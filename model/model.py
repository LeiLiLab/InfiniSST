import os
import logging

import torch
from torch.utils.data import DataLoader

import transformers
from transformers import AutoProcessor
from transformers.optimization import (
    get_cosine_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
    get_constant_schedule_with_warmup
)
from transformers.models.seamless_m4t_v2.configuration_seamless_m4t_v2 import SeamlessM4Tv2Config

import lightning as L
# Optional DeepSpeed import - fallback to standard Adam if not available
try:
    from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    FusedAdam = None
    DeepSpeedCPUAdam = None

from peft import LoraConfig, get_peft_model

from train.dataset import (
    SpeechSampler, 
    PromptSpeechToTextDatasetCreator, 
    DataCollatorForSupervisedDataset,
    DataCollatorForSupervisedInstructDataset,
    DataCollatorForTrajectoryDataset,
    DataCollatorForTrajectoryInstructDataset,
    DataCollatorForTrajectoryInstructMultiLatencyDataset,
    DataCollatorForTrajectoryInstructMultiLatencyQwenDataset,
    DataCollatorForOfflineSeamlessDataset,
    DataCollatorForTrajectoryInstructMultiLatencySeamlessDataset
)
from model.llama31 import SpeechLlamaForCausalLM
from model.qwen25 import SpeechQwenForCausalLM

from model.w2v2 import SpeechEncoderW2V2RoPE
from model.patches.patch_w2v2 import patch_w2v2
from model.patches.patch_llama31 import patch_llama31
from model.patches.patch_qwen25 import patch_qwen25
from model.patches.patch_hf import patch_hf

from model.qwen2ac import Qwen2AudioForConditionalGeneration
from model.patches.patch_qwen2ac import patch_qwen2ac

from model.seamlessm4t_v2_encoder import SeamlessM4Tv2SpeechEncoder

logger = logging.getLogger(__name__)

collator_classes = {
    0: DataCollatorForSupervisedDataset,
    1: DataCollatorForSupervisedInstructDataset,
    2: DataCollatorForTrajectoryDataset,
    3: DataCollatorForTrajectoryInstructDataset,
    4: DataCollatorForTrajectoryInstructMultiLatencyDataset,
    5: DataCollatorForOfflineSeamlessDataset,
    6: DataCollatorForTrajectoryInstructMultiLatencySeamlessDataset,
    7: DataCollatorForTrajectoryInstructMultiLatencyQwenDataset,
    8: DataCollatorForTrajectoryInstructMultiLatencyQwenDataset,
    9: DataCollatorForTrajectoryInstructMultiLatencyQwenDataset,
}

class SLlamaLightning(L.LightningModule):
    def __init__(
            self, speech_args, model_args, data_args, training_args,
            lr=2e-4, warmup_updates=4000, min_lr=0.
        ):
        super().__init__()
        self.model = None

        self.speech_args = speech_args
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_args.llm_path,
            padding_side="right",
            use_fast=False,
        )
        self.tokenizer.pad_token = "<|finetune_right_pad_id|>"

        self.fast_tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_args.llm_path,
            padding_side="right",
            use_fast=True,
        )

        # load speech encoder
        logger.info("rope: {}".format(self.speech_args.rope))
        speech_encoder_args = [
            self.speech_args.w2v2_path,
            self.speech_args.ctc_finetuned,
            self.speech_args.length_shrink_cfg,
            
            self.speech_args.block_size,
            self.speech_args.max_cache_size,
            1,
            None,
            self.speech_args.rope,
        ]
        if self.speech_args.w2v2_type == 'w2v2':
            speech_encoder = SpeechEncoderW2V2RoPE(*speech_encoder_args) 
        else:
            raise ValueError(f"Unsupported type: {self.speech_args.w2v2_type}")

        self.length_shrink_func = speech_encoder._get_feat_extract_output_lengths # TODO: make this more efficient

        self.optimizer_params = {
            "lr": lr,
            "warmup_updates": warmup_updates,
            "min_lr": min_lr,
        }

    def configure_model(self):
        if self.model is not None:
            return
        
        patch_w2v2(self.speech_args.rope)
        patch_llama31()
        patch_hf()

        logger.info("use_flash_attn: {}".format(self.model_args.use_flash_attn))
        model = SpeechLlamaForCausalLM.from_pretrained(
            self.model_args.llm_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if self.model_args.use_flash_attn else 'sdpa'
        )
        model.config.use_cache = False
        model.rdrop = self.training_args.rdrop
        model.text_weight = self.training_args.text_weight

        if self.model_args.llm_freeze:
            model.model.requires_grad_(False)
            model.model.embed_tokens.requires_grad_(True)
        if self.model_args.llm_emb_freeze:
            model.model.embed_tokens.requires_grad_(False)
        if self.model_args.llm_head_freeze:
            model.lm_head.requires_grad_(False)

        # load speech encoder
        speech_encoder_args = [
            self.speech_args.w2v2_path,
            self.speech_args.ctc_finetuned,
            self.speech_args.length_shrink_cfg,
            
            self.speech_args.block_size,
            self.speech_args.max_cache_size,
            model.model.embed_tokens.embedding_dim,
            None,
            self.speech_args.rope,
        ]
        if self.speech_args.w2v2_type == 'w2v2':
            speech_encoder = SpeechEncoderW2V2RoPE(*speech_encoder_args) 
        else:
            raise ValueError(f"Unsupported type: {self.speech_args.w2v2_type}")

        speech_encoder.to(dtype=model.dtype, device=model.device)
        model.model.speech_encoder = speech_encoder

        model.cpo_beta = self.training_args.cpo_beta

        if self.speech_args.w2v2_freeze:
            model.model.speech_encoder.requires_grad_(False)

        model.preprocess(tokenizer=self.tokenizer, max_multiplier=self.data_args.trajectory_max_multiplier)

        if self.model_args.sllm_weight_path is not None:
            logger.info("Loading SLLM weights from {}".format(self.model_args.sllm_weight_path))
            state_dict = torch.load(self.model_args.sllm_weight_path, map_location='cpu', weights_only=True)
            if state_dict['model.speech_encoder.proj.weight'].size() != model.model.speech_encoder.proj.weight.size():
                state_dict.pop('model.speech_encoder.proj.weight')
                state_dict.pop('model.speech_encoder.proj.bias')
            model.load_state_dict(state_dict, strict=False)

        if self.model_args.lora_rank > 0:
            lora_config = LoraConfig(
                task_type="CAUSAL_LM",
                inference_mode=False,
                r=self.model_args.lora_rank,
                target_modules='all-linear',
                lora_alpha=16,
                lora_dropout=0.1,
            )
            model = get_peft_model(model, lora_config, adapter_name='lora_adapter')
            model.print_trainable_parameters()
    
        self.model = model
    
    def train_dataloader(self):
        import time
        t0 = time.time()
        logger.info("[DL] Building train dataset from %s (%s.tsv)", self.data_args.data_path, self.data_args.data_split_train)
        train_dataset = PromptSpeechToTextDatasetCreator.from_tsv(
            self.data_args.data_path, self.data_args.data_split_train
        )
        logger.info("[DL] Train dataset ready: %d samples (%.2fs)", len(train_dataset), time.time() - t0)
        collator_cls = collator_classes.get(self.data_args.trajectory, DataCollatorForTrajectoryInstructMultiLatencyQwenDataset)

        logger.info("collator class: {}".format(collator_cls))

        data_collator = collator_cls(
            self.tokenizer, 
            self.length_shrink_func, 
            self.data_args.source_lang,
            self.data_args.target_lang,
            block_size=self.speech_args.block_size,
            perturb=self.data_args.trajectory_perturb,
            multiplier_step_size=self.data_args.trajectory_step_size,
            max_multiplier=self.data_args.trajectory_max_multiplier,
            po_max_multiplier=self.data_args.preference_optimization_max_multiplier,
            prob_aug=self.data_args.trajectory_prob_aug,
            audio_normalize=self.data_args.audio_normalize,
            trainer=self.trainer,
            dataset=train_dataset
        )

        # if self.data_args.trajectory >= 1:
        #     data_collator.validate(train_dataset)

        logger.info("train_bsz: {}, bsz_sent: {}".format(self.training_args.train_bsz, self.training_args.bsz_sent))

        t1 = time.time()
        train_sampler = SpeechSampler(
            train_dataset, 
            shuffle=True, 
            batch_size=self.training_args.train_bsz, 
            batch_size_sent=self.training_args.bsz_sent,
            min_ms=320,
            multiplier=self.training_args.n_device * self.training_args.grad_acc_steps,
            filter=True,
            tokenizer=self.fast_tokenizer,
        )
        # SpeechSampler 会打印 skipped; 这里补充批次数
        try:
            n_batches = len(train_sampler.batch_indices)
        except Exception:
            n_batches = -1
        logger.info("[DL] Train sampler ready: %s batches (%.2fs)", n_batches, time.time() - t1)
        # 更高吞吐的 DataLoader 设置（可通过环境变量调整）
        num_workers = int(os.environ.get("DL_NUM_WORKERS", "24"))
        prefetch = int(os.environ.get("DL_PREFETCH", "6"))
        train_dataloader = DataLoader(
            train_dataset, 
            batch_sampler=train_sampler, 
            collate_fn=data_collator,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            pin_memory=True,
            prefetch_factor=prefetch if num_workers > 0 else None
        )
        return train_dataloader
    
    def val_dataloader(self):
        import time
        t0 = time.time()
        logger.info("[DL] Building eval dataset from %s (%s.tsv)", self.data_args.data_path, self.data_args.data_split_eval)
        eval_dataset = PromptSpeechToTextDatasetCreator.from_tsv(
            self.data_args.data_path, self.data_args.data_split_eval
        )
        logger.info("[DL] Eval dataset ready: %d samples (%.2fs)", len(eval_dataset), time.time() - t0)
        collator_cls = collator_classes.get(self.data_args.trajectory, DataCollatorForTrajectoryInstructMultiLatencyQwenDataset)
        data_collator = collator_cls(
            self.tokenizer, 
            self.length_shrink_func,
            self.data_args.source_lang,
            self.data_args.target_lang,
            block_size=self.speech_args.block_size,
            max_multiplier=self.data_args.trajectory_max_multiplier,
            po_max_multiplier=self.data_args.preference_optimization_max_multiplier,
            multiplier_step_size=self.data_args.trajectory_step_size,
            prob_aug=self.data_args.trajectory_prob_aug,
            audio_normalize=self.data_args.audio_normalize,
            dataset=eval_dataset
        )

        # if self.data_args.trajectory >= 1:
        #     data_collator.validate(eval_dataset)

        t1 = time.time()
        eval_sampler = SpeechSampler(
            eval_dataset, 
            shuffle=False, 
            batch_size=self.training_args.eval_bsz, 
            batch_size_sent=self.training_args.bsz_sent,
            min_ms=320,
            multiplier=self.training_args.n_device * self.training_args.grad_acc_steps,
            filter=True,
            tokenizer=self.fast_tokenizer,
        )
        try:
            n_batches = len(eval_sampler.batch_indices)
        except Exception:
            n_batches = -1
        logger.info("[DL] Eval sampler ready: %s batches (%.2fs)", n_batches, time.time() - t1)
        num_workers = int(os.environ.get("DL_NUM_WORKERS", "16"))
        prefetch = int(os.environ.get("DL_PREFETCH", "4"))
        eval_dataloader = DataLoader(
            eval_dataset, 
            batch_sampler=eval_sampler, 
            collate_fn=data_collator,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            pin_memory=True,
            prefetch_factor=prefetch if num_workers > 0 else None
        )
        return eval_dataloader

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        if not loss.isnan():
            self.log("train/loss", loss, batch_size=batch["src_lengths"].sum() / 16000)
            if "multiplier" in batch:
                self.log("train/loss_mult{}".format(batch["multiplier"]), loss, batch_size=batch["src_lengths"].sum() / 16000)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        if not loss.isnan():
            self.log("eval/loss", loss, batch_size=batch["src_lengths"].sum() / 16000, sync_dist=True)
            if "multiplier" in batch:
                self.log("eval/loss_mult{}".format(batch["multiplier"]), loss, batch_size=batch["src_lengths"].sum() / 16000, sync_dist=True)

    def setup(self, stage):
        if stage == 'fit':
            train_batches = len(self.train_dataloader()) // (self.training_args.n_device * self.training_args.grad_acc_steps)
            self.max_train_steps = self.training_args.max_epochs * train_batches
            print("Max number of training steps", self.max_train_steps)
    
    # def on_before_optimizer_step(self, optimizer):
    #     # Compute the 2-norm for each layer
    #     # If using mixed precision, the gradients are already unscaled here
    #     norms = grad_norm(self.model, norm_type=2)
    #     self.log_dict(norms)

    def configure_optimizers(self):
        lr = self.optimizer_params["lr"]
        warmup_updates = self.optimizer_params["warmup_updates"]

        # optimizer_cls = DeepSpeedCPUAdam if self.training_args.deepspeed_offload else FusedAdam
        optimizer_cls = torch.optim.Adam
        optimizer = optimizer_cls(self.parameters(), lr=lr, weight_decay=self.training_args.weight_decay)  

        if self.training_args.scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=warmup_updates,
                num_training_steps=self.max_train_steps,
            )
        elif self.training_args.scheduler == "cosine_minlr":
            scheduler = get_cosine_with_min_lr_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=warmup_updates,
                num_training_steps=self.max_train_steps,
                min_lr=self.training_args.min_learning_rate
            )
        elif self.training_args.scheduler == "constant":
            scheduler = get_constant_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=warmup_updates,
            )
        else:
            raise NotImplementedError

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
    
    def forward(self, batch):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "device: %s, batch size: %s, multiplier: %s",
                self.device, batch['input_ids'].size(), batch.get('multiplier', 'n/a')
            )
        output = self.model(
            **batch,
            return_dict=True
        )
        return output.loss


class SQwen25Lightning(SLlamaLightning):
    def __init__(
            self, speech_args, model_args, data_args, training_args,
            lr=2e-4, warmup_updates=4000, min_lr=0.
        ):
        super(SQwen25Lightning, self).__init__(
            speech_args, model_args, data_args, training_args, 
            lr, warmup_updates, min_lr
        )
    
    def configure_model(self):
        if self.model is not None:
            return
        
        patch_w2v2(self.speech_args.rope)
        patch_qwen25()
        patch_hf()

        # Determine whether to use FlashAttention2. If requested but not installed, fallback to SDPA.
        requested_flash = bool(self.model_args.use_flash_attn)
        effective_flash = requested_flash
        if requested_flash:
            try:
                import flash_attn  # noqa: F401
            except Exception as e:
                effective_flash = False
                logger.warning(
                    "FlashAttention2 requested but unavailable ({}). Falling back to SDPA.", str(e)
                )

        logger.info(
            "use_flash_attn (requested={}, effective={})".format(requested_flash, effective_flash)
        )
        attn_impl = "flash_attention_2" if effective_flash else "sdpa"
        model = SpeechQwenForCausalLM.from_pretrained(
            self.model_args.llm_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl
        )
        model.config.use_cache = False
        # Speed/memory knobs
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        # Reduce activations with checkpointing
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
        # Optional compile for speed: set TORCH_COMPILE=1 [mode: reduce-overhead|max-autotune]
        try:
            if os.environ.get("TORCH_COMPILE", "0") == "1":
                mode = os.environ.get("TORCH_COMPILE_MODE", "reduce-overhead")
                model = torch.compile(model, mode=mode)
        except Exception:
            pass

        if self.model_args.llm_freeze:
            model.model.requires_grad_(False)
            model.model.embed_tokens.requires_grad_(True)
        if self.model_args.llm_emb_freeze:
            model.model.embed_tokens.requires_grad_(False)
        if self.model_args.llm_head_freeze:
            model.lm_head.requires_grad_(False)

        # load speech encoder
        speech_encoder_args = [
            self.speech_args.w2v2_path,
            self.speech_args.ctc_finetuned,
            self.speech_args.length_shrink_cfg,
            
            self.speech_args.block_size,
            self.speech_args.max_cache_size,
            model.model.embed_tokens.embedding_dim,
            None,
            self.speech_args.rope,
        ]
        if self.speech_args.w2v2_type == 'w2v2':
            speech_encoder = SpeechEncoderW2V2RoPE(*speech_encoder_args) 
        else:
            raise ValueError(f"Unsupported type: {self.speech_args.w2v2_type}")

        speech_encoder.to(dtype=model.dtype, device=model.device)
        model.model.speech_encoder = speech_encoder

        if self.speech_args.w2v2_freeze:
            model.model.speech_encoder.requires_grad_(False)

        model.preprocess(tokenizer=self.tokenizer)

        if self.model_args.sllm_weight_path is not None:
            logger.info("Loading SLLM weights from {}".format(self.model_args.sllm_weight_path))
            state_dict = torch.load(self.model_args.sllm_weight_path, map_location='cpu', weights_only=True)
            if state_dict['model.speech_encoder.proj.weight'].size() != model.model.speech_encoder.proj.weight.size():
                state_dict.pop('model.speech_encoder.proj.weight')
                state_dict.pop('model.speech_encoder.proj.bias')
            model.load_state_dict(state_dict, strict=False)

        if self.model_args.lora_rank > 0:
            lora_config = LoraConfig(
                task_type="CAUSAL_LM",
                inference_mode=False,
                r=self.model_args.lora_rank,
                target_modules='all-linear',
                lora_alpha=16,
                lora_dropout=0.1,
            )
            model = get_peft_model(model, lora_config, adapter_name='lora_adapter')
            model.print_trainable_parameters()
    
        self.model = model


class Qwen2ACLightning(SLlamaLightning):
    def __init__(
            self, speech_args, model_args, data_args, training_args,
            lr=2e-4, warmup_updates=4000, min_lr=0.
        ):
        super(SLlamaLightning, self).__init__()
        self.model = None

        self.speech_args = speech_args
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

        self.processor = AutoProcessor.from_pretrained(
            self.model_args.llm_path,
            use_fast=False,
        )

        self.optimizer_params = {
            "lr": lr,
            "warmup_updates": warmup_updates,
            "min_lr": min_lr,
        }

    def configure_model(self):
        if self.model is not None:
            return
        
        patch_qwen2ac()

        logger.info("use_flash_attn: {}".format(self.model_args.use_flash_attn))
        
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.model_args.llm_path, 
            torch_dtype=torch.bfloat16 if self.model_args.use_flash_attn else None, 
            attn_implementation="flash_attention_2" if self.model_args.use_flash_attn else 'sdpa'
        )
        model.config.audio_config.sliding_window_size = self.speech_args.max_cache_size
        model.config.audio_config.blocksize = self.speech_args.block_size

        if self.model_args.llm_freeze:
            model.language_model.requires_grad_(False)

        if self.speech_args.whisper_freeze:
            model.audio_tower.requires_grad_(False)

        if self.speech_args.adapter_freeze:
            model.multi_modal_projector.requires_grad_(False)

        if self.model_args.sllm_weight_path is not None:
            logger.info("Loading weights from {}".format(self.model_args.sllm_weight_path))
            state_dict = torch.load(self.model_args.sllm_weight_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict, strict=True)
    
        self.model = model
    
    def train_dataloader(self):
        train_dataset = PromptSpeechToTextDatasetCreator.from_tsv(
            self.data_args.data_path, self.data_args.data_split_train
        )
        collator_cls = collator_classes[self.data_args.trajectory]

        logger.info("collator class: {}".format(collator_cls))

        data_collator = collator_cls(
            self.processor, 
            self.data_args.source_lang,
            self.data_args.target_lang,
            block_size=self.speech_args.block_size,
            max_multiplier=self.data_args.trajectory_max_multiplier,
            prob_aug=self.data_args.trajectory_prob_aug,
        )

        logger.info("train_bsz: {}, bsz_sent: {}".format(self.training_args.train_bsz, self.training_args.bsz_sent))

        train_sampler = SpeechSampler(
            train_dataset, 
            shuffle=True, 
            batch_size=self.training_args.train_bsz, 
            batch_size_sent=self.training_args.bsz_sent,
            min_ms=320,
            multiplier=self.training_args.n_device * self.training_args.grad_acc_steps,
            filter=True,
            tokenizer=self.processor.tokenizer,
            model_type="qwen2ac"
        )
        train_dataloader = DataLoader(
            train_dataset, 
            batch_sampler=train_sampler, 
            collate_fn=data_collator
        )
        return train_dataloader
    
    def val_dataloader(self):
        eval_dataset = PromptSpeechToTextDatasetCreator.from_tsv(
        self.data_args.data_path, self.data_args.data_split_eval
        )
        collator_cls = collator_classes[self.data_args.trajectory]
        data_collator = collator_cls(
            self.processor, 
            self.data_args.source_lang,
            self.data_args.target_lang,
            block_size=self.speech_args.block_size,
            max_multiplier=self.data_args.trajectory_max_multiplier,
            prob_aug=self.data_args.trajectory_prob_aug,
        )

        eval_sampler = SpeechSampler(
            eval_dataset, 
            shuffle=False, 
            batch_size=self.training_args.eval_bsz, 
            batch_size_sent=self.training_args.bsz_sent,
            min_ms=320,
            multiplier=self.training_args.n_device * self.training_args.grad_acc_steps,
            filter=True,
            tokenizer=self.processor.tokenizer,
            model_type="qwen2ac"
        )
        eval_dataloader = DataLoader(
            eval_dataset, 
            batch_sampler=eval_sampler, 
            collate_fn=data_collator
        )
        return eval_dataloader
    
class SeamlessLightning(SLlamaLightning):
    def __init__(
            self, speech_args, model_args, data_args, training_args,
            lr=2e-4, warmup_updates=4000, min_lr=0.
        ):
        super(SLlamaLightning, self).__init__()
        self.model = None

        self.speech_args = speech_args
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_args.llm_path,
            padding_side="right",
            use_fast=False,
        )
        self.tokenizer.pad_token = "<|finetune_right_pad_id|>"

        self.processor = AutoProcessor.from_pretrained(self.speech_args.seamless_path)

        self.optimizer_params = {
            "lr": lr,
            "warmup_updates": warmup_updates,
            "min_lr": min_lr,
        }

    def configure_model(self):
        if self.model is not None:
            return
        
        patch_llama31()
        patch_hf()

        logger.info("use_flash_attn: {}".format(self.model_args.use_flash_attn))
        model = SpeechLlamaForCausalLM.from_pretrained(
            self.model_args.llm_path,
            torch_dtype=torch.bfloat16 if self.model_args.use_flash_attn else None,
            attn_implementation="flash_attention_2" if self.model_args.use_flash_attn else 'sdpa'
        )
        model.config.use_cache = False
        model.rdrop = self.training_args.rdrop
        model.text_weight = self.training_args.text_weight

        if self.model_args.llm_freeze:
            model.model.requires_grad_(False)
            model.model.embed_tokens.requires_grad_(True)
        if self.model_args.llm_emb_freeze:
            model.model.embed_tokens.requires_grad_(False)
        if self.model_args.llm_head_freeze:
            model.lm_head.requires_grad_(False)

        # load speech encoder
        
        config = SeamlessM4Tv2Config.from_pretrained(self.speech_args.seamless_path)
        config.llm_embedding_dim = model.model.embed_tokens.embedding_dim
        config.speech_encoder_layerdrop = 0
        config.position_embeddings_type = 'rope'
        config.speech_encoder_chunk_size = self.speech_args.block_size
        config.speech_encoder_left_chunk_num = self.speech_args.max_cache_size // self.speech_args.block_size
        speech_encoder = SeamlessM4Tv2SpeechEncoder(config)
        speech_encoder_state_dict = torch.load(os.path.join(self.speech_args.seamless_path, "speech_encoder.pt"), weights_only=True)
        speech_encoder.load_state_dict(speech_encoder_state_dict, strict=False)        
        speech_encoder.to(dtype=model.dtype, device=model.device)        
        model.model.speech_encoder = speech_encoder

        model.cpo_beta = self.training_args.cpo_beta

        if self.speech_args.seamless_freeze:
            model.model.speech_encoder.requires_grad_(False)

        model.preprocess(tokenizer=self.tokenizer, max_multiplier=self.data_args.trajectory_max_multiplier)

        if self.model_args.sllm_weight_path is not None:
            logger.info("Loading SLLM weights from {}".format(self.model_args.sllm_weight_path))
            state_dict = torch.load(self.model_args.sllm_weight_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict, strict=True)
    
        self.model = model
    
    def train_dataloader(self):
        train_dataset = PromptSpeechToTextDatasetCreator.from_tsv(
            self.data_args.data_path, self.data_args.data_split_train
        )
        collator_cls = collator_classes.get(self.data_args.trajectory, DataCollatorForTrajectoryInstructMultiLatencyQwenDataset)

        logger.info("collator class: {}".format(collator_cls))

        data_collator = collator_cls(
            self.tokenizer, 
            self.processor,
            self.data_args.source_lang,
            self.data_args.target_lang,
            block_size=self.speech_args.block_size,
            max_multiplier=self.data_args.trajectory_max_multiplier,
            prob_aug=self.data_args.trajectory_prob_aug,
        )

        # if self.data_args.trajectory >= 1:
        #     data_collator.validate(train_dataset)

        logger.info("train_bsz: {}, bsz_sent: {}".format(self.training_args.train_bsz, self.training_args.bsz_sent))

        train_sampler = SpeechSampler(
            train_dataset, 
            shuffle=True, 
            batch_size=self.training_args.train_bsz, 
            batch_size_sent=self.training_args.bsz_sent,
            min_ms=320,
            multiplier=self.training_args.n_device * self.training_args.grad_acc_steps,
            filter=True,
            tokenizer=self.tokenizer,
        )
        train_dataloader = DataLoader(
            train_dataset, 
            batch_sampler=train_sampler, 
            collate_fn=data_collator
        )
        return train_dataloader
    
    def val_dataloader(self):
        eval_dataset = PromptSpeechToTextDatasetCreator.from_tsv(
            self.data_args.data_path, self.data_args.data_split_eval
        )
        collator_cls = collator_classes.get(self.data_args.trajectory, DataCollatorForTrajectoryInstructMultiLatencyQwenDataset)
        data_collator = collator_cls(
            self.tokenizer, 
            self.processor,
            self.data_args.source_lang,
            self.data_args.target_lang,
            block_size=self.speech_args.block_size,
            max_multiplier=self.data_args.trajectory_max_multiplier,
            prob_aug=self.data_args.trajectory_prob_aug,
        )

        # if self.data_args.trajectory >= 1:
        #     data_collator.validate(eval_dataset)

        eval_sampler = SpeechSampler(
            eval_dataset, 
            shuffle=False, 
            batch_size=self.training_args.eval_bsz, 
            batch_size_sent=self.training_args.bsz_sent,
            min_ms=320,
            multiplier=self.training_args.n_device * self.training_args.grad_acc_steps,
            filter=True,
            tokenizer=self.tokenizer,
        )
        eval_dataloader = DataLoader(
            eval_dataset, 
            batch_sampler=eval_sampler, 
            collate_fn=data_collator
        )
        return eval_dataloader
