import os
import logging

import torch
from torch.utils.data import DataLoader

import transformers
from transformers.optimization import (
    get_cosine_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
    get_constant_schedule_with_warmup
)
import lightning as L
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam

from train.dataset import (
    SpeechSampler, 
    PromptSpeechToTextDatasetCreator, 
    DataCollatorForSupervisedDataset,
    DataCollatorForSupervisedInstructDataset,
    DataCollatorForTrajectoryDataset,
    DataCollatorForTrajectoryInstructDataset,
    DataCollatorForTrajectoryInstructMultiLatencyDataset,
)
from model.llm import SpeechLlamaForCausalLM
from model.speech_encoder import SpeechEncoderW2V2RoPE
from model.patches.patch_w2v2 import patch_w2v2
from model.patches.patch_llama31 import patch_llama31
from model.patches.patch_hf import patch_hf

logger = logging.getLogger(__name__)

collator_classes = {
    0: DataCollatorForSupervisedDataset,
    1: DataCollatorForSupervisedInstructDataset,
    2: DataCollatorForTrajectoryDataset,
    3: DataCollatorForTrajectoryInstructDataset,
    4: DataCollatorForTrajectoryInstructMultiLatencyDataset,
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
            self.speech_args.xpos,
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
        
        patch_w2v2(self.speech_args.xpos, self.speech_args.rope)
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
        speech_encoder_args = [
            self.speech_args.w2v2_path,
            self.speech_args.ctc_finetuned,
            self.speech_args.length_shrink_cfg,
            
            self.speech_args.block_size,
            self.speech_args.max_cache_size,
            model.model.embed_tokens.embedding_dim,
            None,
            self.speech_args.xpos,
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
            model.load_state_dict(state_dict, strict=True)
    
        self.model = model
    
    def train_dataloader(self):
        train_dataset = PromptSpeechToTextDatasetCreator.from_tsv(
            self.data_args.data_path, self.data_args.data_split_train
        )
        collator_cls = collator_classes[self.data_args.trajectory]

        logger.info("collator class: {}".format(collator_cls))

        data_collator = collator_cls(
            self.tokenizer, 
            self.length_shrink_func, 
            self.data_args.source_lang,
            self.data_args.target_lang,
            block_size=self.speech_args.block_size,
            perturb=self.data_args.trajectory_perturb,
            max_multiplier=self.data_args.trajectory_max_multiplier,
            po_max_multiplier=self.data_args.preference_optimization_max_multiplier,
            prob_aug=self.data_args.trajectory_prob_aug,
            trainer=self.trainer
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
        collator_cls = collator_classes[self.data_args.trajectory]
        data_collator = collator_cls(
            self.tokenizer, 
            self.length_shrink_func,
            self.data_args.source_lang,
            self.data_args.target_lang,
            block_size=self.speech_args.block_size,
            max_multiplier=self.data_args.trajectory_max_multiplier,
            po_max_multiplier=self.data_args.preference_optimization_max_multiplier,
            prob_aug=self.data_args.trajectory_prob_aug
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
            self.log("eval/loss", loss, batch_size=batch["src_lengths"].sum() / 16000)
            if "multiplier" in batch:
                self.log("eval/loss_mult{}".format(batch["multiplier"]), loss, batch_size=batch["src_lengths"].sum() / 16000)

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

        optimizer_cls = DeepSpeedCPUAdam if self.training_args.deepspeed_offload else FusedAdam
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
        logger.info("device: {}, batch size: {}".format(self.device, batch['input_ids'].size()))
        output = self.model(
            **batch,
            return_dict=True
        )
        return output.loss