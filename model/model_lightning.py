import os

import torch
from torch.utils.data import DataLoader

import transformers
import lightning as L
from torch.optim import Adam
from apex.optimizers import FusedAdam

from train.dataset import (
    SpeechSampler, 
    PromptSpeechToTextDatasetCreator, 
    SpeechToTextDatasetItem,
    DataCollatorForSupervisedDataset
)
from model.model_new import SpeechLlamaForCausalLM
from model.speech_encoder import (
    SpeechEncoderHuBERTRope,
    SpeechEncoderW2V2RoPE
)

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
        speech_encoder_args = [
            self.speech_args.w2v2_path,
            self.speech_args.ctc_finetuned,
            self.speech_args.length_shrink_cfg,
            
            self.speech_args.block_size,
            self.speech_args.max_cache_size,
            1,
            None,
            self.speech_args.xpos,
        ]
        if self.speech_args.w2v2_type == 'hubert':
            speech_encoder = SpeechEncoderHuBERTRope(*speech_encoder_args)
        else:
            speech_encoder = SpeechEncoderW2V2RoPE(*speech_encoder_args)
        self.length_shrink_func = speech_encoder._get_feat_extract_output_lengths

        self.optimizer_params = {
            "lr": lr,
            "warmup_updates": warmup_updates,
            "min_lr": min_lr,
        }

    def configure_model(self):
        if self.model is not None:
            return

        model = SpeechLlamaForCausalLM.from_pretrained(
            self.model_args.llm_path,
            torch_dtype=torch.bfloat16,
            device_map='cuda'
        )
        model.config.use_cache = False
        model.rdrop = self.training_args.rdrop
        model.text_weight = self.training_args.text_weight

        if self.model_args.llm_freeze:
            model.model.requires_grad_(False)
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
        ]
        if self.speech_args.w2v2_type == 'hubert':
            speech_encoder = SpeechEncoderHuBERTRope(*speech_encoder_args)
        else:
            speech_encoder = SpeechEncoderW2V2RoPE(*speech_encoder_args) 

        speech_encoder.to(dtype=model.dtype, device=model.device)
        model.model.speech_encoder = speech_encoder

        if self.speech_args.w2v2_freeze:
            model.model.speech_encoder.requires_grad_(False)

        model.preprocess(tokenizer=self.tokenizer)
        model.model.orig_embeds_params = self.model_args.orig_embeds_params

        if self.model_args.sllm_weight_path is not None:
            state_dict = torch.load(self.model_args.sllm_weight_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict, strict=True)
    
        self.model = model
    
    def train_dataloader(self):
        train_dataset = PromptSpeechToTextDatasetCreator.from_tsv(
            self.data_args.data_path, self.data_args.data_split_train
        )
        data_collator = DataCollatorForSupervisedDataset(
            self.tokenizer, 
            self.length_shrink_func, 
            self.data_args.source_lang,
            self.data_args.target_lang
        )

        train_sampler = SpeechSampler(
            train_dataset, 
            shuffle=True, 
            batch_size=self.training_args.train_bsz, 
            min_ms=320,
            multiplier=self.training_args.n_device * self.training_args.grad_acc_steps
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
        data_collator = DataCollatorForSupervisedDataset(
            self.tokenizer, 
            self.length_shrink_func,
            self.data_args.source_lang,
            self.data_args.target_lang
        )

        eval_sampler = SpeechSampler(
            eval_dataset, 
            shuffle=False, 
            batch_size=self.training_args.eval_bsz, 
            min_ms=320,
            multiplier=self.training_args.n_device * self.training_args.grad_acc_steps
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
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        if not loss.isnan():
            self.log("eval/loss", loss, batch_size=batch["src_lengths"].sum() / 16000)

    def configure_optimizers(self):
        lr = self.optimizer_params["lr"]
        min_lr = self.optimizer_params["min_lr"]
        warmup_updates = self.optimizer_params["warmup_updates"]

        optimizer = FusedAdam(self.parameters(), lr=lr)        
        warmup_init_lr = 0 if warmup_updates > 0 else lr
        lr_step = (lr - warmup_init_lr) / warmup_updates
        decay_factor = lr * warmup_updates**0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, 
            lambda x: max(decay_factor * x**-0.5 if x >= warmup_updates \
                else warmup_init_lr + x * lr_step, min_lr) / lr
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
    
    def forward(self, batch):
        output = self.model(
            **batch,
            return_dict=True
        )
        return output.loss