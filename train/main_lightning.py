# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
import json
import pathlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import torch
import torch.distributed
import torch.nn as nn
from torch.utils.data import Dataset

import transformers
from transformers import Trainer, set_seed

from fairseq.data.audio.speech_to_text_dataset import _collate_frames

import conversation as conversation_lib
from train.dataset import PromptSpeechToTextDatasetCreator, SpeechToTextDatasetItem
from model.model_new import SpeechLlamaForCausalLM
from model.speech_encoder import (
    SpeechEncoderHuBERTRope,
    SpeechEncoderW2V2RoPE
)

import lightning as L
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.plugins.precision import FSDPPrecision
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DeepSpeedStrategy, FSDPStrategy

from model.model_lightning import SLlamaLightning
# TODO: import and use code from ../data/dataset.py

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_SPEECH_TOKEN = "<speech>"
DEFAULT_SPEECH_PATCH_TOKEN = "<sp_patch>"
DEFAULT_SPEECH_START_TOKEN = "<sp_start>"
DEFAULT_SPEECH_END_TOKEN = "<sp_end>"

@dataclass
class SpeechEncoderArguments:
    w2v2_path: Optional[str] = field(default=None)
    w2v2_type: Optional[str] = field(default=None)
    w2v2_freeze: bool = field(default=False)
    ctc_finetuned: bool = field(default=False)
    length_shrink_cfg: str = field(default=None)
    block_size: int = field(default=48)
    max_cache_size: int = field(default=500)
    xpos: bool = field(default=True)

@dataclass
class ModelArguments:
    llm_path: Optional[str] = field(default="facebook/opt-125m")
    llm_freeze: bool = field(default=False)
    orig_embeds_params: bool = field(default=False)
    sllm_weight_path: Optional[str] = field(default=None)

@dataclass
class DataArguments:
    data_path: str = field(
        default=None,
        metadata={"help": "Path to the training data."}
    )
    data_split_train: str = field(
        default=None,
        metadata={"help": "Path to the training data."}
    )
    data_split_eval: str = field(
        default=None,
        metadata={"help": "Path to the training data."}
    )
    source_lang: str = field(
        default="English",
        metadata={"help": "Source language name"}
    )
    target_lang: str = field(
        default="Spanish",
        metadata={"help": "Target language name"}
    )
                            
@dataclass
class TrainingArguments:
    seed: int = field(default=1)
    stage: int = field(default=1)
    rdrop: float = field(default=0.)
    text_weight: float = field(default=0.)
    train_bsz: int = field(default=8) # in terms of number of frames
    eval_bsz: int = field(default=8) # in terms of number of frames
    learning_rate: float = field(default=2e-4)
    scheduler: str = field(default="cosine")
    min_learning_rate: float = field(default=0.)
    weight_decay: float = field(default=0.)
    warmup_steps: int = field(default=400)
    run_name: str = field(default=None)

    n_device: int = field(default=1)
    deepspeed_stage: int = field(default=2)
    deepspeed_offload: bool = field(default=False)
    precision: str = field(default="bf16-mixed")
    max_epochs: int = field(default=1)
    grad_acc_steps: int = field(default=1)
    clip_norm: float = field(default=1.)
    save_dir: str = field(default=None)
    log_step: int = field(default=1)
    eval_step: int = field(default=1)
    debug_mode: bool = field(default=False)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer, length_shrink_func, model, source_lang, target_lang):
        self.tokenizer = tokenizer
        self.length_shrink_func = length_shrink_func
        self.model = model
        self.source_lang = source_lang
        self.target_lang = target_lang
     
    def __call__(self, samples: List[SpeechToTextDatasetItem]) -> Dict[str, torch.Tensor]:
        self.model.model.speech_features_extracted = False
        indices = torch.tensor([x.index for x in samples], dtype=torch.long)
        speech_batch = _collate_frames([x.source for x in samples], is_audio_input=True)
        n_frames = torch.tensor([x.source.size(0) for x in samples], dtype=torch.long)
        speech_lens = self.length_shrink_func(n_frames)

        texts = [x.target for x in samples]
     
        # Create speech tokens based on length
        speech_tokens = [int(speech_len)*DEFAULT_SPEECH_PATCH_TOKEN for speech_len in speech_lens]
        speech_tokens = [DEFAULT_SPEECH_START_TOKEN + tokens + DEFAULT_SPEECH_END_TOKEN for tokens in speech_tokens]

        text_tokens = [DEFAULT_SPEECH_START_TOKEN + x.src_text + DEFAULT_SPEECH_END_TOKEN for x in samples]

        # Create prompts
        instruction = f"Translate the following speech from {self.source_lang} to {self.target_lang}:"
        prompts = [f"{instruction} {speech_token} {text}<|end_of_text|>" for speech_token, text in zip(speech_tokens, texts)]
        text_prompts = [f"{instruction} {text_token} {text}<|end_of_text|>" for text_token, text in zip(text_tokens, texts)]
        
        # Get instruction length for masking
        instruction_ids = self.tokenizer(instruction + " ", add_special_tokens=False).input_ids
        instruction_len = len(instruction_ids)

        # Tokenize with explicit padding settings
        tokenized = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        )
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

        # Create targets and handle padding properly
        targets = input_ids.clone()
        for i in range(len(samples)):
            # 1. Mask instruction tokens
            targets[i, :instruction_len] = IGNORE_INDEX
            
            # 2. Mask speech tokens
            start_pos = (input_ids[i] == self.tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_START_TOKEN)).nonzero()
            end_pos = (input_ids[i] == self.tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_END_TOKEN)).nonzero()
            if len(start_pos) > 0 and len(end_pos) > 0:
                targets[i, start_pos[0][0] : end_pos[0][0] + 1] = IGNORE_INDEX
            
            # 3. Mask padding tokens
            targets[i, attention_mask[i] == 0] = IGNORE_INDEX

        # Tokenize with explicit padding settings
        text_tokenized = self.tokenizer(
            text_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        )
        text_input_ids = text_tokenized.input_ids
        text_attention_mask = text_tokenized.attention_mask

        # Create targets and handle padding properly
        text_targets = text_input_ids.clone()
        for i in range(len(samples)):
            # 1. Mask instruction tokens
            text_targets[i, :instruction_len] = IGNORE_INDEX
            # 2. Mask speech tokens
            start_pos = (text_input_ids[i] == self.tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_START_TOKEN)).nonzero()
            end_pos = (text_input_ids[i] == self.tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_END_TOKEN)).nonzero()
            if len(start_pos) > 0 and len(end_pos) > 0:
                text_targets[i, start_pos[0][0] : end_pos[0][0] + 1] = IGNORE_INDEX
            # 3. Mask padding tokens
            text_targets[i, text_attention_mask[i] == 0] = IGNORE_INDEX
                
        batch = dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=attention_mask,
            speech_batch=speech_batch,
            src_lengths=n_frames,
            after_lens=speech_lens,

            text_input_ids=text_input_ids,
            text_labels=text_targets,
            text_attention_mask=text_attention_mask,
        )

        return batch

def make_supervised_data_module(
        tokenizer: transformers.PreTrainedTokenizer,
        data_args,
        length_shrink_func,
        model
    ):
    """Make dataset and collator for supervised fine-tuning."""

    train_dataset = PromptSpeechToTextDatasetCreator.from_tsv(
        data_args.data_path, data_args.data_split_train
    )
    eval_dataset = PromptSpeechToTextDatasetCreator.from_tsv(
        data_args.data_path, data_args.data_split_eval
    ) if data_args.data_split_eval is not None else None

    data_collator = DataCollatorForSupervisedDataset(
        tokenizer, 
        length_shrink_func, 
        model,
        data_args.source_lang,
        data_args.target_lang
    )

    return train_dataset, eval_dataset, data_collator

def train():
    parser = transformers.HfArgumentParser(
        (SpeechEncoderArguments, ModelArguments, DataArguments, TrainingArguments))
    speech_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set seed before initializing model.
    set_seed(training_args.seed) 
  
    model_lightning = SLlamaLightning(
        speech_args=speech_args,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        lr=training_args.learning_rate,
        warmup_updates=training_args.warmup_steps,
        min_lr=0.,
    )

    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=training_args.save_dir,
    #     monitor='eval/loss',
    #     save_top_k=1,
    #     mode='min',
    #     every_n_train_steps=training_args.eval_step
    # )
    lr_monitor = LearningRateMonitor(
        logging_interval='step'
    )

    wandb_logger = WandbLogger(
        name=training_args.run_name,
        # log_model="all"
    )

    strategy = DeepSpeedStrategy(
        stage=training_args.deepspeed_stage,
        offload_optimizer=training_args.deepspeed_offload,
        offload_parameters=training_args.deepspeed_offload,
    )
    # strategy = FSDPStrategy(
    #     sharding_strategy=training_args.sharding,
    #     state_dict_type="sharded"
    # )

    trainer = L.Trainer(
        accelerator='gpu',
        devices=training_args.n_device,
        strategy=strategy,
        precision=training_args.precision,
        max_steps=-1,
        max_epochs=training_args.max_epochs,
        accumulate_grad_batches=training_args.grad_acc_steps,
        gradient_clip_val=training_args.clip_norm,
        use_distributed_sampler=False,
        default_root_dir=training_args.save_dir,
        log_every_n_steps=training_args.log_step,
        val_check_interval=training_args.eval_step,
        logger=wandb_logger,
        callbacks=[lr_monitor],
        fast_dev_run=training_args.debug_mode,
        enable_checkpointing=False,
    )

    # start training
    if os.path.exists(training_args.save_dir) and len(os.listdir(training_args.save_dir)) >= 1:
        ckpt_path = os.path.join(training_args.save_dir, os.listdir(training_args.save_dir)[0])
        trainer.fit(model_lightning, ckpt_path=ckpt_path)
    else:
        trainer.fit(model_lightning)

    trainer.save_checkpoint(training_args.save_dir, weights_only=True)


if __name__ == "__main__":
    train()