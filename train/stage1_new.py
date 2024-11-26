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

import os, sys, random
# os.environ['WANDB_DISABLED'] = 'true'
# sys.path.append('/home/xixu/sllama')
import copy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence

import torch
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
    ctc_finetuned: bool = field(default=False)
    length_shrink_cfg: str = field(default=None)
    block_size: int = field(default=48)
    max_cache_size: int = field(default=500)

@dataclass
class ModelArguments:
    llm_path: Optional[str] = field(default="facebook/opt-125m")
    llm_freeze: bool = field(default=False)

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    data_split_train: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    data_split_eval: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    source_lang: str = field(default="English",
                           metadata={"help": "Source language name"})
    target_lang: str = field(default="Spanish",
                           metadata={"help": "Target language name"})
 
                           
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA."}
    )
    lora_config: Optional[str] = field(
        default=None,
        metadata={"help": "LoRA config file."},
    )

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

        # Create prompts
        instruction = f"Translate the following speech from {self.source_lang} to {self.target_lang}:"
        prompts = [f"{instruction} {speech_token} {text}<|end_of_text|>" for speech_token, text in zip(speech_tokens, texts)]
        
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
                targets[i, start_pos[0][0]:end_pos[0][0] + 1] = IGNORE_INDEX
            
            # 3. Mask padding tokens
            targets[i, attention_mask[i] == 0] = IGNORE_INDEX
                
        batch = dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=attention_mask,
            speech_batch=speech_batch,
            src_lengths=n_frames,
            after_lens=speech_lens,
        )

        return batch

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args,
                                length_shrink_func,
                                model) -> Dict:
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
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )

def train():
    parser = transformers.HfArgumentParser(
        (SpeechEncoderArguments, ModelArguments, DataArguments, TrainingArguments))
    speech_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set seed before initializing model.
    set_seed(training_args.seed) 

    # load LLM
    model = SpeechLlamaForCausalLM.from_pretrained(
        model_args.llm_path,
        cache_dir=training_args.cache_dir,
    )
    model.config.use_cache = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.llm_path,
        cache_dir=training_args.cache_dir,
        padding_side="right",
        use_fast=False,
    )
 
    tokenizer.pad_token = "<|finetune_right_pad_id|>"

    if model_args.llm_freeze:
        model.model.requires_grad_(False)
        model.lm_head.requires_grad_(False)

    # load speech encoder
    speech_encoder_args = [
        speech_args.w2v2_path,
        speech_args.ctc_finetuned,
        speech_args.length_shrink_cfg,
        
        speech_args.block_size,
        speech_args.max_cache_size,
        model.model.embed_tokens.embedding_dim,
        None,
    ]
    if speech_args.w2v2_type == 'hubert':
        speech_encoder = SpeechEncoderHuBERTRope(*speech_encoder_args)
    else:
        speech_encoder = SpeechEncoderW2V2RoPE(*speech_encoder_args) 

    speech_encoder.to(device=model.device, dtype=model.dtype)
    model.model.speech_encoder = speech_encoder

    model.preprocess(tokenizer=tokenizer) # only in stage 1
  
    # load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        length_shrink_func=speech_encoder._get_feat_extract_output_lengths,
        model=model
    )

    # start training
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        print("Start training")
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir
    )


if __name__ == "__main__":
    
    train()