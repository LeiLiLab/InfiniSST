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
from typing import Iterator, List, Optional, Tuple, Union
import json
import logging
import pathlib
import argparse
from typing import Dict

import torch
import torch.nn as nn

import transformers
from transformers import set_seed
from torch.utils.data import DataLoader
from train.dataset import PromptSpeechToTextDatasetCreator, SpeechToTextDatasetItem
from model.model import SpeechLlamaForCausalLM
from model.speech_encoder import SpeechEncoder
from train.uni_wav2vec_monkey_patch import replace_uni_train
from fairseq.data.audio.speech_to_text_dataset import _collate_frames

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from train.options import add_speech_encoder_args

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


def parse_args():
    parser = argparse.ArgumentParser()
    # lightning module
    add_speech_encoder_args(parser)
    # trainer
    parser.add_argument("--strategy", type=str)
    parser.add_argument("--device-type", type=str)
    parser.add_argument("--n-device", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--save-dir", type=str)
    parser.add_argument("--precision", type=str)
    parser.add_argument("--wandb-run-name", type=str)
    parser.add_argument("--save-step", type=int)
    parser.add_argument("--eval-step", type=int)
    parser.add_argument("--log-step", type=int)
    parser.add_argument("--grad-acc-steps", type=int)
    parser.add_argument("--clip-norm", type=float)
    parser.add_argument("--seed", type=int, default=998244353)
    parser.add_argument("--debug-mode", action="store_true")
    # data
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--train-split", type=str)
    parser.add_argument("--dev-split", type=str)
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--dev-batch-size", type=int)

    args = parser.parse_args()
    return args

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    
    def __call__(self, samples: List[SpeechToTextDatasetItem]) -> Dict[str, torch.Tensor]:
        src_speech = _collate_frames([x.source for x in samples], is_audio_input=True)
        n_frames = torch.tensor([x.source.size(0) for x in samples], dtype=torch.long)

        src_text = [x.src_text for x in samples]
        src_text = self.tokenizer(
            src_text,
            return_tensors="pt",
            padding="longest",
            truncation=False,
            add_special_tokens=False
        ).input_ids

        src_text_len = [(id != self.tokenizer.pad_token_id).sum() for id in src_text]
        src_text_len = torch.tensor(src_text_len, dtype=torch.long)

        text_word = [x.text_word for x in samples]
        speech_word = []
        for i, x in enumerate(samples):
            w = x.speech_word
            if w is not None and len(w) > 0:
                w = torch.FloatTensor(w)
                speech_word.append(w)
            else:
                speech_word.append(None)

        batch = dict(
            src_text=src_text,
            src_speech=src_speech,
            text_word=text_word,
            speech_word=speech_word,
            src_speech_lengths=n_frames,
            src_text_lengths=src_text_len,
        )

        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_path,
    train_split,
    dev_split,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    train_dataset = PromptSpeechToTextDatasetCreator.from_tsv(data_path, train_split)
    dev_dataset = PromptSpeechToTextDatasetCreator.from_tsv(data_path, dev_split)    
    data_collator = DataCollatorForSupervisedDataset(tokenizer)

    return train_dataset, dev_dataset, data_collator

def train():
    args = parse_args()
    # Set seed before initializing model.
    set_seed(args.seed)
    torch.set_float32_matmul_precision('high')

    llm = SpeechLlamaForCausalLM.from_pretrained(
        args.llm_path,
        low_cpu_mem_usage=True,
        load_in_8bit=False,
        device_map='cpu',
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.llm_path,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = "<|finetune_right_pad_id|>"


    train_ds, dev_ds, collate_fn = make_supervised_data_module(
        tokenizer=tokenizer,
        data_path=args.data_path,
        train_split=args.train_split,
        dev_split=args.dev_split,
    )

    model = SpeechEncoder(
        args.feature_extractor_cfg,
        args.feature_extractor_state_dict_path,
        args.feature_extractor_freeze,
        args.length_shrink_cfg,
        
        args.n_attn_layers,
        args.n_dim,
        args.n_heads,
        args.dropout,
        args.block_size,
        args.max_cache_size,      
        copy.deepcopy(llm.model.embed_tokens),

        train_ds, 
        dev_ds, 
        args.train_batch_size, 
        args.dev_batch_size, 
        collate_fn,
        
        args.lr,
        args.warmup_updates,
        args.min_lr,
        args.temp,
        args.loss_fn,
    )
    del llm

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_dir,
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        every_n_train_steps=args.save_step
    )
    lr_monitor = LearningRateMonitor(
        logging_interval='step'
    )

    wandb_logger = WandbLogger(
        name=args.wandb_run_name,
        # log_model="all"
    )

    trainer = L.Trainer(
        accelerator=args.device_type,
        devices=args.n_device,
        strategy=args.strategy,
        precision=args.precision,
        max_steps=args.max_steps,
        accumulate_grad_batches=args.grad_acc_steps,
        gradient_clip_val=args.clip_norm,
        use_distributed_sampler=False,
        default_root_dir=args.save_dir,
        log_every_n_steps=args.log_step,
        val_check_interval=args.eval_step,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        fast_dev_run=args.debug_mode,
    )

    if os.path.exists(args.save_dir) and len(os.listdir(args.save_dir)) >= 1:
        ckpt_path = os.path.join(args.save_dir, os.listdir(args.save_dir)[0])
        trainer.fit(model, ckpt_path=ckpt_path)
    else:
        trainer.fit(model)


if __name__ == "__main__":
    train()