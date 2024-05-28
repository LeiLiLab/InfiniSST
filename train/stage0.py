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
from model.model import SpeechEncoder, SpeechLlamaForCausalLM
from train.uni_wav2vec_monkey_patch import replace_uni_train
from fairseq.data.audio.speech_to_text_dataset import _collate_frames

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

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
    parser.add_argument("--llm-path", type=str)
    parser.add_argument("--speech-encoder-path", type=str)
    parser.add_argument("--ssl-finetuned", action="store_true")
    parser.add_argument("--len-adapter-channels", type=int, default=1024)
    parser.add_argument("--len-adapter-kernel-sizes", type=str, default="3,3")
    parser.add_argument("--unidirectional", action="store_true")
    parser.add_argument("--blocksize", type=int, default=1)
    parser.add_argument("--temp", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--warmup-updates", type=int)
    parser.add_argument("--loss-fn", type=str, default='waco')
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
    length_after_ssl: None
    length_after_adp: None
    prompt_list_asr = ['<speech_here> Try to decipher the spoken language and write it down.']
    prompt_list_st = ['<speech_here>']

    
    def __call__(self, samples: List[SpeechToTextDatasetItem]) -> Dict[str, torch.Tensor]:
        # todo: sort samples by descending number of frames
        indices = torch.tensor([x.index for x in samples], dtype=torch.long)
        src_speech = _collate_frames([x.source for x in samples], is_audio_input=True)
        n_frames = torch.tensor([x.source.size(0) for x in samples], dtype=torch.long)
        speech_lens = self.length_after_adp(self.length_after_ssl(n_frames)) # after forward ssl model and length adapter

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
                duration = n_frames[i] / 16000
                w /= duration
                w[:, 0] = (w[:, 0] * speech_lens[i]).floor()
                w[:, 1] = (w[:, 1] * speech_lens[i]).ceil() - 1
                speech_word.append(w.long())
            else:
                speech_word.append(None)

        batch = dict(
            src_text=src_text,
            src_speech=src_speech,
            text_word=text_word,
            speech_word=speech_word,
            src_speech_lengths=n_frames,
            after_speech_lengths=speech_lens,
            src_text_lengths=src_text_len,
        )

        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_path,
    train_split,
    dev_split,
    length_after_ssl,
    length_after_adp,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    train_dataset = PromptSpeechToTextDatasetCreator.from_tsv(data_path, train_split)
    dev_dataset = PromptSpeechToTextDatasetCreator.from_tsv(data_path, dev_split)    
    data_collator = DataCollatorForSupervisedDataset(tokenizer, length_after_ssl, length_after_adp)

    return train_dataset, dev_dataset, data_collator


    train_sampler = SpeechSampler(train_dataset, shuffle=True, batch_size=train_batch_size)
    dev_sampler = SpeechSampler(dev_dataset, shuffle=False, batch_size=dev_batch_size)

    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=data_collator)
    dev_dataloader = DataLoader(dev_dataset, batch_sampler=dev_sampler, collate_fn=data_collator)

    return train_dataloader, dev_dataloader

def train():
    args = parse_args()
    # Set seed before initializing model.
    set_seed(args.seed)
    torch.set_float32_matmul_precision('high')

    if args.unidirectional:
        replace_uni_train(args.blocksize)

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
    tokenizer.pad_token = tokenizer.eos_token

    model = SpeechEncoder(
        args.speech_encoder_path,
        args.ssl_finetuned,
        args.len_adapter_channels,
        args.len_adapter_kernel_sizes,
        copy.deepcopy(llm.model.embed_tokens),
        args.unidirectional,
        args.temp,
        lr=args.lr,
        warmup_updates=args.warmup_updates,
        loss_fn=args.loss_fn,
    )
    del llm
    model.speech_tower.requires_grad_(False)
    for param in model.speech_tower.encoder.parameters():
        param.requires_grad = True
    for param in model.speech_tower.layer_norm.parameters():
        param.requires_grad = True

    data = make_supervised_data_module(
        tokenizer=tokenizer,
        data_path=args.data_path,
        train_split=args.train_split,
        dev_split=args.dev_split,
        length_after_ssl=model.length_after_ssl,
        length_after_adp=model.length_after_adp,
    )

    model.train_ds, model.dev_ds, model.collate = data
    model.train_bsz = args.train_batch_size
    model.dev_bsz = args.dev_batch_size

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
        callbacks=[checkpoint_callback, lr_monitor]
    )

    trainer.fit(model)


if __name__ == "__main__":
    train()