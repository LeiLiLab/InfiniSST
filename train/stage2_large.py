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
import copy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

import transformers
from transformers import Trainer, set_seed
from torch.utils.data import Dataset
import conversation as conversation_lib
from train.dataset import PromptSpeechToTextDatasetCreator, SpeechToTextDatasetItem
from model.model import SpeechLlamaForCausalLM
from fairseq.data.audio.speech_to_text_dataset import _collate_frames
# from train.uni_wav2vec_monkey_patch import replace_uni_train
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
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    freeze_speech_foundation: bool = field(default=False)
    freeze_length_adapter: bool = field(default=False)
    freeze_mm_adapter: bool = field(default=False)
    only_tune_adapter: bool = field(default=False)
    speech_tower_path: Optional[str] = field(default=None)
    speech_tower_type: Optional[str] = field(default=None)
    ssl_fintuned: bool = field(default=False)
    pretrain_mm_adapter: Optional[str] = field(default=None)
    len_adapter_channels: int = field(
        default=1024,
        metadata={"help": "# of channels in the Length adapter (Conv1d)"}
    )
    len_adapter_kernel_sizes: str = field(
        default="3,3",
        metadata={"help": "kernel sizes of the Length adapter (Conv1d)"}
    )    
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    unidirectional: bool = field(default=False)
    blocksize: int = field(default=1)


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
    freeze_mm_mlp_adapter: bool = field(default=False)
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

    # tokenizer: transformers.PreTrainedTokenizer
    # length_after_ssl: None
    # length_after_adp: None
    # model: SpeechLlamaForCausalLM
    def __init__(self, tokenizer, length_after_ssl, length_after_adp, model, source_lang, target_lang):
        self.tokenizer = tokenizer
        self.length_after_ssl = length_after_ssl
        self.length_after_adp = length_after_adp
        self.model = model
        self.source_lang = source_lang
        self.target_lang = target_lang    
 
    def reset_speech_features_flag(self):
        self.model.model.speech_features_extracted = False
    
    # def __call__(self, samples: List[SpeechToTextDatasetItem]) -> Dict[str, torch.Tensor]:
    #     # todo: sort samples by descending number of frames
    #     self.reset_speech_features_flag()
    #     indices = torch.tensor([x.index for x in samples], dtype=torch.long)
    #     speech_batch = _collate_frames([x.source for x in samples], is_audio_input=True)
    #     n_frames = torch.tensor([x.source.size(0) for x in samples], dtype=torch.long)
    #     speech_lens = self.length_after_adp(self.length_after_ssl(n_frames)) # after forward ssl model and length adapter

    #     texts = [x.target for x in samples]
     
    #     to_adds = [int(speech_len)*DEFAULT_SPEECH_PATCH_TOKEN for speech_len in speech_lens]
    #     to_adds = [DEFAULT_SPEECH_START_TOKEN + to_add + DEFAULT_SPEECH_END_TOKEN for to_add in to_adds]


    #     conv = conversation_lib.default_conversation.copy()
    #     # conv = random.choice(list(conversation_lib.conv_templates.values())).copy()
    #     conversations = []
    #     for to_add, text in zip(to_adds, texts):
    #         conv.messages = []
    #         # before, after = prompt.split('<speech_here>')
    #         # mm_prompt = before + to_add + after
    #         conv.append_message(conv.roles[0], to_add)
    #         conv.append_message(conv.roles[1], text)
    #         conversations.append(conv.get_prompt())
    #     input_ids = self.tokenizer(
    #         conversations,
    #         return_tensors="pt",
    #         padding="longest",
    #         #max_length=tokenizer.model_max_length,
    #         truncation=False,
    #     ).input_ids
    #     targets = input_ids.clone()
    #     sep = conv.sep + conv.roles[1] + ": "
    #     for conversation, target in zip(conversations, targets):
    #         total_len = int(target.ne(self.tokenizer.pad_token_id).sum())
    #         rounds = conversation.split(conv.sep2)
    #         cur_len = 1
    #         target[:cur_len] = IGNORE_INDEX
    #         for i, rou in enumerate(rounds):
    #             if rou == "":
    #                 break
    #             parts = rou.split(sep)
    #             if len(parts) != 2:
    #                 break
    #             parts[0] += sep
    #             round_len = len(self.tokenizer(rou).input_ids)
    #             instruction_len = len(self.tokenizer(parts[0]).input_ids) - 2
    #             target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
    #             cur_len += round_len
    #         # target[cur_len:] = IGNORE_INDEX
    #                     # Ensure the final EOS token is always in targets
    #         if cur_len < total_len:
    #             target[cur_len:total_len-1] = IGNORE_INDEX
    #             target[total_len-1] = self.tokenizer.eos_token_id

    #     #print("conversations:", conversations[0])
    #     #print("input_ids:", input_ids[0])
    #     #print("targets:", targets[0])
    #     #print(self.tokenizer.convert_ids_to_tokens(input_ids[0]))
                
    #     batch = dict(
    #         input_ids=input_ids,
    #         labels=targets,
    #         attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
    #         speech_batch=speech_batch,
    #         src_lengths=n_frames, # src length,ssl_fintuned
    #         after_lens=speech_lens, # length after forward ssl and adapter
    #     )

    #     return batch

    def __call__(self, samples: List[SpeechToTextDatasetItem]) -> Dict[str, torch.Tensor]:
        self.reset_speech_features_flag()
        indices = torch.tensor([x.index for x in samples], dtype=torch.long)
        speech_batch = _collate_frames([x.source for x in samples], is_audio_input=True)
        n_frames = torch.tensor([x.source.size(0) for x in samples], dtype=torch.long)
        speech_lens = self.length_after_adp(self.length_after_ssl(n_frames))

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
                                length_after_ssl,
                                length_after_adp,
                                model) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    train_dataset = PromptSpeechToTextDatasetCreator.from_tsv(data_args.data_path, data_args.data_split_train)
    eval_dataset = PromptSpeechToTextDatasetCreator.from_tsv(data_args.data_path, data_args.data_split_eval) if data_args.data_split_eval is not None else None
    data_collator = DataCollatorForSupervisedDataset(
        tokenizer, 
        length_after_ssl, 
        length_after_adp, 
        model,
        data_args.source_lang,
        data_args.target_lang
    )
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set seed before initializing model.   """
    set_seed(training_args.seed)
    # load model
    config = json.load(open(os.path.join(model_args.model_name_or_path, 'config.json')))
    config['large_model'] = True
    update_config = os.path.join(model_args.model_name_or_path, 'config_large.json')
    json.dump(config, open(update_config, 'w'), indent=2)  
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if world_size != 1 else "auto"

    # replace uni wav2vec forward
    if model_args.unidirectional:
        replace_uni_train(model_args.blocksize)
    model = SpeechLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        #low_cpu_mem_usage=True,
        config=update_config,
        #device_map=device_map,
    )
    length_after_ssl, length_after_adp = model.model.initialize_speech_modules(
        '/data/user_data/yuanjinw/models/wav2_vec_vox_960h_pl.pt',
        speech_tower_type=None,
        len_adapter_channels=model.config.len_adapter_channels,
        len_adapter_kernel_sizes=model.config.len_adapter_kernel_sizes,
        ssl_fintuned=model.config.ssl_fintuned,
    )
    
    length_adapter_weights = torch.load(os.path.join(model_args.model_name_or_path, 'length_adapter.bin'), map_location='cpu')
    mlp_adapter_weights = torch.load(os.path.join(model_args.model_name_or_path, 'mlp_adapter.bin'), map_location='cpu')
    speech_tower_weights = torch.load(os.path.join(model_args.model_name_or_path, 'speech_tower.bin'), map_location='cpu')
    
    model.model.mm_length_adapter.load_state_dict(length_adapter_weights)
    model.model.mm_mlp_adapter.load_state_dict(mlp_adapter_weights)
    model.model.speech_tower.load_state_dict(speech_tower_weights)

    model.config.use_cache = False
    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = "<|finetune_right_pad_id|>"

    model.model.speech_tower.to(device=training_args.device)
    model.model.mm_length_adapter.to(device=training_args.device)
    model.model.mm_mlp_adapter.to(device=training_args.device)  
    
    if model_args.freeze_backbone: # freeze llama model
        model.model.requires_grad_(False)
        model.lm_head.requires_grad_(False)
        model.model.speech_tower.requires_grad_(True)
        model.model.mm_length_adapter.requires_grad_(True) 
        model.model.mm_mlp_adapter.requires_grad_(True) 
          
    if model_args.freeze_speech_foundation: # freeze speech foundation model  
        model.model.speech_tower.requires_grad_(False) 
    else: # train transformer encoder
        model.model.speech_tower.requires_grad_(False)
        for param in model.model.speech_tower.encoder.parameters():
            param.requires_grad = True
        for param in model.model.speech_tower.layer_norm.parameters():
            param.requires_grad = True
    
    if model_args.freeze_length_adapter: # freeze length adapter 
        model.model.mm_length_adapter.requires_grad_(False)  
    if model_args.freeze_mm_adapter: # freeze mm adapter 
        model.model.mm_mlp_adapter.requires_grad_(False)          
        
    model.initialize_speech_tokenizer(tokenizer=tokenizer, device=training_args.device,
                                      only_tune_adapter=model_args.only_tune_adapter, stage1=False)
                                                   

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args,
                                              length_after_ssl=length_after_ssl,
                                              length_after_adp=length_after_adp,
                                              model=model)
    trainer = Trainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()