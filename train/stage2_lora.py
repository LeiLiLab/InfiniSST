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

import os, sys
os.environ['WANDB_DISABLED'] = 'true'
sys.path.append('/home/xixu/sllama')
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

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
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
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    freeze_speech_foundation: bool = field(default=False)
    only_tune_adapter: bool = field(default=False)
    speech_tower_path: Optional[str] = field(default=None)
    speech_tower_type: Optional[str] = field(default=None)
    pretrain_mm_adapter: Optional[str] = field(default=None)
    len_adapter_channels: int = field(
        default=1024,
        metadata={"help": "# of channels in the Length adapter (Conv1d)"}
    )
    len_adapter_kernel_sizes: str = field(
        default="3,3",
        metadata={"help": "kernel sizes of the Length adapter (Conv1d)"}
    )    

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    data_split: str = field(default=None,
                           metadata={"help": "Path to the training data."})
                           
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

    tokenizer: transformers.PreTrainedTokenizer
    length_after_ssl: None
    length_after_adp: None
    prompt_template = 'USER: {} Assiant: <s>'
    # text = "hello"
    # if without <s>, tokenizer(prompt_template+text) != tokenizer(prompt_template) + tokenizer(text)
    end_sym = "</s>"
    prompt_list = ['<speech_here> Describe the speech concisely', '<speech_here> describe with you see']

    def __call__(self, samples: List[SpeechToTextDatasetItem]) -> Dict[str, torch.Tensor]:
        # todo: sort samples by descending number of frames
        indices = torch.tensor([x.index for x in samples], dtype=torch.long)
        speech_batch = _collate_frames([x.source for x in samples], is_audio_input=True)
        n_frames = torch.tensor([x.source.size(0) for x in samples], dtype=torch.long)
        speech_lens = self.length_after_adp(self.length_after_ssl(n_frames)) # after forward ssl model and length adapter

        texts = [x.target for x in samples]
        texts = [text + self.end_sym for text in texts]
           
        to_adds = [int(speech_len)*DEFAULT_SPEECH_PATCH_TOKEN for speech_len in speech_lens]
        to_adds = [DEFAULT_SPEECH_START_TOKEN + to_add + DEFAULT_SPEECH_END_TOKEN for to_add in to_adds]

        prompt = self.prompt_template.format(self.prompt_list[0])
        before, after = prompt.split('<speech_here>')
        mm_prompts = [before + to_add + after for to_add in to_adds]
        # USER: <sp><sp_patch></sp> prompt_text Assiant: <s> 
        all_ = [mm_prompt + text for mm_prompt,text in zip(mm_prompts, texts)]  
        # USER: <sp><sp_patch></sp> prompt_text Assiant: <s>text
        all_tokens = self.tokenizer(all_,
                                    return_tensors="pt",
                                    padding="longest",
                                    truncation=False,)        

        mm_prompt_tokens = self.tokenizer(mm_prompts,
                                          return_tensors="pt",
                                          padding="longest",
                                          truncation=False,)
                                          
        labels = copy.deepcopy(all_tokens.input_ids) 
        for i, label in enumerate(labels):   
            mm_prompt_len = mm_prompt_tokens.input_ids[i].ne(self.tokenizer.pad_token_id).sum().item()
            label[:mm_prompt_len] = IGNORE_INDEX
            
        input_ids = all_tokens.input_ids 

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            speech_batch=speech_batch,
            src_lengths=n_frames, # src length,
            after_lens=speech_lens, # length after forward ssl and adapter
        )      

        return batch

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args,
                                length_after_ssl,
                                length_after_adp) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    train_dataset = PromptSpeechToTextDatasetCreator.from_tsv(data_args.data_path, data_args.data_split)
    data_collator = DataCollatorForSupervisedDataset(tokenizer, length_after_ssl, length_after_adp)

    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set seed before initializing model.
    set_seed(training_args.seed)
    # load model   
    model = SpeechLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    length_after_ssl, length_after_adp = model.model.length_after_ssl, model.model.length_after_adp
            
    model.config.use_cache = False
    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token

    if model_args.freeze_backbone: # freeze llama model, adapter and ssl model is not added here
        model.model.requires_grad_(False)

    model.model.speech_tower.to(device=training_args.device)
    model.model.mm_length_adapter.to(device=training_args.device)
    model.model.mm_mlp_adapter.to(device=training_args.device) 
       
    if model_args.freeze_speech_foundation: # freeze the ssl model after add the ssl model by initialize_speech_modules   
        model.model.speech_tower.requires_grad_(False) 
         
    # lora training
    if training_args.use_lora:
        lora_config = json.load(open(training_args.lora_config))
        config = LoraConfig(
            r=lora_config['lora_r'],
            lora_alpha=lora_config['lora_alpha'],
            target_modules=lora_config['lora_target_modules'],
            lora_dropout=lora_config['lora_dropout'],
            bias="none",
            task_type="CAUSAL_LM",
        )
        # "RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn"
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        model = get_peft_model(model, config)
        model.print_trainable_parameters()    
                                                      
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args,
                                              length_after_ssl=length_after_ssl,
                                              length_after_adp=length_after_adp)
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
