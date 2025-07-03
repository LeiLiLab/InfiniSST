from typing import List, Optional, Tuple, Union

import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Qwen2Config, 
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast, 
    CausalLMOutputWithPast
)
from model.flashinfer.modeling_qwen2 import (
    Qwen2Model,
    Qwen2ForCausalLM
)

from train.dataset import (
    DEFAULT_SPEECH_PATCH_TOKEN,
    DEFAULT_SPEECH_START_TOKEN,
    DEFAULT_SPEECH_END_TOKEN,    
    DEFAULT_LATENCY_TOKEN,
    IGNORE_INDEX
)

import logging
logger = logging.getLogger(__name__)

class SpeechQwenFastConfig(Qwen2Config):
    model_type = "SpeechQwenFast"

class SpeechQwenFastModel(Qwen2Model):
    config_class = SpeechQwenFastConfig

    def __init__(self, config: Qwen2Config):
        super(SpeechQwenFastModel, self).__init__(config)
        self.speech_encoder = None

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        return self.speech_encoder._get_feat_extract_output_lengths(input_lengths)    # type: ignore[attr-defined]            
              
    def forward(
        self,
        requests,
        pagetable,
        speech_features=None,
        output_hidden_states=False,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        input_ids = torch.cat([request['input_ids'] for request in requests], dim=0)
        inputs_embeds = self.embed_tokens(input_ids)
        if speech_features is not None:           
            indices = torch.arange(input_ids.shape[0], device=input_ids.device)

            user_mask = input_ids == self.config.user_token_id
            user_pos = indices[user_mask]

            assist_mask = input_ids == self.config.assist_token_id
            assist_pos = indices[assist_mask]

            user_pos = [
                pos for pos in user_pos if input_ids[pos - 1] == self.config.start_header_id
            ]
            assist_pos = [
                pos for pos in assist_pos if input_ids[pos - 1] == self.config.start_header_id
            ]

            offset = 0
            for u_p, a_p in zip(user_pos, assist_pos):
                inputs_embeds[u_p + 2 : a_p - 3] = speech_features[offset : offset + a_p - u_p - 5]
                offset += a_p - u_p - 5

        hidden_state, requests, pagetable, layer_results = super(SpeechQwenFastModel, self).forward(
            inputs_embeds=inputs_embeds,
            requests=requests,
            pagetable=pagetable,
            output_hidden_states=output_hidden_states,
        )
        return hidden_state, requests, pagetable, layer_results
    
class SpeechQwenFastForCausalLM(Qwen2ForCausalLM):
    config_class = SpeechQwenFastConfig

    def __init__(self, config):
        super(SpeechQwenFastForCausalLM, self).__init__(config)
        self.model = SpeechQwenFastModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
    
    def preprocess(self, tokenizer, resize=True):      
        tokenizer.add_tokens(
            [
                DEFAULT_SPEECH_PATCH_TOKEN, 
                DEFAULT_SPEECH_START_TOKEN, 
                DEFAULT_SPEECH_END_TOKEN,                
            ],
            special_tokens=True
        )
        if tokenizer.pad_token_id is None:
            logger.info("No pad token found, adding it")
            tokenizer.add_tokens(
                [tokenizer.pad_token],
                special_tokens=True
            )
        self.resize_token_embeddings(len(tokenizer), mean_resizing=resize)

        sp_patch_token_id, sp_start_token_id, sp_end_token_id = \
            tokenizer.convert_tokens_to_ids(
                [
                    DEFAULT_SPEECH_PATCH_TOKEN, 
                    DEFAULT_SPEECH_START_TOKEN, 
                    DEFAULT_SPEECH_END_TOKEN
                ]
            )
        self.config.sp_patch_token_id = sp_patch_token_id
        self.config.sp_start_token_id = sp_start_token_id
        self.config.sp_end_token_id = sp_end_token_id

        self.config.user_token_id = tokenizer.convert_tokens_to_ids('user')
        self.config.assist_token_id = tokenizer.convert_tokens_to_ids('assistant')
        self.config.start_header_id = tokenizer.convert_tokens_to_ids('<|im_start|>')

    def forward(
        self,
        requests,
        pagetable,
        speech_features=None,
        output_hidden_states=False,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        hidden_state, requests, pagetable, layer_results = self.model(
            requests=requests,
            pagetable=pagetable,
            speech_features=speech_features,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        logits = self.lm_head(hidden_state)

        offset = 0
        logits_list = []
        for request in requests:
            seq_len = request['input_ids'].shape[0]
            logits_list.append(logits[offset + seq_len - 1])
            offset += seq_len

        logits_list = torch.stack(logits_list, dim=0)

        return logits_list, requests, pagetable, layer_results
                   
AutoConfig.register("SpeechQwenFast", SpeechQwenFastConfig)
AutoModelForCausalLM.register(SpeechQwenFastConfig, SpeechQwenFastForCausalLM)
