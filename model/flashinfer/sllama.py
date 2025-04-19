from typing import List, Optional, Tuple, Union

import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig, 
)
from model.flashinfer.modeling_llama import (
    LlamaModel, LlamaForCausalLM
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast, 
    CausalLMOutputWithPast
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

class SpeechLlamaFastConfig(LlamaConfig):
    model_type = "SpeechLlamaFast"

class SpeechLlamaFastModel(LlamaModel):
    config_class = SpeechLlamaFastConfig

    def __init__(self, config: LlamaConfig):
        super(SpeechLlamaFastModel, self).__init__(config)
        self.speech_encoder = None

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        return self.speech_encoder._get_feat_extract_output_lengths(input_lengths)
              
    def forward(
        self,
        requests,
        pagetable,
        speech_features=None,
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

            for i, (u_p, a_p) in enumerate(zip(user_pos, assist_pos)):
                inputs_embeds[u_p + 3 : a_p - 2] = speech_features[i]

        hidden_state, pagetable = super(SpeechLlamaFastModel, self).forward(
            inputs_embeds=inputs_embeds,
            requests=requests,
            pagetable=pagetable,
        )
        return hidden_state, pagetable
    
    
class SpeechLlamaFastForCausalLM(LlamaForCausalLM):
    config_class = SpeechLlamaFastConfig

    def __init__(self, config):
        super(SpeechLlamaFastForCausalLM, self).__init__(config)
        self.model = SpeechLlamaFastModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
    
    def preprocess(self, tokenizer, max_multiplier=4, resize=True):      
        tokenizer.add_tokens(
            [
                DEFAULT_SPEECH_PATCH_TOKEN, 
                DEFAULT_SPEECH_START_TOKEN, 
                DEFAULT_SPEECH_END_TOKEN,                
            ] + [
                DEFAULT_LATENCY_TOKEN.format(i)
                for i in range(1, max_multiplier + 1)
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
        self.config.start_header_id = tokenizer.convert_tokens_to_ids('<|start_header_id|>')

        self.config.latency_token_ids = tokenizer.convert_tokens_to_ids(
            [
                DEFAULT_LATENCY_TOKEN.format(i)
                for i in range(1, max_multiplier + 1)
            ]
        )

    def forward(
        self,
        requests,
        pagetable,
        speech_features=None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        hidden_state, pagetable = self.model(
            requests=requests,
            pagetable=pagetable,
            speech_features=speech_features,
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

        return logits_list, pagetable
    
AutoConfig.register("SpeechLlamaFast", SpeechLlamaFastConfig)
AutoModelForCausalLM.register(SpeechLlamaFastConfig, SpeechLlamaFastForCausalLM)
