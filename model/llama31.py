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
    LlamaModel, 
    LlamaForCausalLM
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

class SpeechLlamaConfig(LlamaConfig):
    model_type = "SpeechLlama"

class SpeechLlamaModel(LlamaModel):
    config_class = SpeechLlamaConfig

    def __init__(self, config: LlamaConfig):
        super(SpeechLlamaModel, self).__init__(config)
        self.speech_features_extracted = False
        self.inference = False

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        return self.speech_encoder._get_feat_extract_output_lengths(input_lengths)                
              
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        speech_batch: Optional[torch.FloatTensor] = None,
        src_lengths: Optional[List[torch.FloatTensor]] = None,
        after_lens: Optional[List[torch.FloatTensor]] = None,
        return_dict: Optional[bool] = None,
        states: Optional[object] = None,
        multiplier: Optional[int] = 1,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        if speech_batch is not None and not self.speech_features_extracted:
            self.speech_encoder.set_blocksize(multiplier)
            if states is None:
                speech_features, _ = self.speech_encoder.encode_speech(
                    speech_batch, 
                    src_lengths,
                )
            else:
                speech_features, states.speech_cache = self.speech_encoder.encode_speech(
                    speech_batch, 
                    src_lengths,
                    cache=states.speech_cache
                )

            if self.inference:
                self.speech_features_extracted = True

            inputs_embeds = self.embed_tokens(input_ids)
            filled_inputs_embeds = []
            for i in range(input_ids.size(0)):
                user_pos = (input_ids[i] == self.config.user_token_id).nonzero()
                assist_pos = (input_ids[i] == self.config.assist_token_id).nonzero()

                user_pos = [
                    pos for pos in user_pos if input_ids[i, pos[0] - 1] == self.config.start_header_id
                ]
                assist_pos = [
                    pos for pos in assist_pos if input_ids[i, pos[0] - 1] == self.config.start_header_id
                ]

                filled_inputs_embed = inputs_embeds[i]
                index = 0
                for u_p, a_p in zip(user_pos, assist_pos):
                    filled_inputs_embed = torch.cat(
                        [
                            filled_inputs_embed[: u_p + 3],
                            speech_features[i, index : index + a_p - u_p - 5],
                            filled_inputs_embed[a_p - 2 :]
                        ],
                        dim=0                            
                    )
                    index += a_p - u_p - 5
                filled_inputs_embeds.append(filled_inputs_embed)

            inputs_embeds = torch.stack(filled_inputs_embeds)
        else:
            inputs_embeds = self.embed_tokens(input_ids[:, -1:])

        return super(SpeechLlamaModel, self).forward(
            input_ids=None, 
            attention_mask=None,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, 
            use_cache=use_cache,
            output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
    
class SpeechLlamaForCausalLM(LlamaForCausalLM):
    config_class = SpeechLlamaConfig

    def __init__(self, config):
        super(SpeechLlamaForCausalLM, self).__init__(config)
        self.model = SpeechLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
        
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def get_output_embeddings(self):
        return self.lm_head
    
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
        input_ids: torch.LongTensor = None,
        text_input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        text_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        speech_batch: Optional[torch.FloatTensor] = None,
        src_lengths: Optional[List[torch.FloatTensor]] = None,
        after_lens: Optional[List[torch.FloatTensor]] = None,
        return_dict: Optional[bool] = None,
        states: Optional[object] = None,
        multiplier: Optional[int] = 1,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            speech_batch=speech_batch,
            src_lengths=src_lengths,
            after_lens=after_lens,
            states=states,
            multiplier=multiplier,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.transpose(-1, -2), shift_labels)

            if self.cpo_beta > 0:
                bsz = labels.size(0) // 2
                loss = loss.sum(dim=-1)
                logp_w = -loss[:bsz]
                logp_l = -loss[bsz:]
                cpo_loss = -F.logsigmoid(self.cpo_beta * (logp_w - logp_l)).mean()
                nll_loss = -logp_w.sum() / (shift_labels[:bsz] != IGNORE_INDEX).sum()
                loss = nll_loss + cpo_loss
                logger.info("nll_loss {:.2f} cpo_loss {:.2f}".format(nll_loss.item(), cpo_loss.item()))
            else:
                loss = loss.sum() / (shift_labels != IGNORE_INDEX).sum()
            
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # if past_key_values:
        #     input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": True,
                "attention_mask": attention_mask,
                "speech_batch": kwargs.get("speech_batch", None),
                "src_lengths": kwargs.get("src_lengths", None),
                "after_lens": kwargs.get("after_lens", None),
                "states": kwargs.get("states", None),
                "multiplier": kwargs.get("multiplier", 1),
            }
        )
        return model_inputs
    
                   
AutoConfig.register("SpeechLlama", SpeechLlamaConfig)
AutoModelForCausalLM.register(SpeechLlamaConfig, SpeechLlamaForCausalLM)
