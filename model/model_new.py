import os
import copy
import random
import time
import fairseq
from typing import List, Optional, Tuple, Union
from lightning.pytorch.utilities.types import EVAL_DATALOADERS

import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from fairseq.models.speech_to_text import (
    lengths_to_padding_mask,
    Conv1dSubsampler,
)
from fairseq.models.wav2vec import Wav2VecEncoder
from fairseq.models.hubert import HubertEncoder
from fairseq.optim.lr_scheduler.inverse_square_root_schedule import (
    InverseSquareRootLRScheduleConfig,
    InverseSquareRootSchedule
)
from torch.optim.optimizer import Optimizer
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

DEFAULT_SPEECH_PATCH_TOKEN = "<sp_patch>"
DEFAULT_SPEECH_START_TOKEN = "<sp_start>"
DEFAULT_SPEECH_END_TOKEN = "<sp_end>"

class SpeechLlamaConfig(LlamaConfig):
    model_type = "SpeechLlama"
    inference = False

class SpeechLlamaModel(LlamaModel):
    config_class = SpeechLlamaConfig

    def __init__(self, config: LlamaConfig):
        super(SpeechLlamaModel, self).__init__(config)
        self.speech_features_extracted = False

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        return self.speech_encoder._get_feat_extract_output_lengths(input_lengths)                
       
    def get_hubert_features(self, src_tokens, src_lengths):
        padding_mask = lengths_to_padding_mask(src_lengths)
        hubert_args = {
            "source": src_tokens,
            "padding_mask": padding_mask,
            "mask": False,
        }
        x, padding_mask = self.hubert_model.extract_features(**hubert_args)
        output_length = (1 - padding_mask.int()).sum(dim=1)
        return x, padding_mask, output_length
              
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
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
    
        orig_embeds_params = getattr(self, 'orig_embeds_params', False)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if speech_batch is not None:
            speech_features = None
            if self.training or not self.speech_features_extracted:
                speech_features, _ = self.speech_encoder.encode_speech(speech_batch, src_lengths)

            if self.config.inference:
                self.speech_features_extracted = True
                
            # inputs_embeds: B*T*d
            # speech_features: B*T1*d
            if speech_features is not None:

                speech_start_pos = torch.where(input_ids[0] == self.config.sp_start_token_id)[0]
                speech_end_pos = torch.where(input_ids[0] == self.config.sp_end_token_id)[0]

                if orig_embeds_params:
                    inputs_embeds = torch.cat(
                        (
                            inputs_embeds[:, :speech_start_pos].detach(), 
                            inputs_embeds[:, speech_start_pos], 
                            speech_features, 
                            inputs_embeds[:, speech_end_pos], 
                            inputs_embeds[:, speech_end_pos + 1:].detach()
                        ), 
                        dim=1
                    )
                else:
                    inputs_embeds = torch.cat(
                        (
                            inputs_embeds[:, :speech_start_pos+1], 
                            speech_features, 
                            inputs_embeds[:, speech_end_pos:]
                        ), 
                        dim=1
                    )

        return super(SpeechLlamaModel, self).forward(
            input_ids=None, 
            # position_ids=position_ids[:, -inputs_embeds.size(1):], # only ignore for sid's model
            attention_mask=attention_mask,
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
    
    def preprocess(self, tokenizer):      
        num_new_tokens = tokenizer.add_tokens([DEFAULT_SPEECH_PATCH_TOKEN, DEFAULT_SPEECH_START_TOKEN, DEFAULT_SPEECH_END_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer), mean_resizing=True)
        # input_embeddings = self.get_input_embeddings().weight.data
        # output_embeddings = self.get_output_embeddings().weight.data

        # input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
        #         dim=0, keepdim=True)
        # output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
        #         dim=0, keepdim=True)

        # input_embeddings[-num_new_tokens:] = input_embeddings_avg
        # output_embeddings[-num_new_tokens:] = output_embeddings_avg

        sp_patch_token_id, sp_start_token_id, sp_end_token_id = tokenizer.convert_tokens_to_ids([DEFAULT_SPEECH_PATCH_TOKEN, DEFAULT_SPEECH_START_TOKEN, DEFAULT_SPEECH_END_TOKEN])                
        self.config.sp_patch_token_id = sp_patch_token_id
        self.config.sp_start_token_id = sp_start_token_id
        self.config.sp_end_token_id = sp_end_token_id 

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
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if getattr(self, "rdrop", 0.) != 0:
            with torch.no_grad():
                outputs_2 = self.model(
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
                )

                hidden_states_2 = outputs_2[0]
                logits_2 = self.lm_head(hidden_states_2)

            log_probs = F.log_softmax(logits, dim=-1)
            log_probs_2 = F.log_softmax(logits_2, dim=-1)

            rdrop_loss_fct = nn.KLDivLoss(log_target=True)
            rdrop_loss = rdrop_loss_fct(log_probs, log_probs_2) + \
                rdrop_loss_fct(log_probs_2, log_probs)
            # print(loss, rdrop_loss)
            loss = loss + self.rdrop * rdrop_loss

        if getattr(self, "text_weight", 0.) != 0:
            outputs_3 = self.model(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                states=states,
            )

            hidden_states_3 = outputs_3[0]
            logits_3 = self.lm_head(hidden_states_3)

            loss_3 = None
            if text_labels is not None:
                # Shift so that tokens < n predict n
                shift_logits_3 = logits_3[..., :-1, :].contiguous()
                shift_labels_3 = text_labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits_3 = shift_logits_3.view(-1, self.config.vocab_size)
                shift_labels_3 = shift_labels_3.view(-1)
                # Enable model/pipeline parallelism
                shift_labels_3 = shift_labels_3.to(shift_logits_3.device)
                loss_3 = loss_fct(shift_logits_3, shift_labels_3)

            if wandb.run is not None:
                log_key = "mt_train_loss" if self.training else "mt_eval_loss"
                wandb.log({log_key: loss_3.item()})

            loss = loss + self.text_weight * loss_3

        # print(loss, self.training)

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
        if past_key_values:
            input_ids = input_ids[:, -1:]

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
            }
        )
        return model_inputs        
    
                   
AutoConfig.register("SpeechLlama", SpeechLlamaConfig)
AutoModelForCausalLM.register(SpeechLlamaConfig, SpeechLlamaForCausalLM)
