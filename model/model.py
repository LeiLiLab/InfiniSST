import fairseq
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from fairseq.models.speech_to_text import (
    lengths_to_padding_mask,
    Conv1dSubsampler,
)
from fairseq.models.wav2vec import Wav2VecEncoder
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

DEFAULT_SPEECH_PATCH_TOKEN = "<sp_patch>"
DEFAULT_SPEECH_START_TOKEN = "<sp_start>"
DEFAULT_SPEECH_END_TOKEN = "<sp_end>"


class SpeechLlamaConfig(LlamaConfig):
    model_type = "SpeechLlama"


class SpeechLlamaModel(LlamaModel):
    config_class = SpeechLlamaConfig

    def __init__(self, config: LlamaConfig, mm_vision_tower=None, mm_hidden_size=None):
        super(SpeechLlamaModel, self).__init__(config)
        large_model = getattr(config, 'large_model', False)
        lora_train = getattr(config, 'lora_train', False)
        if hasattr(config, "stage1_complete") and not large_model and not lora_train:
            ssl_fintuned = getattr(config, 'ssl_fintuned', False)
            self.length_after_ssl, self.length_after_adp = self.initialize_speech_modules(config.speech_tower_path, None,
                                                         config.len_adapter_channels, config.len_adapter_kernel_sizes,
                                                         config.stage1_complete, ssl_fintuned)      
        self.speech_features_extracted = False

    def initialize_speech_modules(self, speech_tower_path, speech_tower_type=None,
                                   len_adapter_channels=None, len_adapter_kernel_sizes=None,
                                   stage1_complete=False, ssl_fintuned=False):
        # loading pretrained ssl model
        # wav2vec 2.0
        if not ssl_fintuned: # ssl model
            state = fairseq.checkpoint_utils.load_checkpoint_to_cpu(speech_tower_path)
            w2v_args = state["args"]
            task = fairseq.tasks.setup_task(w2v_args)
            model = task.build_model(w2v_args)
            model.load_state_dict(state["model"], strict=True)
            speech_dimension = w2v_args.encoder_embed_dim
        else: # ctc finetune, w2v-ctc
            state = fairseq.checkpoint_utils.load_checkpoint_to_cpu(speech_tower_path)
            model = Wav2VecEncoder(state['cfg']['model'], None)
            new = {}
            for key in state['model'].keys():
                new_key = key.replace('w2v_encoder.', '')
                if not new_key.startswith('proj'):
                    new[new_key] = state['model'][key]
            model.load_state_dict(new, strict=True)
            model = model.w2v_model
            speech_dimension = state['cfg']['model']['w2v_args']['model'].encoder_embed_dim
              
        self.speech_tower = model
        self.mm_length_adapter = Conv1dSubsampler(
                                     speech_dimension,
                                     len_adapter_channels,
                                     speech_dimension,
                                     [int(k) for k in len_adapter_kernel_sizes.split(',')]
                                 ) 
        self.mm_mlp_adapter = nn.Linear(speech_dimension, self.config.hidden_size)
        length_after_ssl = self.speech_tower._get_feat_extract_output_lengths
        length_after_adp = self.mm_length_adapter.get_out_seq_lens_tensor
        
        if not stage1_complete:
            self.config.speech_tower_path = speech_tower_path
            self.config.len_adapter_channels = len_adapter_channels
            self.config.len_adapter_kernel_sizes = len_adapter_kernel_sizes
            self.config.stage1_complete = True
            self.config.ssl_fintuned = ssl_fintuned
                  
        return (length_after_ssl, length_after_adp) 
                
    def get_ssl_feature_w2v(self, src_tokens, src_lengths, after_lens):
        padding_mask = lengths_to_padding_mask(src_lengths)
        res = self.speech_tower.extract_features(src_tokens, padding_mask)
        feature, padding_mask = res["x"], res["padding_mask"]
        if padding_mask is None:
        # Create a padding mask of shape [batch_size, seq_length] with all False values
            padding_mask = torch.zeros(feature.shape[:2], dtype=torch.bool, device=feature.device)
        output_length = (1 - padding_mask.int()).sum(dim=1)
        feature, input_lengths = self.mm_length_adapter(feature, output_length)
        assert after_lens.equal(input_lengths), "pre calculate length not match with the forward length"
        feature = self.mm_mlp_adapter(feature)       
        return feature
        
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
    
    def forward_incremental(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        speech_batch: Optional[torch.FloatTensor] = None,
        src_lengths: Optional[List[torch.FloatTensor]] = None,
        after_lens: Optional[List[torch.FloatTensor]] = None,
        return_dict: Optional[bool] = None,
        states: Optional[object] = None,
    ):
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # speech_features = self.get_ssl_feature_w2v(speech_batch, src_lengths, after_lens).transpose(0, 1)
        speech_features = None
        if not self.speech_features_extracted:
            speech_features = self.get_ssl_feature_w2v(speech_batch, src_lengths, after_lens, states=states).transpose(0, 1)

            if states.past_key_values is not None:
                speech_past_key_values = [
                    [p[i][:, :, :states.speech_prefix_length, :] for i in range(2)]
                    for p in states.past_key_values
                ]
                text_past_key_values = [
                    [p[i][:, :, states.speech_prefix_length:, :] for i in range(2)]
                    for p in states.past_key_values
                ]
                 
                speech_of_llama_output = super(SpeechLlamaModel, self).forward(
                    input_ids=None, 
                    attention_mask=None, 
                    past_key_values=speech_past_key_values,
                    inputs_embeds=speech_features, 
                    use_cache=use_cache,
                    output_attentions=False, 
                    output_hidden_states=False,
                    return_dict=return_dict
                )

                past_key_values = [
                    [
                        torch.cat([speech_of_llama_output.past_key_values[i][j], text_past_key_values[i][j]], dim=2)
                        for j in range(2)
                    ]
                    for i in range(len(self.layers))
                ]
                states.speech_prefix_length = speech_of_llama_output.past_key_values[0][0].shape[2]

            else:
                cur_input_embeds = inputs_embeds[0]
                cur_input_ids = input_ids[0]

                speech_start_pos = torch.where(cur_input_ids == self.config.sp_start_token_id)[0]
                speech_end_pos = torch.where(cur_input_ids == self.config.sp_end_token_id)[0]

                states.speech_prefix_length = speech_start_pos + 1 + speech_features[0].shape[0]
                
                cur_new_input_embeds = torch.cat((cur_input_embeds[:speech_start_pos+1], speech_features[0], cur_input_embeds[speech_end_pos:]), dim=0)
                cur_new_input_embeds = cur_new_input_embeds.unsqueeze(0)

                speech_of_llama_output = super(SpeechLlamaModel, self).forward(
                    input_ids=None, 
                    attention_mask=None, 
                    past_key_values=None,
                    inputs_embeds=cur_new_input_embeds[:, :-1, :], 
                    use_cache=use_cache,
                    output_attentions=False, 
                    output_hidden_states=False,
                    return_dict=return_dict
                )

                past_key_values = speech_of_llama_output.past_key_values
            
            self.speech_features_extracted = True
        else:
            past_key_values = states.past_key_values
        
        sllama_output = super(SpeechLlamaModel, self).forward(
            input_ids=None, 
            attention_mask=None, 
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds[:, -1:, :], 
            use_cache=use_cache,
            output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        states.past_key_values = sllama_output.past_key_values


        return speech_of_llama_output

        return sllama_output
              
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
    ) -> Union[Tuple, BaseModelOutputWithPast]:
    
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # speech_features = self.get_ssl_feature_w2v(speech_batch, src_lengths, after_lens).transpose(0, 1)
        speech_features = None
        if not self.speech_features_extracted:
            speech_features = self.get_ssl_feature_w2v(speech_batch, src_lengths, after_lens).transpose(0, 1)
            self.speech_features_extracted = True
            
        new_input_embeds = []
        cur_speech_idx = 0
        # inputs_embeds: B*T*d
        # speech_features: B*T1*d
        if speech_features is not None:
            for i in range(inputs_embeds.size(0)):
                cur_speech_features = speech_features[i][:after_lens[i]]
                cur_input_embeds = inputs_embeds[i]
                cur_input_ids = input_ids[i]                
                if (cur_input_ids == self.config.sp_start_token_id).sum() == 0:
                    new_input_embeds.append(cur_input_embeds)
                    continue
                speech_start_pos = torch.where(cur_input_ids == self.config.sp_start_token_id)[0]
                speech_end_pos = torch.where(cur_input_ids == self.config.sp_end_token_id)[0]
                if orig_embeds_params is not None:
                    cur_new_input_embeds = torch.cat((cur_input_embeds[:speech_start_pos].detach(), cur_input_embeds[speech_start_pos], cur_speech_features, cur_input_embeds[speech_end_pos], cur_input_embeds[speech_end_pos + 1:].detach()), dim=0)
                else:
                    cur_new_input_embeds = torch.cat((cur_input_embeds[:speech_start_pos+1], cur_speech_features, cur_input_embeds[speech_end_pos:]), dim=0)
                new_input_embeds.append(cur_new_input_embeds)  
            inputs_embeds = torch.stack(new_input_embeds, dim=0)  

        return super(SpeechLlamaModel, self).forward(
            input_ids=None, 
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, 
            use_cache=use_cache,
            output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        ) 
    
## try not add prompt
    
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

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        speech_batch: Optional[torch.FloatTensor] = None,
        src_lengths: Optional[List[torch.FloatTensor]] = None,
        after_lens: Optional[List[torch.FloatTensor]] = None,
        return_dict: Optional[bool] = None,
        states: Optional[object] = None,
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
    
    def initialize_speech_tokenizer(self, tokenizer, device,
                                    only_tune_adapter=False, stage1=True): 
        if stage1:                            
            num_new_tokens = tokenizer.add_tokens([DEFAULT_SPEECH_PATCH_TOKEN, DEFAULT_SPEECH_START_TOKEN, DEFAULT_SPEECH_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))        
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

            sp_patch_token_id, sp_start_token_id, sp_end_token_id = tokenizer.convert_tokens_to_ids([DEFAULT_SPEECH_PATCH_TOKEN, DEFAULT_SPEECH_START_TOKEN, DEFAULT_SPEECH_END_TOKEN])                
            self.config.sp_patch_token_id = sp_patch_token_id
            self.config.sp_start_token_id = sp_start_token_id
            self.config.sp_end_token_id = sp_end_token_id 

        if only_tune_adapter: 
            self.get_model().orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)]
            for p in self.get_input_embeddings().parameters():
                p.requires_grad = True                 
            for p in self.get_output_embeddings().parameters():
                p.requires_grad = False
                   
AutoConfig.register("SpeechLlama", SpeechLlamaConfig)
AutoModelForCausalLM.register(SpeechLlamaConfig, SpeechLlamaForCausalLM)
