import re
from typing import Optional
from simuleval.utils import entrypoint
from simuleval.agents.actions import WriteAction, ReadAction

import numpy as np
import torch
import transformers

from peft import LoraConfig, get_peft_model

from tqdm import tqdm
from model.flashinfer.sllama import SpeechLlamaFastForCausalLM
from model.flashinfer.sqwen import SpeechQwenFastForCausalLM
from model.w2v2 import SpeechEncoderW2V2RoPE

import logging
logger = logging.getLogger(__name__)

from model.flashinfer.beam_search import (
    beam_search,
    Request
)
from model.flashinfer.engine import (
    init_paged_kv_cache,
    duplicate_paged_kv_cache,
    pop_paged_kv_cache,
    SpeechCache,
    LLMCache
)
from agents.infinisst import (
    synchronized_timer, 
    S2TAgentStates, 
    InfiniSST
)

@entrypoint
class InfiniSSTFaster(InfiniSST):

    def __init__(self, args):
        self.dtype = torch.bfloat16

        super().__init__(args)

        self.length_penalty = args.length_penalty

        self.blocksize = args.block_size
        speech_encoder = self.model.model.speech_encoder.speech_encoder
        llm = self.model.model

        self.speech_pagetable, self.llm_prefill_pagetable, self.llm_decode_pagetable = \
            init_paged_kv_cache(
                args.pseudo_batch_size,
                args.max_cache_size,
                speech_encoder.cfg.encoder_layers,
                speech_encoder.cfg.encoder_attention_heads,
                speech_encoder.cfg.encoder_embed_dim // speech_encoder.cfg.encoder_attention_heads,
                args.max_llm_cache_size,
                llm.config.num_hidden_layers,
                llm.config.num_attention_heads,
                llm.config.num_key_value_heads,
                llm.config.hidden_size // llm.config.num_attention_heads,
                dtype=self.dtype,
                device_prefill='cuda:0',
                device_decode='cuda:0'
            )
    
    def load_w2v2_qwen25(self, args):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_name,
            padding_side="right",
            use_fast=False,
        )
        self.tokenizer.pad_token = "<|finetune_right_pad_id|>"

        self.model = SpeechQwenFastForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=self.dtype,
            attn_implementation="eager",
            device_map='cuda',
        ).eval()

        speech_encoder_args = [
            args.w2v2_path,
            args.ctc_finetuned,
            args.length_shrink_cfg,
            
            args.block_size,
            args.max_cache_size,
            self.model.model.embed_tokens.embedding_dim,
            None,
            bool(args.rope),
            True
        ]
        if args.w2v2_type == 'w2v2':
            speech_encoder = SpeechEncoderW2V2RoPE(*speech_encoder_args)
        else:
            raise ValueError(f"Unsupported type: {args.w2v2_type}")
        speech_encoder.eval()
        speech_encoder.to(dtype=self.model.dtype, device=self.model.device)
        self.length_shrink_func = speech_encoder._get_feat_extract_output_lengths
        
        self.model.model.speech_encoder = speech_encoder
        self.model.preprocess(tokenizer=self.tokenizer, resize=False)

        logger.info("Loading SLLM weights from {}".format(args.state_dict_path))
        state_dict = torch.load(args.state_dict_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state_dict, strict=False)

        if args.lora_path:
            logger.info(f"Loading LORA weights from {args.lora_path}")
            lora_config = LoraConfig(
                task_type="CAUSAL_LM",
                inference_mode=True,
                r=args.lora_rank,
                target_modules='all-linear',
                lora_alpha=16,
                lora_dropout=0.1,
            )
            self.model = get_peft_model(self.model, lora_config, adapter_name='lora_adapter')
            
            lora_state_dict = torch.load(args.lora_path, map_location='cpu', weights_only=True)
            self.model.load_state_dict(lora_state_dict, strict=False)
            self.model = self.model.merge_and_unload()

        self.model.model.inference = True

    @staticmethod
    def add_args(parser):
        InfiniSST.add_args(parser)
        parser.add_argument('--length-penalty', type=float, default=1.0)

    @torch.inference_mode()
    def policy(self, states: Optional[S2TAgentStates] = None):
        if states is None:
            states = self.states

        if states.source_sample_rate == 0:
            # empty source, source_sample_rate not set yet
            length_in_seconds = 0
        else:
            length_in_seconds = float(len(states.source)) / states.source_sample_rate

        if not states.source_finished and length_in_seconds < self.min_start_sec:
            return ReadAction()
        
        if states.source_finished and length_in_seconds < 0.32:
            return WriteAction(content="", finished=True)
        
        with synchronized_timer('generate'):
            speech_batch = self._prepare_speech(states)
            input_ids = self._prepare_inputs(states)

            requests = [
                Request(
                    input_ids.view(-1),
                    speech_batch.view(-1),
                    self.latency_multiplier * self.blocksize,
                    self.max_new_tokens,
                    
                    self.args.max_cache_size,
                    states.speech_cache[i] if states.speech_cache is not None else None,

                    self.args.max_llm_cache_size,
                    self.system_prompt_size,
                    states.past_key_values[i] if states.past_key_values is not None else None
                )
                for i in range(self.pseudo_batch_size)
            ]

            # speech_batch = speech_batch.repeat(self.pseudo_batch_size, 1)
            # input_ids = input_ids.repeat(self.pseudo_batch_size, 1)
            # if states.speech_cache is not None:
            #     for i, (k, v) in enumerate(states.past_key_values):
            #         states.past_key_values[i] = (
            #             k.repeat(self.pseudo_batch_size, 1, 1, 1),
            #             v.repeat(self.pseudo_batch_size, 1, 1, 1)
            #         )
            
            if states.source_finished:
                states.segment_idx = -1

            while not all(request.decode_finished for request in requests):
                requests, self.speech_pagetable, self.llm_prefill_pagetable, self.llm_decode_pagetable = beam_search(
                    requests,
                    self.model,
                    self.tokenizer,
                    self.beam,
                    self.length_penalty,
                    self.speech_pagetable,
                    self.llm_prefill_pagetable,
                    self.llm_decode_pagetable
                )

            output_ids = requests[0].results['sequence'][:-1]
            states.speech_cache = [request.speech_cache for request in requests]
            states.past_key_values = [request.llm_cache for request in requests]
        
        if states.source_finished:
            for cache in states.speech_cache:
                pop_paged_kv_cache(
                    self.speech_pagetable, 
                    cache.paged_kv_indices, 
                    cache.paged_kv_last_page_len, 
                    0
                )
            for cache in states.past_key_values:
                pop_paged_kv_cache(
                    self.llm_prefill_pagetable, 
                    cache.paged_kv_indices, 
                    cache.paged_kv_last_page_len, 
                    0
                )
     
        states.target_ids.extend(output_ids)
        translation = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        translation = re.sub(r'[（）()"“”�]', '', translation)

        # print(f"{length_in_seconds / 60:.2f}", ':', self.tokenizer.decode(states.target_ids))
        # print(f"Speech length in minutes: {length_in_seconds / 60:.2f}")
        print(self.tokenizer.decode(states.target_ids))

        # print(states.segment_idx, ":", translation)
        states.segment_idx += 1

        if translation != '' or states.source_finished:
            return WriteAction(
                content=translation,
                finished=states.source_finished,
            )
        else:
            return ReadAction()