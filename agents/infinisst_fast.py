import os
import contextlib
from time import perf_counter

from typing import Optional
from simuleval.agents.states import AgentStates
from simuleval.utils import entrypoint
from simuleval.data.segments import SpeechSegment
from simuleval.agents import SpeechToTextAgent
from simuleval.agents.actions import WriteAction, ReadAction
from simuleval.agents.states import AgentStates
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoProcessor

from peft import LoraConfig, get_peft_model

from tqdm import tqdm
from model.llama31 import SpeechLlamaForCausalLM
from model.qwen25 import SpeechQwenForCausalLM
from model.patches.patch_w2v2 import patch_w2v2
from model.patches.patch_llama31 import patch_llama31
from model.patches.patch_qwen25 import patch_qwen25
from model.patches.patch_hf import patch_hf

from agents.options import (
    add_speech_encoder_args,
    add_simuleval_args,
    add_gen_args
)
from model.w2v2 import SpeechEncoderW2V2RoPE
from model.seamlessm4t_v2_encoder import (
    SeamlessM4Tv2Config,
    SeamlessM4Tv2SpeechEncoder
)
from train.dataset import (
    DEFAULT_SPEECH_PATCH_TOKEN,
    DEFAULT_LATENCY_TOKEN,
    normalize
)

import logging
logger = logging.getLogger(__name__)

from model.flashinfer.beam_search import beam_search_pseudo
from agents.infinisst import synchronized_timer, S2TAgentStates, InfiniSST

@entrypoint
class InfiniSSTFast(InfiniSST):

    def __init__(self, args):
        super().__init__(args)
    
    @staticmethod
    def add_args(parser):
        InfiniSST.add_args(parser)
    
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

            speech_batch = speech_batch.repeat(self.pseudo_batch_size, 1)
            input_ids = input_ids.repeat(self.pseudo_batch_size, 1)
            if states.speech_cache is not None:
                for i, (k, v) in enumerate(states.past_key_values):
                    states.past_key_values[i] = (
                        k.repeat(self.pseudo_batch_size, 1, 1, 1),
                        v.repeat(self.pseudo_batch_size, 1, 1, 1)
                    )
            
            if states.source_finished:
                states.segment_idx = -1

            results = beam_search_pseudo(
                self.model,
                self.tokenizer,
                input_ids,
                speech_batch,
                self.latency_multiplier, 
                self.beam,
                self.max_new_tokens,
                states,
            )

            states.past_key_values = results[0]['past_key_values']
            cur_llm_cache_size = states.past_key_values[0][0].size(2)
            self.cache_checkpoints.append(cur_llm_cache_size)

            if cur_llm_cache_size > self.max_llm_cache_size:
                new_llm_cache_size = 0
                for i, ckpt in enumerate(self.cache_checkpoints):
                    new_llm_cache_size = cur_llm_cache_size - ckpt
                    if new_llm_cache_size <= self.max_llm_cache_size:
                        self.cache_checkpoints = self.cache_checkpoints[i + 1:]
                        n_cache_trimmed = ckpt
                        if self.always_cache_system_prompt:
                            n_cache_trimmed -= self.system_prompt_size
                        self.cache_checkpoints = [
                            ckpt - n_cache_trimmed for ckpt in self.cache_checkpoints
                        ]
                        break

                for i, (k, v) in enumerate(states.past_key_values):
                    k_cache = k[:, :, -new_llm_cache_size:]
                    v_cache = v[:, :, -new_llm_cache_size:]
                    if self.always_cache_system_prompt:
                        k_cache = torch.cat([k[:, :, :self.system_prompt_size], k_cache], dim=2)
                        v_cache = torch.cat([v[:, :, :self.system_prompt_size], v_cache], dim=2)
                    states.past_key_values[i] = (k_cache, v_cache)

        output_ids = results[0]['sequences'][:-1]        
        states.target_ids.extend(output_ids)
        translation = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        translation = translation.replace('�', '')

        # print(f"{length_in_seconds / 60:.2f}", ':', self.tokenizer.decode(states.target_ids))
        # print(f"Speech length in minutes: {length_in_seconds / 60:.2f}")
        print(states.past_key_values[0][0].size(2), self.tokenizer.decode(states.target_ids))

        # print(states.segment_idx, ":", translation)
        states.segment_idx += 1

        if translation != '' or states.source_finished:
            return WriteAction(
                content=translation,
                finished=states.source_finished,
            )
        else:
            return ReadAction()