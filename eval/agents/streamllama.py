import argparse, os, sys, time, json
import contextlib
from time import perf_counter
from collections import Counter

from typing import Optional
from simuleval.agents.states import AgentStates
from simuleval.utils import entrypoint
from simuleval.data.segments import SpeechSegment
from simuleval.agents import SpeechToTextAgent
from simuleval.agents.actions import WriteAction, ReadAction
from simuleval.agents.states import AgentStates
from dataclasses import dataclass

import numpy
import torch
import torch.nn.functional as F
import transformers

import conversation as conversation_lib
from conversation import SeparatorStyle
from eval.utils import disable_torch_init
from model.model_new import SpeechLlamaForCausalLM
from model.utils import SpaceStoppingCriteria, KeywordsStoppingCriteria
# from train.uni_wav2vec_monkey_patch import replace_uni_train
from fairseq.data.audio.speech_to_text_dataset import _collate_frames

from train.options import (
    add_speech_encoder_args,
    add_simuleval_args,
    add_gen_args
)
from model.speech_encoder import (
    SpeechEncoderHuBERTRope,
    SpeechEncoderW2V2RoPE,
    SpeechEncoderW2VBERT2
)
from train.dataset import (
    DEFAULT_SPEECH_PATCH_TOKEN,
    DEFAULT_SPEECH_START_TOKEN,
    DEFAULT_SPEECH_END_TOKEN,
    DEFAULT_LATENCY_TOKEN
)

def synchronized_timer(description: str):
    @contextlib.contextmanager
    def timer_with_sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = perf_counter()
        yield
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_time = perf_counter() - start
        print(f"{description}: {elapsed_time:.4f} seconds")
    return timer_with_sync()

@dataclass
class S2TAgentStates(AgentStates):
    src_len: int
    speech_cache: None
    past_key_values: None
    target_ids: list
    segment_idx: int

    def reset(self):
        super().reset()
        self.src_len = 0
        self.speech_cache = None
        self.past_key_values = None
        self.target_ids = []
        self.segment_idx = 0

@entrypoint
class StreamLlama(SpeechToTextAgent):

    def __init__(self, args):
        super().__init__(args)
        # transformers.set_seed(998244353)

        # simuleval
        self.min_start_sec = args.min_start_sec
        self.source_segment_size = args.source_segment_size
        self.latency_multiplier = args.latency_multiplier
        self.source_lang = args.source_lang
        self.target_lang = args.target_lang
        
        # gen
        self.beam = args.beam
        assert self.beam == 1 # only support beam=1 for now due to the implementation of HF beam search
        self.no_repeat_ngram_size = args.no_repeat_ngram_size
        self.repetition_penalty = args.repetition_penalty
        self.max_len_a = args.max_len_a
        self.max_len_b = args.max_len_b
        self.do_sample = args.do_sample
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.temperature = args.temperature

        # cache
        self.max_llm_cache_size = args.max_llm_cache_size
        self.always_cache_system_prompt = args.always_cache_system_prompt
        
        # model
        self.load_model(args)

    def build_states(self):
        return S2TAgentStates(0, None, None, [], 0)

    def load_model(self, args):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_name,
            padding_side="right",
            use_fast=False,
        )
        self.tokenizer.pad_token = "<|finetune_right_pad_id|>"

        self.model = SpeechLlamaForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
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
            bool(args.xpos)
        ]
        if args.w2v2_type == 'hubert':
            speech_encoder = SpeechEncoderHuBERTRope(*speech_encoder_args)
        elif args.w2v2_type == 'w2v-bert':
            speech_encoder = SpeechEncoderW2VBERT2(
                args.w2v2_path,
                args.length_shrink_cfg,
                args.block_size,
                args.max_cache_size,
                self.model.model.embed_tokens.embedding_dim,
            )
        else:
            speech_encoder = SpeechEncoderW2V2RoPE(*speech_encoder_args)
        speech_encoder.eval()
        speech_encoder.to(dtype=self.model.dtype, device=self.model.device)
        self.length_shrink_func = speech_encoder._get_feat_extract_output_lengths
        
        self.model.model.speech_encoder = speech_encoder
        self.model.preprocess(tokenizer=self.tokenizer) 

        state_dict = torch.load(args.state_dict_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.model.inference = True


    @staticmethod
    def add_args(parser):
        add_simuleval_args(parser)
        add_speech_encoder_args(parser)
        add_gen_args(parser)
        parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
        parser.add_argument("--state-dict-path", type=str, default=None)
        parser.add_argument("--latency-multiplier", type=int, default=1)
        parser.add_argument("--max-llm-cache-size", type=int, default=10000)
        parser.add_argument("--always-cache-system-prompt", action='store_true') # LLM-Inf

    def _prepare_speech(self, states):
        sp_seg_frame = int(self.args.block_size // 4 * 0.08 * 16000)
        
        # Only tensorize the new part
        source = torch.tensor(states.source[states.src_len:])
        
        # Pad if needed
        if source.size(0) % sp_seg_frame != 0:
            n_pad = sp_seg_frame - source.size(0) % sp_seg_frame
            source = torch.cat([source, torch.zeros(n_pad).to(source)], dim=0)
            
        # Add offset only for first chunk
        if states.src_len == 0:
            offset = torch.zeros(79 + 320).to(source)
            source = torch.cat([offset, source], dim=0)
            
        states.src_len = len(states.source)

        speech_batch = source.unsqueeze(0).to(device=self.model.device, dtype=self.model.dtype)
        return speech_batch

    def _prepare_inputs(self, states):
        messages = []
        if states.speech_cache is None:
            latency_token = DEFAULT_LATENCY_TOKEN.format(self.latency_multiplier)
            messages.append(
                {
                    "role": "system",
                    "content": f"Translate the following speech from {self.source_lang} to {self.target_lang} with latency {latency_token}."
                }
            )
            self.system_prompt_size = self.tokenizer.apply_chat_template(
                [messages],
                return_tensors='pt',
                padding=True, 
                truncation=False, 
                add_special_tokens=False
            ).size(1)
        messages.append(
            {
                "role": "user",
                "content": self.args.block_size // 4 * self.latency_multiplier * DEFAULT_SPEECH_PATCH_TOKEN
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": "",
            }
        )
        input_ids = self.tokenizer.apply_chat_template(
            [messages],
            return_tensors='pt',
            padding=True, 
            truncation=False, 
            add_special_tokens=False
        )[:, :-1]
        if states.speech_cache is not None:
            input_ids = input_ids[:, 25:] # to remove system prompt
        input_ids = input_ids.cuda()
        return input_ids

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
            
            max_number_of_tokens = int(length_in_seconds * self.max_len_a + self.max_len_b)

            if states.source_finished:
                states.segment_idx = -1

            self.model.model.speech_features_extracted = False
            outputs = self.model.generate(
                attention_mask=None,
                input_ids=input_ids,
                speech_batch=speech_batch,
                do_sample=self.do_sample,
                top_p=self.top_p,
                top_k=self.top_k,
                temperature=self.temperature,
                num_beams=self.beam,
                max_new_tokens=max(1, max_number_of_tokens - len(states.target_ids)),
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                return_legacy_cache=False,
                use_cache=True,
                past_key_values=states.past_key_values,
                states=states,
            )

            states.past_key_values = outputs.past_key_values

            # cut cache
            max_total_cache_size = self.max_llm_cache_size
            if self.always_cache_system_prompt:
                max_total_cache_size += self.system_prompt_size
            if states.past_key_values[0][0].size(2) > max_total_cache_size:
                for i, (k, v) in enumerate(states.past_key_values):
                    k_cache = k[:, :, -self.max_llm_cache_size:]
                    v_cache = v[:, :, -self.max_llm_cache_size:]
                    if self.always_cache_system_prompt:
                        k_cache = torch.cat([k[:, :, :self.system_prompt_size], k_cache], dim=2)
                        v_cache = torch.cat([v[:, :, :self.system_prompt_size], v_cache], dim=2)
                    states.past_key_values.key_cache[i] = k_cache
                    states.past_key_values.value_cache[i] = v_cache

        output_ids = outputs.sequences[0, input_ids.size(1):-1].tolist()
        
        states.target_ids.extend(output_ids)
        translation = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        # print(f"{length_in_seconds / 60:.2f}", ':', self.tokenizer.decode(states.target_ids))
        # print(f"Speech length in minutes: {length_in_seconds / 60:.2f}")
        print(self.tokenizer.decode(states.target_ids, skip_special_tokens=True).strip())

        states.segment_idx += 1

        if translation != '' or states.source_finished:
            return WriteAction(
                content=translation,
                finished=states.source_finished,
            )
        else:
            return ReadAction()