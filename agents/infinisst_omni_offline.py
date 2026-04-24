import os
import re
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
from transformers import (
    AutoProcessor, 
    Qwen3OmniMoeThinkerForConditionalGeneration, 
    Qwen3OmniMoeForConditionalGeneration, 
    Qwen3OmniMoeProcessor, 
    GenerationConfig, 
    Qwen3OmniMoeConfig
)
from qwen_omni_utils import process_mm_info

from vllm import LLM, SamplingParams

from tqdm import tqdm

from agents.options import (
    add_simuleval_args,
    add_gen_args,
)

import logging
logger = logging.getLogger(__name__)

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

@entrypoint
class InfiniSSTOmniOffline(SpeechToTextAgent):

    def __init__(self, args):
        super().__init__(args)
        transformers.set_seed(998244353)

        # simuleval
        self.source_lang = args.source_lang
        self.target_lang = args.target_lang
        
        # gen
        self.beam = args.beam
        # assert self.beam > 1
        self.max_new_tokens = args.max_new_tokens
        self.do_sample = args.do_sample
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.temperature = args.temperature

        self.generation_config = GenerationConfig(  
            num_beams=self.beam,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_new_tokens=self.max_new_tokens,
        )
        
        # model
        self.load_model(args)
    
    @staticmethod
    def add_args(parser):
        add_simuleval_args(parser)
        add_gen_args(parser)
        parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")

    def load_model(self, args):
        self.model = LLM(
            model=args.model_name, trust_remote_code=True, gpu_memory_utilization=0.95,
            tensor_parallel_size=torch.cuda.device_count(),
            limit_mm_per_prompt={'audio': 1},
            max_num_seqs=1,
            max_model_len=32768,
            enable_prefix_caching=True
        )
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_new_tokens,
        )
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_name)

    def _prepare_inputs_offline(self, speech):

        messages = []
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Listen to the provided {self.source_lang} speech and produce a translation in {self.target_lang} text."},
                    {"type": "audio", "audio": speech},
                ]
            }
        )

        print("len(messages):", len(messages))

        text = self.processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=False
        )
        print(text)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
        print("len(audios):", len(audios))

        inputs = {
            'prompt': text,
            'multi_modal_data': {
                'audio': audios,
            },
            "mm_processor_kwargs": {
                "use_audio_in_video": False,
            },
        }

        return inputs

    @torch.inference_mode()
    def policy(self, states: Optional[AgentStates] = None):
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
            speech = np.array(states.source)
            inputs = self._prepare_inputs_offline(speech)

            outputs = self.model.generate(
                [inputs], 
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            translation = outputs[0].outputs[0].text.replace('\n', '')

        print(translation)

        if translation != '' or states.source_finished:
            return WriteAction(
                content=translation,
                finished=states.source_finished,
            )
        else:
            return ReadAction()