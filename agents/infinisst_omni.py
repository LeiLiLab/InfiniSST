import os

# ⚠️ CRITICAL: Set VLLM_USE_V1=0 BEFORE importing vllm
# This forces vLLM to use the stable v0 engine instead of the experimental v1 engine
# Must be set before any vllm imports
os.environ['VLLM_USE_V1'] = '0'

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


@dataclass
class S2TAgentStates(AgentStates):
    src_len: int
    target_ids: list
    segment_idx: int
    messages: list
    MAX_SRC_LEN = 16000 * 30

    def reset(self):
        super().reset()
        self.src_len = 0
        self.target_ids = []
        self.segment_idx = 0
        self.messages = []


@entrypoint
class InfiniSSTOmniVLLMRAG(SpeechToTextAgent):

    def __init__(self, args):
        super().__init__(args)
        transformers.set_seed(998244353)

        # simuleval
        self.min_start_sec = args.min_start_sec
        self.source_lang = args.source_lang
        self.target_lang = args.target_lang
        
        # gen
        self.beam = args.beam
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

        # cache
        self.max_cache_chunks = args.max_cache_chunks
        self.keep_cache_chunks = args.keep_cache_chunks
        
        # model
        self.use_vllm = args.use_vllm
        self.load_model(args)
    
    @staticmethod
    def add_args(parser):
        add_simuleval_args(parser)
        add_gen_args(parser)
        parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
        parser.add_argument("--use-vllm", type=int, default=0)
        parser.add_argument("--max-cache-chunks", type=int, default=120)
        parser.add_argument("--keep-cache-chunks", type=int, default=60)

    def build_states(self):
        return S2TAgentStates(
            src_len=0,
            target_ids=[],
            segment_idx=0,
            messages=[],
        )

    def load_model(self, args):
        if args.use_vllm:
            """
            vllm serve /data/user_data/siqiouya/ckpts/test_swift/Qwen3-Omni-30B-A3B-Instruct-lora/v1-20251104-033331-hf \
                --gpu-memory-utilization 0.9 \
                --tensor-parallel-size 2 \
                --limit-mm-per-prompt '{"audio": 60}' \
                --max-model-len 2048 \
                --enable-prefix-caching  
            """
            # GPU Allocation Strategy:
            # - vLLM uses TP=2, which will automatically use GPU 0 and 1
            # - Reduced gpu_memory_utilization to 0.9 to leave more headroom for MoE models
            gpu_memory_util = 0.9  # Reduced from 0.95 for stability
            tp_size = 2
            
            logger.info(f"="*80)
            logger.info(f"vLLM Configuration:")
            logger.info(f"  Model: {args.model_name}")
            logger.info(f"  Tensor Parallel Size: {tp_size}")
            logger.info(f"  GPU Memory Utilization: {gpu_memory_util}")
            logger.info(f"  Max Model Len: 2048")
            logger.info(f"  Enable Prefix Caching: True")
            logger.info(f"  Limit MM Per Prompt (audio): {self.max_cache_chunks}")
            logger.info(f"="*80)
            
            logger.info(f"Initializing vLLM engine... This may take a few minutes.")
            
            self.model = LLM(
                model=args.model_name, 
                trust_remote_code=True, 
                gpu_memory_utilization=gpu_memory_util,
                tensor_parallel_size=tp_size,
                limit_mm_per_prompt={'audio': self.max_cache_chunks},
                max_num_seqs=1,
                max_model_len=4096,
                enable_prefix_caching=True,
                enforce_eager=False,  # Use CUDA graphs for better performance
            )
            
            logger.info(f"✅ vLLM engine initialized successfully!")
            self.sampling_params = SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_tokens=self.max_new_tokens,
            )
        else:
            self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                args.model_name,
                dtype="auto",
                device_map="auto",
                attn_implementation="flash_attention_2",
                enable_audio_output=False,
            )
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_name)

    def _prepare_speech(self, states):        
        # Only tensorize the new part
        if len(states.source) > states.MAX_SRC_LEN:
            states.src_len -= len(states.source) - states.MAX_SRC_LEN
            states.source = states.source[-states.MAX_SRC_LEN:]

        increment = np.array(states.source[states.src_len:])   

        if len(increment) < 15360:
            increment = np.pad(increment, (0, 15360 - len(increment)), mode='constant', constant_values=0)

        states.src_len = len(states.source)
        return increment

    def _prepare_inputs(self, states, increment):
        if len(states.messages) == 0:
            states.messages.append(
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": f"You are a professional simultaneous interpreter. You will be given chunks of English audio and you need to translate the audio into Chinese text."}
                    ]
                }
            )
        
        # Build user content with audio only
        user_content = [{"type": "audio", "audio": increment}]
        
        states.messages.append(
            {
                "role": "user",
                "content": user_content
            }
        )

        print("len(messages):", len(states.messages))

        text = self.processor.apply_chat_template(
            states.messages, 
            add_generation_prompt=True, 
            tokenize=False
        )
        audios, images, videos = process_mm_info(states.messages, use_audio_in_video=False)
        print("len(audios):", len(audios))

        if self.use_vllm:
            inputs = {
                'prompt': text,
                'multi_modal_data': {
                    'audio': audios,
                },
                "mm_processor_kwargs": {
                    "use_audio_in_video": False,
                },
            }

            input_ids = self.processor(
                text=text, 
                audio=audios, 
                images=images, 
                videos=videos, 
                return_tensors="pt", 
                padding=True, 
                use_audio_in_video=False
            )['input_ids']
            print("input_ids size:", input_ids.size())
        else:
            inputs = self.processor(
                text=text, 
                audio=audios, 
                images=images, 
                videos=videos, 
                return_tensors="pt", 
                padding=True, 
                use_audio_in_video=False
            )
            inputs['input_features'] = inputs['input_features'].to(self.model.dtype)
        return inputs

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
            increment = self._prepare_speech(states)
            inputs = self._prepare_inputs(states, increment)
            print(f"inputs:\n{inputs}")

            if self.use_vllm:
                outputs = self.model.generate(
                    [inputs], 
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                )
                translation = outputs[0].outputs[0].text
                print(f"translation:{translation}\n")
            else:
                text_ids, _ = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    return_audio=False,
                    thinker_return_dict_in_generate=True,
                    use_audio_in_video=False,
                )
                translation = self.processor.batch_decode(
                    text_ids.sequences[:, inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]

            states.messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": translation}]
                }
            )

            if len(states.messages) >= 2 * self.max_cache_chunks + 1:
                print("before trim:", len(states.messages))
                states.messages = [states.messages[0]] + states.messages[-2 * self.keep_cache_chunks:]
                print("after trim:", len(states.messages))

            if states.source_finished:
                states.segment_idx = -1

        print(''.join(states.target))

        # print(states.segment_idx, ":", translation)
        states.segment_idx += 1

        if translation != '' or states.source_finished:
            return WriteAction(
                content=translation,
                finished=states.source_finished,
            )
        else:
            return ReadAction()

