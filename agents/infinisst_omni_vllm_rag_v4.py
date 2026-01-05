import os

# ⚠️ CRITICAL: Set VLLM_USE_V1=0 BEFORE importing vllm
# This forces vLLM to use the stable v0 engine instead of the experimental v1 engine
# Must be set before any vllm imports
os.environ['VLLM_USE_V1'] = '0'

import re
import json
import pickle
import contextlib
from time import perf_counter, time

from typing import Optional, List, Dict
from simuleval.agents.states import AgentStates
from simuleval.utils import entrypoint
from simuleval.data.segments import SpeechSegment
from simuleval.agents import SpeechToTextAgent
from simuleval.agents.actions import WriteAction, ReadAction
from simuleval.agents.states import AgentStates
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
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

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None

# Import streaming Qwen3 RAG retriever V4
from agents.streaming_qwen3_rag_retriever_v4 import StreamingQwen3RAGRetrieverV4

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


class ProcessorAudioAlias:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, *args, **kwargs):
        if "audios" in kwargs and "audio" not in kwargs:
            kwargs["audio"] = kwargs.pop("audios")
        return self.processor(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self.processor, item)


class TokenizerKwCleaner:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, *args, **kwargs):
        kwargs.pop("audio", None)
        kwargs.pop("audios", None)
        return self.tokenizer(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self.tokenizer, item)


@dataclass
class S2TAgentStates(AgentStates):
    src_len: int
    target_ids: list
    segment_idx: int
    messages: list
    references: list
    # Track audio samples processed by RAG
    rag_processed_samples: int
    # Last vLLM call position
    last_vllm_src_len: int
    MAX_SRC_LEN = 16000 * 3600  # 1 hour

    def reset(self):
        super().reset()
        self.src_len = 0
        self.target_ids = []
        self.segment_idx = 0
        self.messages = []
        self.references = []
        self.rag_processed_samples = 0
        self.last_vllm_src_len = 0


@entrypoint
class InfiniSSTOmniVLLMRAGV4(SpeechToTextAgent):

    def __init__(self, args):
        super().__init__(args)
        transformers.set_seed(998244353)

        # simuleval
        self.min_start_sec = args.min_start_sec
        self.source_lang = args.source_lang
        self.target_lang = args.target_lang
        # vLLM call interval
        self.vllm_segment_sec = getattr(args, "vllm_segment_sec", 0.96)
        
        # Log sample
        self.log_sample = int(getattr(args, "log_sample", 0))
        self._log_sample_count = 0 
        
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
        
        # RAG retriever (Qwen3 version V4)
        self.rag_retriever: Optional[StreamingQwen3RAGRetrieverV4] = None
        self.rag_top_k = getattr(args, "rag_top_k", 5)
        self.rag_target_lang = getattr(args, "rag_target_lang", "zh")
        self.rag_conf_threshold = getattr(args, "rag_confidence_threshold", 0.5)
        self.rag_min_terms = int(getattr(args, "rag_min_terms", 0))
        
        # Sliding window parameters
        self.rag_chunk_size = getattr(args, "rag_chunk_size", 1.92)  # 1.92s as requested
        self.rag_hop_size = getattr(args, "rag_hop_size", 0.96)      # 0.96s as requested
        
        self.debug_audio_dir = getattr(args, "debug_audio_dir", "") or ""
        self._vllm_call_count = 0

        if getattr(args, "rag_enabled", False):
            logger.info("Initializing StreamingQwen3RAGRetrieverV4...")
            self.rag_retriever = StreamingQwen3RAGRetrieverV4(
                index_path=getattr(args, "rag_index_path", None),
                model_path=getattr(args, "rag_model_path", None),
                base_model_name=getattr(args, "rag_base_model", "Atotti/Qwen3-Omni-AudioTransformer"),
                device=getattr(args, "rag_device", "cuda:1"),
                lora_r=getattr(args, "rag_lora_r", 32),
                lora_alpha=getattr(args, "rag_lora_alpha", 64),
                text_lora_r=getattr(args, "rag_text_lora_r", 16),
                top_k=self.rag_top_k,
                voting_k=getattr(args, "rag_voting_k", 20),
                target_lang=self.rag_target_lang,
                score_threshold=self.rag_conf_threshold,
                chunk_size=self.rag_chunk_size,
                hop_size=self.rag_hop_size,
                aggregation_strategy=getattr(args, "rag_strategy", "voting"),
                debug_audio_dir=self.debug_audio_dir if self.debug_audio_dir else None,
            )
            if not self.rag_retriever or not self.rag_retriever.enabled:
                logger.warning("RAG retriever not operational")
                self.rag_retriever = None
        
        # model
        self.use_vllm = args.use_vllm
        self.gpu_memory_utilization = getattr(args, "gpu_memory_utilization", 0.8)
        self.debug_llm_io = bool(getattr(args, "debug_llm_io", False))
        self.debug_filter_term = (getattr(args, "debug_filter_term", "") or "").strip()
        self.debug_max_chars = int(getattr(args, "debug_max_chars", 6000))
        self.debug_llm_io_file = (getattr(args, "debug_llm_io_file", "") or "").strip() or None

        self.runtime_log_dir = (getattr(args, "runtime_log_dir", "/mnt/gemini/data2/jiaxuanluo/converted_logs") or "").strip()
        self.runtime_log_enabled = bool(getattr(args, "runtime_log_enabled", True))
        self.runtime_log_path = None

        if self.runtime_log_enabled and self.runtime_log_dir:
            try:
                os.makedirs(self.runtime_log_dir, exist_ok=True)
                self.runtime_log_path = os.path.join(
                    self.runtime_log_dir,
                    f"runtime_omni_vllm_rag_v4_{int(time())}_pid{os.getpid()}.jsonl",
                )
                logger.info("Runtime log enabled, writing JSONL to %s", self.runtime_log_path)
            except Exception as e:
                logger.warning("Failed to initialize runtime log dir: %s", e)
                self.runtime_log_path = None
        
        self.load_model(args)
    
    @staticmethod
    def add_args(parser):
        add_simuleval_args(parser)
        add_gen_args(parser)
        parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
        parser.add_argument("--use-vllm", type=int, default=0)
        parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
        parser.add_argument("--max-cache-chunks", type=int, default=120)
        parser.add_argument("--keep-cache-chunks", type=int, default=60)
        parser.add_argument("--vllm-segment-sec", type=float, default=0.96)
        parser.add_argument("--log-sample", type=int, default=0)
        parser.add_argument("--rag-enabled", action="store_true")
        parser.add_argument("--rag-index-path", type=str, default=None)
        parser.add_argument("--rag-model-path", type=str, default=None)
        parser.add_argument("--rag-base-model", type=str, default="Atotti/Qwen3-Omni-AudioTransformer")
        parser.add_argument("--rag-device", type=str, default="cuda:1")
        parser.add_argument("--rag-top-k", type=int, default=5)
        parser.add_argument("--rag-voting-k", type=int, default=20)
        parser.add_argument("--rag-target-lang", type=str, default="zh")
        parser.add_argument("--rag-lora-r", type=int, default=32)
        parser.add_argument("--rag-lora-alpha", type=int, default=64)
        parser.add_argument("--rag-text-lora-r", type=int, default=16)
        parser.add_argument("--rag-confidence-threshold", type=float, default=0.5)
        parser.add_argument("--rag-min-terms", type=int, default=0)
        parser.add_argument("--rag-strategy", type=str, default="max_pool", choices=["voting", "max_pool"])
        parser.add_argument("--rag-chunk-size", type=float, default=1.92)
        parser.add_argument("--rag-hop-size", type=float, default=0.96)
        parser.add_argument("--debug-audio-dir", type=str, default="", help="Directory to save debug audio chunks. Empty to disable.")
        parser.add_argument("--debug-llm-io", action="store_true")
        parser.add_argument("--debug-filter-term", type=str, default="")
        parser.add_argument("--debug-max_chars", type=int, default=6000)
        parser.add_argument("--debug-llm-io-file", type=str, default="")
        parser.add_argument("--runtime-log-enabled", type=int, default=1)
        parser.add_argument("--runtime-log-dir", type=str, default="/mnt/gemini/data2/jiaxuanluo/converted_logs")

    def build_states(self):
        if hasattr(self, 'rag_retriever') and self.rag_retriever:
            self.rag_retriever.reset()
        self._vllm_call_count = 0
        return S2TAgentStates(
            src_len=0,
            target_ids=[],
            segment_idx=0,
            messages=[],
            references=[],
            rag_processed_samples=0,
            last_vllm_src_len=0,
        )

    def load_model(self, args):
        if args.use_vllm:
            gpu_memory_util = self.gpu_memory_utilization
            tp_size = 2
            self.model = LLM(
                model=args.model_name, 
                trust_remote_code=True, 
                gpu_memory_utilization=gpu_memory_util,
                tensor_parallel_size=tp_size,
                limit_mm_per_prompt={'audio': self.max_cache_chunks},
                max_num_seqs=1,
                max_model_len=32768,
                enable_prefix_caching=True,
                enforce_eager=False,
            )
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

    def _normalize_references(self, references: List[Dict[str, str]]) -> Dict[str, str]:
        norm_refs: Dict[str, str] = {}
        seen_keys: set = set()
        for r in references:
            key = (r.get("key") or "").strip().lower()
            if not key: continue
            term = (r.get("term") or "").strip()
            if not term: continue
            if key in seen_keys: continue
            translation = (r.get("translation") or "").strip()
            norm_refs[term] = translation
            seen_keys.add(key)
        return norm_refs

    @staticmethod
    def _format_term_map_kv(term_map: Dict[str, str]) -> str:
        lines: List[str] = []
        for k, v in (term_map or {}).items():
            kk = str(k).replace("\n", " ").strip()
            vv = str(v).replace("\n", " ").strip()
            if not kk or not vv: continue
            lines.append(f"{kk}={vv}")
        return "\n".join(lines)

    def _prepare_speech(self, states):        
        if len(states.source) > states.MAX_SRC_LEN:
            diff = len(states.source) - states.MAX_SRC_LEN
            states.src_len = max(0, states.src_len - diff)
            states.source = states.source[-states.MAX_SRC_LEN:]
            if hasattr(states, "rag_processed_samples"):
                states.rag_processed_samples = max(0, states.rag_processed_samples - diff)
            if hasattr(states, "last_vllm_src_len"):
                states.last_vllm_src_len = max(0, states.last_vllm_src_len - diff)

        increment = np.array(states.source[states.src_len:])   
        if len(increment) < 15360:
            increment = np.pad(increment, (0, 15360 - len(increment)), mode='constant', constant_values=0)

        states.src_len = len(states.source)
        return increment

    def _prepare_inputs(self, states, increment, references):
        rag_enabled = getattr(self, "rag_enabled", False)
        if len(states.messages) == 0:
            if rag_enabled:
                system_text = f"You are a professional simultaneous interpreter. Your task is to translate {self.source_lang} audio chunks into accurate and fluent {self.target_lang}. Use the ‘term_map’ as a reference for terminology if provided."
            else:
                system_text = f"You are a professional simultaneous interpreter. Your task is to translate {self.source_lang} audio chunks into accurate and fluent {self.target_lang}."
            states.messages.append({"role": "system", "content": [{"type": "text", "text": system_text}]})
        
        user_content = [{"type": "audio", "audio": increment}]
        norm_refs = self._normalize_references(references)
        if norm_refs:
            kv = self._format_term_map_kv(norm_refs)
            if kv:
                reference_text = f"\n\nterm_map:\n{kv}"
                user_content.append({"type": "text", "text": reference_text})
            else:
                user_content.append({"type": "text", "text": "\n\nterm_map:NONE"})
        elif rag_enabled:
            user_content.append({"type": "text", "text": "\n\nterm_map:NONE"})
        
        states.messages.append({"role": "user", "content": user_content})

        text = self.processor.apply_chat_template(states.messages, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(states.messages, use_audio_in_video=False)

        if self.use_vllm:
            inputs = {
                'prompt': text,
                'multi_modal_data': {'audio': audios},
                "mm_processor_kwargs": {"use_audio_in_video": False},
            }
        else:
            inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
            inputs['input_features'] = inputs['input_features'].to(self.model.dtype)
        return inputs

    def _truncate_text(self, text: str) -> str:
        if not text or self.debug_max_chars <= 0 or len(text) <= self.debug_max_chars:
            return text
        return text[: self.debug_max_chars] + "\n...[truncated]..."

    def _sampling_params_payload(self):
        sp = getattr(self, "sampling_params", None)
        try:
            return dict(sp.__dict__) if hasattr(sp, "__dict__") else str(sp)
        except: return None

    def _append_runtime_jsonl(self, record: Dict[str, object]) -> None:
        if not self.runtime_log_path: return
        try:
            with open(self.runtime_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except: pass

    @torch.inference_mode()
    def policy(self, states: Optional[S2TAgentStates] = None):
        if states is None: states = self.states
        length_in_seconds = float(len(states.source)) / states.source_sample_rate if states.source_sample_rate > 0 else 0

        if not states.source_finished and length_in_seconds < self.min_start_sec:
            return ReadAction()
        
        if states.source_finished and length_in_seconds < 0.32:
            return WriteAction(content="", finished=True)
        
        samples_since_last_vllm = len(states.source) - states.last_vllm_src_len
        samples_for_vllm_call = int(self.vllm_segment_sec * states.source_sample_rate) if states.source_sample_rate > 0 else 15360
        
        should_call_vllm = (states.source_finished or samples_since_last_vllm >= samples_for_vllm_call)
        
        if self.rag_retriever:
            if states.rag_processed_samples == 0 and self.rag_retriever.get_audio_duration() > 0:
                logger.warning("New sample detected but RAG buffer was not empty. Forcing reset.")
                self.rag_retriever.reset()
                
            new_samples_start = states.rag_processed_samples
            new_samples_end = len(states.source)
            new_audio = None
            if new_samples_end > new_samples_start:
                new_audio = np.array(states.source[new_samples_start:new_samples_end], dtype=np.float32)
                states.rag_processed_samples = new_samples_end
            
            if new_audio is not None or should_call_vllm or states.source_finished:
                self.rag_retriever.accumulate_audio(
                    new_audio, 
                    force_process=states.source_finished
                )
        
        if not should_call_vllm: return ReadAction()
        
        with synchronized_timer('generate'):
            increment = self._prepare_speech(states)
            
            if self.debug_audio_dir:
                import soundfile as sf
                vllm_wav_path = os.path.join(self.debug_audio_dir, f"vllm_inc_call{self._vllm_call_count:03d}.wav")
                sf.write(vllm_wav_path, increment, 16000)
                logger.info(f"Saved vLLM increment audio to {vllm_wav_path}")
                self._vllm_call_count += 1

            references = []
            if self.rag_retriever:
                references = self.rag_retriever.get_current_references(min_terms=self.rag_min_terms)
                states.references = references
                rag_duration = self.rag_retriever.get_audio_duration()
                if references:
                    self._append_runtime_jsonl({"type": "rag", "segment_idx": int(getattr(states, "segment_idx", -1)), "rag_audio_duration": round(rag_duration, 2), "references": references})
            
            states.last_vllm_src_len = len(states.source)
            inputs = self._prepare_inputs(states, increment, references)
            
            if references:
                ref_str = ", ".join([f"{r['term']}->{r['translation']}" for r in references])
                print(f"\n[VLLM] 📥 Final TermMap (Top-{len(references)}): {ref_str}")
            else:
                print(f"\n[VLLM] 📥 Final TermMap: [None]")
            
            self._append_runtime_jsonl({"type": "llm_input", "segment_idx": int(getattr(states, "segment_idx", -1)), "prompt": self._truncate_text(inputs.get("prompt", "")) if isinstance(inputs, dict) else "", "references": references})
            
            if self.use_vllm:
                outputs = self.model.generate([inputs], sampling_params=self.sampling_params, use_tqdm=False)
                translation = outputs[0].outputs[0].text
                print(f"[VLLM] 📤 Output Translation: {translation}")
                self._append_runtime_jsonl({"type": "llm_output", "segment_idx": int(getattr(states, "segment_idx", -1)), "text": self._truncate_text(translation)})
            else:
                translation = "Translation logic for non-vLLM Qwen3-Omni not implemented in this snippet"
            
            states.messages.append({"role": "assistant", "content": [{"type": "text", "text": translation}]})
            if len(states.messages) >= 2 * self.max_cache_chunks + 1:
                states.messages = [states.messages[0]] + states.messages[-2 * self.keep_cache_chunks:]

            if states.source_finished: states.segment_idx = -1

        print(''.join(states.target))
        states.segment_idx += 1
        return WriteAction(content=translation, finished=states.source_finished) if translation != '' or states.source_finished else ReadAction()

