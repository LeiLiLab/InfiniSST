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

# Import streaming RAG retriever
from agents.streaming_rag_retriever import StreamingTermRAGRetriever

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


# Note: TermRAGRetriever has been replaced by StreamingTermRAGRetriever
# imported from agents.streaming_rag_retriever


@dataclass
class S2TAgentStates(AgentStates):
    src_len: int
    target_ids: list
    segment_idx: int
    messages: list
    references: list
    # Track audio samples processed by RAG (independent of src_len for vLLM)
    rag_processed_samples: int
    # Last vLLM call position (for decoupling RAG and vLLM)
    last_vllm_src_len: int
    MAX_SRC_LEN = 16000 * 30

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
        
        # RAG retriever (streaming version with sliding window support)
        self.rag_retriever: Optional[StreamingTermRAGRetriever] = None
        self.rag_top_k = getattr(args, "rag_top_k", 5)
        self.rag_target_lang = getattr(args, "rag_target_lang", "zh")
        self.rag_conf_threshold = getattr(args, "rag_confidence_threshold", 0.5)
        # Minimum number of terms to keep per vLLM call (avoid forcing negatives)
        # Default: 0 (do NOT force 5 terms; let score_threshold/top-N decide)
        self.rag_min_terms = int(getattr(args, "rag_min_terms", 0))
        # Sliding window parameters (consistent with eval_local_sliding_window)
        self.rag_chunk_size = getattr(args, "rag_chunk_size", 2.0)  # seconds
        self.rag_hop_size = getattr(args, "rag_hop_size", 1.0)      # seconds
        self.rag_terms_per_second = getattr(args, "rag_terms_per_second", 2.5)
        self.rag_enable_top_n_filter = getattr(args, "rag_enable_top_n_filter", True)
        
        if getattr(args, "rag_enabled", False):
            logger.info("Initializing StreamingTermRAGRetriever with sliding window...")
            logger.info("  chunk_size=%.1fs, hop_size=%.1fs, terms_per_second=%.1f",
                       self.rag_chunk_size, self.rag_hop_size, self.rag_terms_per_second)
            self.rag_retriever = StreamingTermRAGRetriever(
                index_path=getattr(args, "rag_index_path", None),
                model_path=getattr(args, "rag_model_path", None),
                base_model_name=getattr(args, "rag_base_model", "Qwen/Qwen2-Audio-7B-Instruct"),
                device=getattr(args, "rag_device", "cuda:1"),
                lora_r=getattr(args, "rag_lora_r", 16),
                lora_alpha=getattr(args, "rag_lora_alpha", 32),
                lora_dropout=getattr(args, "rag_lora_dropout", 0.0),
                top_k=self.rag_top_k,
                target_lang=self.rag_target_lang,
                score_threshold=self.rag_conf_threshold,
                chunk_size=self.rag_chunk_size,
                hop_size=self.rag_hop_size,
                terms_per_second=self.rag_terms_per_second,
                enable_top_n_filter=self.rag_enable_top_n_filter,
            )
            if not self.rag_retriever or not self.rag_retriever.enabled:
                logger.warning("RAG retriever not operational; continuing without references")
                self.rag_retriever = None
        else:
            logger.info("RAG retrieval disabled for InfiniSSTOmniVLLMRAG agent")
        
        # model
        self.use_vllm = args.use_vllm
        self.vllm_segment_sec = args.vllm_segment_sec
        self.log_sample = args.log_sample
        self._log_sample_count = 0
        # Debug (LLM IO dump)
        self.debug_llm_io = bool(getattr(args, "debug_llm_io", False))
        self.debug_filter_term = (getattr(args, "debug_filter_term", "") or "").strip()
        self.debug_max_chars = int(getattr(args, "debug_max_chars", 6000))
        self.debug_llm_io_file = (getattr(args, "debug_llm_io_file", "") or "").strip() or None

        # Runtime log (persistent JSONL, enabled by default; does NOT rely on stdout/stderr redirection)
        self.runtime_log_dir = (getattr(args, "runtime_log_dir", "/mnt/gemini/data2/jiaxuanluo/converted_logs") or "").strip()
        self.runtime_log_enabled = bool(getattr(args, "runtime_log_enabled", True))
        self.runtime_log_path = None



        if self.runtime_log_enabled and self.runtime_log_dir:
            try:
                os.makedirs(self.runtime_log_dir, exist_ok=True)
                self.runtime_log_path = os.path.join(
                    self.runtime_log_dir,
                    f"runtime_omni_vllm_rag_{int(time())}_pid{os.getpid()}.jsonl",
                )
                logger.info("Runtime log enabled, writing JSONL to %s", self.runtime_log_path)
            except Exception as e:
                logger.warning("Failed to initialize runtime log dir %s: %s", self.runtime_log_dir, e)
                self.runtime_log_path = None
        self.load_model(args)
    
    @staticmethod
    def add_args(parser):
        add_simuleval_args(parser)
        add_gen_args(parser)
        parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
        parser.add_argument("--use-vllm", type=int, default=0)
        parser.add_argument("--max-cache-chunks", type=int, default=120)
        parser.add_argument("--keep-cache-chunks", type=int, default=60)
        parser.add_argument(
            "--vllm-segment-sec",
            type=float,
            default=0.96,
            help="vLLM call interval in seconds. Default: 0.96 (960ms). "
                 "This controls how much audio to accumulate before calling vLLM for translation.",
        )
        parser.add_argument(
            "--log-sample",
            type=int,
            default=0,
            help="Print detailed input/output for the first N vLLM calls. "
                 "Includes RAG results, vLLM prompt, and translation output. Default: 0 (disabled).",
        )
        parser.add_argument("--rag-enabled", action="store_true", help="Enable glossary RAG retrieval for prompt augmentation")
        parser.add_argument("--rag-index-path", type=str, default=None, help="Path to prebuilt RAG FAISS index (.pkl)")
        parser.add_argument("--rag-model-path", type=str, default=None, help="Path to trained RAG contrastive checkpoint")
        parser.add_argument("--rag-base-model", type=str, default="Qwen/Qwen2-Audio-7B-Instruct", help="Base model name for RAG encoder")
        parser.add_argument("--rag-device", type=str, default="cuda:1", help="Device identifier for RAG encoder")
        parser.add_argument("--rag-top-k", type=int, default=5, help="Number of glossary terms to retrieve per chunk")
        parser.add_argument("--rag-target-lang", type=str, default="zh", help="Target language key for term translations")
        parser.add_argument("--rag-lora-r", type=int, default=16, help="LoRA rank for RAG model loading")
        parser.add_argument("--rag-lora-alpha", type=int, default=32, help="LoRA alpha for RAG model loading")
        parser.add_argument("--rag-lora-dropout", type=float, default=0.0, help="LoRA dropout for RAG model loading")
        parser.add_argument(
            "--rag-confidence-threshold",
            type=float,
            default=0.5,
            help="Minimum confidence (0-1) required to keep retrieved glossary references",
        )
        parser.add_argument(
            "--rag-min-terms",
            type=int,
            default=0,
            help="Minimum number of terms to keep per vLLM call (default: 0). "
                 "Setting this to 5 can inject many negative terms and confuse the model.",
        )
        # Sliding window parameters for streaming RAG
        parser.add_argument(
            "--rag-chunk-size",
            type=float,
            default=2.0,
            help="Sliding window chunk size in seconds (default: 2.0)",
        )
        parser.add_argument(
            "--rag-hop-size",
            type=float,
            default=1.0,
            help="Sliding window hop size in seconds (default: 1.0)",
        )
        parser.add_argument(
            "--rag-terms-per-second",
            type=float,
            default=2.5,
            help="Number of terms to keep per second of audio for top-N filtering (default: 2.5)",
        )
        parser.add_argument(
            "--rag-enable-top-n-filter",
            type=int,
            default=1,
            help="Enable top-N filtering based on audio duration (1=enabled, 0=disabled)",
        )
        parser.add_argument(
            "--debug-llm-io",
            action="store_true",
            help="Dump LLM request/response (prompt, params, output) for debugging",
        )
        parser.add_argument(
            "--debug-filter-term",
            type=str,
            default="",
            help="Only dump debug LLM IO when this term appears in references/term_map (e.g., 'NLP')",
        )
        parser.add_argument(
            "--debug-max-chars",
            type=int,
            default=6000,
            help="Max characters to print for prompt/output when debug dump is enabled",
        )
        parser.add_argument(
            "--debug-llm-io-file",
            type=str,
            default="",
            help="Optional path to append debug LLM IO as JSONL (UTF-8). If empty, only prints to stdout/stderr.",
        )
        parser.add_argument(
            "--runtime-log-enabled",
            type=int,
            default=1,
            help="Enable persistent runtime JSONL logging (1) or disable (0). Default: 1.",
        )
        parser.add_argument(
            "--runtime-log-dir",
            type=str,
            default="/mnt/gemini/data2/jiaxuanluo/converted_logs",
            help="Directory for persistent runtime JSONL logs (UTF-8).",
        )

    def build_states(self):
        # Reset RAG retriever state for new utterance
        # Use hasattr because build_states may be called before __init__ completes
        if hasattr(self, 'rag_retriever') and self.rag_retriever:
            self.rag_retriever.reset()
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
            """
            vllm serve /data/user_data/siqiouya/ckpts/test_swift/Qwen3-Omni-30B-A3B-Instruct-lora/v1-20251104-033331-hf \
                --gpu-memory-utilization 0.9 \
                --tensor-parallel-size 2 \
                --limit-mm-per-prompt '{"audio": 60}' \
                --max-model-len 2048 \
                --enable-prefix-caching  
            """
            # GPU Allocation Strategy:
            # - RAG uses cuda:2 (specified in --rag-device)
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
            
            if self.rag_retriever and self.rag_retriever.enabled:
                logger.info(f"  RAG Status: Enabled on separate GPU")
            else:
                logger.info(f"  RAG Status: Disabled")
            logger.info(f"="*80)
            
            logger.info(f"Initializing vLLM engine... This may take a few minutes.")
            
            self.model = LLM(
                model=args.model_name, 
                trust_remote_code=True, 
                gpu_memory_utilization=gpu_memory_util,
                tensor_parallel_size=tp_size,
                limit_mm_per_prompt={'audio': self.max_cache_chunks},
                max_num_seqs=1,
                max_model_len=32768,
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

    def _normalize_references(self, references: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Normalize references into a simple {term: translation} mapping,
        matching the training format:

            term_map:
            jungle diary=丛林日记
            los angeles=洛杉矶
        """
        norm_refs: Dict[str, str] = {}
        seen_keys: set = set()
        for r in references:
            # Strict mode: require canonical lowercase key to be present.
            key = (r.get("key") or "").strip().lower()
            if not key:
                raise ValueError("Invalid RAG reference: missing required 'key' field")
            term = (r.get("term") or "").strip()
            if not term:
                continue
            if key in seen_keys:
                continue
            translation = (r.get("translation") or "").strip()
            norm_refs[term] = translation
            seen_keys.add(key)
        return norm_refs

    @staticmethod
    def _format_term_map_kv(term_map: Dict[str, str]) -> str:
        """
        Format term_map as key=value lines.
        - Keep insertion order from dict.
        - Strip newlines to avoid prompt injection / broken formatting.
        """
        lines: List[str] = []
        for k, v in (term_map or {}).items():
            kk = (str(k) if k is not None else "").replace("\n", " ").strip()
            vv = (str(v) if v is not None else "").replace("\n", " ").strip()
            if not kk or not vv:
                continue
            lines.append(f"{kk}={vv}")
        return "\n".join(lines)

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


    def _prepare_inputs(self, states, increment, references):
        if len(states.messages) == 0:
            system_text = (
                f"You are a professional simultaneous interpreter. "
                f"You will be given chunks of {self.source_lang} audio and you need to "
                f"translate the audio into {self.target_lang} text. "
                f"Use the 'term_map' as a reference for terminology if provided."
            )
            
            states.messages.append(
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": system_text}
                    ]
                }
            )
        
        # Build user content with audio and references
        user_content = [{"type": "audio", "audio": increment}]
        
        # ✅ Match training format exactly:
        #
        # <audio>
        #
        # term_map:
        # a=b
        # c=d
        norm_refs = self._normalize_references(references)
        if norm_refs:
            kv = self._format_term_map_kv(norm_refs)
            if kv:
                reference_text = f"\n\nterm_map:\n{kv}"
                user_content.append({"type": "text", "text": reference_text})
        else:
            user_content.append({"type": "text", "text": "\n\nterm_map:NONE"})
        
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

    def _truncate_text(self, text: str) -> str:
        if text is None:
            return ""
        if self.debug_max_chars <= 0:
            return text
        if len(text) <= self.debug_max_chars:
            return text
        return text[: self.debug_max_chars] + "\n...[truncated]..."

    def _sampling_params_payload(self):
        sp = getattr(self, "sampling_params", None)
        if sp is None:
            return None
        # Best-effort conversion to JSON-serializable payload.
        try:
            if hasattr(sp, "__dict__"):
                return dict(sp.__dict__)
        except Exception:
            pass
        return str(sp)

    def _append_debug_jsonl(self, record: Dict[str, object]) -> None:
        if not self.debug_llm_io_file:
            return
        try:
            parent = os.path.dirname(self.debug_llm_io_file)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self.debug_llm_io_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            # Never crash inference due to debug persistence.
            logger.warning("Failed to append debug JSONL to %s: %s", self.debug_llm_io_file, e)

    def _append_runtime_jsonl(self, record: Dict[str, object]) -> None:
        if not self.runtime_log_path:
            return
        try:
            with open(self.runtime_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            # Never crash inference due to logging.
            logger.warning("Failed to append runtime JSONL to %s: %s", self.runtime_log_path, e)

    def _should_dump_llm_io(self, references: List[Dict[str, str]], prompt_text: str) -> bool:
        if not self.debug_llm_io:
            return False
        if not self.debug_filter_term:
            return True
        t = self.debug_filter_term
        for r in references or []:
            if t in (r.get("term") or ""):
                return True
            if t in (r.get("translation") or ""):
                return True
        return t in (prompt_text or "")

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
        
        # === DECOUPLED RAG PROCESSING ===
        # RAG is called every step (~120ms) with incremental audio
        # vLLM is only called when accumulated audio reaches vllm_segment_sec or source_finished
        
        # Step 1: Check if we should call vLLM first (to determine if we need cold start)
        samples_since_last_vllm = len(states.source) - states.last_vllm_src_len
        # Use vllm_segment_sec (default 960ms) to control vLLM call interval
        samples_for_vllm_call = int(self.vllm_segment_sec * states.source_sample_rate) if states.source_sample_rate > 0 else 15360
        
        should_call_vllm = (
            states.source_finished or
            samples_since_last_vllm >= samples_for_vllm_call
        )
        
        # Step 2: Accumulate new audio to RAG (every ~120ms)
        if self.rag_retriever:
            # Get new audio samples since last RAG processing
            new_samples_start = states.rag_processed_samples
            new_samples_end = len(states.source)
            
            if new_samples_end > new_samples_start:
                new_audio = np.array(states.source[new_samples_start:new_samples_end], dtype=np.float32)
                
                # Accumulate to RAG with sliding window processing
                # force_process=True only when source is finished
                self.rag_retriever.accumulate_audio(
                    new_audio,
                    force_process=states.source_finished,
                )
                states.rag_processed_samples = new_samples_end
        
        if not should_call_vllm:
            # Not enough audio for vLLM yet, but RAG is still accumulating
            return ReadAction()
        
        with synchronized_timer('generate'):
            increment = self._prepare_speech(states)
            
            # Check if we should log this sample (for debugging first N calls)
            should_log_sample = self.log_sample > 0 and self._log_sample_count < self.log_sample
            
            # Get accumulated RAG references (with sliding window aggregation)
            # KEY FIX: Use increment samples (not total buffer) for top-N calculation
            # top_n = max(ceil(2.5 * increment_sec), 5)
            references: List[Dict[str, str]] = []
            vllm_increment_samples = samples_since_last_vllm
            increment_sec = vllm_increment_samples / states.source_sample_rate if states.source_sample_rate > 0 else 0
            
            if self.rag_retriever:
                # Calculate increment samples for this vLLM call
                references = self.rag_retriever.get_current_references(
                    min_terms=self.rag_min_terms,
                )
                states.references = references
                rag_duration = self.rag_retriever.get_audio_duration()
                
                if references:
                    print(f"[RAG] total_duration={rag_duration:.2f}s, increment={increment_sec:.2f}s, {json.dumps({'reference': references}, ensure_ascii=False)}")
                    self._append_runtime_jsonl(
                        {
                            "type": "rag",
                            "segment_idx": int(getattr(states, "segment_idx", -1)),
                            "rag_audio_duration": round(rag_duration, 2),
                            "vllm_increment_sec": round(increment_sec, 2),
                            "references": references,
                        }
                    )
                
                # Log sample: detailed RAG output
                if should_log_sample:
                    print(f"\n{'='*80}")
                    print(f"[LOG_SAMPLE {self._log_sample_count + 1}/{self.log_sample}] RAG Results")
                    print(f"{'='*80}")
                    print(f"  Total audio duration: {rag_duration:.2f}s")
                    print(f"  vLLM increment: {increment_sec:.2f}s ({vllm_increment_samples} samples)")
                    print(f"  RAG references ({len(references)} terms):")
                    for i, ref in enumerate(references):
                        print(f"    {i+1}. {ref.get('term', '')} -> {ref.get('translation', '')}")
                    if not references:
                        print(f"    (no references)")
            else:
                states.references = []
            
            # Update last vLLM call position
            states.last_vllm_src_len = len(states.source)
            
            inputs = self._prepare_inputs(states, increment, references)
            #print(f"inputs:\n{inputs}")
            self._append_runtime_jsonl(
                {
                    "type": "llm_input",
                    "segment_idx": int(getattr(states, "segment_idx", -1)),
                    "prompt": self._truncate_text(inputs.get("prompt", "")) if isinstance(inputs, dict) else "",
                    "references": references,
                    "sampling_params": self._sampling_params_payload() if self.use_vllm else None,
                }
            )
            
            # Log sample: vLLM input
            if should_log_sample:
                print(f"\n[LOG_SAMPLE {self._log_sample_count + 1}/{self.log_sample}] vLLM Input")
                print(f"{'-'*80}")
                prompt_text = inputs.get("prompt", "") if isinstance(inputs, dict) else ""
                # Print last 2000 chars of prompt (usually contains the most recent audio and term_map)
                if len(prompt_text) > 2000:
                    print(f"  Prompt (last 2000 chars):\n{prompt_text[-2000:]}")
                else:
                    print(f"  Prompt:\n{prompt_text}")
                audio_count = len(inputs.get("multi_modal_data", {}).get("audio", []) or []) if isinstance(inputs, dict) else 0
                print(f"  Audio chunks: {audio_count}")

            if self.use_vllm:
                dump_ok = self._should_dump_llm_io(references, inputs.get("prompt", ""))
                if dump_ok:
                    debug_in = {
                        "type": "llm_input",
                        "segment_idx": int(getattr(states, "segment_idx", -1)),
                        "prompt": self._truncate_text(inputs.get("prompt", "")),
                        "references": references,
                        "sampling_params": self._sampling_params_payload(),
                        "audio_count": len(inputs.get("multi_modal_data", {}).get("audio", []) or []),
                    }
                    print(f"[LLM_INPUT] {json.dumps(debug_in, ensure_ascii=False)}")
                    self._append_debug_jsonl(debug_in)
                outputs = self.model.generate(
                    [inputs], 
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                )
                translation = outputs[0].outputs[0].text
                self._append_runtime_jsonl(
                    {
                        "type": "llm_output",
                        "segment_idx": int(getattr(states, "segment_idx", -1)),
                        "text": self._truncate_text(translation),
                    }
                )
                
                # Log sample: vLLM output
                if should_log_sample:
                    print(f"\n[LOG_SAMPLE {self._log_sample_count + 1}/{self.log_sample}] vLLM Output")
                    print(f"{'-'*80}")
                    print(f"  Translation: {translation}")
                    print(f"{'='*80}\n")
                    self._log_sample_count += 1
                
                if dump_ok:
                    out0 = outputs[0] if outputs else None
                    gen0 = out0.outputs[0] if out0 and getattr(out0, "outputs", None) else None
                    debug_out = {
                        "type": "llm_output",
                        "segment_idx": int(getattr(states, "segment_idx", -1)),
                        "text": self._truncate_text(translation),
                        "finish_reason": getattr(gen0, "finish_reason", None) if gen0 else None,
                        "output_tokens": len(getattr(gen0, "token_ids", []) or []) if gen0 else None,
                    }
                    print(f"[LLM_OUTPUT] {json.dumps(debug_out, ensure_ascii=False)}")
                    self._append_debug_jsonl(debug_out)
            # else:
            #     text_ids, _ = self.model.generate(
            #         **inputs,
            #         generation_config=self.generation_config,
            #         return_audio=False,
            #         thinker_return_dict_in_generate=True,
            #         use_audio_in_video=False,
            #     )
            #     translation = self.processor.batch_decode(
            #         text_ids.sequences[:, inputs["input_ids"].shape[1] :],
            #         skip_special_tokens=True,
            #         clean_up_tokenization_spaces=False
            #     )[0]
            #     self._append_runtime_jsonl(
            #         {
            #             "type": "llm_output",
            #             "segment_idx": int(getattr(states, "segment_idx", -1)),
            #             "text": self._truncate_text(translation),
            #         }
            #     )
            #     dump_ok = self._should_dump_llm_io(references, "")
            #     if dump_ok:
            #         debug_out = {
            #             "type": "llm_output",
            #             "segment_idx": int(getattr(states, "segment_idx", -1)),
            #             "text": self._truncate_text(translation),
            #             "generation_config": dict(self.generation_config.to_dict()) if hasattr(self.generation_config, "to_dict") else str(self.generation_config),
            #         }
            #         print(f"[LLM_OUTPUT] {json.dumps(debug_out, ensure_ascii=False)}")
            #         self._append_debug_jsonl(debug_out)

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
