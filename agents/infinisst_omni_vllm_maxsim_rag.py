import os

os.environ['VLLM_USE_V1'] = '0'

# ======Configuration=====
EFF_PRINT_DECIMALS = 4
DEFAULT_GEN_SEED = 998244353
DEFAULT_VLLM_ENABLE_PREFIX_CACHING = 1
VLLM_ENABLE_PREFIX_CACHING_ENV = "VLLM_ENABLE_PREFIX_CACHING"
UNIT_DURATION_SEC = 0.96
DEFAULT_RAG_TIMELINE_LOOKBACK_SEC = 1.92
# ======Configuration=====

import re
import json
import contextlib
from time import perf_counter, time

from typing import Optional, List, Dict
from simuleval.agents.states import AgentStates
from simuleval.utils import entrypoint
from simuleval.data.segments import SpeechSegment
from simuleval.agents import SpeechToTextAgent
from simuleval.agents.actions import WriteAction, ReadAction
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
    Qwen3OmniMoeConfig,
)
from qwen_omni_utils import process_mm_info

from vllm import LLM, SamplingParams

from agents.options import add_simuleval_args, add_gen_args

import logging

logger = logging.getLogger(__name__)


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return int(default)
    return int(value)


def _env_int_or_auto(name, default):
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return int(default)
    if str(value).strip().lower() == "auto":
        return int(default)
    return int(value)


from agents.streaming_maxsim_retriever import (
    MAXSIM_STRIDE,
    MAXSIM_WINDOWS,
    StreamingMaxSimRetriever,
)


@contextlib.contextmanager
def synchronized_elapsed_timer():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = perf_counter()
    out = {"sec": 0.0}
    try:
        yield out
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        out["sec"] = float(perf_counter() - t0)


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
    rag_processed_samples: int
    last_vllm_src_len: int
    rag_total_sec: float
    rag_call_count: int
    rag_blocking_total_sec: float
    rag_blocking_call_count: int
    rag_last_retrieve_src_len: int
    accumulated_rag_results: list
    MAX_SRC_LEN = 16000 * 3600

    def reset(self):
        super().reset()
        self.src_len = 0
        self.target_ids = []
        self.segment_idx = 0
        self.messages = []
        self.references = []
        self.rag_processed_samples = 0
        self.last_vllm_src_len = 0
        self.rag_total_sec = 0.0
        self.rag_call_count = 0
        self.rag_blocking_total_sec = 0.0
        self.rag_blocking_call_count = 0
        self.rag_last_retrieve_src_len = 0
        self.accumulated_rag_results = []


@entrypoint
class InfiniSSTOmniVLLMMaxSimRAG(SpeechToTextAgent):

    def __init__(self, args):
        super().__init__(args)
        self.seed = int(getattr(args, "seed", DEFAULT_GEN_SEED))
        transformers.set_seed(self.seed)

        self.min_start_sec = args.min_start_sec
        self.source_lang = args.source_lang
        self.target_lang = args.target_lang
        self.system_prompt_style = (
            getattr(args, "system_prompt_style", "translate_task") or "translate_task"
        ).strip()
        if self.system_prompt_style not in {"translate_task", "given_chunks"}:
            logger.warning(
                "Unknown system_prompt_style=%r; falling back to translate_task",
                self.system_prompt_style,
            )
            self.system_prompt_style = "translate_task"
        self.vllm_segment_sec = getattr(args, "vllm_segment_sec", 0.96)

        self.log_sample = int(getattr(args, "log_sample", 0))
        self._log_sample_count = 0

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

        self.max_cache_chunks = args.max_cache_chunks
        self.keep_cache_chunks = args.keep_cache_chunks

        self.rag_retriever: Optional[StreamingMaxSimRetriever] = None
        self.rag_top_k = getattr(args, "rag_top_k", 10)
        self.rag_target_lang = getattr(args, "rag_target_lang", "zh")
        self.term_map_format = (getattr(args, "term_map_format", "plain") or "plain").strip()
        if self.term_map_format not in {"plain", "tagged", "xml_tagged"}:
            logger.warning(
                "Unknown term_map_format=%r; falling back to plain",
                self.term_map_format,
            )
            self.term_map_format = "plain"
        self.empty_term_map_policy = (
            getattr(args, "empty_term_map_policy", "none_block") or "none_block"
        ).strip()
        if self.empty_term_map_policy not in {"none_block", "omit"}:
            logger.warning(
                "Unknown empty_term_map_policy=%r; falling back to none_block",
                self.empty_term_map_policy,
            )
            self.empty_term_map_policy = "none_block"
        self.rag_streaming_mode = (
            getattr(args, "rag_streaming_mode", "timeline") or "timeline"
        ).strip()
        valid_rag_modes = {"timeline", "timeline_stride_debug", "stride_merge", "direct"}
        if self.rag_streaming_mode not in valid_rag_modes:
            logger.warning(
                "Unknown rag_streaming_mode=%r; falling back to timeline",
                self.rag_streaming_mode,
            )
            self.rag_streaming_mode = "timeline"

        raw_stride = getattr(args, "rag_retrieve_stride_sec", 0.0)
        if raw_stride > 0:
            self.rag_retrieve_stride_sec = raw_stride
        else:
            self.rag_retrieve_stride_sec = self.vllm_segment_sec
        self.rag_timeline_lookback_sec = max(
            0.0,
            float(
                getattr(
                    args,
                    "rag_timeline_lookback_sec",
                    DEFAULT_RAG_TIMELINE_LOOKBACK_SEC,
                )
            ),
        )
        self.rag_sliding_window_enabled = (
            self.rag_streaming_mode in {"stride_merge", "timeline_stride_debug"}
            and self.rag_retrieve_stride_sec > 0
        )

        self.debug_audio_dir = getattr(args, "debug_audio_dir", "") or ""
        self._vllm_call_count = 0
        self.oracle_term_map_path = (getattr(args, "oracle_term_map_path", "") or "").strip()
        self.oracle_term_map = self._load_oracle_term_map(self.oracle_term_map_path)

        if self.oracle_term_map_path and getattr(args, "rag_enabled", False):
            logger.warning(
                "--oracle-term-map-path was provided; disabling learned retriever "
                "and using oracle term_map entries."
            )

        if getattr(args, "rag_enabled", False) and not self.oracle_term_map_path:
            rag_window_sec = (
                self.vllm_segment_sec + self.rag_timeline_lookback_sec
                if self.rag_streaming_mode in {
                    "timeline",
                    "timeline_stride_debug",
                    "stride_merge",
                }
                else 0.0
            )
            logger.info(
                "Initializing StreamingMaxSimRetriever "
                "(mode=%s, retrieval_span=%.2fs, current=%.2fs, lookback=%.2fs, "
                "stride=%.2fs)...",
                self.rag_streaming_mode,
                rag_window_sec,
                self.vllm_segment_sec,
                self.rag_timeline_lookback_sec,
                self.rag_retrieve_stride_sec,
            )
            self.rag_retriever = StreamingMaxSimRetriever(
                model_path=getattr(args, "rag_model_path", None),
                index_path=getattr(args, "rag_index_path", None),
                device=getattr(args, "rag_device", "cuda:1"),
                top_k=self.rag_top_k,
                lora_rank=getattr(args, "rag_lora_r", 128),
                text_lora_rank=getattr(args, "rag_text_lora_r", 128),
                target_lang=self.rag_target_lang,
                window_sec=rag_window_sec,
                score_threshold=float(getattr(args, "rag_score_threshold", 0.0)),
                maxsim_windows=list(getattr(args, "rag_maxsim_windows", MAXSIM_WINDOWS)),
                maxsim_stride=int(getattr(args, "rag_maxsim_stride", MAXSIM_STRIDE)),
            )
            if not self.rag_retriever or not self.rag_retriever.enabled:
                logger.warning("MaxSim RAG retriever not operational")
                self.rag_retriever = None

        self.use_vllm = args.use_vllm
        self.gpu_memory_utilization = getattr(args, "gpu_memory_utilization", 0.8)
        self.vllm_enforce_eager = int(getattr(args, "vllm_enforce_eager", 0))
        self.vllm_prompt_audio_limit = _env_int_or_auto(
            "VLLM_LIMIT_AUDIO_OVERRIDE", self.max_cache_chunks
        )
        self.debug_llm_io = bool(getattr(args, "debug_llm_io", False))
        self.debug_max_chars = int(getattr(args, "debug_max_chars", 6000))
        self.debug_llm_io_file = (getattr(args, "debug_llm_io_file", "") or "").strip() or None

        self.runtime_log_dir = (
            getattr(args, "runtime_log_dir", "/mnt/gemini/data2/jiaxuanluo/converted_logs") or ""
        ).strip()
        self.runtime_log_enabled = bool(getattr(args, "runtime_log_enabled", True))
        self.runtime_log_path = None

        if self.runtime_log_enabled and self.runtime_log_dir:
            try:
                os.makedirs(self.runtime_log_dir, exist_ok=True)
                self.runtime_log_path = os.path.join(
                    self.runtime_log_dir,
                    f"runtime_omni_vllm_maxsim_rag_{int(time())}_pid{os.getpid()}.jsonl",
                )
                logger.info("Runtime log: %s", self.runtime_log_path)
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
        parser.add_argument(
            "--vllm-enforce-eager", type=int, default=0,
            help="If 1, disable vLLM torch.compile/cudagraph (reduces startup latency).",
        )
        parser.add_argument("--max-cache-chunks", type=int, default=120)
        parser.add_argument("--keep-cache-chunks", type=int, default=60)
        parser.add_argument("--vllm-segment-sec", type=float, default=0.96)
        parser.add_argument("--log-sample", type=int, default=0)
        parser.add_argument("--rag-enabled", action="store_true")
        parser.add_argument("--rag-index-path", type=str, default=None)
        parser.add_argument("--rag-model-path", type=str, default=None)
        parser.add_argument("--rag-device", type=str, default="cuda:1")
        parser.add_argument("--rag-top-k", type=int, default=10)
        parser.add_argument(
            "--rag-score-threshold", type=float, default=0.0,
            help="Drop retrieved glossary candidates with MaxSim score below "
                 "this threshold. 0.0 = no filtering (keep all top-k). "
                 "Current dev-calibrated default in launchers is tau=0.73.",
        )
        parser.add_argument("--rag-target-lang", type=str, default="zh")
        parser.add_argument("--rag-lora-r", type=int, default=128)
        parser.add_argument("--rag-text-lora-r", type=int, default=128)
        parser.add_argument(
            "--rag-maxsim-windows",
            type=int,
            nargs="+",
            default=MAXSIM_WINDOWS,
            help="MaxSim pooling windows in encoder frames; must match the retriever checkpoint.",
        )
        parser.add_argument(
            "--rag-maxsim-stride",
            type=int,
            default=MAXSIM_STRIDE,
            help="MaxSim pooling stride in encoder frames; must match the retriever checkpoint.",
        )
        parser.add_argument(
            "--rag-retrieve-stride-sec", type=float, default=0.0,
            help="Legacy/intermediate retriever stride in seconds "
                 "(0 = auto = vllm_segment_sec). Timeline inference retrieves once "
                 "per vLLM generation step.",
        )
        parser.add_argument(
            "--rag-timeline-lookback-sec",
            type=float,
            default=DEFAULT_RAG_TIMELINE_LOOKBACK_SEC,
            help="Timeline retrieval lookback before the current vLLM chunk. "
                 "Default 1.92s means each retrieval scores current lm*0.96s "
                 "audio plus a 1.92s left context.",
        )
        parser.add_argument(
            "--rag-streaming-mode",
            type=str,
            default="timeline",
            choices=["timeline", "timeline_stride_debug", "stride_merge", "direct"],
            help=(
                "RAG streaming strategy. 'timeline' encodes previous+current audio "
                "at each vLLM call, then filters MaxSim windows ending before the "
                "current chunk. 'timeline_stride_debug' additionally runs the old "
                "intermediate stride retrieves for runtime logging, but still feeds "
                "only the final timeline-aware term_map to vLLM. 'stride_merge' is "
                "the older multi-retrieve term-list merge. 'direct' retrieves only "
                "the newly accumulated audio."
            ),
        )
        parser.add_argument("--debug-audio-dir", type=str, default="")
        parser.add_argument("--debug-llm-io", action="store_true")
        parser.add_argument("--debug-max_chars", type=int, default=6000)
        parser.add_argument("--debug-llm-io-file", type=str, default="")
        parser.add_argument(
            "--oracle-term-map-path",
            type=str,
            default="",
            help=(
                "Optional JSON term map for oracle/GT evaluation. Expected schema: "
                "a list of rows with start_sec, end_sec, and references=[{term, translation}]. "
                "When set, learned MaxSim retrieval is skipped."
            ),
        )
        parser.add_argument(
            "--term-map-format",
            type=str,
            default="plain",
            choices=["plain", "tagged", "xml_tagged"],
            help=(
                "Term-map serialization in the Speech LLM prompt. "
                "plain uses 'source=target'; tagged uses '[TERM] source => target [/TERM]'; "
                "xml_tagged uses '<term>source => target</term>'."
            ),
        )
        parser.add_argument(
            "--empty-term-map-policy",
            type=str,
            default="none_block",
            choices=["none_block", "omit"],
            help=(
                "How to serialize empty RAG results. none_block emits "
                "'term_map: NONE'; omit leaves the user turn as audio only."
            ),
        )
        parser.add_argument(
            "--system-prompt-style",
            type=str,
            default="translate_task",
            choices=["translate_task", "given_chunks"],
            help=(
                "System prompt wording. translate_task keeps the historical "
                "'Your task is to translate ... audio chunks ...' prompt; "
                "given_chunks matches the cap16 denoise-budget SFT JSONL prompt."
            ),
        )
        parser.add_argument("--runtime-log-enabled", type=int, default=1)
        parser.add_argument("--runtime-log-dir", type=str, default="/mnt/gemini/data2/jiaxuanluo/converted_logs")

    def build_states(self):
        if hasattr(self, "rag_retriever") and self.rag_retriever:
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
            rag_total_sec=0.0,
            rag_call_count=0,
            rag_blocking_total_sec=0.0,
            rag_blocking_call_count=0,
            rag_last_retrieve_src_len=0,
            accumulated_rag_results=[],
        )

    def load_model(self, args):
        if args.use_vllm:
            gpu_memory_util = self.gpu_memory_utilization
            tp_size = int(os.environ.get("VLLM_TP_SIZE_OVERRIDE", "2"))
            enforce_eager = bool(int(getattr(self, "vllm_enforce_eager", 0)))
            max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN_OVERRIDE", "32768"))
            limit_audio = _env_int_or_auto("VLLM_LIMIT_AUDIO_OVERRIDE", self.max_cache_chunks)
            enable_prefix_caching = bool(
                int(
                    os.environ.get(
                        VLLM_ENABLE_PREFIX_CACHING_ENV,
                        str(DEFAULT_VLLM_ENABLE_PREFIX_CACHING),
                    )
                )
            )

            disable_custom_ar = bool(
                int(os.environ.get("VLLM_DISABLE_CUSTOM_ALL_REDUCE", "0"))
            )
            llm_kwargs = dict(
                model=args.model_name,
                trust_remote_code=True,
                gpu_memory_utilization=gpu_memory_util,
                tensor_parallel_size=tp_size,
                limit_mm_per_prompt={"audio": limit_audio},
                max_num_seqs=1,
                max_model_len=max_model_len,
                enable_prefix_caching=enable_prefix_caching,
                enforce_eager=enforce_eager,
            )
            if disable_custom_ar:
                llm_kwargs["disable_custom_all_reduce"] = True
                logger.info(
                    "VLLM_DISABLE_CUSTOM_ALL_REDUCE=1: forcing NCCL all-reduce "
                    "(bypasses CUDA IPC P2P; useful when node P2P is corrupted)"
                )
            try:
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        free, total = torch.cuda.mem_get_info(i)
                        logger.info(
                            "[PRE-VLLM MEM] cuda:%d free=%.2f GiB total=%.2f GiB (used=%.2f GiB)",
                            i, free / 1024**3, total / 1024**3,
                            (total - free) / 1024**3,
                        )
            except Exception as _e:
                logger.warning("[PRE-VLLM MEM] probe failed: %r", _e)
            self.model = LLM(**llm_kwargs)
            self.sampling_params = SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_tokens=self.max_new_tokens,
                seed=self.seed,
            )
        else:
            attn_impl = os.environ.get(
                "TRANSFORMERS_ATTN_IMPLEMENTATION_OVERRIDE",
                "flash_attention_2",
            )
            self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                args.model_name,
                dtype="auto",
                device_map="auto",
                attn_implementation=attn_impl,
                enable_audio_output=False,
            )
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_name)

    def _format_term_map_kv(self, term_map: Dict[str, str]) -> str:
        lines: List[str] = []
        for k, v in (term_map or {}).items():
            kk = str(k).replace("\n", " ").strip()
            vv = str(v).replace("\n", " ").strip()
            if not kk or not vv:
                continue
            if self.term_map_format == "xml_tagged":
                lines.append(f"<term>{kk} => {vv}</term>")
            elif self.term_map_format == "tagged":
                lines.append(f"[TERM] {kk} => {vv} [/TERM]")
            else:
                lines.append(f"{kk}={vv}")
        return "\n".join(lines)

    def _use_chinese_training_prompt(self) -> bool:
        return (
            (self.source_lang or "").strip().lower() in {"english", "en"}
            and (self.target_lang or "").strip().lower() in {"chinese", "zh", "zh-cn", "中文"}
        )

    def _build_system_prompt(self, rag_enabled: bool) -> str:
        if self._use_chinese_training_prompt():
            return (
                "You are a professional simultaneous interpreter. "
                "Your task is to translate English audio chunks into accurate and fluent "
                "Chinese. Use the ‘term_map’ as a reference for terminology if provided."
            )
        if self.system_prompt_style == "given_chunks":
            system_text = (
                f"You are a professional simultaneous interpreter. "
                f"You will be given chunks of {self.source_lang} audio and you need to "
                f"translate the audio into {self.target_lang} text."
            )
        else:
            system_text = (
                f"You are a professional simultaneous interpreter. "
                f"Your task is to translate {self.source_lang} audio chunks into "
                f"accurate and fluent {self.target_lang}."
            )
        if rag_enabled:
            system_text += " Use the 'term_map' as a reference for terminology if provided."
        return system_text

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
            increment = np.pad(
                increment, (0, 15360 - len(increment)), mode="constant", constant_values=0
            )

        states.src_len = len(states.source)
        return increment

    def _reset_rag_stream_state(self, states, reason: str) -> None:
        """Reset per-source RAG buffers when SimulEval reuses the agent object."""
        if not self.rag_retriever:
            return
        self.rag_retriever.reset()
        states.rag_processed_samples = 0
        states.rag_total_sec = 0.0
        states.rag_call_count = 0
        states.rag_blocking_total_sec = 0.0
        states.rag_blocking_call_count = 0
        states.rag_last_retrieve_src_len = 0
        states.accumulated_rag_results = []
        states.references = []
        self._vllm_call_count = 0
        print(f"[RAG] Reset stream state for new source ({reason})")

    def _maybe_reset_rag_for_new_source(self, states) -> None:
        if not self.rag_retriever:
            return

        processed = int(getattr(states, "rag_processed_samples", 0))
        if len(states.source) < processed:
            self._reset_rag_stream_state(states, "source length moved backwards")
            return

        # SimulEval may reuse one agent object across multiple full-corpus
        # instances and call states.reset() instead of agent.build_states().
        # In that case state counters go back to zero, but the retriever audio
        # buffer can still contain the previous talk unless reset here.
        state_looks_new = (
            processed == 0
            and int(getattr(states, "segment_idx", 0)) == 0
            and not getattr(states, "messages", [])
        )
        if not state_looks_new:
            return

        rag_audio_sec = float(self.rag_retriever.get_audio_duration())
        if rag_audio_sec > 1e-6:
            self._reset_rag_stream_state(
                states,
                f"state reset while retriever still had {rag_audio_sec:.2f}s audio",
            )

    def _prepare_inputs(self, states, increment, references):
        rag_enabled = self.rag_retriever is not None or bool(self.oracle_term_map_path)
        if len(states.messages) == 0:
            system_text = self._build_system_prompt(rag_enabled=rag_enabled)
            states.messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_text}]}
            )
            print(
                f"lang: {self.source_lang} -> {self.target_lang}, "
                f"system_text: {system_text}, rag_enabled: {rag_enabled}"
            )

        user_content = [{"type": "audio", "audio": increment}]

        norm_refs: Dict[str, str] = {}
        for r in references:
            term = (r.get("term") or "").strip()
            translation = (r.get("translation") or "").strip()
            if term and translation:
                norm_refs[term] = translation

        if norm_refs:
            kv = self._format_term_map_kv(norm_refs)
            if kv:
                user_content.append({"type": "text", "text": f"\n\nterm_map:\n{kv}"})
            elif rag_enabled and self.empty_term_map_policy == "none_block":
                user_content.append({"type": "text", "text": "\n\nterm_map:\nNONE"})
        elif rag_enabled and self.empty_term_map_policy == "none_block":
            user_content.append({"type": "text", "text": "\n\nterm_map:\nNONE"})

        if self.use_vllm and self.vllm_prompt_audio_limit > 0:
            keep_pairs = max(0, self.vllm_prompt_audio_limit - 1)
            if keep_pairs == 0:
                states.messages = states.messages[:1]
            else:
                body_messages = states.messages[1:]
                max_body_messages = 2 * keep_pairs
                if len(body_messages) > max_body_messages:
                    states.messages = [states.messages[0]] + body_messages[-max_body_messages:]

        states.messages.append({"role": "user", "content": user_content})

        text = self.processor.apply_chat_template(
            states.messages, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(states.messages, use_audio_in_video=False)

        if self.use_vllm:
            inputs = {
                "prompt": text,
                "multi_modal_data": {"audio": audios},
                "mm_processor_kwargs": {"use_audio_in_video": False},
            }
        else:
            inputs = self.processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=False,
            )
            inputs["input_features"] = inputs["input_features"].to(self.model.dtype)
        return inputs

    def _truncate_text(self, text: str) -> str:
        if not text or self.debug_max_chars <= 0 or len(text) <= self.debug_max_chars:
            return text
        return text[: self.debug_max_chars] + "\n...[truncated]..."

    def _append_runtime_jsonl(self, record: Dict[str, object]) -> None:
        if not self.runtime_log_path:
            return
        try:
            with open(self.runtime_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    @staticmethod
    def _load_oracle_term_map(path: str) -> List[Dict]:
        if not path:
            return []
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Oracle term_map file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Oracle term_map must be a JSON list: {path}")

        rows: List[Dict] = []
        for idx, row in enumerate(data):
            if not isinstance(row, dict):
                raise ValueError(f"Oracle term_map row {idx} is not an object")
            try:
                start_sec = float(row["start_sec"])
                end_sec = float(row["end_sec"])
            except KeyError as exc:
                raise ValueError(f"Oracle term_map row {idx} missing {exc.args[0]}") from exc
            if end_sec < start_sec:
                raise ValueError(
                    f"Oracle term_map row {idx} has end_sec < start_sec: "
                    f"{start_sec} > {end_sec}"
                )
            references = row.get("references", [])
            if not isinstance(references, list):
                raise ValueError(f"Oracle term_map row {idx} references must be a list")
            clean_refs: List[Dict] = []
            for ref_idx, ref in enumerate(references):
                if not isinstance(ref, dict):
                    raise ValueError(
                        f"Oracle term_map row {idx} reference {ref_idx} is not an object"
                    )
                term = str(ref.get("term") or "").strip()
                translation = str(ref.get("translation") or "").strip()
                if not term or not translation:
                    raise ValueError(
                        f"Oracle term_map row {idx} reference {ref_idx} "
                        "requires non-empty term and translation"
                    )
                clean_refs.append({"term": term, "translation": translation})
            rows.append(
                {
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "references": clean_refs,
                }
            )
        logger.info("Loaded %d oracle term_map rows from %s", len(rows), path)
        return rows

    @staticmethod
    def _time_overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
        return a_start < b_end and b_start < a_end

    def _oracle_references_for_window(
        self,
        states: S2TAgentStates,
        current_start_sec: float,
        current_end_sec: float,
    ) -> List[Dict]:
        refs_by_term: Dict[str, Dict] = {}
        for row in self.oracle_term_map:
            row_start = float(row["start_sec"])
            row_end = float(row["end_sec"])
            if not self._time_overlaps(current_start_sec, current_end_sec, row_start, row_end):
                continue
            for ref in row["references"]:
                term = ref["term"]
                refs_by_term[term] = {
                    "term": term,
                    "translation": ref["translation"],
                    "key": term,
                    "score": 1.0,
                    "retrieval_mode": "oracle",
                    "time_start": row_start,
                    "time_end": row_end,
                    "oracle_row_start_sec": row_start,
                    "oracle_row_end_sec": row_end,
                }
        refs = list(refs_by_term.values())[: self.rag_top_k]
        self._append_runtime_jsonl({
            "type": "rag_window",
            "trigger": "vllm_timeline",
            "segment_idx": int(getattr(states, "segment_idx", -1)),
            "rag_call_count": int(getattr(states, "rag_call_count", 0)),
            "rag_audio_duration": round(float(current_end_sec), 2),
            "rag_sec": 0.0,
            "blocking_for_vllm": False,
            "source_samples": int(len(getattr(states, "source", []))),
            "current_start_sec": round(float(current_start_sec), 3),
            "current_end_sec": round(float(current_end_sec), 3),
            "lookback_sec": 0.0,
            "oracle_term_map_path": self.oracle_term_map_path,
            "references": refs,
        })
        return refs

    def _do_retrieve(
        self,
        states: S2TAgentStates,
        trigger: str = "retrieve",
        blocking_for_vllm: bool = False,
    ) -> tuple[List[Dict], float]:
        """Run one retriever call, update timing stats, and return results."""
        assert self.rag_retriever is not None
        with synchronized_elapsed_timer() as t_rag:
            if self.rag_streaming_mode in {"stride_merge", "timeline_stride_debug"}:
                results = self.rag_retriever.retrieve_window_with_times(self.rag_top_k)
            else:
                results = self.rag_retriever.retrieve(self.rag_top_k)
        rag_sec = float(t_rag["sec"])
        states.rag_total_sec += rag_sec
        states.rag_call_count += 1
        if blocking_for_vllm:
            states.rag_blocking_total_sec += rag_sec
            states.rag_blocking_call_count += 1
        print(f"[TIME] maxsim_retrieve_sec={rag_sec:.{EFF_PRINT_DECIMALS}f}")
        self._append_runtime_jsonl({
            "type": "rag_window",
            "trigger": trigger,
            "segment_idx": int(getattr(states, "segment_idx", -1)),
            "rag_call_count": int(getattr(states, "rag_call_count", 0)),
            "rag_audio_duration": round(self.rag_retriever.get_audio_duration(), 2),
            "rag_sec": round(rag_sec, 6),
            "blocking_for_vllm": bool(blocking_for_vllm),
            "source_samples": int(len(getattr(states, "source", []))),
            "references": results,
        })
        return results, rag_sec

    def _do_retrieve_timeline(
        self,
        states: S2TAgentStates,
        current_start_sec: float,
        current_end_sec: float,
    ) -> tuple[List[Dict], float]:
        """Run timeline-aware previous+current MaxSim retrieval for vLLM."""
        assert self.rag_retriever is not None
        with synchronized_elapsed_timer() as t_rag:
            lookback_sec = self.rag_timeline_lookback_sec
            results = self.rag_retriever.retrieve_timeline(
                top_k=self.rag_top_k,
                current_start_sec=current_start_sec,
                current_end_sec=current_end_sec,
                lookback_sec=lookback_sec,
            )
        rag_sec = float(t_rag["sec"])
        states.rag_total_sec += rag_sec
        states.rag_call_count += 1
        states.rag_blocking_total_sec += rag_sec
        states.rag_blocking_call_count += 1
        print(f"[TIME] maxsim_retrieve_timeline_sec={rag_sec:.{EFF_PRINT_DECIMALS}f}")
        self._append_runtime_jsonl({
            "type": "rag_window",
            "trigger": "vllm_timeline",
            "segment_idx": int(getattr(states, "segment_idx", -1)),
            "rag_call_count": int(getattr(states, "rag_call_count", 0)),
            "rag_audio_duration": round(self.rag_retriever.get_audio_duration(), 2),
            "rag_sec": round(rag_sec, 6),
            "blocking_for_vllm": True,
            "source_samples": int(len(getattr(states, "source", []))),
            "current_start_sec": round(float(current_start_sec), 3),
            "current_end_sec": round(float(current_end_sec), 3),
            "lookback_sec": round(float(lookback_sec), 3),
            "references": results,
        })
        return results, rag_sec

    @torch.inference_mode()
    def policy(self, states: Optional[S2TAgentStates] = None):
        if states is None:
            states = self.states
        length_in_seconds = (
            float(len(states.source)) / states.source_sample_rate
            if states.source_sample_rate > 0
            else 0
        )

        if not states.source_finished and length_in_seconds < self.min_start_sec:
            return ReadAction()

        if states.source_finished and length_in_seconds < 0.32:
            return WriteAction(content="", finished=True)

        sr = states.source_sample_rate if states.source_sample_rate > 0 else 16000
        self._maybe_reset_rag_for_new_source(states)

        samples_since_last_vllm = len(states.source) - states.last_vllm_src_len
        samples_for_vllm_call = int(self.vllm_segment_sec * sr)

        should_call_vllm = (
            states.source_finished
            or samples_since_last_vllm >= samples_for_vllm_call
        )

        if self.rag_retriever:
            new_samples_start = states.rag_processed_samples
            new_samples_end = len(states.source)
            if new_samples_end > new_samples_start:
                new_audio = np.array(
                    states.source[new_samples_start:new_samples_end], dtype=np.float32
                )
                self.rag_retriever.accumulate_audio(new_audio)
                states.rag_processed_samples = new_samples_end

        # --- Legacy decoupled retriever: runs at stride frequency (2x vLLM) ---
        if self.rag_retriever and self.rag_sliding_window_enabled:
            rag_samples_since = len(states.source) - states.rag_last_retrieve_src_len
            stride_samples = int(self.rag_retrieve_stride_sec * sr)
            should_call_retriever = (
                states.source_finished
                or rag_samples_since >= stride_samples
            )
            if should_call_retriever and not should_call_vllm:
                results, _rag_sec = self._do_retrieve(
                    states, trigger="interim", blocking_for_vllm=False
                )
                states.accumulated_rag_results.append(results)
                states.rag_last_retrieve_src_len = len(states.source)

        if not should_call_vllm:
            return ReadAction()

        with synchronized_elapsed_timer() as t_gen:
            increment = self._prepare_speech(states)

            if self.debug_audio_dir:
                import soundfile as sf

                vllm_wav_path = os.path.join(
                    self.debug_audio_dir, f"vllm_inc_call{self._vllm_call_count:03d}.wav"
                )
                sf.write(vllm_wav_path, increment, 16000)
                logger.info("Saved vLLM increment audio to %s", vllm_wav_path)
                self._vllm_call_count += 1

            references = []
            rag_blocking_sec = 0.0
            if self.oracle_term_map_path:
                current_start_sec = float(states.last_vllm_src_len) / float(sr)
                current_end_sec = float(len(states.source)) / float(sr)
                references = self._oracle_references_for_window(
                    states,
                    current_start_sec=current_start_sec,
                    current_end_sec=current_end_sec,
                )
                states.references = references
                states.rag_call_count += 1
                self._append_runtime_jsonl({
                    "type": "rag",
                    "segment_idx": int(getattr(states, "segment_idx", -1)),
                    "rag_audio_duration": round(float(current_end_sec), 2),
                    "rag_blocking_sec": 0.0,
                    "rag_blocking_total_sec": round(
                        float(getattr(states, "rag_blocking_total_sec", 0.0)), 6
                    ),
                    "rag_blocking_call_count": int(
                        getattr(states, "rag_blocking_call_count", 0)
                    ),
                    "rag_total_sec": round(float(getattr(states, "rag_total_sec", 0.0)), 6),
                    "rag_call_count": int(getattr(states, "rag_call_count", 0)),
                    "rag_streaming_mode": "oracle",
                    "oracle_term_map_path": self.oracle_term_map_path,
                    "references": references,
                })
            elif self.rag_retriever:
                if self.rag_streaming_mode in {"timeline", "timeline_stride_debug"}:
                    current_start_sec = float(states.last_vllm_src_len) / float(sr)
                    current_end_sec = float(len(states.source)) / float(sr)
                    references, rag_blocking_sec = self._do_retrieve_timeline(
                        states,
                        current_start_sec=current_start_sec,
                        current_end_sec=current_end_sec,
                    )
                    states.rag_last_retrieve_src_len = len(states.source)
                    # In timeline_stride_debug mode these are only diagnostics.
                    # Do not merge stale stride term lists into the vLLM prompt.
                    states.accumulated_rag_results = []
                else:
                    final_results, rag_blocking_sec = self._do_retrieve(
                        states, trigger="vllm_final", blocking_for_vllm=True
                    )
                    states.rag_last_retrieve_src_len = len(states.source)

                    if self.rag_sliding_window_enabled and states.accumulated_rag_results:
                        states.accumulated_rag_results.append(final_results)
                        current_start_sec = float(states.last_vllm_src_len) / float(sr)
                        references = StreamingMaxSimRetriever.merge_results(
                            states.accumulated_rag_results,
                            self.rag_top_k,
                            min_time_end=current_start_sec,
                        )
                        n_windows = len(states.accumulated_rag_results)
                        states.accumulated_rag_results = []
                        print(
                            f"[RAG] Merged {n_windows} sliding windows "
                            f"(drop time_end <= {current_start_sec:.2f}s)"
                        )
                    else:
                        references = final_results

                states.references = references
                rag_duration = self.rag_retriever.get_audio_duration()
                if references:
                    self._append_runtime_jsonl({
                        "type": "rag",
                        "segment_idx": int(getattr(states, "segment_idx", -1)),
                        "rag_audio_duration": round(rag_duration, 2),
                        "rag_blocking_sec": round(float(rag_blocking_sec), 6),
                        "rag_blocking_total_sec": round(
                            float(getattr(states, "rag_blocking_total_sec", 0.0)), 6
                        ),
                        "rag_blocking_call_count": int(
                            getattr(states, "rag_blocking_call_count", 0)
                        ),
                        "rag_total_sec": round(float(getattr(states, "rag_total_sec", 0.0)), 6),
                        "rag_call_count": int(getattr(states, "rag_call_count", 0)),
                        "rag_streaming_mode": self.rag_streaming_mode,
                        "references": references,
                    })

            states.last_vllm_src_len = len(states.source)
            inputs = self._prepare_inputs(states, increment, references)

            if references:
                ref_str = ", ".join(
                    [f"{r['term']}->{r['translation']}" for r in references]
                )
                print(f"\n[VLLM] Final TermMap (Top-{len(references)}): {ref_str}")
            else:
                print("\n[VLLM] Final TermMap: [None]")

            self._append_runtime_jsonl({
                "type": "llm_input",
                "segment_idx": int(getattr(states, "segment_idx", -1)),
                "prompt": self._truncate_text(inputs.get("prompt", ""))
                if isinstance(inputs, dict)
                else "",
                "references": references,
            })

            if self.use_vllm:
                outputs = self.model.generate(
                    [inputs], sampling_params=self.sampling_params, use_tqdm=False
                )
                translation = outputs[0].outputs[0].text
                print(f"[VLLM] Output Translation: {translation}")
                self._append_runtime_jsonl({
                    "type": "llm_output",
                    "segment_idx": int(getattr(states, "segment_idx", -1)),
                    "text": self._truncate_text(translation),
                })
            else:
                translation = ""

            states.messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": translation}]}
            )
            body_messages = states.messages[1:]
            if len(body_messages) > 2 * self.max_cache_chunks:
                states.messages = [states.messages[0]] + body_messages[
                    -2 * self.keep_cache_chunks:
                ]

            if states.source_finished:
                states.segment_idx = -1

        gen_sec = float(t_gen["sec"])
        print(f"[TIME] vllm_generate_total_sec={gen_sec:.{EFF_PRINT_DECIMALS}f}")

        print("".join(states.target))
        states.segment_idx += 1
        return (
            WriteAction(content=translation, finished=states.source_finished)
            if translation != "" or states.source_finished
            else ReadAction()
        )
