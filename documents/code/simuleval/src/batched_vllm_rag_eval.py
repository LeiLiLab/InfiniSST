#!/usr/bin/env python3
"""Run multiple streaming RAG eval streams through one shared vLLM instance.

This is a throughput prototype that intentionally does not modify the existing
SimulEval agent or serial launchers.  It reads SimulEval-style source/target
lists, schedules independent (sample, latency-multiplier) streams, batches the
vLLM generate calls, and writes per-lm `instances.log` plus runtime JSONL files
compatible with `offline_streamlaal_eval.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("VLLM_USE_V1", "0")

import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[4]
UNIT_DURATION_SEC = 0.96
EXPECTED_SAMPLE_RATE = 16000
TARGET_LANG_NAME = {"zh": "Chinese", "ja": "Japanese", "de": "German"}


@dataclass
class StreamState:
    instance_index: int
    lm: int
    source_path: str
    reference: str
    audio: np.ndarray
    samplerate: int
    segment_samples: int
    messages: List[Dict[str, Any]] = field(default_factory=list)
    cursor_samples: int = 0
    last_vllm_samples: int = 0
    segment_idx: int = 0
    finished: bool = False
    start_perf: Optional[float] = None
    prediction_parts: List[str] = field(default_factory=list)
    delays: List[float] = field(default_factory=list)
    elapsed: List[float] = field(default_factory=list)
    runtime_records: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def audio_sec(self) -> float:
        return float(len(self.audio)) / float(self.samplerate)

    @property
    def prediction(self) -> str:
        return "".join(self.prediction_parts)


def _repo_path(path: str) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def _load_lines(path: Path, *, name: str) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty {name}: {path}")
    return lines


def _load_audio(path: str) -> Tuple[np.ndarray, int]:
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"Missing audio: {path}")
    audio, sr = sf.read(str(p), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32).flatten()
    if sr != EXPECTED_SAMPLE_RATE:
        raise ValueError(
            f"Unsupported sample rate for {path}: got {sr}, expected {EXPECTED_SAMPLE_RATE}. "
            "Resampling is intentionally not implicit."
        )
    if audio.size == 0:
        raise ValueError(f"Empty audio: {path}")
    return audio, sr


def _format_term_map(refs: Sequence[Dict[str, Any]], mode: str) -> str:
    lines: List[str] = []
    seen: set[str] = set()
    for ref in refs:
        term = str(ref.get("term") or "").replace("\n", " ").strip()
        translation = str(ref.get("translation") or "").replace("\n", " ").strip()
        if not term or not translation or term in seen:
            continue
        seen.add(term)
        if mode == "xml_tagged":
            lines.append(f"<term>{term} => {translation}</term>")
        elif mode == "tagged":
            lines.append(f"[TERM] {term} => {translation} [/TERM]")
        else:
            lines.append(f"{term}={translation}")
    return "\n".join(lines)


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]..."


def _latency_unit_count(text: str, lang_code: str) -> int:
    if not text:
        return 0
    if lang_code == "de":
        return len(text.split())
    return len(text)


def _simuleval_prediction_piece(text: str, lang_code: str) -> str:
    """Mirror SimulEval's stored prediction form for character-level targets.

    The serial agent appends the raw vLLM text to the conversational history,
    but SimulEval stores zh/ja character-level predictions without whitespace in
    ``instances.log``.  Keeping raw text in prompts while writing normalized
    instance text is necessary for serial-compatible BLEU/latency validation.
    """
    text = str(text or "")
    if lang_code in {"zh", "ja"}:
        return re.sub(r"\s+", "", text)
    return text


def _join_prediction_parts(parts: Sequence[str], lang_code: str) -> str:
    """Join generated chunks in the representation consumed by StreamLAAL.

    SimulEval latency for German is word-level.  Each chunk's delay trace is
    counted from that chunk's whitespace-tokenized output, so concatenating
    chunks without an intervening boundary can merge words across chunk edges
    and make the hypothesis shorter than its delay trace.  Character-level
    zh/ja must remain concatenated without whitespace.
    """
    if lang_code != "de":
        return "".join(parts)
    out: List[str] = []
    for part in parts:
        if not part:
            continue
        if out and not out[-1].endswith(tuple(" \t\r\n")) and not part.startswith(tuple(" \t\r\n")):
            out.append(" ")
        out.append(part)
    return "".join(out)


def _instance_row(state: StreamState, lang_code: str) -> Dict[str, Any]:
    prediction = _join_prediction_parts(state.prediction_parts, lang_code)
    expected_units = _latency_unit_count(prediction, lang_code)
    if expected_units != len(state.delays):
        raise ValueError(
            "prediction/delay unit mismatch for "
            f"lang={lang_code} lm={state.lm} instance={state.instance_index}: "
            f"prediction_units={expected_units} delays={len(state.delays)}"
        )
    return {
        "index": int(state.instance_index),
        "prediction": prediction,
        "delays": state.delays,
        "elapsed": state.elapsed,
        "prediction_length": len(state.delays),
        "reference": state.reference,
        "source": [state.source_path],
        "source_length": float(state.audio_sec * 1000.0),
    }


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _build_states(args: argparse.Namespace) -> List[StreamState]:
    source_paths = _load_lines(_repo_path(args.source_list), name="source list")
    target_lines = _load_lines(_repo_path(args.target_list), name="target list")
    if len(source_paths) != len(target_lines):
        raise ValueError(
            f"source/target length mismatch: {len(source_paths)} vs {len(target_lines)}"
        )

    states: List[StreamState] = []
    for idx, (source_path, reference) in enumerate(zip(source_paths, target_lines)):
        audio, sr = _load_audio(source_path)
        for lm in args.lms:
            segment_samples = max(1, int(round(float(lm) * UNIT_DURATION_SEC * sr)))
            states.append(
                StreamState(
                    instance_index=idx,
                    lm=int(lm),
                    source_path=source_path,
                    reference=reference,
                    audio=audio,
                    samplerate=sr,
                    segment_samples=segment_samples,
                )
            )
    return states


def _build_rag_system_prompt(
    source_lang: str, target_lang: str, prompt_policy: str
) -> str:
    if prompt_policy == "translate_task":
        system_text = (
            f"You are a professional simultaneous interpreter. "
            f"Your task is to translate {source_lang} audio chunks into accurate and fluent "
            f"{target_lang}."
        )
    elif prompt_policy == "given_chunks":
        system_text = (
            f"You are a professional simultaneous interpreter. "
            f"You will be given chunks of {source_lang} audio and you need to translate "
            f"the audio into {target_lang} text."
        )
    else:
        raise ValueError(f"Unsupported rag_prompt_policy={prompt_policy!r}")
    return system_text + " Use the 'term_map' as a reference for terminology if provided."


def _append_system_if_needed(
    state: StreamState, source_lang: str, target_lang: str, prompt_policy: str
) -> None:
    if state.messages:
        return
    system_text = _build_rag_system_prompt(source_lang, target_lang, prompt_policy)
    state.messages.append({"role": "system", "content": [{"type": "text", "text": system_text}]})


def _append_norag_system_if_needed(
    state: StreamState,
    source_lang: str,
    target_lang: str,
    prompt_policy: str,
) -> None:
    if state.messages:
        return
    if prompt_policy == "serial_compat":
        system_text = (
            f"You are a professional simultaneous interpreter. "
            f"Your task is to translate {source_lang} audio chunks into accurate and fluent "
            f"{target_lang}."
        )
    elif prompt_policy == "term_map_if_available":
        system_text = (
            f"You are a professional simultaneous interpreter. "
            f"Your task is to translate {source_lang} audio chunks into accurate and fluent "
            f"{target_lang}. Use the 'term_map' as a reference for terminology if provided."
        )
    else:
        raise ValueError(f"Unsupported norag_prompt_policy={prompt_policy!r}")
    state.messages.append({"role": "system", "content": [{"type": "text", "text": system_text}]})


def _prepare_vllm_input(
    *,
    state: StreamState,
    processor: Any,
    process_mm_info: Any,
    increment: np.ndarray,
    references: Sequence[Dict[str, Any]],
    source_lang: str,
    target_lang: str,
    term_map_format: str,
    empty_term_map_policy: str,
    disable_rag: bool,
    rag_prompt_policy: str,
    norag_prompt_policy: str,
) -> Dict[str, Any]:
    if disable_rag:
        _append_norag_system_if_needed(
            state,
            source_lang=source_lang,
            target_lang=target_lang,
            prompt_policy=norag_prompt_policy,
        )
    else:
        _append_system_if_needed(
            state,
            source_lang=source_lang,
            target_lang=target_lang,
            prompt_policy=rag_prompt_policy,
        )
    user_content: List[Dict[str, Any]] = [{"type": "audio", "audio": increment}]
    kv = _format_term_map(references, term_map_format)
    if kv:
        user_content.append({"type": "text", "text": f"\n\nterm_map:\n{kv}"})
    elif empty_term_map_policy == "none_block":
        user_content.append({"type": "text", "text": "\n\nterm_map:\nNONE"})
    elif empty_term_map_policy == "omit":
        pass
    else:
        raise ValueError(f"Unsupported empty_term_map_policy={empty_term_map_policy!r}")
    state.messages.append({"role": "user", "content": user_content})

    text = processor.apply_chat_template(
        state.messages, add_generation_prompt=True, tokenize=False
    )
    audios, _images, _videos = process_mm_info(state.messages, use_audio_in_video=False)
    return {
        "prompt": text,
        "multi_modal_data": {"audio": audios},
        "mm_processor_kwargs": {"use_audio_in_video": False},
    }


def _cache_chunks(
    *,
    state: StreamState,
    seconds: float,
    fallback_chunks: int,
    min_chunks: int,
) -> int:
    if seconds <= 0:
        return int(fallback_chunks)
    segment_sec = float(state.segment_samples) / float(state.samplerate)
    return max(int(min_chunks), int(float(seconds) / max(segment_sec, 1e-6)))


def _trim_messages(
    state: StreamState,
    *,
    max_cache_chunks: int,
    keep_cache_chunks: int,
) -> None:
    if len(state.messages) >= 2 * max_cache_chunks + 1:
        state.messages = [state.messages[0]] + state.messages[-2 * keep_cache_chunks :]


def _retrieve_references(
    *,
    retriever: Any,
    state: StreamState,
    top_k: int,
    lookback_sec: float,
) -> List[Dict[str, Any]]:
    current_start_sec = float(state.last_vllm_samples) / float(state.samplerate)
    current_end_sec = float(state.cursor_samples) / float(state.samplerate)
    retriever._audio_buffer = np.asarray(state.audio[: state.cursor_samples], dtype=np.float32)
    refs = retriever.retrieve_timeline(
        top_k=top_k,
        current_start_sec=current_start_sec,
        current_end_sec=current_end_sec,
        lookback_sec=lookback_sec,
    )
    state.runtime_records.append(
        {
            "type": "rag_window",
            "lm": int(state.lm),
            "instance_index": int(state.instance_index),
            "source_path": state.source_path,
            "trigger": "vllm_timeline",
            "segment_idx": int(state.segment_idx),
            "current_start_sec": current_start_sec,
            "current_end_sec": current_end_sec,
            "lookback_sec": float(lookback_sec),
        }
    )
    state.runtime_records.append(
        {
            "type": "rag",
            "lm": int(state.lm),
            "instance_index": int(state.instance_index),
            "source_path": state.source_path,
            "segment_idx": int(state.segment_idx),
            "rag_audio_duration": round(current_end_sec, 6),
            "rag_streaming_mode": "timeline_batched",
            "references": refs,
        }
    )
    return refs


def _retrieve_references_for_batch(
    *,
    args: argparse.Namespace,
    retriever: Any,
    batch_pairs: Sequence[Tuple[StreamState, np.ndarray]],
) -> List[List[Dict[str, Any]]]:
    """Retrieve term_map references for a scheduled vLLM batch."""
    if args.disable_rag or int(args.rag_top_k) <= 0:
        for state, _increment in batch_pairs:
            state.runtime_records.append(
                {
                    "type": "rag",
                    "lm": int(state.lm),
                    "instance_index": int(state.instance_index),
                    "source_path": state.source_path,
                    "segment_idx": int(state.segment_idx),
                    "rag_disabled": True,
                    "references": [],
                }
            )
        return [[] for _state, _increment in batch_pairs]

    if not args.rag_batch_retrieval:
        return [
            _retrieve_references(
                retriever=retriever,
                state=state,
                top_k=args.rag_top_k,
                lookback_sec=args.rag_timeline_lookback_sec,
            )
            for state, _increment in batch_pairs
        ]

    if not hasattr(retriever, "retrieve_timeline_batch"):
        raise AttributeError(
            "rag_batch_retrieval=1 requires retriever.retrieve_timeline_batch"
        )

    requests: List[Dict[str, Any]] = []
    for state, _increment in batch_pairs:
        current_start_sec = float(state.last_vllm_samples) / float(state.samplerate)
        current_end_sec = float(state.cursor_samples) / float(state.samplerate)
        requests.append(
            {
                "audio_buffer": np.asarray(state.audio[: state.cursor_samples], dtype=np.float32),
                "current_start_sec": current_start_sec,
                "current_end_sec": current_end_sec,
                "lookback_sec": float(args.rag_timeline_lookback_sec),
            }
        )

    refs_by_state = retriever.retrieve_timeline_batch(
        requests,
        top_k=args.rag_top_k,
        lookback_sec=args.rag_timeline_lookback_sec,
    )
    if len(refs_by_state) != len(batch_pairs):
        raise RuntimeError(
            f"retrieve_timeline_batch returned {len(refs_by_state)} rows for "
            f"{len(batch_pairs)} requests"
        )

    for (state, _increment), refs, req in zip(batch_pairs, refs_by_state, requests):
        state.runtime_records.append(
            {
                "type": "rag_window",
                "lm": int(state.lm),
                "instance_index": int(state.instance_index),
                "source_path": state.source_path,
                "trigger": "vllm_timeline",
                "segment_idx": int(state.segment_idx),
                "current_start_sec": float(req["current_start_sec"]),
                "current_end_sec": float(req["current_end_sec"]),
                "lookback_sec": float(args.rag_timeline_lookback_sec),
                "rag_batch_retrieval": True,
            }
        )
        state.runtime_records.append(
            {
                "type": "rag",
                "lm": int(state.lm),
                "instance_index": int(state.instance_index),
                "source_path": state.source_path,
                "segment_idx": int(state.segment_idx),
                "rag_audio_duration": round(float(req["current_end_sec"]), 6),
                "rag_streaming_mode": "timeline_batch",
                "rag_batch_retrieval": True,
                "references": refs,
            }
        )
    return refs_by_state


def _advance_ready_states(states: Sequence[StreamState]) -> List[Tuple[StreamState, np.ndarray]]:
    ready: List[Tuple[StreamState, np.ndarray]] = []
    for state in states:
        if state.finished:
            continue
        previous_cursor = state.cursor_samples
        state.cursor_samples = min(
            len(state.audio), state.cursor_samples + state.segment_samples
        )
        state.finished = state.cursor_samples >= len(state.audio)
        if state.cursor_samples <= state.last_vllm_samples:
            continue
        increment = state.audio[state.last_vllm_samples : state.cursor_samples]
        if increment.size == 0 and not state.finished:
            state.cursor_samples = previous_cursor
            continue
        ready.append((state, increment))
    return ready


def _chunked(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    if size <= 0:
        yield items
        return
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _run_offline_eval(args: argparse.Namespace, lm: int, out_dir: Path) -> None:
    if args.skip_offline_eval:
        return
    instances_log = out_dir / "instances.log"
    if not instances_log.is_file() or instances_log.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing instances.log for offline eval: {instances_log}")
    eval_tsv = out_dir / "eval_results.tsv"
    eval_log = out_dir / "eval_results.log"
    glossary = args.eval_glossary or args.glossary
    cmd = [
        sys.executable,
        str(_repo_path(args.offline_eval_script)),
        "--mode",
        args.eval_mode,
        "--instances-log",
        str(instances_log),
        "--lang-code",
        args.lang_code,
        "--source-file",
        str(_repo_path(args.source_text_file)),
        "--ref-file",
        str(_repo_path(args.ref_file)),
        "--audio-yaml",
        str(_repo_path(args.audio_yaml)),
        "--glossary-acl6060",
        str(_repo_path(glossary)),
        "--strip-output-tags",
        args.strip_output_tags,
        "--term-fcr-policy",
        args.term_fcr_policy,
        "--output-tsv",
        str(eval_tsv),
        "--output-log",
        str(eval_log),
        "--work-dir",
        str(out_dir / "offline_work"),
    ]
    print(f"[OFFLINE] lm={lm} {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def _write_outputs(
    *,
    args: argparse.Namespace,
    states: Sequence[StreamState],
) -> Dict[int, Path]:
    out_dirs: Dict[int, Path] = {}
    for lm in args.lms:
        lm_states = sorted(
            [s for s in states if s.lm == lm], key=lambda s: s.instance_index
        )
        if args.density_tag and args.glossary_tag:
            tau = str(args.rag_score_threshold)
            out_name = (
                f"d{args.density_tag}_lm{lm}_k{args.rag_top_k}"
                f"_th{tau}_g{args.glossary_tag}"
            )
        else:
            out_name = f"batchvllm_lm{lm}_{args.run_tag}"
        out_dir = Path(args.output_base) / args.lang_code / out_name
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(
            out_dir / "instances.log",
            (_instance_row(s, args.lang_code) for s in lm_states),
        )
        runtime_path = out_dir / f"runtime_omni_vllm_maxsim_rag_batched_lm{lm}.jsonl"
        runtime_rows: List[Dict[str, Any]] = []
        for state in lm_states:
            runtime_rows.extend(state.runtime_records)
            # compute_sentence_term_adoption.py splits concatenated runtime
            # records when a new segment-0 follows a previous positive segment.
            # Add a zero-duration ignored marker for one-segment instances so
            # very short rows cannot merge with the next row's runtime records.
            has_positive = any(
                rec.get("type") in {"rag_window", "rag", "llm_input", "llm_output"}
                and int(rec.get("segment_idx", -1)) > 0
                for rec in state.runtime_records
            )
            if not has_positive:
                runtime_rows.append(
                    {
                        "type": "rag_window",
                        "trigger": "vllm_timeline",
                        "segment_idx": 1,
                        "current_start_sec": 0.0,
                        "current_end_sec": 0.0,
                        "sentinel": "one_segment_instance_boundary",
                    }
                )
        _write_jsonl(runtime_path, runtime_rows)
        metadata = {
            "lang_code": args.lang_code,
            "lm": lm,
            "run_tag": args.run_tag,
            "density_tag": args.density_tag,
            "glossary_tag": args.glossary_tag,
            "source_list": str(_repo_path(args.source_list)),
            "target_list": str(_repo_path(args.target_list)),
            "glossary": str(_repo_path(args.glossary)),
            "rag_model_path": args.rag_model_path,
            "rag_index_path": args.rag_index_path,
            "rag_top_k": args.rag_top_k,
            "rag_score_threshold": args.rag_score_threshold,
            "rag_timeline_lookback_sec": args.rag_timeline_lookback_sec,
            "empty_term_map_policy": args.empty_term_map_policy,
            "rag_prompt_policy": args.rag_prompt_policy,
            "stream_count": len(lm_states),
        }
        (out_dir / "batched_vllm_rag_meta.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        out_dirs[lm] = out_dir
    return out_dirs


def _process_ready_pairs(
    *,
    args: argparse.Namespace,
    pairs: Sequence[Tuple[StreamState, np.ndarray]],
    processor: Any,
    process_mm_info: Any,
    llm: Any,
    sampling_params_by_lm: Dict[int, Any],
    fixed_sampling_params: Any,
    retriever: Any,
    source_lang: str,
    target_lang: str,
    t0: float,
) -> int:
    completed = 0
    for batch in _chunked(pairs, args.scheduler_batch_size):
        prepared: List[Dict[str, Any]] = []
        batch_pairs = list(batch)
        batch_sampling_params: List[Any] = []
        refs_by_state = _retrieve_references_for_batch(
            args=args,
            retriever=retriever,
            batch_pairs=batch_pairs,
        )
        for (state, increment), refs in zip(batch_pairs, refs_by_state):
            if state.start_perf is None:
                state.start_perf = time.perf_counter()
            inputs = _prepare_vllm_input(
                state=state,
                processor=processor,
                process_mm_info=process_mm_info,
                increment=increment,
                references=refs,
                source_lang=source_lang,
                target_lang=target_lang,
                term_map_format=args.term_map_format,
                empty_term_map_policy=args.empty_term_map_policy,
                disable_rag=bool(args.disable_rag),
                rag_prompt_policy=args.rag_prompt_policy,
                norag_prompt_policy=args.norag_prompt_policy,
            )
            state.runtime_records.append(
                {
                    "type": "llm_input",
                    "lm": int(state.lm),
                    "instance_index": int(state.instance_index),
                    "source_path": state.source_path,
                    "segment_idx": int(state.segment_idx),
                    "prompt": _truncate(inputs.get("prompt", ""), args.runtime_prompt_max_chars),
                    "references": refs,
                    "prompt_chars": len(inputs.get("prompt", "")),
                    "input_samples": int(len(increment)),
                    "cursor_samples": int(state.cursor_samples),
                    "last_vllm_samples": int(state.last_vllm_samples),
                    "max_new_tokens": int(
                        sampling_params_by_lm[int(state.lm)].max_tokens
                        if args.max_new_tokens_policy == "lm_scaled"
                        else fixed_sampling_params.max_tokens
                    ),
                }
            )
            prepared.append(inputs)
            if args.max_new_tokens_policy == "lm_scaled":
                batch_sampling_params.append(sampling_params_by_lm[int(state.lm)])
        sampling_params = (
            batch_sampling_params
            if args.max_new_tokens_policy == "lm_scaled"
            else fixed_sampling_params
        )
        outputs = llm.generate(prepared, sampling_params=sampling_params, use_tqdm=False)
        now_perf = time.perf_counter()
        for (state, _increment), out in zip(batch_pairs, outputs):
            text = out.outputs[0].text if out.outputs else ""
            instance_text = _simuleval_prediction_piece(text, args.lang_code)
            state.prediction_parts.append(instance_text)
            units = _latency_unit_count(instance_text, args.lang_code)
            current_delay_ms = round(
                float(state.cursor_samples) / float(state.samplerate) * 1000.0,
                6,
            )
            stream_start = state.start_perf if state.start_perf is not None else t0
            now_elapsed_ms = (now_perf - stream_start) * 1000.0
            state.delays.extend([current_delay_ms] * units)
            state.elapsed.extend([now_elapsed_ms] * units)
            state.runtime_records.append(
                {
                    "type": "llm_output",
                    "lm": int(state.lm),
                    "instance_index": int(state.instance_index),
                    "source_path": state.source_path,
                    "segment_idx": int(state.segment_idx),
                    "text": _truncate(text, args.runtime_prompt_max_chars),
                    "text_chars": len(text),
                    "instance_text": _truncate(instance_text, args.runtime_prompt_max_chars),
                    "instance_text_chars": len(instance_text),
                    "delay_ms": current_delay_ms,
                    "elapsed_ms": now_elapsed_ms,
                }
            )
            state.messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": text}]}
            )
            _trim_messages(
                state,
                max_cache_chunks=_cache_chunks(
                    state=state,
                    seconds=args.max_cache_seconds,
                    fallback_chunks=args.max_cache_chunks,
                    min_chunks=args.min_cache_chunks,
                ),
                keep_cache_chunks=_cache_chunks(
                    state=state,
                    seconds=args.keep_cache_seconds,
                    fallback_chunks=args.keep_cache_chunks,
                    min_chunks=args.min_cache_chunks,
                ),
            )
            state.last_vllm_samples = state.cursor_samples
            state.segment_idx += 1
            if state.finished:
                completed += 1
    return completed


def run(args: argparse.Namespace) -> int:
    states = _build_states(args)
    if args.dry_run:
        summary = {
            "stream_count": len(states),
            "lms": args.lms,
            "source_count": len({s.instance_index for s in states}),
            "total_audio_hours": sum(s.audio_sec for s in states) / 3600.0,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 0

    from qwen_omni_utils import process_mm_info
    from transformers import Qwen3OmniMoeProcessor
    from vllm import LLM, SamplingParams

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if not args.disable_rag and int(args.rag_top_k) > 0:
        from agents.streaming_maxsim_retriever import (  # noqa: WPS433
            MAXSIM_STRIDE,
            MAXSIM_WINDOWS,
            StreamingMaxSimRetriever,
        )

    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_name)
    llm_kwargs = {
        "model": args.model_name,
        "trust_remote_code": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.vllm_tp_size,
        "limit_mm_per_prompt": {"audio": args.vllm_limit_audio},
        "max_num_seqs": args.max_num_seqs,
        "max_model_len": args.max_model_len,
        "enable_prefix_caching": bool(args.enable_prefix_caching),
        "enforce_eager": bool(args.vllm_enforce_eager),
        "safetensors_load_strategy": args.safetensors_load_strategy,
    }
    if args.disable_custom_all_reduce:
        llm_kwargs["disable_custom_all_reduce"] = True
    print(f"[LOAD] vLLM model={args.model_name} kwargs={llm_kwargs}", flush=True)
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    sampling_params_by_lm = {
        int(lm): SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=(
                int(args.max_new_tokens) * int(lm)
                if args.max_new_tokens_policy == "lm_scaled"
                else int(args.max_new_tokens)
            ),
            seed=args.seed,
        )
        for lm in args.lms
    }
    print(
        "[DECODE] max_new_tokens_policy={} {}".format(
            args.max_new_tokens_policy,
            {
                int(lm): int(sampling_params_by_lm[int(lm)].max_tokens)
                for lm in args.lms
            },
        ),
        flush=True,
    )

    retriever = None
    if args.disable_rag or int(args.rag_top_k) <= 0:
        print(
            "[LOAD] retriever disabled; norag_prompt_policy={} empty_term_map_policy={}".format(
                args.norag_prompt_policy,
                args.empty_term_map_policy,
            ),
            flush=True,
        )
    else:
        if not args.rag_model_path or not args.rag_index_path:
            raise ValueError("--rag-model-path and --rag-index-path are required unless --disable-rag is set")
        retriever_device = args.rag_device
        print(
            f"[LOAD] retriever ckpt={args.rag_model_path} index={args.rag_index_path} "
            f"device={retriever_device} tau={args.rag_score_threshold} "
            f"rag_prompt_policy={args.rag_prompt_policy} "
            f"empty_term_map_policy={args.empty_term_map_policy}",
            flush=True,
        )
        retriever = StreamingMaxSimRetriever(
            model_path=args.rag_model_path,
            index_path=args.rag_index_path,
            device=retriever_device,
            top_k=args.rag_top_k,
            lora_rank=args.rag_lora_r,
            text_lora_rank=args.rag_text_lora_r,
            target_lang=args.lang_code,
            window_sec=0.0,
            score_threshold=args.rag_score_threshold,
            maxsim_windows=MAXSIM_WINDOWS,
            maxsim_stride=MAXSIM_STRIDE,
        )

    source_lang = args.source_lang
    target_lang = TARGET_LANG_NAME.get(args.lang_code, args.lang_code)
    active = len(states)
    step_idx = 0
    t0 = time.perf_counter()
    if args.schedule_mode == "serial_by_lm":
        ordered_states = sorted(states, key=lambda s: (s.lm, s.instance_index))
        for state in ordered_states:
            while not state.finished:
                ready = _advance_ready_states([state])
                if not ready:
                    break
                print(
                    f"[STEP] {step_idx} mode=serial_by_lm lm={state.lm} "
                    f"idx={state.instance_index} ready={len(ready)} active={active}",
                    flush=True,
                )
                active -= _process_ready_pairs(
                    args=args,
                    pairs=ready,
                    processor=processor,
                    process_mm_info=process_mm_info,
                    llm=llm,
                    sampling_params_by_lm=sampling_params_by_lm,
                    fixed_sampling_params=sampling_params,
                    retriever=retriever,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    t0=t0,
                )
                step_idx += 1
    else:
        while active:
            ready = _advance_ready_states(states)
            if not ready:
                break
            print(f"[STEP] {step_idx} ready={len(ready)} active={active}", flush=True)
            active -= _process_ready_pairs(
                args=args,
                pairs=ready,
                processor=processor,
                process_mm_info=process_mm_info,
                llm=llm,
                sampling_params_by_lm=sampling_params_by_lm,
                fixed_sampling_params=sampling_params,
                retriever=retriever,
                source_lang=source_lang,
                target_lang=target_lang,
                t0=t0,
            )
            step_idx += 1

    out_dirs = _write_outputs(args=args, states=states)
    for lm, out_dir in out_dirs.items():
        _run_offline_eval(args, lm, out_dir)
    print("[DONE] batched vLLM RAG eval outputs:", flush=True)
    for lm, out_dir in out_dirs.items():
        print(f"  lm={lm}: {out_dir}", flush=True)
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-list", required=True)
    p.add_argument("--target-list", required=True)
    p.add_argument("--source-text-file", required=True)
    p.add_argument("--ref-file", required=True)
    p.add_argument("--audio-yaml", required=True)
    p.add_argument("--glossary", required=True)
    p.add_argument("--eval-glossary", default="")
    p.add_argument("--output-base", required=True)
    p.add_argument("--run-tag", required=True)
    p.add_argument("--density-tag", default="")
    p.add_argument("--glossary-tag", default="")
    p.add_argument("--lang-code", choices=["zh", "ja", "de"], required=True)
    p.add_argument("--source-lang", default="English")
    p.add_argument("--lms", type=int, nargs="+", default=[1, 2, 3, 4])

    p.add_argument("--model-name", required=True)
    p.add_argument("--vllm-tp-size", type=int, default=2)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--max-num-seqs", type=int, default=8)
    p.add_argument("--scheduler-batch-size", type=int, default=8)
    p.add_argument(
        "--schedule-mode",
        choices=["round_robin", "serial_by_lm"],
        default="round_robin",
        help=(
            "round_robin batches all active streams at each segment; serial_by_lm "
            "runs one (lm, source) stream to completion before the next stream, "
            "which is useful for serial SimulEval equivalence checks."
        ),
    )
    p.add_argument("--vllm-limit-audio", type=int, default=128)
    p.add_argument("--enable-prefix-caching", type=int, default=1)
    p.add_argument("--vllm-enforce-eager", type=int, default=0)
    p.add_argument("--safetensors-load-strategy", default="lazy")
    p.add_argument("--disable-custom-all-reduce", type=int, default=0)
    p.add_argument("--max-cache-chunks", type=int, default=16)
    p.add_argument("--keep-cache-chunks", type=int, default=8)
    p.add_argument(
        "--max-cache-seconds",
        type=float,
        default=0.0,
        help="If >0, emulate serial SimulEval cache length as floor(seconds / segment_sec).",
    )
    p.add_argument(
        "--keep-cache-seconds",
        type=float,
        default=0.0,
        help="If >0, emulate serial SimulEval keep length as floor(seconds / segment_sec).",
    )
    p.add_argument("--min-cache-chunks", type=int, default=1)

    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument(
        "--max-new-tokens-policy",
        choices=["fixed", "lm_scaled"],
        default="lm_scaled",
        help=(
            "fixed uses --max-new-tokens for every vLLM request; lm_scaled uses "
            "--max-new-tokens * lm, matching older serial launchers that budget "
            "more target tokens for longer latency multipliers."
        ),
    )
    p.add_argument("--seed", type=int, default=998244353)

    p.add_argument("--disable-rag", action="store_true")
    p.add_argument("--rag-model-path", default="")
    p.add_argument("--rag-index-path", default="")
    p.add_argument("--rag-device", default="cuda:1")
    p.add_argument("--rag-top-k", type=int, default=10)
    p.add_argument("--rag-score-threshold", type=float, default=0.78)
    p.add_argument("--rag-timeline-lookback-sec", type=float, default=1.92)
    p.add_argument("--rag-lora-r", type=int, default=128)
    p.add_argument("--rag-text-lora-r", type=int, default=128)
    p.add_argument(
        "--rag-batch-retrieval",
        type=int,
        default=1,
        help="Use batched MaxSim audio encoding for all ready streams in a vLLM batch.",
    )
    p.add_argument("--term-map-format", choices=["plain", "tagged", "xml_tagged"], default="plain")
    p.add_argument(
        "--empty-term-map-policy",
        choices=["none_block", "omit"],
        default="none_block",
        help=(
            "Prompt policy when retrieval returns no usable references. "
            "none_block keeps the historical explicit 'term_map: NONE' block; "
            "omit leaves the user message audio-only, matching empty chunks in "
            "the TM-SFT training JSONL."
        ),
    )
    p.add_argument(
        "--norag-prompt-policy",
        choices=["term_map_if_available", "serial_compat"],
        default="term_map_if_available",
        help=(
            "System prompt policy when --disable-rag is set. "
            "term_map_if_available keeps the existing batch prompt. "
            "serial_compat removes the term-map instruction to match the "
            "serial InfiniSST no-RAG agent."
        ),
    )
    p.add_argument(
        "--rag-prompt-policy",
        choices=["translate_task", "given_chunks"],
        default="translate_task",
        help=(
            "System prompt policy when RAG is enabled. translate_task keeps the "
            "historical batch prompt; given_chunks matches the cap16 "
            "denoise-budget SFT JSONL prompt."
        ),
    )

    p.add_argument("--offline-eval-script", default="documents/code/offline_sst_eval/offline_streamlaal_eval.py")
    p.add_argument("--eval-mode", choices=["acl6060", "extracted_by_paper"], default="acl6060")
    p.add_argument("--strip-output-tags", choices=["none", "term", "term_t"], default="term")
    p.add_argument(
        "--term-fcr-policy",
        choices=[
            "term_map_if_available",
            "term_map_source_ref_negative_sentence",
            "source_ref_negative_sentence",
        ],
        default="term_map_source_ref_negative_sentence",
    )
    p.add_argument("--skip-offline-eval", action="store_true")
    p.add_argument("--runtime-prompt-max-chars", type=int, default=4000)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    args.lms = sorted({int(x) for x in args.lms})
    if any(x <= 0 for x in args.lms):
        raise ValueError(f"Invalid lms: {args.lms}")
    if args.scheduler_batch_size <= 0:
        args.scheduler_batch_size = args.max_num_seqs
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
