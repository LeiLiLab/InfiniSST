#!/usr/bin/env python3

"""
Offline StreamLAAL evaluation from pre-generated SimulEval-style instances.log.

Supported modes:
- acl6060: evaluate BLEU/StreamLAAL/TERM metrics using glossary_acl6060.json.
- extracted_by_paper: keep BLEU/StreamLAAL from one full-run evaluation, but compute TERM metrics
  by splitting instances per paper (talk id) and using per-paper glossaries derived from the
  extracted glossary (with source_paper field).

All user-facing strings are in English.
"""

from __future__ import annotations

import os

# ======Configuration=====
DEFAULT_ROOT_DIR = os.environ.get("INFINISST_ROOT", "/home/jiaxuanluo/InfiniSST")

DEFAULT_DATA_ROOT = "/mnt/taurus/data/siqiouyang/datasets/acl6060"
DEFAULT_AUDIO_YAML = f"{DEFAULT_DATA_ROOT}/dev.yaml"
DEFAULT_REF_FILE_TEMPLATE = f"{DEFAULT_DATA_ROOT}/dev/text/txt/ACL.6060.dev.en-xx.{{lang}}.txt"

DEFAULT_GLOSSARY_ACL6060 = (
    f"{DEFAULT_ROOT_DIR}/retriever/gigaspeech/data_pre/glossary_acl6060.json"
)
DEFAULT_EXTRACTED_GLOSSARY = (
    f"{DEFAULT_ROOT_DIR}/retriever/gigaspeech/data_pre/extracted_glossary_with_translations.json"
)
DEFAULT_EXTRACTED_GLOSSARY_MANIFEST = (
    f"{DEFAULT_ROOT_DIR}/documents/data/data_pre/extracted_glossary_by_paper_manifest.json"
)

DEFAULT_FBK_FAIRSEQ_ROOT = os.environ.get(
    "FBK_FAIRSEQ_ROOT",
    "/mnt/taurus/home/jiaxuanluo/FBK-fairseq",
)
DEFAULT_STREAM_LAAL_TOOL_REL = (
    "examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py"
)

DEFAULT_CONDA_PYTHON = ""  # If empty, use current python executable.

DEFAULT_TERM_MISMATCH_EXAMPLES = "0"

LANG_DEFAULTS = {
    "zh": {"sacrebleu_tokenizer": "zh", "latency_unit": "char", "term_lang": "zh"},
    "ja": {"sacrebleu_tokenizer": "ja-mecab", "latency_unit": "char", "term_lang": "ja"},
    "de": {"sacrebleu_tokenizer": "13a", "latency_unit": "word", "term_lang": "de"},
}

EXIT_CONFIG_ERROR = 2
EXIT_DATA_ERROR = 3
EXIT_RUNTIME_ERROR = 4
# ======Configuration=====

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import yaml


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def _err(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _normalise_glossary_for_stream_laal(glossary_path: Path) -> Tuple[Path, Optional[Path]]:
    data = json.loads(_read_text(glossary_path))
    if isinstance(data, dict):
        return glossary_path, None
    if not isinstance(data, list):
        raise ValueError(f"Unsupported glossary format for stream_laal_term.py: {glossary_path}")

    normalised: Dict[str, Any] = {}
    for idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("term") or entry.get("source") or idx)
        if key in normalised:
            key = f"{key}__{idx}"
        normalised[key] = entry

    tmp = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".streamlaal_glossary.json", delete=False
    )
    tmp_path = Path(tmp.name)
    with tmp:
        json.dump(normalised, tmp, ensure_ascii=False, indent=2)
        tmp.write("\n")
    return tmp_path, tmp_path


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)


def _basename(path_str: str) -> str:
    return os.path.basename(str(path_str).strip())


def _paper_id_from_wav_basename(wav_base: str) -> Optional[str]:
    s = str(wav_base).strip()
    if not s:
        return None
    if s.lower().endswith(".wav"):
        s = s[: -len(".wav")]
    return s.strip() or None


@dataclass(frozen=True)
class ParsedMetrics:
    bleu: str
    stream_laal: str
    stream_laal_ca: str
    term_acc: str
    term_correct: str
    term_total: str
    fcr: str = ""
    neg_false_copy: str = ""
    neg_total: str = ""


_NUMBER_RE = r"-?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)"
_METRIC_TRIPLE_RE = re.compile(rf"^\s*({_NUMBER_RE})\s+({_NUMBER_RE})\s+({_NUMBER_RE})\s*$")
_TERM_OUTPUT_TAG_RE = re.compile(r"</?\s*term\s*>", flags=re.IGNORECASE)
_TERM_OR_T_OUTPUT_TAG_RE = re.compile(r"</?\s*(?:term|t)\s*>", flags=re.IGNORECASE)
_TERM_OUTPUT_MALFORMED_PREFIX_RE = re.compile(
    r"<\s*term\b(?!\s*>)|<\s*term(?=[^\s>/])",
    flags=re.IGNORECASE,
)


def _tag_edge_space_regexes(*, include_short_t: bool) -> Tuple[re.Pattern[str], ...]:
    tag = r"(?:term|t)" if include_short_t else r"term"
    right_punct = r"[,.;:!?，。；：！？)\]\}]"
    return (
        re.compile(rf"</\s*{tag}\s*>(?P<ws>\s+)(?=-)", flags=re.IGNORECASE),
        re.compile(rf"-(?P<ws>\s+)(?=<\s*{tag}\s*>)", flags=re.IGNORECASE),
        re.compile(rf"</\s*{tag}\s*>(?P<ws>\s+)(?={right_punct})", flags=re.IGNORECASE),
        re.compile(rf"[(\[\{{](?P<ws>\s+)(?=<\s*{tag}\s*>)", flags=re.IGNORECASE),
    )

COMPUTE_TCR_SCRIPT_REL = "documents/code/offline_sst_eval/compute_tcr_from_runtime_log.py"
COMPUTE_ADOPTION_SCRIPT_REL = "documents/code/offline_sst_eval/compute_sentence_term_adoption.py"


def _strip_term_output_tags_with_mask(
    text: str,
    *,
    include_short_t: bool,
) -> Tuple[str, List[bool], int]:
    """Strip assistant-side term markers while preserving term content.

    Returns the cleaned text, a char-level keep mask for the original text, and
    the number of removed tag spans.  The mask lets us keep delay/elapsed arrays
    aligned with the cleaned hypothesis when SimulEval stores one timing value
    per character.
    """
    text = str(text or "")
    keep = [True] * len(text)
    removed = 0
    proper_tag_re = _TERM_OR_T_OUTPUT_TAG_RE if include_short_t else _TERM_OUTPUT_TAG_RE
    for regex in (proper_tag_re, _TERM_OUTPUT_MALFORMED_PREFIX_RE):
        for match in regex.finditer(text):
            removed += 1
            for idx in range(match.start(), match.end()):
                keep[idx] = False
    if removed == 0:
        return text, keep, 0
    for pattern in _tag_edge_space_regexes(include_short_t=include_short_t):
        for match in pattern.finditer(text):
            for idx in range(match.start("ws"), match.end("ws")):
                keep[idx] = False
    cleaned = "".join(ch for ch, flag in zip(text, keep) if flag)
    return cleaned, keep, removed


def _filter_sequence_by_mask(values: Any, keep: List[bool]) -> Any:
    if not isinstance(values, list) or len(values) != len(keep):
        return values
    return [value for value, flag in zip(values, keep) if flag]


def _strip_term_output_tags_from_word_timing(
    row: Dict[str, Any],
    prediction: str,
    *,
    include_short_t: bool,
) -> Tuple[str, int, int]:
    """Strip term markers while preserving word-level timing alignment.

    SimulEval German instances store one delay per whitespace token.  For that
    case, stripping must happen at the same granularity: tag-only tokens are
    dropped together with their timing entry, and tags attached to a word are
    removed while keeping that word's timing.
    """
    words = str(prediction or "").split()
    delays = row.get("delays")
    elapsed = row.get("elapsed")
    if isinstance(delays, list) and len(delays) != len(words):
        raise ValueError(
            f"word timing mismatch before strip for index={row.get('index')}: "
            f"prediction_words={len(words)} delays={len(delays)}"
        )
    if isinstance(elapsed, list) and len(elapsed) != len(words):
        raise ValueError(
            f"word timing mismatch before strip for index={row.get('index')}: "
            f"prediction_words={len(words)} elapsed={len(elapsed)}"
        )

    clean_words: List[str] = []
    clean_delays: List[Any] = []
    clean_elapsed: List[Any] = []
    removed_spans = 0
    dropped_tokens = 0
    proper_tag_re = _TERM_OR_T_OUTPUT_TAG_RE if include_short_t else _TERM_OUTPUT_TAG_RE
    for idx, word in enumerate(words):
        cleaned, proper_removed = proper_tag_re.subn("", word)
        cleaned, malformed_removed = _TERM_OUTPUT_MALFORMED_PREFIX_RE.subn("", cleaned)
        removed_spans += proper_removed + malformed_removed
        if not cleaned:
            dropped_tokens += 1
            continue
        clean_words.append(cleaned)
        if isinstance(delays, list):
            clean_delays.append(delays[idx])
        if isinstance(elapsed, list):
            clean_elapsed.append(elapsed[idx])

    cleaned_prediction = " ".join(clean_words)
    row["prediction"] = cleaned_prediction
    row["prediction_length"] = len(clean_words)
    if isinstance(delays, list):
        row["delays"] = clean_delays
    if isinstance(elapsed, list):
        row["elapsed"] = clean_elapsed
    return cleaned_prediction, removed_spans, dropped_tokens


def _validate_prediction_timing_lengths(row: Dict[str, Any], *, latency_unit: str) -> None:
    prediction = str(row.get("prediction") or "")
    if latency_unit == "word":
        expected = len(prediction.split())
    else:
        expected = len(prediction)
    for field in ("delays", "elapsed"):
        values = row.get(field)
        if isinstance(values, list) and len(values) != expected:
            raise ValueError(
                f"{latency_unit} timing mismatch after strip for index={row.get('index')}: "
                f"prediction_units={expected} {field}={len(values)}"
            )
    row["prediction_length"] = expected


def _strip_output_tags_from_instances(
    instances_path: Path,
    *,
    mode: str,
    latency_unit: str,
) -> Tuple[Path, Dict[str, int]]:
    """Create a sanitized instances.log if requested.

    The original log is never modified.  For ``mode=term`` this strips only
    legacy ``<term>`` markers.  For ``mode=term_t`` it also strips the short
    ``<t>`` markers used by denoising-budget SLM variants.
    """
    if mode == "none":
        return instances_path, {"strip_output_tags_mode": 0}
    if mode not in {"term", "term_t"}:
        raise ValueError(f"Unsupported strip output tag mode: {mode}")
    include_short_t = mode == "term_t"

    out_path = instances_path.with_name("instances.strip_term.log")
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    stats: Counter = Counter()

    with tmp_path.open("w", encoding="utf-8") as out:
        for row in _iter_jsonl(instances_path):
            stats["instances_seen"] += 1
            prediction = str(row.get("prediction") or "")
            if latency_unit == "word":
                cleaned, removed, dropped_tokens = _strip_term_output_tags_from_word_timing(
                    row,
                    prediction,
                    include_short_t=include_short_t,
                )
                if dropped_tokens:
                    stats["removed_tag_only_tokens"] += dropped_tokens
                keep: List[bool] = []
            else:
                cleaned, keep, removed = _strip_term_output_tags_with_mask(
                    prediction,
                    include_short_t=include_short_t,
                )
            if removed:
                stats["instances_with_removed_tags"] += 1
                stats["removed_tag_spans"] += removed
                stats["removed_tag_chars"] += len(prediction) - len(cleaned)
                if latency_unit != "word":
                    row["prediction"] = cleaned
                    row["prediction_length"] = len(cleaned)
                    # SimulEval zh/ja latency uses character-level timings.
                    # When lengths match exactly, drop timings for the removed
                    # marker characters so StreamLAAL consumes a consistent
                    # hypothesis.
                    row["delays"] = _filter_sequence_by_mask(row.get("delays"), keep)
                    row["elapsed"] = _filter_sequence_by_mask(row.get("elapsed"), keep)
            _validate_prediction_timing_lengths(row, latency_unit=latency_unit)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    tmp_path.replace(out_path)
    stats[f"strip_output_tags_mode_{mode}"] = 1
    stats["sanitized_instances_log"] = str(out_path)
    return out_path, dict(stats)


def _find_runtime_log(output_dir: Path) -> Optional[Path]:
    """Find the most recent runtime JSONL log in the output directory."""
    candidates = sorted(
        list(output_dir.glob("runtime_omni_vllm_maxsim_rag_*.jsonl"))
        + list(output_dir.glob("runtime_omni_vllm_rag_v4_*.jsonl")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _compute_tcr(
    python_bin: str,
    root_dir: str,
    runtime_log: Path,
    ref_file: Path,
    glossary_path: Optional[Path] = None,
    target_lang: str = "zh",
) -> Optional[Dict[str, str]]:
    """Run compute_tcr_from_runtime_log.py and parse output."""
    script = Path(root_dir) / COMPUTE_TCR_SCRIPT_REL
    if not script.is_file():
        _warn(f"TCR script not found: {script}")
        return None

    cmd = [
        python_bin or sys.executable,
        str(script),
        "--runtime-log", str(runtime_log),
        "--ref-file", str(ref_file),
        "--target-lang", target_lang,
    ]
    if glossary_path and glossary_path.is_file():
        cmd += ["--glossary-path", str(glossary_path)]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            _warn(f"TCR computation failed: {result.stderr.strip()}")
            return None
    except Exception as e:
        _warn(f"TCR computation error: {e}")
        return None

    for line in result.stdout.splitlines():
        if line.startswith("TCR\t"):
            parts = line.split("\t")
            if len(parts) >= 6:
                return {
                    "tcr": parts[1],
                    "tcr_adopted": parts[3],
                    "tcr_total": parts[5],
                }
    _warn("TCR output not parseable")
    return None


def _compute_term_adoption(
    python_bin: str,
    root_dir: str,
    instances_log: Path,
    source_file: Path,
    ref_file: Path,
    glossary_path: Path,
    target_lang: str,
    latency_unit: str,
    runtime_log: Optional[Path] = None,
    audio_yaml: Optional[Path] = None,
    sentence_term_map: Optional[Path] = None,
    output_json: Optional[Path] = None,
    fcr_policy: str = "term_map_if_available",
) -> Optional[Dict[str, str]]:
    """Run compute_sentence_term_adoption.py and parse TERM_ADOPTION output."""
    script = Path(root_dir) / COMPUTE_ADOPTION_SCRIPT_REL
    if not script.is_file():
        _warn(f"TERM_ADOPTION script not found: {script}")
        return None

    cmd = [
        python_bin or sys.executable,
        str(script),
        "--instances-log", str(instances_log),
        "--source-file", str(source_file),
        "--reference-file", str(ref_file),
        "--glossary-path", str(glossary_path),
        "--target-lang", target_lang,
        "--latency-unit", latency_unit,
    ]
    if runtime_log is not None and runtime_log.is_file() and audio_yaml is not None and audio_yaml.is_file():
        cmd += ["--runtime-log", str(runtime_log), "--audio-yaml", str(audio_yaml)]
    if sentence_term_map is not None and sentence_term_map.is_file():
        cmd += ["--sentence-term-map", str(sentence_term_map)]
    if output_json is not None:
        cmd += ["--output-json", str(output_json)]
    cmd += ["--fcr-policy", fcr_policy]

    strict_fcr = fcr_policy == "term_map_source_ref_negative_sentence"
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            msg = f"TERM_ADOPTION computation failed: {result.stderr.strip()}"
            if strict_fcr:
                raise RuntimeError(msg)
            _warn(msg)
            return None
    except Exception as e:
        if strict_fcr:
            raise
        _warn(f"TERM_ADOPTION computation error: {e}")
        return None

    parsed: Dict[str, str] = {}
    for line in result.stdout.splitlines():
        if line.startswith("TERM_ADOPTION\t"):
            parts = line.split("\t")
            if len(parts) >= 10:
                parsed.update(
                    {
                        "term_adoption": parts[1],
                        "adopted": parts[3],
                        "total": parts[5],
                        "sentences": parts[7],
                        "micro": parts[9],
                    }
                )
        if line.startswith("TERM_SENT_FCR\t"):
            parts = line.split("\t")
            if len(parts) >= 8:
                parsed.update(
                    {
                        "term_fcr": parts[1],
                        "false_copy_sentences": parts[3],
                        "no_gold_sentences": parts[5],
                        "false_copy_terms": parts[7],
                    }
                )
                if len(parts) >= 10 and parts[8] == "MODE":
                    parsed["term_fcr_mode"] = parts[9]
        if line.startswith("SOURCE_TERM_SENT_FCR\t"):
            parts = line.split("\t")
            if len(parts) >= 8:
                parsed.update(
                    {
                        "source_term_sent_fcr": parts[1],
                        "source_false_copy_sentences": parts[3],
                        "source_no_gold_sentences": parts[5],
                        "source_false_copy_terms": parts[7],
                    }
                )
        if line.startswith("REAL_TERM_ADOPT\t"):
            parts = line.split("\t")
            if len(parts) >= 10 and parts[1] != "N/A":
                parsed.update(
                    {
                        "real_term_adopt": parts[1],
                        "real_adopted": parts[3],
                        "real_total": parts[5],
                        "real_sentences": parts[7],
                        "real_micro": parts[9],
                    }
                )
    if "term_adoption" not in parsed:
        _warn("TERM_ADOPTION output not parseable")
        return None
    parsed.setdefault("term_fcr", "N/A")
    parsed.setdefault("false_copy_sentences", "N/A")
    parsed.setdefault("no_gold_sentences", "N/A")
    parsed.setdefault("false_copy_terms", "N/A")
    parsed.setdefault("term_fcr_mode", "unknown")
    parsed.setdefault("source_term_sent_fcr", "N/A")
    parsed.setdefault("source_false_copy_sentences", "N/A")
    parsed.setdefault("source_no_gold_sentences", "N/A")
    parsed.setdefault("source_false_copy_terms", "N/A")
    parsed.setdefault("real_term_adopt", "N/A")
    parsed.setdefault("real_adopted", "N/A")
    parsed.setdefault("real_total", "N/A")
    parsed.setdefault("real_sentences", "N/A")
    parsed.setdefault("real_micro", "N/A")
    return parsed


def _parse_stream_laal_output(text: str) -> ParsedMetrics:
    bleu = stream_laal = stream_laal_ca = ""
    term_acc = term_correct = term_total = ""
    fcr = neg_false_copy = neg_total = ""

    for line in text.splitlines():
        m = _METRIC_TRIPLE_RE.match(line)
        if m and not bleu:
            bleu, stream_laal, stream_laal_ca = m.group(1), m.group(2), m.group(3)

        if line.startswith("TERM_ACC"):
            parts = line.split()
            if len(parts) >= 6:
                term_acc = parts[1]
                term_correct = parts[3]
                term_total = parts[5]

        if line.startswith("TERM_FCR"):
            parts = line.split()
            if len(parts) >= 6:
                fcr = parts[1]
                neg_false_copy = parts[3]
                neg_total = parts[5]

    if not bleu or not stream_laal or not stream_laal_ca:
        raise ValueError("Failed to parse BLEU/StreamLAAL/StreamLAAL_CA from stream_laal_term.py output.")
    if not term_acc or not term_correct or not term_total:
        raise ValueError("Failed to parse TERM metrics from stream_laal_term.py output.")

    return ParsedMetrics(
        bleu=bleu,
        stream_laal=stream_laal,
        stream_laal_ca=stream_laal_ca,
        term_acc=term_acc,
        term_correct=term_correct,
        term_total=term_total,
        fcr=fcr,
        neg_false_copy=neg_false_copy,
        neg_total=neg_total,
    )


def _run_stream_laal_term(
    python_bin: str,
    tool_path: Path,
    instances_path: Path,
    ref_file: Path,
    source_reference_file: Optional[Path],
    audio_yaml: Path,
    sacrebleu_tokenizer: str,
    latency_unit: str,
    glossary_path: Path,
    term_lang: str,
    term_mismatch_examples: str,
) -> str:
    glossary_for_tool, tmp_glossary = _normalise_glossary_for_stream_laal(glossary_path)
    cmd = [
        python_bin,
        str(tool_path),
        "--simuleval-instances",
        str(instances_path),
        "--reference",
        str(ref_file),
        *(["--source-reference", str(source_reference_file)] if source_reference_file is not None else []),
        "--audio-yaml",
        str(audio_yaml),
        "--sacrebleu-tokenizer",
        str(sacrebleu_tokenizer),
        "--latency-unit",
        str(latency_unit),
        "--glossary",
        str(glossary_for_tool),
        "--term-lang",
        str(term_lang),
        "--term-mismatch-examples",
        str(term_mismatch_examples),
    ]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    finally:
        if tmp_glossary is not None:
            tmp_glossary.unlink(missing_ok=True)
    if p.returncode != 0:
        raise RuntimeError(
            "stream_laal_term.py failed.\n"
            f"returncode={p.returncode}\n"
            f"cmd={' '.join(cmd)}\n"
            "output:\n"
            f"{p.stdout}"
        )
    return p.stdout


def _load_acl6060_dev(audio_yaml: Path, ref_file: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    data = yaml.safe_load(_read_text(audio_yaml))
    refs = _read_text(ref_file).splitlines()
    if not isinstance(data, list):
        raise ValueError(f"Invalid audio yaml format (expect list): {audio_yaml}")
    if len(data) != len(refs):
        raise ValueError(
            f"dev.yaml entries {len(data)} != reference lines {len(refs)}. "
            f"audio_yaml={audio_yaml} ref_file={ref_file}"
        )
    return data, refs


def _subset_audio_and_refs_by_wav_basename(
    full_audio: List[Dict[str, Any]],
    full_refs: List[str],
    wav_basename: str,
) -> Tuple[List[Dict[str, Any]], List[str], List[int]]:
    indices: List[int] = []
    for i, x in enumerate(full_audio):
        if isinstance(x, dict) and _basename(x.get("wav", "")) == wav_basename:
            indices.append(i)
    return [full_audio[i] for i in indices], [full_refs[i] for i in indices], indices


def _group_extracted_glossary_by_source_paper(extracted: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Returns: {source_paper_basename: {term_key: term_obj}}
    source_paper_basename example: "2022.acl-long.110.pdf"
    """
    out: Dict[str, Dict[str, Any]] = {}
    for term_key, term_obj in extracted.items():
        if not isinstance(term_obj, dict):
            continue
        sp = str(term_obj.get("source_paper", "")).strip()
        if not sp:
            continue
        sp_base = _basename(sp)
        if not sp_base:
            continue
        out.setdefault(sp_base, {})[term_key] = term_obj
    return out


def _load_glossary_manifest(manifest_path: Path) -> Dict[str, Path]:
    """
    Returns: {paper_id: glossary_json_path}
    paper_id example: "2022.acl-long.110"
    """
    obj = json.loads(_read_text(manifest_path))
    papers = obj.get("papers", {})
    out: Dict[str, Path] = {}
    if not isinstance(papers, dict):
        return out
    for pid, info in papers.items():
        if not isinstance(info, dict):
            continue
        gp = str(info.get("glossary_path", "")).strip()
        if not gp:
            continue
        p = Path(gp)
        if p.is_file():
            out[str(pid)] = p
    return out


def _instances_papers(instances_path: Path) -> List[str]:
    papers: List[str] = []
    for obj in _iter_jsonl(instances_path):
        src = obj.get("source")
        if not isinstance(src, list) or not src:
            continue
        wav_base = _basename(src[0])
        pid = _paper_id_from_wav_basename(wav_base)
        if pid and pid not in papers:
            papers.append(pid)
    papers.sort()
    return papers


def _write_instances_subset_by_paper(instances_path: Path, paper_id: str, out_path: Path) -> int:
    wav_base = f"{paper_id}.wav"
    kept = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as w:
        for obj in _iter_jsonl(instances_path):
            src = obj.get("source")
            if not isinstance(src, list) or not src:
                continue
            if _basename(src[0]) != wav_base:
                continue
            w.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1
    return kept


def _compute_term_sums_extracted_by_paper(
    python_bin: str,
    tool_path: Path,
    instances_path: Path,
    full_audio_yaml: Path,
    full_source_file: Path,
    full_ref_file: Path,
    sacrebleu_tokenizer: str,
    latency_unit: str,
    extracted_glossary_manifest: Path,
    term_lang: str,
    term_mismatch_examples: str,
    work_dir: Path,
) -> Tuple[int, int, Dict[str, Tuple[int, int]]]:
    paper_to_gloss = _load_glossary_manifest(extracted_glossary_manifest)
    if not paper_to_gloss:
        raise ValueError(f"No valid papers found in manifest: {extracted_glossary_manifest}")

    full_audio, full_refs = _load_acl6060_dev(full_audio_yaml, full_ref_file)
    full_sources = _read_text(full_source_file).splitlines()
    if len(full_sources) != len(full_audio):
        raise ValueError(
            f"source lines {len(full_sources)} != audio entries {len(full_audio)}. "
            f"source_file={full_source_file}"
        )

    total_correct = 0
    total_terms = 0
    per_paper_counts: Dict[str, Tuple[int, int]] = {}

    papers = _instances_papers(instances_path)
    if not papers:
        raise ValueError(f"No papers found from instances source[0]: {instances_path}")

    tmp_dir = work_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for pid in papers:
        gloss_path_in_manifest = paper_to_gloss.get(pid)
        if not gloss_path_in_manifest:
            _warn(f"Skip paper_id={pid}: glossary_path not found in manifest.")
            continue

        paper_wav = f"{pid}.wav"
        sub_audio, sub_refs, indices = _subset_audio_and_refs_by_wav_basename(full_audio, full_refs, paper_wav)
        sub_sources = [full_sources[i] for i in indices]
        if not sub_audio or not sub_refs:
            _warn(f"Skip paper_id={pid}: no matching examples in dev.yaml/reference for wav={paper_wav}.")
            continue

        gloss_path = gloss_path_in_manifest
        inst_path = tmp_dir / f"instances__{pid}.log"
        audio_path = tmp_dir / f"audio__{pid}.yaml"
        ref_path = tmp_dir / f"ref__{pid}.txt"
        source_path = tmp_dir / f"source__{pid}.txt"

        kept = _write_instances_subset_by_paper(instances_path, pid, inst_path)
        if kept <= 0:
            _warn(f"Skip paper_id={pid}: no instances selected.")
            continue

        _write_text(audio_path, yaml.safe_dump(sub_audio, allow_unicode=True))
        _write_text(ref_path, "\n".join(sub_refs) + "\n")
        _write_text(source_path, "\n".join(sub_sources) + "\n")

        out = _run_stream_laal_term(
            python_bin=python_bin,
            tool_path=tool_path,
            instances_path=inst_path,
            ref_file=ref_path,
            source_reference_file=source_path,
            audio_yaml=audio_path,
            sacrebleu_tokenizer=sacrebleu_tokenizer,
            latency_unit=latency_unit,
            glossary_path=gloss_path,
            term_lang=term_lang,
            term_mismatch_examples=term_mismatch_examples,
        )
        m = _parse_stream_laal_output(out)
        c = int(float(m.term_correct))
        t = int(float(m.term_total))
        total_correct += c
        total_terms += t
        per_paper_counts[pid] = (c, t)

    return total_correct, total_terms, per_paper_counts


def _compute_adoption_sums_extracted_by_paper(
    python_bin: str,
    instances_path: Path,
    full_audio_yaml: Path,
    full_source_file: Path,
    full_ref_file: Path,
    extracted_glossary_manifest: Path,
    term_lang: str,
    latency_unit: str,
    work_dir: Path,
    runtime_log: Optional[Path] = None,
    fcr_policy: str = "term_map_if_available",
) -> Tuple[
    float,
    int,
    int,
    int,
    float,
    int,
    int,
    int,
    int,
    int,
    int,
    str,
    int,
    int,
    int,
    Dict[str, Dict[str, str]],
]:
    paper_to_gloss = _load_glossary_manifest(extracted_glossary_manifest)
    full_audio, full_refs = _load_acl6060_dev(full_audio_yaml, full_ref_file)
    full_sources = _read_text(full_source_file).splitlines()
    if len(full_sources) != len(full_audio):
        raise ValueError(
            f"source lines {len(full_sources)} != audio entries {len(full_audio)}. "
            f"source_file={full_source_file}"
        )

    adopted_sum = 0
    term_sum = 0
    sentence_sum = 0.0
    sentence_count = 0
    real_adopted_sum = 0
    real_term_sum = 0
    real_sentence_sum = 0.0
    real_sentence_count = 0
    false_copy_sentence_sum = 0
    no_gold_sentence_sum = 0
    false_copy_term_sum = 0
    source_false_copy_sentence_sum = 0
    source_no_gold_sentence_sum = 0
    source_false_copy_term_sum = 0
    fcr_modes = set()
    per_paper: Dict[str, Dict[str, str]] = {}

    papers = _instances_papers(instances_path)
    runtime_for_paper = runtime_log if runtime_log is not None and len(papers) == 1 else None
    tmp_dir = work_dir / "tmp_adoption"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for pid in papers:
        gloss_path = paper_to_gloss.get(pid)
        if not gloss_path:
            continue
        paper_wav = f"{pid}.wav"
        indices: List[int] = []
        for i, x in enumerate(full_audio):
            if isinstance(x, dict) and _basename(x.get("wav", "")) == paper_wav:
                indices.append(i)
        if not indices:
            continue

        inst_path = tmp_dir / f"instances__{pid}.log"
        audio_path = tmp_dir / f"audio__{pid}.yaml"
        src_path = tmp_dir / f"source__{pid}.txt"
        ref_path = tmp_dir / f"ref__{pid}.txt"
        json_path = tmp_dir / f"term_adoption__{pid}.json"

        kept = _write_instances_subset_by_paper(instances_path, pid, inst_path)
        if kept <= 0:
            continue
        _write_text(audio_path, yaml.safe_dump([full_audio[i] for i in indices], allow_unicode=True))
        _write_text(src_path, "\n".join(full_sources[i] for i in indices) + "\n")
        _write_text(ref_path, "\n".join(full_refs[i] for i in indices) + "\n")

        result = _compute_term_adoption(
            python_bin=python_bin,
            root_dir=DEFAULT_ROOT_DIR,
            instances_log=inst_path,
            source_file=src_path,
            ref_file=ref_path,
            glossary_path=gloss_path,
            target_lang=term_lang,
            latency_unit=latency_unit,
            runtime_log=runtime_for_paper,
            audio_yaml=audio_path if runtime_for_paper is not None else None,
            output_json=json_path,
            fcr_policy=fcr_policy,
        )
        if not result:
            continue
        adopted = int(float(result["adopted"]))
        total = int(float(result["total"]))
        sentences = int(float(result["sentences"]))
        macro = float(result["term_adoption"])
        adopted_sum += adopted
        term_sum += total
        sentence_sum += macro * sentences
        sentence_count += sentences
        if result.get("real_term_adopt") != "N/A":
            real_adopted = int(float(result["real_adopted"]))
            real_total = int(float(result["real_total"]))
            real_sentences = int(float(result["real_sentences"]))
            real_macro = float(result["real_term_adopt"])
            real_adopted_sum += real_adopted
            real_term_sum += real_total
            real_sentence_sum += real_macro * real_sentences
            real_sentence_count += real_sentences
        false_copy_sentence_sum += int(float(result.get("false_copy_sentences", "0") or 0))
        no_gold_sentence_sum += int(float(result.get("no_gold_sentences", "0") or 0))
        false_copy_term_sum += int(float(result.get("false_copy_terms", "0") or 0))
        source_false_copy_sentence_sum += int(
            float(result.get("source_false_copy_sentences", "0") or 0)
        )
        source_no_gold_sentence_sum += int(
            float(result.get("source_no_gold_sentences", "0") or 0)
        )
        source_false_copy_term_sum += int(
            float(result.get("source_false_copy_terms", "0") or 0)
        )
        fcr_modes.add(str(result.get("term_fcr_mode", "unknown")))
        per_paper[pid] = result

    return (
        sentence_sum,
        sentence_count,
        adopted_sum,
        term_sum,
        real_sentence_sum,
        real_sentence_count,
        real_adopted_sum,
        real_term_sum,
        false_copy_sentence_sum,
        no_gold_sentence_sum,
        false_copy_term_sum,
        ",".join(sorted(fcr_modes)) if fcr_modes else "unknown",
        source_false_copy_sentence_sum,
        source_no_gold_sentence_sum,
        source_false_copy_term_sum,
        per_paper,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["acl6060", "extracted_by_paper"], required=True)
    ap.add_argument("--instances-log", required=True)
    ap.add_argument("--lang-code", choices=sorted(LANG_DEFAULTS.keys()), required=True)

    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--source-file", default="")
    ap.add_argument("--ref-file", default="")
    ap.add_argument("--audio-yaml", default=DEFAULT_AUDIO_YAML)
    ap.add_argument(
        "--sentence-term-map",
        default="",
        help="Optional sentence-indexed oracle/retriever term_map JSON for REAL_TERM_ADOPT and term-map-gated FCR.",
    )

    ap.add_argument("--glossary-acl6060", default=DEFAULT_GLOSSARY_ACL6060)
    ap.add_argument("--extracted-glossary", default=DEFAULT_EXTRACTED_GLOSSARY)
    ap.add_argument("--extracted-glossary-manifest", default=DEFAULT_EXTRACTED_GLOSSARY_MANIFEST)

    ap.add_argument("--fbk-fairseq-root", default=DEFAULT_FBK_FAIRSEQ_ROOT)
    ap.add_argument("--stream-laal-tool-rel", default=DEFAULT_STREAM_LAAL_TOOL_REL)
    ap.add_argument("--python-bin", default=DEFAULT_CONDA_PYTHON)

    ap.add_argument("--sacrebleu-tokenizer", default="")
    ap.add_argument("--latency-unit", default="")
    ap.add_argument("--term-lang", default="")
    ap.add_argument("--term-mismatch-examples", default=DEFAULT_TERM_MISMATCH_EXAMPLES)
    ap.add_argument(
        "--strip-output-tags",
        choices=["none", "term", "term_t"],
        default="none",
        help=(
            "Sanitize hypothesis-side markup before scoring. "
            "'term' removes <term>...</term>; 'term_t' also removes short <t>...</t> markers, "
            "while preserving the inner translation."
        ),
    )
    ap.add_argument(
        "--term-fcr-policy",
        choices=[
            "term_map_if_available",
            "term_map_source_ref_negative_sentence",
            "source_ref_negative_sentence",
        ],
        default="term_map_if_available",
        help=(
            "TERM_FCR policy. Use term_map_source_ref_negative_sentence when "
            "false-copy candidates must be gated by the aligned runtime term_map."
        ),
    )

    ap.add_argument("--output-tsv", default="")
    ap.add_argument("--output-log", default="")
    ap.add_argument("--work-dir", default="")

    args = ap.parse_args()

    lang_defaults = LANG_DEFAULTS.get(args.lang_code)
    if not lang_defaults:
        _err(f"Unsupported lang-code: {args.lang_code}")
        return EXIT_CONFIG_ERROR

    sacrebleu_tokenizer = args.sacrebleu_tokenizer or lang_defaults["sacrebleu_tokenizer"]
    latency_unit = args.latency_unit or lang_defaults["latency_unit"]
    term_lang = args.term_lang or lang_defaults["term_lang"]

    raw_instances_path = Path(args.instances_log)
    if not raw_instances_path.is_file() or raw_instances_path.stat().st_size <= 0:
        _err(f"Missing/empty instances log: {raw_instances_path}")
        return EXIT_DATA_ERROR
    instances_path, strip_stats = _strip_output_tags_from_instances(
        raw_instances_path,
        mode=args.strip_output_tags,
        latency_unit=latency_unit,
    )

    data_root = Path(args.data_root)
    audio_yaml = Path(args.audio_yaml)
    if not audio_yaml.is_file():
        _err(f"Missing audio yaml: {audio_yaml}")
        return EXIT_DATA_ERROR

    ref_file = Path(args.ref_file) if args.ref_file else Path(
        DEFAULT_REF_FILE_TEMPLATE.format(lang=args.lang_code)
    )
    if not ref_file.is_file():
        _err(f"Missing reference file: {ref_file}")
        return EXIT_DATA_ERROR

    source_file = Path(args.source_file) if args.source_file else (
        data_root / "dev/text/txt/ACL.6060.dev.en-xx.en.txt"
    )
    if not source_file.is_file():
        _err(f"Missing source text file: {source_file}")
        return EXIT_DATA_ERROR

    glossary_acl6060 = Path(args.glossary_acl6060)
    if not glossary_acl6060.is_file():
        _err(f"Missing glossary_acl6060: {glossary_acl6060}")
        return EXIT_DATA_ERROR

    extracted_glossary_manifest = Path(args.extracted_glossary_manifest)
    if args.mode == "extracted_by_paper" and not extracted_glossary_manifest.is_file():
        _err(f"Missing extracted glossary manifest: {extracted_glossary_manifest}")
        return EXIT_DATA_ERROR

    tool_path = Path(args.fbk_fairseq_root) / args.stream_laal_tool_rel
    if not tool_path.is_file():
        _err(f"stream_laal_term.py not found: {tool_path}")
        return EXIT_CONFIG_ERROR

    python_bin = args.python_bin.strip() or sys.executable

    out_tsv = Path(args.output_tsv) if args.output_tsv else None
    out_log = Path(args.output_log) if args.output_log else None

    # Work dir
    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        tmp_ctx = None
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="offline_streamlaal_")
        work_dir = Path(tmp_ctx.name)

    try:
        _info(f"Mode: {args.mode}")
        _info(f"Lang code: {args.lang_code}")
        _info(f"Instances: {instances_path}")
        if args.strip_output_tags != "none":
            _info(f"Raw instances: {raw_instances_path}")
            _info(f"Strip output tags stats: {strip_stats}")
        _info(f"Reference: {ref_file}")
        _info(f"Audio yaml: {audio_yaml}")
        _info(f"Tokenizer: {sacrebleu_tokenizer}")
        _info(f"Latency unit: {latency_unit}")
        _info(f"Term lang: {term_lang}")
        runtime_log = _find_runtime_log(instances_path.parent)
        if runtime_log is not None:
            _info(f"Runtime log: {runtime_log}")
        else:
            _warn(f"No runtime JSONL found in {instances_path.parent}; REAL_TERM_ADOPT will be N/A.")

        if args.mode == "acl6060":
            eval_out = _run_stream_laal_term(
                python_bin=python_bin,
                tool_path=tool_path,
                instances_path=instances_path,
                ref_file=ref_file,
                source_reference_file=source_file,
                audio_yaml=audio_yaml,
                sacrebleu_tokenizer=sacrebleu_tokenizer,
                latency_unit=latency_unit,
                glossary_path=glossary_acl6060,
                term_lang=term_lang,
                term_mismatch_examples=str(args.term_mismatch_examples),
            )
            m = _parse_stream_laal_output(eval_out)

            if out_log:
                _write_text(out_log, eval_out)

            adoption = {
                "term_adoption": "N/A",
                "adopted": "N/A",
                "total": "N/A",
                "sentences": "N/A",
                "micro": "N/A",
                "term_fcr": "N/A",
                "false_copy_sentences": "N/A",
                "no_gold_sentences": "N/A",
                "false_copy_terms": "N/A",
                "term_fcr_mode": "unknown",
                "source_term_sent_fcr": "N/A",
                "source_false_copy_sentences": "N/A",
                "source_no_gold_sentences": "N/A",
                "source_false_copy_terms": "N/A",
                "real_term_adopt": "N/A",
                "real_adopted": "N/A",
                "real_total": "N/A",
                "real_sentences": "N/A",
                "real_micro": "N/A",
            }
            adoption_result = _compute_term_adoption(
                python_bin=python_bin,
                root_dir=DEFAULT_ROOT_DIR,
                instances_log=instances_path,
                source_file=source_file,
                ref_file=ref_file,
                glossary_path=glossary_acl6060,
                target_lang=term_lang,
                latency_unit=latency_unit,
                runtime_log=runtime_log,
                audio_yaml=audio_yaml,
                sentence_term_map=Path(args.sentence_term_map) if args.sentence_term_map else None,
                output_json=instances_path.parent / "term_adoption.json",
                fcr_policy=args.term_fcr_policy,
            )
            if adoption_result:
                adoption = adoption_result

            header = [
                "mode",
                "lang_code",
                "BLEU",
                "StreamLAAL",
                "StreamLAAL_CA",
                "TERM_ACC",
                "TERM_CORRECT",
                "TERM_TOTAL",
                "TERM_ADOPTION",
                "TERM_ADOPTED",
                "TERM_ADOPTION_TOTAL",
                "TERM_ADOPTION_SENTENCES",
                "TERM_ADOPTION_MICRO",
                "REAL_TERM_ADOPT",
                "REAL_TERM_ADOPTED",
                "REAL_TERM_ADOPT_TOTAL",
                "REAL_TERM_ADOPT_SENTENCES",
                "REAL_TERM_ADOPT_MICRO",
                "TERM_FCR",
                "FALSE_COPY",
                "NEG_TOTAL",
                "FALSE_COPY_TERMS",
                "instances_log",
                "TERM_FCR_MODE",
                "SOURCE_TERM_SENT_FCR",
                "SOURCE_FALSE_COPY",
                "SOURCE_NEG_TOTAL",
                "SOURCE_FALSE_COPY_TERMS",
            ]
            row = [
                args.mode,
                args.lang_code,
                m.bleu,
                m.stream_laal,
                m.stream_laal_ca,
                m.term_acc,
                m.term_correct,
                m.term_total,
                adoption["term_adoption"],
                adoption["adopted"],
                adoption["total"],
                adoption["sentences"],
                adoption["micro"],
                adoption["real_term_adopt"],
                adoption["real_adopted"],
                adoption["real_total"],
                adoption["real_sentences"],
                adoption["real_micro"],
                adoption["term_fcr"],
                adoption["false_copy_sentences"],
                adoption["no_gold_sentences"],
                adoption["false_copy_terms"],
                str(instances_path),
                adoption["term_fcr_mode"],
                adoption["source_term_sent_fcr"],
                adoption["source_false_copy_sentences"],
                adoption["source_no_gold_sentences"],
                adoption["source_false_copy_terms"],
            ]
            if out_tsv:
                _write_text(out_tsv, "\t".join(header) + "\n" + "\t".join(row) + "\n")
                _info(f"Wrote TSV: {out_tsv}")
            else:
                print("\t".join(header))
                print("\t".join(row))
            return 0

        # extracted_by_paper
        # 1) BLEU/StreamLAAL from a single full-run evaluation (glossary-independent).
        eval_out_full = _run_stream_laal_term(
            python_bin=python_bin,
            tool_path=tool_path,
            instances_path=instances_path,
            ref_file=ref_file,
            source_reference_file=source_file,
            audio_yaml=audio_yaml,
            sacrebleu_tokenizer=sacrebleu_tokenizer,
            latency_unit=latency_unit,
            glossary_path=glossary_acl6060,
            term_lang=term_lang,
            term_mismatch_examples=str(args.term_mismatch_examples),
        )
        m_full = _parse_stream_laal_output(eval_out_full)

        # 2) TERM metrics by paper using extracted glossary.
        total_correct, total_terms, per_paper_counts = _compute_term_sums_extracted_by_paper(
            python_bin=python_bin,
            tool_path=tool_path,
            instances_path=instances_path,
            full_audio_yaml=audio_yaml,
            full_source_file=source_file,
            full_ref_file=ref_file,
            sacrebleu_tokenizer=sacrebleu_tokenizer,
            latency_unit=latency_unit,
            extracted_glossary_manifest=extracted_glossary_manifest,
            term_lang=term_lang,
            term_mismatch_examples=str(args.term_mismatch_examples),
            work_dir=work_dir,
        )
        assert total_terms > 0, f"Invalid total_terms={total_terms} for extracted_by_paper aggregation."

        term_acc = f"{(float(total_correct) / float(total_terms)):.6f}"
        (
            adoption_sentence_sum,
            adoption_sentence_count,
            adoption_adopted,
            adoption_total,
            real_adoption_sentence_sum,
            real_adoption_sentence_count,
            real_adoption_adopted,
            real_adoption_total,
            false_copy_sentences,
            no_gold_sentences,
            false_copy_terms,
            term_fcr_mode,
            source_false_copy_sentences,
            source_no_gold_sentences,
            source_false_copy_terms,
            per_paper_adoption,
        ) = (
            _compute_adoption_sums_extracted_by_paper(
                python_bin=python_bin,
                instances_path=instances_path,
                full_audio_yaml=audio_yaml,
                full_source_file=source_file,
                full_ref_file=ref_file,
                extracted_glossary_manifest=extracted_glossary_manifest,
                term_lang=term_lang,
                latency_unit=latency_unit,
                work_dir=work_dir,
                runtime_log=runtime_log,
                fcr_policy=args.term_fcr_policy,
            )
        )
        adoption_macro = (
            adoption_sentence_sum / float(adoption_sentence_count)
            if adoption_sentence_count > 0 else 0.0
        )
        adoption_micro = (
            adoption_adopted / float(adoption_total)
            if adoption_total > 0 else 0.0
        )
        real_adoption_macro = (
            real_adoption_sentence_sum / float(real_adoption_sentence_count)
            if real_adoption_sentence_count > 0 else 0.0
        )
        real_adoption_micro = (
            real_adoption_adopted / float(real_adoption_total)
            if real_adoption_total > 0 else 0.0
        )
        term_fcr = false_copy_sentences / float(no_gold_sentences) if no_gold_sentences > 0 else 0.0
        source_term_fcr = (
            source_false_copy_sentences / float(source_no_gold_sentences)
            if source_no_gold_sentences > 0 else 0.0
        )

        if out_log:
            details = {
                "mode": args.mode,
                "lang_code": args.lang_code,
                "instances_log": str(instances_path),
                "extracted_glossary_manifest": str(extracted_glossary_manifest),
                "bleu": m_full.bleu,
                "stream_laal": m_full.stream_laal,
                "stream_laal_ca": m_full.stream_laal_ca,
                "term_correct_sum": total_correct,
                "term_total_sum": total_terms,
                "term_acc": term_acc,
                "term_adoption": f"{adoption_macro:.6f}",
                "term_adoption_adopted": str(adoption_adopted),
                "term_adoption_total": str(adoption_total),
                "term_adoption_sentences": str(adoption_sentence_count),
                "term_adoption_micro": f"{adoption_micro:.6f}",
                "real_term_adopt": f"{real_adoption_macro:.6f}" if real_adoption_sentence_count > 0 else "N/A",
                "real_term_adopted": str(real_adoption_adopted) if real_adoption_sentence_count > 0 else "N/A",
                "real_term_adopt_total": str(real_adoption_total) if real_adoption_sentence_count > 0 else "N/A",
                "real_term_adopt_sentences": str(real_adoption_sentence_count) if real_adoption_sentence_count > 0 else "N/A",
                "real_term_adopt_micro": f"{real_adoption_micro:.6f}" if real_adoption_sentence_count > 0 else "N/A",
                "term_fcr": f"{term_fcr:.6f}",
                "false_copy_sentences": str(false_copy_sentences),
                "no_gold_sentences": str(no_gold_sentences),
                "false_copy_terms": str(false_copy_terms),
                "term_fcr_mode": term_fcr_mode,
                "source_term_sent_fcr": f"{source_term_fcr:.6f}",
                "source_false_copy_sentences": str(source_false_copy_sentences),
                "source_no_gold_sentences": str(source_no_gold_sentences),
                "source_false_copy_terms": str(source_false_copy_terms),
                "per_paper_counts": {k: {"correct": v[0], "total": v[1]} for k, v in per_paper_counts.items()},
                "per_paper_adoption": per_paper_adoption,
                "full_run_stream_laal_term_output": eval_out_full,
            }
            _write_text(out_log, json.dumps(details, ensure_ascii=False, indent=2) + "\n")

        header = [
            "mode",
            "lang_code",
            "BLEU",
            "StreamLAAL",
            "StreamLAAL_CA",
            "TERM_ACC",
            "TERM_CORRECT",
            "TERM_TOTAL",
            "TERM_ADOPTION",
            "TERM_ADOPTED",
            "TERM_ADOPTION_TOTAL",
            "TERM_ADOPTION_SENTENCES",
            "TERM_ADOPTION_MICRO",
            "REAL_TERM_ADOPT",
            "REAL_TERM_ADOPTED",
            "REAL_TERM_ADOPT_TOTAL",
            "REAL_TERM_ADOPT_SENTENCES",
            "REAL_TERM_ADOPT_MICRO",
            "TERM_FCR",
            "FALSE_COPY",
            "NEG_TOTAL",
            "FALSE_COPY_TERMS",
            "instances_log",
            "TERM_FCR_MODE",
            "SOURCE_TERM_SENT_FCR",
            "SOURCE_FALSE_COPY",
            "SOURCE_NEG_TOTAL",
            "SOURCE_FALSE_COPY_TERMS",
        ]
        row = [
            args.mode,
            args.lang_code,
            m_full.bleu,
            m_full.stream_laal,
            m_full.stream_laal_ca,
            term_acc,
            str(total_correct),
            str(total_terms),
            f"{adoption_macro:.6f}",
            str(adoption_adopted),
            str(adoption_total),
            str(adoption_sentence_count),
            f"{adoption_micro:.6f}",
            f"{real_adoption_macro:.6f}" if real_adoption_sentence_count > 0 else "N/A",
            str(real_adoption_adopted) if real_adoption_sentence_count > 0 else "N/A",
            str(real_adoption_total) if real_adoption_sentence_count > 0 else "N/A",
            str(real_adoption_sentence_count) if real_adoption_sentence_count > 0 else "N/A",
            f"{real_adoption_micro:.6f}" if real_adoption_sentence_count > 0 else "N/A",
            f"{term_fcr:.6f}",
            str(false_copy_sentences),
            str(no_gold_sentences),
            str(false_copy_terms),
            str(instances_path),
            term_fcr_mode,
            f"{source_term_fcr:.6f}",
            str(source_false_copy_sentences),
            str(source_no_gold_sentences),
            str(source_false_copy_terms),
        ]
        if out_tsv:
            _write_text(out_tsv, "\t".join(header) + "\n" + "\t".join(row) + "\n")
            _info(f"Wrote TSV: {out_tsv}")
        else:
            print("\t".join(header))
            print("\t".join(row))
        return 0
    except subprocess.CalledProcessError as e:
        _err(str(e))
        return EXIT_RUNTIME_ERROR
    except Exception as e:
        _err(str(e))
        return EXIT_RUNTIME_ERROR
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
