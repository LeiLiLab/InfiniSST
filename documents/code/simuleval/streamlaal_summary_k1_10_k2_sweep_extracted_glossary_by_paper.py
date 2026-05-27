#!/usr/bin/env python3

"""
Re-evaluate TERM metrics for the "paper extracted glossary" runs by using a per-talk/per-paper glossary.

Goal
- The extracted glossary JSON contains `source_paper` per term (e.g. "2022.acl-long.110.pdf").
- The SimulEval instances contain the wav path (e.g. ".../2022.acl-long.110.wav") under `source[0]`.
- Instead of evaluating TERM_ACC using the *merged* glossary, we:
  - split instances by paper_id (talk id),
  - evaluate each split with its corresponding per-paper glossary,
  - aggregate TERM_CORRECT/TOTAL across splits,
  - keep BLEU/StreamLAAL metrics from a single full-run stream_laal_term evaluation (glossary-independent).

All user-facing strings are in English.
"""

from __future__ import annotations

# ======Configuration=====
DEFAULT_OUTPUT_BASE = (
    "/mnt/gemini/data2/jiaxuanluo/"
    "infinisst_omni_vllm_rag_rank32_iter_0000452_hf_zh_k1_10_k2_sweep_glossary2"
)
DEFAULT_LANG_CODE = "zh"

DEFAULT_DATA_ROOT = "/mnt/taurus/data/siqiouyang/datasets/acl6060"
DEFAULT_REF_FILE = f"{DEFAULT_DATA_ROOT}/dev/text/txt/ACL.6060.dev.en-xx.zh.txt"
DEFAULT_AUDIO_YAML = f"{DEFAULT_DATA_ROOT}/dev.yaml"

DEFAULT_EXTRACTED_GLOSSARY_PATH = (
    "/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/"
    "extracted_glossary_with_translations.json"
)

DEFAULT_FBK_FAIRSEQ_ROOT = "/mnt/taurus/home/jiaxuanluo/FBK-fairseq"
DEFAULT_STREAM_LAAL_TOOL_REL = (
    "examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py"
)

DEFAULT_SACREBLEU_TOKENIZER = "zh"
DEFAULT_LATENCY_UNIT = "char"
DEFAULT_TERM_LANG = "zh"
DEFAULT_TERM_MISMATCH_EXAMPLES = "0"

DEFAULT_INSTANCES_FILE = "instances.log"
DEFAULT_POST_EVAL_LOG_NAME = "post_eval_by_paper.log"

DEFAULT_SUMMARY_TSV_NAME = "k1_10_k2_sweep_extracted_glossary_by_paper_streamlaal_summary.tsv"

DEFAULT_SKIP_DIR_GLOB_SUBSTR = "_partial_backup_"
DEFAULT_MAX_OUTPUT_DIRS = 0  # 0 means no limit

DEFAULT_PAPER_GLOSSARIES_DIRNAME = "__paper_glossaries__"
DEFAULT_TMP_DIRNAME = "__paper_eval_tmp__"
DEFAULT_MAX_SOURCE_PREVIEW_CHARS = 200

EXIT_CONFIG_ERROR = 2
EXIT_DATA_ERROR = 3
# ======Configuration=====

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


DIR_RE = re.compile(
    r"_g(?P<g>.+?)_cs(?P<cs>[0-9.]+)_hs(?P<hs>[0-9.]+)_lm(?P<lm>[0-9]+)_k2(?P<k2>[0-9]+)_k1(?P<k1>[0-9]+)$"
)


def _err(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _paper_id_from_source_paper(source_paper: str) -> Optional[str]:
    s = str(source_paper).strip()
    if not s:
        return None
    base = os.path.basename(s)
    if base.lower().endswith(".pdf"):
        base = base[: -len(".pdf")]
    return base.strip() or None


def _paper_id_from_wav_path(wav_path: str) -> Optional[str]:
    s = str(wav_path).strip()
    if not s:
        return None
    base = os.path.basename(s)
    if base.lower().endswith(".wav"):
        base = base[: -len(".wav")]
    return base.strip() or None


def _extract_wav_path_from_instance(inst: Dict[str, Any]) -> Optional[str]:
    # Observed format: "source": ["/path/to.wav", "samplerate: ...", ...]
    src = inst.get("source")
    if isinstance(src, list) and src:
        first = src[0]
        if isinstance(first, str) and first.strip():
            return first.strip()
    if isinstance(src, str) and src.strip():
        return src.strip()
    return None


@dataclass(frozen=True)
class DirMeta:
    glossary_tag: str
    vllm_segment_sec: str
    hop_size: str
    latency_multiplier: int
    k2: int
    k1: int


def _as_int(x: str) -> Optional[int]:
    try:
        return int(str(x).strip())
    except Exception:
        return None


def parse_meta_from_dir_name(dir_name: str) -> Optional[DirMeta]:
    m = DIR_RE.search(dir_name)
    if not m:
        return None
    gd = m.groupdict()
    lm = _as_int(gd["lm"])
    k2 = _as_int(gd["k2"])
    k1 = _as_int(gd["k1"])
    if lm is None or k2 is None or k1 is None:
        return None
    return DirMeta(
        glossary_tag=gd["g"],
        vllm_segment_sec=gd["cs"],
        hop_size=gd["hs"],
        latency_multiplier=lm,
        k2=k2,
        k1=k1,
    )


@dataclass(frozen=True)
class StreamLAALOut:
    stdout: str
    rc: int


@dataclass(frozen=True)
class Metrics:
    bleu: Optional[float]
    stream_laal: Optional[float]
    stream_laal_ca: Optional[float]
    term_correct: Optional[int]
    term_total: Optional[int]


def _run_stream_laal_term(
    python_exe: str,
    tool_path: Path,
    instances_path: Path,
    ref_file: Path,
    audio_yaml: Path,
    sacrebleu_tokenizer: str,
    latency_unit: str,
    glossary_path: Path,
    term_lang: str,
    term_mismatch_examples: str,
) -> StreamLAALOut:
    cmd = [
        python_exe,
        str(tool_path),
        "--simuleval-instances",
        str(instances_path),
        "--reference",
        str(ref_file),
        "--audio-yaml",
        str(audio_yaml),
        "--sacrebleu-tokenizer",
        sacrebleu_tokenizer,
        "--latency-unit",
        latency_unit,
        "--glossary",
        str(glossary_path),
        "--term-lang",
        term_lang,
        "--term-mismatch-examples",
        str(term_mismatch_examples),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return StreamLAALOut(stdout=p.stdout, rc=int(p.returncode))


def _parse_metrics_from_stream_laal_output(txt: str) -> Metrics:
    bleu = stream_laal = stream_laal_ca = None
    term_correct = term_total = None

    lines = txt.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "BLEU\tStreamLAAL\tStreamLAAL_CA" and i + 1 < len(lines):
            parts = lines[i + 1].split("\t")
            if len(parts) >= 3:
                try:
                    bleu = float(parts[0])
                    stream_laal = float(parts[1])
                    stream_laal_ca = float(parts[2])
                except Exception:
                    pass
            break

    for line in lines:
        if line.startswith("TERM_ACC"):
            # Format: TERM_ACC\t0.8162\tCORRECT_TERMS\t1239\tTOTAL_TERMS\t1518
            parts = line.split("\t")
            if len(parts) >= 6:
                try:
                    term_correct = int(parts[3])
                    term_total = int(parts[5])
                except Exception:
                    pass
            break

    return Metrics(
        bleu=bleu,
        stream_laal=stream_laal,
        stream_laal_ca=stream_laal_ca,
        term_correct=term_correct,
        term_total=term_total,
    )


def _load_extracted_glossary(path: Path) -> Dict[str, Dict[str, Any]]:
    obj = json.loads(_read_text(path))
    if not isinstance(obj, dict):
        raise ValueError("Extracted glossary JSON must be a dict keyed by term.")
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in obj.items():
        if isinstance(v, dict):
            out[str(k)] = v
    return out


def _build_paper_glossaries(
    extracted: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    paper2gloss: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for term_key, entry in extracted.items():
        paper = _paper_id_from_source_paper(str(entry.get("source_paper", "")).strip())
        if not paper:
            continue
        paper2gloss.setdefault(paper, {})[term_key] = entry
    return paper2gloss


def _write_paper_glossary_files(paper2gloss: Dict[str, Dict[str, Dict[str, Any]]], out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paper2path: Dict[str, Path] = {}
    for paper_id, gloss in sorted(paper2gloss.items()):
        p = out_dir / f"{paper_id}.json"
        if not p.is_file():
            _write_json(p, gloss)
        paper2path[paper_id] = p
    return paper2path


def _group_instances_by_paper(instances_path: Path) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    warnings: List[str] = []
    for inst in _iter_jsonl(instances_path):
        wav = _extract_wav_path_from_instance(inst)
        if not wav:
            warnings.append("missing wav path in instance (no usable 'source').")
            continue
        paper = _paper_id_from_wav_path(wav)
        if not paper:
            warnings.append(f"cannot parse paper_id from wav path: {wav[:DEFAULT_MAX_SOURCE_PREVIEW_CHARS]}")
            continue
        grouped.setdefault(paper, []).append(inst)
    return grouped, warnings


def _write_instances_subset(path: Path, instances: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in instances:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _scan_output_dirs(output_zh_base: Path, extracted_glossary_tag: str, skip_dir_substr: str) -> List[Path]:
    outs: List[Path] = []
    if not output_zh_base.is_dir():
        return outs
    for p in sorted(output_zh_base.iterdir()):
        if not p.is_dir():
            continue
        if skip_dir_substr and skip_dir_substr in p.name:
            continue
        meta = parse_meta_from_dir_name(p.name)
        if meta is None:
            continue
        if meta.glossary_tag != extracted_glossary_tag:
            continue
        outs.append(p)
    return outs


def _load_audio_yaml_items(path: Path) -> List[Dict[str, Any]]:
    obj = yaml.safe_load(_read_text(path))
    if not isinstance(obj, list):
        raise ValueError("audio_yaml must be a list.")
    out: List[Dict[str, Any]] = []
    for x in obj:
        if isinstance(x, dict):
            out.append(x)
    return out


def _load_reference_lines(path: Path) -> List[str]:
    return _read_text(path).splitlines()


def _wav_name_from_paper_id(paper_id: str) -> str:
    return f"{paper_id}.wav"


def _subset_audio_and_ref_by_wav(
    audio_items: List[Dict[str, Any]], ref_lines: List[str], wav_name: str
) -> Tuple[List[Dict[str, Any]], List[str]]:
    if len(ref_lines) < len(audio_items):
        raise ValueError(
            f"REF_FILE has fewer lines than audio_yaml items: ref_lines={len(ref_lines)} audio_items={len(audio_items)}"
        )
    idxs: List[int] = []
    for i, item in enumerate(audio_items):
        w = item.get("wav")
        if isinstance(w, str) and w.strip() == wav_name:
            idxs.append(i)
    sub_audio = [audio_items[i] for i in idxs]
    sub_ref = [ref_lines[i] for i in idxs]
    return sub_audio, sub_ref


def _write_audio_yaml(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve a compact, readable flow style similar to the original file.
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(items, f, allow_unicode=True, default_flow_style=True, sort_keys=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-base", default=DEFAULT_OUTPUT_BASE)
    ap.add_argument("--lang-code", default=DEFAULT_LANG_CODE)
    ap.add_argument("--ref-file", default=DEFAULT_REF_FILE)
    ap.add_argument("--audio-yaml", default=DEFAULT_AUDIO_YAML)
    ap.add_argument("--extracted-glossary", default=DEFAULT_EXTRACTED_GLOSSARY_PATH)
    ap.add_argument("--fbk-fairseq-root", default=DEFAULT_FBK_FAIRSEQ_ROOT)
    ap.add_argument("--stream-laal-tool-rel", default=DEFAULT_STREAM_LAAL_TOOL_REL)
    ap.add_argument("--sacrebleu-tokenizer", default=DEFAULT_SACREBLEU_TOKENIZER)
    ap.add_argument("--latency-unit", default=DEFAULT_LATENCY_UNIT)
    ap.add_argument("--term-lang", default=DEFAULT_TERM_LANG)
    ap.add_argument("--term-mismatch-examples", default=DEFAULT_TERM_MISMATCH_EXAMPLES)
    ap.add_argument("--instances-file", default=DEFAULT_INSTANCES_FILE)
    ap.add_argument("--post-eval-log-name", default=DEFAULT_POST_EVAL_LOG_NAME)
    ap.add_argument("--summary-tsv-name", default=DEFAULT_SUMMARY_TSV_NAME)
    ap.add_argument("--skip-dir-substr", default=DEFAULT_SKIP_DIR_GLOB_SUBSTR)
    ap.add_argument("--max-output-dirs", type=int, default=DEFAULT_MAX_OUTPUT_DIRS)
    ap.add_argument("--paper-glossaries-dirname", default=DEFAULT_PAPER_GLOSSARIES_DIRNAME)
    ap.add_argument("--tmp-dirname", default=DEFAULT_TMP_DIRNAME)
    ap.add_argument("--python-exe", default=sys.executable)
    args = ap.parse_args()

    output_base = Path(args.output_base)
    output_zh_base = output_base / str(args.lang_code)
    ref_file = Path(args.ref_file)
    audio_yaml = Path(args.audio_yaml)
    extracted_glossary = Path(args.extracted_glossary)
    stream_laal_tool = Path(args.fbk_fairseq_root) / str(args.stream_laal_tool_rel)

    if not stream_laal_tool.is_file():
        _err(f"stream_laal_term.py not found: {stream_laal_tool}")
        return EXIT_CONFIG_ERROR
    if not extracted_glossary.is_file():
        _err(f"Extracted glossary not found: {extracted_glossary}")
        return EXIT_DATA_ERROR
    if not ref_file.is_file():
        _err(f"REF_FILE missing: {ref_file}")
        return EXIT_DATA_ERROR
    if not audio_yaml.is_file():
        _err(f"AUDIO_YAML missing: {audio_yaml}")
        return EXIT_DATA_ERROR

    _info(f"Loading audio_yaml: {audio_yaml}")
    audio_items = _load_audio_yaml_items(audio_yaml)
    _info(f"Loading reference file: {ref_file}")
    ref_lines = _load_reference_lines(ref_file)

    extracted_tag = extracted_glossary.stem
    paper_glossaries_dir = output_zh_base / str(args.paper_glossaries_dirname)

    _info(f"Loading extracted glossary: {extracted_glossary}")
    extracted = _load_extracted_glossary(extracted_glossary)
    paper2gloss = _build_paper_glossaries(extracted)
    if not paper2gloss:
        _err("No per-paper glossaries were built (source_paper may be missing).")
        return EXIT_DATA_ERROR

    paper2path = _write_paper_glossary_files(paper2gloss, paper_glossaries_dir)
    _info(f"Prepared per-paper glossaries: {len(paper2path)} papers at {paper_glossaries_dir}")

    out_dirs = _scan_output_dirs(output_zh_base, extracted_tag, str(args.skip_dir_substr))
    if not out_dirs:
        _warn(f"No output directories found under: {output_zh_base} for glossary_tag={extracted_tag}")
        return 0

    if args.max_output_dirs and args.max_output_dirs > 0:
        out_dirs = out_dirs[: int(args.max_output_dirs)]

    summary_path = output_zh_base / str(args.summary_tsv_name)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "timestamp",
                "glossary_tag",
                "vllm_segment_sec",
                "latency_multiplier",
                "K2",
                "K1",
                "BLEU",
                "StreamLAAL",
                "StreamLAAL_CA",
                "TERM_ACC",
                "TERM_CORRECT",
                "TERM_TOTAL",
                "output_path",
            ]
        )

        processed = 0
        for out_dir in out_dirs:
            meta = parse_meta_from_dir_name(out_dir.name)
            if meta is None:
                continue

            instances_path = out_dir / str(args.instances_file)
            if not instances_path.is_file() or instances_path.stat().st_size <= 0:
                _warn(f"Missing/empty instances file (skip): {instances_path}")
                continue

            tmp_dir = out_dir / str(args.tmp_dirname)
            tmp_dir.mkdir(parents=True, exist_ok=True)

            post_eval_log = out_dir / str(args.post_eval_log_name)
            log_parts: List[str] = []

            # 1) Full-run evaluation for BLEU/StreamLAAL (glossary-independent)
            _info(f"Full eval (BLEU/StreamLAAL) for: {out_dir}")
            full_out = _run_stream_laal_term(
                python_exe=str(args.python_exe),
                tool_path=stream_laal_tool,
                instances_path=instances_path,
                ref_file=ref_file,
                audio_yaml=audio_yaml,
                sacrebleu_tokenizer=str(args.sacrebleu_tokenizer),
                latency_unit=str(args.latency_unit),
                glossary_path=extracted_glossary,
                term_lang=str(args.term_lang),
                term_mismatch_examples=str(args.term_mismatch_examples),
            )
            log_parts.append("[SECTION] full_run\n")
            log_parts.append(full_out.stdout)
            if full_out.rc != 0:
                _warn(f"Full eval returned non-zero (rc={full_out.rc}): {out_dir}")
            full_metrics = _parse_metrics_from_stream_laal_output(full_out.stdout)

            # 2) Per-paper TERM evaluation (additive via correct/total)
            grouped, inst_warnings = _group_instances_by_paper(instances_path)
            for w in inst_warnings:
                _warn(f"{out_dir}: {w}")

            term_correct_sum = 0
            term_total_sum = 0
            seen_any_term = False

            log_parts.append("\n[SECTION] per_paper\n")
            for paper_id, insts in sorted(grouped.items()):
                gloss_path = paper2path.get(paper_id)
                if gloss_path is None:
                    _warn(f"{out_dir}: missing per-paper glossary for paper_id={paper_id} (skip this paper)")
                    continue

                subset_path = tmp_dir / f"instances_{paper_id}.jsonl"
                _write_instances_subset(subset_path, insts)

                wav_name = _wav_name_from_paper_id(paper_id)
                try:
                    sub_audio, sub_ref = _subset_audio_and_ref_by_wav(audio_items, ref_lines, wav_name)
                except Exception as e:
                    _warn(f"{out_dir}: failed to subset audio/ref for wav={wav_name}: {e}")
                    continue

                if not sub_audio or not sub_ref:
                    _warn(f"{out_dir}: empty subset for wav={wav_name} (skip this paper)")
                    continue

                subset_audio_yaml = tmp_dir / f"audio_{paper_id}.yaml"
                subset_ref_file = tmp_dir / f"ref_{paper_id}.txt"
                _write_audio_yaml(subset_audio_yaml, sub_audio)
                _write_text(subset_ref_file, "\n".join(sub_ref) + "\n")

                _info(f"TERM eval for paper_id={paper_id} instances={len(insts)} out={out_dir.name}")
                paper_out = _run_stream_laal_term(
                    python_exe=str(args.python_exe),
                    tool_path=stream_laal_tool,
                    instances_path=subset_path,
                    ref_file=subset_ref_file,
                    audio_yaml=subset_audio_yaml,
                    sacrebleu_tokenizer=str(args.sacrebleu_tokenizer),
                    latency_unit=str(args.latency_unit),
                    glossary_path=gloss_path,
                    term_lang=str(args.term_lang),
                    term_mismatch_examples=str(args.term_mismatch_examples),
                )
                log_parts.append(f"\n[SUBSECTION] paper_id={paper_id}\n")
                log_parts.append(paper_out.stdout)
                if paper_out.rc != 0:
                    _warn(f"Paper eval returned non-zero (rc={paper_out.rc}) paper_id={paper_id} out={out_dir}")

                m = _parse_metrics_from_stream_laal_output(paper_out.stdout)
                if m.term_correct is None or m.term_total is None:
                    _warn(f"{out_dir}: cannot parse TERM metrics for paper_id={paper_id}")
                    continue

                term_correct_sum += int(m.term_correct)
                term_total_sum += int(m.term_total)
                seen_any_term = True

            term_acc = ""
            if seen_any_term and term_total_sum > 0:
                term_acc = f"{(float(term_correct_sum) / float(term_total_sum)):.6f}"

            _write_text(post_eval_log, "".join(log_parts))

            from datetime import datetime

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow(
                [
                    ts,
                    meta.glossary_tag,
                    meta.vllm_segment_sec,
                    meta.latency_multiplier,
                    meta.k2,
                    meta.k1,
                    "" if full_metrics.bleu is None else f"{full_metrics.bleu}",
                    "" if full_metrics.stream_laal is None else f"{full_metrics.stream_laal}",
                    "" if full_metrics.stream_laal_ca is None else f"{full_metrics.stream_laal_ca}",
                    term_acc,
                    str(term_correct_sum),
                    str(term_total_sum),
                    str(out_dir),
                ]
            )

            processed += 1

    _info(f"Summary written: {summary_path} (rows={processed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


