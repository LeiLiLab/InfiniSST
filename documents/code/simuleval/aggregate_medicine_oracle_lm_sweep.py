#!/usr/bin/env python3
"""Aggregate ESO medicine oracle SimulEval outputs across talks and LMs.

This script intentionally separates metric sources:

* BLEU / StreamLAAL are recomputed on a concatenated corpus for each LM.
* term metrics are pooled from the per-talk ``eval_results.tsv`` count columns,
  because the current sentence-level term-map metrics need one runtime log per
  talk and are not valid on a four-talk concatenated ``instances.log``.

The output layout matches ``wandb_eval_logger.py`` scanning conventions, so the
generated ``eval_results.tsv`` files can be logged as one combined readout run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


SAMPLE_IDS = ("404", "596001", "606", "545006")

FINAL_HEADER = [
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


def _die(msg: str) -> None:
    raise SystemExit(f"[ERROR] {msg}")


def _read_tsv_one(path: Path) -> Dict[str, str]:
    if not path.is_file() or path.stat().st_size == 0:
        _die(f"missing non-empty TSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        _die(f"empty TSV rows: {path}")
    return rows[-1]


def _as_int(row: Mapping[str, str], key: str) -> int:
    raw = (row.get(key) or "").strip()
    if raw in {"", "N/A"}:
        return 0
    return int(float(raw))


def _safe_rate(num: int, den: int) -> str:
    if den <= 0:
        return "N/A"
    return f"{num / den:.6f}"


def _zero_when_no_candidates_rate(num: int, den: int) -> str:
    if den <= 0:
        return "0.000000"
    return f"{num / den:.6f}"


def _write_concat(paths: Iterable[Path], output: Path) -> None:
    with output.open("w", encoding="utf-8") as out:
        for path in paths:
            if not path.is_file():
                _die(f"missing input file for concat: {path}")
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    out.write(line)
                if output.suffix != ".jsonl":
                    out.write("" if line.endswith("\n") else "\n")


def _write_concat_json_arrays(paths: Iterable[Path], output: Path) -> None:
    merged: List[object] = []
    for path in paths:
        if not path.is_file():
            _die(f"missing JSON array input for concat: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            _die(f"expected JSON array: {path}")
        merged.extend(data)
    output.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _merge_glossaries(paths: Iterable[Path], output: Path) -> None:
    merged: Dict[str, object] = {}
    for path in paths:
        if not path.is_file():
            _die(f"missing glossary: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            _die(f"glossary must be a JSON object: {path}")
        for key, value in data.items():
            merged.setdefault(key, value)
    output.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _glossary_tag(args: argparse.Namespace, sample: str) -> str:
    return args.glossary_tag_pattern.format(sample=sample)


def _oracle_term_map_tag(args: argparse.Namespace, sample: str) -> str:
    return args.oracle_term_map_tag_pattern.format(sample=sample)


def _sample_output_dir(args: argparse.Namespace, sample: str, lm: int) -> Path:
    if sample == "404":
        base = Path(args.onetalk_output_base)
        density = args.onetalk_density
    else:
        base = Path(args.remaining_output_base)
        density = args.remaining_density
    glossary_tag = _glossary_tag(args, sample)
    suffix = (
        f"d{density}_oraclegt_lm{lm}_k{args.rag_top_k}_th{args.rag_score_threshold}"
        f"_g{glossary_tag}_ppmedicine_{sample}"
    )
    return base / args.lang_code / suffix


def _sample_inputs_dir(args: argparse.Namespace, sample: str) -> Path:
    base = Path(args.onetalk_output_base if sample == "404" else args.remaining_output_base)
    return base / args.lang_code / "__medicine_inputs__" / "lists"


def _latest_runtime_jsonl(output_dir: Path) -> Path | None:
    paths = sorted(output_dir.glob("runtime_omni_vllm_maxsim_rag_*.jsonl"))
    return paths[-1] if paths else None


def _run_corpus_eval(args: argparse.Namespace, lm_dir: Path, inputs: Dict[str, Path]) -> Dict[str, str]:
    raw_tsv = lm_dir / "raw_corpus_eval_results.tsv"
    raw_log = lm_dir / "raw_corpus_eval_results.log"
    cmd = [
        args.python_bin,
        args.offline_eval_script,
        "--mode",
        "acl6060",
        "--instances-log",
        str(inputs["instances"]),
        "--lang-code",
        args.lang_code,
        "--ref-file",
        str(inputs["ref"]),
        "--source-file",
        str(inputs["source_text"]),
        "--audio-yaml",
        str(inputs["audio_yaml"]),
        "--sentence-term-map",
        str(inputs["sentence_term_map"]),
        "--glossary-acl6060",
        str(inputs["glossary"]),
        "--term-fcr-policy",
        args.term_fcr_policy,
        "--output-tsv",
        str(raw_tsv),
        "--output-log",
        str(raw_log),
        "--work-dir",
        str(lm_dir / "work"),
    ]
    env = os.environ.copy()
    env.setdefault("MWERSEGMENTER_ROOT", "/mnt/taurus/home/jiaxuanluo/mwerSegmenter")
    env["PATH"] = f"{env['MWERSEGMENTER_ROOT']}:{env.get('PATH', '')}"
    subprocess.run(cmd, check=True, env=env)
    return _read_tsv_one(raw_tsv)


def _aggregate_lm(args: argparse.Namespace, lm: int) -> Dict[str, str]:
    out_base = Path(args.aggregate_output_base)
    combined_glossary_tag = args.combined_glossary_tag
    lm_dir = (
        out_base
        / args.lang_code
        / f"d{args.aggregate_density}_oraclegt_lm{lm}_k{args.rag_top_k}_g{combined_glossary_tag}"
    )
    lm_dir.mkdir(parents=True, exist_ok=True)

    sample_rows: List[Dict[str, str]] = []
    instance_paths: List[Path] = []
    runtime_paths: List[Path] = []
    ref_paths: List[Path] = []
    source_paths: List[Path] = []
    audio_paths: List[Path] = []
    glossary_paths: List[Path] = []
    sentence_term_map_paths: List[Path] = []

    for sample in args.samples:
        out_dir = _sample_output_dir(args, sample, lm)
        eval_tsv = out_dir / "eval_results.tsv"
        sample_rows.append(_read_tsv_one(eval_tsv))
        instance_paths.append(out_dir / "instances.log")
        runtime = _latest_runtime_jsonl(out_dir)
        if runtime is not None:
            runtime_paths.append(runtime)

        inputs_dir = _sample_inputs_dir(args, sample)
        prefix = f"medicine_{sample}"
        ref_paths.append(inputs_dir / f"medicine.ref.{args.lang_code}__{prefix}.txt")
        source_paths.append(inputs_dir / f"medicine.source_text.en__{prefix}.txt")
        audio_paths.append(inputs_dir / f"medicine.audio__{prefix}.yaml")
        glossary_paths.append(inputs_dir / f"{_glossary_tag(args, sample)}.json")
        sentence_term_map_paths.append(inputs_dir / f"{_oracle_term_map_tag(args, sample)}.json")

    inputs = {
        "instances": lm_dir / "instances.log",
        "runtime": lm_dir / f"runtime_omni_vllm_maxsim_rag_combined_lm{lm}.jsonl",
        "ref": lm_dir / f"medicine.ref.{args.lang_code}.txt",
        "source_text": lm_dir / "medicine.source_text.en.txt",
        "audio_yaml": lm_dir / "medicine.audio.yaml",
        "glossary": lm_dir / f"{combined_glossary_tag}.json",
        "sentence_term_map": lm_dir / "medicine.oracle_term_map.json",
    }
    _write_concat(instance_paths, inputs["instances"])
    if runtime_paths:
        _write_concat(runtime_paths, inputs["runtime"])
    _write_concat(ref_paths, inputs["ref"])
    _write_concat(source_paths, inputs["source_text"])
    _write_concat(audio_paths, inputs["audio_yaml"])
    _merge_glossaries(glossary_paths, inputs["glossary"])
    _write_concat_json_arrays(sentence_term_map_paths, inputs["sentence_term_map"])

    raw = _run_corpus_eval(args, lm_dir, inputs)

    term_correct = sum(_as_int(r, "TERM_CORRECT") for r in sample_rows)
    term_total = sum(_as_int(r, "TERM_TOTAL") for r in sample_rows)
    adopted = sum(_as_int(r, "TERM_ADOPTED") for r in sample_rows)
    adopt_total = sum(_as_int(r, "TERM_ADOPTION_TOTAL") for r in sample_rows)
    adopt_sentences = sum(_as_int(r, "TERM_ADOPTION_SENTENCES") for r in sample_rows)
    real_adopted = sum(_as_int(r, "REAL_TERM_ADOPTED") for r in sample_rows)
    real_total = sum(_as_int(r, "REAL_TERM_ADOPT_TOTAL") for r in sample_rows)
    real_sentences = sum(_as_int(r, "REAL_TERM_ADOPT_SENTENCES") for r in sample_rows)
    false_copy = sum(_as_int(r, "FALSE_COPY") for r in sample_rows)
    neg_total = sum(_as_int(r, "NEG_TOTAL") for r in sample_rows)
    false_copy_terms = sum(_as_int(r, "FALSE_COPY_TERMS") for r in sample_rows)
    source_false = sum(_as_int(r, "SOURCE_FALSE_COPY") for r in sample_rows)
    source_neg = sum(_as_int(r, "SOURCE_NEG_TOTAL") for r in sample_rows)
    source_false_terms = sum(_as_int(r, "SOURCE_FALSE_COPY_TERMS") for r in sample_rows)

    final = {
        "mode": "medicine4_oracle_gt",
        "lang_code": args.lang_code,
        "BLEU": raw["BLEU"],
        "StreamLAAL": raw["StreamLAAL"],
        "StreamLAAL_CA": raw["StreamLAAL_CA"],
        "TERM_ACC": _safe_rate(term_correct, term_total),
        "TERM_CORRECT": str(term_correct),
        "TERM_TOTAL": str(term_total),
        "TERM_ADOPTION": _safe_rate(adopted, adopt_total),
        "TERM_ADOPTED": str(adopted),
        "TERM_ADOPTION_TOTAL": str(adopt_total),
        "TERM_ADOPTION_SENTENCES": str(adopt_sentences),
        "TERM_ADOPTION_MICRO": _safe_rate(adopted, adopt_total),
        "REAL_TERM_ADOPT": _safe_rate(real_adopted, real_total),
        "REAL_TERM_ADOPTED": str(real_adopted),
        "REAL_TERM_ADOPT_TOTAL": str(real_total),
        "REAL_TERM_ADOPT_SENTENCES": str(real_sentences),
        "REAL_TERM_ADOPT_MICRO": _safe_rate(real_adopted, real_total),
        "TERM_FCR": _zero_when_no_candidates_rate(false_copy, neg_total),
        "FALSE_COPY": str(false_copy),
        "NEG_TOTAL": str(neg_total),
        "FALSE_COPY_TERMS": str(false_copy_terms),
        "instances_log": str(inputs["instances"]),
        "TERM_FCR_MODE": f"{args.term_fcr_policy}_pooled_from_per_talk",
        "SOURCE_TERM_SENT_FCR": _safe_rate(source_false, source_neg),
        "SOURCE_FALSE_COPY": str(source_false),
        "SOURCE_NEG_TOTAL": str(source_neg),
        "SOURCE_FALSE_COPY_TERMS": str(source_false_terms),
    }

    final_tsv = lm_dir / "eval_results.tsv"
    with final_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FINAL_HEADER, delimiter="\t")
        writer.writeheader()
        writer.writerow(final)

    manifest = {
        "lm": lm,
        "samples": list(args.samples),
        "sample_output_dirs": [str(_sample_output_dir(args, s, lm)) for s in args.samples],
        "combined_eval_dir": str(lm_dir),
        "raw_corpus_eval_tsv": str(lm_dir / "raw_corpus_eval_results.tsv"),
        "final_eval_tsv": str(final_tsv),
        "metric_policy": {
            "BLEU": "corpus recomputed from concatenated instances/ref/audio",
            "StreamLAAL": "corpus recomputed from concatenated instances/ref/audio",
            "term_metrics": "pooled from per-talk eval_results.tsv count columns",
        },
    }
    (lm_dir / "aggregate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lms", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--samples", nargs="+", default=list(SAMPLE_IDS))
    parser.add_argument("--lang-code", default="zh")
    parser.add_argument("--rag-top-k", default="10")
    parser.add_argument("--rag-score-threshold", default="1.0")
    parser.add_argument(
        "--onetalk-output-base",
        default="/mnt/gemini/data2/jiaxuanluo/medicine_onetalk_oracle_gt_sft_oraclegt_r32a64_20260519",
    )
    parser.add_argument(
        "--remaining-output-base",
        default="/mnt/gemini/data2/jiaxuanluo/medicine_remaining_oracle_gt_sft_oraclegt_r32a64_20260519",
    )
    parser.add_argument("--onetalk-density", default="medicine1_oraclegt_r32a64")
    parser.add_argument("--remaining-density", default="medicine_remaining_oraclegt_r32a64")
    parser.add_argument(
        "--aggregate-output-base",
        default="/mnt/gemini/data2/jiaxuanluo/medicine4_oracle_gt_sft_oraclegt_r32a64_lm_sweep_20260519",
    )
    parser.add_argument("--aggregate-density", default="medicine4_oraclegt_r32a64")
    parser.add_argument("--combined-glossary-tag", default="medicine_gt_strict_translated_four_samples")
    parser.add_argument(
        "--glossary-tag-pattern",
        default="medicine_gt_strict_translated__medicine_{sample}",
        help="Per-sample glossary filename stem pattern; {sample} is replaced by the ESO sample id.",
    )
    parser.add_argument(
        "--oracle-term-map-tag-pattern",
        default="medicine.oracle_term_map__medicine_{sample}",
        help="Per-sample oracle term_map filename stem pattern; {sample} is replaced by the ESO sample id.",
    )
    parser.add_argument(
        "--offline-eval-script",
        default="documents/code/offline_sst_eval/offline_streamlaal_eval.py",
    )
    parser.add_argument("--python-bin", default="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python")
    parser.add_argument(
        "--term-fcr-policy",
        choices=[
            "term_map_if_available",
            "term_map_source_ref_negative_sentence",
            "source_ref_negative_sentence",
        ],
        default="term_map_if_available",
    )
    args = parser.parse_args()

    summaries = [_aggregate_lm(args, lm) for lm in args.lms]
    summary_path = Path(args.aggregate_output_base) / args.lang_code / "summary_lm_sweep.tsv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lm"] + FINAL_HEADER, delimiter="\t")
        writer.writeheader()
        for lm, row in zip(args.lms, summaries):
            out = {"lm": str(lm)}
            out.update(row)
            writer.writerow(out)
    print(f"[INFO] Wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
