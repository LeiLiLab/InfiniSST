#!/usr/bin/env python3
"""Aggregate ESO medicine retriever-RAG SimulEval outputs across talks and LMs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Mapping


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
            text = path.read_text(encoding="utf-8")
            out.write(text)
            if text and not text.endswith("\n"):
                out.write("\n")


def _merge_json_objects(paths: Iterable[Path], output: Path) -> None:
    merged: Dict[str, object] = {}
    for path in paths:
        if not path.is_file():
            _die(f"missing glossary: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            items = data.items()
        elif isinstance(data, list):
            items = []
            for row in data:
                if not isinstance(row, dict):
                    continue
                term = row.get("term") or row.get("source")
                if term:
                    items.append((str(term).casefold(), row))
        else:
            _die(f"unsupported glossary format: {path}")
        for key, value in items:
            merged.setdefault(str(key), value)
    output.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sample_output_dir(args: argparse.Namespace, sample: str, lm: int) -> Path:
    suffix = (
        f"d{args.density_tag}_lm{lm}_k{args.rag_top_k}_th{args.rag_score_threshold}"
        f"_g{args.runtime_glossary_tag}_ppmedicine_{sample}"
    )
    return Path(args.output_base) / args.lang_code / suffix


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
        "--glossary-acl6060",
        str(inputs["glossary"]),
        "--strip-output-tags",
        args.strip_output_tags,
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
    lm_dir = (
        Path(args.output_base)
        / args.lang_code
        / f"d{args.aggregate_density_tag}_lm{lm}_k{args.rag_top_k}_th{args.rag_score_threshold}_g{args.combined_glossary_tag}"
    )
    lm_dir.mkdir(parents=True, exist_ok=True)

    sample_rows = []
    instance_paths = []
    ref_paths = []
    source_paths = []
    audio_paths = []
    glossary_paths = []
    inputs_dir = Path(args.output_base) / args.lang_code / "__medicine_inputs__" / "lists"

    for sample in args.samples:
        out_dir = _sample_output_dir(args, sample, lm)
        sample_rows.append(_read_tsv_one(out_dir / "eval_results.tsv"))
        instance_paths.append(out_dir / "instances.log")
        prefix = f"medicine_{sample}"
        ref_paths.append(inputs_dir / f"medicine.ref.{args.lang_code}__{prefix}.txt")
        source_paths.append(inputs_dir / f"medicine.source_text.en__{prefix}.txt")
        audio_paths.append(inputs_dir / f"medicine.audio__{prefix}.yaml")
        glossary_paths.append(inputs_dir / f"{args.glossary_tag_pattern.format(sample=sample)}.json")

    inputs = {
        "instances": lm_dir / "instances.log",
        "ref": lm_dir / f"medicine.ref.{args.lang_code}.txt",
        "source_text": lm_dir / "medicine.source_text.en.txt",
        "audio_yaml": lm_dir / "medicine.audio.yaml",
        "glossary": lm_dir / f"{args.combined_glossary_tag}.json",
    }
    _write_concat(instance_paths, inputs["instances"])
    _write_concat(ref_paths, inputs["ref"])
    _write_concat(source_paths, inputs["source_text"])
    _write_concat(audio_paths, inputs["audio_yaml"])
    _merge_json_objects(glossary_paths, inputs["glossary"])

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
        "mode": "medicine_hardraw_retriever",
        "lang_code": args.lang_code,
        "BLEU": raw["BLEU"],
        "StreamLAAL": raw["StreamLAAL"],
        "StreamLAAL_CA": raw.get("StreamLAAL_CA", "N/A"),
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

    with (lm_dir / "eval_results.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FINAL_HEADER, delimiter="\t")
        writer.writeheader()
        writer.writerow(final)
    return final


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-base", required=True)
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--lms", nargs="+", type=int, required=True)
    ap.add_argument("--lang-code", default="zh")
    ap.add_argument("--density-tag", required=True)
    ap.add_argument("--aggregate-density-tag", required=True)
    ap.add_argument("--runtime-glossary-tag", required=True)
    ap.add_argument("--combined-glossary-tag", required=True)
    ap.add_argument("--glossary-tag-pattern", required=True)
    ap.add_argument("--rag-top-k", default="10")
    ap.add_argument("--rag-score-threshold", default="0.78")
    ap.add_argument("--offline-eval-script", default="documents/code/offline_sst_eval/offline_streamlaal_eval.py")
    ap.add_argument("--python-bin", default="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python")
    ap.add_argument("--term-fcr-policy", default="term_map_source_ref_negative_sentence")
    ap.add_argument("--strip-output-tags", default="term")
    args = ap.parse_args()

    summaries = [_aggregate_lm(args, lm) for lm in args.lms]
    summary_path = Path(args.output_base) / args.lang_code / "summary_lm_sweep.tsv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lm"] + FINAL_HEADER, delimiter="\t")
        writer.writeheader()
        for lm, row in zip(args.lms, summaries):
            out = {"lm": str(lm)}
            out.update(row)
            writer.writerow(out)

    md = Path(args.output_base) / "summary_medicine_hardraw_main.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# Medicine Hard-Raw Main Results\n\n")
        f.write("| lm | BLEU | TERM_ACC | REAL_ADOPT | TERM_FCR | SOURCE_SENT_FCR | StreamLAAL |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|\n")
        for lm, row in zip(args.lms, summaries):
            def pct(key: str) -> str:
                raw = row.get(key, "N/A")
                return "N/A" if raw == "N/A" else f"{float(raw) * 100:.2f}"
            f.write(
                f"| {lm} | {float(row['BLEU']):.2f} | {pct('TERM_ACC')} | "
                f"{pct('REAL_TERM_ADOPT')} | {pct('TERM_FCR')} | "
                f"{pct('SOURCE_TERM_SENT_FCR')} | {float(row['StreamLAAL']):.1f} |\n"
            )
        f.write(f"\nTSV: `{summary_path}`\n")
    print(f"[INFO] Wrote summary: {summary_path}")
    print(md.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
