#!/usr/bin/env python3
"""Rescore legacy medicine oracle outputs with glossary-derived metrics.

Legacy medicine oracle runs built the prompt term_map from ESO
``sentences[*].terms``.  That field is not the metric source of truth.  This
script keeps the generated hypotheses unchanged, but regenerates per-sample
source/reference-matched medicine glossaries and recomputes the offline metrics.

This is intended for Panel B / appendix analysis only: the prompt is still the
legacy sentence_terms oracle, while TERM metrics are corrected to a glossary
source-of-truth denominator.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


SAMPLES = ("404", "596001", "606", "545006")
FINAL_HEADER = [
    "lm",
    "samples",
    "sample_count",
    "complete",
    "BLEU",
    "StreamLAAL",
    "StreamLAAL_CA",
    "TERM_ACC",
    "TERM_CORRECT",
    "TERM_TOTAL",
    "REAL_TERM_ADOPT",
    "REAL_TERM_ADOPTED",
    "REAL_TERM_ADOPT_TOTAL",
    "TERM_FCR",
    "FALSE_COPY",
    "NEG_TOTAL",
    "SOURCE_TERM_SENT_FCR",
    "SOURCE_FALSE_COPY",
    "SOURCE_NEG_TOTAL",
    "combined_dir",
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


def _rate(num: int, den: int) -> str:
    return "N/A" if den <= 0 else f"{num / den:.6f}"


def _write_concat(paths: Iterable[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as out:
        for path in paths:
            if not path.is_file():
                _die(f"missing concat input: {path}")
            text = path.read_text(encoding="utf-8", errors="replace")
            out.write(text)
            if text and not text.endswith("\n"):
                out.write("\n")


def _merge_json_objects(paths: Iterable[Path], output: Path) -> None:
    merged: Dict[str, object] = {}
    for path in paths:
        if not path.is_file():
            _die(f"missing glossary: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            _die(f"glossary must be a JSON object: {path}")
        for key, value in data.items():
            merged.setdefault(key, value)
    output.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _latest_runtime(output_dir: Path) -> Path | None:
    paths = sorted(output_dir.glob("runtime_omni_vllm_maxsim_rag_*.jsonl"))
    return paths[-1] if paths else None


def _legacy_dir(args: argparse.Namespace, sample: str, lm: int) -> Path:
    if sample == "404":
        base = Path(args.legacy_onetalk_output_base)
        density = args.legacy_onetalk_density
    else:
        base = Path(args.legacy_remaining_output_base)
        density = args.legacy_remaining_density
    glossary_tag = f"{args.legacy_glossary_tag_prefix}__medicine_{sample}"
    suffix = (
        f"d{density}_oraclegt_lm{lm}_k{args.rag_top_k}_th{args.rag_score_threshold}"
        f"_g{glossary_tag}_ppmedicine_{sample}"
    )
    return base / args.lang_code / suffix


def _input_dir(args: argparse.Namespace, sample: str) -> Path:
    return Path(args.output_base) / args.lang_code / "__medicine_inputs__" / "lists"


def _rescore_dir(args: argparse.Namespace, sample: str, lm: int) -> Path:
    glossary_tag = f"{args.panelb_glossary_tag_prefix}__medicine_{sample}"
    suffix = (
        f"dmedicine_panelB_legacy_sentence_terms_rescored_oraclegt_lm{lm}"
        f"_k{args.rag_top_k}_g{glossary_tag}_ppmedicine_{sample}"
    )
    return Path(args.output_base) / args.lang_code / suffix


def _combined_dir(args: argparse.Namespace, lm: int, complete: bool) -> Path:
    label = "complete" if complete else "partial"
    suffix = (
        f"dmedicine4_panelB_legacy_sentence_terms_rescored_oraclegt_lm{lm}"
        f"_k{args.rag_top_k}_g{args.combined_glossary_tag}_{label}"
    )
    return Path(args.output_base) / args.lang_code / suffix


def _run(cmd: Sequence[str], env: Mapping[str, str] | None = None) -> None:
    print("[RUN]", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True, env=dict(env) if env is not None else None)


def _prepare_inputs(args: argparse.Namespace, sample: str) -> None:
    input_dir = _input_dir(args, sample)
    cmd = [
        sys.executable,
        args.prepare_script,
        "--sample-id",
        sample,
        "--lang-code",
        args.lang_code,
        "--output-dir",
        str(input_dir),
        "--term-source",
        "glossary_match",
        "--oracle-glossary",
        args.medicine_glossary,
        "--eval-glossary",
        args.medicine_glossary,
        "--glossary-source-filter",
        "medicine_gt",
        "--glossary-tag",
        f"{args.panelb_glossary_tag_prefix}__medicine_{sample}",
        "--oracle-term-map-tag",
        f"medicine.oracle_term_map__panelB_fixed_metric__medicine_{sample}",
    ]
    _run(cmd)


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def _offline_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MWERSEGMENTER_ROOT", "/mnt/taurus/home/jiaxuanluo/mwerSegmenter")
    env["PATH"] = f"{env['MWERSEGMENTER_ROOT']}:{env.get('PATH', '')}"
    return env


def _run_offline_eval(
    args: argparse.Namespace,
    instances: Path,
    source_text: Path,
    ref: Path,
    audio_yaml: Path,
    glossary: Path,
    out_dir: Path,
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = out_dir / "eval_results.tsv"
    out_log = out_dir / "raw_corpus_eval_results.log"
    cmd = [
        args.python_bin,
        args.offline_eval_script,
        "--mode",
        "acl6060",
        "--instances-log",
        str(instances),
        "--lang-code",
        args.lang_code,
        "--ref-file",
        str(ref),
        "--source-file",
        str(source_text),
        "--audio-yaml",
        str(audio_yaml),
        "--glossary-acl6060",
        str(glossary),
        "--term-fcr-policy",
        args.term_fcr_policy,
        "--output-tsv",
        str(out_tsv),
        "--output-log",
        str(out_log),
        "--work-dir",
        str(out_dir / "work"),
        "--python-bin",
        args.python_bin,
    ]
    _run(cmd, env=_offline_env())
    return _read_tsv_one(out_tsv)


def _rescore_sample(args: argparse.Namespace, sample: str, lm: int) -> Dict[str, str] | None:
    legacy = _legacy_dir(args, sample, lm)
    if not (legacy / "instances.log").is_file():
        print(f"[WARN] missing legacy output for sample={sample} lm={lm}: {legacy}", flush=True)
        return None

    inputs = _input_dir(args, sample)
    prefix = f"medicine_{sample}"
    out = _rescore_dir(args, sample, lm)
    _link_or_copy(legacy / "instances.log", out / "instances.log")
    runtime = _latest_runtime(legacy)
    if runtime is not None:
        _link_or_copy(runtime, out / runtime.name)

    row = _run_offline_eval(
        args=args,
        instances=out / "instances.log",
        source_text=inputs / f"medicine.source_text.en__{prefix}.txt",
        ref=inputs / f"medicine.ref.{args.lang_code}__{prefix}.txt",
        audio_yaml=inputs / f"medicine.audio__{prefix}.yaml",
        glossary=inputs / f"{args.panelb_glossary_tag_prefix}__medicine_{sample}.json",
        out_dir=out,
    )
    row["sample"] = sample
    row["lm"] = str(lm)
    row["output_dir"] = str(out)
    return row


def _aggregate_lm(args: argparse.Namespace, lm: int, samples: Sequence[str]) -> Dict[str, str]:
    rows = [_read_tsv_one(_rescore_dir(args, sample, lm) / "eval_results.tsv") for sample in samples]
    complete = tuple(samples) == tuple(args.samples)
    out_dir = _combined_dir(args, lm, complete)
    out_dir.mkdir(parents=True, exist_ok=True)

    instance_paths = [_rescore_dir(args, sample, lm) / "instances.log" for sample in samples]
    input_dirs = [_input_dir(args, sample) for sample in samples]
    prefixes = [f"medicine_{sample}" for sample in samples]

    instances = out_dir / "instances.log"
    ref = out_dir / f"medicine.ref.{args.lang_code}.txt"
    source_text = out_dir / "medicine.source_text.en.txt"
    audio_yaml = out_dir / "medicine.audio.yaml"
    glossary = out_dir / f"{args.combined_glossary_tag}.json"

    _write_concat(instance_paths, instances)
    _write_concat(
        [d / f"medicine.ref.{args.lang_code}__{p}.txt" for d, p in zip(input_dirs, prefixes)],
        ref,
    )
    _write_concat(
        [d / f"medicine.source_text.en__{p}.txt" for d, p in zip(input_dirs, prefixes)],
        source_text,
    )
    _write_concat([d / f"medicine.audio__{p}.yaml" for d, p in zip(input_dirs, prefixes)], audio_yaml)
    _merge_json_objects(
        [d / f"{args.panelb_glossary_tag_prefix}__medicine_{sample}.json" for d, sample in zip(input_dirs, samples)],
        glossary,
    )

    raw = _run_offline_eval(args, instances, source_text, ref, audio_yaml, glossary, out_dir)

    term_correct = sum(_as_int(r, "TERM_CORRECT") for r in rows)
    term_total = sum(_as_int(r, "TERM_TOTAL") for r in rows)
    real_adopted = sum(_as_int(r, "REAL_TERM_ADOPTED") for r in rows)
    real_total = sum(_as_int(r, "REAL_TERM_ADOPT_TOTAL") for r in rows)
    false_copy = sum(_as_int(r, "FALSE_COPY") for r in rows)
    neg_total = sum(_as_int(r, "NEG_TOTAL") for r in rows)
    source_false = sum(_as_int(r, "SOURCE_FALSE_COPY") for r in rows)
    source_neg = sum(_as_int(r, "SOURCE_NEG_TOTAL") for r in rows)

    final = {
        "lm": str(lm),
        "samples": ",".join(samples),
        "sample_count": str(len(samples)),
        "complete": str(complete).lower(),
        "BLEU": raw["BLEU"],
        "StreamLAAL": raw["StreamLAAL"],
        "StreamLAAL_CA": raw["StreamLAAL_CA"],
        "TERM_ACC": _rate(term_correct, term_total),
        "TERM_CORRECT": str(term_correct),
        "TERM_TOTAL": str(term_total),
        "REAL_TERM_ADOPT": _rate(real_adopted, real_total),
        "REAL_TERM_ADOPTED": str(real_adopted),
        "REAL_TERM_ADOPT_TOTAL": str(real_total),
        "TERM_FCR": _rate(false_copy, neg_total),
        "FALSE_COPY": str(false_copy),
        "NEG_TOTAL": str(neg_total),
        "SOURCE_TERM_SENT_FCR": _rate(source_false, source_neg),
        "SOURCE_FALSE_COPY": str(source_false),
        "SOURCE_NEG_TOTAL": str(source_neg),
        "combined_dir": str(out_dir),
    }
    with (out_dir / "eval_results.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FINAL_HEADER, delimiter="\t")
        writer.writeheader()
        writer.writerow(final)
    return final


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", default=list(SAMPLES))
    ap.add_argument("--lms", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--lang-code", default="zh")
    ap.add_argument("--rag-top-k", default="10")
    ap.add_argument("--rag-score-threshold", default="1.0")
    ap.add_argument("--legacy-onetalk-output-base", default="/mnt/gemini/data2/jiaxuanluo/medicine_onetalk_oracle_gt_sft_oraclegt_r32a64_20260519")
    ap.add_argument("--legacy-remaining-output-base", default="/mnt/gemini/data2/jiaxuanluo/medicine_remaining_oracle_gt_sft_oraclegt_r32a64_20260519")
    ap.add_argument("--legacy-onetalk-density", default="medicine1_oraclegt_r32a64")
    ap.add_argument("--legacy-remaining-density", default="medicine_remaining_oraclegt_r32a64")
    ap.add_argument("--legacy-glossary-tag-prefix", default="medicine_gt_strict_translated")
    ap.add_argument("--output-base", default="/mnt/gemini/data2/jiaxuanluo/medicine_panelB_legacy_sentence_terms_rescored_fixedgt_20260519")
    ap.add_argument("--panelb-glossary-tag-prefix", default="medicine_panelB_fixedmetric")
    ap.add_argument("--combined-glossary-tag", default="medicine_panelB_fixedmetric_four_samples")
    ap.add_argument("--medicine-glossary", default="/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/medicine_glossary_gt_plus_medicine_wiki_gs10000_translated.json")
    ap.add_argument("--prepare-script", default="documents/code/simuleval/prepare_medicine_one_talk_inputs.py")
    ap.add_argument("--offline-eval-script", default="documents/code/offline_sst_eval/offline_streamlaal_eval.py")
    ap.add_argument("--python-bin", default="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python")
    ap.add_argument(
        "--term-fcr-policy",
        choices=[
            "term_map_if_available",
            "term_map_source_ref_negative_sentence",
            "source_ref_negative_sentence",
        ],
        default="source_ref_negative_sentence",
    )
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    if not Path(args.python_bin).is_file():
        _die(f"python-bin not found: {args.python_bin}")

    for sample in args.samples:
        _prepare_inputs(args, sample)

    missing: List[Dict[str, str]] = []
    for lm in args.lms:
        for sample in args.samples:
            row = _rescore_sample(args, sample, lm)
            if row is None:
                missing.append({"lm": str(lm), "sample": sample, "legacy_dir": str(_legacy_dir(args, sample, lm))})

    summary_rows: List[Dict[str, str]] = []
    for lm in args.lms:
        available = [
            sample
            for sample in args.samples
            if (_rescore_dir(args, sample, lm) / "eval_results.tsv").is_file()
        ]
        if len(available) != len(args.samples) and not args.allow_partial:
            print(f"[WARN] skip aggregate lm={lm}; available={available}", flush=True)
            continue
        if not available:
            continue
        summary_rows.append(_aggregate_lm(args, lm, available))

    out_root = Path(args.output_base) / args.lang_code
    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "summary_lm_sweep.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FINAL_HEADER, delimiter="\t")
        writer.writeheader()
        writer.writerows(summary_rows)
    with (out_root / "missing_outputs.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lm", "sample", "legacy_dir"], delimiter="\t")
        writer.writeheader()
        writer.writerows(missing)
    print(f"[INFO] wrote {out_root / 'summary_lm_sweep.tsv'}")
    print(f"[INFO] wrote {out_root / 'missing_outputs.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
