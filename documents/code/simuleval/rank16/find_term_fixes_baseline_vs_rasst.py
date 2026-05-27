#!/usr/bin/env python3

"""
Find cases where baseline misses a glossary term but RASST includes it (zh/ja/de).

We compare two per-paper SimulEval outputs that share the same references and per-paper glossaries:
- baseline output base
- rasst output base

For each paper:
- load per-paper glossary (from paper_inputs_map.json)
- resegment both runs to segment-level predictions using the same references (mWERSegmenter)
- for each segment and each glossary term appearing in REF:
    baseline misses term AND rasst contains term  -> record as a "fix" example

Outputs a TSV to stdout (and optionally to a file).

All user-facing strings are in English.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


STREAM_LAAL_TERM_PY = (
    "/mnt/taurus/home/jiaxuanluo/FBK-fairseq/examples/speech_to_text/"
    "simultaneous_translation/scripts/stream_laal_term.py"
)

DEFAULT_DATA_ROOT = "/mnt/taurus/data/siqiouyang/datasets/acl6060"
DEFAULT_MWERSEGMENTER_ROOT = "/mnt/taurus/home/jiaxuanluo/mwerSegmenter"

LANG_DEFAULTS = {
    "zh": {"latency_unit": "char"},
    "ja": {"latency_unit": "char"},
    "de": {"latency_unit": "word"},
}


def _die(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    raise SystemExit(2)


def _load_stream_laal_term_module():
    p = Path(STREAM_LAAL_TERM_PY)
    if not p.is_file():
        _die(f"stream_laal_term.py not found: {p}")
    spec = importlib.util.spec_from_file_location("stream_laal_term", str(p))
    if spec is None or spec.loader is None:
        _die("Failed to load stream_laal_term.py module spec.")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _find_output_dir(output_base: Path, lang_code: str, glossary_tag: str, lm: int, k2: int, k1: int, th_tag: str) -> Path:
    # Match the naming used by rank16/rank32 scripts:
    #   <model_short>_g<GLOSSARY_TAG>_cs*_hs0.48_lm<LM>_k2<K2>_k1<K1>_th<TH>
    # Pick the most recently modified match.
    root = output_base / lang_code
    if not root.is_dir():
        _die(f"Missing lang dir: {root}")
    candidates = sorted(
        root.glob(f"*_g{glossary_tag}_cs*_hs0.48_lm{lm}_k2{k2}_k1{k1}_th{th_tag}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        _die(f"No output dir match under {root} for glossary_tag={glossary_tag} lm={lm} k2={k2} k1={k1} th={th_tag}")
    return candidates[0]


@dataclass(frozen=True)
class FixExample:
    paper_id: str
    seg_idx: int
    term_target: str
    term_en: str
    src: str
    ref: str
    baseline_pred: str
    rasst_pred: str


def _truncate(s: str, n: int) -> str:
    s = (s or "").replace("\t", " ").replace("\n", " ").strip()
    if len(s) <= n:
        return s
    return s[:n] + "…"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-output-base", required=True)
    ap.add_argument("--rasst-output-base", required=True)
    ap.add_argument("--lang-code", default="zh", choices=sorted(LANG_DEFAULTS.keys()))
    ap.add_argument("--lm", type=int, default=2)
    ap.add_argument("--k2", type=int, default=10)
    ap.add_argument("--k1", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--map-json", default="")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--source-ref", default="")  # optional, defaults to en-xx.en.txt
    ap.add_argument("--max-output-examples", type=int, default=50)
    ap.add_argument("--max-text-chars", type=int, default=220)
    ap.add_argument("--output-tsv", default="")
    args = ap.parse_args()

    # Ensure mwerSegmenter is available (required for resegmentation).
    if not os.environ.get("MWERSEGMENTER_ROOT"):
        p = Path(DEFAULT_MWERSEGMENTER_ROOT)
        if p.is_dir():
            os.environ["MWERSEGMENTER_ROOT"] = str(p)
            # Some installs expect the binary under MWERSEGMENTER_ROOT/mwerSegmenter
            os.environ["PATH"] = f"{p}:{os.environ.get('PATH','')}"

    baseline_base = Path(args.baseline_output_base)
    rasst_base = Path(args.rasst_output_base)
    if not baseline_base.is_dir():
        _die(f"baseline-output-base not found: {baseline_base}")
    if not rasst_base.is_dir():
        _die(f"rasst-output-base not found: {rasst_base}")

    lang_code = args.lang_code
    latency_unit = LANG_DEFAULTS[lang_code]["latency_unit"]
    th_tag = str(args.threshold).replace(".", "p")

    map_json = Path(args.map_json) if args.map_json else (baseline_base / lang_code / "__paper_inputs__" / "paper_inputs_map.json")
    if not map_json.is_file():
        _die(f"paper_inputs_map.json not found: {map_json}")

    data_root = Path(args.data_root)
    if not data_root.is_dir():
        _die(f"data-root not found: {data_root}")
    source_ref = Path(args.source_ref) if args.source_ref else (data_root / "dev/text/txt/ACL.6060.dev.en-xx.en.txt")
    if not source_ref.is_file():
        print(f"[WARN] source reference missing (SRC column will be empty): {source_ref}", file=sys.stderr)

    mp = _read_json(map_json)
    papers: Dict[str, Dict[str, str]] = mp.get("papers", {}) if isinstance(mp, dict) else {}
    if not papers:
        _die(f"No papers found in mapping: {map_json}")

    stream_mod = _load_stream_laal_term_module()

    fixes: List[FixExample] = []

    for paper_id, info in sorted(papers.items()):
        glossary_path = Path(str(info.get("glossary_path", "")))
        if not glossary_path.is_file():
            print(f"[WARN] Skip {paper_id}: missing glossary_path={glossary_path}", file=sys.stderr)
            continue
        glossary_tag = glossary_path.stem  # extracted_glossary__<paper_id>

        base_out = _find_output_dir(baseline_base, lang_code, glossary_tag, args.lm, args.k2, args.k1, th_tag)
        rasst_out = _find_output_dir(rasst_base, lang_code, glossary_tag, args.lm, args.k2, args.k1, th_tag)

        base_instances = base_out / "instances.log"
        rasst_instances = rasst_out / "instances.log"
        if not base_instances.is_file() or base_instances.stat().st_size == 0:
            print(f"[WARN] Skip {paper_id}: missing/empty baseline instances.log: {base_instances}", file=sys.stderr)
            continue
        if not rasst_instances.is_file() or rasst_instances.stat().st_size == 0:
            print(f"[WARN] Skip {paper_id}: missing/empty rasst instances.log: {rasst_instances}", file=sys.stderr)
            continue

        # Prefer the per-paper ref/audio already generated under output dirs.
        ref_file = base_out / f"ref_{paper_id}.txt"
        audio_yaml = base_out / f"audio_{paper_id}.yaml"
        if not ref_file.is_file() or not audio_yaml.is_file():
            # Fall back to rasst dir
            ref_file = rasst_out / f"ref_{paper_id}.txt"
            audio_yaml = rasst_out / f"audio_{paper_id}.yaml"
        if not ref_file.is_file() or not audio_yaml.is_file():
            _die(f"Missing per-paper ref/audio for {paper_id} under {base_out} or {rasst_out}")

        # Build references dict with optional source_reference.
        # Note: per-paper ref files are subsets, so a full dev source ref file may not align.
        source_ref_arg = str(source_ref) if source_ref.is_file() else None
        try:
            references = stream_mod.parse_references(  # type: ignore[attr-defined]
                str(ref_file),
                str(audio_yaml),
                source_ref_arg,
            )
        except AssertionError:
            if source_ref_arg:
                print(
                    f"[WARN] {paper_id}: source reference does not align with per-paper refs; "
                    f"retrying without SRC. source_ref={source_ref_arg}",
                    file=sys.stderr,
                )
            references = stream_mod.parse_references(str(ref_file), str(audio_yaml), None)  # type: ignore[attr-defined]

        base_pred = stream_mod.parse_simuleval_instances(str(base_instances), latency_unit)  # type: ignore[attr-defined]
        rasst_pred = stream_mod.parse_simuleval_instances(str(rasst_instances), latency_unit)  # type: ignore[attr-defined]

        # Keys should match (single wav for per-paper runs)
        if set(base_pred.keys()) != set(references.keys()):
            _die(f"{paper_id}: baseline predictions keys != references keys: {base_pred.keys()} vs {references.keys()}")
        if set(rasst_pred.keys()) != set(references.keys()):
            _die(f"{paper_id}: rasst predictions keys != references keys: {rasst_pred.keys()} vs {references.keys()}")

        base_segs = stream_mod.resegment_instances(base_pred, references, latency_unit)  # type: ignore[attr-defined]
        rasst_segs = stream_mod.resegment_instances(rasst_pred, references, latency_unit)  # type: ignore[attr-defined]
        if len(base_segs) != len(rasst_segs):
            _die(f"{paper_id}: segment count mismatch after resegmentation: {len(base_segs)} vs {len(rasst_segs)}")

        target_terms = stream_mod.load_glossary(str(glossary_path), lang_code)  # type: ignore[attr-defined]
        if not target_terms:
            print(f"[WARN] {paper_id}: no target terms for lang={lang_code} in {glossary_path}", file=sys.stderr)
            continue

        for i, (b, r) in enumerate(zip(base_segs, rasst_segs)):
            ref = getattr(b, "reference", "") or ""
            src = getattr(b, "source_reference", "") or ""
            bpred = getattr(b, "prediction", "") or ""
            rpred = getattr(r, "prediction", "") or ""
            if not ref:
                continue
            # For each term present in ref, check miss->hit
            for t in target_terms:
                term_target = t["target"]
                if term_target in ref and (term_target not in bpred) and (term_target in rpred):
                    fixes.append(
                        FixExample(
                            paper_id=paper_id,
                            seg_idx=i,
                            term_target=term_target,
                            term_en=t.get("en", ""),
                            src=src,
                            ref=ref,
                            baseline_pred=bpred,
                            rasst_pred=rpred,
                        )
                    )

    # Output
    header = [
        "paper_id",
        "seg_idx",
        "term_target",
        "term_en",
        "src",
        "ref",
        "baseline_pred",
        "rasst_pred",
    ]

    rows: List[List[str]] = []
    for ex in fixes[: max(0, int(args.max_output_examples))]:
        rows.append(
            [
                ex.paper_id,
                str(ex.seg_idx),
                _truncate(ex.term_target, args.max_text_chars),
                _truncate(ex.term_en, args.max_text_chars),
                _truncate(ex.src, args.max_text_chars),
                _truncate(ex.ref, args.max_text_chars),
                _truncate(ex.baseline_pred, args.max_text_chars),
                _truncate(ex.rasst_pred, args.max_text_chars),
            ]
        )

    out_lines = ["\t".join(header)]
    out_lines += ["\t".join(r) for r in rows]
    out_text = "\n".join(out_lines) + "\n"

    if args.output_tsv:
        Path(args.output_tsv).write_text(out_text, encoding="utf-8")
    sys.stdout.write(out_text)

    print(f"[INFO] Found {len(fixes)} fix candidates (showing {len(rows)}).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

