#!/usr/bin/env python3
"""Sentence-level comparison for two medicine SimulEval instances.log files."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


LANG_DEFAULTS = {
    "zh": {"latency_unit": "char", "term_lang": "zh"},
    "ja": {"latency_unit": "char", "term_lang": "ja"},
    "de": {"latency_unit": "word", "term_lang": "de"},
}


def _load_stream_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("stream_laal_term", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import stream_laal_term.py: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["stream_laal_term"] = mod
    spec.loader.exec_module(mod)
    return mod


def _normalise_glossary(path: Path, output_path: Path) -> Path:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return output_path
    if not isinstance(data, list):
        raise ValueError(f"Unsupported glossary format: {path}")

    normalised: Dict[str, Any] = {}
    for idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("term") or entry.get("source") or idx)
        if key in normalised:
            key = f"{key}__{idx}"
        normalised[key] = entry
    output_path.write_text(json.dumps(normalised, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-label", default="right")
    parser.add_argument("--left-instances", type=Path, required=True)
    parser.add_argument("--right-instances", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source-reference", type=Path, required=True)
    parser.add_argument("--audio-yaml", type=Path, required=True)
    parser.add_argument("--glossary", type=Path, required=True)
    parser.add_argument("--lang-code", choices=sorted(LANG_DEFAULTS), required=True)
    parser.add_argument("--stream-laal-tool", type=Path, required=True)
    parser.add_argument("--mwersegmenter-root", required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output-normalized-glossary", type=Path, default=None)
    args = parser.parse_args()

    for path in [
        args.left_instances,
        args.right_instances,
        args.reference,
        args.source_reference,
        args.audio_yaml,
        args.glossary,
        args.stream_laal_tool,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    defaults = LANG_DEFAULTS[args.lang_code]
    latency_unit = defaults["latency_unit"]
    term_lang = defaults["term_lang"]
    os.environ.setdefault("MWERSEGMENTER_ROOT", args.mwersegmenter_root)
    os.environ["PATH"] = f"{os.environ['MWERSEGMENTER_ROOT']}:{os.environ.get('PATH', '')}"

    stream_mod = _load_stream_module(args.stream_laal_tool)
    norm_glossary = args.output_normalized_glossary or args.output_tsv.with_suffix(".streamlaal_glossary.json")
    _normalise_glossary(args.glossary, norm_glossary)

    references = stream_mod.parse_references(
        str(args.reference),
        str(args.audio_yaml),
        str(args.source_reference),
    )
    left_predictions = stream_mod.parse_simuleval_instances(str(args.left_instances), latency_unit)
    right_predictions = stream_mod.parse_simuleval_instances(str(args.right_instances), latency_unit)
    left_sentences = stream_mod.resegment_instances(left_predictions, references, latency_unit)
    right_sentences = stream_mod.resegment_instances(right_predictions, references, latency_unit)
    target_terms = stream_mod.load_glossary(str(norm_glossary), term_lang)

    if len(left_sentences) != len(right_sentences):
        raise ValueError(f"sentence count mismatch: {len(left_sentences)} vs {len(right_sentences)}")

    rows: List[Dict[str, Any]] = []
    total_terms = 0
    left_correct = 0
    right_correct = 0
    changed_sentences = 0
    changed_term_hit_sentences = 0

    for idx, (left, right) in enumerate(zip(left_sentences, right_sentences)):
        source_ref = getattr(left, "source_reference", "")
        reference = left.reference
        left_pred = left.prediction
        right_pred = right.prediction
        sentence_terms = []
        sent_left = 0
        sent_right = 0
        for term_info in target_terms:
            target = term_info["target"]
            term_en = term_info.get("en", "")
            source_has = stream_mod.source_contains(source_ref, term_en) if source_ref else True
            target_has = target in reference
            if not (source_has and target_has):
                continue
            total_terms += 1
            hit_left = target in left_pred
            hit_right = target in right_pred
            sent_left += int(hit_left)
            sent_right += int(hit_right)
            left_correct += int(hit_left)
            right_correct += int(hit_right)
            sentence_terms.append(
                {
                    "term_en": term_en,
                    "target": target,
                    args.left_label: hit_left,
                    args.right_label: hit_right,
                }
            )

        equal_prediction = left_pred == right_pred
        changed = not equal_prediction
        changed_sentences += int(changed)
        changed_term_hit_sentences += int(changed and sent_left != sent_right)
        rows.append(
            {
                "sentence_index": idx,
                "equal_prediction": int(equal_prediction),
                "term_total": len(sentence_terms),
                f"{args.left_label}_term_correct": sent_left,
                f"{args.right_label}_term_correct": sent_right,
                "term_hit_delta": sent_right - sent_left,
                f"{args.left_label}_chars": len(left_pred),
                f"{args.right_label}_chars": len(right_pred),
                "source": source_ref,
                "reference": reference,
                f"{args.left_label}_prediction": left_pred,
                f"{args.right_label}_prediction": right_pred,
                "terms_json": json.dumps(sentence_terms, ensure_ascii=False),
            }
        )

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "sentence_index",
        "equal_prediction",
        "term_total",
        f"{args.left_label}_term_correct",
        f"{args.right_label}_term_correct",
        "term_hit_delta",
        f"{args.left_label}_chars",
        f"{args.right_label}_chars",
        "source",
        "reference",
        f"{args.left_label}_prediction",
        f"{args.right_label}_prediction",
        "terms_json",
    ]
    with args.output_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "left_label": args.left_label,
        "right_label": args.right_label,
        "sentence_count": len(rows),
        "changed_sentences": changed_sentences,
        "changed_sentence_rate": changed_sentences / len(rows) if rows else 0.0,
        "changed_term_hit_sentences": changed_term_hit_sentences,
        "term_total": total_terms,
        "left_term_correct": left_correct,
        "right_term_correct": right_correct,
        "left_term_acc": left_correct / total_terms if total_terms else 0.0,
        "right_term_acc": right_correct / total_terms if total_terms else 0.0,
        "term_correct_delta": right_correct - left_correct,
        "output_tsv": str(args.output_tsv),
    }
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
