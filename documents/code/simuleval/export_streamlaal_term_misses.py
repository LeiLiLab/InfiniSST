#!/usr/bin/env python3
"""Export exact-match term misses from SimulEval instances.log.

This mirrors stream_laal_term.py TERM_ACC:
- source sentence contains the English term
- reference sentence contains the target translation
- prediction does not contain the target translation

The script resegments long SimulEval instances with stream_laal_term.py and
mwerSegmenter, then writes occurrence-level misses plus a unique miss summary.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
from collections import Counter
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


def _wav_to_sample(wav: str) -> str:
    base = os.path.basename(str(wav))
    base = re.sub(r"\.wav$", "", base)
    base = re.sub(r"_v2$", "", base)
    return base


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instances-log", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--source-reference", type=Path, required=True)
    ap.add_argument("--audio-yaml", type=Path, required=True)
    ap.add_argument("--glossary", type=Path, required=True)
    ap.add_argument("--lang-code", choices=sorted(LANG_DEFAULTS), required=True)
    ap.add_argument(
        "--stream-laal-tool",
        type=Path,
        default=Path(
            "/mnt/taurus/home/jiaxuanluo/FBK-fairseq/"
            "examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py"
        ),
    )
    ap.add_argument("--mwersegmenter-root", default="/mnt/taurus/home/jiaxuanluo/mwerSegmenter")
    ap.add_argument("--latency-unit", default="")
    ap.add_argument("--term-lang", default="")
    ap.add_argument("--output-misses", type=Path, required=True)
    ap.add_argument("--output-summary", type=Path, required=True)
    ap.add_argument("--output-normalized-glossary", type=Path, default=None)
    args = ap.parse_args()

    defaults = LANG_DEFAULTS[args.lang_code]
    latency_unit = args.latency_unit or defaults["latency_unit"]
    term_lang = args.term_lang or defaults["term_lang"]

    os.environ.setdefault("MWERSEGMENTER_ROOT", args.mwersegmenter_root)
    os.environ["PATH"] = f"{os.environ['MWERSEGMENTER_ROOT']}:{os.environ.get('PATH', '')}"

    stream_mod = _load_stream_module(args.stream_laal_tool)
    norm_glossary = args.output_normalized_glossary or args.output_misses.with_suffix(".streamlaal_glossary.json")
    _normalise_glossary(args.glossary, norm_glossary)

    predictions = stream_mod.parse_simuleval_instances(str(args.instances_log), latency_unit)
    references = stream_mod.parse_references(
        str(args.reference),
        str(args.audio_yaml),
        str(args.source_reference),
    )
    resegmented = stream_mod.resegment_instances(predictions, references, latency_unit)
    target_terms = stream_mod.load_glossary(str(norm_glossary), term_lang)

    metadata: List[Dict[str, Any]] = []
    for wav, ref_sentences in references.items():
        sample_id = _wav_to_sample(wav)
        for local_idx, sent in enumerate(ref_sentences):
            metadata.append(
                {
                    "sample_id": sample_id,
                    "wav": wav,
                    "sentence_index_in_sample": local_idx,
                    "source_sentence": sent.source_content.strip(),
                    "reference": sent.content.strip(),
                }
            )
    if len(metadata) != len(resegmented):
        raise ValueError(f"metadata rows {len(metadata)} != resegmented rows {len(resegmented)}")

    misses: List[Dict[str, Any]] = []
    total = 0
    correct = 0
    for idx, (ins, meta) in enumerate(zip(resegmented, metadata)):
        ref = ins.reference
        pred = ins.prediction
        source_ref = getattr(ins, "source_reference", "")
        for term_info in target_terms:
            target = term_info["target"]
            term_en = term_info.get("en", "")
            source_has = stream_mod.source_contains(source_ref, term_en) if source_ref else True
            target_has = target in ref
            if source_has and target_has:
                total += 1
                if target in pred:
                    correct += 1
                else:
                    misses.append(
                        {
                            "lang": args.lang_code,
                            "sample_id": meta["sample_id"],
                            "global_sentence_index": idx,
                            "sentence_index_in_sample": meta["sentence_index_in_sample"],
                            "term_en": term_en,
                            "target_translation": target,
                            "source_sentence": source_ref,
                            "reference": ref,
                            "prediction": pred,
                        }
                    )

    args.output_misses.parent.mkdir(parents=True, exist_ok=True)
    miss_fields = [
        "lang",
        "sample_id",
        "global_sentence_index",
        "sentence_index_in_sample",
        "term_en",
        "target_translation",
        "source_sentence",
        "reference",
        "prediction",
    ]
    with args.output_misses.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=miss_fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(misses)

    counter = Counter((m["term_en"], m["target_translation"]) for m in misses)
    with args.output_summary.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["term_en", "target_translation", "miss_count"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for (term_en, target), count in sorted(
            counter.items(), key=lambda x: (-x[1], x[0][0].casefold(), x[0][1])
        ):
            writer.writerow({"term_en": term_en, "target_translation": target, "miss_count": count})

    result = {
        "correct": correct,
        "total": total,
        "term_acc": correct / total if total else 0.0,
        "miss_occurrences": len(misses),
        "unique_missed_term_translations": len(counter),
        "misses_tsv": str(args.output_misses),
        "summary_tsv": str(args.output_summary),
        "normalised_glossary": str(norm_glossary),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
