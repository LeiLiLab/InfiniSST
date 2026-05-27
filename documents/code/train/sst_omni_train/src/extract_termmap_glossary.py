#!/usr/bin/env python3
"""Extract a target-language glossary from Speech LLM JSONL term_map entries."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{lineno}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object on {path}:{lineno}")
            yield lineno, obj


def _parse_term_map(content: str) -> List[Tuple[str, str]]:
    content = str(content or "")
    marker = "term_map:"
    idx = content.find(marker)
    if idx < 0 or "term_map:NONE" in content:
        return []
    body = content[idx + len(marker) :].strip()
    out: List[Tuple[str, str]] = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        term, translation = line.split("=", 1)
        term = term.strip()
        translation = translation.strip()
        if term and translation:
            out.append((term, translation))
    return out


def _term_key(term: str) -> str:
    return " ".join(str(term or "").casefold().split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--stats-json", type=Path, required=True)
    parser.add_argument("--lang-code", choices=["zh", "de", "ja"], required=True)
    parser.add_argument("--max-terms", type=int, default=100000)
    parser.add_argument("--min-source-chars", type=int, default=2)
    parser.add_argument("--min-target-chars", type=int, default=1)
    args = parser.parse_args()

    if not args.input_jsonl.is_file():
        raise FileNotFoundError(args.input_jsonl)
    if args.output_json.exists():
        raise FileExistsError(args.output_json)
    if args.stats_json.exists():
        raise FileExistsError(args.stats_json)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.parent.mkdir(parents=True, exist_ok=True)

    pair_counts: Counter[Tuple[str, str]] = Counter()
    first_surface: Dict[Tuple[str, str], Tuple[str, str]] = {}
    by_term: dict[str, Counter[str]] = defaultdict(Counter)
    stats: Counter[str] = Counter()

    for _lineno, obj in _iter_jsonl(args.input_jsonl):
        stats["rows"] += 1
        messages = obj.get("messages")
        if not isinstance(messages, list):
            raise ValueError("Missing messages list")
        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = str(msg.get("content") or "")
            if not content.startswith("<audio>"):
                continue
            stats["chunks"] += 1
            entries = _parse_term_map(content)
            stats["termmap_entries"] += len(entries)
            if entries:
                stats["chunks_with_termmap"] += 1
            for term, translation in entries:
                if len("".join(term.split())) < args.min_source_chars:
                    stats["skipped_short_source"] += 1
                    continue
                if len("".join(translation.split())) < args.min_target_chars:
                    stats["skipped_short_target"] += 1
                    continue
                key = _term_key(term)
                if not key:
                    stats["skipped_empty_key"] += 1
                    continue
                pair = (key, translation)
                pair_counts[pair] += 1
                by_term[key][translation] += 1
                first_surface.setdefault(pair, (term, translation))

    ranked_terms = []
    for key, translations in by_term.items():
        translation, count = translations.most_common(1)[0]
        surface_term, surface_translation = first_surface[(key, translation)]
        ranked_terms.append(
            {
                "term": surface_term,
                "term_key": key,
                "translation": surface_translation,
                args.lang_code: surface_translation,
                "target_translations": {args.lang_code: surface_translation},
                "source": "speech_llm_train_termmap",
                "count": int(count),
                "num_translations_for_term": len(translations),
            }
        )
    ranked_terms.sort(key=lambda x: (-int(x["count"]), str(x["term_key"])))
    if args.max_terms > 0:
        ranked_terms = ranked_terms[: args.max_terms]
    if not ranked_terms:
        raise ValueError(f"No glossary entries extracted from {args.input_jsonl}")

    stats_dict: Dict[str, Any] = dict(stats)
    stats_dict.update(
        {
            "input_jsonl": str(args.input_jsonl),
            "output_json": str(args.output_json),
            "lang_code": args.lang_code,
            "max_terms": args.max_terms,
            "unique_terms_before_cap": len(by_term),
            "output_terms": len(ranked_terms),
            "ambiguous_terms": sum(1 for x in by_term.values() if len(x) > 1),
            "avg_entries_per_chunk": stats["termmap_entries"] / stats["chunks"] if stats["chunks"] else 0.0,
            "termmap_chunk_rate": stats["chunks_with_termmap"] / stats["chunks"] if stats["chunks"] else 0.0,
        }
    )

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(args.output_json.parent),
        prefix=args.output_json.name + ".",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
        json.dump(ranked_terms, tmp, ensure_ascii=False, indent=2)
        tmp.write("\n")
    tmp_path.replace(args.output_json)
    args.stats_json.write_text(
        json.dumps(stats_dict, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats_dict, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
