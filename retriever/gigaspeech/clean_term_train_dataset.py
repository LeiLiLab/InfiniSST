#!/usr/bin/env python3
"""
Clean term_train_dataset.jsonl:

1) Repair truncated hyphenated terms like "-limits" by searching chunk_src_text for a
   full hyphenated compound (e.g., "off-limits") and replacing the term.
2) Drop terms that start with a non-alphanumeric symbol other than hyphen/dash.
3) Count terms containing interjections (yeah/oh/yes) and print 10 samples.
4) Count terms containing stopwords (provided list) and print 10 samples.

All logs/messages are in English.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


INTERJECTIONS = {"yeah", "oh", "yes"}
STOPWORDS = {
    "the",
    "a",
    "an",
    "this",
    "that",
    "these",
    "those",
    "my",
    "your",
    "his",
    "her",
    "its",
    "our",
    "their",
    "it",
    "they",
    "them",
    "who",
    "whom",
    "whose",
}

_DASH_CHARS = "-–—‑‐‒"
_DASH_TRANS = str.maketrans({c: "-" for c in _DASH_CHARS})


def _safe_json_loads(line: str) -> Optional[Dict[str, Any]]:
    line = (line or "").strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
        if isinstance(obj, dict):
            return obj
        return None
    except Exception:
        return None


def _starts_with_bad_symbol(term: str) -> bool:
    """
    Return True if term starts with a symbol that is NOT a hyphen/dash and NOT alphanumeric.
    Example to drop: ', black macho'
    """
    if not term:
        return False
    first = term[0]
    if first.isalnum():
        return False
    if first in {"-", "–", "—"}:
        return False
    return True


def _repair_truncated_hyphen_term(term: str, chunk_src_text: str) -> Tuple[str, bool]:
    """
    If term looks like a truncated hyphen suffix (e.g., "-limits"), try to find a full
    hyphenated compound in chunk_src_text and return the repaired term.
    """
    t = (term or "").strip()
    if not t:
        return term, False

    # Only attempt for leading dash terms like "-limits"
    if not (t.startswith("-") or t.startswith("–") or t.startswith("—")):
        return term, False

    suffix = t.lstrip("-–—").strip()
    if len(suffix) < 2:
        return term, False

    src = (chunk_src_text or "").strip()
    if not src:
        return term, False
    # Normalize all dash variants to ASCII '-' to make matching robust.
    src_n = src.translate(_DASH_TRANS)

    # Robust approach:
    # 1) Extract hyphenated compounds from src: e.g., "off-limits", "up-side-down", "re-formed".
    # 2) Pick the first one that ends with "-<suffix>".
    hyphen_pat = re.compile(r"\b[A-Za-z]{2,}(?:-[A-Za-z]{2,})+\b")
    suffix_l = suffix.lower()
    for m in hyphen_pat.finditer(src_n):
        tok = m.group(0).strip()
        tl = tok.lower()
        if tl.endswith("-" + suffix_l):
            repaired = tok
            break
    else:
        # Also handle tokenization with spaces around dash: "off -limits" / "off - limits"
        spaced = re.compile(
            rf"\b([A-Za-z]{{2,}}(?:\s*-\s*[A-Za-z]{{2,}})+)(?![A-Za-z])",
            re.IGNORECASE,
        )
        for m in spaced.finditer(src_n):
            tok = re.sub(r"\\s+", " ", m.group(1)).replace(" -", "-").replace("- ", "-").strip()
            tl = tok.lower()
            if tl.endswith("-" + suffix_l):
                repaired = tok
                break
        else:
            return term, False

    # Normalize to lowercase for consistency with later training code paths.
    repaired_norm = repaired.lower()
    return repaired_norm, True


def _contains_any_word(term: str, words: Iterable[str]) -> bool:
    if not term:
        return False
    t = term.lower()
    for w in words:
        if re.search(rf"\b{re.escape(w)}\b", t):
            return True
    return False


@dataclass
class Stats:
    total_lines: int = 0
    parsed_lines: int = 0
    kept_lines: int = 0
    dropped_bad_symbol: int = 0
    dropped_unrepaired_hyphen: int = 0
    dropped_interjection: int = 0
    dropped_stopword: int = 0
    repaired_hyphen: int = 0
    unrepaired_hyphen: int = 0
    interjection_terms: int = 0
    stopword_terms: int = 0


def _format_sample(obj: Dict[str, Any]) -> str:
    keys = [
        "term",
        "translation",
        "chunk_src_text",
        "chunk_tgt_text",
        "chunk_audio_path",
        "utter_id",
        "chunk_idx",
        "global_idx",
        "line_idx",
    ]
    slim = {k: obj.get(k) for k in keys if k in obj}
    return json.dumps(slim, ensure_ascii=False)


def clean_file(input_path: Path, output_path: Path, max_samples: int = 10) -> None:
    stats = Stats()
    interj_samples: List[str] = []
    stop_samples: List[str] = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            stats.total_lines += 1
            obj = _safe_json_loads(line)
            if obj is None:
                continue
            stats.parsed_lines += 1

            term = str(obj.get("term", "")).strip()
            if not term:
                continue

            # 2) Drop terms that start with a bad symbol (not '-' and not alnum)
            if _starts_with_bad_symbol(term):
                stats.dropped_bad_symbol += 1
                continue

            # 1) Repair truncated hyphen terms like "-limits"
            repaired = False
            if term.startswith(("-", "–", "—")):
                new_term, repaired = _repair_truncated_hyphen_term(term, str(obj.get("chunk_src_text", "")))
                if repaired:
                    stats.repaired_hyphen += 1
                    obj["term"] = new_term
                    term = new_term
                else:
                    stats.unrepaired_hyphen += 1
                    # v2 behavior: drop unrepaired hyphen-leading terms
                    stats.dropped_unrepaired_hyphen += 1
                    continue

            # 3) Interjection stats
            if _contains_any_word(term, INTERJECTIONS):
                stats.interjection_terms += 1
                if len(interj_samples) < max_samples:
                    interj_samples.append(_format_sample(obj))
                # v2 behavior: drop any term containing interjections
                stats.dropped_interjection += 1
                continue

            # 4) Stopword stats
            if _contains_any_word(term, STOPWORDS):
                stats.stopword_terms += 1
                if len(stop_samples) < max_samples:
                    stop_samples.append(_format_sample(obj))
                # v2 behavior: drop any term containing stopwords
                stats.dropped_stopword += 1
                continue

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            stats.kept_lines += 1

    print("=== Summary ===")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"total_lines={stats.total_lines}")
    print(f"parsed_lines={stats.parsed_lines}")
    print(f"kept_lines={stats.kept_lines}")
    print(f"dropped_bad_symbol={stats.dropped_bad_symbol}")
    print(f"dropped_unrepaired_hyphen={stats.dropped_unrepaired_hyphen}")
    print(f"dropped_interjection={stats.dropped_interjection}")
    print(f"dropped_stopword={stats.dropped_stopword}")
    print(f"repaired_hyphen={stats.repaired_hyphen}")
    print(f"unrepaired_hyphen={stats.unrepaired_hyphen}")
    print(f"interjection_terms={stats.interjection_terms}")
    print(f"stopword_terms={stats.stopword_terms}")

    print("\n=== Interjection Samples (up to 10) ===")
    for s in interj_samples:
        print(s)

    print("\n=== Stopword Samples (up to 10) ===")
    for s in stop_samples:
        print(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSONL path")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument("--max_samples", type=int, default=10, help="Number of samples to print for each category")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    clean_file(in_path, out_path, max_samples=int(args.max_samples))


if __name__ == "__main__":
    main()


