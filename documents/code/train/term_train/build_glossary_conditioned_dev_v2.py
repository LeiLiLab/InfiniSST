#!/usr/bin/env python3
"""Build a glossary-conditioned dev set with unseen wiki with-term chunks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Set, Tuple


SAMPLE_RATE = 16000
CHUNK_SEC = 1.92
WORD_NORMALIZE_PATTERN = re.compile(r"[^a-z0-9']+")


def normalize_word(word: str) -> str:
    word = word.strip().lower().replace("\u2019", "'")
    if word.endswith("'s"):
        word = word[:-2]
    return WORD_NORMALIZE_PATTERN.sub("", word)


def normalize_term(text: object) -> str:
    return str(text or "").strip().lower()


def term_tokens(text: str, max_tokens: int) -> Tuple[str, ...]:
    toks = tuple(tok for tok in (normalize_word(w) for w in str(text).split()) if tok)
    if not toks or len(toks) > max_tokens:
        return ()
    return toks


def iter_jsonl(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_glossary(path: str) -> Tuple[Set[str], Dict[str, dict]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    terms: Set[str] = set()
    meta: Dict[str, dict] = {}
    values: Iterable[object]
    if isinstance(data, dict):
        values = data.keys()
    else:
        values = data
    for item in values:
        if isinstance(item, dict):
            term = normalize_term(item.get("term_key") or item.get("term") or "")
            raw = dict(item)
        else:
            term = normalize_term(item)
            raw = {"term": str(item)}
        if not term:
            continue
        terms.add(term)
        meta.setdefault(term, raw)
    return terms, meta


def load_term_set(paths: Sequence[str]) -> Set[str]:
    terms: Set[str] = set()
    for path in paths:
        for row in iter_jsonl(path):
            term = normalize_term(row.get("term_key") or row.get("term") or row.get("term_text") or "")
            if term:
                terms.add(term)
    return terms


def remap_path(path: str, remaps: Sequence[Tuple[str, str]]) -> str:
    for src, dst in remaps:
        if path.startswith(src):
            return dst + path[len(src):]
    return path


def parse_remaps(items: Sequence[str]) -> List[Tuple[str, str]]:
    if len(items) % 2 != 0:
        raise ValueError("--remap arguments must be FROM TO pairs")
    return [(items[i], items[i + 1]) for i in range(0, len(items), 2)]


def select_wiki_withterm(args: argparse.Namespace) -> None:
    active_terms, active_meta = load_glossary(args.eval_glossary)
    excluded_terms = load_term_set(args.train_jsonl + args.exclude_jsonl)
    rng = random.Random(args.seed)
    remaps = parse_remaps(args.remap)
    reservoir: List[dict] = []
    eligible = 0
    stats = Counter()

    for source_path in args.candidate_jsonl:
        for row in iter_jsonl(source_path):
            stats["source_rows"] += 1
            if args.max_scan_rows > 0 and stats["source_rows"] > args.max_scan_rows:
                break
            term = normalize_term(row.get("term_key") or row.get("term") or "")
            if not term:
                stats["skip_empty_term"] += 1
                continue
            if term not in active_terms:
                stats["skip_not_active"] += 1
                continue
            if term in excluded_terms:
                stats["skip_seen_term"] += 1
                continue
            utterance = str(row.get("utterance") or row.get("chunk_src_text") or "")
            if term_tokens(term, args.max_term_tokens) not in {
                tuple(term_tokens(" ".join(utterance.split()[i:i + len(term.split())]), args.max_term_tokens))
                for i in range(max(0, len(utterance.split()) - len(term.split()) + 1))
            }:
                # Keep this as a cheap guard; MFA alignment will do the final check.
                stats["skip_term_not_literal_in_utterance"] += 1
                continue
            audio_path = str(row.get("clean_audio_path") or row.get("tts_audio_path") or "")
            audio_path = remap_path(audio_path, remaps)
            if not audio_path or not os.path.isfile(audio_path):
                stats["skip_missing_audio"] += 1
                continue
            out = {
                "term": row.get("term") or row.get("term_key"),
                "utterance": utterance,
                "variant_idx": row.get("variant_idx", 0),
                "clean_audio_path": audio_path,
                "source_jsonl": source_path,
                "source_rank": active_meta.get(term, {}).get("rank", ""),
                "source": active_meta.get(term, {}).get("source", ""),
            }
            eligible += 1
            if len(reservoir) < args.target_withterm:
                reservoir.append(out)
            else:
                j = rng.randint(0, eligible - 1)
                if j < args.target_withterm:
                    reservoir[j] = out

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for row in reservoir:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats.update(
        {
            "active_terms": len(active_terms),
            "excluded_terms": len(excluded_terms),
            "eligible": eligible,
            "selected": len(reservoir),
            "target_withterm": args.target_withterm,
        }
    )
    stats_path = args.output_jsonl.replace(".jsonl", "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(dict(stats), f, indent=2, sort_keys=True)
    print(json.dumps(dict(stats), indent=2, sort_keys=True), flush=True)
    print(f"[DONE] selected={len(reservoir)} -> {args.output_jsonl}", flush=True)


def parse_audio_field(audio_field: str) -> Tuple[str, int, int]:
    path, start, length = audio_field.rsplit(":", 2)
    return path, int(start), int(length)


def load_needed_manifest_starts(paths: Sequence[str], needed_ids: Set[str]) -> Dict[str, int]:
    starts: Dict[str, int] = {}
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                seg_id = row.get("id", "")
                if not seg_id or seg_id not in needed_ids or seg_id in starts:
                    continue
                try:
                    _, start, _ = parse_audio_field(row.get("audio", ""))
                except Exception:
                    continue
                starts[seg_id] = start
                if len(starts) == len(needed_ids):
                    return starts
    return starts


def parse_textgrid_words(path: Path) -> List[Tuple[float, float, str]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines()]
    tier_name_idx = None
    for i, line in enumerate(lines):
        if line == '"words"':
            tier_name_idx = i
            break
    if tier_name_idx is None:
        return []
    n_intervals = int(lines[tier_name_idx + 3])
    intervals: List[Tuple[float, float, str]] = []
    cursor = tier_name_idx + 4
    for _ in range(n_intervals):
        start = float(lines[cursor])
        end = float(lines[cursor + 1])
        word = lines[cursor + 2]
        if word.startswith('"') and word.endswith('"') and len(word) >= 2:
            word = word[1:-1]
        intervals.append((start, end, word))
        cursor += 3
    return intervals


def row_window_text(row: dict, textgrid_dir: str, manifest_starts: Dict[str, int]) -> str:
    text = str(row.get("chunk_src_text", "") or "")
    if text:
        return text
    seg_id = str(row.get("source_seg_id", "") or "")
    source_start = row.get("source_start_sample")
    if not seg_id or source_start is None:
        return ""
    tg_path = Path(textgrid_dir) / f"{seg_id}.TextGrid"
    if not tg_path.exists() or seg_id not in manifest_starts:
        return ""
    rel_start = (int(source_start) - manifest_starts[seg_id]) / SAMPLE_RATE
    rel_end = rel_start + CHUNK_SEC
    words = []
    for start, end, word in parse_textgrid_words(tg_path):
        mid = 0.5 * (start + end)
        if rel_start <= mid < rel_end and normalize_word(word):
            words.append(word)
    return " ".join(words)


def load_active_term_tuples(path: str, max_tokens: int) -> Set[Tuple[str, ...]]:
    active, _ = load_glossary(path)
    tuples = set()
    for term in active:
        toks = term_tokens(term, max_tokens)
        if toks:
            tuples.add(toks)
    return tuples


def active_hits(text: str, active_tuples: Set[Tuple[str, ...]], max_tokens: int) -> List[str]:
    toks = [tok for tok in (normalize_word(w) for w in str(text).split()) if tok]
    hits: Set[str] = set()
    for n in range(1, min(max_tokens, len(toks)) + 1):
        for i in range(0, len(toks) - n + 1):
            tup = tuple(toks[i:i + n])
            if tup in active_tuples:
                hits.add(" ".join(tup))
    return sorted(hits)


def combine_dev(args: argparse.Namespace) -> None:
    active_tuples = load_active_term_tuples(args.eval_glossary, args.max_term_tokens)
    base_rows = list(iter_jsonl(args.base_dev_jsonl))
    needed_seg_ids = {
        str(row.get("source_seg_id", "") or "")
        for row in base_rows
        if not (row.get("chunk_src_text") or "") and row.get("source_seg_id")
    }
    needed_seg_ids.discard("")
    manifest_starts = load_needed_manifest_starts(args.manifest_tsv, needed_seg_ids)
    stats = Counter()
    output_rows: List[dict] = []

    for row in base_rows:
        row = dict(row)
        term = normalize_term(row.get("term_key") or row.get("term") or "")
        if term:
            output_rows.append(row)
            stats["base_withterm_kept"] += 1
            continue
        text = row_window_text(row, args.textgrid_dir, manifest_starts)
        hits = active_hits(text, active_tuples, args.max_term_tokens) if text else []
        if hits:
            stats["base_noterm_dropped_active_hit"] += 1
            stats[f"drop_audio_type:{row.get('audio_type', 'unknown')}"] += 1
            continue
        row["chunk_src_text"] = text
        row["active_glossary_hits"] = []
        output_rows.append(row)
        stats["base_noterm_kept"] += 1

    for idx, row in enumerate(iter_jsonl(args.wiki_withterm_jsonl)):
        row = dict(row)
        term = normalize_term(row.get("term_key") or row.get("term") or "")
        if not term:
            stats["wiki_skip_empty_term"] += 1
            continue
        if not os.path.isfile(str(row.get("chunk_audio_path", ""))):
            stats["wiki_skip_missing_audio"] += 1
            continue
        row["term_key"] = term
        row["audio_type"] = "wiki_unseen_withterm"
        row["source_mix"] = "wiki_unseen_dev_v2"
        row["dev_v2_idx"] = idx
        output_rows.append(row)
        stats["wiki_withterm_kept"] += 1

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output_jsonl + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, args.output_jsonl)
    stats.update(
        {
            "base_rows": len(base_rows),
            "output_rows": len(output_rows),
            "active_term_tuples": len(active_tuples),
        }
    )
    with open(args.stats_json, "w", encoding="utf-8") as f:
        json.dump(dict(stats), f, indent=2, sort_keys=True)
    print(json.dumps(dict(stats), indent=2, sort_keys=True), flush=True)
    print(f"[DONE] wrote {args.output_jsonl}", flush=True)


def rebalance_dev(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    rows = list(iter_jsonl(args.input_jsonl))

    def has_term(row: dict) -> bool:
        return bool(normalize_term(row.get("term_key") or row.get("term") or ""))

    wiki_with = [
        row for row in rows
        if has_term(row) and str(row.get("audio_type", "")) == "wiki_unseen_withterm"
    ]
    other_with = [
        row for row in rows
        if has_term(row) and str(row.get("audio_type", "")) != "wiki_unseen_withterm"
    ]
    noterm = [row for row in rows if not has_term(row)]

    n_wiki = min(args.target_wiki_withterm, len(wiki_with))
    n_other = min(args.target_other_withterm, len(other_with))
    n_noterm = min(args.target_noterm, len(noterm))
    selected: List[dict] = []
    selected.extend(rng.sample(wiki_with, n_wiki))
    selected.extend(rng.sample(other_with, n_other))
    selected.extend(rng.sample(noterm, n_noterm))
    rng.shuffle(selected)

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output_jsonl + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for idx, row in enumerate(selected):
            out = dict(row)
            out["dev_v3_idx"] = idx
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    os.replace(tmp, args.output_jsonl)

    stats = Counter(str(row.get("audio_type", "unknown")) for row in selected)
    summary = {
        "input_jsonl": args.input_jsonl,
        "output_jsonl": args.output_jsonl,
        "seed": args.seed,
        "source_rows": len(rows),
        "source_wiki_withterm": len(wiki_with),
        "source_other_withterm": len(other_with),
        "source_noterm": len(noterm),
        "selected_rows": len(selected),
        "selected_wiki_withterm": n_wiki,
        "selected_other_withterm": n_other,
        "selected_noterm": n_noterm,
        "audio_type_counts": dict(stats),
    }
    with open(args.stats_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"[DONE] wrote {args.output_jsonl}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sel = sub.add_parser("select-wiki-withterm")
    p_sel.add_argument("--candidate_jsonl", action="append", required=True)
    p_sel.add_argument("--eval_glossary", required=True)
    p_sel.add_argument("--train_jsonl", action="append", default=[])
    p_sel.add_argument("--exclude_jsonl", action="append", default=[])
    p_sel.add_argument("--output_jsonl", required=True)
    p_sel.add_argument("--target_withterm", type=int, default=1800)
    p_sel.add_argument("--max_scan_rows", type=int, default=600000)
    p_sel.add_argument("--max_term_tokens", type=int, default=8)
    p_sel.add_argument("--seed", type=int, default=17)
    p_sel.add_argument("--remap", nargs="*", default=[])
    p_sel.set_defaults(func=select_wiki_withterm)

    p_combine = sub.add_parser("combine")
    p_combine.add_argument("--base_dev_jsonl", required=True)
    p_combine.add_argument("--wiki_withterm_jsonl", required=True)
    p_combine.add_argument("--eval_glossary", required=True)
    p_combine.add_argument("--output_jsonl", required=True)
    p_combine.add_argument("--stats_json", required=True)
    p_combine.add_argument("--textgrid_dir", required=True)
    p_combine.add_argument("--manifest_tsv", action="append", default=[])
    p_combine.add_argument("--max_term_tokens", type=int, default=8)
    p_combine.set_defaults(func=combine_dev)

    p_rebalance = sub.add_parser("rebalance")
    p_rebalance.add_argument("--input_jsonl", required=True)
    p_rebalance.add_argument("--output_jsonl", required=True)
    p_rebalance.add_argument("--stats_json", required=True)
    p_rebalance.add_argument("--target_wiki_withterm", type=int, default=1592)
    p_rebalance.add_argument("--target_other_withterm", type=int, default=1592)
    p_rebalance.add_argument("--target_noterm", type=int, default=3184)
    p_rebalance.add_argument("--seed", type=int, default=29)
    p_rebalance.set_defaults(func=rebalance_dev)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
