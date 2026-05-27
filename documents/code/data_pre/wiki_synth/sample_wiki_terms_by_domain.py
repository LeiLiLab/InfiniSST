#!/usr/bin/env python3
"""
P31-based uniform-over-type sampling of wiki terms for retriever training.

Motivation:
  Uniform random sampling from Wikidata is dominated by mega-types
  (human ~33%, taxon ~16%, scholarly article ~10%). This causes domain
  imbalance and poor generalization.

Method:
  For each entity e with primary type t_e, the sampling weight is:

    w(e) = 1 / f(t_e)

  where f(t) = number of entities with type t. Every P31 type contributes
  an equal number of expected samples. No hyperparameters.

Filtering:
  1. Must have a short_description in glossary (non-empty, not echoing term)
     - This naturally removes non-English terms (no description available)
  2. Must have at least one P31 type
  3. Basic quality filters already applied by extract_rdf_terms_with_p31.py

Input:
  - JSONL from extract_rdf_terms_with_p31.py (terms + P31 types)
  - glossary_filtered_from_wiki.json (for short_description filtering)

Output: JSON array (--format json) or JSONL (--format jsonl), each record:
  {"term", "term_key", "short_description", "p31_qids", "p31_labels"}

Usage:
    python sample_wiki_terms_by_domain.py
    python sample_wiki_terms_by_domain.py --max_terms 1000000
    python sample_wiki_terms_by_domain.py --max_terms 4524378 --format jsonl \\
        --output /path/wiki_synth_terms_p31_balanced_full.jsonl
    python sample_wiki_terms_by_domain.py --analyze_only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter

# ======Configuration=====
INPUT_JSONL = "/mnt/gemini/data1/jiaxuanluo/wiki_synth_data/wiki_rdf_terms_with_p31.jsonl"
GLOSSARY_PATH = (
    "/mnt/taurus/home/jiaxuanluo/InfiniSST/"
    "retriever/gigaspeech/data/terms/glossary_filtered_from_wiki.json"
)
OUTPUT_DIR = "/mnt/gemini/data1/jiaxuanluo/wiki_synth_data"
DEFAULT_MAX_TERMS = 1_000_000
# Full pool size after glossary + short_description filters (see stats from prior run).
RANDOM_SEED = 42
# ======Configuration=====


def pick_primary_type(p31_qids, type_freq):
    """Pick the most specific (least frequent) P31 type as primary."""
    assert len(p31_qids) > 0
    best_qid = p31_qids[0]
    best_freq = type_freq.get(best_qid, 0)
    for qid in p31_qids[1:]:
        freq = type_freq.get(qid, 0)
        if freq < best_freq:
            best_freq = freq
            best_qid = qid
    return best_qid


def main():
    parser = argparse.ArgumentParser(
        description="P31 uniform-over-type sampling of wiki terms"
    )
    parser.add_argument("--input", type=str, default=INPUT_JSONL)
    parser.add_argument("--glossary", type=str, default=GLOSSARY_PATH)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--max_terms", type=int, default=DEFAULT_MAX_TERMS)
    parser.add_argument(
        "--full_pool", action="store_true",
        help="After filters, set --max_terms to the full entity count (no subsampling).",
    )
    parser.add_argument(
        "--format", type=str, choices=("json", "jsonl"), default="",
        help="Output format. Default: jsonl if --output ends with .jsonl, else json",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--analyze_only", action="store_true",
        help="Print type distribution without sampling",
    )
    args = parser.parse_args()

    if not args.format:
        if args.output:
            args.format = "jsonl" if args.output.endswith(".jsonl") else "json"
        elif args.full_pool:
            args.format = "jsonl"
        else:
            args.format = "json"

    suffix = ".jsonl" if args.format == "jsonl" else ".json"
    if args.full_pool and not args.output:
        default_name = f"wiki_synth_terms_p31_balanced_full{suffix}"
    else:
        stem_k = args.max_terms // 1000
        default_name = f"wiki_synth_terms_p31_balanced_{stem_k}k{suffix}"
    output_path = args.output or os.path.join(OUTPUT_DIR, default_name)

    # ========== Step 1: Load glossary for short_description ==========
    print("=" * 70)
    print(f"[Step 1] Loading glossary from {args.glossary} ...")
    assert os.path.isfile(args.glossary), f"Glossary not found: {args.glossary}"

    t0 = time.time()
    with open(args.glossary, "r", encoding="utf-8") as f:
        glossary = json.load(f)
    print(f"  Loaded {len(glossary):,} entries ({time.time()-t0:.1f}s)\n")

    # ========== Step 2: Load P31 JSONL + filter by description ==========
    print("=" * 70)
    print(f"[Step 2] Loading P31 terms + filtering by short_description ...")
    assert os.path.isfile(args.input), f"Input not found: {args.input}"

    entities = []
    no_glossary = 0
    empty_desc = 0
    echo_desc = 0
    t0 = time.time()

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            key = obj["term_key"]
            entry = glossary.get(key)
            if entry is None:
                no_glossary += 1
                continue

            desc = entry.get("short_description", "")
            if not desc.strip():
                empty_desc += 1
                continue

            desc_stripped = desc.lower().replace(key, "").replace(",", "").strip()
            if not desc_stripped:
                echo_desc += 1
                continue

            obj["short_description"] = desc
            entities.append(obj)

    del glossary

    elapsed = time.time() - t0
    print(f"  Loaded {len(entities):,} entities ({elapsed:.1f}s)")
    print(f"  Filtered - no glossary entry: {no_glossary:,}")
    print(f"  Filtered - empty description: {empty_desc:,}")
    print(f"  Filtered - description echoes term: {echo_desc:,}")
    print()

    if args.full_pool:
        args.max_terms = len(entities)
        print("=" * 70)
        print(f"[full_pool] Sampling {args.max_terms:,} terms (all entities after filters).")
        print("=" * 70)
        print()

    # ========== Step 3: Type frequencies ==========
    print("=" * 70)
    print("[Step 3] Computing P31 type frequencies ...")

    type_freq = Counter()
    for ent in entities:
        for tqid in ent["p31_qids"]:
            type_freq[tqid] += 1

    type_labels = {}
    for ent in entities:
        for qid, label in zip(ent["p31_qids"], ent.get("p31_labels", [])):
            if qid not in type_labels and label:
                type_labels[qid] = label

    print(f"  Unique P31 types: {len(type_freq):,}")
    print(f"\n  Top-30 P31 types by entity count:")
    for rank, (tqid, cnt) in enumerate(type_freq.most_common(30), 1):
        tlabel = type_labels.get(tqid, "?")
        pct = 100.0 * cnt / len(entities)
        print(f"    {rank:3d}. {tqid:12s}  {cnt:>10,}  ({pct:5.1f}%)  {tlabel}")

    # ========== Step 4: Assign primary types + weights ==========
    print(f"\n{'=' * 70}")
    print("[Step 4] Computing inverse-frequency weights (uniform over types) ...")

    primary_types = []
    for ent in entities:
        pt = pick_primary_type(ent["p31_qids"], type_freq)
        primary_types.append(pt)

    primary_freq = Counter(primary_types)
    print(f"  Unique primary types: {len(primary_freq):,}")

    weights = []
    for pt in primary_types:
        f_t = primary_freq[pt]
        assert f_t > 0
        weights.append(1.0 / f_t)

    print(f"\n  Effect on top-5 mega-types (each type -> equal contribution):")
    for tqid, cnt in primary_freq.most_common(5):
        tlabel = type_labels.get(tqid, "?")
        print(f"    {tqid:12s}  count={cnt:>10,}  w=1/{cnt}  effective=1.0  {tlabel}")

    if args.analyze_only:
        print("\n[--analyze_only] Exiting without sampling.")
        return

    # ========== Step 5: Efraimidis-Spirakis sampling ==========
    print(f"\n{'=' * 70}")
    print(f"[Step 5] Efraimidis-Spirakis sampling (target={args.max_terms:,}) ...")

    n = len(entities)
    if args.max_terms > n:
        raise RuntimeError(
            f"Only {n:,} entities available after filters; "
            f"requested {args.max_terms:,}. Re-run extract or lower --max_terms."
        )

    rng = random.Random(args.seed)
    t0 = time.time()

    # Efraimidis-Spirakis: key_i = u_i^(1/w_i), select top-k by key.
    # Use log-space to avoid float underflow when 1/w is large:
    #   log(key_i) = (1/w_i) * log(u_i)
    # Sort descending by log(key).
    log_keys = []
    for i, w in enumerate(weights):
        u = rng.random()
        while u == 0.0:
            u = rng.random()
        log_key = math.log(u) / w
        log_keys.append((log_key, i))

    log_keys.sort(reverse=True)
    selected_indices = [idx for _, idx in log_keys[:args.max_terms]]

    elapsed = time.time() - t0
    print(f"  Sampling done ({elapsed:.1f}s)")

    sampled_primary = Counter()
    for idx in selected_indices:
        sampled_primary[primary_types[idx]] += 1

    print(f"\n  Sampled type distribution (top-20):")
    for rank, (tqid, cnt) in enumerate(sampled_primary.most_common(20), 1):
        orig = primary_freq[tqid]
        tlabel = type_labels.get(tqid, "?")
        pct = 100.0 * cnt / args.max_terms
        print(
            f"    {rank:3d}. {tqid:12s}  sampled={cnt:>8,} ({pct:5.1f}%)  "
            f"original={orig:>10,}  {tlabel}"
        )

    print(f"\n  Unique types in sample: {len(sampled_primary):,} / {len(primary_freq):,}")

    # ========== Write output ==========
    sampled_terms = []
    for idx in selected_indices:
        ent = entities[idx]
        sampled_terms.append({
            "term": ent["term"],
            "term_key": ent["term_key"],
            "short_description": ent["short_description"],
            "p31_qids": ent["p31_qids"],
            "p31_labels": ent.get("p31_labels", []),
        })

    sampled_terms.sort(key=lambda x: x["term_key"])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if args.format == "jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for row in sampled_terms:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sampled_terms, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Saved {len(sampled_terms):,} terms to {output_path} ({args.format})")
    print(f"{'=' * 70}")

    if output_path.endswith(".jsonl"):
        stats_path = output_path[:-6] + "_stats.json"
    else:
        stats_path = output_path.replace(".json", "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_sampled": len(sampled_terms),
                "total_available_after_filter": len(entities),
                "filtered_no_glossary": no_glossary,
                "filtered_empty_desc": empty_desc,
                "filtered_echo_desc": echo_desc,
                "method": "inverse_frequency_uniform_over_p31_types",
                "unique_types_in_pool": len(primary_freq),
                "unique_types_sampled": len(sampled_primary),
                "top_30_sampled_types": [
                    {
                        "qid": tqid,
                        "label": type_labels.get(tqid, "?"),
                        "sampled_count": cnt,
                        "original_count": primary_freq[tqid],
                    }
                    for tqid, cnt in sampled_primary.most_common(30)
                ],
            },
            f, indent=2,
        )
    print(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
