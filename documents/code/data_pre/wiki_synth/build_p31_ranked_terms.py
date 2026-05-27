#!/usr/bin/env python3
"""
Build a P31-balanced ranked term list from the full pool using
Efraimidis-Spirakis weighted reservoir sampling.

Each term's primary type = least frequent P31 type among its P31 types.
Weight w(e) = 1 / primary_type_frequency  (uniform-over-type).
Key k(e)    = log(U) / w(e)  where U ~ Uniform(0,1).
Terms are sorted by key DESCENDING, so any prefix of length N is a
valid P31-balanced sample of size N.

This avoids the problem of pure inverse-frequency sorting, which
pushes mega-types (human, taxon) to the very end. With this method,
every type contributes proportionally at every cutoff level.

Output: JSONL with a 0-based rank field, so training can simply take
the first `wiki_rank` lines.

Usage:
    python build_p31_ranked_terms.py
    python build_p31_ranked_terms.py --smoke_test 1000
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
FULL_POOL_JSONL = (
    "/mnt/gemini/data1/jiaxuanluo/wiki_synth_data/"
    "wiki_synth_terms_p31_balanced_full.jsonl"
)
INFERENCE_GLOSSARIES = [
    "/mnt/taurus/home/jiaxuanluo/InfiniSST/"
    "documents/code/data_pre/glossary_scale/wiki_glossary_medicine.json",
    "/mnt/taurus/home/jiaxuanluo/InfiniSST/"
    "documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs.json",
]
OUTPUT_PATH = (
    "/mnt/gemini/data1/jiaxuanluo/wiki_synth_data/p31_full/"
    "wiki_synth_terms_p31_ranked.jsonl"
)
# ======Configuration=====


def load_inference_terms(paths: list[str]) -> set[str]:
    terms: set[str] = set()
    for p in paths:
        assert os.path.isfile(p), f"Not found: {p}"
        with open(p) as f:
            for item in json.load(f):
                terms.add(item["term"].strip().lower())
    return terms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank wiki terms by P31 inverse frequency (rarest first)"
    )
    parser.add_argument("--full_pool", type=str, default=FULL_POOL_JSONL)
    parser.add_argument("--output", type=str, default=OUTPUT_PATH)
    parser.add_argument("--smoke_test", type=int, default=0)
    args = parser.parse_args()

    # Step 1: Load inference exclusion terms
    print("=" * 70)
    print("[Step 1] Loading inference exclusion terms...")
    blocked = load_inference_terms(INFERENCE_GLOSSARIES)
    print(f"  Blocked terms: {len(blocked):,}")

    # Step 2: Load full pool
    print(f"\n[Step 2] Loading full pool: {args.full_pool}")
    t0 = time.time()
    entities = []
    skipped_blocked = 0
    with open(args.full_pool, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.smoke_test > 0 and i >= args.smoke_test:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tk = obj.get("term_key", obj["term"].strip().lower())
            if tk in blocked:
                skipped_blocked += 1
                continue
            entities.append(obj)
    print(f"  Loaded: {len(entities):,} ({time.time()-t0:.1f}s)")
    print(f"  Skipped (inference blocked): {skipped_blocked:,}")

    # Step 3: Compute type frequencies
    print(f"\n[Step 3] Computing P31 type frequencies...")
    type_freq = Counter()
    for ent in entities:
        for qid in ent.get("p31_qids", []):
            type_freq[qid] += 1

    type_labels = {}
    for ent in entities:
        for qid, label in zip(ent.get("p31_qids", []), ent.get("p31_labels", [])):
            if qid not in type_labels and label:
                type_labels[qid] = label

    print(f"  Unique P31 types: {len(type_freq):,}")

    # Step 4: Assign primary type, compute Efraimidis-Spirakis keys, and sort
    print(f"\n[Step 4] Computing E-S keys (uniform-over-type sampling order)...")
    SEED = 42
    rng = random.Random(SEED)
    print(f"  Random seed: {SEED}")

    primary_freq_counter = Counter()
    primary_info = []
    for ent in entities:
        qids = ent.get("p31_qids", [])
        assert len(qids) > 0, f"No P31 types for {ent.get('term')}"
        primary_qid = min(qids, key=lambda q: type_freq.get(q, 0))
        primary_info.append(primary_qid)
        primary_freq_counter[primary_qid] += 1

    print(f"  Unique primary types: {len(primary_freq_counter):,}")

    ranked = []
    for i, ent in enumerate(entities):
        pqid = primary_info[i]
        pfreq = primary_freq_counter[pqid]
        plabel = type_labels.get(pqid, "")
        w = 1.0 / pfreq
        u = rng.random()
        while u == 0.0:
            u = rng.random()
        log_key = math.log(u) / w
        ranked.append((log_key, ent["term_key"], ent, pqid, plabel, pfreq))

    ranked.sort(key=lambda x: -x[0])

    # Step 5: Verify balance at milestones
    print(f"\n[Step 5] Verifying P31 balance at key cutoffs...")
    total = len(ranked)
    milestones = [100_000, 500_000, 1_000_000, 2_000_000, 3_000_000, total]
    for m in milestones:
        if m > total:
            m = total
        type_counts = Counter()
        for j in range(m):
            type_counts[ranked[j][3]] += 1
        top5 = type_counts.most_common(5)
        top5_str = ", ".join(
            f"{type_labels.get(q, q)}={c}" for q, c in top5
        )
        print(f"  Top {m:>9,}: {len(type_counts):,} types | top-5: {top5_str}")
        if m == total:
            break

    # Step 6: Write output
    print(f"\n[Step 6] Writing ranked output: {args.output}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for rank, (log_key, tk, ent, pqid, plabel, pfreq) in enumerate(ranked):
            out = {
                "rank": rank,
                "term": ent["term"],
                "term_key": tk,
                "primary_p31_freq": pfreq,
                "primary_p31_qid": pqid,
                "primary_p31_label": plabel,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 70}")
    print(f"[SUMMARY]")
    print(f"  Total ranked terms: {total:,}")
    print(f"  Method: Efraimidis-Spirakis weighted sampling order (seed={SEED})")
    print(f"  Output: {args.output}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
