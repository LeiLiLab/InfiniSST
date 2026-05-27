#!/usr/bin/env python3
"""
Extract English terms with P31 (instance of) types from Wikidata RDF truthy dump.

Single-pass streaming through latest-truthy.nt:
  1. Collect entity QID -> English label (rdfs:label)
  2. Collect entity QID -> P31 type QID(s) (wdt:P31)
  3. Post-process: resolve type QID labels, apply term quality filters
  4. Output JSONL: {qid, term, term_key, p31_types: ["Q5", ...], p31_labels: ["human", ...]}

This output is consumed by sample_wiki_terms_by_domain.py for inverse-frequency
sampling based on P31 type.

Usage:
    # Full run (SLURM recommended, ~1-2h for 382GB dump)
    python extract_rdf_terms_with_p31.py

    # Smoke test
    python extract_rdf_terms_with_p31.py --smoke_test 50000000

    # With bz2 input
    python extract_rdf_terms_with_p31.py --input /path/to/latest-truthy.nt.bz2
"""

from __future__ import annotations

import argparse
import codecs
import json
import os
import re
import time

# ======Configuration=====
RDF_INPUT_PATH = "/mnt/gemini/data1/jiaxuanluo/glossary/latest-truthy.nt"
OUTPUT_DIR = "/mnt/gemini/data1/jiaxuanluo/wiki_synth_data"

LABEL_URI = "http://www.w3.org/2000/01/rdf-schema#label"
P31_URI = "http://www.wikidata.org/prop/direct/P31"
ENTITY_PREFIX = "<http://www.wikidata.org/entity/"

MAX_WORDS = 5
MIN_CHARS = 2
MAX_CHARS = 80
PARENS_RE = re.compile(r"\s*\([^)]*\)")
HAS_DIGIT_RE = re.compile(r"\d")
HAS_PUNCT_RE = re.compile(r"[^a-zA-Z\s\-]")
MAX_WORD_CHARS = 25

PROGRESS_INTERVAL = 10_000_000
# ======Configuration=====


def extract_qid(uri):
    """Extract QID from '<http://www.wikidata.org/entity/Q42>'."""
    if not uri.startswith(ENTITY_PREFIX) or not uri.endswith(">"):
        return None
    inner = uri[len(ENTITY_PREFIX):-1]
    if not inner.startswith("Q"):
        return None
    return inner


def decode_escapes(value):
    """Decode N-Triples \\uXXXX escapes."""
    try:
        return codecs.decode(value, "unicode_escape")
    except Exception:
        return value


def parse_en_label(line):
    """Extract (subject_QID, English_label) from an rdfs:label triple.

    Returns (None, None) if not an English label.
    """
    lit_start = line.find('"')
    if lit_start == -1:
        return None, None

    lit_end = line.rfind('"@en')
    if lit_end <= lit_start:
        return None, None

    s_end = line.find(">")
    if s_end == -1:
        return None, None
    subj_uri = line[:s_end + 1]
    qid = extract_qid(subj_uri)
    if qid is None:
        return None, None

    raw_label = line[lit_start + 1:lit_end]
    label = decode_escapes(raw_label)
    return qid, label


def parse_p31_triple(line):
    """Extract (subject_QID, object_QID) from a P31 triple."""
    parts = line.split(" ", 3)
    if len(parts) < 3:
        return None, None
    subj_qid = extract_qid(parts[0])
    obj_qid = extract_qid(parts[2])
    return subj_qid, obj_qid


def is_valid_term(cleaned):
    if len(cleaned) < MIN_CHARS or len(cleaned) > MAX_CHARS:
        return False
    if not cleaned[0].isalpha():
        return False
    if HAS_DIGIT_RE.search(cleaned):
        return False
    if HAS_PUNCT_RE.search(cleaned):
        return False
    words = cleaned.split()
    if len(words) < 1 or len(words) > MAX_WORDS:
        return False
    if any(len(w) > MAX_WORD_CHARS for w in words):
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Extract terms with P31 types from Wikidata RDF dump"
    )
    parser.add_argument(
        "--input", type=str, default=RDF_INPUT_PATH,
        help="Path to latest-truthy.nt or .nt.bz2",
    )
    parser.add_argument("--output", type=str, default="")
    parser.add_argument(
        "--smoke_test", type=int, default=0,
        help="If >0, only read the first N lines",
    )
    args = parser.parse_args()

    assert os.path.isfile(args.input), f"RDF dump not found: {args.input}"

    output_path = args.output or os.path.join(
        OUTPUT_DIR, "wiki_rdf_terms_with_p31.jsonl"
    )

    is_bz2 = args.input.endswith(".bz2")
    if is_bz2:
        import bz2
        open_fn = lambda p: bz2.open(p, "rt", encoding="utf-8", errors="ignore")
    else:
        open_fn = lambda p: open(p, "r", encoding="utf-8", errors="ignore")

    print(f"[INFO] Input:  {args.input}")
    print(f"[INFO] Output: {output_path}")
    print(f"[INFO] Mode:   {'bz2' if is_bz2 else 'plain text'}")
    if args.smoke_test > 0:
        print(f"[SMOKE TEST] Reading only {args.smoke_test:,} lines")

    # ========== Single pass: collect labels + P31 ==========
    entity_labels = {}
    entity_p31 = {}
    lines_read = 0
    label_count = 0
    p31_count = 0
    start_time = time.time()

    EN_SUFFIX = '"@en '

    with open_fn(args.input) as f:
        for line in f:
            lines_read += 1

            if args.smoke_test > 0 and lines_read > args.smoke_test:
                break

            if P31_URI in line:
                subj_q, obj_q = parse_p31_triple(line)
                if subj_q and obj_q:
                    if subj_q not in entity_p31:
                        entity_p31[subj_q] = []
                    entity_p31[subj_q].append(obj_q)
                    p31_count += 1

            elif LABEL_URI in line and EN_SUFFIX in line:
                qid, label = parse_en_label(line)
                if qid and label:
                    if qid not in entity_labels:
                        entity_labels[qid] = label
                    label_count += 1

            if lines_read % PROGRESS_INTERVAL == 0:
                elapsed = time.time() - start_time
                rate = lines_read / elapsed if elapsed > 0 else 0
                mem_labels = len(entity_labels)
                mem_p31 = len(entity_p31)
                print(
                    f"  [{lines_read:>13,} lines] "
                    f"labels={mem_labels:,}  p31_entities={mem_p31:,}  "
                    f"rate={rate:,.0f} lines/s  ({elapsed:.0f}s)"
                )

    elapsed = time.time() - start_time
    print(f"\n[INFO] Pass complete: {lines_read:,} lines in {elapsed:.0f}s")
    print(f"[INFO] Entities with English labels: {len(entity_labels):,}")
    print(f"[INFO] Entities with P31 types: {len(entity_p31):,}")
    print(f"[INFO] Total P31 triples: {p31_count:,}")
    print(f"[INFO] Total label triples (en): {label_count:,}")

    # ========== Post-process: filter, resolve types, output ==========
    print(f"\n[INFO] Post-processing ...")
    t0 = time.time()

    type_freq = {}
    for qid, types in entity_p31.items():
        for t in types:
            type_freq[t] = type_freq.get(t, 0) + 1

    print(f"[INFO] Unique P31 type QIDs: {len(type_freq):,}")
    top_types = sorted(type_freq.items(), key=lambda x: -x[1])[:30]
    print("[INFO] Top-30 P31 types by frequency:")
    for rank, (tqid, cnt) in enumerate(top_types, 1):
        tlabel = entity_labels.get(tqid, "?")
        print(f"  {rank:3d}. {tqid:12s}  {cnt:>10,}  ({tlabel})")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    written = 0
    skipped_no_label = 0
    skipped_filter = 0
    skipped_no_p31 = 0
    seen_keys = set()
    skipped_dup = 0

    with open(output_path, "w", encoding="utf-8") as fout:
        for qid, raw_label in entity_labels.items():
            cleaned = PARENS_RE.sub("", raw_label).strip()
            if not is_valid_term(cleaned):
                skipped_filter += 1
                continue

            key = cleaned.lower()
            if key in seen_keys:
                skipped_dup += 1
                continue
            seen_keys.add(key)

            types = entity_p31.get(qid)
            if not types:
                skipped_no_p31 += 1
                continue

            type_labels = []
            for tqid in types:
                tl = entity_labels.get(tqid, "")
                type_labels.append(tl)

            obj = {
                "qid": qid,
                "term": cleaned,
                "term_key": key,
                "p31_qids": types,
                "p31_labels": type_labels,
            }
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1

    elapsed = time.time() - t0
    print(f"\n[INFO] Post-processing done ({elapsed:.1f}s)")
    print(f"  Written:          {written:,}")
    print(f"  Skipped (filter): {skipped_filter:,}")
    print(f"  Skipped (no P31): {skipped_no_p31:,}")
    print(f"  Skipped (dup):    {skipped_dup:,}")
    print(f"\n{'=' * 60}")
    print(f"Saved {written:,} terms to {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
