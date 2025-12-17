#!/usr/bin/env python3
import argparse
import bz2
import codecs
import json
import logging
import os
import time
from typing import Any, Dict, Optional, Set

from opencc import OpenCC
from tqdm import tqdm

LABEL_URI = "<http://www.w3.org/2000/01/rdf-schema#label>"
DESC_URI = "<http://schema.org/description>"
P31_URI = "<http://www.wikidata.org/prop/direct/P31>"

TARGET_LANGS = {"en", "es", "de", "zh"}
ZH_CONVERTER = OpenCC("t2s")

# instance-of classes we consider "not useful"
BAD_INSTANCE_QIDS: Set[str] = {
    # 人、名字、Wikimedia meta
    "Q5",        # human
    "Q202444",   # given name
    "Q101352",   # family name
    "Q4167410",  # Wikimedia disambiguation page
    "Q4167836",  # Wikimedia category
    "Q13406463", # Wikimedia list article
    "Q22808320", # Wikimedia human name disambiguation page
    "Q18340514", # events in a specific year or time period

    # 地理实体 / 行政区
    "Q486972",   # human settlement
    "Q515",      # city
    "Q532",      # village
    "Q3957",     # town
    "Q6256",     # country
    "Q82794",    # geographic region
    "Q618123",   # geographical object
    "Q2221906",  # district
    "Q10864048", # county
    "Q185441",   # municipality
    "Q123705",   # island
    "Q1190",     # Indian state
    "Q35657",    # U.S. state
}


class Entity:
    __slots__ = ("labels", "descriptions", "instance_of")
    def __init__(self):
        self.labels: Dict[str, str] = {}
        self.descriptions: Dict[str, str] = {}
        self.instance_of: Set[str] = set()


def get_qid_from_uri(uri: str) -> Optional[str]:
    """
    uri is like '<http://www.wikidata.org/entity/Q42>'
    """
    prefix = "<http://www.wikidata.org/entity/"
    if not uri.startswith(prefix) or not uri.endswith(">"):
        return None
    inner = uri[len(prefix):-1]
    if not inner.startswith("Q"):
        return None
    return inner


def parse_literal_line(line: str):
    """
    Parse lines like:
    <http://www.wikidata.org/entity/Q42> <...#label> "Douglas Adams"@en .
    Returns (subject_uri, literal_value, lang) or (None, None, None).
    """
    line = line.strip()
    if not line or line[0] != "<":
        return None, None, None

    s_end = line.find(">")
    if s_end == -1:
        return None, None, None
    subject_uri = line[:s_end+1]

    lit_start = line.find('"', s_end+1)
    if lit_start == -1:
        return None, None, None

    i = lit_start + 1
    lit_end = -1
    while True:
        quote_pos = line.find('"', i)
        if quote_pos == -1:
            return None, None, None
        next_char = line[quote_pos+1] if quote_pos + 1 < len(line) else ""
        if next_char in ("@", "^", " "):
            lit_end = quote_pos
            break
        i = quote_pos + 1

    literal = line[lit_start+1:lit_end]
    literal = decode_literal_escapes(literal)
    rest = line[lit_end+1:].strip()
    lang = None
    if rest.startswith("@"):
        for j, ch in enumerate(rest[1:], start=1):
            if ch in (" ", "^", "."):
                lang = rest[1:j]
                break
        else:
            lang = rest[1:]
    return subject_uri, literal, lang


def decode_literal_escapes(value: str) -> str:
    """
    Decode N-Triples style escape sequences like \\uXXXX, \\UXXXXXXXX, \\t, etc.
    """
    try:
        return codecs.decode(value, "unicode_escape")
    except Exception:
        return value


def looks_like_numeric_junk(term: str) -> bool:
    stripped = term.strip()
    if not stripped:
        return True
    has_digit = any(c.isdigit() for c in stripped)
    has_alpha = any(c.isalpha() for c in stripped)
    if has_alpha:
        return False
    return has_digit


def looks_too_weird(term: str) -> bool:
    t = term.strip()
    if len(t) < 2:
        return True
    if len(t) > 100:
        return True
    counts = {}
    for ch in t:
        counts[ch] = counts.get(ch, 0) + 1
    if counts:
        max_freq = max(counts.values())
        if max_freq / len(t) > 0.8:
            return True
    return False


def get_total_lines(bz2_path: str) -> int:
    """
    预先统计 latest-truthy.nt.bz2 的总行数，并缓存到同目录下的 .lines 文件。
    """
    cache_path = bz2_path + ".lines"
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                val = f.read().strip()
                total = int(val)
                logging.info("Loaded cached total line count: %d from %s", total, cache_path)
                return total
        except Exception:
            logging.warning("Failed to read cached line count from %s, will recount.", cache_path)

    logging.info("Counting total lines in %s (first time, may take a while)...", bz2_path)
    total = 0
    with bz2.open(bz2_path, mode="rt", encoding="utf-8", errors="ignore") as fin:
        for _ in fin:
            total += 1
            if total % 5_000_000 == 0:
                logging.info("Counted %d lines so far...", total)

    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(str(total))
    logging.info("Finished counting. Total lines: %d (cached to %s)", total, cache_path)
    return total


def load_processed_qids(output_path: str) -> Set[str]:
    """
    从已有的 JSONL output 中读出已经处理过的 id（qid），用于断点续跑。
    """
    processed: Set[str] = set()
    if not os.path.exists(output_path):
        return processed

    logging.info("Loading processed qids from existing output: %s", output_path)
    bad_lines = 0
    total_lines = 0
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                qid = obj.get("id")
                if isinstance(qid, str):
                    processed.add(qid)
            except Exception:
                bad_lines += 1
                continue
    logging.info(
        "Loaded %d processed qids from %d lines (bad lines: %d)",
        len(processed),
        total_lines,
        bad_lines,
    )
    return processed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=False,
                   default="/mnt/taurus/home/jiaxuanluo/InfiniSST/latest-truthy.nt.bz2",
                   help="Path to latest-truthy.nt.bz2")
    ap.add_argument("--output", default="/mnt/gemini/data1/jiaxuanluo/glossary/glossary_truthy.jsonl",
                   help="Path to output JSONL file")
    ap.add_argument("--max_entities", type=int, default=0,
                   help="Stop after seeing this many distinct Q-items (0 = all)")
    ap.add_argument("--max_output", type=int, default=0,
                   help="Global max glossary entries (0 = unlimited, counts existing + new)")
    ap.add_argument("--log_every", type=int, default=200000,
                   help="Emit a progress log every N lines (0 to disable)")
    ap.add_argument("--sample_lines", type=int, default=0,
                   help="Print the first N label/description triples for debugging")
    ap.add_argument("--no_progress_bar", action="store_true",
                   help="Disable tqdm progress bar (even if total line count is known)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )

    # 1) 预先统计总行数（仅用于 tqdm）
    total_lines = 0
    if not args.no_progress_bar:
        try:
            total_lines = get_total_lines(args.input)
        except Exception as e:
            logging.warning("Failed to get total line count for progress bar: %s", e)

    # 2) 读取已经写进 output 的 qid（断点续跑）
    processed_qids = load_processed_qids(args.output)
    already_written = len(processed_qids)
    logging.info("Already written (from previous runs): %d", already_written)

    if args.max_output and already_written >= args.max_output:
        logging.info(
            "max_output=%d already reached (%d existing entries), nothing to do.",
            args.max_output,
            already_written,
        )
        return

    entities: Dict[str, Entity] = {}
    seen_entities = 0
    written_this_run = 0
    last_log_time = time.time()
    sample_printed = 0

    # 3) 解析 N-Triples
    with bz2.open(args.input, mode="rt", encoding="utf-8", errors="ignore") as fin:
        if total_lines > 0 and not args.no_progress_bar:
            iterator = tqdm(fin, total=total_lines, unit="lines", desc="Parsing N-Triples")
        else:
            iterator = fin

        for line_idx, line in enumerate(iterator, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                if P31_URI in line:
                    parts = line.split(" ", 3)
                    if len(parts) < 3:
                        continue
                    subj_uri = parts[0]
                    obj_uri = parts[2]
                    subj_q = get_qid_from_uri(subj_uri)
                    obj_q = get_qid_from_uri(obj_uri)
                    if not subj_q or not obj_q:
                        continue
                    ent = entities.get(subj_q)
                    if ent is None:
                        ent = Entity()
                        entities[subj_q] = ent
                        seen_entities += 1
                    ent.instance_of.add(obj_q)

                elif LABEL_URI in line or DESC_URI in line:
                    subj_uri, literal, lang = parse_literal_line(line)
                    if not subj_uri or not literal or not lang:
                        continue
                    if lang not in TARGET_LANGS:
                        continue
                    subj_q = get_qid_from_uri(subj_uri)
                    if not subj_q:
                        continue
                    if args.sample_lines and sample_printed < args.sample_lines:
                        logging.info(
                            "Sample triple #%d [%s] qid=%s lang=%s value=%r",
                            sample_printed + 1,
                            "LABEL" if LABEL_URI in line else "DESC",
                            subj_q,
                            lang,
                            literal[:120],
                        )
                        sample_printed += 1
                    ent = entities.get(subj_q)
                    if ent is None:
                        ent = Entity()
                        entities[subj_q] = ent
                        seen_entities += 1

                    if LABEL_URI in line:
                        if lang not in ent.labels:
                            ent.labels[lang] = literal
                    elif DESC_URI in line:
                        if lang not in ent.descriptions:
                            ent.descriptions[lang] = literal

                else:
                    continue

            except Exception:
                continue

            if args.log_every and line_idx % args.log_every == 0:
                now = time.time()
                logging.info(
                    "Processed %d lines (Δ%.1fs), tracking %d entities, written_this_run=%d",
                    line_idx,
                    now - last_log_time,
                    seen_entities,
                    written_this_run,
                )
                last_log_time = now

            if args.max_entities and seen_entities >= args.max_entities:
                break

    # 4) 写出 JSONL（append 模式，跳过已经写过的 qid）
    with open(args.output, "a", encoding="utf-8") as fout:
        for qid, ent in entities.items():
            # 已经在旧文件里写过的，直接跳过（断点续跑关键）
            if qid in processed_qids:
                continue

            def has_lang(lang: str) -> bool:
                return lang in ent.labels or lang in ent.descriptions

            if not all(has_lang(lang) for lang in TARGET_LANGS):
                continue

            if any(inst in BAD_INSTANCE_QIDS for inst in ent.instance_of):
                continue

            term = ent.labels.get("en") or ent.descriptions.get("en")
            if not term:
                continue

            if looks_like_numeric_junk(term) or looks_too_weird(term):
                continue

            target_translations: Dict[str, str] = {}
            for lang in ("es", "de", "zh"):
                val = ent.labels.get(lang) or ent.descriptions.get(lang)
                if val:
                    if lang == "zh":
                        val = ZH_CONVERTER.convert(val)
                    target_translations[lang] = val

            short_desc = ent.descriptions.get("en", "")

            obj: Dict[str, Any] = {
                "id": qid,
                "term": term,
                "short_description": short_desc,
                "target_translations": target_translations,
            }
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written_this_run += 1
            processed_qids.add(qid)

            if args.max_output and (already_written + written_this_run) >= args.max_output:
                logging.info(
                    "Reached global max_output=%d (existing=%d, new=%d), stopping.",
                    args.max_output,
                    already_written,
                    written_this_run,
                )
                break

    logging.info(
        "Done. Collected entities in-memory: %d, already_existing=%d, written_this_run=%d, total_now=%d",
        len(entities),
        already_written,
        written_this_run,
        already_written + written_this_run,
    )


if __name__ == "__main__":
    main()
