import bz2
import json
import os
import re
from multiprocessing import Pool, cpu_count
from opencc import OpenCC
import codecs
cc = OpenCC('t2s')

LABEL_PRED = "<http://www.w3.org/2000/01/rdf-schema#label>"
DESC_PRED = "<http://schema.org/description>"

label_re = re.compile(
    r'^<http://www\.wikidata\.org/entity/(Q\d+)>\s+<http://www\.w3\.org/2000/01/rdf-schema#label>\s+"([^"]+)"@([a-z]+)\s+\.$'
)
desc_re = re.compile(
    r'^<http://www\.wikidata\.org/entity/(Q\d+)>\s+<http://schema\.org/description>\s+"([^"]+)"@([a-z]+)\s+\.$'
)

TARGET_LANGS = {"zh", "de", "es", "en"}

def process_lines(lines):
    labels = {}
    descriptions = {}

    for line in lines:
        line = line.strip()
        if LABEL_PRED in line:
            m = label_re.match(line)
            if m:
                qid, label, lang = m.groups()
                if lang in TARGET_LANGS:
                    label = codecs.decode(label, 'unicode_escape')  # 把\uXXXX转成真正的中文
                    if lang == "zh":
                        label = cc.convert(label)
                    labels.setdefault(qid, {})[lang] = label
        elif DESC_PRED in line:
            m = desc_re.match(line)
            if m:
                qid, desc, lang = m.groups()
                if lang == "en":
                    descriptions[qid] = desc
    return labels, descriptions

def merge_results(results):
    merged_labels = {}
    merged_desc = {}
    for labels, descs in results:
        for qid, lang_map in labels.items():
            if qid not in merged_labels:
                merged_labels[qid] = {}
            merged_labels[qid].update(lang_map)
        merged_desc.update(descs)
    return merged_labels, merged_desc

import time

import string
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
punct_set = set(string.punctuation)

def is_valid_term(term):
    words = term.lower().split()
    if term.lower().startswith("category:"):
        return False
    if all(w in stop_words or w in punct_set for w in words):
        return False
    return True

def save_chunk(labels, descriptions, chunk_idx, output_dir):
    terms = []
    for qid, langs in labels.items():
        if "en" in langs and any(l in langs for l in ["zh", "de", "es"]):
            term_en = langs["en"]
            if not is_valid_term(term_en):
                continue
            terms.append({
                "term": term_en,
                "target_translations": {lang: label for lang, label in langs.items() if lang != "en"},
                "short_description": descriptions.get(qid, "")
            })

    chunk_path = os.path.join(output_dir, f"terms_part{chunk_idx}.json")
    with open(chunk_path, 'w', encoding='utf-8') as f_out:
        json.dump(terms, f_out, ensure_ascii=False, indent=2)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 💾 Saved {len(terms)} terms to {chunk_path}", flush=True)

def parse_rdf(filepath, output_dir, chunk_size=5000000, limit=None):
    is_bz2 = filepath.endswith('.bz2')
    open_func = bz2.open if is_bz2 else open
    os.makedirs(output_dir, exist_ok=True)

    batch = []
    total_processed = 0
    chunk_idx = 1

    with open_func(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            batch.append(line)
            total_processed += 1

            if len(batch) >= chunk_size:
                labels, descs = process_lines(batch)
                save_chunk(labels, descs, chunk_idx, output_dir)
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processed {total_processed} lines.", flush=True)
                batch = []
                chunk_idx += 1

            if limit and total_processed >= limit:
                break

        # 处理最后不足 chunk_size 的 batch
        if batch:
            labels, descs = process_lines(batch)
            save_chunk(labels, descs, chunk_idx, output_dir)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processed {total_processed} lines.", flush=True)

if __name__ == "__main__":


    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output_dir', default='split_terms')
    parser.add_argument('--chunk_size', type=int, default=5000000)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    parse_rdf(args.input, args.output_dir, chunk_size=args.chunk_size, limit=args.limit)