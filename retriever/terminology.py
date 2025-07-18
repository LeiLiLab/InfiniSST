import os
import json
import re

def is_term_valid(term, max_words):
    return term.isascii() and len(term.split()) <= max_words and len(re.findall(r"[A-Za-z]", term)) >= 3

def build_wiki_term_set(input_dir="data/final_split_terms", max_words=5, output_path="data/terms/wiki_term_set.txt"):
    term_set = set()

    # Step 1: 读取所有 JSON 文件
    for fname in os.listdir(input_dir):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(input_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            for term in data:
                term = term["term"]
                if is_term_valid(term, max_words):
                    term_set.add(term)

    # Step 2: 排序后写入
    sorted_terms = sorted(term_set)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for term in sorted_terms:
            f.write(term + "\n")

    print(f"✅ Saved {len(sorted_terms)} valid terms to {output_path}")

if __name__ == "__main__":
    build_wiki_term_set()
