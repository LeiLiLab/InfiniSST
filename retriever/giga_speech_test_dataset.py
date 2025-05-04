import json
import random
from tqdm import tqdm
from datasets import Dataset
import ahocorasick
import os
import pickle
import re
from retriever import load_and_clean_glossary


def load_gigaspeech_and_filter(input_path, output_path, terms_path, max_samples=None, max_terms = None):
    with open(input_path, 'r') as f:
        raw = json.load(f)
        data = raw.get("audios")
    print(f"Total documents: {len(data)}")

    glossary = load_and_clean_glossary(terms_path,max_terms)
    term_set = set([item["term"].lower() for item in glossary])
    print(f"Total terms: {len(term_set)}")

    def normalize(text):
        return text.replace("<COMMA>", ",").replace("<PERIOD>", ".").replace("<QUESTIONMARK>", "?").title()

    all_items = []
    for doc in tqdm(data, desc="Filtering samples"):
        segments = doc.get("segments", [])
        for item in segments:
            text = item.get("text_tn", "").strip()
            begin_time = item.get("begin_time", 0)
            end_time = item.get("end_time", 0)
            duration = end_time - begin_time

            if text and 1 <= duration <= 5:
                all_items.append({
                    "sid": item.get('sid', ''),
                    "text": normalize(text),
                    "begin_time": begin_time,
                    "end_time": end_time
                })

    if max_samples:
        all_items = random.sample(all_items, max_samples)

    dataset = Dataset.from_list(all_items)

    import re

    def extract_ground_truth_terms(batch):
        results = []
        for text in batch["text"]:
            tokens = re.findall(r'\b\w+\b', text.lower())
            n = len(tokens)
            matched = set()

            for i in range(n):
                for j in range(i+1, min(i+6, n+1)):  # 最多支持 5-gram 匹配
                    phrase = ' '.join(tokens[i:j])
                    if phrase in term_set:
                        matched.add(phrase)

            results.append(sorted(matched, key=lambda x: -len(x)))  # 按长度降序
        return {"ground_truth_terms": results}

    dataset = dataset.map(extract_ground_truth_terms, batched=True, batch_size=32)
    samples = dataset.to_list()

    samples = [s for s in samples if s.get("ground_truth_terms") and s["ground_truth_terms"]]

    print(f"Found {len(samples)} valid samples.")

    with open(output_path, 'w') as f_out:
        json.dump(samples, f_out, ensure_ascii=False, indent=2)

    print(f"Saved {len(samples)} samples to {output_path}")


# 调用
load_gigaspeech_and_filter(
    input_path="/mnt/taurus/data/siqiouyang/datasets/gigaspeech/GigaSpeech.json",
    output_path="/home/jiaxuanluo/InfiniSST/retriever/data/gigaspeech_test_samples.json",
    terms_path="/home/jiaxuanluo/InfiniSST/retriever/final_split_terms/",
    max_samples=10000,
    max_terms=1000000
)