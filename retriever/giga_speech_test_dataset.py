import json
import random
from tqdm import tqdm
from datasets import Dataset
import os
import pickle
import re
from glossary_utils import load_and_clean_glossary

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

import spacy
nlp = spacy.load("en_core_web_sm")

def load_gigaspeech_and_filter(input_path, output_path, terms_path, max_samples=None, max_terms = None):
    with open(input_path, 'r') as f:
        raw = json.load(f)
        data = raw.get("audios")
    print(f"Total documents: {len(data)}")

    glossary = load_and_clean_glossary(terms_path,max_terms)
    term_dict = {
        item["term"].lower(): {
            "is_proper": not item["term"].islower()
        }
        for item in glossary
    }
    term_set = set(term_dict.keys())
    print(f"Total terms: {len(term_set)}")

    def normalize(text):
        return text.replace("<COMMA>", ",").replace("<PERIOD>", ".").replace("<QUESTIONMARK>", "?").lower()

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
        term_list = []
        type_list = []

        def is_stopword_like(term):
            return term in stop_words or len(term) <= 3

        def is_likely_noun_phrase(text, phrase):
            if not text.strip().endswith(('.', '?', '!')):
                text += '.'
            doc = nlp(text)
            for np in doc.noun_chunks:
                if phrase.lower() == np.text.lower():
                    return True
            phrase_doc = nlp(phrase)
            if all(token.pos_ in {"NOUN", "PROPN"} for token in phrase_doc):
                return True
            return False

        for text in batch["text"]:
            tokens = re.findall(r"\b[\w']+\b", text.lower())
            n = len(tokens)
            matched = []

            for i in range(n):
                for j in range(i + 1, min(i + 6, n + 1)):
                    phrase = ' '.join(tokens[i:j])
                    normalized = phrase.lower()
                    if normalized.startswith("the "):
                        normalized = normalized[4:]

                    if normalized in term_set:
                        if term_dict[normalized]["is_proper"]:
                            match_type = "proper"
                        elif is_stopword_like(normalized):
                            match_type = "stopword_like"
                        elif not is_likely_noun_phrase(text, normalized):
                            match_type = "not_noun_phrase"
                        else:
                            match_type = "exact"
                        matched.append((phrase, match_type, i, j))  # 包含起止位置便于后续去重

            # 贪心去重：优先保留长的、不重叠的
            matched.sort(key=lambda x: -(x[3] - x[2]))  # 按 span 长度降序
            final_terms = []
            used = set()
            for phrase, ttype, start, end in matched:
                if any(pos in used for pos in range(start, end)):
                    continue
                final_terms.append((phrase, ttype))
                used.update(range(start, end))

            term_list.append([x[0] for x in final_terms])
            type_list.append([x[1] for x in final_terms])

        return {
            "ground_truth_terms": term_list,
            "ground_truth_types": type_list,
        }

    dataset = dataset.map(extract_ground_truth_terms, batched=True, batch_size=32)
    samples = dataset.to_list()

    total_terms = 0
    ambiguous_terms = 0
    for s in samples:
        terms = s.get("ground_truth_terms", [])
        types = s.get("ground_truth_types", [])
        total_terms += len(terms)
        ambiguous_terms += sum(1 for t in types if t == "ambiguous")
    print(f"Ambiguous terms: {ambiguous_terms}/{total_terms} ({(ambiguous_terms / total_terms * 100):.2f}%)")

    samples = [s for s in samples if s.get("ground_truth_terms")]

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
    #max_terms=1000000
)