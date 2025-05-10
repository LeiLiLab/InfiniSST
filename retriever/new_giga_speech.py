import os
import re
import json
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset, Dataset
from tqdm import tqdm
import numpy as np
import torch
import torchaudio.functional as AF
import spacy
nlp = spacy.load("en_core_web_sm")

from glossary_utils import load_clean_glossary_from_file


def normalize(text):
    text = text.replace("<COMMA>", ",").replace("<PERIOD>", ".").replace("<QUESTIONMARK>", "?")
    text = re.sub(r"<[^>]+>", "", text)  # remove all other <...> tags
    return text.lower()

def filter_train_set(train_set, min_duration=2.0, max_duration=10.0, limit=None):
    filtered = []
    for item in train_set:
        duration = item["end_time"] - item["begin_time"]
        if not (min_duration < duration < max_duration):
            continue
        word_count = len(item["text"].split())
        # if not (3 <= word_count <= 20):
        #     continue
        doc = nlp(item["text"])
        if not any(token.pos_ in {"NOUN", "PROPN"} for token in doc):
            continue
        filtered.append(item)
        if limit and len(filtered) >= limit:
            break
    return filtered


def extract_ground_truth_terms(text, term_set):
    tokens = re.findall(r"\b[\w']+\b", text.lower())
    n = len(tokens)
    matched = []

    for i in range(n):
        for j in range(i + 1, min(i + 6, n + 1)):
            phrase = ' '.join(tokens[i:j])
            if phrase in term_set:
                matched.append((phrase, i, j))  # 记录匹配项

    # 贪心去重：优先保留长、不重叠的
    matched.sort(key=lambda x: -(x[2] - x[1]))  # 长度降序
    selected = []
    occupied = set()
    for phrase, start, end in matched:
        if not any(pos in occupied for pos in range(start, end)):
            selected.append(phrase)
            occupied.update(range(start, end))

    return selected if selected else None


def extract_array_from_sample(sample):
    try:
        if "audio" in sample and "array" in sample["audio"]:
            arr = sample["audio"]["array"]
            sr = sample["audio"].get("sampling_rate", 16000)

            # 保证为 float32 且在 [-1, 1]
            arr = np.asarray(arr, dtype=np.float32)
            arr = np.clip(arr, -1.0, 1.0)
            tensor = torch.tensor(arr).unsqueeze(0)  # shape: (1, T)

            if sr != 48000:
                tensor = AF.resample(tensor, orig_freq=sr, new_freq=48000)

            if tensor.shape[-1] < 24000:  # 少于 0.5 秒就跳过
                return None

            return tensor  # shape: (1, T)

        else:
            return None
    except Exception as e:
        print(f"[ERROR] Failed to extract audio array for {sample.get('segment_id', 'unknown')}: {e}")
        return None


# step2 构造{text, term}二元组
def process_item(item, term_set):
    speech_text = item["text"]
    speech_text = normalize(speech_text)
    ground_truth_terms = extract_ground_truth_terms(speech_text, term_set)
    if not ground_truth_terms:
        return None

    item["text"] = speech_text
    item["ground_truth_term"] = ground_truth_terms

    # For JSON serialization: remove array, keep path
    # json_safe_item = item.copy()
    # if "audio" in json_safe_item and isinstance(json_safe_item["audio"], dict):
    #     json_safe_item["audio"] = json_safe_item["audio"].get("path")

    # resample to 48000
    tensor = extract_array_from_sample(item)
    if tensor is None:
        print(f"[ERROR] tensor None, Failed to extract audio arrays for {item['segment_id']}")
        return None
    item["audio_tensor"] = tensor

    return item


def handle_giga_speech_train_samples(name="s", split="train",limit=None):
    # step1 处理glossary
    glossary = load_clean_glossary_from_file()
    term_set = set(item["term"].lower() for item in glossary)
    print(f"Total terms: {len(glossary)}")

    gs = load_dataset(
        path="speechcolab/gigaspeech",
        name=name,
        trust_remote_code=True,
        token=os.getenv("HF_TOKEN")
    )
    train_set = gs[split]

    train_set = filter_train_set(train_set,limit=limit)

    results = list(tqdm(
        ThreadPoolExecutor(max_workers=os.cpu_count()).map(
            lambda item: process_item(item, term_set), train_set
        ),
        total=len(train_set), desc="Processing"
    ))

    # 过滤掉 None
    results = [r for r in results if r is not None]

    print(f"Total items: {len(results)}")

    results = [
        item for item in results
        if item is not None and
           isinstance(item.get("audio_tensor"), torch.Tensor) and item["audio_tensor"].numel() > 0 and
           bool(item.get("ground_truth_term"))
    ]
    print(f"Total items after filter: {len(results)}")

    # Filter out items whose audio_tensor is too short for patch embedding
    results = [
        item for item in results
        if isinstance(item.get("audio_tensor"), torch.Tensor) and item["audio_tensor"].shape[-1] >= 100000
    ]
    print(f"Total items after length filter: {len(results)}")
    return results
