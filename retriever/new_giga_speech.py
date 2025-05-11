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


def extract_ground_truth_terms(text, term_set,alt2main ,glossary):
    tokens = re.findall(r"\b[\w']+\b", text.lower())
    n = len(tokens)
    matched = []

    for i in range(n):
        for j in range(i + 1, min(i + 6, n + 1)):
            phrase = ' '.join(tokens[i:j])
            if phrase in term_set:
                matched.append((phrase, i, j))  # 记录匹配项

    matched.sort(key=lambda x: -(x[2] - x[1]))  # 长度降序
    selected = []
    occupied = set()
    for phrase, start, end in matched:
        if not any(pos in occupied for pos in range(start, end)):
            # 优先查 glossary，其次查 alt2main 再跳转
            if phrase in glossary:
                desc = glossary[phrase]["short_description"]
            elif phrase in alt2main and alt2main[phrase] in glossary:
                desc = glossary[alt2main[phrase]]["short_description"]
            else:
                print(f"error, {phrase} not in glossary")
                desc = phrase  # fallback 到原 phrase（避免报错）
            selected.append(desc)
            occupied.update(range(start, end))

    return selected if selected else None


def extract_array_from_sample(sample):
    segment_id = sample.get("segment_id")
    save_path = f"data/audio_tensor/{segment_id}.pt"

    if os.path.exists(save_path):
        try:
            tensor = torch.load(save_path)
            return tensor
        except Exception as e:
            print(f"[WARNING] Failed to load cached tensor for {segment_id}: {e}")
            # fallback to reprocessing below

    try:
        if "audio" in sample and "array" in sample["audio"]:
            arr = sample["audio"]["array"]
            sr = sample["audio"].get("sampling_rate", 16000)

            arr = np.asarray(arr, dtype=np.float32)
            arr = np.clip(arr, -1.0, 1.0)
            tensor = torch.tensor(arr).unsqueeze(0)

            if sr != 48000:
                tensor = AF.resample(tensor, orig_freq=sr, new_freq=48000)

            if tensor.shape[-1] < 24000:
                return None

            # 保存 tensor
            torch.save(tensor, save_path)
            return tensor

        else:
            return None
    except Exception as e:
        print(f"[ERROR] Failed to extract/save tensor for {segment_id}: {e}")
        return None


# step2 构造{text, term}二元组
def process_item(item, term_set, alt2main ,glossary):
    speech_text = item["text"]
    speech_text = normalize(speech_text)
    ground_truth_terms = extract_ground_truth_terms(speech_text, term_set, alt2main ,glossary)
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


# term_set, alt2main, glossary = process_named_entities(
#         input_path=args.input,
#         max_words_length=args.max_words_length,
#         max_workers=args.max_workers
#     )
#
#     # 可选：保存调试信息
#     with open("data/alt2main.json", "w", encoding="utf-8") as f:
#         json.dump(alt2main, f, indent=2, ensure_ascii=False)
#     with open("data/glossary_filtered.json", "w", encoding="utf-8") as f:
#         json.dump(glossary, f, indent=2, ensure_ascii=False)

from multiprocessing import Pool

def handle_giga_speech_train_samples(term_set_path, alt2main_path, glossary_path, name="s", split="train", sample_limit=None):
    os.makedirs("data/audio_tensor", exist_ok=True)
    # step1 处理 glossary
    term_set, alt2main, glossary = load_clean_glossary_from_file(term_set_path, alt2main_path, glossary_path)
    print(f"Total terms: {len(term_set)}, total entities: {len(glossary)}")

    gs = load_dataset(
        path="speechcolab/gigaspeech",
        name=name,
        trust_remote_code=True,
        token=os.getenv("HF_TOKEN")
    )
    train_set = filter_train_set(gs[split], limit=sample_limit)

    # 为多进程构造全局变量，避免 pickle 问题
    global _term_set_for_pool, _alt2main_for_pool, _glossary_for_pool
    _term_set_for_pool = term_set
    _alt2main_for_pool = alt2main
    _glossary_for_pool = glossary

    def _process_wrapper(item):
        return process_item(item, _term_set_for_pool, _alt2main_for_pool, _glossary_for_pool)

    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    with Pool(processes=slurm_cpus) as pool:
        results = list(tqdm(pool.imap(_process_wrapper, train_set), total=len(train_set), desc="Processing"))

    print(f"Total items: {len(results)}")

    # 后处理过滤
    results = [
        item for item in results
        if item is not None and
           isinstance(item.get("audio_tensor"), torch.Tensor) and item["audio_tensor"].numel() > 0
           and item["audio_tensor"].shape[-1] >= 100000
           and bool(item.get("ground_truth_term"))
    ]
    print(f"Total items after filter: {len(results)}")
    # return samples
    return results


import json

def serialize_for_json(samples):
    """
    Remove non-serializable fields like 'audio_tensor'.
    """
    clean_samples = []
    for item in samples:
        item = dict(item)  # shallow copy
        if "audio_tensor" in item:
            del item["audio_tensor"]
        clean_samples.append(item)
    return clean_samples

if __name__ == "__main__":
    term_set_path = "data/terms/term_set.txt"
    alt2main_path = "data/terms/alt2main.json"
    glossary_path = "data/terms/glossary_filtered.json"

    # You can change name="s" to other splits like "m", "l", "xl"
    samples = handle_giga_speech_train_samples(
        term_set_path=term_set_path,
        alt2main_path=alt2main_path,
        glossary_path=glossary_path,
        name="s",
        split="train",
        sample_limit=10000,  # limit number for quick testing
    )

    json_ready = serialize_for_json(samples)

    with open("test.json", "w", encoding="utf-8") as f:
        json.dump(json_ready, f, indent=2, ensure_ascii=False)

    print("✅ test.json written successfully with", len(json_ready), "samples.")
