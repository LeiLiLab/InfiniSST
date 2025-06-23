import os
import re
from tkinter.font import names

from datasets import load_dataset, Dataset
from tqdm import tqdm
import numpy as np
import torch
import torchaudio.functional as AF

from glossary_utils import load_clean_glossary_from_file
#from preprocess.prep_mfa import split

_term_set_for_pool = None
_alt2main_for_pool = None
_glossary_for_pool = None
_return_tensor_for_pool = True

def _process_wrapper(args):
    item, named_entities, return_tensor, phrase2desc = args
    return process_item(item, phrase2desc, return_tensor, named_entities)


def normalize(text):
    text = text.replace("<COMMA>", ",").replace("<PERIOD>", ".").replace("<QUESTIONMARK>", "?")
    text = re.sub(r"<[^>]+>", "", text)  # remove all other <...> tags
    return text.lower()


def build_phrase_desc_index(term_set, alt2main, glossary,text_field):
    phrase2desc = {}
    for phrase in term_set:
        if phrase in glossary:
            phrase2desc[phrase] = glossary[phrase][text_field]
        elif phrase in alt2main and alt2main[phrase] in glossary:
            phrase2desc[phrase] = glossary[alt2main[phrase]][text_field]
    return phrase2desc

def extract_ground_truth_terms(text, phrase2desc, named_entities):
    if not named_entities:
        return None

    # Tokenization
    tokens = re.findall(r"\b[\w']+\b", text.lower())
    named_entity_phrases = set(' '.join(re.findall(r"\b[\w']+\b", ne.lower())) for ne in named_entities)
    n = len(tokens)

    matched = []
    for i in range(n):
        for j in range(i + 1, min(i + 6, n + 1)):
            phrase = ' '.join(tokens[i:j])
            if phrase not in named_entity_phrases:
                continue  # 剪枝：只考虑 named entity 范围内的片段
            if phrase in phrase2desc:
                matched.append((phrase, i, j))

    matched.sort(key=lambda x: -(x[2] - x[1]))  # 优先选长的
    selected = []
    occupied = set()
    for phrase, start, end in matched:
        if not any(pos in occupied for pos in range(start, end)):
            desc = phrase2desc.get(phrase)
            if desc:
                selected.append((phrase, desc))
                occupied.update(range(start, end))

    filtered = [desc for phrase, desc in selected if phrase in named_entity_phrases]
    return filtered if filtered else None


import time

def safe_resample(tensor, orig_freq, new_freq, timeout=30):
    result = {}

    def run():
        try:
            result["tensor"] = AF.resample(tensor, orig_freq=orig_freq, new_freq=new_freq)
        except Exception as e:
            result["error"] = e

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        print("[TIMEOUT] Resample operation exceeded timeout")
        return None
    if "error" in result:
        print(f"[ERROR] Resample failed: {result['error']}")
        return None
    return result.get("tensor")


import threading

def safe_resample(tensor, orig_freq, new_freq, timeout=30):
    result = {}

    def run():
        try:
            result["tensor"] = AF.resample(tensor, orig_freq=orig_freq, new_freq=new_freq)
        except Exception as e:
            result["error"] = e

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        print("[TIMEOUT] Resample operation exceeded timeout")
        return None
    if "error" in result:
        print(f"[ERROR] Resample failed: {result['error']}")
        return None
    return result.get("tensor")

def extract_array_from_sample(sample):
    segment_id = sample.get("segment_id")
    save_path = f"data/audio_tensor/{segment_id}.pt"

    try:
        if os.path.exists(save_path):
            try:
                with open(save_path, "rb") as f:
                    tensor = torch.load(f, weights_only=True)
                return tensor
            except Exception as e:
                print(f"[WARNING] Failed to load cached tensor for {segment_id}: {e}")
                # fallback to reprocess

        start_time = time.time()
        if "audio" in sample and "array" in sample["audio"]:
            arr = sample["audio"]["array"]
            sr = sample["audio"].get("sampling_rate", 16000)

            arr = np.asarray(arr, dtype=np.float32)
            arr = np.clip(arr, -1.0, 1.0)

            tensor = torch.tensor(arr).unsqueeze(0)

            # if tensor.shape[-1] > 480000:  # 超过 10 秒
            #     print(f"[SKIP] Too long tensor: {segment_id} with {tensor.shape[-1]} samples")
            #     return None

            # if sr != 48000:
            #     print(f"[ERROR] Audio array has different sampling rate for {segment_id}")
            #     tensor = safe_resample(tensor, orig_freq=sr, new_freq=48000)
            #     if tensor is None:
            #         print(f"[ERROR] Resample failed or timed out for {segment_id}, sample: {sample}")
            #         # 将失败的样本记录下来
            #         with open("data/resample_blacklist.txt", "a") as f:
            #             f.write(segment_id + "\n")
            #         return None

            if tensor.shape[-1] < 24000:
                print(f"[INFO] Skipping too short tensor: {segment_id}")
                return None
            torch.save(tensor, save_path)
            return tensor

        else:
            return None
    except Exception as e:
        print(f"[ERROR] Failed to extract/save tensor for {segment_id}: {e}")
        return None


# step2 构造{text, term}二元组
def process_item(item, phrase2desc, return_tensor=True,named_entities=None):
    speech_text = item["text"]
    speech_text = normalize(speech_text)
    ground_truth_terms = extract_ground_truth_terms(speech_text, phrase2desc,named_entities)

    item["text"] = speech_text
    item["ground_truth_term"] = ground_truth_terms or []  # 保留空 term 列表
    item["has_target"] = bool(ground_truth_terms)  # True / False 标志

    # For JSON serialization: remove array, keep path
    json_safe_item = item.copy()
    if "audio" in json_safe_item and isinstance(json_safe_item["audio"], dict):
        json_safe_item["audio"] = json_safe_item["audio"].get("path")

    # resample to 48000
    # tensor = extract_array_from_sample(item)
    # if tensor is None:
    #     print(f"[ERROR] tensor None, Failed to extract audio arrays for {item['segment_id']}")
    #     return None
    # if return_tensor:
    #     item["audio_tensor"] = tensor
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

from concurrent.futures import ProcessPoolExecutor, as_completed


def load_preprocessed_samples(json_path, with_tensor=True):
    with open(json_path, "r") as f:
        samples = json.load(f)

    if with_tensor:
        for item in samples:
            seg_id = item.get("segment_id")
            pt_path = f"data/audio_tensor/{seg_id}.pt"
            if os.path.exists(pt_path):
                try:
                    item["audio_tensor"] = torch.load(pt_path)
                except Exception as e:
                    print(f"[WARNING] Failed to load tensor for {seg_id}: {e}")
                    item["audio_tensor"] = None
    return [
        item for item in samples
        if not with_tensor or (
            isinstance(item.get("audio_tensor"), torch.Tensor)
            and item["audio_tensor"].numel() > 0
            and item["audio_tensor"].shape[-1] >= 48000
        )
    ]

# 不主动返回tensor, 而是从文件中读取，等待子进程都处理完后，防止pickle错误
def handle_giga_speech_train_samples(term_set_path, alt2main_path, glossary_path,
                                     name="s", split="train", sample_limit=None,
                                     return_tensor=False, start_offset=0, text_field = "term"):
    os.makedirs("data/audio_tensor", exist_ok=True)
    term_set, alt2main, glossary = load_clean_glossary_from_file(term_set_path, alt2main_path, glossary_path)
    print(f"Total terms: {len(term_set)}, total entities: {len(glossary)}")

    gs = load_dataset(
        path="speechcolab/gigaspeech",
        name=name,
        trust_remote_code=True,
        token=os.getenv("HF_TOKEN")
    )

    total_size = len(gs[split])
    end = start_offset + sample_limit if sample_limit else total_size
    train_set = gs[split].select(range(start_offset, min(end, total_size)))

    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    print(f"slurm_cpus: {slurm_cpus}")
    # 固定使用全量 NER 文件
    # TODO fix
    named_entities_path = f"data/named_entities_{name}_{split}_None.json"
    if os.path.exists(named_entities_path):
        print(f"[INFO] Loading full cached NER from {named_entities_path}")
        with open(named_entities_path, "r", encoding="utf-8") as f:
            all_named_entities = json.load(f)
        named_entities_list = all_named_entities[start_offset: end]
    else:
        raise FileNotFoundError(
            f"[ERROR] {named_entities_path} not found. "
            f"Please run extract_ner_cache.py in a spaCy-enabled environment first."
        )

    blacklist_path = "data/resample_blacklist.txt"
    blacklist = set()
    if os.path.exists(blacklist_path):
        with open(blacklist_path, "r") as f:
            blacklist = set(line.strip() for line in f)

    # 注意：这里统一传 False，确保子进程不携带 tensor

    phrase2desc = build_phrase_desc_index(term_set, alt2main, glossary,text_field)

    args_list = []
    for sample, named_entities in zip(train_set, named_entities_list):
        segment_id = sample["segment_id"]
        if segment_id in blacklist:
            continue
        audio_path = sample.get("audio", {}).get("path")
        if audio_path and not os.path.exists(audio_path):
            print(f"[SKIP] Audio path not found: {audio_path}")
            continue
        if audio_path:
            try:
                import torchaudio
                torchaudio.load(audio_path)  # 尝试加载
            except Exception as e:
                print(f"[SKIP] Failed to load audio {audio_path}: {e}")
                continue
        args_list.append((sample, named_entities, False, phrase2desc))

    results = []
    for args in tqdm(args_list, desc="Processing"):
        result = _process_wrapper(args)
        if result is not None:
            results.append(result)

    print(f"Total items: {len(results)}")

    # 主进程完成后，手动加载：
    if return_tensor:
        for item in results:
            if item is None:
                continue
            segment_id = item.get("segment_id")
            tensor_path = f"data/audio_tensor/{segment_id}.pt"
            if os.path.exists(tensor_path):
                try:
                    item["audio_tensor"] = torch.load(tensor_path)
                except Exception as e:
                    print(f"[WARNING] Failed to load tensor for {segment_id}: {e}")
                    item["audio_tensor"] = None  # 保底

    results = [
        item for item in results
        if item is not None and
           (not return_tensor or (
                   isinstance(item.get("audio_tensor"), torch.Tensor) and
                   item["audio_tensor"].numel() > 0 and
                   item["audio_tensor"].shape[-1] >= 48000
           ))
    ]
    print(f"Total items after filter: {len(results)}")
    return results


import json

def serialize_for_json(samples):
    clean_samples = []
    for item in samples:
        item = dict(item)  # shallow copy
        if "audio_tensor" in item:
            del item["audio_tensor"]
        if "audio" in item and isinstance(item["audio"], dict):
            item["audio"] = item["audio"].get("path")  # ❗只保留路径字符串
        clean_samples.append(item)
    return clean_samples

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        '--text_field', type=str, default="term", choices=["term", "short_description"],
        help="Which field to use as input text (term: comma-split title, short_description: full description)"
    )
    args = parser.parse_args()
    term_set_path = "data/terms/term_set.txt"
    alt2main_path = "data/terms/alt2main.json"
    glossary_path = "data/terms/glossary_filtered.json"

    # You can change name="s" to other splits like "m", "l", "xl"
    samples = handle_giga_speech_train_samples(
        term_set_path=term_set_path,
        alt2main_path=alt2main_path,
        glossary_path=glossary_path,
        name=args.name,
        split=args.split,
        sample_limit=args.limit,
        return_tensor=False,
        start_offset=args.start,
        text_field = args.text_field
    )

    json_ready = serialize_for_json(samples)
    if args.name == 'dev' or args.split == 'validation':
        out_path = f"data/test_preprocessed_samples_merged.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(serialize_for_json(samples), f, indent=2, ensure_ascii=False)
    else:
        sample_path = f'{args.text_field}_preprocessed_samples' if args.text_field == 'term' else f'preprocessed_samples'
        prefix = f"data/samples/{args.name}"
        import os
        os.makedirs(prefix, exist_ok=True)
        out_path = f"{prefix}/{sample_path}_{args.start}_{args.start + (args.limit or 'end')}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(serialize_for_json(samples), f, indent=2, ensure_ascii=False)

    print("✅ test.json written successfully with", len(json_ready), "samples.")
