from collections import OrderedDict
import signal
import time

from sentence_transformers import SentenceTransformer
import faiss
import json
import torchaudio
import numpy as np
from typing import List, Dict, Tuple
#from clap import CLAP_Model  # Assuming you have a CLAP text encoder ready
from transformers import ClapModel, ClapProcessor
from concurrent.futures import ProcessPoolExecutor
from audio_cache import AudioCache
from audio_utils import get_audio_full_path

# ---------- CONFIG ----------
TOP_K = 5


import functools
print = functools.partial(print, flush=True)

# ---------- LOAD GLOSSARY ----------
def load_glossary(glossary_path: str) -> List[Dict]:
    with open(glossary_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------- BUILD INDEX ----------
class Retriever:
    def __init__(self, model_name: str = "laion/clap-htsat-fused", fallback_mode: str = "safe", device: str = "cpu", max_gpus: int = None):
        self.fallback_mode = fallback_mode
        self.device = device
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name).to(device)
        self.index = None
        self.term_list = []
        self.max_gpus = max_gpus

    def encode_texts(self, texts: List[str], batch_size: int = 512) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                embeddings = self.model.get_text_features(**inputs)
            all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings, dim=0).numpy()

    def build_index(self, glossary: List[Dict]):
        # 🔁 Step 1: 去重 glossary（忽略大小写）
        if self.fallback_mode == "safe":
            unique_items = OrderedDict()
            for item in glossary:
                key = item['term'].strip().lower()
                if key not in unique_items:
                    unique_items[key] = item
            glossary = list(unique_items.values())

        # 🔍 Step 2: 用 term-only 构建 embedding

        if self.fallback_mode == "safe":
            texts = [item["term"] for item in glossary]
        elif self.fallback_mode == "flexible":
            texts = [
                f"{item['term']}  ({item['summary']})" if item["summary"]
                else item["term"]
                for item in glossary
            ]
        else:
            texts = [item["term"] for item in glossary]

        embeddings = self.encode_texts(texts)
        print(f"[DEBUG] encode_texts Embeddings: {embeddings.shape}")

        # ✅ Step 3: 构建 FAISS index
        self.term_list = [item["term"] for item in glossary]
        dim = embeddings.shape[1]

        # ✅ 构建 CPU index
        cpu_index = faiss.IndexFlatL2(dim)

        # 分批添加 embedding 到 CPU index（降低峰值内存）
        batch_size = 10000
        for i in range(0, len(embeddings), batch_size):
            cpu_index.add(embeddings[i:i + batch_size])

        # ✅ 多卡 GPU 分布索引（手动分 shard 到所有可用 GPU 上）
        ngpu = self.max_gpus or faiss.get_num_gpus()
        print(f"[INFO] FAISS using {ngpu} GPUs for indexing (manual sharding/merging)")
        co = faiss.GpuClonerOptions()
        co.shard = False
        shard_index = faiss.IndexShards(dim, True, True)
        per_gpu = (len(embeddings) + ngpu - 1) // ngpu

        for i in range(ngpu):
            start = i * per_gpu
            end = min((i + 1) * per_gpu, len(embeddings))
            if start >= end:
                continue
            sub_embeds = embeddings[start:end]
            print(f"[DEBUG] Shard {i}: sub_embeds shape = {sub_embeds.shape}")

            res = faiss.StandardGpuResources()
            sub_cpu_index = faiss.IndexFlatL2(dim)
            sub_cpu_index.add(sub_embeds)

            try:
                gpu_sub_index = faiss.index_cpu_to_gpu(res, i, sub_cpu_index, co)
                shard_index.add_shard(gpu_sub_index)
            except Exception as e:
                print(f"[ERROR] Failed to build GPU shard {i}, range ({start}-{end}): {e}")

        self.index = shard_index

    def query(self, text: str, top_k: int = TOP_K) -> List[str]:
        embedding = self.encode_texts([text])  # 这里是 (1, hidden_dim) 的 GPU tensor
        D, I = self.index.search(embedding, top_k)
        return [self.term_list[i] for i in I[0]]

    def save_index(self):
        index_path = f"retriever_{self.fallback_mode}.index"
        terms_path = f"term_list_{self.fallback_mode}.json"
        # 🔥 提取 GPU shard 中的所有向量，构建统一 CPU index 保存
        dim = self.index.d
        xb = []
        # Fix: IndexShards does not have a .shards attribute; use .at(i) and .count()
        for shard in [self.index.at(i) for i in range(self.index.count())]:
            try:
                xb_shard = shard.reconstruct_n(0, shard.ntotal)
                xb.append(xb_shard)
            except Exception as e:
                print(f"[ERROR] Failed to reconstruct shard: {e}")
        if not xb:
            raise RuntimeError("No vectors reconstructed from shards")
        xb = np.vstack(xb)
        cpu_index = faiss.IndexFlatL2(dim)
        cpu_index.add(xb)
        faiss.write_index(cpu_index, index_path)
        with open(terms_path, "w", encoding="utf-8") as f:
            json.dump(self.term_list, f)

    def load_index(self):
        index_path = f"retriever_{self.fallback_mode}.index"
        terms_path = f"term_list_{self.fallback_mode}.json"
        cpu_index = faiss.read_index(index_path)
        ngpu = self.max_gpus or faiss.get_num_gpus()
        print(f"[INFO] Loading FAISS index onto {ngpu} GPUs (shard mode enabled)")
        dim = cpu_index.d
        co = faiss.GpuClonerOptions()
        co.shard = False
        shard_index = faiss.IndexShards(dim, True, True)
        per_gpu = (cpu_index.ntotal + ngpu - 1) // ngpu

        xb = cpu_index.reconstruct_n(0, cpu_index.ntotal)

        for i in range(ngpu):
            start = i * per_gpu
            end = min((i + 1) * per_gpu, cpu_index.ntotal)
            if start >= end:
                continue
            sub_xb = xb[start:end]

            sub_cpu_index = faiss.IndexFlatL2(dim)
            sub_cpu_index.add(sub_xb)

            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, i, sub_cpu_index, co)
            shard_index.add_shard(gpu_index)

        self.index = shard_index

        with open(terms_path, "r", encoding="utf-8") as f:
            self.term_list = json.load(f)

# get auido full path
def get_audio_full_path(sid):
    doc_id = sid.split("_")[0]  # 提取文档ID，比如 'POD0000001165'
    source_prefix = doc_id[:3]  # POD, AUD, YOU
    id_num = int(doc_id[3:])  # 比如 '0000001165' -> 1165
    subdir_num = (id_num + 99) // 100  # ceil division for every 100 samples
    subdir = f"P{subdir_num:04d}"  # 格式化成P0001这样a

    # source到文件夹名字的映射
    source_map = {
        "POD": "podcast",
        "YOU": "youtube",
        "AUD": "audiobook"
    }
    source_folder = source_map.get(source_prefix)
    if source_folder is None:
        raise ValueError(f"Unknown source prefix: {source_prefix}")

    return os.path.join(
        "/mnt/taurus/data/siqiouyang/datasets/gigaspeech/audio",
        source_folder,
        subdir,
        f"{doc_id}.opus"
    )


import librosa
import torch
import numpy as np

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)


# TODO delete this function
def load_audio(audio_path: str, start_time: float = None, end_time: float = None, target_sr: int = 48000) -> torch.Tensor:
    waveform, sr = torchaudio.load(audio_path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    if start_time is not None and end_time is not None:
        start_sample = int(start_time * target_sr)
        end_sample = int(end_time * target_sr)
        waveform = waveform[:, start_sample:end_sample]

    audio_data = waveform.mean(dim=0).numpy()

    usable_length = (audio_data.shape[0] // target_sr) * target_sr
    if usable_length < target_sr:
        raise ValueError("Audio too short after processing.")
    audio_data = audio_data[:usable_length]

    audio_data = audio_data.reshape(1, -1)
    audio_data = torch.from_numpy(
        int16_to_float32(float32_to_int16(audio_data))
    ).float()

    return audio_data

def evaluate_audio_retrieval(retriever: Retriever, test_samples: List[Dict], device: str = "cuda",max_limit=None, filter_missing_gt=False):
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor

    # 初始化音频缓存
    audio_cache = AudioCache()

    def load_audio_for_sample(sample):
        try:
            audio_path = get_audio_full_path(sample['sid'])
            return audio_cache.load_audio(audio_path, sample.get('begin_time'), sample.get('end_time'))
        except Exception as e:
            print(f"[ERROR] Failed to load audio for {sample['sid']}: {e}")
            return None

    top1, top5 = 0, 0

    # Prepare lowercase term set for efficient matching
    term_set = set([t.lower() for t in retriever.term_list])
    if max_limit is not None:
        test_samples = test_samples[:int(max_limit)]

    # 预处理所有音频文件
    print("Preprocessing all audio files...")
    audio_files = [get_audio_full_path(sample['sid']) for sample in test_samples]
    audio_cache.preprocess_all(audio_files)
    print("Preprocessing completed!")

    batch_size = 8
    output_dir = "./data/audio_embeddings"
    for b in tqdm(range(0, len(test_samples), batch_size), desc="Extracting audio embeddings"):
        batch_samples = test_samples[b:b + batch_size]
        audio_start = time.time()
        with ProcessPoolExecutor(max_workers=batch_size) as pool:
            audio_batch = list(pool.map(load_audio_for_sample, batch_samples))
        audio_end = time.time()
        print(f"[TIME] Audio loading took {audio_end - audio_start:.2f} seconds for batch {b // batch_size}")

        valid_indices = [i for i, a in enumerate(audio_batch) if a is not None]

        if not valid_indices:
            continue

        audios_to_process = [audio_batch[i].squeeze().numpy() for i in valid_indices]

        proc_start = time.time()
        try:
            inputs = retriever.processor(audios=audios_to_process, sampling_rate=48000, return_tensors="pt",
                                         padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                audio_emb_batch = retriever.model.get_audio_features(**inputs)
        except Exception as e:
            print(
                f"[ERROR] Batch inference failed on samples {[(batch_samples[i]['sid']) for i in valid_indices]}: {e}")
            continue
        proc_end = time.time()
        print(f"[TIME] Processor+embedding took {proc_end - proc_start:.2f} seconds for batch {b // batch_size}")

        audio_emb_batch = audio_emb_batch.cpu().numpy()

        query_start = time.time()
        # Now do FAISS search and evaluation
        for idx_in_batch, sample_idx in enumerate(valid_indices):
            sample = batch_samples[sample_idx]
            sid = sample['sid']
            ground_truth_text = sample['text']

            embedding_path = os.path.join(output_dir, f"{sid}.npy")
            if os.path.exists(embedding_path):
                query_emb = np.load(embedding_path)
            else:
                query_emb = audio_emb_batch[idx_in_batch]
                np.save(embedding_path, query_emb)

            try:
                D, I = retriever.index.search(query_emb[None, :], TOP_K)
            except Exception as faiss_e:
                print(f"[ERROR] FAISS search crash at sample {sid}: {faiss_e}")
                continue

            query_end = time.time()
            print(f"[TIME] Query took {query_end - query_start:.2f} seconds for batch {b // batch_size}")

            retrieved_indices = I[0]

            # 🧠 Efficient ground-truth term matching using n-gram scan
            text = ground_truth_text.lower()
            words = text.split()
            max_ngram = 8
            found_terms = set()
            for i in range(len(words)):
                for j in range(i + 1, min(len(words), i + max_ngram) + 1):
                    sub = ' '.join(words[i:j])
                    if sub in term_set:
                        found_terms.add(sub)

            gt_terms = []
            if found_terms:
                for ft in found_terms:
                    for t in retriever.term_list:
                        if t.lower() == ft:
                            gt_terms.append(t)
                            break
            elif filter_missing_gt:
                continue

            if gt_terms:
                retrieved_terms = [retriever.term_list[i] for i in retrieved_indices]
                if retrieved_terms[0] in gt_terms:
                    top1 += 1
                if any(t in retrieved_terms for t in gt_terms):
                    top5 += 1
        evaluate_end = time.time()
        print(f"[TIME] Evaluation took {evaluate_end - query_end:.2f} seconds for batch {b // batch_size}")

        torch.cuda.empty_cache()

    print(f"Top-1 Accuracy: {top1}/{len(test_samples)} = {top1 / len(test_samples):.2%}")
    print(f"Top-5 Accuracy: {top5}/{len(test_samples)} = {top5 / len(test_samples):.2%}")

import glob
import json
import os
import torch


def load_glossary_by_dir(input_file):
    glossary_files = sorted(glob.glob(input_file + "*.json"))
    glossary = []
    for file in glossary_files:
        with open(file, "r", encoding="utf-8") as f:
            glossary.extend(json.load(f))
    return glossary


def generate(input_file, mode,max_gpu, max_terms=None):
    if input_file.endswith(".json"):
        glossary = load_glossary(input_file)
    else:
        glossary = load_glossary_by_dir(input_file)

    glossary = glossary[:max_terms] if max_terms else glossary

    parsed_glossary = []
    for item in glossary:
        parsed_glossary.append({
            "term": item["term"],
            "summary": item.get("short_description", "")
        })

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n🔍 Running in {mode} mode on {device}")
    retriever = Retriever(fallback_mode=mode, device=device, max_gpus=max_gpu)
    index_file = f"retriever_{mode}.index"
    terms_file = f"term_list_{mode}.json"

    # TODO need fix
    if False and os.path.exists(index_file) and os.path.exists(terms_file):
        print("✅ Loading existing index...")
        retriever.load_index()
    else:
        print("⚙️ Building new index...")
        retriever.build_index(parsed_glossary)
        print("⚙️ Building new index done!")
        retriever.save_index()
        print("⚙️ saved new index done!")
    return retriever


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--mode', required=True)
    parser.add_argument('--max_limit', required=False)
    parser.add_argument('--max_gpu', type=int, required=False)
    parser.add_argument('--max_terms', type=int, required=False)
    parser.add_argument('--filter_missing_gt', action="store_true")
    args = parser.parse_args()

    retriever = generate(input_file=args.input, mode=args.mode,max_gpu = args.max_gpu, max_terms=args.max_terms)
    with open('data/gigaspeech_test_samples.json') as f:
        test_samples = json.load(f)
    evaluate_audio_retrieval(retriever, test_samples, max_limit=args.max_limit, filter_missing_gt=args.filter_missing_gt)
