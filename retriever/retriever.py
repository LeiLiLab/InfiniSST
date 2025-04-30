from collections import OrderedDict
import signal

from sentence_transformers import SentenceTransformer
import faiss
import json
import torchaudio
import numpy as np
from typing import List, Dict, Tuple
#from clap import CLAP_Model  # Assuming you have a CLAP text encoder ready
from transformers import ClapModel, ClapProcessor

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
    def __init__(self, model_name: str = "laion/clap-htsat-unfused", fallback_mode: str = "safe", device: str = "cpu", max_gpus: int = None):
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

        # ✅ 多卡 GPU 分布索引（将 CPU index 分 shard 拆到所有可用 GPU 上）
        co = faiss.GpuClonerOptions()
        co.shard = True
        ngpu = self.max_gpus or faiss.get_num_gpus()
        print(f"[INFO] FAISS using {ngpu} GPUs for indexing (shard mode enabled)")
        gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, ngpu=ngpu, co=co)
        self.index = gpu_index

    def query(self, text: str, top_k: int = TOP_K) -> List[str]:
        embedding = self.encode_texts([text])  # 这里是 (1, hidden_dim) 的 GPU tensor
        D, I = self.index.search(embedding, top_k)
        return [self.term_list[i] for i in I[0]]

    def save_index(self):
        index_path = f"retriever_{self.fallback_mode}.index"
        terms_path = f"term_list_{self.fallback_mode}.json"
        # 🔥 先把GPU index转成CPU index，然后保存
        cpu_index = faiss.index_gpu_to_cpu(self.index)
        faiss.write_index(cpu_index, index_path)
        with open(terms_path, "w", encoding="utf-8") as f:
            json.dump(self.term_list, f)

    def load_index(self):
        index_path = f"retriever_{self.fallback_mode}.index"
        terms_path = f"term_list_{self.fallback_mode}.json"
        cpu_index = faiss.read_index(index_path)
        co = faiss.GpuClonerOptions()
        co.shard = True
        ngpu = self.max_gpus or faiss.get_num_gpus()
        print(f"[INFO] Loading FAISS index onto {ngpu} GPUs (shard mode enabled)")
        self.index = faiss.index_cpu_to_all_gpus(cpu_index, ngpu=ngpu, co=co)

        with open(terms_path, "r", encoding="utf-8") as f:
            self.term_list = json.load(f)

# get auido full path
def get_audio_full_path(sid):
    doc_id = sid.split("_")[0]  # 提取文档ID，比如 'POD0000001165'
    source_prefix = doc_id[:3]  # POD, AUD, YOU
    id_num = int(doc_id[3:])  # 比如 '0000001165' -> 1165
    subdir_num = (id_num + 99) // 100  # ceil division for every 100 samples
    subdir = f"P{subdir_num:04d}"  # 格式化成P0001这样

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
    top1, top5 = 0, 0

    # Prepare lowercase term set for efficient matching
    term_set = set([t.lower() for t in retriever.term_list])
    if max_limit is not None:
        test_samples = test_samples[:int(max_limit)]
    for idx, sample in enumerate(tqdm(test_samples, desc="Extracting audio embeddings")):
        sid = sample['sid']
        audio_path = get_audio_full_path(sid)
        ground_truth_text = sample['text']
        # Load and process audio
        start_time = sample.get('begin_time', None)
        end_time = sample.get('end_time', None)
        try:
            audio_tensor = load_audio(audio_path, start_time=start_time, end_time=end_time)  # shape (1, T)
            raw_tensor = audio_tensor.squeeze(0)
            if raw_tensor is None or not torch.isfinite(raw_tensor).all():
                print(f"[ERROR] Invalid audio input (NaN or None) for sample #{idx}: {sid}")
                continue

            print(f"[DEBUG] Processing audio input shape: {raw_tensor.shape}, dtype: {raw_tensor.dtype}")

            try:
                print(f"[INFO] Running sample #{idx}: {sid}")
                inputs = retriever.processor(audios=raw_tensor, sampling_rate=48000, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    audio_emb = retriever.model.get_audio_features(**inputs)
                #
                # # Save to file
                # torch.save(audio_emb, os.path.join(output_dir, f"{sid}.pt"))
                # print(f"[✅] Saved {sid} embedding.")
                # 🔍 FAISS 检索并评估当前样本
                try:
                    query_emb = audio_emb.squeeze(0).cpu().numpy()
                    D, I = retriever.index.search(query_emb[None, :], TOP_K)
                except Exception as faiss_e:
                    print(f"[ERROR] FAISS search crash at sample #{idx}: {sid} | {faiss_e}")
                    continue
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
                del inputs
                del audio_emb
                torch.cuda.empty_cache()
            except Exception as inner_e:
                print(f"[ERROR] Exception during audio embedding extraction: {sid}: {inner_e}")
        except BaseException as crash:
            print(f"[CRITICAL] Low-level crash during audio embedding for sample #{idx}: {sid} | {crash}")

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
        # TODO 暂时不保存index
        #retriever.save_index()
    return retriever


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--mode', required=True)
    parser.add_argument('--max_limit', required=False)
    parser.add_argument('--max_gpu', required=False)
    parser.add_argument('--max_terms', type=int, required=False)
    parser.add_argument('--filter_missing_gt', action="store_true")
    args = parser.parse_args()

    retriever = generate(input_file=args.input, mode=args.mode,max_gpu = args.max_gpu, max_terms=args.max_terms)
    with open('gigaspeech_test_samples.json') as f:
        test_samples = json.load(f)
    output_dir = "./audio_embeddings"
    os.makedirs(output_dir, exist_ok=True)
    evaluate_audio_retrieval(retriever, test_samples, max_limit=args.max_limit, filter_missing_gt=args.filter_missing_gt)
