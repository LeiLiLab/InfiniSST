from collections import OrderedDict
import signal

from sentence_transformers import SentenceTransformer
import faiss
import json
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
    def __init__(self, model_name: str = "laion/clap-htsat-unfused", fallback_mode: str = "safe", device: str = "cpu"):
        self.fallback_mode = fallback_mode
        self.device = device
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name).to(device)
        self.index = None
        self.term_list = []

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            embeddings = self.model.get_text_features(**inputs)
        return embeddings.cpu().numpy()

    def build_index(self, glossary: List[Dict]):
        # 🔁 Step 1: 去重 glossary（忽略大小写）
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

        # ✅ Step 3: 构建 FAISS index
        self.term_list = [item["term"] for item in glossary]
        dim = embeddings.shape[1]

        res = faiss.StandardGpuResources()  # 创建 GPU 资源
        cpu_index = faiss.IndexFlatL2(dim)  # 先建一个普通的 CPU index
        self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)  # 把CPU index转成GPU index

        # ⚡️直接加 GPU tensor
        self.index.add(embeddings)

    def query(self, text: str, top_k: int = TOP_K) -> List[str]:
        embedding = self.encode_texts([text])  # 这里是 (1, hidden_dim) 的 GPU tensor
        D, I = self.index.search(embedding.detach().cpu().numpy(), top_k)
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

        res = faiss.StandardGpuResources()
        self.index = cpu_index  # 默认放CPU

        try:
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            if 'GpuIndex' in type(gpu_index).__name__:
                print(f"✅ Successfully moved index to GPU: {type(gpu_index)}")
                self.index = gpu_index
            else:
                print(f"⚠️  Moved index type suspicious: {type(gpu_index)}, staying on CPU")
        except Exception as e:
            print(f"⚠️  Failed to move index to GPU: {e}")
            print("⚙️  Falling back to CPU index")

        with open(terms_path, "r", encoding="utf-8") as f:
            self.term_list = json.load(f)

# get auido full path
def get_audio_full_path(sid):
    doc_id = sid.split("_")[0]  # 提取文档ID，比如 'POD0000001165'
    source_prefix = doc_id[:3]  # POD, AUD, YOU
    id_num = int(doc_id[3:])  # 比如 '0000001165' -> 1165
    subdir_num = id_num // 100 + 1  # 每1000个放一个Pxxxx子目录，比如0-999是P0001，1000-1999是P0002
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

# def load_audio(audio_path: str, start_time: float = None, end_time: float = None, target_sr: int = 48000) -> torch.Tensor:
#     audio_data, sr = librosa.load(audio_path, sr=target_sr)  # librosa自动resample
#     if start_time is not None and end_time is not None:
#         start_sample = int(start_time * target_sr)
#         end_sample = int(end_time * target_sr)
#         audio_data = audio_data[start_sample:end_sample]
#     if audio_data.ndim > 1:
#         audio_data = np.mean(audio_data, axis=0)  # 转单通道
#     audio_data = audio_data.reshape(1, -1)  # (1, T)，CLAP要求(1, length)
#     # 规范化到[-1,1]，同时确保是float32
#     audio_data = torch.from_numpy(
#         int16_to_float32(float32_to_int16(audio_data))
#     ).float()
#     print(f"[DEBUG] Loaded audio shape: {audio_data.shape}, path: {audio_path}, start_time: {start_time}, end_time: {end_time}")
#     return audio_data  # 返回Tensor，shape: (1, T)


def load_audio(audio_path: str, start_time: float = None, end_time: float = None, target_sr: int = 48000) -> torch.Tensor:
    audio_data, sr = librosa.load(audio_path, sr=target_sr)
    if start_time is not None and end_time is not None:
        start_sample = int(start_time * target_sr)
        end_sample = int(end_time * target_sr)
        audio_data = audio_data[start_sample:end_sample]

    # 🔥 确保是1D array
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=0)

    # 🔥 保证长度是480000 samples
    desired_length = 48000*5
    current_length = audio_data.shape[0]
    if current_length < desired_length:
        pad_width = desired_length - current_length
        audio_data = np.pad(audio_data, (0, pad_width), mode='constant')
    elif current_length > desired_length:
        audio_data = audio_data[:desired_length]

    # 🔥 转成(1, T)，然后规范float32
    audio_data = audio_data.reshape(1, -1)
    audio_data = torch.from_numpy(
        int16_to_float32(float32_to_int16(audio_data))
    ).float()

    print(f"[DEBUG] Loaded audio shape: {audio_data.shape}, path: {audio_path}, start_time: {start_time}, end_time: {end_time}")
    return audio_data

def evaluate_audio_retrieval(retriever: Retriever, test_samples: List[Dict], device: str = "cuda"):
    from tqdm import tqdm
    top1, top5 = 0, 0
    audio_emb_list = []
    ground_truth_terms = []

    class TimeoutException(Exception):
        pass

    def timeout_handler(signum, frame):
        raise TimeoutException

    for idx, sample in enumerate(tqdm(test_samples, desc="Extracting audio embeddings")):
        sid = sample['sid']
        audio_path = get_audio_full_path(sid)
        ground_truth_text = sample['text']
        # Load and process audio
        start_time = sample.get('begin_time', None)
        end_time = sample.get('end_time', None)
        audio_tensor = load_audio(audio_path, start_time=start_time, end_time=end_time)  # shape (1, T)
        raw_tensor = audio_tensor.squeeze(0)
        try:
            if raw_tensor is None or not torch.isfinite(raw_tensor).all():
                print(f"[ERROR] Invalid audio input (NaN or None) for sample #{idx}: {sid}")
                continue

            print(f"[DEBUG] Processing audio input shape: {raw_tensor.shape}, dtype: {raw_tensor.dtype}")

            try:
                inputs = retriever.processor(audios=raw_tensor, sampling_rate=48000, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(600)  # 600 seconds = 10 minutes
                try:
                    with torch.no_grad():
                        audio_emb = retriever.model.get_audio_features(**inputs)
                except TimeoutException:
                    print(f"[ERROR] Timeout during inference for sample #{idx}: {sid}")
                    continue
                finally:
                    signal.alarm(0)  # Cancel alarm
                audio_emb_list.append(audio_emb.squeeze(0))  # Remove batch dim
            except Exception as inner_e:
                print(f"[ERROR] Exception during audio embedding extraction: {inner_e}")
        except BaseException as crash:
            print(f"[CRITICAL] Low-level crash during audio embedding for sample #{idx}: {sid} | {crash}")

        # Find ground-truth term
        gt_term = None
        for term in retriever.term_list:
            if term.lower() in ground_truth_text.lower():
                gt_term = term
                break
        ground_truth_terms.append(gt_term)

    audio_emb_batch = torch.stack(audio_emb_list, dim=0).contiguous().to(device)  # (B, hidden_dim)
    # Ensure embeddings are (B, hidden_dim) shape
    D, I = retriever.index.search(audio_emb_batch.detach().contiguous(), TOP_K)

    for idx, retrieved_indices in enumerate(I):
        gt_term = ground_truth_terms[idx]
        if gt_term is None:
            continue
        retrieved_terms = [retriever.term_list[i] for i in retrieved_indices]
        if retrieved_terms[0] == gt_term:
            top1 += 1
        if gt_term in retrieved_terms:
            top5 += 1

    total = len([gt for gt in ground_truth_terms if gt is not None])
    print(f"Top-1 Accuracy: {top1/total:.2%}")
    print(f"Top-5 Accuracy: {top5/total:.2%}")

import glob
import json
import os
import torch

#
# def load_glossary(input_file):
#     glossary_files = sorted(glob.glob(input_file + "*.json"))
#     glossary = []
#     for file in glossary_files:
#         with open(file, "r", encoding="utf-8") as f:
#             glossary.extend(json.load(f))
#     return glossary


def generate(input_file, mode):
    glossary = load_glossary(input_file)

    parsed_glossary = []
    for item in glossary:
        parsed_glossary.append({
            "term": item["term"],
            "summary": item.get("short_description", "")
        })

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n🔍 Running in {mode} mode on {device}")
    retriever = Retriever(fallback_mode=mode, device=device)
    index_file = f"retriever_{mode}.index"
    terms_file = f"term_list_{mode}.json"

    if os.path.exists(index_file) and os.path.exists(terms_file):
        print("✅ Loading existing index...")
        retriever.load_index()
    else:
        print("⚙️ Building new index...")
        retriever.build_index(parsed_glossary)
        retriever.save_index()
    return retriever


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--mode', required=True)
    args = parser.parse_args()

    retriever = generate(input_file=args.input, mode=args.mode)
    with open('gigaspeech_test_samples.json') as f:
        test_samples = json.load(f)
    # print(f"Index type: {type(retriever.index)}, On GPU: {isinstance(retriever.index, faiss.GpuIndexFlatL2)}")
    evaluate_audio_retrieval(retriever, test_samples)
