from datasets import load_dataset
import os
import time
from new_giga_speech import extract_array_from_sample, filter_train_set
import faiss
from typing import List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor


from inbatch_clap_train import handle_giga_speech_train_samples
# ---------- CONFIG ----------
TOP_K = 10

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import functools
print = functools.partial(print, flush=True)

import torch
import torch.nn.functional as F

import datetime

def log_with_time(message):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}")

# ---------- LOAD GLOSSARY ----------
def load_glossary(glossary_path: str) -> List[Dict]:
    with open(glossary_path, "r", encoding="utf-8") as f:
        return json.load(f)

def gpu_worker(gpu_id, batch_list, ret_dict, cache_dir = "data/text_embeddings"):
    import torch.nn.functional as F
    from laion_clap import CLAP_Module
    enable_fusion = True
    model = CLAP_Module(enable_fusion=enable_fusion)
    model.load_ckpt()
    model = model.to(f'cuda:{gpu_id}')
    log_with_time(f"[GPU {gpu_id}] Start loading weights")
    state_dict = torch.load(f"data/clap_inbatch_{enable_fusion}.pt", map_location=f'cuda:{gpu_id}')
    log_with_time(f"[GPU {gpu_id}] Weights loaded from file, now applying to model")
    model.load_state_dict(state_dict, strict=False)
    log_with_time(f"[GPU {gpu_id}] Model weights loaded")
    with torch.no_grad():
        import hashlib
        import os

        def hash_text(t):  # 用于文件名避免过长
            return hashlib.md5(t.encode('utf-8')).hexdigest()

        cached_results = []

        for batch in batch_list:
            for text in batch:
                fname = os.path.join(cache_dir, f"{hash_text(text)}.npy")
                if os.path.exists(fname):
                    try:
                        emb_arr = np.load(fname)
                        cached_results.append(torch.from_numpy(emb_arr))
                    except Exception as e:
                        print(f"[WARN] Failed to load {fname}: {e}")
                else:
                    log_with_time(f"[GPU {gpu_id}] Computing embedding for: {text[:30]}...")
                    emb = model.get_text_embedding([text], use_tensor=True).to(f'cuda:{gpu_id}')
                    emb = F.normalize(emb, dim=-1)
                    emb_cpu = emb.cpu()
                    np.save(fname, emb_cpu.numpy())
                    cached_results.append(emb_cpu)

        ret_dict[gpu_id] = torch.cat(cached_results, dim=0)

# ---------- BUILD INDEX ----------
class Retriever:
    def __init__(self,enable_fusion = True, fallback_mode: str = "safe", device: str = "cpu", max_gpus: int = None):
        self.fallback_mode = fallback_mode
        self.device = device
        self.enable_fusion = enable_fusion
        self.text_embedding_cache_dir = "data/text_embeddings"
        os.makedirs(self.text_embedding_cache_dir, exist_ok=True)
        import laion_clap
        # self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = laion_clap.CLAP_Module(enable_fusion=enable_fusion)
        self.model.load_ckpt()
        self.model = self.model.to(device)
        try:
            self.model.load_state_dict(torch.load(f"data/clap_inbatch_{enable_fusion}", map_location=device), strict=False)
            print(f"[INFO] Loaded fine-tuned model from data/clap_inbatch.pt")
        except Exception as e:
            print(f"[WARN] Failed to load fine-tuned weights: {e}")
        self.index = None
        self.term_list = []
        self.max_gpus = max_gpus
        self.return_summary = False

    def encode_texts_multi_gpu(self, texts,mode,enable_fusion, batch_size=512):
        import torch.multiprocessing as mp
        from multiprocessing import Manager

        mp.set_start_method("spawn", force=True)
        num_gpus = torch.cuda.device_count()
        print(f"[INFO] Multi-GPU embedding: using {num_gpus} GPUs")

        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        gpu_batches = [[] for _ in range(num_gpus)]
        for i, batch in enumerate(batches):
            gpu_batches[i % num_gpus].append(batch)

        manager = Manager()
        return_dict = manager.dict()
        cache_dir = f"data/new_text_embeddings_{mode}_{enable_fusion}"
        os.makedirs(cache_dir, exist_ok=True)

        processes = []
        for gpu_id in range(num_gpus):
            p = mp.Process(target=gpu_worker, args=(gpu_id, gpu_batches[gpu_id], return_dict, cache_dir))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        return torch.cat([return_dict[i] for i in range(num_gpus)], dim=0)

    def build_index(self, glossary: List[Dict]):
        # # 🔁 Step 1: 去重 glossary（忽略大小写）
        # if self.fallback_mode == "safe":
        #     unique_items = OrderedDict()
        #     for item in glossary:
        #         key = item['term'].strip().lower()
        #         if key not in unique_items:
        #             unique_items[key] = item
        #     glossary = list(unique_items.values())

        # 🔍 Step 2: 用 term-only 构建 embedding

        if self.fallback_mode == "safe":
            texts = [item["term"] for item in glossary]
        elif self.fallback_mode == "flexible":
            texts = [
                f"{item['term']}, {item['short_description']}" if item["short_description"]
                else item["term"]
                for item in glossary
            ]
        else:
            texts = [item["term"] for item in glossary]

        with torch.no_grad():
            print(f"[DEBUG] Number of terms: {len(texts)}")
            embeddings = self.encode_texts_multi_gpu(texts,self.fallback_mode,self.enable_fusion, batch_size=512).numpy()

        print(f"[DEBUG] encode_texts Embeddings: {embeddings.shape}")

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

    def save_index(self):
        index_path = f"retriever_{self.fallback_mode}_{self.enable_fusion}.index"
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

    def load_index(self):
        index_path = f"retriever_{self.fallback_mode}_{self.enable_fusion}.index"
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


import librosa
import torch
import numpy as np

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

def evaluate_audio_retrieval(retriever: Retriever, test_samples: List[Dict], device: str = "cuda"):
    from tqdm import tqdm
    import traceback

    recall_scores = []

    batch_size = 8
    output_dir = f"./data/new_audio_embeddings_{retriever.fallback_mode}_{retriever.enable_fusion}"

    print(f"[DEBUG] Starting evaluation loop with {len(test_samples)} samples, batch size = {batch_size}")

    for b in tqdm(range(0, len(test_samples), batch_size), desc="Extracting audio embeddings"):
        batch_samples = test_samples[b:b + batch_size]
        print(f"[DEBUG] Processing batch {b // batch_size}: {len(batch_samples)} samples")

        audio_batch = [sample['audio_tensor'] for sample in batch_samples]

        try:
            audio_start = time.time()  # ✅ 补充这一行

            with torch.no_grad():
                max_len = max([a.shape[-1] for a in audio_batch])
                padded_audio = torch.stack([
                    F.pad(torch.tensor(a).squeeze(), (0, max_len - a.shape[-1])) for a in audio_batch
                ]).to(device)  # shape: [B, T]

                print(f"[DEBUG] Audio padded shape: {padded_audio.shape}")
                audio_emb_batch = retriever.model.get_audio_embedding_from_data(x=padded_audio, use_tensor=True)
                audio_emb_batch = F.normalize(audio_emb_batch, dim=-1)

            proc_end = time.time()
            print(f"[TIME] Processor+embedding took {proc_end - audio_start:.2f} seconds for batch {b // batch_size}")
        except Exception as e:
            print(f"[ERROR] Exception during embedding at batch {b // batch_size}: {e}")
            traceback.print_exc()
            continue

        audio_emb_batch = audio_emb_batch.cpu().numpy()

        query_start = time.time()
        # Now do FAISS search and evaluation
        for idx_in_batch, sample_idx in enumerate(range(len(batch_samples))):
            sample = batch_samples[sample_idx]
            sid = sample['segment_id']
            ground_truth_text = sample['text']

            embedding_path = os.path.join(output_dir, f"{sid}.npy")
            if os.path.exists(embedding_path):
                query_emb = np.load(embedding_path)
            else:
                query_emb = audio_emb_batch[idx_in_batch]
                np.save(embedding_path, query_emb)

            # --- DEBUG: print SID, GT Text, embedding norm
            print(f"[DEBUG] Evaluating SID: {sid}, GT Text: {ground_truth_text[:100]}...")  # Truncate long texts
            print(f"[DEBUG] Embedding norm: {np.linalg.norm(query_emb):.4f}")

            try:
                D, I = retriever.index.search(query_emb[None, :], TOP_K)
            except Exception as faiss_e:
                print(f"[ERROR] FAISS search crash at sample {sid}: {faiss_e}")
                continue

            # --- DEBUG: print retrieved top-K terms
            retrieved_terms = [retriever.term_list[i] for i in I[0]]
            # Get ground-truth terms directly from sample
            gt_terms = sample.get("ground_truth_term", [])
            if not gt_terms:
                print(f"[WARN] No ground_truth_term found in sample: {sample['segment_id']}")
                print(f"[DEBUG] Sample keys: {sample.keys()}")
                continue
            print(f"[DEBUG] Top-{TOP_K} Retrieved terms: {retrieved_terms}")
            if b == 0 and idx_in_batch < 10:
                print(f"[DEBUG] Text: {ground_truth_text}")
                print(f"[DEBUG] GT Terms: {gt_terms}")
                print(f"[DEBUG] Retrieved (Top-{TOP_K}): {retrieved_terms[:TOP_K]}")

            query_end = time.time()
            print(f"[TIME] Query took {query_end - query_start:.2f} seconds for batch {b // batch_size}")

            retrieved_indices = I[0]

            if gt_terms:
                # Only match against the term part (before comma) in a case-insensitive way
                def get_term_part(s):
                    return s["term"]
                retrieved_term_texts = [get_term_part(rt).lower() for rt in retrieved_terms]
                matched_count = 0
                for gt in gt_terms:
                    if gt.lower() in retrieved_term_texts:
                        matched_count += 1
                recall = matched_count / len(gt_terms)
                print(f"[RECALL] {sid}: Matched {matched_count}/{len(gt_terms)} terms, Recall@{TOP_K} = {recall:.2%}")
                recall_scores.append(recall)
        evaluate_end = time.time()
        print(f"[TIME] Evaluation took {evaluate_end - query_end:.2f} seconds for batch {b // batch_size}")

        torch.cuda.empty_cache()

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    print(f"\n📊 Average Recall@{TOP_K}: {avg_recall:.2%} over {len(recall_scores)} evaluated samples")

import glob
import json
import os
import torch
from glossary_utils import load_and_clean_glossary


def generate(enable_fusion,input_file, mode, max_gpu, max_terms=None):
    parsed_glossary = load_and_clean_glossary(input_file, max_terms)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n🔍 Running in {mode} mode on {device}")
    retriever = Retriever(enable_fusion=enable_fusion, fallback_mode=mode, device=device, max_gpus=max_gpu)
    retriever.term_list = parsed_glossary
    index_file = f"retriever_{mode}_{enable_fusion}.index"

    if os.path.exists(index_file):
        print("✅ Loading existing index...")
        retriever.load_index()
        print("✅ Loading existing done!")
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
    parser.add_argument('--max_limit', type=int, required=False)
    parser.add_argument('--max_gpu', type=int, required=False)
    parser.add_argument('--max_terms', type=int, required=False)
    args = parser.parse_args()

    retriever = generate(enable_fusion = True, input_file=args.input, mode=args.mode,max_gpu = args.max_gpu, max_terms=args.max_terms)


    eval_set = handle_giga_speech_train_samples(name="dev",split="validation")

    print(f'got eval_set: {len(eval_set)}')
    evaluate_audio_retrieval(retriever, eval_set)
