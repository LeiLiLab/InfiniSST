from datasets import load_dataset
import os
import time
import faiss
import json
from typing import List, Dict
from new_giga_speech import load_preprocessed_samples
import torch.nn.functional as F
from laion_clap import CLAP_Module
#from sentence_transformers import SentenceTransformer
from transformers import WhisperModel, WhisperProcessor

# ---------- CONFIG ----------
top_ks = [5, 10, 50]
recall_scores_dict = {k: [] for k in top_ks}

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import functools
print = functools.partial(print, flush=True)

import torch
import torch.nn.functional as F

import datetime

def log_with_time(message):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}")


import sys

class FilteredStdout:
    def __init__(self, original_stdout):
        self.original = original_stdout
    def write(self, text):
        if "Loaded" not in text:
            self.original.write(text)
    def flush(self):
        self.original.flush()


def gpu_worker(gpu_id, batch_list, ret_dict, cache_dir = None):
    model = CLAP_Module(enable_fusion=True)
    # 启用
    sys.stdout = FilteredStdout(sys.__stdout__)
    model.load_ckpt()
    sys.stdout = sys.__stdout__  # 恢复
    model = model.to(f'cuda:{gpu_id}')
    log_with_time(f"[GPU {gpu_id}] Start loading weights")
    state_dict = torch.load(f"data/clap_inbatch.pt", map_location=f'cuda:{gpu_id}')
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
    def __init__(self,enable_fusion = True, device: str = "cpu", max_gpus: int = None):
        self.device = device
        self.enable_fusion = enable_fusion
        # whisper to encode audio and sonar to handle text embedding
        # 加载 Whisper 模型
        # self.whisper_model = whisper.load_model("medium.en")
        # self.whisper_model.to(device)

        # 加载 SONAR 模型（假设是句子嵌入模型）
        # self.sonar_model = SentenceTransformer("SonarModel/checkpoints/sonar-base")
        # self.sonar_model.to(device)

        # 启用
        sys.stdout = FilteredStdout(sys.__stdout__)
        sys.stdout = sys.__stdout__  # 恢复
        # try:
        #     self.model.load_state_dict(torch.load(f"data/clap_inbatch.pt", map_location=device), strict=False)
        #     print(f"[INFO] Loaded fine-tuned model from data/clap_inbatch.pt")
        # except Exception as e:
        #     print(f"[WARN] Failed to load fine-tuned weights: {e}")
        self.index = None
        self.term_list = []
        self.max_gpus = max_gpus

    def encode_texts_multi_gpu(self, texts, batch_size=512):
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
        cache_dir = f"data/new_text_embeddings"
        os.makedirs(cache_dir, exist_ok=True)

        processes = []
        for gpu_id in range(num_gpus):
            p = mp.Process(target=gpu_worker, args=(gpu_id, gpu_batches[gpu_id], return_dict, cache_dir))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        return torch.cat([return_dict[i] for i in range(num_gpus)], dim=0)

    # def build_index(self, glossary: List[Dict]):
    #     texts = [item["short_description"] for item in glossary]
    #
    #     with torch.no_grad():
    #         print(f"[DEBUG] Number of terms: {len(texts)}")
    #         embeddings = self.encode_texts_multi_gpu(texts, batch_size=512).numpy()
    #
    #     dim = embeddings.shape[1]
    #
    #     # ✅ 构建 CPU index
    #     cpu_index = faiss.IndexFlatL2(dim)
    #
    #     # 分批添加 embedding 到 CPU index（降低峰值内存）
    #     batch_size = 10000
    #     for i in range(0, len(embeddings), batch_size):
    #         cpu_index.add(embeddings[i:i + batch_size])
    #
    #     # ✅ 多卡 GPU 分布索引（手动分 shard 到所有可用 GPU 上）
    #     ngpu = self.max_gpus or faiss.get_num_gpus()
    #     print(f"[INFO] FAISS using {ngpu} GPUs for indexing (manual sharding/merging)")
    #     co = faiss.GpuClonerOptions()
    #     co.shard = False
    #     shard_index = faiss.IndexShards(dim, True, True)
    #     per_gpu = (len(embeddings) + ngpu - 1) // ngpu
    #
    #     for i in range(ngpu):
    #         start = i * per_gpu
    #         end = min((i + 1) * per_gpu, len(embeddings))
    #         if start >= end:
    #             continue
    #         sub_embeds = embeddings[start:end]
    #         print(f"[DEBUG] Shard {i}: sub_embeds shape = {sub_embeds.shape}")
    #
    #         res = faiss.StandardGpuResources()
    #         sub_cpu_index = faiss.IndexFlatL2(dim)
    #         sub_cpu_index.add(sub_embeds)
    #
    #         try:
    #             gpu_sub_index = faiss.index_cpu_to_gpu(res, i, sub_cpu_index, co)
    #             shard_index.add_shard(gpu_sub_index)
    #         except Exception as e:
    #             print(f"[ERROR] Failed to build GPU shard {i}, range ({start}-{end}): {e}")
    #
    #     self.index = shard_index

    # def save_index(self):
    #     index_path = f"data/retriever.index"
    #     # 🔥 提取 GPU shard 中的所有向量，构建统一 CPU index 保存
    #     dim = self.index.d
    #     xb = []
    #     # Fix: IndexShards does not have a .shards attribute; use .at(i) and .count()
    #     for shard in [self.index.at(i) for i in range(self.index.count())]:
    #         try:
    #             xb_shard = shard.reconstruct_n(0, shard.ntotal)
    #             xb.append(xb_shard)
    #         except Exception as e:
    #             print(f"[ERROR] Failed to reconstruct shard: {e}")
    #     if not xb:
    #         raise RuntimeError("No vectors reconstructed from shards")
    #     xb = np.vstack(xb)
    #     cpu_index = faiss.IndexFlatL2(dim)
    #     cpu_index.add(xb)
    #     faiss.write_index(cpu_index, index_path)

    def load_index(self,index_file):
        cpu_index = faiss.read_index(index_file)
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

def evaluate_audio_retrieval(retriever: Retriever, test_samples: List[Dict], device: str = "cuda", text_field: str = "short_description"):
    from tqdm import tqdm
    import traceback

    recall_scores = []

    batch_size = 64
    print(f"[DEBUG] Starting evaluation loop with {len(test_samples)} samples, batch size = {batch_size}")

    for b in tqdm(range(0, len(test_samples), batch_size), desc="Extracting audio embeddings"):
        batch_samples = test_samples[b:b + batch_size]
        audio_batch = [sample['audio_tensor'] for sample in batch_samples]

        try:
            with torch.no_grad():
                max_len = max([a.shape[-1] for a in audio_batch])
                padded_audio = torch.stack([
                    F.pad(torch.tensor(a).squeeze(), (0, max_len - a.shape[-1])) for a in audio_batch
                ]).to(device)  # shape: [B, T]

                audio_emb_batch = retriever.model.get_audio_embedding_from_data(x=padded_audio, use_tensor=True)
                audio_emb_batch = F.normalize(audio_emb_batch, dim=-1)

            proc_end = time.time()
        except Exception as e:
            print(f"[ERROR] Exception during embedding at batch {b // batch_size}: {e}")
            traceback.print_exc()
            continue

        audio_emb_batch = audio_emb_batch.detach().cpu().numpy()

        # Now do FAISS search and evaluation
        for idx_in_batch, sample_idx in enumerate(range(len(batch_samples))):
            sample = batch_samples[sample_idx]
            sid = sample['segment_id']
            gt_terms = sample['ground_truth_term']
            # 跳过不存在ground_truth_terms的
            if not gt_terms:
                print("ground-truth term is empty")
                continue

            query_emb = audio_emb_batch[idx_in_batch]


            for top_k in top_ks:
                try:
                    D, I = retriever.index.search(query_emb[None, :], top_k)
                except Exception as faiss_e:
                    print(f"[ERROR] FAISS search crash at sample {sid}: {faiss_e}")
                    continue

                retrieved_terms = [retriever.term_list[i] for i in I[0]]
                retrieved_texts = [rt[text_field].lower() for rt in retrieved_terms]
                matched_count = sum(gt.lower() in retrieved_texts for gt in gt_terms)

                recall = matched_count / len(gt_terms)
                recall_scores_dict[top_k].append(recall)

                if b == 0 and idx_in_batch < 3:  # 可视化前几条
                    print(f"[DEBUG@{top_k}] GT Terms: {gt_terms}")
                    print(f"[DEBUG@{top_k}] Retrieved: {retrieved_texts}")
                    print(f"[DEBUG@{top_k}] Matched: {matched_count}/{len(gt_terms)}")

        torch.cuda.empty_cache()

    for k in top_ks:
        avg_recall = sum(recall_scores_dict[k]) / len(recall_scores_dict[k]) if recall_scores_dict[k] else 0.0
        print(f"\n📊 Average Recall@{k}: {avg_recall:.2%} over {len(recall_scores_dict[k])} evaluated samples")

import os
import torch
from glossary_utils import load_clean_glossary_from_file


# generate retrieve model
def generate(enable_fusion, max_gpu):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    retriever = Retriever(enable_fusion=enable_fusion, device=device, max_gpus=max_gpu)

    with open("data/alignment_terms.json", "r", encoding="utf-8") as f:
        filtered_terms = json.load(f)

    retriever.term_list = filtered_terms
    index_file = f"data/retriever.index"
    # # TODO test
    # index_file = "data/train_index/train_retriever.index"
    # with open("data/train_index/train_terms.json") as f:
    #     retriever.term_list = json.load(f)

    if os.path.exists(index_file):
        retriever.load_index(index_file)
    else:
        raise FileNotFoundError("❌ FAISS index file not found. Please run build_glossary_index.py first.")
    return retriever

if __name__ == "__main__":
    # retriever = generate(
    #         enable_fusion=True,
    #         max_gpu=1
    #     )

    path = "data/test_preprocessed_samples_merged.json"
    test_samples = load_preprocessed_samples(path)[:5]

    # try medium
    processor = WhisperProcessor.from_pretrained("openai/whisper-medium.en")
    model = WhisperModel.from_pretrained("openai/whisper-medium.en")

    # 重采样 audio_tensor 从 48k 到 16k
    import torchaudio
    orig_tensor = test_samples[0]["audio_tensor"]
    if isinstance(orig_tensor, torch.Tensor):
        orig_tensor = orig_tensor.squeeze()
    else:
        orig_tensor = torch.tensor(orig_tensor).squeeze()

    resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)
    sample = resampler(orig_tensor).numpy()

    print(f'最大振幅 (resampled): {np.max(np.abs(sample)):.6f}')

    print(sample)
    inputs = processor(sample, language="en", sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        encoder_outputs = model.encoder(inputs.input_features)
        embeddings = encoder_outputs.last_hidden_state
    print(embeddings.shape)
    print(embeddings)

    # Whisper 推理生成文本
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    import librosa

    # 加载原始音频（非 audio_tensor）
    audio_path = test_samples[0]["audio"]
    waveform, _ = librosa.load(audio_path, sr=16000)

    processor = WhisperProcessor.from_pretrained("openai/whisper-medium.en")
    model_gen = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium.en")
    model_gen.to("cuda" if torch.cuda.is_available() else "cpu")
    model_gen.eval()

    inputs = processor(
        sample,
        sampling_rate=16000,
        language="en",
        task="transcribe",
        return_tensors="pt"
    ).to(model_gen.device)

    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")

    with torch.no_grad():
        predicted_ids = model_gen.generate(
            inputs["input_features"],
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=128
        )
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        print(f"[TRANSCRIPTION] {transcription}")




# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--max_limit', type=int, required=False)
#     parser.add_argument('--max_gpu', type=int, required=False)
#     parser.add_argument('--max_terms', type=int, required=False)
#     parser.add_argument('--samples_path', type=str, default="data/test_preprocessed_samples_merged.json")
#     parser.add_argument('--term_set_path', type=str, default="data/terms/term_set.txt")
#     parser.add_argument('--alt2main_path', type=str, default="data/terms/alt2main.json")
#     parser.add_argument('--glossary_path', type=str, default="data/terms/glossary_filtered.json")
#     args = parser.parse_args()
#
#     retriever = generate(
#         enable_fusion=True,
#         max_gpu=args.max_gpu
#     )
#     test =False
#     if test:
#         term_emb = retriever.model.get_text_embedding(
#             ["well , thank you so much , waldo ."],
#             use_tensor=True
#         ).to(retriever.device)
#
#         term_emb = F.normalize(term_emb, dim=-1).detach().cpu().numpy()
#         D, I = retriever.index.search(term_emb, 10)
#         retrieved_terms = [retriever.term_list[i] for i in I[0]]
#         print(retrieved_terms)
#     else:
#         #path = "data/preprocessed_samples_merged.json"
#         test_samples = load_preprocessed_samples(args.samples_path)
#         test_samples = [sample for sample in test_samples if sample.get('ground_truth_term')]
#         print(f'got eval_set: {len(test_samples)}')
#         evaluate_audio_retrieval(retriever, test_samples, device="cuda", text_field="term")
