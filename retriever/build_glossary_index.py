import os
import numpy as np
import faiss
import json
from glossary_utils import load_clean_glossary_from_file
from hashlib import md5
import argparse

def hash_text(t): return md5(t.encode("utf-8")).hexdigest()

def load_text_embeddings(texts, cache_dirs):
    from collections import defaultdict

    embeddings = []
    filtered_texts = []

    # Step 1: 计算目标 hash
    target_hashes = {hash_text(t): t for t in texts}
    remaining_hashes = set(target_hashes.keys())

    # Step 2: 扫描所有 .npz 文件，收集目标 embedding
    hash2emb = {}

    for dir in cache_dirs:
        for fname in os.listdir(dir):
            if not fname.endswith(".npz"):
                continue
            path = os.path.join(dir, fname)
            try:
                npz = np.load(path)
                for key in npz.files:
                    if key in remaining_hashes:
                        hash2emb[key] = npz[key]
                        remaining_hashes.remove(key)
                if not remaining_hashes:
                    break
            except Exception as e:
                print(f"[WARN] Failed to load {path}: {e}")

    # Step 3: 按顺序组装结果
    for h, t in target_hashes.items():
        if h in hash2emb:
            embeddings.append(hash2emb[h])
            filtered_texts.append(t)
        else:
            print(f"[WARN] Missing embedding for: {t[:30]}...")

    return np.stack(embeddings), filtered_texts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--text_field', type=str, default="term", choices=["term", "short_description"],
        help="Which field to use as input text (term: comma-split title, short_description: full description)"
    )
    args = parser.parse_args()

    term_set_path = "data/terms/term_set.txt"
    alt2main_path = "data/terms/alt2main.json"
    glossary_path = "data/terms/glossary_filtered.json"

    # 自动发现所有 embedding 缓存目录
    import glob
    cache_dirs = sorted(glob.glob(f"data/{args.text_field}/text_embedding_cache_batch_*"))
    print(f"[INFO] Found {len(cache_dirs)} embedding cache dirs.")

    _, _, glossary = load_clean_glossary_from_file(term_set_path, alt2main_path, glossary_path)
    # --text_field 可为 "term" 或 "short_description"
    texts = [ v[args.text_field] for v in glossary.values()]

    print(f"[INFO] Loading text embeddings for {len(texts)} terms...")
    xb,filtered_texts = load_text_embeddings(texts, cache_dirs)

    text2term = {v[args.text_field]: v for v in glossary.values()}
    filtered_terms = [text2term[t] for t in filtered_texts]
    with open("data/alignment_terms.json", "w", encoding="utf-8") as f:
        json.dump(filtered_terms, f, indent=2, ensure_ascii=False)

    # dim = xb.shape[1]
    # index = faiss.IndexFlatL2(dim)
    # index.add(xb)
    # faiss.write_index(index, "data/retriever.index")

    dim = xb.shape[1]

    # 自动检测 GPU 数量或手动指定
    ngpu = min(4, faiss.get_num_gpus())  # 最多用 4 张
    print(f"[INFO] Using {ngpu} GPUs for sharded FAISS index.")

    # 初始化主 sharded index
    shard_index = faiss.IndexShards(dim, True, True)
    co = faiss.GpuClonerOptions()
    co.shard = False  # 每个 index 是一个完整的 copy（手动 shard）

    # 手动分 shard
    per_gpu = (len(xb) + ngpu - 1) // ngpu
    for i in range(ngpu):
        start = i * per_gpu
        end = min((i + 1) * per_gpu, len(xb))
        if start >= end:
            continue

        sub_embeds = xb[start:end]
        print(f"[DEBUG] GPU-{i} indexing range: [{start}, {end})")

        try:
            res = faiss.StandardGpuResources()
            cpu_sub_index = faiss.IndexFlatL2(dim)
            cpu_sub_index.add(sub_embeds)

            gpu_index = faiss.index_cpu_to_gpu(res, i, cpu_sub_index, co)
            shard_index.add_shard(gpu_index)
        except Exception as e:
            print(f"[ERROR] Failed on GPU-{i}: {e}")

    # 保存合并好的 CPU 版索引
    final_index = faiss.index_gpu_to_cpu(shard_index)
    faiss.write_index(final_index, "data/retriever.index")
    #print(f"[INFO] Multi-GPU FAISS index saved to data/retriever.index")

    print(f"[INFO] FAISS index built and saved to data/retriever.index")