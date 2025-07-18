import json, os
import torch
import numpy as np
from laion_clap import CLAP_Module
import torch.nn.functional as F
from hashlib import md5
from tqdm import tqdm

def hash_text(t):
    return md5(t.encode("utf-8")).hexdigest()

def encode_texts(texts, gpu_id, output_dir, batch_size=1024, skip_if_exists=False):
    device = f"cuda:{gpu_id}"
    model = CLAP_Module(enable_fusion=True)
    model.load_ckpt()
    model.load_state_dict(torch.load("data/clap_inbatch.pt", map_location=device), strict=False)
    model = model.to(device)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        # 构建已存在哈希集合（仅检查 .npz 中是否已有该 hash）
        existing_hashes = set()
        if skip_if_exists:
            for fname in os.listdir(output_dir):
                if fname.endswith(".npz"):
                    try:
                        npz = np.load(os.path.join(output_dir, fname))
                        existing_hashes.update(npz.files)
                    except:
                        continue

        # 过滤文本
        if skip_if_exists:
            texts_to_encode = [t for t in texts if hash_text(t) not in existing_hashes]
        else:
            texts_to_encode = texts

        for i in tqdm(range(0, len(texts_to_encode), batch_size), desc=f"[Chunk {gpu_id}] Batching"):
            batch = texts_to_encode[i:i + batch_size]
            if not batch:
                continue

            emb = model.get_text_embedding(batch, use_tensor=True).to(device)
            emb = F.normalize(emb, dim=-1).cpu().numpy()

            batch_dict = {hash_text(text): vec for text, vec in zip(batch, emb)}
            npz_path = os.path.join(output_dir, f"batch_{i//batch_size:05d}.npz")
            np.savez(npz_path, **batch_dict)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_id", type=int, required=True)
    parser.add_argument("--gpu_id", type=int, required=True)
    parser.add_argument("--output_dir", type=str, default="data/text_embedding_cache_batch")
    parser.add_argument("--chunk_dir", type=str, default="data/glossary_chunks")  # 可选修改 chunk 路径
    parser.add_argument("--skip_if_exists", action="store_true", help="Skip texts if output .npy file already exists")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)


    # 从 chunk 文件中读取文本列表
    chunk_path = os.path.join(args.chunk_dir, f"text_chunk_{args.chunk_id}.json")
    with open(chunk_path, encoding="utf-8") as f:
        texts = json.load(f)

    # 输出目录带上 chunk_id 后缀
    output = f"{args.output_dir}_{args.chunk_id}"
    encode_texts(texts, gpu_id=args.gpu_id, output_dir=output)