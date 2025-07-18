import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from hashlib import md5
from tqdm import tqdm
from laion_clap import CLAP_Module
from new_giga_speech import load_preprocessed_samples

def hash_text(t): return md5(t.encode("utf-8")).hexdigest()

def main():
    # 加载训练集样本
    print("[INFO] Loading training samples...")
    samples = load_preprocessed_samples("data/preprocessed_samples_merged.json", with_tensor=False)
    gt_terms = set()
    for s in samples:
        if s.get("ground_truth_term"):
            gt_terms.update(t.strip() for t in s["ground_truth_term"])

    gt_terms = sorted(list(gt_terms))
    print(f"[INFO] Total unique ground_truth_term: {len(gt_terms)}")

    # 初始化模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLAP_Module(enable_fusion=True)
    model.load_ckpt()
    model.load_state_dict(torch.load("data/clap_inbatch.pt", map_location=device), strict=False)
    model = model.to(device)
    model.eval()

    # 批量生成 embedding
    print("[INFO] Encoding terms...")
    batch_size = 256
    all_embeddings = []
    for i in tqdm(range(0, len(gt_terms), batch_size)):
        batch = gt_terms[i:i+batch_size]
        with torch.no_grad():
            emb = model.get_text_embedding(batch, use_tensor=True).to(device)
            emb = F.normalize(emb, dim=-1).cpu().numpy()
            all_embeddings.append(emb)

    xb = np.vstack(all_embeddings)
    print(f"[INFO] Final embedding shape: {xb.shape}")

    # 构建 FAISS index
    dim = xb.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(xb)

    os.makedirs("data/train_index", exist_ok=True)
    faiss.write_index(index, "data/train_index/train_retriever.index")
    print("[INFO] Saved FAISS index to data/train_index/train_retriever.index")

    # 保存 term list（带 short_description 字段）
    train_terms = [{"short_description": t} for t in gt_terms]
    with open("data/train_index/train_terms.json", "w", encoding="utf-8") as f:
        json.dump(train_terms, f, indent=2, ensure_ascii=False)
    print("[INFO] Saved term list to data/train_index/train_terms.json")

if __name__ == "__main__":
    main()