#!/usr/bin/env python3
import os
import json
import torch
import numpy as np
import faiss
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F

class BgeM3TextEncoder(torch.nn.Module):
    def __init__(self, model_id="BAAI/bge-m3", lora_rank=16):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16,
            add_pooling_layer=False
        )
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            target_modules=["query", "key", "value"],
            lora_dropout=0.05,
            bias="none",
            task_type=None
        )
        self.encoder = get_peft_model(self.encoder, lora_config)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]
        return F.normalize(embeddings, p=2, dim=-1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--text_lora_r", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--min_ckpt_matched_keys", type=int, default=200, help="If loaded checkpoint matches fewer keys than this, warn (or fail if --fail_on_low_match).")
    parser.add_argument("--fail_on_low_match", action="store_true", help="Fail fast if checkpoint seems not loaded (matched keys too low).")
    args = parser.parse_args()

    device = torch.device(args.device)

    # 1. Load data
    print(f"[INFO] Loading unique terms from {args.train_jsonl}...", flush=True)
    term_to_translations = {}
    with open(args.train_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                term = item["term"].strip().lower()
                trans = item.get("translation", "").strip().lower()
                if term not in term_to_translations:
                    term_to_translations[term] = set()
                if trans:
                    term_to_translations[term].add(trans)
            except: continue
    
    unique_terms = sorted(list(term_to_translations.keys()))
    num_terms = len(unique_terms)
    print(f"[INFO] Found {num_terms} unique terms.", flush=True)

    # 2. Load Model
    print(f"[INFO] Loading model from {args.model_path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    model = BgeM3TextEncoder(lora_rank=args.text_lora_r).to(device).to(torch.bfloat16)
    
    # IMPORTANT: this mining script is single-process (no DDP). We must ensure checkpoint actually loads.
    checkpoint = torch.load(args.model_path, map_location="cpu")
    if "text_model_state_dict" in checkpoint:
        state_dict = checkpoint["text_model_state_dict"]
        # Strip potential DDP prefix
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(new_state_dict.keys())
        matched = len(model_keys & ckpt_keys)
        print(f"[INFO] CKPT load stats: matched_keys={matched}/{len(model_keys)} (ckpt_keys={len(ckpt_keys)})", flush=True)
        if matched < args.min_ckpt_matched_keys:
            msg = f"[WARN] Suspicious checkpoint load: matched_keys={matched} < {args.min_ckpt_matched_keys}. HN mining may be using near-random text encoder weights."
            if args.fail_on_low_match:
                raise RuntimeError(msg)
            else:
                print(msg, flush=True)

        incompatible = model.load_state_dict(new_state_dict, strict=False)
        print(f"[INFO] CKPT incompatible: missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}", flush=True)
    else:
        raise KeyError(f"[ERROR] 'text_model_state_dict' not found in checkpoint: {args.model_path}")
    model.eval()

    # 3. Encode all terms (directly into a pre-allocated numpy array to save memory)
    print(f"[INFO] Encoding {num_terms} terms...", flush=True)
    dim = 1024
    all_embeddings = np.zeros((num_terms, dim), dtype=np.float32)
    
    with torch.no_grad():
        for i in tqdm(range(0, num_terms, args.batch_size), desc="Encoding"):
            batch_terms = unique_terms[i : i + args.batch_size]
            inputs = tokenizer(batch_terms, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                embeddings = model(inputs.input_ids, inputs.attention_mask)
            all_embeddings[i : i + len(batch_terms)] = embeddings.cpu().float().numpy()
    
    # Pre-normalization for Inner Product (Cosine Similarity)
    print("[INFO] Normalizing embeddings...", flush=True)
    faiss.normalize_L2(all_embeddings)

    # 4. Build HNSW Index (ANN)
    # M=32, efConstruction=128 are good defaults for balance
    print(f"[INFO] Building HNSW index (M=32)...", flush=True)
    index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 128
    index.add(all_embeddings)
    
    print(f"[INFO] Searching top-{args.top_k} neighbors with HNSW...", flush=True)
    index.hnsw.efSearch = 64 # Increase for better recall, decrease for speed
    
    # Batch search to keep progress updated
    all_I = []
    search_batch_size = 10000
    for i in tqdm(range(0, num_terms, search_batch_size), desc="Searching"):
        _, I_batch = index.search(all_embeddings[i : i + search_batch_size], args.top_k)
        all_I.append(I_batch)
    
    I = np.vstack(all_I)
    del all_embeddings # Clear memory
    
    # 5. Filter and save
    print("[INFO] Filtering hard negatives and saving...", flush=True)
    hard_negatives = {}
    for i, term in enumerate(unique_terms):
        term_trans = term_to_translations[term]
        neg_candidates = []
        for idx in I[i]:
            if idx < 0: continue # HNSW might return -1 if search fails
            neg_term = unique_terms[idx]
            if neg_term == term:
                continue
            
            neg_trans = term_to_translations[neg_term]
            if term_trans & neg_trans:
                continue
            
            neg_candidates.append(neg_term)
            if len(neg_candidates) >= 50:
                break
        
        hard_negatives[term] = neg_candidates

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(hard_negatives, f, ensure_ascii=False, indent=2)
    
    print(f"[INFO] Done! Saved to {args.output_path}", flush=True)

if __name__ == "__main__":
    main()
