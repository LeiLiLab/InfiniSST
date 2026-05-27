#!/usr/bin/env python3
import os
import json
import torch
import numpy as np
import faiss
import pickle
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ======Configuration=====
GLOSSARY_TERMS_FIELD_NAME = "terms"
GLOSSARY_META_FIELD_NAME = "meta"
WRAPPED_GLOSSARY_ALLOWED_KEYS = {GLOSSARY_TERMS_FIELD_NAME, GLOSSARY_META_FIELD_NAME}
# ======Configuration=====

class BgeM3TextEncoder(torch.nn.Module):
    def __init__(self, model_id="BAAI/bge-m3", lora_rank=16,
                 lora_target_modules=None):
        super().__init__()
        if lora_target_modules is None:
            lora_target_modules = ["query", "key", "value"]
        self.encoder = AutoModel.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16,
            add_pooling_layer=False
        )
        
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            target_modules=lora_target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=None
        )
        self.encoder = get_peft_model(self.encoder, lora_config)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Take CLS token and normalize
        embeddings = outputs.last_hidden_state[:, 0, :]
        return F.normalize(embeddings, p=2, dim=-1)

def main():
    parser = argparse.ArgumentParser(description="Build FAISS index for glossary using tuned Qwen3 V4 Text Encoder")
    parser.add_argument("--glossary_path", type=str, 
                        default="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_acl6060.json",
                        help="Path to glossary.json")
    parser.add_argument("--model_path", type=str, 
                        default="/mnt/gemini/data2/jiaxuanluo/q3rag_unfrozen_lora-r32-tr16_bs4k_w1.0-0.0_sampled_best.pt",
                        help="Path to V4 model checkpoint (.pt)")
    parser.add_argument("--output_path", type=str, 
                        default="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_acl6060_index_v4.pkl",
                        help="Path to save the new index (.pkl)")
    parser.add_argument("--text_lora_r", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument(
        "--target_lang_code",
        type=str,
        default="zh",
        help="Target language code to store in target_translations (e.g., zh, ja, de).",
    )
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # 1. Load Glossary
    print(f"[INFO] Loading glossary from {args.glossary_path}...")
    with open(args.glossary_path, "r", encoding="utf-8") as f:
        glossary = json.load(f)

    # Support both formats:
    # - Flat dict: { "<term>": {..payload..}, ... }
    # - Wrapped dict: { "meta": {...}, "terms": { "<term>": {..payload..}, ... } }
    # IMPORTANT:
    # Avoid false positives when a normal flat glossary contains a real term key named "terms".
    # We only treat it as wrapped format when top-level keys are exactly:
    #   {"terms"} or {"terms", "meta"}.
    is_wrapped_glossary = (
        isinstance(glossary, dict)
        and GLOSSARY_TERMS_FIELD_NAME in glossary
        and isinstance(glossary.get(GLOSSARY_TERMS_FIELD_NAME), dict)
        and set(glossary.keys()).issubset(WRAPPED_GLOSSARY_ALLOWED_KEYS)
    )
    if is_wrapped_glossary:
        glossary = glossary[GLOSSARY_TERMS_FIELD_NAME]
    
    # Pre-process glossary entries to match StreamingQwen3RAGRetriever expectations
    # Expected format: list of dicts with {"key": canonical_lc, "term": surface, "target_translations": {"<lang>": ...}}
    filtered_entries = []
    seen_keys = set()
    
    for term, payload in glossary.items():
        canonical_key = term.strip().lower()
        if canonical_key in seen_keys:
            continue
        seen_keys.add(canonical_key)
        
        # Build entry metadata
        entry = {
            "key": canonical_key,
            "term": term.strip(),
            "target_translations": {}
        }
        
        if isinstance(payload, dict):
            # If payload is a dict, preserve other fields and set translations
            for k, v in payload.items():
                if k != "key":
                    entry[k] = v
            # Ensure target_translations exists for target lang
            if "translation" in payload:
                entry["target_translations"][args.target_lang_code] = payload["translation"]
            elif "target_translations" in payload:
                entry["target_translations"].update(payload["target_translations"])
        elif isinstance(payload, str):
            # Simple term: translation mapping
            entry["target_translations"][args.target_lang_code] = payload
            
        filtered_entries.append(entry)
    
    print(f"[INFO] Total unique terms to encode: {len(filtered_entries)}")
    
    # 2. Load Model
    print(f"[INFO] Loading tuned Text Encoder from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    
    checkpoint = torch.load(args.model_path, map_location=device)
    ckpt_args = checkpoint.get("args", {})
    if isinstance(ckpt_args, dict):
        ckpt_lora_rank = ckpt_args.get("text_lora_rank", args.text_lora_r)
        ckpt_target_modules = ckpt_args.get("text_lora_target_modules",
                                            ["query", "key", "value"])
    else:
        ckpt_lora_rank = getattr(ckpt_args, "text_lora_rank", args.text_lora_r)
        ckpt_target_modules = getattr(ckpt_args, "text_lora_target_modules",
                                      ["query", "key", "value"])
    print(f"[INFO] Checkpoint LoRA config: rank={ckpt_lora_rank}, "
          f"target_modules={ckpt_target_modules}")
    
    model = BgeM3TextEncoder(
        lora_rank=ckpt_lora_rank,
        lora_target_modules=ckpt_target_modules,
    ).to(device).to(torch.bfloat16)
    
    if "text_model_state_dict" in checkpoint:
        state_dict = checkpoint["text_model_state_dict"]
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        print("[INFO] Successfully loaded tuned weights.")
    else:
        print("[WARN] text_model_state_dict not found in checkpoint! Using base BGE-M3.")
    
    model.eval()
    
    # 3. Encode Terms
    print(f"[INFO] Encoding terms (batch_size={args.batch_size})...")
    all_embeddings = []
    
    # Extract only the keys for encoding
    all_keys = [e["key"] for e in filtered_entries]
    
    for i in tqdm(range(0, len(all_keys), args.batch_size)):
        batch_keys = all_keys[i : i + args.batch_size]
        inputs = tokenizer(
            batch_keys, 
            padding=True, 
            truncation=True, 
            max_length=64, 
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                embeddings = model(inputs.input_ids, inputs.attention_mask)
            all_embeddings.append(embeddings.cpu().float().numpy())
            
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    faiss.normalize_L2(all_embeddings)
    
    # 4. Build FAISS Index
    print(f"[INFO] Building FAISS IndexFlatIP...")
    dim = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(all_embeddings)
    
    # 5. Save Index
    print(f"[INFO] Saving index to {args.output_path}...")
    index_data = {
        "faiss_index": faiss.serialize_index(index),
        "term_list": filtered_entries,
        "num_terms": len(filtered_entries),
        "embedding_dim": dim
    }
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "wb") as f:
        pickle.dump(index_data, f)
        
    print(f"[INFO] Done! Index saved successfully.")

if __name__ == "__main__":
    main()

