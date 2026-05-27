import json
import os
import pickle
import numpy as np
import torch
import faiss
from tqdm import tqdm
from FlagEmbedding import FlagModel

def process_glossary(input_path, output_path):
    print(f"Loading glossary from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        glossary_data = json.load(f)
    
    term_list = []
    seen_keys = set()
    
    for _, info in glossary_data.items():
        term = info.get("term", "").strip()
        zh_trans = info.get("target_translations", {}).get("zh", "").strip()
        
        if not term or not zh_trans:
            continue
            
        key = term.lower()
        if key in seen_keys:
            continue
            
        term_list.append({
            "key": key,
            "term": term,
            "target_translations": {"zh": zh_trans}
        })
        seen_keys.add(key)
    
    print(f"Extracted {len(term_list)} unique terms.")
    
    # 2. Encode terms with BGE-M3
    print("Loading BGE-M3 model...")
    model = FlagModel('BAAI/bge-m3', use_fp16=True)
    
    texts_to_encode = [item["term"] for item in term_list]
    print(f"Encoding {len(texts_to_encode)} terms...")
    embeddings = model.encode(texts_to_encode, batch_size=128)
    embeddings = embeddings.astype('float32')
    
    # Normalize embeddings for Inner Product (Cosine Similarity)
    faiss.normalize_L2(embeddings)
    
    # 3. Build FAISS Index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    # 4. Save to .pkl
    save_data = {
        "faiss_index": faiss.serialize_index(index),
        "term_list": term_list,
        "embedding_dim": dim
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"Index saved to {output_path}")

if __name__ == "__main__":
    # input_file = "/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_acl6060.json"
    # output_file = "/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_acl6060_index.pkl"
    input_file = "/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/acl_terminology_glossary_lowercase.json"
    output_file = "/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_acl6060_curated_terms_index_bge_m3.pkl"
    process_glossary(input_file, output_file)

