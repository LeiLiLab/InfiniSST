#!/usr/bin/env python3
import json
import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from FlagEmbedding import FlagModel

def precompute_embeddings(args):
    # 1. 扫描总行数并计算本分片的范围
    print(f"Counting lines in {args.input_jsonl}...")
    with open(args.input_jsonl, 'r') as f:
        total_lines = sum(1 for _ in f)
    
    # 计算分片范围
    lines_per_shard = total_lines // args.total_shards
    start_idx = args.shard_id * lines_per_shard
    end_idx = total_lines if args.shard_id == args.total_shards - 1 else (args.shard_id + 1) * lines_per_shard
    num_shard_samples = end_idx - start_idx
    
    print(f"Shard {args.shard_id}/{args.total_shards}: lines {start_idx} to {end_idx} (total {num_shard_samples})")

    # 2. 收集本分片文本
    terms = []
    transcripts = []
    print(f"Reading shard data...")
    with open(args.input_jsonl, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < start_idx: continue
            if i >= end_idx: break
            try:
                data = json.loads(line)
                terms.append(data.get("term", "").strip().lower() or "[NO_TERM]")
                transcripts.append(data.get("chunk_src_text", "").strip().lower())
            except:
                terms.append("[NO_TERM]")
                transcripts.append("")

    # 3. 初始化 BGE-M3 编码
    print("Loading BGE-M3 model...")
    model = FlagModel('BAAI/bge-m3', use_fp16=True) 
    dim = 1024

    def encode_with_local_dedup(text_list, desc):
        unique_list = list(set(text_list))
        mapping = {}
        print(f"Encoding {desc} ({len(unique_list)} unique in shard)...")
        embeddings = model.encode(unique_list, batch_size=args.batch_size)
        for s, e in zip(unique_list, embeddings):
            mapping[s] = e
        return np.array([mapping[s] for s in text_list], dtype='float32')

    shard_term_embs = encode_with_local_dedup(terms, "Terms")
    shard_trans_embs = encode_with_local_dedup(transcripts, "Transcripts")

    # 4. 保存分片结果
    os.makedirs(args.output_dir, exist_ok=True)
    term_shard_path = os.path.join(args.output_dir, f"terms_shard_{args.shard_id}.npy")
    trans_shard_path = os.path.join(args.output_dir, f"trans_shard_{args.shard_id}.npy")
    
    np.save(term_shard_path, shard_term_embs)
    np.save(trans_shard_path, shard_trans_embs)
    
    # 只有主分片保存元信息副本
    if args.shard_id == 0:
        meta = {"num_samples": total_lines, "dim": dim, "input_file": args.input_jsonl}
        with open(os.path.join(args.output_dir, "meta.json"), 'w') as f:
            json.dump(meta, f)
            
    print(f"Shard {args.shard_id} saved to {args.output_dir}")

def merge_shards(args):
    print("Merging shards...")
    with open(os.path.join(args.output_dir, "meta.json"), 'r') as f:
        meta = json.load(f)
    
    total_samples = meta["num_samples"]
    dim = meta["dim"]
    
    term_mmap = np.memmap(os.path.join(args.output_dir, "terms.mmap"), dtype='float32', mode='w+', shape=(total_samples, dim))
    trans_mmap = np.memmap(os.path.join(args.output_dir, "trans.mmap"), dtype='float32', mode='w+', shape=(total_samples, dim))
    
    curr_idx = 0
    for i in range(args.total_shards):
        print(f"Processing shard {i}...")
        t_shard = np.load(os.path.join(args.output_dir, f"terms_shard_{i}.npy"))
        tr_shard = np.load(os.path.join(args.output_dir, f"trans_shard_{i}.npy"))
        
        num_in_shard = t_shard.shape[0]
        term_mmap[curr_idx : curr_idx + num_in_shard] = t_shard
        trans_mmap[curr_idx : curr_idx + num_in_shard] = tr_shard
        curr_idx += num_in_shard
        
        # 删除中间文件节省空间
        os.remove(os.path.join(args.output_dir, f"terms_shard_{i}.npy"))
        os.remove(os.path.join(args.output_dir, f"trans_shard_{i}.npy"))
        
    term_mmap.flush()
    trans_mmap.flush()
    print(f"Final memmaps saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--total_shards", type=int, default=1)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    
    if args.merge:
        merge_shards(args)
    else:
        precompute_embeddings(args)
