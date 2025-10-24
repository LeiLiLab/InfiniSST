#!/usr/bin/env python3
"""
从现有 checkpoint 离线挖 Hard Negatives：
- 读取 train/test 样本（mmap 或路径皆可）
- 用当前 best ckpt 编码每个 audio 样本
- 在"已建好的 term FAISS 索引（512维）"上搜 TopK
- 去掉 GT，保存每样本的 hard_neg_terms 列表
输出：JSONL，每行 { "audio_key": "...", "hard_negs": ["t1","t2",...], "topk": 200 }
"""
import os
import sys
import json
import argparse
import numpy as np
import faiss
import torch
import torch.nn.functional as F
from tqdm import tqdm

# 禁用 tokenizers 的并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from Qwen2_Audio_train import (
    Qwen2AudioSpeechEncoder, 
    Qwen2AudioTextEncoder, 
    ContrastiveQwen2AudioModel
)
from mmap_audio_reader import MMapAudioCollection, extract_audio_key_from_path


def load_retriever_terms(index_pkl):
    """加载预构建的 FAISS 索引和术语列表"""
    import pickle
    print(f"[INFO] Loading FAISS index from: {index_pkl}")
    with open(index_pkl, "rb") as f:
        data = pickle.load(f)
    
    index = faiss.deserialize_index(data["faiss_index"])
    term_list = [d["term"].lower() for d in data["term_list"]]
    
    print(f"[INFO] Loaded index with {index.ntotal} vectors, {len(term_list)} terms")
    return index, term_list


def iter_samples(json_path):
    """迭代有效样本"""
    print(f"[INFO] Loading samples from: {json_path}")
    with open(json_path, "r") as f:
        all_samples = json.load(f)
    
    valid_count = 0
    for x in all_samples:
        if x.get("term_chunk_text") and x.get("term_chunk_audio"):
            gts = [t.lower() for t in x.get("term_chunk_audio_ground_truth_terms", []) 
                   if isinstance(t, str) and len(t.strip()) >= 3]
            if gts:
                x_copy = dict(x)
                x_copy["term_chunk_audio_ground_truth_terms"] = gts
                valid_count += 1
                yield x_copy
    
    print(f"[INFO] Found {valid_count} valid samples")


def main():
    parser = argparse.ArgumentParser(description="Mine hard negatives from trained model")
    parser.add_argument("--samples_path", required=True, help="训练/测试样本JSON路径")
    parser.add_argument("--mmap_dir", default=None, help="mmap音频分片目录（可选）")
    parser.add_argument("--faiss_index_pkl", required=True, help="预构建的FAISS索引pkl文件")
    parser.add_argument("--model_path", default=None, help="训练好的模型checkpoint路径（可选）")
    parser.add_argument("--model_name", default="Qwen/Qwen2-Audio-7B-Instruct", help="基础模型名称")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout（推理时设为0）")
    parser.add_argument("--out_path", required=True, help="输出JSONL文件路径")
    parser.add_argument("--topk", type=int, default=200, help="每个样本检索的候选数")
    parser.add_argument("--batch_size", type=int, default=128, help="批处理大小")
    parser.add_argument("--device", default="cuda:0", help="使用的设备")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")
    
    # 初始化模型
    print(f"[INFO] Initializing model: {args.model_name}")
    speech_encoder = Qwen2AudioSpeechEncoder(model_name=args.model_name, device=device)
    text_encoder = Qwen2AudioTextEncoder(
        model_name=args.model_name, 
        device=device, 
        shared_model=speech_encoder.get_shared_model()
    )
    
    model = ContrastiveQwen2AudioModel(
        speech_encoder, 
        text_encoder, 
        proj_dim=512,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    ).to(device)
    
    # 加载训练好的权重
    if args.model_path and os.path.exists(args.model_path):
        print(f"[INFO] Loading model weights from: {args.model_path}")
        try:
            checkpoint = torch.load(args.model_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            
            # 处理DDP前缀
            if list(state_dict.keys())[0].startswith("module."):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            
            model.load_state_dict(state_dict, strict=False)
            print("[INFO] ✅ Model weights loaded successfully")
        except Exception as e:
            print(f"[ERROR] ❌ Failed to load model weights: {e}")
            print("[WARN] Will use randomly initialized model")
    else:
        print(f"[WARN] No model checkpoint provided or file not found")
        print("[WARN] Using randomly initialized model for mining")
    
    model.eval()
    
    # 加载 FAISS 索引和术语列表
    index, term_list = load_retriever_terms(args.faiss_index_pkl)
    term_set = set(term_list)  # 用于快速查找
    
    # 初始化 mmap 数据库（如果需要）
    use_mmap = args.mmap_dir and os.path.exists(args.mmap_dir)
    mmap_db = None
    if use_mmap:
        print(f"[INFO] Using mmap audio database from: {args.mmap_dir}")
        mmap_db = MMapAudioCollection(args.mmap_dir)
    else:
        print("[INFO] Using file-based audio loading")
    
    # 收集样本
    samples = list(iter_samples(args.samples_path))
    if len(samples) == 0:
        print("[ERROR] No valid samples found!")
        return 1
    
    print(f"\n{'='*80}")
    print(f"HARD NEGATIVE MINING")
    print(f"{'='*80}")
    print(f"[INFO] Total samples: {len(samples)}")
    print(f"[INFO] FAISS index terms: {len(term_list)}")
    print(f"[INFO] Top-K per sample: {args.topk}")
    print(f"[INFO] Batch size: {args.batch_size}")
    print(f"{'='*80}\n")
    
    # 准备输出文件
    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    def get_audio_tensor(sample):
        """获取音频数据（支持mmap和文件路径）"""
        if use_mmap:
            key = extract_audio_key_from_path(sample["term_chunk_audio"])
            try:
                wav, sr, _, _ = mmap_db.get_by_key(key)
                return key, torch.from_numpy(wav.copy()).float()
            except Exception as e:
                print(f"[WARN] Failed to load audio key {key}: {e}")
                return key, torch.zeros(16000, dtype=torch.float32)
        else:
            # 文件路径模式
            path = sample["term_chunk_audio"]
            return path, path
    
    # 批量编码 + 搜索
    out_file = open(args.out_path, "w", encoding="utf-8")
    
    buf_audio = []
    buf_metadata = []  # (key, gt_set)
    processed_count = 0
    failed_count = 0
    
    for sample in tqdm(samples, desc="Mining hard negatives"):
        key, audio_data = get_audio_tensor(sample)
        gt_set = set(t.lower() for t in sample["term_chunk_audio_ground_truth_terms"])
        
        buf_audio.append(audio_data)
        buf_metadata.append((key, gt_set))
        
        # 达到batch大小或最后一个样本
        if len(buf_audio) >= args.batch_size or sample == samples[-1]:
            try:
                # 编码音频
                with torch.no_grad():
                    audio_emb = model.encode_audio(buf_audio)
                    audio_emb = audio_emb.detach().cpu().float().numpy()
                
                # FAISS搜索
                D, I = index.search(audio_emb, args.topk)
                
                # 处理每个样本的结果
                for (audio_key, gt_set), indices in zip(buf_metadata, I):
                    # 过滤掉ground truth术语
                    hard_negs = []
                    for idx in indices:
                        term = term_list[idx]
                        if term not in gt_set:
                            hard_negs.append(term)
                    
                    # 写入结果
                    result = {
                        "audio_key": audio_key,
                        "hard_negs": hard_negs,
                        "topk": args.topk,
                        "num_gt": len(gt_set)
                    }
                    out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    processed_count += 1
                
            except Exception as e:
                print(f"\n[ERROR] Batch processing failed: {e}")
                failed_count += len(buf_audio)
                # 写入空结果
                for audio_key, gt_set in buf_metadata:
                    result = {
                        "audio_key": audio_key,
                        "hard_negs": [],
                        "topk": args.topk,
                        "num_gt": len(gt_set)
                    }
                    out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            
            # 清空缓冲区
            buf_audio = []
            buf_metadata = []
    
    out_file.close()
    
    if mmap_db:
        mmap_db.close()
    
    print(f"\n{'='*80}")
    print(f"MINING COMPLETED")
    print(f"{'='*80}")
    print(f"[INFO] ✅ Successfully processed: {processed_count} samples")
    if failed_count > 0:
        print(f"[WARN] ⚠️  Failed to process: {failed_count} samples")
    print(f"[INFO] Output saved to: {args.out_path}")
    
    # 统计信息
    print(f"\n[INFO] Analyzing results...")
    hn_counts = []
    with open(args.out_path, "r") as f:
        for line in f:
            rec = json.loads(line)
            hn_counts.append(len(rec["hard_negs"]))
    
    if hn_counts:
        print(f"[INFO] Hard negatives statistics:")
        print(f"  - Average per sample: {np.mean(hn_counts):.1f}")
        print(f"  - Median: {np.median(hn_counts):.0f}")
        print(f"  - Min: {np.min(hn_counts)}")
        print(f"  - Max: {np.max(hn_counts)}")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

