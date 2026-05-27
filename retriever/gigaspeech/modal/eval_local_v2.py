#!/usr/bin/env python3
"""
本地评估脚本 - 单进程版本
用于在本地评估训练好的模型，不需要DDP
"""

import os
import sys
import argparse
import json
import torch
import faiss
import pickle

# 禁用 tokenizers 的并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 导入模型相关
from Qwen2_Audio_train import (
    Qwen2AudioSpeechEncoder,
    Qwen2AudioTextEncoder,
    ContrastiveQwen2AudioModel,
    encode_texts_in_batches,
    SimpleRetriever,
)
from mmap_audio_reader import MMapAudioCollection, extract_audio_key_from_path


class LocalTermLevelDataset:
    """本地评估用的简化数据集（支持mmap）"""
    
    def __init__(self, samples_path, mmap_shard_dir=None):
        print(f"[INFO] Loading samples from: {samples_path}")
        with open(samples_path, "r") as f:
            all_samples = json.load(f)
        
        # 初始化 mmap 数据库（如果提供）
        self.audio_db = None
        if mmap_shard_dir and os.path.exists(mmap_shard_dir):
            print(f"[INFO] Initializing mmap audio database from: {mmap_shard_dir}")
            self.audio_db = MMapAudioCollection(mmap_shard_dir)
        
        # 过滤有效样本
        valid_samples = []
        for sample in all_samples:
            if not (sample.get('term_chunk_text', '').strip() and sample.get('term_chunk_audio', '')):
                continue
            
            terms = sample.get('term_chunk_audio_ground_truth_terms', [])
            if not isinstance(terms, list):
                terms = []
            
            filtered_terms = [
                t for t in terms
                if isinstance(t, str) and len(t.strip()) >= 3
            ]
            
            if not filtered_terms:
                continue
            
            # 如果使用mmap，检查音频是否存在
            if self.audio_db:
                audio_path = sample.get("term_chunk_audio", "")
                audio_key = extract_audio_key_from_path(audio_path)
                if audio_key in self.audio_db.k2loc:
                    sample = dict(sample)
                    sample['term_chunk_audio_ground_truth_terms'] = filtered_terms
                    sample['audio_key'] = audio_key
                    valid_samples.append(sample)
            else:
                sample = dict(sample)
                sample['term_chunk_audio_ground_truth_terms'] = filtered_terms
                valid_samples.append(sample)
        
        self.samples = valid_samples
        print(f"[INFO] Loaded {len(self.samples)} valid samples")
    
    def __getitem__(self, index):
        sample = self.samples[index]
        chunk_text = sample["term_chunk_text"]
        ground_truth_terms = sample.get('term_chunk_audio_ground_truth_terms', [])
        
        # 从 mmap 或文件路径加载音频
        if self.audio_db and 'audio_key' in sample:
            audio_key = sample["audio_key"]
            try:
                wav, sr, _, _ = self.audio_db.get_by_key(audio_key)
                audio_tensor = torch.from_numpy(wav.copy()).float()
            except Exception as e:
                print(f"[WARN] Failed to load audio for key {audio_key}: {e}")
                audio_tensor = torch.zeros(16000, dtype=torch.float32)
        else:
            # 返回路径，让模型自己加载
            audio_tensor = sample["term_chunk_audio"]
        
        return ground_truth_terms, audio_tensor, chunk_text
    
    def __len__(self):
        return len(self.samples)
    
    def close(self):
        if self.audio_db:
            self.audio_db.close()


def encode_audio_tensors_in_batches(model, audio_tensors, batch_size=128, device="cuda"):
    """Encode audio tensors in batches using the model's audio encoder"""
    all_embeddings = []
    
    for i in range(0, len(audio_tensors), batch_size):
        batch_tensors = audio_tensors[i:i + batch_size]
        try:
            processed_tensors = []
            for tensor in batch_tensors:
                if isinstance(tensor, torch.Tensor):
                    tensor = tensor.float().to(device)
                processed_tensors.append(tensor)
            
            if model.training:
                embeddings = model.encode_audio(processed_tensors)
            else:
                with torch.no_grad():
                    embeddings = model.encode_audio(processed_tensors)
            
            embeddings = embeddings.float()
            if not model.training:
                embeddings = embeddings.detach()
            all_embeddings.append(embeddings)
        except Exception as e:
            print(f"[ERROR] Failed to encode audio tensor batch {i//batch_size}: {e}")
            print(f"[DEBUG] Batch tensor types: {[type(t) for t in batch_tensors]}")
            print(f"[DEBUG] Batch tensor shapes: {[t.shape if isinstance(t, torch.Tensor) else 'Not tensor' for t in batch_tensors]}")
            print(f"[DEBUG] Batch tensor dtypes: {[t.dtype if isinstance(t, torch.Tensor) else 'Not tensor' for t in batch_tensors]}")
            dummy_emb = torch.zeros(len(batch_tensors), 512, dtype=torch.float32, device=device)
            all_embeddings.append(dummy_emb)
    
    return torch.cat(all_embeddings, dim=0)

def extract_all_used_terms(dataset):
    """Extract all used terms from dataset"""
    used_terms = set()
    for i, sample in enumerate(dataset):
        if sample is None:
            continue
        try:
            ground_truth_terms, _, _ = sample
            for term in ground_truth_terms:
                if isinstance(term, str) and len(term.strip()) > 0:
                    used_terms.add(term.lower())
        except Exception as e:
            print(f"[DEBUG] Error extracting terms from sample {i}: {e}")
            print(f"[DEBUG] Sample: {sample}")
            continue
    
    print(f"[DEBUG] extract_all_used_terms found {len(used_terms)} terms from {len(dataset)} samples")
    return list(used_terms)


def evaluate_topk_recall(model, retriever, dataset, device, top_ks=(1, 5, 10), max_eval=1000, skip_term_encoding=False):
    """Evaluate recall@k using used_terms index
    
    Args:
        skip_term_encoding: If True, skip term encoding (use prebuilt index embeddings)
    """
    model.eval()
    recall_dict = {k: [] for k in top_ks}
    
    if skip_term_encoding:
        # Use prebuilt index directly
        print(f"[INFO] Using prebuilt index with {retriever.index.ntotal} embeddings")
    else:
        # Rebuild index from used_terms
        text_terms = [term['term'] for term in retriever.term_list]
        print(f"[INFO] Encoding {len(text_terms)} terms...")
        text_emb = encode_texts_in_batches(model, text_terms, device=device)
        
        if text_emb.size(0) == 0:
            print("[WARN] No valid text embeddings, skipping evaluation")
            return {k: [0.0] for k in top_ks}
        
        retriever.index.reset()
        text_emb_numpy = text_emb.detach().cpu().float().numpy()
        retriever.index.add(text_emb_numpy)
    
    # 随机采样评估样本
    import random
    random.seed(42)
    eval_indices = random.sample(range(len(dataset)), min(max_eval, len(dataset)))
    
    # 收集样本
    valid_samples = []
    valid_audio_tensors = []
    for i in eval_indices:
        sample = dataset[i]
        if sample is not None:
            ground_truth_terms, audio_tensor, chunk_text = sample
            if ground_truth_terms and isinstance(audio_tensor, torch.Tensor) and audio_tensor.numel() > 0:
                valid_samples.append(sample)
                valid_audio_tensors.append(audio_tensor)
    
    if not valid_samples:
        print("[WARN] No valid samples found for evaluation")
        return recall_dict
    
    print(f"[INFO] Evaluating on {len(valid_samples)} valid samples")
    
    audio_embs = encode_audio_tensors_in_batches(model, valid_audio_tensors, batch_size=128, device=device)
    audio_embs = audio_embs.detach().cpu().float().numpy()
    
    for j, sample in enumerate(valid_samples):
        ground_truth_terms, _, _ = sample
        gt_terms = [t.lower() for t in ground_truth_terms]
        audio_emb = audio_embs[j:j+1]
        
        for top_k in top_ks:
            D, I = retriever.index.search(audio_emb, top_k)
            retrieved_terms = [retriever.term_list[idx]['term'].lower() for idx in I[0]]
            matched = sum(gt_term in retrieved_terms for gt_term in gt_terms)
            sample_recall = matched / len(gt_terms) if gt_terms else 0.0
            recall_dict[top_k].append(sample_recall)
    
    # 打印结果
    for top_k in top_ks:
        if recall_dict[top_k]:
            avg_recall = sum(recall_dict[top_k]) / len(recall_dict[top_k])
            print(f"[EVAL] Overall Recall@{top_k}: {avg_recall:.2%} ({len(recall_dict[top_k])} samples)")
    
    model.train()
    
    return recall_dict


def main():
    parser = argparse.ArgumentParser(description="本地评估脚本")
    
    # 必需参数
    parser.add_argument('--model_path', type=str, required=True, help='训练好的模型路径')
    parser.add_argument('--train_samples_path', type=str, required=True, help='训练样本JSON路径')
    parser.add_argument('--test_samples_path', type=str, required=True, help='测试样本JSON路径')
    
    # 可选参数
    parser.add_argument('--mmap_shard_dir', type=str, default=None, help='mmap音频分片目录')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2-Audio-7B-Instruct", help='基础模型名称')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
    parser.add_argument('--max_eval', type=int, default=1000, help='最大评估样本数')
    parser.add_argument('--device', type=str, default='cuda', help='使用的GPU设备（单GPU）')
    parser.add_argument('--prebuilt_index', type=str, default=None, help='预构建索引路径（.pkl），如果提供则跳过term编码')
    
    args = parser.parse_args()
    
    # 检查CUDA
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available!")
        return 1
    
    device = torch.device(args.device)
    print(f"[INFO] Using device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"[INFO] Evaluation mode: Single GPU (7B model ~14GB fp16, no sharding needed)")
    
    # 加载数据集
    print("\n" + "="*80)
    print("LOADING DATASETS")
    print("="*80)
    train_dataset = LocalTermLevelDataset(
        args.train_samples_path,
        mmap_shard_dir=args.mmap_shard_dir
    )
    test_dataset = LocalTermLevelDataset(
        args.test_samples_path,
        mmap_shard_dir=args.mmap_shard_dir
    )
    
    # 初始化模型（与训练结构保持一致）
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    print(f"[INFO] Loading base model: {args.model_name}")
    print(f"[INFO] Available GPUs: {torch.cuda.device_count()}")
    
    # 设置环境变量以避免内存碎片
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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
    
    # 清理GPU缓存
    torch.cuda.empty_cache()
    
    # 打印内存使用情况
    print("\n[INFO] GPU Memory Usage after model loading:")
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    print(f"  {device}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Total={total:.2f}GB")
    
    # 加载训练好的权重
    print(f"\n[INFO] Loading trained weights from: {args.model_path}")
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 处理DDP前缀
        if list(state_dict.keys())[0].startswith('module.'):
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v
            state_dict = new_state_dict
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print("[INFO] ✅ Model weights loaded successfully")
        if missing_keys:
            print(f"[INFO] Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"[WARN] Unexpected keys: {unexpected_keys[:5]}...")
        
    except Exception as e:
        print(f"[ERROR] ❌ Failed to load model weights: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 设置为评估模式
    model.eval()
    print("\n[INFO] Model set to evaluation mode")
    
    # 设置检索器
    print("\n" + "="*80)
    print("SETTING UP RETRIEVER")
    print("="*80)
    retriever = SimpleRetriever(enable_fusion=True, device=device)
    retriever.model = model
    
    # 检查是否使用预构建索引
    use_prebuilt_index = args.prebuilt_index and os.path.exists(args.prebuilt_index)
    
    if use_prebuilt_index:
        # 加载预构建索引
        print(f"[INFO] Loading prebuilt index from: {args.prebuilt_index}")
        with open(args.prebuilt_index, 'rb') as f:
            index_data = pickle.load(f)
        
        # faiss 索引是序列化存储的，需要反序列化
        serialized_index = index_data['faiss_index']
        retriever.index = faiss.deserialize_index(serialized_index)
        retriever.term_list = index_data['term_list']
        
        # metadata 可能在顶层或嵌套
        metadata = index_data.get('metadata', index_data)
        print(f"[INFO] ✅ Loaded prebuilt index:")
        print(f"[INFO]   - Terms: {len(retriever.term_list)}")
        print(f"[INFO]   - Index size: {retriever.index.ntotal}")
        print(f"[INFO]   - Embedding dim: {metadata.get('embedding_dim', 'unknown')}")
        if 'model_path' in metadata:
            print(f"[INFO]   - Source model: {metadata.get('model_path')}")
        if 'glossary_path' in metadata:
            print(f"[INFO]   - Source glossary: {metadata.get('glossary_path')}")
        
        # 使用预构建索引时，used_terms 仅用于统计
        used_terms = [t['term'] if isinstance(t, dict) else t for t in retriever.term_list]
    else:
        # 使用 used terms 构建索引
        if args.prebuilt_index:
            print(f"[WARN] Prebuilt index not found: {args.prebuilt_index}")
        print("[INFO] Building index from used terms...")
        
        used_terms_train = extract_all_used_terms(train_dataset)
        used_terms_test = extract_all_used_terms(test_dataset)
        train_terms_set = set(t.lower() for t in used_terms_train if t and len(t.strip()) >= 3)
        test_terms_set = set(t.lower() for t in used_terms_test if t and len(t.strip()) >= 3)
        used_terms = list(train_terms_set | test_terms_set)
        if not used_terms:
            print("[ERROR] No used terms found in dataset")
            train_dataset.close()
            test_dataset.close()
            return 1
        
        retriever.index = faiss.IndexFlatL2(512)
        retriever.term_list = [{'term': t} for t in used_terms]
        print(f"[INFO] Train terms: {len(train_terms_set)}, Test terms: {len(test_terms_set)}")
        print(f"[INFO] Retriever index size: {len(used_terms)} terms (train+test glossary)")
    
    # 运行评估
    print("\n" + "="*80)
    print("RUNNING EVALUATION")
    print("="*80)
    recall_results = evaluate_topk_recall(
        model,
        retriever,
        test_dataset,
        device,
        top_ks=(1, 5, 10, 20, 50),
        max_eval=args.max_eval,
        skip_term_encoding=use_prebuilt_index,
    )
    
    print("\n[RESULTS] ========== EVALUATION RESULTS ==========")
    print(f"[RESULTS] Model: {args.model_path}")
    print(f"[RESULTS] Test Dataset: {len(test_dataset)} total samples")
    if use_prebuilt_index:
        print(f"[RESULTS] Index: {args.prebuilt_index} ({len(used_terms)} terms)")
    else:
        print(f"[RESULTS] Index Size: {len(used_terms)} terms (train+test glossary)")
    print(f"[RESULTS] Evaluated on: {min(args.max_eval, len(test_dataset))} samples")
    print(f"[RESULTS] " + "="*48)
    
    for top_k in [1, 5, 10, 20, 50, 100]:
        if recall_results.get(top_k) and len(recall_results[top_k]) > 0:
            avg_recall = sum(recall_results[top_k]) / len(recall_results[top_k])
            print(f"[RESULTS] Recall@{top_k:3d}: {avg_recall:.2%} ({len(recall_results[top_k])} samples)")
        else:
            print(f"[RESULTS] Recall@{top_k:3d}: No valid results")
    
    # 清理
    train_dataset.close()
    test_dataset.close()
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETED")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
