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
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import faiss

# 禁用 tokenizers 的并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 导入模型相关
from Qwen2_Audio_train import (
    Qwen2AudioSpeechEncoder, 
    Qwen2AudioTextEncoder, 
    ContrastiveQwen2AudioModel,
    encode_texts_in_batches,
    SimpleRetriever,
    load_glossary_terms
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
        
        return ground_truth_terms, audio_tensor, chunk_text, sample
    
    def __len__(self):
        return len(self.samples)
    
    def close(self):
        if self.audio_db:
            self.audio_db.close()


def encode_audio_tensors_in_batches(model, audio_tensors, batch_size=64, device="cuda"):
    """批量编码音频张量"""
    all_embeddings = []
    
    for i in range(0, len(audio_tensors), batch_size):
        batch_tensors = audio_tensors[i:i + batch_size]
        try:
            with torch.no_grad():
                embeddings = model.encode_audio(batch_tensors)
            embeddings = embeddings.float().detach()
            all_embeddings.append(embeddings)
            
            # 定期清理GPU缓存
            if (i // batch_size) % 10 == 0:
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[ERROR] Failed to encode audio batch {i//batch_size}: {e}")
            dummy_emb = torch.zeros(len(batch_tensors), 512, dtype=torch.float32, device=device)
            all_embeddings.append(dummy_emb)
            torch.cuda.empty_cache()
    
    return torch.cat(all_embeddings, dim=0)


def evaluate_with_failure_analysis(model, retriever, dataset, device, top_ks=(5, 10), max_eval=1000, use_prebuilt=False):
    """评估模型并记录失败案例"""
    model.eval()
    
    # 如果使用预生成索引，跳过重新编码
    if not use_prebuilt:
        # 重建索引
        text_terms = [term['term'] for term in retriever.term_list]
        print(f"[INFO] Encoding {len(text_terms)} terms...")
        # 使用标准编码函数（单GPU，无设备不匹配问题）
        text_emb = encode_texts_in_batches(model, text_terms, batch_size=256, device=device)
        
        if text_emb.size(0) == 0:
            print("[WARN] No valid text embeddings")
            return {}, []
        
        retriever.index.reset()
        text_emb_numpy = text_emb.detach().cpu().float().numpy()
        retriever.index.add(text_emb_numpy)
    else:
        print(f"[INFO] Using prebuilt index with {retriever.index.ntotal} vectors")
        print(f"[INFO] Skipping term encoding (already done)")
    
    # 随机采样评估样本
    import random
    random.seed(42)
    eval_indices = random.sample(range(len(dataset)), min(max_eval, len(dataset)))
    
    # 收集样本
    valid_samples = []
    valid_audio_tensors = []
    valid_metadata = []
    
    print("[INFO] Loading evaluation samples...")
    for i in tqdm(eval_indices, desc="Loading samples"):
        sample = dataset[i]
        if sample is not None:
            ground_truth_terms, audio_tensor, chunk_text, metadata = sample
            if ground_truth_terms and isinstance(audio_tensor, torch.Tensor) and audio_tensor.numel() > 0:
                valid_samples.append((ground_truth_terms, audio_tensor, chunk_text))
                valid_audio_tensors.append(audio_tensor)
                valid_metadata.append(metadata)
    
    if not valid_samples:
        print("[WARN] No valid samples found")
        return {}, []
    
    print(f"[INFO] Evaluating on {len(valid_samples)} samples...")
    
    # 编码音频（使用小batch size避免OOM）
    audio_embs = encode_audio_tensors_in_batches(model, valid_audio_tensors, batch_size=16, device=device)
    audio_embs = audio_embs.detach().cpu().float().numpy()
    
    # 评估并记录失败案例
    recall_dict = {k: [] for k in top_ks}
    failed_cases = []
    
    for j, sample in enumerate(tqdm(valid_samples, desc="Evaluating")):
        ground_truth_terms, _, chunk_text = sample
        gt_terms = [t.lower() for t in ground_truth_terms]
        audio_emb = audio_embs[j:j+1]
        metadata = valid_metadata[j]
        
        # 检索
        max_k = max(top_ks)
        D, I = retriever.index.search(audio_emb, max_k)
        retrieved_terms = [retriever.term_list[idx]['term'].lower() for idx in I[0]]
        
        # 计算各个k的recall
        sample_failed = False
        for top_k in top_ks:
            retrieved_at_k = retrieved_terms[:top_k]
            matched = sum(gt_term in retrieved_at_k for gt_term in gt_terms)
            sample_recall = matched / len(gt_terms) if gt_terms else 0.0
            recall_dict[top_k].append(sample_recall)
            
            # 记录在top-10失败的案例
            if top_k == 10 and sample_recall == 0.0:
                sample_failed = True
        
        # 记录失败案例（recall@10 = 0）
        if sample_failed:
            failed_cases.append({
                'index': j,
                'text': chunk_text,
                'ground_truth_terms': gt_terms,
                'retrieved_top10': retrieved_terms[:10],
                'retrieved_top50': retrieved_terms[:50],
                'audio_path': metadata.get('term_chunk_audio', 'N/A'),
                'distances': D[0][:10].tolist()
            })
    
    # 打印结果
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    for top_k in top_ks:
        if recall_dict[top_k]:
            avg_recall = sum(recall_dict[top_k]) / len(recall_dict[top_k])
            print(f"Recall@{top_k:3d}: {avg_recall:.2%} ({len(recall_dict[top_k])} samples)")
    
    return recall_dict, failed_cases


def main():
    parser = argparse.ArgumentParser(description="本地评估脚本")
    
    # 必需参数
    parser.add_argument('--model_path', type=str, required=True, help='训练好的模型路径')
    parser.add_argument('--test_samples_path', type=str, required=True, help='测试样本JSON路径')
    parser.add_argument('--glossary_path', type=str, required=True, help='词汇表JSON路径')
    
    # 可选参数
    parser.add_argument('--mmap_shard_dir', type=str, default=None, help='mmap音频分片目录')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2-Audio-7B-Instruct", help='基础模型名称')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
    parser.add_argument('--max_eval', type=int, default=1000, help='最大评估样本数')
    parser.add_argument('--device', type=str, default='cuda:0', help='使用的GPU设备（单GPU）')
    parser.add_argument('--num_failed_cases', type=int, default=10, help='打印失败案例的数量')
    parser.add_argument('--prebuilt_index', type=str, default=None, help='预先生成的索引文件路径（.pkl）')
    
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
    print("LOADING DATASET")
    print("="*80)
    test_dataset = LocalTermLevelDataset(
        args.test_samples_path, 
        mmap_shard_dir=args.mmap_shard_dir
    )
    
    # 加载词汇表
    print("\n" + "="*80)
    print("LOADING GLOSSARY")
    print("="*80)
    print(f"[INFO] Loading glossary from: {args.glossary_path}")
    all_glossary_terms = load_glossary_terms(args.glossary_path)
    all_glossary_terms = list(set(t.lower() for t in all_glossary_terms if t and len(t.strip()) >= 3))
    print(f"[INFO] Loaded {len(all_glossary_terms)} unique terms")
    
    # 初始化模型（使用device_map自动分片）
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    print(f"[INFO] Loading base model: {args.model_name}")
    print(f"[INFO] Available GPUs: {torch.cuda.device_count()}")
    
    # 设置环境变量以避免内存碎片
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # 使用device_map加载模型，自动分片到多GPU
    # 注意：这里我们不能直接使用device参数，因为需要device_map
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
    from peft import LoraConfig, get_peft_model, TaskType
    
    print("[INFO] Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_name)
    
    print("[INFO] Loading model to single GPU (evaluation doesn't need sharding)...")
    # 评估模式：直接加载到单GPU（7B模型fp16约14GB，单卡足够）
    shared_qwen2_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
    ).to(device)
    
    try:
        if hasattr(shared_qwen2_model, "config"):
            setattr(shared_qwen2_model.config, "use_cache", False)
        shared_qwen2_model.gradient_checkpointing_enable()
        print("[INFO] Enabled gradient checkpointing")
    except Exception as e:
        print(f"[WARN] Failed to enable gradient checkpointing: {e}")
    
    # 应用LoRA配置（与训练时保持一致）
    print("\n[INFO] Applying LoRA configuration for evaluation...")
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,  # 评估时禁用dropout
        target_modules=target_modules,
        bias="none",
    )
    shared_qwen2_model = get_peft_model(shared_qwen2_model, lora_config)
    print(f"[INFO] LoRA applied with r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"[INFO] Target modules: {target_modules}")
    
    # 创建encoder wrappers，共享已加载的模型
    speech_encoder = Qwen2AudioSpeechEncoder.__new__(Qwen2AudioSpeechEncoder)
    speech_encoder.device = device
    speech_encoder.model_name = args.model_name
    speech_encoder.processor = processor
    speech_encoder.model = shared_qwen2_model
    speech_encoder._analyze_model_structure()
    
    text_encoder = Qwen2AudioTextEncoder.__new__(Qwen2AudioTextEncoder)
    text_encoder.device = device
    text_encoder.processor = processor
    text_encoder.model = shared_qwen2_model
    text_encoder._analyze_model_structure()
    
    # 获取hidden sizes
    speech_hidden = speech_encoder.get_hidden_size()
    text_hidden = text_encoder.get_hidden_size()
    print(f"[INFO] Speech hidden size: {speech_hidden}, Text hidden size: {text_hidden}")
    
    # 创建一个简化的wrapper类来替代ContrastiveQwen2AudioModel
    import torch.nn as nn
    
    class SimpleContrastiveModel(nn.Module):
        def __init__(self, speech_encoder, text_encoder, speech_hidden, text_hidden, proj_dim, device):
            super().__init__()
            self.speech_encoder = speech_encoder
            self.text_encoder = text_encoder
            self.proj_speech = nn.Linear(speech_hidden, proj_dim).to(device)
            self.proj_text = nn.Linear(text_hidden, proj_dim).to(device)
            self.lora_config = None
            self.actual_lora_params = 0
            self._logged_speech_shape = False
        
        def encode_audio(self, audio_inputs):
            with torch.no_grad():
                emb = self.speech_encoder.predict(audio_inputs)
            if not isinstance(emb, torch.Tensor):
                emb = torch.as_tensor(emb)
            emb = emb.float().to(self.proj_speech.weight.device)
            if emb.dim() == 3:
                emb = emb.mean(dim=1)
            return F.normalize(self.proj_speech(emb), dim=-1)
        
        def encode_text(self, texts):
            with torch.no_grad():
                emb = self.text_encoder.predict(texts)
            if not isinstance(emb, torch.Tensor):
                emb = torch.as_tensor(emb)
            emb = emb.float().to(self.proj_text.weight.device)
            return F.normalize(self.proj_text(emb), dim=-1)
    
    model = SimpleContrastiveModel(
        speech_encoder, text_encoder, 
        speech_hidden, text_hidden, 
        512, device
    )
    
    # 清理GPU缓存
    torch.cuda.empty_cache()
    
    # 打印内存使用情况
    print("\n[INFO] GPU Memory Usage after model loading:")
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    print(f"  {device}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Total={total:.2f}GB")
    
    # 加载训练好的权重（包括LoRA权重和投影层）
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
        
        # 分离投影层权重和LoRA权重
        proj_state_dict = {}
        lora_state_dict = {}
        
        for k, v in state_dict.items():
            if 'proj_speech' in k or 'proj_text' in k:
                proj_state_dict[k] = v
            elif 'lora_' in k or 'base_model' in k:
                # LoRA权重，需要映射到PEFT包装后的模型
                # 从 'speech_qwen2_model.base_model.model.xxx' 
                # 映射到 shared_qwen2_model 的对应路径
                if k.startswith('speech_qwen2_model.') or k.startswith('text_qwen2_model.'):
                    # 去掉 'speech_qwen2_model.' 或 'text_qwen2_model.' 前缀
                    new_key = k.split('.', 1)[1] if '.' in k else k
                    lora_state_dict[new_key] = v
                else:
                    lora_state_dict[k] = v
        
        # 加载投影层权重
        if proj_state_dict:
            model.load_state_dict(proj_state_dict, strict=False)
            print(f"[INFO] ✅ Loaded {len(proj_state_dict)} projection layer weights")
        else:
            print("[WARN] ⚠️  No projection layer weights found in checkpoint")
        
        # 加载LoRA权重到共享的模型
        if lora_state_dict:
            missing_keys, unexpected_keys = shared_qwen2_model.load_state_dict(lora_state_dict, strict=False)
            print(f"[INFO] ✅ Loaded {len(lora_state_dict)} LoRA weights")
            if missing_keys:
                print(f"[INFO] Missing keys (expected for non-LoRA params): {len(missing_keys)} keys")
            if unexpected_keys:
                print(f"[WARN] Unexpected keys: {unexpected_keys[:5]}...")
        else:
            print("[WARN] ⚠️  No LoRA weights found in checkpoint")
            print("[WARN] Model will use randomly initialized LoRA weights")
        
        # 打印加载的权重统计
        print(f"[INFO] Projection weights: {list(proj_state_dict.keys())}")
        print(f"[INFO] LoRA weight count by type:")
        lora_types = {}
        for k in lora_state_dict.keys():
            if 'lora_A' in k:
                lora_types['lora_A'] = lora_types.get('lora_A', 0) + 1
            elif 'lora_B' in k:
                lora_types['lora_B'] = lora_types.get('lora_B', 0) + 1
        for lora_type, count in lora_types.items():
            print(f"  - {lora_type}: {count} layers")
        
    except Exception as e:
        print(f"[ERROR] ❌ Failed to load model weights: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 设置为评估模式
    model.eval()
    shared_qwen2_model.eval()
    print("\n[INFO] Model set to evaluation mode")
    
    # 设置检索器
    print("\n" + "="*80)
    print("SETTING UP RETRIEVER")
    print("="*80)
    retriever = SimpleRetriever(enable_fusion=True, device=device)
    retriever.model = model
    
    # 检查是否使用预生成的索引
    if args.prebuilt_index and os.path.exists(args.prebuilt_index):
        print(f"[INFO] Loading prebuilt index from: {args.prebuilt_index}")
        import pickle
        with open(args.prebuilt_index, 'rb') as f:
            index_data = pickle.load(f)
        
        # 反序列化FAISS索引
        retriever.index = faiss.deserialize_index(index_data['faiss_index'])
        retriever.term_list = index_data['term_list']
        
        print(f"[INFO] ✅ Loaded prebuilt index with {len(retriever.term_list)} terms")
        print(f"[INFO] FAISS index size: {retriever.index.ntotal} vectors")
    else:
        # 现场生成索引（原有逻辑）
        if args.prebuilt_index:
            print(f"[WARN] Prebuilt index not found: {args.prebuilt_index}")
        print(f"[INFO] Building index from scratch...")
        retriever.index = faiss.IndexFlatL2(512)
        retriever.term_list = [{'term': t} for t in all_glossary_terms]
        print(f"[INFO] Retriever index size: {len(all_glossary_terms)} terms")
    
    # 运行评估
    print("\n" + "="*80)
    print("RUNNING EVALUATION")
    print("="*80)
    use_prebuilt = args.prebuilt_index and os.path.exists(args.prebuilt_index)
    recall_results, failed_cases = evaluate_with_failure_analysis(
        model, retriever, test_dataset, device, 
        top_ks=(1, 5, 10, 20, 50), 
        max_eval=args.max_eval,
        use_prebuilt=use_prebuilt
    )
    
    # 打印失败案例
    if failed_cases:
        print("\n" + "="*80)
        print(f"FAILED CASES (Recall@10 = 0, showing first {args.num_failed_cases})")
        print("="*80)
        for i, case in enumerate(failed_cases[:args.num_failed_cases]):
            print(f"\n--- Failed Case #{i+1} ---")
            print(f"Text chunk: {case['text'][:200]}...")
            print(f"Ground truth terms: {case['ground_truth_terms']}")
            print(f"Retrieved top-50: {case['retrieved_top50']}")
            print(f"Audio path: {case['audio_path']}")
            print(f"Top-10 distances: {[f'{d:.4f}' for d in case['distances']]}")
        
        print(f"\n[INFO] Total failed cases (Recall@10=0): {len(failed_cases)}")
        print(f"[INFO] Failure rate: {len(failed_cases)/args.max_eval*100:.1f}%")
    else:
        print("\n[INFO] ✅ No failed cases! All samples had at least one term recalled in top-10.")
    
    # 清理
    test_dataset.close()
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETED")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

