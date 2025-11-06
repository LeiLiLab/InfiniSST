#!/usr/bin/env python3
"""
多GPU并行挖掘Hard Negatives

从现有 checkpoint 离线挖 Hard Negatives：
- 使用多个GPU并行处理音频样本
- 用当前 best ckpt 编码每个 audio 样本
- 在"已建好的 term FAISS 索引（512维）"上搜 TopK
- 去掉 GT，保存每样本的 hard_neg_terms 列表

输出：JSONL，每行 { "audio_key": "...", "hard_negs": ["t1","t2",...], "topk": 200 }

使用说明：
1. 建议设置环境变量: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
2. 如果遇到CUDA OOM错误，减小 --batch_size
3. 每个GPU需要约20-30GB显存

更新模式：
- 使用 --update_existing 参数可以在已有文件基础上增量更新
- 会自动合并新旧hard negatives（去重）
- 保留更多样化的负例，有助于模型学习

示例运行命令：

# 首次生成
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python mine_hard_negatives_multi_gpu.py \
    --samples_path /path/to/samples.json \
    --mmap_dir /path/to/mmap \
    --faiss_index_pkl /path/to/index.pkl \
    --model_path /path/to/model.pt \
    --out_path /path/to/output.jsonl \
    --num_gpus 2 \
    --batch_size 64

# 增量更新（在原有基础上添加新的hard negatives）
python mine_hard_negatives_multi_gpu.py \
    --samples_path /path/to/samples.json \
    --mmap_dir /path/to/mmap \
    --faiss_index_pkl /path/to/index.pkl \
    --model_path /path/to/new_model.pt \
    --out_path /path/to/output.jsonl \
    --num_gpus 2 \
    --batch_size 64 \
    --update_existing
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
import threading
import queue
import time

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


def mine_on_gpu(gpu_id, samples_chunk, chunk_idx, model_config, index, term_list, 
                mmap_dir, batch_size, topk, results_queue, model_load_lock):
    """在指定GPU上处理样本chunk"""
    try:
        device = torch.device(f"cuda:{gpu_id}")
        
        # 使用锁避免并发加载模型时的冲突
        print(f"[GPU {gpu_id}] Waiting to initialize model...")
        with model_load_lock:
            print(f"[GPU {gpu_id}] Initializing model (locked)...")
            
            # 初始化模型
            speech_encoder = Qwen2AudioSpeechEncoder(
                model_name=model_config['model_name'], 
                device=device
            )
            text_encoder = Qwen2AudioTextEncoder(
                model_name=model_config['model_name'], 
                device=device, 
                shared_model=speech_encoder.get_shared_model()
            )
            
            model = ContrastiveQwen2AudioModel(
                speech_encoder, 
                text_encoder, 
                proj_dim=512,
                lora_r=model_config['lora_r'],
                lora_alpha=model_config['lora_alpha'],
                lora_dropout=0.0  # 推理时禁用dropout
            ).to(device)
            
            # 加载训练好的权重
            if model_config['state_dict']:
                model.load_state_dict(model_config['state_dict'], strict=False)
                print(f"[GPU {gpu_id}] ✅ Model weights loaded")
            else:
                print(f"[GPU {gpu_id}] ⚠️ Using randomly initialized model")
            
            model.eval()
            
            # 清理GPU缓存
            torch.cuda.empty_cache()
            print(f"[GPU {gpu_id}] Model initialization complete")
        
        # 初始化 mmap 数据库（如果需要）
        use_mmap = mmap_dir and os.path.exists(mmap_dir)
        mmap_db = None
        if use_mmap:
            print(f"[GPU {gpu_id}] Loading mmap database...")
            mmap_db = MMapAudioCollection(mmap_dir)
        
        def get_audio_tensor(sample):
            """获取音频数据（支持mmap和文件路径）"""
            if use_mmap:
                key = extract_audio_key_from_path(sample["term_chunk_audio"])
                try:
                    wav, sr, _, _ = mmap_db.get_by_key(key)
                    return key, torch.from_numpy(wav.copy()).float()
                except Exception as e:
                    print(f"[GPU {gpu_id} WARN] Failed to load audio key {key}: {e}")
                    return key, torch.zeros(16000, dtype=torch.float32)
            else:
                # 文件路径模式
                path = sample["term_chunk_audio"]
                return path, path
        
        # 打印GPU显存使用情况
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"[GPU {gpu_id}] Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        print(f"[GPU {gpu_id}] Processing {len(samples_chunk)} samples...")
        
        # 批量编码 + 搜索
        results = []
        buf_audio = []
        buf_metadata = []  # (key, gt_set)
        processed_count = 0
        failed_count = 0
        
        for sample in tqdm(samples_chunk, desc=f"GPU {gpu_id}", position=gpu_id):
            key, audio_data = get_audio_tensor(sample)
            gt_set = set(t.lower() for t in sample["term_chunk_audio_ground_truth_terms"])
            
            buf_audio.append(audio_data)
            buf_metadata.append((key, gt_set))
            
            # 达到batch大小或最后一个样本
            if len(buf_audio) >= batch_size or sample == samples_chunk[-1]:
                try:
                    # 编码音频
                    with torch.no_grad():
                        audio_emb = model.encode_audio(buf_audio)
                        audio_emb = audio_emb.detach().cpu().float().numpy()
                    
                    # FAISS搜索
                    D, I = index.search(audio_emb, topk)
                    
                    # 处理每个样本的结果
                    for (audio_key, gt_set), indices in zip(buf_metadata, I):
                        # 过滤掉ground truth术语
                        hard_negs = []
                        for idx in indices:
                            term = term_list[idx]
                            if term not in gt_set:
                                hard_negs.append(term)
                        
                        # 保存结果
                        result = {
                            "audio_key": audio_key,
                            "hard_negs": hard_negs,
                            "topk": topk,
                            "num_gt": len(gt_set)
                        }
                        results.append(result)
                        processed_count += 1
                    
                    # 清理GPU缓存
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"\n[GPU {gpu_id} ERROR] Batch processing failed: {e}")
                    failed_count += len(buf_audio)
                    # 写入空结果
                    for audio_key, gt_set in buf_metadata:
                        result = {
                            "audio_key": audio_key,
                            "hard_negs": [],
                            "topk": topk,
                            "num_gt": len(gt_set)
                        }
                        results.append(result)
                    torch.cuda.empty_cache()
                
                # 清空缓冲区
                buf_audio = []
                buf_metadata = []
        
        # 清理
        if mmap_db:
            mmap_db.close()
        
        del model
        del speech_encoder
        del text_encoder
        torch.cuda.empty_cache()
        
        print(f"[GPU {gpu_id}] ✅ Completed: {processed_count} samples")
        if failed_count > 0:
            print(f"[GPU {gpu_id}] ⚠️ Failed: {failed_count} samples")
        
        # 将结果放入队列
        results_queue.put((chunk_idx, results, processed_count, failed_count))
        
    except Exception as e:
        print(f"[GPU {gpu_id} ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        results_queue.put((chunk_idx, [], 0, len(samples_chunk)))


def main():
    parser = argparse.ArgumentParser(description="Mine hard negatives using multiple GPUs")
    parser.add_argument("--samples_path", required=True, help="训练/测试样本JSON路径")
    parser.add_argument("--mmap_dir", default=None, help="mmap音频分片目录（可选）")
    parser.add_argument("--faiss_index_pkl", required=True, help="预构建的FAISS索引pkl文件")
    parser.add_argument("--model_path", default=None, help="训练好的模型checkpoint路径（可选）")
    parser.add_argument("--model_name", default="Qwen/Qwen2-Audio-7B-Instruct", help="基础模型名称")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--out_path", required=True, help="输出JSONL文件路径")
    parser.add_argument("--topk", type=int, default=200, help="每个样本检索的候选数")
    parser.add_argument("--batch_size", type=int, default=64, help="每个GPU的批处理大小")
    parser.add_argument("--num_gpus", type=int, default=2, help="使用的GPU数量")
    parser.add_argument("--update_existing", action="store_true", help="更新已存在的文件，合并新旧hard negatives")
    
    args = parser.parse_args()
    
    # 检查CUDA
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available!")
        return 1
    
    available_gpus = torch.cuda.device_count()
    if args.num_gpus > available_gpus:
        print(f"[WARN] Requested {args.num_gpus} GPUs but only {available_gpus} available")
        args.num_gpus = available_gpus
    
    print(f"\n{'='*80}")
    print(f"HARD NEGATIVE MINING (MULTI-GPU)")
    print(f"{'='*80}")
    print(f"[INFO] Using {args.num_gpus} GPUs")
    print(f"[INFO] Batch size per GPU: {args.batch_size}")
    print(f"[INFO] Top-K per sample: {args.topk}")
    
    # 检查显存
    for gpu_id in range(args.num_gpus):
        props = torch.cuda.get_device_properties(gpu_id)
        total_memory_gb = props.total_memory / 1024**3
        print(f"[INFO] GPU {gpu_id}: {props.name}, {total_memory_gb:.1f}GB")
    
    # 检查环境变量
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        print("[WARN] PYTORCH_CUDA_ALLOC_CONF not set")
        print("[WARN] Recommend: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    
    # 加载模型配置和权重
    model_config = {
        'model_name': args.model_name,
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'state_dict': None
    }
    
    if args.model_path and os.path.exists(args.model_path):
        print(f"[INFO] Loading model weights from: {args.model_path}")
        try:
            checkpoint = torch.load(args.model_path, map_location='cpu')
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            
            # 处理DDP前缀
            if list(state_dict.keys())[0].startswith("module."):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            
            model_config['state_dict'] = state_dict
            print("[INFO] ✅ Model weights loaded successfully")
        except Exception as e:
            print(f"[ERROR] ❌ Failed to load model weights: {e}")
            print("[WARN] Will use randomly initialized model")
    else:
        print(f"[WARN] No model checkpoint provided or file not found")
        print("[WARN] Using randomly initialized model for mining")
    
    # 加载 FAISS 索引和术语列表
    print(f"\n{'='*80}")
    print("LOADING FAISS INDEX")
    print(f"{'='*80}")
    index, term_list = load_retriever_terms(args.faiss_index_pkl)
    term_set = set(term_list)  # 用于快速查找
    
    # 收集样本
    print(f"\n{'='*80}")
    print("LOADING SAMPLES")
    print(f"{'='*80}")
    samples = list(iter_samples(args.samples_path))
    if len(samples) == 0:
        print("[ERROR] No valid samples found!")
        return 1
    
    print(f"[INFO] Total samples: {len(samples)}")
    
    # 将样本分配到多个GPU
    chunk_size = (len(samples) + args.num_gpus - 1) // args.num_gpus
    sample_chunks = [samples[i:i+chunk_size] for i in range(0, len(samples), chunk_size)]
    
    print(f"[INFO] Split into {len(sample_chunks)} chunks, ~{chunk_size} samples each")
    
    # 准备输出文件
    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    # 启动多个线程，每个GPU一个线程
    print(f"\n{'='*80}")
    print("STARTING PARALLEL MINING")
    print(f"{'='*80}")
    print(f"[INFO] Loading models sequentially to avoid conflicts...")
    
    results_queue = queue.Queue()
    model_load_lock = threading.Lock()  # 用于串行加载模型
    threads = []
    
    for gpu_id, (chunk_idx, sample_chunk) in enumerate(zip(range(len(sample_chunks)), sample_chunks)):
        if gpu_id >= args.num_gpus:
            break
        
        t = threading.Thread(
            target=mine_on_gpu,
            args=(gpu_id, sample_chunk, chunk_idx, model_config, index, term_list,
                  args.mmap_dir, args.batch_size, args.topk, results_queue, model_load_lock)
        )
        t.start()
        threads.append(t)
        
        # 添加小延迟，让线程有序启动，避免并发冲突
        time.sleep(0.5)
    
    # 等待所有线程完成
    for t in threads:
        t.join()
    
    # 收集结果
    print(f"\n{'='*80}")
    print("COLLECTING RESULTS")
    print(f"{'='*80}")
    
    all_results = {}
    total_processed = 0
    total_failed = 0
    
    while not results_queue.empty():
        chunk_idx, results, processed, failed = results_queue.get()
        all_results[chunk_idx] = results
        total_processed += processed
        total_failed += failed
    
    # 合并新旧结果（如果是更新模式）
    if args.update_existing and os.path.exists(args.out_path):
        print(f"\n{'='*80}")
        print("MERGING WITH EXISTING RESULTS")
        print(f"{'='*80}")
        print(f"[INFO] Loading existing hard negatives from: {args.out_path}")
        
        existing_hn = {}
        with open(args.out_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                existing_hn[rec["audio_key"]] = rec
        
        print(f"[INFO] Loaded {len(existing_hn)} existing entries")
        
        # 合并结果
        merged_count = 0
        new_count = 0
        
        for chunk_results in all_results.values():
            for result in chunk_results:
                audio_key = result["audio_key"]
                if audio_key in existing_hn:
                    # 合并hard negatives（去重）
                    old_hn = set(existing_hn[audio_key]["hard_negs"])
                    new_hn = set(result["hard_negs"])
                    combined_hn = list(old_hn | new_hn)  # 合并去重
                    
                    # 更新结果
                    existing_hn[audio_key]["hard_negs"] = combined_hn
                    existing_hn[audio_key]["topk"] = args.topk
                    merged_count += 1
                else:
                    # 新增条目
                    existing_hn[audio_key] = result
                    new_count += 1
        
        print(f"[INFO] ✅ Merged {merged_count} entries, added {new_count} new entries")
        print(f"[INFO] Total entries: {len(existing_hn)}")
        
        # 按顺序写入文件（保持与样本顺序一致）
        print(f"[INFO] Writing merged results to: {args.out_path}")
        with open(args.out_path, "w", encoding="utf-8") as out_file:
            # 先按samples的顺序写入
            use_mmap = args.mmap_dir and os.path.exists(args.mmap_dir)
            for sample in samples:
                if use_mmap:
                    key = extract_audio_key_from_path(sample["term_chunk_audio"])
                else:
                    key = sample["term_chunk_audio"]
                
                if key in existing_hn:
                    out_file.write(json.dumps(existing_hn[key], ensure_ascii=False) + "\n")
                    del existing_hn[key]
            
            # 写入剩余的（可能是旧数据中有但新样本中没有的）
            for result in existing_hn.values():
                out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
    else:
        # 按顺序写入文件
        print(f"[INFO] Writing results to: {args.out_path}")
        with open(args.out_path, "w", encoding="utf-8") as out_file:
            for i in range(len(sample_chunks)):
                if i in all_results:
                    for result in all_results[i]:
                        out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                else:
                    print(f"[WARN] Missing results for chunk {i}")
    
    print(f"\n{'='*80}")
    print("MINING COMPLETED")
    print(f"{'='*80}")
    print(f"[INFO] ✅ Successfully processed: {total_processed} samples")
    if total_failed > 0:
        print(f"[WARN] ⚠️  Failed to process: {total_failed} samples")
    print(f"[INFO] Output saved to: {args.out_path}")
    
    # 统计信息
    print(f"\n[INFO] Analyzing final results...")
    hn_counts = []
    total_entries = 0
    with open(args.out_path, "r") as f:
        for line in f:
            rec = json.loads(line)
            hn_counts.append(len(rec["hard_negs"]))
            total_entries += 1
    
    if hn_counts:
        print(f"[INFO] Hard negatives statistics:")
        print(f"  - Total entries: {total_entries}")
        print(f"  - Average HN per sample: {np.mean(hn_counts):.1f}")
        print(f"  - Median: {np.median(hn_counts):.0f}")
        print(f"  - Min: {np.min(hn_counts)}")
        print(f"  - Max: {np.max(hn_counts)}")
        
        if args.update_existing:
            print(f"\n[INFO] 更新模式统计:")
            print(f"  - 本次处理样本数: {total_processed}")
            print(f"  - 最终条目总数: {total_entries}")
            print(f"  - 平均每样本HN数（合并后）: {np.mean(hn_counts):.1f}")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

