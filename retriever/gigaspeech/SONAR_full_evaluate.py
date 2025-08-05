import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
from tqdm import tqdm
import argparse, os, sys
import faiss
from new_retrieve import Retriever

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

# 导入训练脚本中的模型和数据集类
from SONAR_train import ContrastiveSpeechTextModel, InBatchDataset


def load_glossary_terms(glossary_path):
    """加载完整的术语表"""
    print(f"[INFO] Loading glossary from {glossary_path}")
    sys.stdout.flush()
    with open(glossary_path, "r") as f:
        glossary = json.load(f)
    
    # 提取所有术语，处理不同的数据格式
    terms = []
    if isinstance(glossary, list):
        for item in glossary:
            if isinstance(item, dict):
                # 如果是字典，尝试获取 'term' 或 'text' 字段
                term = item.get('term') or item.get('text') or item.get('word')
                if term:
                    terms.append(term.lower())
            elif isinstance(item, str):
                terms.append(item.lower())
    elif isinstance(glossary, dict):
        # 如果是字典格式，提取所有值
        for key, value in glossary.items():
            if isinstance(value, str):
                terms.append(value.lower())
            elif isinstance(value, dict) and 'term' in value:
                terms.append(value['term'].lower())
    
    # 去重并过滤
    terms = list(set(term for term in terms if term and len(term.strip()) >= 2))
    print(f"[INFO] Loaded {len(terms)} unique terms from glossary")
    sys.stdout.flush()
    return terms


def encode_texts_in_batches(model, texts, batch_size=512, device="cuda", auto_batch_size=True, max_chunk_size=1000000):
    """分批编码文本，支持动态batch_size和分段处理"""
    print(f"[INFO] Text encoding setup:")
    print(f"[INFO] - Model type: {type(model)}")
    print(f"[INFO] - Device count: {torch.cuda.device_count()}")
    print(f"[INFO] - Model device: {next(model.parameters()).device if hasattr(model, 'parameters') else 'N/A'}")
    print(f"[INFO] - Initial batch_size: {batch_size}")
    print(f"[INFO] - Total texts: {len(texts)}")
    sys.stdout.flush()
    
    # 对于大量文本，使用分段处理
    if len(texts) > max_chunk_size:
        print(f"[INFO] 📊 Large dataset detected ({len(texts)} texts)")
        print(f"[INFO] 🔄 Using chunked processing with max_chunk_size={max_chunk_size}")
        sys.stdout.flush()
        
        all_results = []
        num_chunks = (len(texts) + max_chunk_size - 1) // max_chunk_size
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * max_chunk_size
            end_idx = min(start_idx + max_chunk_size, len(texts))
            chunk_texts = texts[start_idx:end_idx]
            
            print(f"[INFO] 📦 Processing chunk {chunk_idx + 1}/{num_chunks} ({len(chunk_texts)} texts)")
            sys.stdout.flush()
            
            # 递归调用处理单个chunk（不会再分段）
            chunk_result = encode_texts_in_batches(
                model, chunk_texts, batch_size, device, auto_batch_size, max_chunk_size=float('inf')
            )
            all_results.append(chunk_result)
            
            # 及时清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"[INFO] ✅ Chunk {chunk_idx + 1}/{num_chunks} completed, shape: {chunk_result.shape}")
            sys.stdout.flush()
        
        # 合并所有chunk的结果
        print(f"[INFO] 🔗 Merging {len(all_results)} chunks...")
        sys.stdout.flush()
        final_result = torch.cat(all_results, dim=0)
        
        # 清理中间结果
        del all_results
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"[INFO] ✅ Chunked processing completed: {final_result.shape}")
        sys.stdout.flush()
        return final_result
    
    # 动态调整batch_size到显存极限
    if auto_batch_size and torch.cuda.is_available():
        print(f"[INFO] Auto-tuning batch size for optimal GPU memory usage...")
        sys.stdout.flush()
        try_bs = batch_size
        test_texts = texts[:min(try_bs * 4, len(texts))]  # 用小样本测试
        
        while try_bs >= 32:  # 最小batch_size
            try:
                print(f"[DEBUG] Testing batch_size: {try_bs}")
                sys.stdout.flush()
                torch.cuda.empty_cache()  # 清理显存
                
                with torch.no_grad():
                    test_batch = test_texts[:try_bs] if len(test_texts) >= try_bs else test_texts
                    _ = model.encode_text(test_batch)
                    torch.cuda.empty_cache()
                
                # 成功了，尝试更大的batch_size（但不超过合理上限）
                max_reasonable_bs = min(1024, batch_size * 2)  # 设置合理上限
                if try_bs < max_reasonable_bs:
                    try_bs = int(try_bs * 1.3)  # 更保守的增长
                else:
                    break
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) or "out of memory" in str(e):
                    try_bs = max(32, try_bs // 2)
                    print(f"[WARNING] OOM, reducing batch_size to {try_bs}")
                    sys.stdout.flush()
                    torch.cuda.empty_cache()
                else:
                    print(f"[ERROR] Unexpected error during batch size testing: {e}")
                    break
        
                         # 再退一步，确保稳定（更保守），并设置绝对上限
        batch_size = max(32, min(1024, int(try_bs * 0.6)))
        print(f"[INFO] ✅ Optimized batch_size: {batch_size} (capped at 1024)")
        sys.stdout.flush()
    
    # 批量编码
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    print(f"[INFO] Encoding {len(texts)} texts in {total_batches} batches of size {batch_size}")
    sys.stdout.flush()
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        if batch_num % max(1, total_batches // 10) == 0 or batch_num <= 3:
            print(f"[INFO] Processing text batch {batch_num}/{total_batches}")
            sys.stdout.flush()
        
        with torch.no_grad():
            try:
                emb = model.encode_text(batch).cpu()
                all_embeddings.append(emb)
                
                # 强制清理显存，防止累积
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"[ERROR] Failed to encode text batch {batch_num}: {e}")
                sys.stdout.flush()
                # 清理显存后重试
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 尝试更小的batch
                for j in range(0, len(batch), batch_size // 4):
                    mini_batch = batch[j:j + batch_size // 4]
                    try:
                        emb = model.encode_text(mini_batch).cpu()
                        all_embeddings.append(emb)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception as e2:
                        print(f"[ERROR] Failed mini-batch encoding: {e2}")
                        sys.stdout.flush()
                        continue
    
    if not all_embeddings:
        raise RuntimeError("No texts were successfully encoded")
    
    result = torch.cat(all_embeddings, dim=0)
    print(f"[INFO] ✅ Text encoding completed: {result.shape}")
    sys.stdout.flush()
    return result


def encode_audios_in_batches(model, audio_paths, batch_size=1000, device="cuda", auto_batch_size=True):
    """分批编码音频，支持动态batch_size优化"""
    print(f"[INFO] Audio encoding setup:")
    print(f"[INFO] - Model type: {type(model)}")
    print(f"[INFO] - Initial batch_size: {batch_size}")
    sys.stdout.flush()
    
    # 动态调整audio batch_size（音频编码更消耗显存）
    if auto_batch_size and torch.cuda.is_available() and len(audio_paths) > batch_size:
        print(f"[INFO] Auto-tuning audio batch size...")
        sys.stdout.flush()
        try_bs = batch_size
        test_paths = audio_paths[:min(try_bs * 2, len(audio_paths))]  # 用小样本测试
        
        while try_bs >= 4:  # 音频最小batch_size
            try:
                print(f"[DEBUG] Testing audio batch_size: {try_bs}")
                sys.stdout.flush()
                torch.cuda.empty_cache()
                
                with torch.no_grad():
                    test_batch = test_paths[:try_bs] if len(test_paths) >= try_bs else test_paths
                    _ = model.encode_audio(test_batch)
                    torch.cuda.empty_cache()
                
                # 成功了，尝试稍大一点的batch_size（音频更保守）
                max_reasonable_bs = min(128, batch_size * 2)  # 音频上限128
                if try_bs < max_reasonable_bs:
                    try_bs = int(try_bs * 1.2)  # 更保守的增长
                else:
                    break
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) or "out of memory" in str(e):
                    try_bs = max(4, try_bs // 2)
                    print(f"[WARNING] Audio OOM, reducing batch_size to {try_bs}")
                    sys.stdout.flush()
                    torch.cuda.empty_cache()
                else:
                    print(f"[ERROR] Unexpected error during audio batch size testing: {e}")
                    break
        
        batch_size = max(4, min(128, int(try_bs * 0.7)))  # 保守一点，上限128
        print(f"[INFO] ✅ Optimized audio batch_size: {batch_size} (capped at 128)")
        sys.stdout.flush()
    
    # 批量编码
    all_embeddings = []
    total_batches = (len(audio_paths) + batch_size - 1) // batch_size
    print(f"[INFO] Encoding {len(audio_paths)} audio files in {total_batches} batches of size {batch_size}")
    sys.stdout.flush()
    
    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        if batch_num % max(1, total_batches // 10) == 0 or batch_num <= 3:
            print(f"[INFO] Processing audio batch {batch_num}/{total_batches}")
            sys.stdout.flush()
        
        with torch.no_grad():
            try:
                emb = model.encode_audio(batch_paths).cpu()
                all_embeddings.append(emb)
            except Exception as e:
                print(f"[ERROR] Failed to encode audio batch {batch_num}: {e}")
                sys.stdout.flush()
                print(f"[INFO] Trying single file processing for this batch...")
                sys.stdout.flush()
                # 如果batch失败，尝试单个处理
                for single_path in batch_paths:
                    try:
                        single_emb = model.encode_audio([single_path]).cpu()
                        all_embeddings.append(single_emb)
                    except Exception as e2:
                        print(f"[ERROR] Failed to encode single audio {single_path}: {e2}")
                        sys.stdout.flush()
                        # 跳过这个音频文件
                        continue
    
    if not all_embeddings:
        raise RuntimeError("No audio files were successfully encoded")
    
    result = torch.cat(all_embeddings, dim=0)
    print(f"[INFO] ✅ Audio encoding completed: {result.shape}")
    sys.stdout.flush()
    return result


def evaluate_topk_recall_full(model, retriever, dataset, device, top_ks=(5, 10, 20), max_eval=1000, field="term", train_terms=None, audio_batch_size=1000):
    """完整评估函数，可被训练和full evaluate脚本复用
    
    Args:
        train_terms: 仅来自训练集的术语列表，用于区分seen/unseen terms
    """
    model.eval()
    recall_dict = {k: [] for k in top_ks}

    # === 重建索引 ===
    text_terms = [term['term'] for term in retriever.term_list]
    print(f'[INFO] Building index with {len(text_terms)} terms')
    sys.stdout.flush()
    
    # 检查模型并行状态
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    is_data_parallel = isinstance(model, torch.nn.DataParallel)
    print(f"[INFO] Model parallel status: DataParallel={is_data_parallel}")
    if is_data_parallel:
        print(f"[INFO] DataParallel will use {len(model.device_ids)} GPUs: {model.device_ids}")
    sys.stdout.flush()
    
    # 文本编码（自动优化batch_size）
    print(f"[INFO] 🚀 Starting text encoding with GPU optimization...")
    sys.stdout.flush()
    text_emb = encode_texts_in_batches(raw_model, text_terms, batch_size=512, device=device, auto_batch_size=True)
    
    # === 构建GPU加速的FAISS索引 ===
    print(f"[INFO] 🚀 Building GPU-accelerated FAISS index...")
    sys.stdout.flush()
    
    embedding_dim = text_emb.shape[1]
    print(f"[INFO] Converting embeddings to numpy (shape: {text_emb.shape})...")
    sys.stdout.flush()
    
    text_emb_np = text_emb.numpy().astype('float32')
    
    # 及时释放PyTorch tensor
    del text_emb
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        # 尝试使用多GPU FAISS索引
        if torch.cuda.device_count() > 1:
            print(f"[INFO] Using multi-GPU FAISS index ({torch.cuda.device_count()} GPUs)")
            sys.stdout.flush()
            
            # 创建CPU索引然后转到所有GPU
            cpu_index = faiss.IndexFlatL2(embedding_dim)
            
            # 分批添加embeddings以避免GPU内存问题
            add_batch_size = min(500000, len(text_emb_np))  # 50万一批
            num_add_batches = (len(text_emb_np) + add_batch_size - 1) // add_batch_size
            
            print(f"[INFO] Adding {len(text_emb_np)} embeddings in {num_add_batches} batches of {add_batch_size}...")
            sys.stdout.flush()
            
            for batch_idx in range(num_add_batches):
                start_idx = batch_idx * add_batch_size
                end_idx = min(start_idx + add_batch_size, len(text_emb_np))
                batch_embeddings = text_emb_np[start_idx:end_idx]
                
                cpu_index.add(batch_embeddings)
                print(f"[INFO] Added batch {batch_idx + 1}/{num_add_batches} to CPU index")
                sys.stdout.flush()
            
            # 转换到GPU
            print(f"[INFO] Converting CPU index to multi-GPU...")
            sys.stdout.flush()
            gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
            
            retriever.index = gpu_index
            print(f"[INFO] ✅ Multi-GPU FAISS index built successfully!")
            
        else:
            # 单GPU FAISS索引
            print(f"[INFO] Using single-GPU FAISS index")
            sys.stdout.flush()
            
            res = faiss.StandardGpuResources()
            # 设置更大的临时内存
            res.setTempMemory(1024 * 1024 * 1024)  # 1GB temp memory
            
            cpu_index = faiss.IndexFlatL2(embedding_dim)
            
            # 分批添加到CPU索引
            add_batch_size = min(1000000, len(text_emb_np))  # 100万一批
            num_add_batches = (len(text_emb_np) + add_batch_size - 1) // add_batch_size
            
            print(f"[INFO] Adding {len(text_emb_np)} embeddings in {num_add_batches} batches...")
            sys.stdout.flush()
            
            for batch_idx in range(num_add_batches):
                start_idx = batch_idx * add_batch_size
                end_idx = min(start_idx + add_batch_size, len(text_emb_np))
                batch_embeddings = text_emb_np[start_idx:end_idx]
                
                cpu_index.add(batch_embeddings)
                print(f"[INFO] Added batch {batch_idx + 1}/{num_add_batches} to CPU index")
                sys.stdout.flush()
            
            # 转换到GPU
            print(f"[INFO] Converting CPU index to GPU...")
            sys.stdout.flush()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            
            retriever.index = gpu_index
            print(f"[INFO] ✅ Single-GPU FAISS index built successfully!")
            
    except Exception as e:
        # 回退到CPU索引
        print(f"[WARNING] GPU FAISS failed ({e}), falling back to CPU index")
        sys.stdout.flush()
        
        cpu_index = faiss.IndexFlatL2(embedding_dim)
        
        # 分批添加以避免内存问题
        add_batch_size = min(2000000, len(text_emb_np))  # 200万一批
        num_add_batches = (len(text_emb_np) + add_batch_size - 1) // add_batch_size
        
        print(f"[INFO] Adding {len(text_emb_np)} embeddings to CPU index in {num_add_batches} batches...")
        sys.stdout.flush()
        
        for batch_idx in range(num_add_batches):
            start_idx = batch_idx * add_batch_size
            end_idx = min(start_idx + add_batch_size, len(text_emb_np))
            batch_embeddings = text_emb_np[start_idx:end_idx]
            
            cpu_index.add(batch_embeddings)
            print(f"[INFO] Added batch {batch_idx + 1}/{num_add_batches} to CPU index")
            sys.stdout.flush()
        
        retriever.index = cpu_index
        print(f"[INFO] ✅ CPU FAISS index built as fallback")
    
    # 清理numpy数组
    del text_emb_np
    sys.stdout.flush()

    print(f"[INFO] Dataset size: {len(dataset)}")
    sys.stdout.flush()
    import random
    random.seed(42)  # 固定随机种子确保可复现
    eval_indices = random.sample(range(len(dataset)), min(max_eval, len(dataset)))
    valid_samples = []
    valid_indices = []
    
    for i in eval_indices:
        sample = dataset[i]
        if sample is not None and sample[3] and sample[0]:  # has_target=True and has ground_truth_terms
            valid_samples.append(sample)
            valid_indices.append(i)

    print(f"[INFO] Selected {len(eval_indices)} samples randomly, {len(valid_samples)} are valid for evaluation")
    print(f"[INFO] Filtered out {len(eval_indices) - len(valid_samples)} samples (no ground truth terms or has_target=False)")
    sys.stdout.flush()
    
    # 使用chunk音频进行编码（分批处理）
    audio_paths = [sample[1] for sample in valid_samples]  # n_chunk_audio paths
    audio_embs = encode_audios_in_batches(raw_model, audio_paths, batch_size=audio_batch_size, device=device, auto_batch_size=True).numpy()

    for j, (i, sample) in enumerate(zip(valid_indices, valid_samples)):
        ground_truth_terms, audio_path, chunk_text, has_target = sample
        audio_emb = audio_embs[j:j+1]  # shape: [1, 512]

        for top_k in top_ks:
            D, I = retriever.index.search(audio_emb, top_k)
            retrieved_terms = [retriever.term_list[idx][field].lower() for idx in I[0]]
            gt_terms = [t.lower() for t in ground_truth_terms]  # 使用n_chunk_audio_ground_truth_terms

            matched = sum(gt in retrieved_terms for gt in gt_terms)
            recall = matched / len(gt_terms) if gt_terms else 0.0
            recall_dict[top_k].append(recall)

            if j < 3 and top_k == top_ks[0]:  # 只打印前3个样本的详细信息
                print(f"[DEBUG] Sample {i}:")
                print(f"[DEBUG] Chunk text: {chunk_text[:100]}...")
                print(f"[DEBUG] GT terms: {gt_terms}")
                print(f"[DEBUG] Retrieved terms: {retrieved_terms}")
                print(f"[DEBUG] Match count: {matched}/{len(gt_terms)}")
                print(f"[DEBUG] Recall: {recall:.2%}")
                sys.stdout.flush()

    # 打印统计结果
    for top_k in top_ks:
        avg_recall = sum(recall_dict[top_k]) / len(recall_dict[top_k]) if recall_dict[top_k] else 0.0
        print(f"[EVAL] Average Recall@{top_k}: {avg_recall:.2%}")
        sys.stdout.flush()

        # === 计算 seen/unseen recall ===
        if train_terms is not None:
            # 只有训练集中的术语才算seen
            seen_terms = set(t.lower() for t in train_terms)
            seen_recalls, unseen_recalls = [], []
            for recall_val, sample in zip(recall_dict[top_k], valid_samples):
                gt_terms = [t.lower() for t in sample[0]]
                if all(gt in seen_terms for gt in gt_terms):
                    seen_recalls.append(recall_val)
                else:
                    unseen_recalls.append(recall_val)

            avg_seen = sum(seen_recalls) / len(seen_recalls) if seen_recalls else 0.0
            avg_unseen = sum(unseen_recalls) / len(unseen_recalls) if unseen_recalls else 0.0
            total_samples = len(seen_recalls) + len(unseen_recalls)
            print(f"[EVAL] Seen Recall@{top_k}: {avg_seen:.2%} ({len(seen_recalls)}/{total_samples} samples), Unseen Recall@{top_k}: {avg_unseen:.2%} ({len(unseen_recalls)}/{total_samples} samples)")
            sys.stdout.flush()
        else:
            print(f"[WARN] train_terms not provided, skipping seen/unseen analysis")
            sys.stdout.flush()

    model.train()
    return recall_dict


def load_model(model_path, device):
    """加载训练好的模型"""
    print(f"[INFO] Loading model from {model_path}")
    sys.stdout.flush()
    
    # 确保device是torch.device对象
    if isinstance(device, str):
        device = torch.device(device)
    
    # 初始化编码器
    try:
        speech_encoder = SpeechToEmbeddingModelPipeline(
            encoder="sonar_speech_encoder_eng", device=device
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize speech encoder: {e}")
        sys.stdout.flush()
        print(f"[INFO] Trying alternative initialization...")
        sys.stdout.flush()
        # 尝试不传递device参数
        speech_encoder = SpeechToEmbeddingModelPipeline(
            encoder="sonar_speech_encoder_eng"
        )
        # 手动移动到设备
        if hasattr(speech_encoder, 'model'):
            speech_encoder.model = speech_encoder.model.to(device)

    try:
        text_encoder = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=device,
            dtype=torch.float32,
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize text encoder: {e}")
        sys.stdout.flush()
        print(f"[INFO] Trying alternative initialization...")
        sys.stdout.flush()
        # 尝试不传递device参数
        text_encoder = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            dtype=torch.float32,
        )
        # 手动移动到设备
        if hasattr(text_encoder, 'model'):
            text_encoder.model = text_encoder.model.to(device)

    # 创建模型（使用默认参数，因为结构需要匹配）
    model = ContrastiveSpeechTextModel(
        speech_encoder, text_encoder, 
        unfreeze_layers=10  # 这个参数不影响推理，只影响训练时的参数冻结
    ).to(device)
    
    # 加载训练好的参数
    state_dict = torch.load(model_path, map_location=device)
    
    # 处理 DataParallel 的情况
    if list(state_dict.keys())[0].startswith('module.'):
        # 移除 'module.' 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v  # 移除 'module.' (7个字符)
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[INFO] Model loaded successfully")
    sys.stdout.flush()
    
    # 自动多GPU包装
    if torch.cuda.device_count() > 1:
        print(f"[INFO] 🚀 Detected {torch.cuda.device_count()} GPUs, wrapping with DataParallel")
        available_gpus = list(range(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model, device_ids=available_gpus)
        print(f"[INFO] ✅ DataParallel enabled on GPUs: {available_gpus}")
        sys.stdout.flush()
    else:
        print(f"[INFO] Single GPU mode: {device}")
        sys.stdout.flush()
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Full evaluation with complete glossary")
    parser.add_argument('--model_path', type=str, required=True,
                       help="Path to trained model (.pt file)")
    parser.add_argument('--test_samples_path', type=str, 
                       default="data/samples/xl/test_mfa_3chunks_samples_0_500000.json",
                       help="Path to test samples")
    parser.add_argument('--glossary_path', type=str, 
                       default="data/terms/glossary_filtered.json",
                       help="Path to complete glossary file")
    parser.add_argument('--train_ratio', type=float, default=0.99,
                       help="Train ratio used during training (to get consistent test split)")
    parser.add_argument('--max_eval', type=int, default=1000,
                       help="Maximum number of samples to evaluate")
    parser.add_argument('--batch_size', type=int, default=512,
                       help="Initial batch size for text encoding (will be auto-optimized, max 1024)")
    parser.add_argument('--audio_batch_size', type=int, default=1000,
                       help="Initial batch size for audio encoding (will be auto-optimized, max 128)")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    sys.stdout.flush()

    # === 加载模型 ===
    device = torch.device(device)  # 转换为torch.device对象
    model = load_model(args.model_path, device)

    # === 加载训练和测试数据集 ===
    print(f"[INFO] Loading test dataset from {args.test_samples_path}")
    sys.stdout.flush()
    test_dataset = InBatchDataset(
        args.test_samples_path, 
        split="test", 
        train_ratio=args.train_ratio
    )
    print(f"[INFO] Test dataset size: {len(test_dataset)}")
    sys.stdout.flush()
    
    # 加载训练数据集以获取训练集术语（用于seen/unseen分析）
    print(f"[INFO] Loading training dataset for seen/unseen analysis...")
    sys.stdout.flush()
    train_dataset = InBatchDataset(
        args.test_samples_path, 
        split="train", 
        train_ratio=args.train_ratio
    )
    
    # 提取训练集中的术语
    def extract_all_used_terms(dataset):
        """提取数据集中所有使用的术语"""
        used_terms = set()
        for i in range(len(dataset)):
            sample = dataset[i]
            if sample is None:
                continue
            ground_truth_terms, audio_path, chunk_text, has_target = sample
            if has_target and ground_truth_terms:
                used_terms.update(t.lower() for t in ground_truth_terms if isinstance(t, str))
        return list(used_terms)
    
    train_terms = extract_all_used_terms(train_dataset)
    test_terms = extract_all_used_terms(test_dataset)
    print(f"[INFO] Found {len(train_terms)} unique terms in training set")
    print(f"[INFO] Found {len(test_terms)} unique terms in test set")
    sys.stdout.flush()
    
    # 分析train/test术语重叠
    train_set = set(train_terms)
    test_set = set(test_terms)
    overlap = train_set.intersection(test_set)
    unseen_terms = test_set - train_set
    print(f"[INFO] Terms overlap between train/test: {len(overlap)} terms")
    print(f"[INFO] Test terms that are unseen in training: {len(unseen_terms)} terms")
    if len(unseen_terms) > 0:
        print(f"[INFO] Example unseen terms: {list(unseen_terms)[:10]}...")
    sys.stdout.flush()

    # === 加载完整术语表 ===
    glossary_terms = load_glossary_terms(args.glossary_path)

    # === 初始化 retriever 使用完整术语表 ===
    retriever = Retriever(enable_fusion=True, device=device)
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    retriever.model = raw_model
    retriever.index = faiss.IndexFlatL2(512)  # 初始化空索引
    retriever.term_list = [{'term': t} for t in glossary_terms]

    print(f"[INFO] Using complete glossary with {len(glossary_terms)} terms")
    sys.stdout.flush()

    # === 分析数据集术语分布 ===
    print(f"\n[INFO] Dataset statistics:")
    print(f"[INFO] - Training terms: {len(train_terms)}")
    print(f"[INFO] - Glossary terms: {len(glossary_terms)}")
    sys.stdout.flush()
    
    # 计算训练集术语在完整词汇表中的覆盖率
    train_terms_set = set(train_terms)
    glossary_terms_set = set(glossary_terms)
    overlap = train_terms_set.intersection(glossary_terms_set)
    coverage = len(overlap) / len(train_terms_set) if train_terms_set else 0
    print(f"[INFO] - Training terms covered in glossary: {len(overlap)}/{len(train_terms)} ({coverage:.1%})")
    sys.stdout.flush()

    # === 执行完整评估 ===
    print("\n" + "="*50)
    print("FULL EVALUATION WITH COMPLETE GLOSSARY")
    print("="*50)
    sys.stdout.flush()
    
    recall_results = evaluate_topk_recall_full(
        model, retriever, test_dataset, device,
        top_ks=(1, 5, 10),
        max_eval=args.max_eval,
        train_terms=train_terms,  # 传入训练集术语用于seen/unseen分析
        audio_batch_size=args.audio_batch_size
    )

    # === 保存评估结果 ===
    results_path = args.model_path.replace('.pt', '_full_eval_results.json')
    eval_summary = {
        'model_path': args.model_path,
        'glossary_path': args.glossary_path,
        'test_samples_path': args.test_samples_path,
        'total_terms': len(glossary_terms),
        'train_terms': len(train_terms),
        'test_terms': len(test_terms),
        'terms_overlap': len(overlap),
        'unseen_terms': len(unseen_terms),
        'test_samples': len(test_dataset),
        'train_samples': len(train_dataset),
        'evaluated_samples': min(args.max_eval, len(test_dataset)),
        'results': {}
    }
    
    for top_k in [1, 5, 10]:
        if top_k in recall_results and recall_results[top_k]:
            avg_recall = sum(recall_results[top_k]) / len(recall_results[top_k])
            eval_summary['results'][f'recall@{top_k}'] = float(avg_recall)
    
    with open(results_path, 'w') as f:
        json.dump(eval_summary, f, indent=2)
    
    print(f"\n[INFO] Evaluation results saved to {results_path}")
    print(f"[INFO] Full evaluation completed!")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
