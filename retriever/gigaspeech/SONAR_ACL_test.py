import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
from tqdm import tqdm
import argparse, os, sys
import faiss
from new_retrieve import Retriever
import soundfile as sf
from chunk_splitter import split_audio_from_path, create_chunked_samples

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

# 导入训练脚本中的模型类
from SONAR_train import ContrastiveSpeechTextModel

# === New imports for offline asset building ===
import math
from pathlib import Path


def l2_normalize_numpy(x: np.ndarray) -> np.ndarray:
    """L2 normalize along last dim for numpy array."""
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def save_offline_assets(
    embeddings: np.ndarray,
    terms: list,
    out_dir: str,
    index_name: str = "glossary_emb.ivfpq.faiss",
    term2idx_name: str = "glossary_term2idx.json",
    terms_txt_name: str = "glossary_terms.txt",
):
    """Save term->idx map and ordered term list for verification."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    term2idx = {t: i for i, t in enumerate(terms)}
    with open(os.path.join(out_dir, term2idx_name), "w") as f:
        json.dump(term2idx, f)
    with open(os.path.join(out_dir, terms_txt_name), "w") as f:
        for t in terms:
            f.write(t + "\n")
    print(f"[ASSET] Saved term2idx -> {os.path.join(out_dir, term2idx_name)}")
    print(f"[ASSET] Saved terms.txt -> {os.path.join(out_dir, terms_txt_name)}")


def build_ivfpq_index(
    xb: np.ndarray,
    use_ip: bool = True,
    nlist: int = 4096,
    pq_m: int = 64,
    pq_bits: int = 8,
    train_size: int = 1000000,
    nprobe: int = 16,
):
    """Build an IVF-PQ index in memory and return it.
    - xb should be L2-normalized if use_ip=True (cosine via inner product).
    """
    d = xb.shape[1]
    metric = faiss.METRIC_INNER_PRODUCT if use_ip else faiss.METRIC_L2

    # Coarse quantizer
    quantizer = faiss.IndexFlatIP(d) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(d)

    # IVF-PQ
    index = faiss.IndexIVFPQ(quantizer, d, nlist, pq_m, pq_bits, metric)

    # Train
    train_size = min(train_size, xb.shape[0])
    print(f"[FAISS] Training IVF-PQ on {train_size} vectors (nlist={nlist}, m={pq_m}, bits={pq_bits})")
    faiss_idx_train = xb[np.random.choice(xb.shape[0], train_size, replace=False)]
    index.train(faiss_idx_train)

    # Add
    index.nprobe = nprobe
    print(f"[FAISS] Adding {xb.shape[0]} vectors to IVF-PQ (nprobe={nprobe})")
    index.add(xb)
    print(f"[FAISS] IVF-PQ ntotal={index.ntotal}")

    return index


def build_sharded_ivfpq_indices(
    xb: np.ndarray,
    terms: list,
    out_dir: str,
    shard_size: int = 2_000_000,
    use_ip: bool = True,
    nlist: int = 4096,
    pq_m: int = 64,
    pq_bits: int = 8,
    train_size: int = 1_000_000,
    nprobe: int = 16,
):
    """Split embeddings into shards and build multiple IVF-PQ indexes: glossary_shard_XX.faiss.
    Returns list of written index paths.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    N = xb.shape[0]
    d = xb.shape[1]
    n_shards = math.ceil(N / shard_size)
    paths = []
    print(f"[FAISS] Sharding {N} vectors into {n_shards} shards (size≈{shard_size})")
    for s in range(n_shards):
        s0 = s * shard_size
        s1 = min((s + 1) * shard_size, N)
        x_shard = xb[s0:s1]
        print(f"[FAISS] Building shard {s+1}/{n_shards} with {x_shard.shape[0]} vectors")
        index = build_ivfpq_index(
            x_shard, use_ip=use_ip, nlist=nlist, pq_m=pq_m, pq_bits=pq_bits, train_size=min(train_size, x_shard.shape[0]), nprobe=nprobe
        )
        shard_path = os.path.join(out_dir, f"glossary_shard_{s:02d}.faiss")
        faiss.write_index(index, shard_path)
        print(f"[ASSET] Shard written -> {shard_path}")
        del index
    return paths


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


def is_audio_valid(audio_path, min_duration=0.01, max_duration=30.0):
    """检查音频文件是否有效"""
    try:
        if not os.path.exists(audio_path):
            return False, "File does not exist"
        
        data, sr = sf.read(audio_path)
        
        # 检查基本属性
        if len(data) == 0:
            return False, "Empty audio file"
        
        duration = len(data) / sr
        if duration < min_duration:
            return False, f"Too short ({duration:.3f}s < {min_duration}s)"
        
        if duration > max_duration:
            return False, f"Too long ({duration:.3f}s > {max_duration}s)"
        
        # 检查是否全静音
        if np.allclose(data, 0, atol=1e-6):
            return False, "All silence"
        
        # 检查是否有NaN或Inf
        if np.isnan(data).any():
            return False, "Contains NaN values"
        
        if np.isinf(data).any():
            return False, "Contains Inf values"
        
        # 检查动态范围
        data_std = np.std(data)
        if data_std < 1e-6:
            return False, f"Very low dynamic range (std={data_std:.2e})"
        
        return True, "Valid"
        
    except Exception as e:
        return False, f"Failed to read: {str(e)}"


def validate_audio_batch(audio_paths, verbose=False):
    """批量验证音频文件，返回有效的路径列表和对应的原始索引"""
    valid_paths = []
    valid_indices = []
    invalid_count = 0
    
    for i, path in enumerate(audio_paths):
        is_valid, reason = is_audio_valid(path)
        if is_valid:
            valid_paths.append(path)
            valid_indices.append(i)
        else:
            invalid_count += 1
            if verbose or invalid_count <= 5:  # 只打印前5个无效文件
                print(f"[WARN] Invalid audio {i}: {path} - {reason}")
    
    if invalid_count > 5:
        print(f"[WARN] ... and {invalid_count - 5} more invalid audio files")
    
    return valid_paths, valid_indices


class TermLevelDataset(Dataset):
    def __init__(self, path=None, split="test", train_ratio=0.99, test_path=None):
        if split == "test" and test_path is not None:
            # 使用独立的测试数据集
            print(f"[INFO] Loading test samples from separate file: {test_path}")
            with open(test_path, "r") as f:
                all_samples = json.load(f)
            # 对于独立测试集，不需要train_ratio分割，直接使用所有样本
            use_split_logic = False
        else:
            # 使用原有的分割逻辑
            if path is None:
                raise ValueError("path must be provided when not using separate test file")
            print(f"[INFO] Loading term-level chunk samples from {path}")
            with open(path, "r") as f:
                all_samples = json.load(f)
            use_split_logic = True
        
        # 过滤有效样本：必须有音频文件、chunk文本和ground truth terms
        valid_samples = []
        invalid_audio_count = 0
        
        for i, s in enumerate(all_samples):
            terms = s.get('term_chunk_audio_ground_truth_terms')
            if not (terms and isinstance(terms, list)):
                continue
            # 过滤术语
            filtered_terms = [
                t for t in terms
                if isinstance(t, str)
                and len(t) >= 3
                and sum(c.isdigit() for c in t) <= len(t) // 2
            ]
            if not filtered_terms:
                continue

            # 过滤前后缀
            black_words = ['yeah','this ']
            black_suffixes = ['years']
            filtered_terms = [
                t for t in filtered_terms 
                if not any(t.lower().startswith(prefix.lower()) for prefix in black_words)
                and not any(t.lower().endswith(suffix.lower()) for suffix in black_suffixes)
            ]
            
            # 替换原列表为过滤后的术语
            s = dict(s)  # 避免直接修改原始数据
            s['term_chunk_audio_ground_truth_terms'] = filtered_terms
            
            # 检查基本条件
            if not (s.get('term_chunk_text', '').strip() and s.get('term_chunk_audio', '')):
                continue
            
            # 检查音频文件有效性
            audio_path = s.get("term_chunk_audio", "")
            is_valid, reason = is_audio_valid(audio_path)
            
            if is_valid:
                valid_samples.append(s)
            else:
                invalid_audio_count += 1
                # 只打印前10个无效音频的详细信息
                if invalid_audio_count <= 10:
                    print(f"[WARN] Skipping sample {i}: {audio_path} - {reason}")
        
        if invalid_audio_count > 10:
            print(f"[WARN] ... and {invalid_audio_count - 10} more samples with invalid audio")
            
        print(f"[INFO] Audio validation: {len(valid_samples)} valid, {invalid_audio_count} invalid")
        
        print(f"[INFO] Filtered {len(valid_samples)} valid term-level samples from {len(all_samples)} total samples")
        
        if use_split_logic:
            # 数据分割：99%训练，1%测试
            import random
            random.seed(42)  # 固定随机种子确保可复现
            random.shuffle(valid_samples)
            
            split_idx = int(len(valid_samples) * train_ratio)
            
            if split == "train":
                self.samples = valid_samples[:split_idx]
                print(f"[INFO] Training split: {len(self.samples)} term-level samples")
            elif split == "test":
                self.samples = valid_samples[split_idx:]
                print(f"[INFO] Test split: {len(self.samples)} term-level samples")
            else:
                raise ValueError(f"Invalid split: {split}. Must be 'train' or 'test'")
        else:
            # 独立测试集，直接使用所有有效样本
            self.samples = valid_samples
            print(f"[INFO] Using separate test dataset: {len(self.samples)} term-level samples")
        
        print(f"[INFO] Loaded {len(self.samples)} term-level samples for {split} split")

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_path = sample["term_chunk_audio"]  # 使用term chunk音频
        chunk_text = sample["term_chunk_text"]   # 使用term chunk文本
        ground_truth_terms = sample.get('term_chunk_audio_ground_truth_terms', [])
        
        return ground_truth_terms, audio_path, chunk_text, True

    def __len__(self):
        return len(self.samples)


def extract_all_used_terms(dataset):
    """提取数据集中所有使用的术语"""
    used_terms = set()
    processed_samples = 0
    valid_samples = 0
    
    for i, sample in enumerate(dataset):
        if sample is None:
            continue
        processed_samples += 1
        ground_truth_terms, audio_path, chunk_text, has_target = sample
        
        if has_target and ground_truth_terms:
            valid_samples += 1
            for t in ground_truth_terms:
                if isinstance(t, str) and len(t.strip()) > 0:
                    used_terms.add(t.lower())
            
            # 调试前几个样本
            if i < 5:
                print(f"[DEBUG] extract_all_used_terms - Sample {i}: ground_truth_terms={ground_truth_terms}, chunk_text='{chunk_text}'")
    
    print(f"[DEBUG] extract_all_used_terms - Processed {processed_samples} samples, {valid_samples} valid samples, {len(used_terms)} unique terms")
    return list(used_terms)


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
                emb = model.encode_text(batch).detach().cpu()
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
                        emb = model.encode_text(mini_batch).detach().cpu()
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
                emb = model.encode_audio(batch_paths).detach().cpu()
                all_embeddings.append(emb)
            except Exception as e:
                print(f"[ERROR] Failed to encode audio batch {batch_num}: {e}")
                sys.stdout.flush()
                print(f"[INFO] Trying single file processing for this batch...")
                sys.stdout.flush()
                # 如果batch失败，尝试单个处理
                for single_path in batch_paths:
                    try:
                        single_emb = model.encode_audio([single_path]).detach().cpu()
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


def evaluate_topk_recall(model, retriever, dataset, device, top_ks=(5, 10, 20), max_eval=1000, field="term", train_terms=None, show_missed_terms=True, glossary_emb_path=None, relaxed_chunk_eval=False):
    """评估top-k召回率，使用sample-level平均，同时收集term-level信息用于分析"""
    model.eval()
    
    # 用于存储sample-level召回率
    recall_dict = {k: [] for k in top_ks}
    
    # 用于存储所有GT术语和对应的检索结果（用于分析未命中术语）
    all_gt_terms_with_retrieval = {k: [] for k in top_ks}  # 每个元素是 (gt_term, is_retrieved, sample_info)
    sample_info_for_debug = []  # 用于调试输出

    # === 构建或加载索引 ===
    if glossary_emb_path and os.path.exists(glossary_emb_path):
        # 直接加载预构建的索引
        print(f'[INFO] Loading pre-built glossary index from {glossary_emb_path}')
        try:
            retriever.index = faiss.read_index(glossary_emb_path)
            print(f'[INFO] Successfully loaded index with {retriever.index.ntotal} vectors')
            
            # 从索引中获取term数量信息，用于统计
            index_size = retriever.index.ntotal
            print(f'[INFO] Pre-built index contains {index_size} terms')
            
            # 如果retriever.term_list为空，创建一个占位符列表用于评估
            if not hasattr(retriever, 'term_list') or not retriever.term_list:
                retriever.term_list = [{'term': f'term_{i}'} for i in range(index_size)]
                print(f'[INFO] Created placeholder term list for evaluation')
        except Exception as e:
            print(f'[WARNING] Failed to load pre-built index: {e}, falling back to text encoding')
            glossary_emb_path = None
    
    if not glossary_emb_path or not os.path.exists(glossary_emb_path):
        # 需要重新构建索引
        text_terms = [term['term'] for term in retriever.term_list]
        print(f'[DEBUG] Building index with {len(text_terms)} terms')
        print(f'[DEBUG] First 10 terms: {text_terms[:10]}')
        print(f'[DEBUG] Last 10 terms: {text_terms[-10:]}')
        
        # 检查是否有重复terms
        unique_terms = set(text_terms)
        print(f'[DEBUG] Unique terms: {len(unique_terms)} / {len(text_terms)}')
        if len(unique_terms) != len(text_terms):
            print(f'[WARNING] Found duplicate terms in retriever.term_list!')
        
        raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        text_emb = encode_texts_in_batches(raw_model, text_terms, device=device)
        
        # 检查embedding是否都相同
        if text_emb.shape[0] > 1:
            first_emb = text_emb[0:1]
            similarities = F.cosine_similarity(first_emb, text_emb, dim=1)
            identical_count = (similarities > 0.99).sum().item()
            print(f'[DEBUG] Embeddings identical to first: {identical_count} / {text_emb.shape[0]}')
            if identical_count > text_emb.shape[0] * 0.8:
                print(f'[ERROR] Most embeddings are identical! This will cause retrieval issues.')

        retriever.index.reset()
        retriever.index.add(text_emb)
        print(f'[DEBUG] Index built with {retriever.index.ntotal} vectors')

    print(f"[INFO] Dataset size: {len(dataset)}")
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
    
    # 检查是否是chunked dataset
    is_chunked_dataset = hasattr(dataset, 'get_audio_tensor')
    
    # 检查是否使用relaxed evaluation模式
    use_relaxed_eval = is_chunked_dataset and hasattr(dataset, 'relaxed_eval') and dataset.relaxed_eval
    if use_relaxed_eval:
        print(f"[INFO] Using relaxed chunk evaluation: all sentence terms assigned to each chunk")
    elif relaxed_chunk_eval:
        print(f"[INFO] Relaxed evaluation requested but dataset doesn't support it")
        use_relaxed_eval = relaxed_chunk_eval  # 兼容手动设置
    
    if is_chunked_dataset:
        print(f"[DEBUG] Using chunked dataset with audio tensors...")
        # 从chunked dataset获取音频tensors
        audio_tensors = []
        chunk_info_list = []
        
        for i in valid_indices:
            audio_tensor = dataset.get_audio_tensor(i)
            chunk_info = dataset.get_chunk_info(i)
            audio_tensors.append(audio_tensor)
            chunk_info_list.append(chunk_info)
        
        print(f"[DEBUG] Collected {len(audio_tensors)} audio tensors for evaluation...")
        
        # 对于chunked数据，我们需要特殊处理音频编码
        # 使用音频tensor而不是文件路径进行编码
        raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        
        # 分批处理音频tensors
        all_audio_embs = []
        batch_size = 100  # 对于tensors，使用较小的batch size
        
        for i in range(0, len(audio_tensors), batch_size):
            batch_tensors = audio_tensors[i:i + batch_size]
            
            # 将tensors保存为临时文件进行编码
            import tempfile
            temp_paths = []
            
            try:
                for j, tensor in enumerate(batch_tensors):
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                        # 确保tensor是正确的格式
                        if tensor.dim() > 1:
                            tensor = tensor.squeeze()
                        
                        # 重采样到16kHz进行SONAR编码
                        if hasattr(torch.nn.functional, 'interpolate'):
                            # 从48kHz重采样到16kHz
                            target_length = int(tensor.shape[-1] * 16000 / 48000)
                            tensor_16k = torch.nn.functional.interpolate(
                                tensor.unsqueeze(0).unsqueeze(0), 
                                size=target_length, 
                                mode='linear', 
                                align_corners=False
                            ).squeeze()
                        else:
                            # 简单的下采样
                            step = 48000 // 16000
                            tensor_16k = tensor[::step]
                        
                        sf.write(tmp_file.name, tensor_16k.detach().numpy(), 16000)
                        temp_paths.append(tmp_file.name)
                
                # 批量编码
                batch_embs = raw_model.encode_audio(temp_paths).detach().cpu()
                all_audio_embs.append(batch_embs)
                
            finally:
                # 清理临时文件
                for temp_path in temp_paths:
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
        
        audio_embs = torch.cat(all_audio_embs, dim=0).detach().numpy()
        
        # 统计chunk信息
        short_chunks = sum(1 for info in chunk_info_list if info['is_short_chunk'])
        print(f"[DEBUG] Audio encoding completed for chunked dataset")
        print(f"[DEBUG] - Total chunks: {len(chunk_info_list)}")
        print(f"[DEBUG] - Short chunks (< 1s): {short_chunks} ({short_chunks/len(chunk_info_list)*100:.1f}%)")
        
    else:
        print(f"[DEBUG] Using traditional dataset with audio paths...")
        # 原有的音频文件路径处理逻辑
        audio_paths = [sample[1] for sample in valid_samples]  # term_chunk_audio paths
        
        # 验证音频文件
        print(f"[DEBUG] Validating {len(audio_paths)} audio files for evaluation...")
        valid_audio_paths, valid_audio_indices = validate_audio_batch(audio_paths, verbose=False)
        
        if len(valid_audio_paths) != len(audio_paths):
            print(f"[WARN] Evaluation: Only {len(valid_audio_paths)}/{len(audio_paths)} audio files are valid")
            # 过滤掉无效的样本
            valid_samples = [valid_samples[i] for i in valid_audio_indices]
            valid_indices = [valid_indices[i] for i in valid_audio_indices]
            audio_paths = valid_audio_paths
        
        if len(audio_paths) == 0:
            print(f"[ERROR] No valid audio files for evaluation!")
            return {k: [] for k in top_ks}
        
        print(f"[DEBUG] Encoding {len(audio_paths)} valid audio files...")
        raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        audio_embs = encode_audios_in_batches(raw_model, audio_paths, batch_size=1000, device=device).detach().numpy()

    for j, (i, sample) in enumerate(zip(valid_indices, valid_samples)):
        ground_truth_terms, audio_path, chunk_text, has_target = sample
        audio_emb = audio_embs[j:j+1]  # shape: [1, 512]
        gt_terms = [t.lower() for t in ground_truth_terms]  # 使用term_chunk_audio_ground_truth_terms

        # 对每个top_k进行检索
        retrieval_results = {}
        for top_k in top_ks:
            D, I = retriever.index.search(audio_emb, top_k)
            retrieved_terms = [retriever.term_list[idx][field].lower() for idx in I[0]]
            retrieval_results[top_k] = (D[0], I[0], retrieved_terms)
            
            # 计算sample-level召回率
            if use_relaxed_eval:
                # 宽松模式：只要有任何一个术语命中就算成功
                has_match = any(gt_term in retrieved_terms for gt_term in gt_terms)
                sample_recall = 1.0 if has_match else 0.0

            else:
                # 严格模式：按比例计算
                matched = sum(gt_term in retrieved_terms for gt_term in gt_terms)
                sample_recall = matched / len(gt_terms) if gt_terms else 0.0

            recall_dict[top_k].append(sample_recall)
            
            # 同时收集term-level信息用于分析未命中术语
            for gt_term in gt_terms:
                is_retrieved = gt_term in retrieved_terms
                sample_info = {
                    'sample_idx': i,
                    'audio_path': audio_path,
                    'chunk_text': chunk_text,
                    'all_gt_terms': gt_terms,
                    'retrieved_terms': retrieved_terms  # 添加检索到的候选术语
                }
                all_gt_terms_with_retrieval[top_k].append((gt_term, is_retrieved, sample_info))

        # 存储样本信息用于调试（只存储第一个top_k的结果）
        if j < 3:  # 只保存前3个样本的详细信息
            first_top_k = top_ks[0]
            D, I, retrieved_terms = retrieval_results[first_top_k]
            matched = sum(gt_term in retrieved_terms for gt_term in gt_terms)
            sample_info_for_debug.append({
                'sample_idx': i,
                'audio_path': audio_path,
                'chunk_text': chunk_text,
                'gt_terms': gt_terms,
                'audio_emb': audio_emb,
                'retrieved_indices': I,
                'retrieved_distances': D,
                'retrieved_terms': retrieved_terms,
                'matched_count': matched,
                'total_gt_count': len(gt_terms)
            })

    # 打印调试信息（前3个样本）
    for debug_info in sample_info_for_debug:
        print(f"[DEBUG] Sample {debug_info['sample_idx']}:")
        print(f"[DEBUG] Audio path: {debug_info['audio_path']}")
        print(f"[DEBUG] Chunk text: {debug_info['chunk_text']}")
        print(f"[DEBUG] GT terms: {debug_info['gt_terms']}")
        print(f"[DEBUG] Audio embedding stats: mean={debug_info['audio_emb'].mean():.4f}, std={debug_info['audio_emb'].std():.4f}")
        print(f"[DEBUG] Retrieved indices: {debug_info['retrieved_indices']}")
        print(f"[DEBUG] Retrieved distances: {debug_info['retrieved_distances']}")
        print(f"[DEBUG] Retrieved terms: {debug_info['retrieved_terms']}")
        print(f"[DEBUG] Match count: {debug_info['matched_count']}/{debug_info['total_gt_count']}")
        
        # 额外检查：看看距离最近的几个terms
        if len(debug_info['retrieved_distances']) > 0:
            print(f"[DEBUG] Closest term distance: {debug_info['retrieved_distances'][0]:.4f}")
            if len(set(debug_info['retrieved_terms'])) == 1:
                print(f"[ERROR] All retrieved terms are identical: '{debug_info['retrieved_terms'][0]}'")
        print(f"[DEBUG] ---")

    # 计算sample-level和term-level召回率
    for top_k in top_ks:
        # Sample-level平均召回率
        avg_recall = sum(recall_dict[top_k]) / len(recall_dict[top_k]) if recall_dict[top_k] else 0.0
        eval_mode = " (Relaxed)" if use_relaxed_eval else " (Strict)"
        print(f"[EVAL] Sample-level Average Recall@{top_k}{eval_mode}: {avg_recall:.2%}")
        
        # Term-level微平均召回率
        term_retrieval_pairs = all_gt_terms_with_retrieval[top_k]
        total_terms = len(term_retrieval_pairs)
        hit_terms = sum(1 for _, is_retrieved, _ in term_retrieval_pairs if is_retrieved)
        term_micro_avg_recall = hit_terms / total_terms if total_terms > 0 else 0.0
        print(f"[EVAL] Term-level Micro-Average Recall@{top_k}: {term_micro_avg_recall:.2%} ({hit_terms}/{total_terms} terms)")
        
        # 计算差异
        diff = avg_recall - term_micro_avg_recall
        if diff > 0:
            print(f"[EVAL] Multi-term sample penalty: -{diff:.2%} (sample-level higher, indicating multi-term samples hurt overall recall)")
        elif diff < 0:
            print(f"[EVAL] Multi-term sample benefit: +{abs(diff):.2%} (term-level higher, indicating multi-term samples help overall recall)")
        else:
            print(f"[EVAL] No difference between sample-level and term-level recall")
        print()
        
    # === 统计和打印未命中的术语 ===
    if show_missed_terms:
        for top_k in top_ks:
            term_retrieval_pairs = all_gt_terms_with_retrieval[top_k]
            missed_terms_info = []
            for gt_term, is_retrieved, sample_info in term_retrieval_pairs:
                if not is_retrieved:
                    missed_terms_info.append((gt_term, sample_info))
            
            print(f"[EVAL] Missed {len(missed_terms_info)} terms for Recall@{top_k}:")
            
            # 按术语分组统计
            missed_terms_count = {}
            for gt_term, sample_info in missed_terms_info:
                if gt_term not in missed_terms_count:
                    missed_terms_count[gt_term] = []
                missed_terms_count[gt_term].append(sample_info)
            
            # 打印未命中术语的详细信息（限制输出数量）
            max_terms_to_show = 20  # 最多显示20个术语
            sorted_missed_terms = sorted(missed_terms_count.items(), key=lambda x: len(x[1]), reverse=True)
            
            for i, (missed_term, sample_infos) in enumerate(sorted_missed_terms):
                if i >= max_terms_to_show:
                    remaining_terms = len(sorted_missed_terms) - max_terms_to_show
                    print(f"[EVAL]   ... and {remaining_terms} more missed terms")
                    break
                    
                print(f"[EVAL]   '{missed_term}' (missed {len(sample_infos)} times):")
                
                # 显示前3个样本的详细信息
                max_samples_to_show = 3
                for j, sample_info in enumerate(sample_infos):
                    if j >= max_samples_to_show:
                        remaining_samples = len(sample_infos) - max_samples_to_show
                        print(f"[EVAL]     ... and {remaining_samples} more samples")
                        break
                        
                    chunk_text_preview = sample_info['chunk_text'][:100] + '...' if len(sample_info['chunk_text']) > 100 else sample_info['chunk_text']
                    audio_basename = sample_info['audio_path'].split('/')[-1] if sample_info['audio_path'] else 'N/A'
                    print(f"[EVAL]     Sample {sample_info['sample_idx']}: {audio_basename}")
                    print(f"[EVAL]       Text: {chunk_text_preview}")
                    print(f"[EVAL]       All GT terms: {sample_info['all_gt_terms']}")
                    print(f"[EVAL]       Retrieved top-{top_k}: {sample_info['retrieved_terms']}")
            
            print()  # 空行分隔
    else:
        for top_k in top_ks:
            term_retrieval_pairs = all_gt_terms_with_retrieval[top_k]
            missed_count = sum(1 for _, is_retrieved, _ in term_retrieval_pairs if not is_retrieved)
            print(f"[EVAL] Missed {missed_count} terms for Recall@{top_k} (details hidden)")
        print()

    # === 计算 seen/unseen recall (both sample-level and term-level) ===
    if train_terms is not None:
        for top_k in top_ks:
            # 只有训练集中的术语才算seen
            seen_terms_set = set(t.lower() for t in train_terms)
            
            # Sample-level seen/unseen分析
            seen_recalls, unseen_recalls = [], []
            for recall_val, sample in zip(recall_dict[top_k], valid_samples):
                gt_terms = [t.lower() for t in sample[0]]
                # 修正逻辑：如果样本中有任何术语在训练集中，则该样本归类为seen
                # 这样可以更好地区分seen和unseen样本，避免过于严格的分类
                if any(gt in seen_terms_set for gt in gt_terms):
                    seen_recalls.append(recall_val)
                else:
                    unseen_recalls.append(recall_val)

            avg_seen = sum(seen_recalls) / len(seen_recalls) if seen_recalls else 0.0
            avg_unseen = sum(unseen_recalls) / len(unseen_recalls) if unseen_recalls else 0.0
            total_samples = len(seen_recalls) + len(unseen_recalls)
            print(f"[EVAL] Sample-level - Seen Recall@{top_k}: {avg_seen:.2%} ({len(seen_recalls)}/{total_samples} samples), Unseen Recall@{top_k}: {avg_unseen:.2%} ({len(unseen_recalls)}/{total_samples} samples)")
            
            # Term-level seen/unseen分析
            term_retrieval_pairs = all_gt_terms_with_retrieval[top_k]
            seen_hits, seen_total = 0, 0
            unseen_hits, unseen_total = 0, 0
            
            for gt_term, is_retrieved, sample_info in term_retrieval_pairs:
                if gt_term in seen_terms_set:
                    seen_total += 1
                    if is_retrieved:
                        seen_hits += 1
                else:
                    unseen_total += 1
                    if is_retrieved:
                        unseen_hits += 1
            
            term_seen_recall = seen_hits / seen_total if seen_total > 0 else 0.0
            term_unseen_recall = unseen_hits / unseen_total if unseen_total > 0 else 0.0
            total_terms = seen_total + unseen_total
            unseen_percentage = unseen_total / total_terms * 100 if total_terms > 0 else 0.0
            
            print(f"[EVAL] Term-level - Seen Recall@{top_k}: {term_seen_recall:.2%} ({seen_hits}/{seen_total} terms), Unseen Recall@{top_k}: {term_unseen_recall:.2%} ({unseen_hits}/{unseen_total} terms)")
            print(f"[EVAL] Unseen Term Percentage: {unseen_percentage:.1f}%")
            print()
    else:
        print(f"[WARN] train_terms not provided, skipping seen/unseen analysis")

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
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    
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


def load_acl_terminology(glossary_csv_path):
    """从ACL术语词汇表中加载英文术语"""
    print(f"[INFO] Loading ACL terminology from {glossary_csv_path}")
    sys.stdout.flush()
    
    terms = []
    with open(glossary_csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if len(lines) > 1:
            # 跳过header行，第一列是英文术语
            for line in lines[1:]:
                parts = line.strip().split(',')
                if len(parts) > 0 and parts[0]:
                    english_term = parts[0].strip().lower()
                    if len(english_term) >= 2:
                        terms.append(english_term)
    
    # 去重
    terms = list(set(terms))
    print(f"[INFO] Loaded {len(terms)} unique English terms from ACL glossary")
    sys.stdout.flush()
    return terms


def parse_acl_tagged_text(tagged_text_path):
    """解析ACL标注的文本文件，提取术语"""
    print(f"[INFO] Parsing ACL tagged text from {tagged_text_path}")
    sys.stdout.flush()
    
    terms = set()
    with open(tagged_text_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 提取方括号中的术语 [term]
            import re
            tagged_terms = re.findall(r'\[([^\]]+)\]', line)
            for term in tagged_terms:
                term = term.strip().lower()
                if len(term) >= 2:
                    terms.add(term)
    
    terms_list = list(terms)
    print(f"[INFO] Extracted {len(terms_list)} unique terms from {tagged_text_path}")
    sys.stdout.flush()
    return terms_list


class ACLDataset(Dataset):
    """ACL数据集类，用于加载ACL segmented音频和对应的术语"""
    
    def __init__(self, acl_root_dir, split="dev", segmentation="gold"):
        """
        Args:
            acl_root_dir: ACL数据集根目录 (如 data/acl-6060/2/acl_6060)
            split: "dev" 或 "eval"
            segmentation: "gold" 或 "shas"
        """
        self.acl_root_dir = acl_root_dir
        self.split = split
        self.segmentation = segmentation
        
        # 构建路径
        self.audio_dir = os.path.join(acl_root_dir, split, "segmented_wavs", segmentation)
        self.text_dir = os.path.join(acl_root_dir, split, "text", "tagged_terminology")
        
        print(f"[INFO] Loading ACL {split} dataset with {segmentation} segmentation")
        print(f"[INFO] Audio dir: {self.audio_dir}")
        print(f"[INFO] Text dir: {self.text_dir}")
        
        # 加载音频文件列表
        self.audio_files = []
        if os.path.exists(self.audio_dir):
            for f in os.listdir(self.audio_dir):
                if f.endswith('.wav'):
                    self.audio_files.append(f)
        
        self.audio_files.sort()  # 按文件名排序
        
        # 加载对应的术语标注文件 (使用英文版本)
        tagged_file = os.path.join(self.text_dir, f"ACL.6060.{split}.tagged.en-xx.en.txt")
        self.tagged_terms = []
        self.sentence_to_terms = {}  # 句子ID到术语的映射
        
        if os.path.exists(tagged_file):
            with open(tagged_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 提取方括号中的术语
                    import re
                    tagged_terms = re.findall(r'\[([^\]]+)\]', line)
                    clean_terms = [term.strip().lower() for term in tagged_terms if len(term.strip()) >= 2]
                    
                    # 句子ID直接从行号+1开始（对应sent_1.wav, sent_2.wav, ...）
                    sent_id = i + 1
                    if clean_terms:
                        self.sentence_to_terms[sent_id] = clean_terms
        
        print(f"[INFO] Loaded {len(self.sentence_to_terms)} sentences with terms from tagged file")
        
        # 过滤出有术语标注的音频文件
        self.valid_samples = []
        for audio_file in self.audio_files:
            # 从文件名提取句子ID (如 sent_1.wav -> 1, sent_401.wav -> 401)
            import re
            match = re.search(r'sent_(\d+)\.wav', audio_file)
            if match:
                sent_id = int(match.group(1))
                if sent_id in self.sentence_to_terms:
                    audio_path = os.path.join(self.audio_dir, audio_file)
                    terms = self.sentence_to_terms[sent_id]
                    
                    # 验证音频文件
                    is_valid, reason = is_audio_valid(audio_path)
                    if is_valid:
                        self.valid_samples.append({
                            'audio_path': audio_path,
                            'terms': terms,
                            'sent_id': sent_id
                        })
                    else:
                        print(f"[WARN] Invalid audio {audio_path}: {reason}")
                else:
                    # 对于没有术语标注的句子，我们跳过
                    pass
        
        print(f"[INFO] ACL {split} dataset: {len(self.valid_samples)} valid samples with terms")
        print(f"[INFO] Sentence ID range: {min(s['sent_id'] for s in self.valid_samples)} - {max(s['sent_id'] for s in self.valid_samples)}")
        sys.stdout.flush()
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        return sample['terms'], sample['audio_path'], f"sent_{sample['sent_id']}", True


class ACLChunkedDataset(Dataset):
    """ACL数据集的chunked版本，将sentence-level音频切分成2秒固定chunk"""
    
    def __init__(self, acl_root_dir, split="dev", segmentation="gold", 
                 chunk_duration=2.0, target_sr=48000, overlap=0.0, 
                 min_chunk_duration=1.0, term_filtering_method="position",
                 save_chunks=False, chunk_save_dir="/mnt/gemini/data/jiaxuanluo/acl_chunks",
                 relaxed_eval=False):
        """
        Args:
            acl_root_dir: ACL数据集根目录 (如 data/acl-6060/2/acl_6060)
            split: "dev" 或 "eval"
            segmentation: "gold" 或 "shas"
            chunk_duration: 每个chunk的时长（秒）
            target_sr: 目标采样率
            overlap: chunk之间的重叠（秒）
            min_chunk_duration: 最小chunk时长，低于此值的chunk会被标记
            term_filtering_method: 术语过滤方法 ("position", "asr", "none")
                - "position": 基于术语在文本中的位置进行分配
                - "asr": 使用ASR转录进行术语匹配（需要额外实现）
                - "none": 不进行过滤，所有chunk包含所有术语（原始方法）
            save_chunks: 是否保存chunk数据到文件
            chunk_save_dir: chunk数据保存目录
            relaxed_eval: 是否使用宽松评估模式（所有sentence术语分配给每个chunk）
        """
        self.acl_root_dir = acl_root_dir
        self.split = split
        self.segmentation = segmentation
        self.chunk_duration = chunk_duration
        self.target_sr = target_sr
        self.overlap = overlap
        self.min_chunk_duration = min_chunk_duration
        self.term_filtering_method = term_filtering_method
        self.save_chunks = save_chunks
        self.chunk_save_dir = chunk_save_dir
        self.relaxed_eval = relaxed_eval
        
        # 构建路径
        self.audio_dir = os.path.join(acl_root_dir, split, "segmented_wavs", segmentation)
        self.text_dir = os.path.join(acl_root_dir, split, "text", "tagged_terminology")
        
        print(f"[INFO] Loading ACL {split} chunked dataset with {segmentation} segmentation")
        print(f"[INFO] Chunk duration: {chunk_duration}s, Min chunk: {min_chunk_duration}s")
        print(f"[INFO] Term filtering method: {term_filtering_method}")
        print(f"[INFO] Relaxed evaluation mode: {relaxed_eval}")
        print(f"[INFO] Audio dir: {self.audio_dir}")
        print(f"[INFO] Text dir: {self.text_dir}")
        
        # 首先加载原始ACL数据
        original_dataset = ACLDataset(acl_root_dir, split, segmentation)
        
        # 加载原始文本（不带标记的版本）用于术语位置分析
        self.sentence_to_raw_text = {}
        if term_filtering_method == "position":
            raw_text_file = os.path.join(self.text_dir, f"ACL.6060.{split}.tagged.en-xx.en.txt")
            if os.path.exists(raw_text_file):
                with open(raw_text_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines):
                        sent_id = i + 1
                        # 移除术语标记 [term] 获得纯文本
                        import re
                        raw_text = re.sub(r'\[([^\]]+)\]', r'\1', line.strip())
                        self.sentence_to_raw_text[sent_id] = raw_text.lower()
                print(f"[INFO] Loaded raw text for {len(self.sentence_to_raw_text)} sentences for position-based filtering")
        
        # 开始chunk切分
        self.chunked_samples = []
        short_chunks_count = 0
        total_chunks_count = 0
        term_filtering_stats = {
            'original_terms': 0,
            'filtered_terms': 0,
            'chunks_with_terms': 0,
            'chunks_without_terms': 0
        }
        
        print(f"[INFO] Processing {len(original_dataset)} original sentences...")
        
        for idx in range(len(original_dataset)):
            terms, audio_path, sent_text, has_target = original_dataset[idx]
            
            # 加载和切分音频
            try:
                # 获取音频信息
                data, sr = sf.read(audio_path)
                original_duration = len(data) / sr
                
                # 使用chunk_splitter进行切分
                audio_chunks = split_audio_from_path(
                    audio_path, 
                    chunk_duration=chunk_duration, 
                    target_sr=target_sr, 
                    overlap=overlap
                )
                
                if not audio_chunks:
                    print(f"[WARN] No chunks generated for {audio_path}")
                    continue
                
                # 为每个chunk创建样本
                for chunk_idx, chunk_tensor in enumerate(audio_chunks):
                    chunk_samples = chunk_tensor.shape[-1]
                    actual_chunk_duration = chunk_samples / target_sr
                    
                    # 检查chunk是否过短
                    is_short_chunk = actual_chunk_duration < min_chunk_duration
                    if is_short_chunk:
                        short_chunks_count += 1
                    
                    total_chunks_count += 1
                    
                    # 进行术语过滤
                    sent_id = original_dataset.valid_samples[idx]['sent_id']
                    if self.relaxed_eval:
                        # 宽松模式：所有chunk使用完整的sentence术语
                        filtered_terms = terms

                    else:
                        # 严格模式：根据设定的方法进行过滤
                        filtered_terms = self._filter_terms_for_chunk(
                            terms, sent_id, chunk_idx, len(audio_chunks), original_duration
                        )

                    
                    # 更新统计
                    term_filtering_stats['original_terms'] += len(terms)
                    term_filtering_stats['filtered_terms'] += len(filtered_terms)
                    if len(filtered_terms) > 0:
                        term_filtering_stats['chunks_with_terms'] += 1
                    else:
                        term_filtering_stats['chunks_without_terms'] += 1
                    
                    # 创建chunk样本信息
                    chunk_sample = {
                        'terms': filtered_terms,  # 使用过滤后的术语
                        'original_terms': terms,  # 保留原始术语用于调试
                        'audio_path': audio_path,  # 原始音频路径（用于调试）
                        'audio_tensor': chunk_tensor,  # chunk音频tensor
                        'chunk_id': f"sent_{sent_id}_chunk_{chunk_idx}",
                        'original_sent_id': sent_id,
                        'chunk_index': chunk_idx,
                        'total_chunks': len(audio_chunks),
                        'chunk_duration': actual_chunk_duration,
                        'original_duration': original_duration,
                        'is_short_chunk': is_short_chunk,  # 标记短chunk
                        'is_chunked': True,
                        'has_target': has_target and (len(filtered_terms) > 0 or self.relaxed_eval)  # 基于过滤后的术语，relaxed模式保留所有chunk
                    }
                    
                    self.chunked_samples.append(chunk_sample)
                    
            except Exception as e:
                print(f"[ERROR] Failed to process {audio_path}: {e}")
                continue
        
        # 统计信息
        valid_chunks = [s for s in self.chunked_samples if s['has_target']]
        print(f"[INFO] ACL chunked dataset statistics:")
        print(f"[INFO] - Total chunks: {total_chunks_count}")
        print(f"[INFO] - Valid chunks (with terms): {len(valid_chunks)}")
        print(f"[INFO] - Short chunks (< {min_chunk_duration}s): {short_chunks_count} ({short_chunks_count/total_chunks_count*100:.1f}%)")
        print(f"[INFO] - Average chunks per sentence: {total_chunks_count / len(original_dataset):.2f}")
        
        # 术语过滤统计
        if term_filtering_method != "none" or relaxed_eval:
            total_original = term_filtering_stats['original_terms']
            total_filtered = term_filtering_stats['filtered_terms']
            chunks_with = term_filtering_stats['chunks_with_terms']
            chunks_without = term_filtering_stats['chunks_without_terms']
            
            if relaxed_eval:
                filtering_desc = f"relaxed (all sentence terms)"
            else:
                filtering_desc = term_filtering_method
            
            print(f"[INFO] Term assignment statistics ({filtering_desc}):")
            print(f"[INFO] - Original terms: {total_original}")
            print(f"[INFO] - Assigned terms: {total_filtered} ({total_filtered/total_original*100:.1f}% retained)")
            print(f"[INFO] - Chunks with terms: {chunks_with}")
            print(f"[INFO] - Chunks without terms: {chunks_without} ({chunks_without/total_chunks_count*100:.1f}%)")
        
        # 只保留有术语的chunks
        self.chunked_samples = valid_chunks
        print(f"[INFO] Final chunked dataset size: {len(self.chunked_samples)}")
        
        # 保存chunk数据（如果启用）
        if save_chunks:
            self._save_chunks_to_files()
        
        sys.stdout.flush()
    
    def __len__(self):
        return len(self.chunked_samples)
    
    def __getitem__(self, idx):
        chunk_sample = self.chunked_samples[idx]
        return (
            chunk_sample['terms'], 
            chunk_sample['chunk_id'],  # 使用chunk_id作为标识
            chunk_sample['chunk_id'],  # chunk text标识
            chunk_sample['has_target']
        )
    
    def get_audio_tensor(self, idx):
        """获取chunk的音频tensor"""
        return self.chunked_samples[idx]['audio_tensor']
    
    def get_chunk_info(self, idx):
        """获取chunk的详细信息"""
        chunk = self.chunked_samples[idx]
        return {
            'chunk_id': chunk['chunk_id'],
            'original_sent_id': chunk['original_sent_id'],
            'chunk_index': chunk['chunk_index'],
            'total_chunks': chunk['total_chunks'],
            'chunk_duration': chunk['chunk_duration'],
            'original_duration': chunk['original_duration'],
            'is_short_chunk': chunk['is_short_chunk'],
            'terms': chunk['terms'],
            'original_terms': chunk.get('original_terms', chunk['terms'])
        }
    
    def _filter_terms_for_chunk(self, terms, sent_id, chunk_idx, total_chunks, original_duration):
        """
        根据不同策略过滤术语，确保每个chunk只包含实际在其内容中的术语
        
        Args:
            terms: 原始术语列表
            sent_id: 句子ID
            chunk_idx: chunk索引
            total_chunks: 总chunk数
            original_duration: 原始音频时长
            
        Returns:
            过滤后的术语列表
        """
        if self.term_filtering_method == "none":
            # 不进行过滤，返回所有术语
            return terms
        
        elif self.term_filtering_method == "position":
            # 基于术语在文本中的位置进行过滤
            return self._filter_terms_by_position(terms, sent_id, chunk_idx, total_chunks)
        
        elif self.term_filtering_method == "asr":
            # 基于ASR的术语过滤（待实现）
            print(f"[WARN] ASR-based term filtering not implemented, falling back to position-based")
            return self._filter_terms_by_position(terms, sent_id, chunk_idx, total_chunks)
        
        else:
            print(f"[WARN] Unknown term filtering method: {self.term_filtering_method}, using 'none'")
            return terms
    
    def _filter_terms_by_position(self, terms, sent_id, chunk_idx, total_chunks):
        """
        基于术语在文本中的位置进行chunk级别的术语过滤
        
        这是一个启发式方法，假设：
        1. 术语在文本中的相对位置对应其在音频中的相对位置
        2. 每个chunk覆盖句子的一个连续时间段
        """
        if sent_id not in self.sentence_to_raw_text:
            # 如果没有原始文本，返回所有术语
            return terms
        
        raw_text = self.sentence_to_raw_text[sent_id]
        text_length = len(raw_text)
        
        if text_length == 0:
            return []
        
        # 计算chunk在文本中的覆盖范围
        chunk_start_ratio = chunk_idx / total_chunks
        chunk_end_ratio = (chunk_idx + 1) / total_chunks
        
        # 转换为文本位置
        text_start = int(chunk_start_ratio * text_length)
        text_end = int(chunk_end_ratio * text_length)
        
        # 扩展边界以处理术语跨边界的情况（添加10%的缓冲区）
        buffer_size = max(10, int(0.1 * (text_end - text_start)))
        text_start = max(0, text_start - buffer_size)
        text_end = min(text_length, text_end + buffer_size)
        
        chunk_text = raw_text[text_start:text_end]
        
        # 过滤术语：只保留在chunk文本范围内的术语
        filtered_terms = []
        for term in terms:
            term_lower = term.lower()
            # 检查术语是否在chunk文本中出现
            if term_lower in chunk_text:
                # 进一步检查：确保是完整的词（避免部分匹配）
                import re
                pattern = r'\b' + re.escape(term_lower) + r'\b'
                if re.search(pattern, chunk_text):
                    filtered_terms.append(term)
        
        return filtered_terms
    
    def _save_chunks_to_files(self):
        """保存chunk数据到文件，包括音频文件和元数据JSON"""
        import os
        from pathlib import Path
        import json
        import soundfile as sf
        
        # 创建保存目录
        save_dir = Path(self.chunk_save_dir)
        audio_dir = save_dir / "audio"
        metadata_dir = save_dir / "metadata"
        
        audio_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[INFO] Saving {len(self.chunked_samples)} chunks to {save_dir}")
        
        # 准备元数据
        chunks_metadata = []
        sample_chunks = []  # 用于抽样检查
        
        for idx, chunk_sample in enumerate(self.chunked_samples):
            chunk_id = chunk_sample['chunk_id']
            
            # 保存音频文件
            audio_tensor = chunk_sample['audio_tensor']
            if audio_tensor.dim() > 1:
                audio_tensor = audio_tensor.squeeze()
            
            audio_filename = f"{chunk_id}.wav"
            audio_path = audio_dir / audio_filename
            
            # 保存为16kHz的wav文件（适合SONAR）
            audio_16k = self._resample_to_16k(audio_tensor)
            sf.write(str(audio_path), audio_16k.detach().numpy(), 16000)
            
            # 准备元数据
            metadata = {
                'chunk_id': chunk_id,
                'original_sent_id': chunk_sample['original_sent_id'],
                'chunk_index': chunk_sample['chunk_index'],
                'total_chunks': chunk_sample['total_chunks'],
                'chunk_duration': chunk_sample['chunk_duration'],
                'original_duration': chunk_sample['original_duration'],
                'is_short_chunk': chunk_sample['is_short_chunk'],
                'audio_file': audio_filename,
                'audio_path_relative': f"audio/{audio_filename}",
                'terms': chunk_sample['terms'],
                'original_terms': chunk_sample.get('original_terms', chunk_sample['terms']),
                'term_filtering_method': self.term_filtering_method,
                'original_audio_path': chunk_sample['audio_path']
            }
            
            chunks_metadata.append(metadata)
            
            # 收集前10个chunk用于抽样检查
            if idx < 10:
                sample_chunks.append(metadata)
        
        # 保存完整元数据
        metadata_file = metadata_dir / f"chunks_{self.split}_{self.segmentation}_{self.term_filtering_method}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_metadata, f, indent=2, ensure_ascii=False)
        
        # 保存抽样检查数据
        sample_file = metadata_dir / f"sample_chunks_{self.split}_{self.segmentation}_{self.term_filtering_method}.json"
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample_chunks, f, indent=2, ensure_ascii=False)
        
        # 创建检查报告
        self._create_inspection_report(save_dir, chunks_metadata, sample_chunks)
        
        print(f"[INFO] ✅ Chunks saved successfully:")
        print(f"[INFO]   - Audio files: {audio_dir} ({len(chunks_metadata)} files)")
        print(f"[INFO]   - Metadata: {metadata_file}")
        print(f"[INFO]   - Sample data: {sample_file}")
        print(f"[INFO]   - Inspection report: {save_dir}/inspection_report.txt")
    
    def _resample_to_16k(self, audio_tensor):
        """将音频从48kHz重采样到16kHz"""
        if audio_tensor.shape[0] == 0:
            return torch.zeros(0)
        
        # 简单的下采样：每3个采样点取1个 (48000/16000 = 3)
        step = 3
        resampled = audio_tensor[::step]
        
        return resampled
    
    def _create_inspection_report(self, save_dir, chunks_metadata, sample_chunks):
        """创建详细的检查报告"""
        report_path = save_dir / "inspection_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("ACL Chunked Dataset Inspection Report\n")
            f.write("=" * 50 + "\n\n")
            
            # 基本统计
            f.write("📊 Basic Statistics:\n")
            f.write(f"  - Total chunks: {len(chunks_metadata)}\n")
            f.write(f"  - Split: {self.split}\n")
            f.write(f"  - Segmentation: {self.segmentation}\n")
            f.write(f"  - Chunk duration: {self.chunk_duration}s\n")
            f.write(f"  - Term filtering method: {self.term_filtering_method}\n")
            f.write(f"  - Target sample rate: {self.target_sr}Hz\n\n")
            
            # 术语统计
            all_terms = []
            all_original_terms = []
            short_chunks = 0
            
            for chunk in chunks_metadata:
                all_terms.extend(chunk['terms'])
                all_original_terms.extend(chunk['original_terms'])
                if chunk['is_short_chunk']:
                    short_chunks += 1
            
            f.write("🔍 Term Filtering Analysis:\n")
            f.write(f"  - Original terms (total): {len(all_original_terms)}\n")
            f.write(f"  - Filtered terms (total): {len(all_terms)}\n")
            f.write(f"  - Retention rate: {len(all_terms)/len(all_original_terms)*100:.1f}%\n")
            f.write(f"  - Unique original terms: {len(set(all_original_terms))}\n")
            f.write(f"  - Unique filtered terms: {len(set(all_terms))}\n")
            f.write(f"  - Short chunks: {short_chunks} ({short_chunks/len(chunks_metadata)*100:.1f}%)\n\n")
            
            # 抽样检查
            f.write("🔬 Sample Inspection (First 10 chunks):\n")
            f.write("-" * 50 + "\n")
            
            for i, chunk in enumerate(sample_chunks):
                f.write(f"\nChunk {i+1}: {chunk['chunk_id']}\n")
                f.write(f"  📁 Audio file: {chunk['audio_file']}\n")
                f.write(f"  ⏱️  Duration: {chunk['chunk_duration']:.2f}s")
                if chunk['is_short_chunk']:
                    f.write(" (SHORT)")
                f.write("\n")
                f.write(f"  📝 Original sentence: {chunk['original_sent_id']}\n")
                f.write(f"  🧩 Chunk position: {chunk['chunk_index']}/{chunk['total_chunks']-1}\n")
                f.write(f"  📚 Original terms: {chunk['original_terms']}\n")
                f.write(f"  ✅ Filtered terms: {chunk['terms']}\n")
                
                if chunk['original_terms'] != chunk['terms']:
                    removed = set(chunk['original_terms']) - set(chunk['terms'])
                    f.write(f"  ❌ Removed terms: {list(removed)}\n")
                
                f.write(f"  🎵 Original audio: {chunk['original_audio_path']}\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("💡 Usage: Check sample chunks to verify term filtering accuracy\n")
            f.write("🎧 Listen to audio files to confirm terms are actually present\n")
            f.write("📈 Compare filtered vs original terms to assess filtering quality\n")


def main():
    parser = argparse.ArgumentParser(description="Full evaluation with complete glossary or ACL dataset")
    parser.add_argument('--model_path', type=str, required=True,
                       help="Path to trained model (.pt file)")
    
    # === ACL evaluation mode ===
    parser.add_argument('--acl_mode', action='store_true',
                       help="Enable ACL evaluation mode")
    parser.add_argument('--acl_root_dir', type=str, 
                       default="data/acl-6060/2/acl_6060",
                       help="Path to ACL dataset root directory")
    parser.add_argument('--acl_glossary_path', type=str,
                       default="data/acl-6060/2/intermediate_files/terminology_glossary.csv",
                       help="Path to ACL terminology glossary CSV file")
    parser.add_argument('--acl_test_split', type=str, default="eval", choices=["dev", "eval"],
                       help="ACL split to use for testing")
    parser.add_argument('--acl_index_split', type=str, default="dev", choices=["dev", "eval"],
                       help="ACL split to use for building index (terminology source)")
    parser.add_argument('--acl_segmentation', type=str, default="gold", choices=["gold", "shas"],
                       help="ACL segmentation type to use")
    parser.add_argument('--acl_chunked', action='store_true',
                       help="Use chunked ACL dataset (2-second chunks)")
    parser.add_argument('--chunk_duration', type=float, default=2.0,
                       help="Duration of each chunk in seconds (for chunked mode)")
    parser.add_argument('--min_chunk_duration', type=float, default=1.0,
                       help="Minimum chunk duration to avoid marking as short")
    parser.add_argument('--term_filtering_method', type=str, default="position", 
                       choices=["position", "asr", "none"],
                       help="Term filtering method for chunked mode")
    parser.add_argument('--save_chunks', action='store_true',
                       help="Save chunked audio and metadata to files for inspection")
    parser.add_argument('--chunk_save_dir', type=str, default="/mnt/gemini/data/jiaxuanluo/acl_chunks",
                       help="Directory to save chunk files")
    parser.add_argument('--relaxed_chunk_eval', action='store_true',
                       help="Use relaxed evaluation: assign all sentence terms to chunks, match if any candidate hits")
    
    # === Original evaluation mode ===
    parser.add_argument('--test_samples_path', type=str, 
                       default="data/xl_term_level_chunks_merged.json",
                       help="Path to test samples (for non-ACL mode)")
    parser.add_argument('--glossary_path', type=str, 
                       default="data/terms/glossary_filtered.json",
                       help="Path to complete glossary file (for non-ACL mode)")
    parser.add_argument('--glossary_emb_path', type=str, default=None,
                       help="Path to pre-built glossary embedding index (.faiss file). If provided, will skip text encoding")
    parser.add_argument('--train_samples_path', type=str,
                       default="data/samples/xl/term_level_chunks_single_0_500000.json",
                       help="Path to training samples for seen/unseen analysis (for non-ACL mode)")
    
    parser.add_argument('--max_eval', type=int, default=1000,
                       help="Maximum number of samples to evaluate")
    parser.add_argument('--batch_size', type=int, default=512,
                       help="Initial batch size for text encoding (will be auto-optimized, max 1024)")
    parser.add_argument('--audio_batch_size', type=int, default=1000,
                       help="Initial batch size for audio encoding (will be auto-optimized, max 128)")

    # === Offline asset building args ===
    parser.add_argument('--build_offline_assets', action='store_true',
                       help='If set, build and save offline glossary assets (embeddings + FAISS index) and exit')
    parser.add_argument('--asset_out_dir', type=str, default='data',
                       help='Output directory for offline assets')
    parser.add_argument('--index_type', type=str, default='ivfpq', choices=['ivfpq', 'flat'],
                       help='Index type to build for offline assets')
    parser.add_argument('--use_ip', action='store_true',
                       help='Use inner-product metric (cosine if vectors are L2-normalized). Default false => L2')
    parser.add_argument('--nlist', type=int, default=4096, help='IVF nlist (coarse clusters)')
    parser.add_argument('--pq_m', type=int, default=64, help='PQ m (number of subvectors)')
    parser.add_argument('--pq_bits', type=int, default=8, help='PQ bits per subvector')
    parser.add_argument('--nprobe', type=int, default=16, help='nprobe for IVF search/add sanity')
    parser.add_argument('--shard_size', type=int, default=0,
                       help='If >0, build multiple sharded indices of at most this many vectors per shard')

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    sys.stdout.flush()

    # === 加载模型 ===
    device = torch.device(device)  # 转换为torch.device对象
    model = load_model(args.model_path, device)

    # === ACL evaluation mode ===
    if args.acl_mode:
        print("\n" + "="*50)
        print("ACL EVALUATION MODE")
        print("="*50)
        sys.stdout.flush()
        
        # 1. 从ACL glossary或dev set构建术语索引
        print(f"[INFO] Building index from ACL {args.acl_index_split} split")
        
        # 方法1: 使用ACL术语词汇表
        if os.path.exists(args.acl_glossary_path):
            print(f"[INFO] Using ACL terminology glossary: {args.acl_glossary_path}")
            index_terms = load_acl_terminology(args.acl_glossary_path)
        else:
            # 方法2: 从dev set的tagged文本中提取术语
            print(f"[INFO] Extracting terms from ACL {args.acl_index_split} tagged text")
            tagged_file = os.path.join(args.acl_root_dir, args.acl_index_split, "text", "tagged_terminology", f"ACL.6060.{args.acl_index_split}.tagged.en-xx.en.txt")
            if os.path.exists(tagged_file):
                index_terms = parse_acl_tagged_text(tagged_file)
            else:
                raise FileNotFoundError(f"ACL tagged text file not found: {tagged_file}")
        
        print(f"[INFO] Building index with {len(index_terms)} ACL terms")
        
        # 2. 加载ACL测试数据集
        if args.acl_chunked:
            print(f"[INFO] Using chunked ACL dataset (chunks: {args.chunk_duration}s)")
            test_dataset = ACLChunkedDataset(
                acl_root_dir=args.acl_root_dir,
                split=args.acl_test_split,
                segmentation=args.acl_segmentation,
                chunk_duration=args.chunk_duration,
                target_sr=48000,
                overlap=0.0,
                min_chunk_duration=args.min_chunk_duration,
                term_filtering_method=args.term_filtering_method,
                save_chunks=args.save_chunks,
                chunk_save_dir=args.chunk_save_dir,
                relaxed_eval=args.relaxed_chunk_eval
            )
        else:
            print(f"[INFO] Using sentence-level ACL dataset")
            test_dataset = ACLDataset(
                acl_root_dir=args.acl_root_dir,
                split=args.acl_test_split,
                segmentation=args.acl_segmentation
            )
        
        # 3. 初始化检索器
        retriever = Retriever(enable_fusion=True, device=device)
        raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        retriever.model = raw_model
        retriever.index = faiss.IndexFlatL2(512)
        retriever.term_list = [{'term': t} for t in index_terms]
        
        # 4. 执行ACL评估
        print(f"\n[INFO] ACL Evaluation Setup:")
        print(f"[INFO] - Index split: {args.acl_index_split} ({len(index_terms)} terms)")
        print(f"[INFO] - Test split: {args.acl_test_split} ({len(test_dataset)} samples)")
        print(f"[INFO] - Segmentation: {args.acl_segmentation}")
        sys.stdout.flush()
        
        recall_results = evaluate_topk_recall(
            model, retriever, test_dataset, device,
            top_ks=(1, 5, 10),
            max_eval=min(args.max_eval, len(test_dataset)),
            train_terms=None,  # ACL模式下不做seen/unseen分析
            show_missed_terms=True,
            glossary_emb_path=None,  # ACL模式下不使用预构建索引
            relaxed_chunk_eval=args.relaxed_chunk_eval
        )
        
        # 5. 保存ACL评估结果
        chunked_suffix = '_chunked' if args.acl_chunked else ''
        relaxed_suffix = '_relaxed' if args.relaxed_chunk_eval else ''
        results_path = args.model_path.replace('.pt', f'_acl_{args.acl_test_split}{chunked_suffix}{relaxed_suffix}_eval_results.json')
        eval_summary = {
            'model_path': args.model_path,
            'acl_mode': True,
            'acl_chunked': args.acl_chunked,
            'relaxed_chunk_eval': args.relaxed_chunk_eval,
            'chunk_duration': args.chunk_duration if args.acl_chunked else None,
            'min_chunk_duration': args.min_chunk_duration if args.acl_chunked else None,
            'term_filtering_method': args.term_filtering_method if args.acl_chunked else None,
            'acl_root_dir': args.acl_root_dir,
            'acl_glossary_path': args.acl_glossary_path,
            'acl_test_split': args.acl_test_split,
            'acl_index_split': args.acl_index_split,
            'acl_segmentation': args.acl_segmentation,
            'index_terms_count': len(index_terms),
            'test_samples': len(test_dataset),
            'evaluated_samples': min(args.max_eval, len(test_dataset)),
            'results': {}
        }
        
        for top_k in [1, 5, 10]:
            if top_k in recall_results and recall_results[top_k]:
                avg_recall = sum(recall_results[top_k]) / len(recall_results[top_k])
                eval_summary['results'][f'recall@{top_k}'] = float(avg_recall)
        
        with open(results_path, 'w') as f:
            json.dump(eval_summary, f, indent=2)
        
        print(f"\n[INFO] ACL evaluation results saved to {results_path}")
        print(f"[INFO] ACL evaluation completed!")
        sys.stdout.flush()
        return

    # === Offline asset building path ===
    if args.build_offline_assets:
        print("[ASSET] Building offline glossary assets...")
        sys.stdout.flush()

        # 1) Load glossary
        glossary_terms = load_glossary_terms(args.glossary_path)
        print(f"[ASSET] Total glossary terms: {len(glossary_terms)}")
        sys.stdout.flush()

        # 2) Encode texts using the SAME text encoder as training
        raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        text_emb_t = encode_texts_in_batches(raw_model, glossary_terms, batch_size=args.batch_size, device=device)
        text_emb = text_emb_t.detach().cpu().numpy()

        # 3) L2 normalize if we plan to use IP (cosine via inner product)
        if args.use_ip:
            text_emb = l2_normalize_numpy(text_emb)
            print("[ASSET] L2-normalized embeddings for IP/cosine metric.")
        else:
            print("[ASSET] Using L2 metric; skipping normalization requirement.")
        sys.stdout.flush()

        # 4) Save maps (term2idx, terms.txt)
        save_offline_assets(text_emb, glossary_terms, args.asset_out_dir)

        # 5) Build FAISS index (ivfpq or flat) — single or sharded
        index_paths = []
        if args.index_type == 'flat':
            # Flat index (primarily for debugging)
            d = text_emb.shape[1]
            if args.use_ip:
                index = faiss.IndexFlatIP(d)
            else:
                index = faiss.IndexFlatL2(d)
            index.add(text_emb)
            out_path = os.path.join(args.asset_out_dir, 'glossary_emb.flat.faiss')
            faiss.write_index(index, out_path)
            print(f"[ASSET] Wrote flat index -> {out_path} (ntotal={index.ntotal})")
            del index
            index_paths.append(out_path)
        else:
            # ivfpq path (recommended)
            if args.shard_size and args.shard_size > 0:
                index_paths = build_sharded_ivfpq_indices(
                    text_emb, glossary_terms, args.asset_out_dir,
                    shard_size=args.shard_size, use_ip=args.use_ip,
                    nlist=args.nlist, pq_m=args.pq_m, pq_bits=args.pq_bits, train_size=min(args.nlist * 100, text_emb.shape[0]), nprobe=args.nprobe
                )
                print(f"[ASSET] Built {len(index_paths)} sharded indices under {args.asset_out_dir}")
            else:
                index = build_ivfpq_index(
                    text_emb, use_ip=args.use_ip, nlist=args.nlist, pq_m=args.pq_m, pq_bits=args.pq_bits, train_size=min(args.nlist * 100, text_emb.shape[0]), nprobe=args.nprobe
                )
                out_path = os.path.join(args.asset_out_dir, 'glossary_emb.ivfpq.faiss')
                faiss.write_index(index, out_path)
                print(f"[ASSET] Wrote IVF-PQ index -> {out_path} (ntotal={index.ntotal})")
                del index
                index_paths.append(out_path)

        print("[ASSET] Offline asset building finished.")
        sys.stdout.flush()
        return

    # === 原有的正常评估模式 ===
    # === 加载测试数据集（独立文件，不按比例切分） ===
    print(f"[INFO] Loading test dataset from {args.test_samples_path}")
    sys.stdout.flush()
    test_dataset = TermLevelDataset(
        None,
        split="test",
        test_path=args.test_samples_path
    )
    print(f"[INFO] Test dataset size: {len(test_dataset)}")
    sys.stdout.flush()

    # === 加载训练样本用于 seen/unseen 统计（使用全部样本） ===
    print(f"[INFO] Loading training samples for seen/unseen from {args.train_samples_path}")
    sys.stdout.flush()
    train_dataset = TermLevelDataset(
        args.train_samples_path,
        split="train",
        train_ratio=1.0
    )
    train_terms = extract_all_used_terms(train_dataset)
    print(f"[INFO] Training terms collected: {len(train_terms)}")
    sys.stdout.flush()

    # === 加载完整术语表并初始化检索器 ===
    if args.glossary_emb_path and os.path.exists(args.glossary_emb_path):
        # 使用预构建索引，不需要加载glossary terms
        print(f"[INFO] Using pre-built glossary index: {args.glossary_emb_path}")
        glossary_terms = []  # 空列表，因为索引中已包含所有信息
    else:
        # 需要加载glossary terms来构建索引
        glossary_terms = load_glossary_terms(args.glossary_path)
        print(f"[INFO] Loaded {len(glossary_terms)} terms from glossary")
    
    retriever = Retriever(enable_fusion=True, device=device)
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    retriever.model = raw_model
    retriever.index = faiss.IndexFlatL2(512)
    
    if not args.glossary_emb_path or not os.path.exists(args.glossary_emb_path):
        # 只有在没有预构建索引时才设置term_list
        retriever.term_list = [{'term': t} for t in glossary_terms]
        print(f"[INFO] Using complete glossary with {len(glossary_terms)} terms")
    else:
        # 使用预构建索引时，term_list会在evaluate函数中从索引获取
        print(f"[INFO] Will load term list from pre-built index")
    
    sys.stdout.flush()

    # === 简要数据集统计 ===
    print(f"\n[INFO] Dataset statistics:")
    if args.glossary_emb_path and os.path.exists(args.glossary_emb_path):
        # 使用预构建索引时的统计
        print(f"[INFO] - Pre-built index terms: {retriever.index.ntotal}")
    else:
        # 使用glossary文件时的统计
        print(f"[INFO] - Glossary terms: {len(glossary_terms)}")
    print(f"[INFO] - Training terms (for seen/unseen): {len(train_terms)}")
    sys.stdout.flush()

    # === 执行完整评估 ===
    print("\n" + "="*50)
    print("FULL EVALUATION WITH COMPLETE GLOSSARY")
    print("="*50)
    sys.stdout.flush()

    recall_results = evaluate_topk_recall(
        model, retriever, test_dataset, device,
        top_ks=(1, 5, 10),
        max_eval=args.max_eval,
        train_terms=train_terms,
        show_missed_terms=True,
        glossary_emb_path=args.glossary_emb_path
    )

    # === 保存评估结果 ===
    results_path = args.model_path.replace('.pt', '_full_eval_results.json')
    eval_summary = {
        'model_path': args.model_path,
        'glossary_path': args.glossary_path,
        'glossary_emb_path': args.glossary_emb_path,
        'test_samples_path': args.test_samples_path,
        'train_samples_path': args.train_samples_path,
        'total_terms': retriever.index.ntotal if args.glossary_emb_path and os.path.exists(args.glossary_emb_path) else len(glossary_terms),
        'train_terms_count': len(train_terms),
        'test_samples': len(test_dataset),
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
