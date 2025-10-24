#!/usr/bin/env python3
"""
简化的Qwen2-Audio Term-Level DDP训练脚本
专为Modal环境优化，移除了复杂的hard negative mining等功能
"""

import os
import sys

# 禁用 tokenizers 的并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from tqdm import tqdm
import soundfile as sf
import faiss

# 导入模型相关
from Qwen2_Audio_train import (
    Qwen2AudioSpeechEncoder, 
    Qwen2AudioTextEncoder, 
    ContrastiveQwen2AudioModel,
    encode_texts_in_batches, 
    SimpleRetriever
)
from mmap_audio_reader import MMapAudioCollection, extract_audio_key_from_path

# 启用TF32以提高性能
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# 启用cudnn benchmark以优化固定输入大小的性能
torch.backends.cudnn.benchmark = True


def collate_keep(batch):
    """保留样本原样（跳过None），返回list[tuple]"""
    batch = [b for b in batch if b is not None]
    return batch


def worker_init_fn(worker_id):
    """Worker进程初始化函数，设置随机种子"""
    import random
    import numpy as np
    random.seed(42 + worker_id)
    np.random.seed(42 + worker_id)


def is_audio_valid(audio_path, min_duration=0.01, max_duration=30.0):
    """检查音频文件是否有效"""
    try:
        if not os.path.exists(audio_path):
            return False, "File does not exist"
        
        data, sr = sf.read(audio_path)
        
        if len(data) == 0:
            return False, "Empty audio file"
        
        duration = len(data) / sr
        if duration < min_duration or duration > max_duration:
            return False, f"Duration {duration:.3f}s out of range"
        
        if np.allclose(data, 0, atol=1e-6):
            return False, "All silence"
        
        if np.isnan(data).any() or np.isinf(data).any():
            return False, "Contains NaN/Inf values"
        
        return True, "Valid"
        
    except Exception as e:
        return False, f"Failed to read: {str(e)}"


def validate_audio_batch(audio_paths, verbose=False):
    """批量验证音频文件"""
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
            if verbose and invalid_count <= 5:
                print(f"[WARN] Invalid audio {i}: {path} - {reason}")
    
    return valid_paths, valid_indices


class TermLevelDatasetMMap(Dataset):
    """基于 mmap 的 Term-Level 数据集"""
    
    def __init__(self, path, mmap_shard_dir, split="train", train_ratio=0.99, test_path=None, hard_neg_jsonl=None):
        # 加载 JSON 元数据
        if split == "test" and test_path is not None:
            if dist.get_rank() == 0:
                print(f"[INFO] Loading test samples from: {test_path}")
            with open(test_path, "r") as f:
                all_samples = json.load(f)
            use_split_logic = False
        else:
            if dist.get_rank() == 0:
                print(f"[INFO] Loading samples from: {path}")
            with open(path, "r") as f:
                all_samples = json.load(f)
            use_split_logic = True
        
        # 加载 hard negatives 映射
        self.hn_map = {}
        if hard_neg_jsonl and os.path.exists(hard_neg_jsonl):
            if dist.get_rank() == 0:
                print(f"[INFO] Loading hard negatives from: {hard_neg_jsonl}")
            with open(hard_neg_jsonl, "r") as f:
                for line in f:
                    rec = json.loads(line)
                    self.hn_map[rec["audio_key"]] = rec.get("hard_negs", [])
            if dist.get_rank() == 0:
                print(f"[INFO] Loaded hard negatives for {len(self.hn_map)} samples")
        else:
            if dist.get_rank() == 0 and hard_neg_jsonl:
                print(f"[WARN] Hard negative file not found: {hard_neg_jsonl}")
                print(f"[INFO] Training without hard negatives")
        
        # 初始化 mmap 音频数据库
        if dist.get_rank() == 0:
            print(f"[INFO] Initializing mmap audio database from: {mmap_shard_dir}")
        self.audio_db = MMapAudioCollection(mmap_shard_dir)
        
        # 过滤有效样本
        valid_samples = []
        invalid_count = 0
        
        for sample in all_samples:
            # 检查基本字段
            if not (sample.get('term_chunk_text', '').strip() and sample.get('term_chunk_audio', '')):
                continue
            
            # 检查术语
            terms = sample.get('term_chunk_audio_ground_truth_terms', [])
            if not isinstance(terms, list):
                terms = []
            
            # 过滤术语
            filtered_terms = [
                t for t in terms
                if isinstance(t, str) and len(t.strip()) >= 3
            ]
            
            # 只保留有术语的样本
            if not filtered_terms:
                continue
            
            # 检查音频是否在 mmap 数据库中
            audio_path = sample.get("term_chunk_audio", "")
            audio_key = extract_audio_key_from_path(audio_path)
            
            if audio_key in self.audio_db.k2loc:
                sample = dict(sample)
                sample['term_chunk_audio_ground_truth_terms'] = filtered_terms
                sample['audio_key'] = audio_key  # 添加 mmap key
                valid_samples.append(sample)
            else:
                invalid_count += 1
        
        if dist.get_rank() == 0:
            print(f"[INFO] Filtered {len(valid_samples)} valid samples from {len(all_samples)} total")
            print(f"[INFO] Audio files not in mmap: {invalid_count}")
        
        # 划分训练/测试集
        if use_split_logic:
            if split == "train":
                split_idx = int(len(valid_samples) * train_ratio)
                self.samples = valid_samples[:split_idx]
            else:  # test
                split_idx = int(len(valid_samples) * train_ratio)
                self.samples = valid_samples[split_idx:]
        else:
            self.samples = valid_samples
        
        if dist.get_rank() == 0:
            print(f"[INFO] {split} dataset: {len(self.samples)} samples")

    def __getitem__(self, index):
        sample = self.samples[index]
        audio_key = sample["audio_key"]
        chunk_text = sample["term_chunk_text"]
        ground_truth_terms = sample.get('term_chunk_audio_ground_truth_terms', [])
        
        # 从 mmap 数据库读取音频
        try:
            wav, sr, _, _ = self.audio_db.get_by_key(audio_key)
            # 转换为 PyTorch tensor
            audio_tensor = torch.from_numpy(wav.copy()).float()
            
            # 调试：检查音频内容与文本的匹配性
            if dist.get_rank() == 0 and index < 5:  # 只打印前5个样本的详细信息
                print(f"\n[DEBUG] Sample {index} MMAP content check:")
                print(f"[DEBUG] Audio key: {audio_key}")
                print(f"[DEBUG] Audio shape: {audio_tensor.shape}, duration: {len(audio_tensor)/16000:.2f}s")
                print(f"[DEBUG] Audio stats: min={audio_tensor.min():.4f}, max={audio_tensor.max():.4f}, mean={audio_tensor.mean():.4f}")
                print(f"[DEBUG] Text chunk: '{chunk_text[:100]}...'")
                print(f"[DEBUG] Ground truth terms: {ground_truth_terms}")
                # 检查音频是否为全零（静音）
                if torch.allclose(audio_tensor, torch.zeros_like(audio_tensor), atol=1e-6):
                    print(f"[DEBUG] ⚠️  WARNING: Audio appears to be silence!")
                else:
                    print(f"[DEBUG] ✅ Audio contains non-zero values")
                    
        except Exception as e:
            # 如果读取失败，返回零音频
            if dist.get_rank() == 0:
                print(f"[WARN] Failed to load audio for key {audio_key}: {e}")
                if index < 5:  # 详细报告前5个失败的样本
                    print(f"[DEBUG] Sample {index} FAILED to load from MMAP:")
                    print(f"[DEBUG] Audio key: {audio_key}")
                    print(f"[DEBUG] Text chunk: '{chunk_text[:100]}...'")
                    print(f"[DEBUG] Ground truth terms: {ground_truth_terms}")
            audio_tensor = torch.zeros(16000, dtype=torch.float32)  # 1秒的静音
        
        # 获取 hard negatives
        hard_negs = self.hn_map.get(audio_key, [])
        
        return ground_truth_terms, audio_tensor, chunk_text, hard_negs

    def __len__(self):
        return len(self.samples)
    
    def close(self):
        """关闭 mmap 数据库"""
        if hasattr(self, 'audio_db'):
            self.audio_db.close()


class TermLevelDataset(Dataset):
    """简化的Term-Level数据集（原版本，用于向后兼容）"""
    
    def __init__(self, path, split="train", train_ratio=0.99, test_path=None, hard_neg_jsonl=None):
        if split == "test" and test_path is not None:
            # 使用独立的测试数据集
            if dist.get_rank() == 0:
                print(f"[INFO] Loading test samples from: {test_path}")
            with open(test_path, "r") as f:
                all_samples = json.load(f)
            use_split_logic = False
        else:
            if dist.get_rank() == 0:
                print(f"[INFO] Loading samples from: {path}")
            with open(path, "r") as f:
                all_samples = json.load(f)
            use_split_logic = True
        
        # 加载 hard negatives 映射（使用文件路径作为key）
        self.hn_map = {}
        if hard_neg_jsonl and os.path.exists(hard_neg_jsonl):
            if dist.get_rank() == 0:
                print(f"[INFO] Loading hard negatives from: {hard_neg_jsonl}")
            with open(hard_neg_jsonl, "r") as f:
                for line in f:
                    rec = json.loads(line)
                    # 文件路径模式：audio_key 可能是路径
                    self.hn_map[rec["audio_key"]] = rec.get("hard_negs", [])
            if dist.get_rank() == 0:
                print(f"[INFO] Loaded hard negatives for {len(self.hn_map)} samples")
        else:
            if dist.get_rank() == 0 and hard_neg_jsonl:
                print(f"[WARN] Hard negative file not found: {hard_neg_jsonl}")
                print(f"[INFO] Training without hard negatives")
        
        # 过滤有效样本
        valid_samples = []
        invalid_count = 0
        
        for sample in all_samples:
            # 检查基本字段
            if not (sample.get('term_chunk_text', '').strip() and sample.get('term_chunk_audio', '')):
                continue
            
            # 检查术语
            terms = sample.get('term_chunk_audio_ground_truth_terms', [])
            if not isinstance(terms, list):
                terms = []
            
            # 过滤术语
            filtered_terms = [
                t for t in terms
                if isinstance(t, str) and len(t.strip()) >= 3
            ]
            
            # 只保留有术语的样本
            if not filtered_terms:
                continue
            
            # 验证音频文件
            audio_path = sample.get("term_chunk_audio", "")
            is_valid, _ = is_audio_valid(audio_path)
            
            if is_valid:
                sample = dict(sample)
                sample['term_chunk_audio_ground_truth_terms'] = filtered_terms
                valid_samples.append(sample)
            else:
                invalid_count += 1
        
        if dist.get_rank() == 0:
            print(f"[INFO] Filtered {len(valid_samples)} valid samples from {len(all_samples)} total")
            print(f"[INFO] Invalid audio files: {invalid_count}")
        
        if use_split_logic:
            # 数据分割
            import random
            random.seed(42)
            random.shuffle(valid_samples)
            
            split_idx = int(len(valid_samples) * train_ratio)
            
            if split == "train":
                self.samples = valid_samples[:split_idx]
            elif split == "test":
                self.samples = valid_samples[split_idx:]
            else:
                raise ValueError(f"Invalid split: {split}")
        else:
            self.samples = valid_samples
        
        if dist.get_rank() == 0:
            print(f"[INFO] {split} dataset: {len(self.samples)} samples")

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_path = sample["term_chunk_audio"]
        chunk_text = sample["term_chunk_text"]
        ground_truth_terms = sample.get('term_chunk_audio_ground_truth_terms', [])
        
        # 获取 hard negatives（使用音频路径作为key）
        hard_negs = self.hn_map.get(audio_path, [])
        
        return ground_truth_terms, audio_path, chunk_text, hard_negs

    def __len__(self):
        return len(self.samples)


def train_step(model, batch, device, args, temperature=0.07, global_random_terms=None):
    """简化的训练步骤 - 支持 hard negatives"""
    raw_model = model.module if isinstance(model, DDP) else model

    if len(batch) < 2:
        return None  # 返回None表示无效batch

    # 解包数据（包含 hard negatives）
    ground_truth_terms_list, audio_paths, chunk_texts, hard_neg_terms_list = zip(*batch)
    
    # 小写处理
    ground_truth_terms_list = [[t.lower() for t in terms if isinstance(t, str)] for terms in ground_truth_terms_list]
    chunk_texts = [text.lower() if isinstance(text, str) else "" for text in chunk_texts]

    # 兼容两种输入：
    # - 文件路径（TermLevelDataset）→ 需要进行路径校验与过滤
    # - 张量（TermLevelDatasetMMap）→ 跳过路径校验，直接送入编码器
    is_path_input = isinstance(audio_paths[0], str)
    if is_path_input:
        valid_audio_paths, valid_indices = validate_audio_batch(audio_paths, verbose=False)
        if len(valid_audio_paths) < 2:
            return None  # 返回None表示无效batch
        if len(valid_audio_paths) != len(audio_paths):
            valid_batch_data = []
            for idx in valid_indices:
                valid_batch_data.append((
                    ground_truth_terms_list[idx],
                    audio_paths[idx], 
                    chunk_texts[idx]
                ))
            ground_truth_terms_list, audio_paths, chunk_texts = zip(*valid_batch_data)
            ground_truth_terms_list = list(ground_truth_terms_list)
            audio_paths = list(audio_paths)
            chunk_texts = list(chunk_texts)
    else:
        # 张量输入：确保数量足够
        if len(audio_paths) < 2:
            return None

    try:
        # 编码音频和文本（encode_audio 需同时支持路径或张量列表）
        audio_emb = raw_model.encode_audio(audio_paths)
        text_emb = raw_model.encode_text(chunk_texts) if args.audio_text_loss_ratio > 0 else torch.zeros_like(audio_emb)
        
        # 检查NaN/Inf
        if torch.isnan(audio_emb).any() or torch.isinf(audio_emb).any():
            return None  # 返回None表示编码失败
        
    except Exception as e:
        if dist.get_rank() == 0:
            print(f"[ERROR] Encoding failed: {e}")
        return None  # 返回None表示编码异常

    batch_size = len(audio_paths)
    
    # 音频-文本对比损失
    contrastive_loss = torch.tensor(0.0, device=device)
    if args.audio_text_loss_ratio > 0:
        try:
            sim_matrix = (audio_emb @ text_emb.T) / temperature
            labels = torch.arange(batch_size, dtype=torch.long, device=device)
            loss_a2t = F.cross_entropy(sim_matrix, labels)
            loss_t2a = F.cross_entropy(sim_matrix.T, labels)
            contrastive_loss = (loss_a2t + loss_t2a) / 2
        except Exception as e:
            if dist.get_rank() == 0:
                print(f"[ERROR] Contrastive loss failed: {e}")
                print(f"[DEBUG] sim_matrix shape: {sim_matrix.shape if 'sim_matrix' in locals() else 'undefined'}")
                print(f"[DEBUG] labels shape: {labels.shape if 'labels' in locals() else 'undefined'}")
                print(f"[DEBUG] batch_size: {batch_size}")

    # 音频-术语对比损失
    audio_term_loss = torch.tensor(0.0, device=device)
    all_gt_terms = []
    audio_term_pairs = []
    
    for i, terms in enumerate(ground_truth_terms_list):
        for term in terms:
            if term and len(term.strip()) > 0:
                term_idx = len(all_gt_terms)
                all_gt_terms.append(term.strip())
                audio_term_pairs.append((i, term_idx))
    
    if len(all_gt_terms) > 0:
        try:
            # 文本端不回传梯度，减小图与显存
            with torch.no_grad():
                terms_emb = raw_model.encode_text(all_gt_terms)
            terms_emb = terms_emb.detach()
            audio_term_sim = (audio_emb @ terms_emb.T) / temperature
            
            # 构建标签
            audio_term_labels = []
            for i in range(batch_size):
                positive_terms = [term_idx for audio_idx, term_idx in audio_term_pairs if audio_idx == i]
                if positive_terms:
                    import random
                    audio_term_labels.append(random.choice(positive_terms))
                else:
                    audio_term_labels.append(-1)
            
            # 计算损失
            valid_indices = [i for i, label in enumerate(audio_term_labels) if label >= 0]
            if len(valid_indices) > 0:
                valid_sim = audio_term_sim[valid_indices]
                valid_labels = torch.tensor([audio_term_labels[i] for i in valid_indices], dtype=torch.long, device=device)
                audio_term_loss = F.cross_entropy(valid_sim, valid_labels)
                
        except Exception as e:
            if dist.get_rank() == 0:
                print(f"[ERROR] Audio-term loss failed: {e}")
                print(f"[DEBUG] audio_emb shape: {audio_emb.shape if 'audio_emb' in locals() else 'undefined'}")
                print(f"[DEBUG] terms_emb shape: {terms_emb.shape if 'terms_emb' in locals() else 'undefined'}")
                print(f"[DEBUG] audio_term_sim shape: {audio_term_sim.shape if 'audio_term_sim' in locals() else 'undefined'}")
                print(f"[DEBUG] valid_indices: {valid_indices if 'valid_indices' in locals() else 'undefined'}")
                print(f"[DEBUG] valid_labels shape: {valid_labels.shape if 'valid_labels' in locals() else 'undefined'}")
                print(f"[DEBUG] all_gt_terms count: {len(all_gt_terms)}")
                print(f"[DEBUG] audio_term_pairs count: {len(audio_term_pairs)}")

    # Hard Negative 对比损失（如果提供了 hard negatives）
    hard_neg_loss = torch.tensor(0.0, device=device)
    if hasattr(args, 'hard_neg_loss_ratio') and args.hard_neg_loss_ratio > 0 and global_random_terms is not None:
        # 检查是否有任何样本有 hard negatives
        has_hn = any(len(hn) > 0 for hn in hard_neg_terms_list)
        
        if has_hn:
            try:
                # 为每个样本构造候选词列表：正例 + hard neg + 少量随机 neg
                candidate_terms_list = []
                pos_label_indices = []  # 每个样本的正例位置（简化为第一个正例）
                max_hn = getattr(args, 'max_hn_per_sample', 15)
                rand_neg = getattr(args, 'rand_neg_per_sample', 5)
                
                import random
                for i, (gts, hns) in enumerate(zip(ground_truth_terms_list, hard_neg_terms_list)):
                    # 正例（去重）
                    gts = list(dict.fromkeys([t for t in gts if t]))
                    if not gts:  # 如果没有正例，跳过
                        candidate_terms_list.append([])
                        pos_label_indices.append(-1)
                        continue
                    
                    # 采样 hard negatives
                    chosen_hn = hns[:max_hn] if len(hns) >= max_hn else hns
                    
                    # 采样随机 negatives（从全局术语库中）
                    chosen_rn = []
                    if rand_neg > 0 and len(global_random_terms) > 0:
                        # 确保不选到 ground truth 或 hard negatives
                        gt_hn_set = set(gts + chosen_hn)
                        available_randoms = [t for t in global_random_terms if t not in gt_hn_set]
                        if available_randoms:
                            chosen_rn = random.sample(available_randoms, min(rand_neg, len(available_randoms)))
                    
                    # 拼接候选：正例放前面（便于标签为0的单正例CE）
                    candidates = list(dict.fromkeys(gts + chosen_hn + chosen_rn))
                    candidate_terms_list.append(candidates)
                    pos_label_indices.append(0)  # 第一个正例
                
                # 过滤掉无效样本
                valid_indices = [i for i, cands in enumerate(candidate_terms_list) if len(cands) > 1]
                
                if len(valid_indices) > 0:
                    # 一次性编码所有候选文本（去重后编码，再映射回）
                    flat_terms = []
                    offsets = []
                    for i in valid_indices:
                        offsets.append(len(flat_terms))
                        flat_terms.extend(candidate_terms_list[i])
                    
                    # 编码候选术语（不需要梯度）
                    with torch.no_grad():
                        term_emb_all = raw_model.encode_text(flat_terms)
                        term_emb_all = term_emb_all.detach()
                    
                    # 计算每个有效样本的 logits
                    logits_list = []
                    labels_list = []
                    for idx, i in enumerate(valid_indices):
                        start = offsets[idx]
                        end = offsets[idx] + len(candidate_terms_list[i])
                        term_emb = term_emb_all[start:end]  # [Ci, 512]
                        
                        # [1,512] x [512,Ci] -> [1,Ci]
                        logit = (audio_emb[i:i+1] @ term_emb.T) / temperature
                        logits_list.append(logit)
                        labels_list.append(torch.tensor([pos_label_indices[i]], device=device, dtype=torch.long))
                    
                    # 拼成批次计算 CE
                    if logits_list:
                        # 由于每个样本的候选数不同，需要逐个计算loss然后平均
                        losses = []
                        for logit, label in zip(logits_list, labels_list):
                            loss_i = F.cross_entropy(logit, label)
                            losses.append(loss_i)
                        hard_neg_loss = torch.stack(losses).mean()
                
            except Exception as e:
                if dist.get_rank() == 0:
                    print(f"[ERROR] Hard negative loss computation failed: {e}")
                    import traceback
                    traceback.print_exc()
    
    # 组合损失
    total_loss = (args.audio_text_loss_ratio * contrastive_loss + 
                  args.audio_term_loss_ratio * audio_term_loss +
                  getattr(args, 'hard_neg_loss_ratio', 0.0) * hard_neg_loss)
    
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        return None  # 返回None表示损失计算异常
    
    return total_loss


def extract_all_used_terms(dataset):
    """提取数据集中所有使用的术语"""
    used_terms = set()
    for i, sample in enumerate(dataset):
        if sample is None:
            continue
        try:
            ground_truth_terms, _, _ = sample
            for t in ground_truth_terms:
                if isinstance(t, str) and len(t.strip()) > 0:
                    used_terms.add(t.lower())
        except Exception as e:
            print(f"[DEBUG] Error extracting terms from sample {i}: {e}")
            print(f"[DEBUG] Sample: {sample}")
            continue
    
    print(f"[DEBUG] extract_all_used_terms found {len(used_terms)} terms from {len(dataset)} samples")
    return list(used_terms)


def encode_audio_tensors_in_batches(model, audio_tensors, batch_size=128, device="cuda"):
    """Encode audio tensors in batches using the model's audio encoder"""
    all_embeddings = []
    
    for i in range(0, len(audio_tensors), batch_size):
        batch_tensors = audio_tensors[i:i + batch_size]
        try:
            # 将张量移动到正确的设备并确保数据类型一致
            processed_tensors = []
            for tensor in batch_tensors:
                if isinstance(tensor, torch.Tensor):
                    # 确保张量是 float32 类型并在正确的设备上
                    tensor = tensor.float().to(device)
                processed_tensors.append(tensor)
            
            # 使用与原始 encode_audio 相同的方式处理，不额外添加AMP（model.encode_audio内部已处理）
            if model.training:
                embeddings = model.encode_audio(processed_tensors)
            else:
                with torch.no_grad():
                    embeddings = model.encode_audio(processed_tensors)
            
            # 确保数据类型为float32，但保持梯度（训练时）或断链（评估时）
            embeddings = embeddings.float()
            if not model.training:
                embeddings = embeddings.detach()
            all_embeddings.append(embeddings)
        except Exception as e:
            print(f"[ERROR] Failed to encode audio tensor batch {i//batch_size}: {e}")
            print(f"[DEBUG] Batch tensor types: {[type(t) for t in batch_tensors]}")
            print(f"[DEBUG] Batch tensor shapes: {[t.shape if isinstance(t, torch.Tensor) else 'Not tensor' for t in batch_tensors]}")
            print(f"[DEBUG] Batch tensor dtypes: {[t.dtype if isinstance(t, torch.Tensor) else 'Not tensor' for t in batch_tensors]}")
            # Create dummy embeddings
            dummy_emb = torch.zeros(len(batch_tensors), 512, dtype=torch.float32, device=device)
            all_embeddings.append(dummy_emb)
    
    return torch.cat(all_embeddings, dim=0)


def evaluate_topk_recall(model, retriever, dataset, device, top_ks=(5, 10), max_eval=1000, train_terms_set=None):
    """简化的评估函数 - 适配mmap数据格式，分别计算seen和unseen术语的recall"""
    if dist.get_rank() != 0:
        return {}
    
    model.eval()
    recall_dict = {k: [] for k in top_ks}
    seen_recall_dict = {k: [] for k in top_ks}  # seen术语的recall
    unseen_recall_dict = {k: [] for k in top_ks}  # unseen术语的recall
    
    # 重建索引
    text_terms = [term['term'] for term in retriever.term_list]
    raw_model = model.module if isinstance(model, DDP) else model
    text_emb = encode_texts_in_batches(raw_model, text_terms, device=device)
    
    # 检查是否有有效的embeddings
    if text_emb.size(0) == 0:
        print("[WARN] No valid text embeddings, skipping evaluation")
        return {k: [0.0] for k in top_ks}
    
    retriever.index.reset()
    if isinstance(text_emb, torch.Tensor):
        text_emb_numpy = text_emb.detach().cpu().float().numpy()
    else:
        text_emb_numpy = text_emb.astype(np.float32)
    retriever.index.add(text_emb_numpy)
    
    # 随机采样评估样本
    import random
    random.seed(42)
    eval_indices = random.sample(range(len(dataset)), min(max_eval, len(dataset)))
    
    # 收集有效样本 - 适配mmap格式: (ground_truth_terms, audio_tensor, chunk_text)
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
    
    # 编码音频张量 - 使用适中的batch size平衡性能和内存
    audio_embs = encode_audio_tensors_in_batches(raw_model, valid_audio_tensors, batch_size=128, device=device)
    if isinstance(audio_embs, torch.Tensor):
        audio_embs = audio_embs.detach().cpu().float().numpy()
    
    # 统计seen和unseen样本数量
    seen_samples = 0
    unseen_samples = 0
    mixed_samples = 0
    
    # 评估
    for j, sample in enumerate(valid_samples):
        ground_truth_terms, _, _ = sample
        gt_terms = [t.lower() for t in ground_truth_terms]
        audio_emb = audio_embs[j:j+1]
        
        # 如果提供了训练术语集合，分别计算seen和unseen术语的recall
        if train_terms_set is not None:
            seen_gt_terms = [t for t in gt_terms if t in train_terms_set]
            unseen_gt_terms = [t for t in gt_terms if t not in train_terms_set]
            
            # 分类样本
            if len(seen_gt_terms) > 0 and len(unseen_gt_terms) > 0:
                mixed_samples += 1
            elif len(seen_gt_terms) > 0:
                seen_samples += 1
            elif len(unseen_gt_terms) > 0:
                unseen_samples += 1
        
        for top_k in top_ks:
            D, I = retriever.index.search(audio_emb, top_k)
            retrieved_terms = [retriever.term_list[idx]['term'].lower() for idx in I[0]]
            
            # 整体recall
            matched = sum(gt_term in retrieved_terms for gt_term in gt_terms)
            sample_recall = matched / len(gt_terms) if gt_terms else 0.0
            recall_dict[top_k].append(sample_recall)
            
            # 分别计算seen和unseen术语的recall
            if train_terms_set is not None:
                seen_gt_terms = [t for t in gt_terms if t in train_terms_set]
                unseen_gt_terms = [t for t in gt_terms if t not in train_terms_set]
                
                # Seen术语recall
                if len(seen_gt_terms) > 0:
                    seen_matched = sum(gt_term in retrieved_terms for gt_term in seen_gt_terms)
                    seen_sample_recall = seen_matched / len(seen_gt_terms)
                    seen_recall_dict[top_k].append(seen_sample_recall)
                
                # Unseen术语recall
                if len(unseen_gt_terms) > 0:
                    unseen_matched = sum(gt_term in retrieved_terms for gt_term in unseen_gt_terms)
                    unseen_sample_recall = unseen_matched / len(unseen_gt_terms)
                    unseen_recall_dict[top_k].append(unseen_sample_recall)
    
    # 打印结果
    for top_k in top_ks:
        if recall_dict[top_k]:
            avg_recall = sum(recall_dict[top_k]) / len(recall_dict[top_k])
            print(f"[EVAL] Overall Recall@{top_k}: {avg_recall:.2%} ({len(recall_dict[top_k])} samples)")
            
            # 打印seen和unseen术语的recall
            if train_terms_set is not None:
                if seen_recall_dict[top_k]:
                    seen_avg_recall = sum(seen_recall_dict[top_k]) / len(seen_recall_dict[top_k])
                    print(f"[EVAL] Seen Recall@{top_k}: {seen_avg_recall:.2%} ({len(seen_recall_dict[top_k])} samples)")
                
                if unseen_recall_dict[top_k]:
                    unseen_avg_recall = sum(unseen_recall_dict[top_k]) / len(unseen_recall_dict[top_k])
                    print(f"[EVAL] Unseen Recall@{top_k}: {unseen_avg_recall:.2%} ({len(unseen_recall_dict[top_k])} samples)")
    
    # 打印样本分布统计
    if train_terms_set is not None:
        total_samples = seen_samples + unseen_samples + mixed_samples
        print(f"[EVAL] Sample distribution: Seen-only: {seen_samples}, Unseen-only: {unseen_samples}, Mixed: {mixed_samples}, Total: {total_samples}")
    
    model.train()
    
    # 返回包含详细结果的字典
    result = {
        'overall': recall_dict,
        'seen': seen_recall_dict,
        'unseen': unseen_recall_dict
    }
    
    return result


def setup_ddp(rank, world_size):
    """设置DDP环境 - Modal版本"""
    # 从环境变量获取配置
    master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
    master_port = os.environ.get('MASTER_PORT', '29500')
    
    print(f"[INFO] Rank {rank}: Setting up DDP with {master_addr}:{master_port}")
    
    # 设置NCCL环境 - 启用P2P/NVLink以提升多卡通信性能
    os.environ['NCCL_DEBUG'] = 'WARN'
    os.environ['NCCL_P2P_DISABLE'] = '0'  # 启用P2P通信
    os.environ['NCCL_IB_DISABLE'] = '1'   # 没有InfiniBand保持禁用
    os.environ['NCCL_SHM_DISABLE'] = '0'  # 启用共享内存
    # 移除SOCKET_IFNAME限制，让NCCL自动选择最佳接口
    os.environ.pop('NCCL_SOCKET_IFNAME', None)
    
    # 初始化进程组
    import datetime
    timeout = datetime.timedelta(minutes=10)
    
    try:
        dist.init_process_group(
            backend="nccl", 
            rank=rank, 
            world_size=world_size, 
            timeout=timeout
        )
        print(f"[INFO] Rank {rank}: DDP initialized successfully")
    except Exception as e:
        print(f"[ERROR] Rank {rank}: DDP initialization failed: {e}")
        raise
    
    torch.cuda.set_device(rank)
    
    if rank == 0:
        print(f"[INFO] Using device: cuda:{rank}")
        print(f"[INFO] Device name: {torch.cuda.get_device_name(rank)}")


def cleanup_ddp():
    """清理DDP环境"""
    dist.destroy_process_group()


def quick_performance_test(model, sample_batch, device, rank):
    """快速性能测试函数"""
    if rank != 0:
        return
    
    import time
    print("[INFO] Running quick performance test...")
    
    # 测试数据加载时间
    t0 = time.time()
    # 模拟一个小 batch
    test_batch = sample_batch[:min(8, len(sample_batch))]
    t1 = time.time()
    
    # 测试音频编码时间
    with torch.cuda.amp.autocast():
        try:
            raw_model = model.module if hasattr(model, 'module') else model
            audio_items = [x[1] for x in test_batch]  # 提取音频数据
            audio_emb = raw_model.encode_audio(audio_items)
            torch.cuda.synchronize()  # 确保 GPU 计算完成
            t2 = time.time()
            
            print(f"[PERF] Data prep: {(t1-t0)*1000:.1f}ms")
            print(f"[PERF] Audio encoding ({len(test_batch)} samples): {(t2-t1)*1000:.1f}ms")
            print(f"[PERF] Audio embedding shape: {audio_emb.shape}")
            print(f"[PERF] GPU memory used: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
            
        except Exception as e:
            print(f"[PERF] Performance test failed: {e}")


def check_lora_training_status(model, step, rank):
    """检查LoRA训练状态的辅助函数"""
    if rank != 0:
        return
    
    try:
        raw_model = model.module if hasattr(model, 'module') else model
        if hasattr(raw_model, 'check_lora_gradients'):
            raw_model.check_lora_gradients(step=step)
        else:
            print(f"[WARN] Model does not have check_lora_gradients method")
    except Exception as e:
        print(f"[ERROR] Failed to check LoRA status: {e}")


def eval_only(rank, world_size, args):
    """仅评估模式：加载训练好的模型，使用全部glossary作为索引进行评估"""
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    
    if rank == 0:
        print(f"[INFO] Starting evaluation-only mode with {world_size} GPUs")
        print(f"[INFO] Will evaluate on maximum {args.eval_max_samples} samples")
    
    # 初始化模型
    speech_encoder = Qwen2AudioSpeechEncoder(model_name=args.model_name, device=device)
    text_encoder = Qwen2AudioTextEncoder(
        model_name=args.model_name, 
        device=device, 
        shared_model=speech_encoder.get_shared_model()
    )
    
    model = ContrastiveQwen2AudioModel(
        speech_encoder, text_encoder, 
        proj_dim=512,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    ).to(device)
    
    # 加载预训练权重（在DDP包装之前）
    eval_model_path = getattr(args, 'eval_model_path', None)
    if eval_model_path and os.path.exists(eval_model_path):
        if rank == 0:
            print(f"[INFO] Loading trained model weights from: {eval_model_path}")
        try:
            checkpoint = torch.load(eval_model_path, map_location=device)
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
            
            model.load_state_dict(state_dict, strict=False)
            if rank == 0:
                print("[INFO] ✅ Model weights loaded successfully")
        except Exception as e:
            if rank == 0:
                print(f"[ERROR] ❌ Failed to load model weights: {e}")
                print(f"[WARN] Will evaluate with randomly initialized model instead")
    else:
        if rank == 0:
            print(f"[WARN] No trained model path provided or file not found: {eval_model_path}")
            print(f"[WARN] Will evaluate with randomly initialized model")
    
    # 包装为DDP（即使只是评估也需要，保持一致性）
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    # 加载数据集
    mmap_shard_dir = getattr(args, 'mmap_shard_dir', None)
    
    if mmap_shard_dir and os.path.exists(mmap_shard_dir):
        # 使用 mmap 数据集
        if rank == 0:
            print(f"[INFO] Using mmap dataset from: {mmap_shard_dir}")
        
        if args.test_samples_path:
            test_dataset = TermLevelDatasetMMap(None, mmap_shard_dir, split="test", test_path=args.test_samples_path)
        else:
            # 使用训练数据的测试部分
            test_dataset = TermLevelDatasetMMap(args.train_samples_path, mmap_shard_dir, split="test", train_ratio=args.train_ratio)
    else:
        # 使用传统数据集
        if rank == 0:
            print("[INFO] Using traditional dataset (file-based audio loading)")
        
        if args.test_samples_path:
            test_dataset = TermLevelDataset(None, split="test", test_path=args.test_samples_path)
        else:
            # 使用训练数据的测试部分
            test_dataset = TermLevelDataset(args.train_samples_path, split="test", train_ratio=args.train_ratio)
    
    # 设置评估用的retriever（仅主进程）
    if rank == 0:
        # 从glossary_cleaned.json加载全部术语作为索引
        if args.glossary_path and os.path.exists(args.glossary_path):
            print(f"[INFO] Loading full glossary from: {args.glossary_path}")
            from Qwen2_Audio_train import load_glossary_terms
            all_glossary_terms = load_glossary_terms(args.glossary_path)
            all_glossary_terms = list(set(t.lower() for t in all_glossary_terms if t and len(t.strip()) >= 3))
            print(f"[INFO] Loaded {len(all_glossary_terms)} unique terms from glossary")
        else:
            print(f"[WARN] Glossary path not provided or not found: {args.glossary_path}")
            print(f"[INFO] Falling back to extracting terms from test dataset")
            all_glossary_terms = extract_all_used_terms(test_dataset)
            all_glossary_terms = list(set(t.lower() for t in all_glossary_terms))
            print(f"[INFO] Extracted {len(all_glossary_terms)} unique terms from test dataset")
        
        if len(all_glossary_terms) == 0:
            print("[ERROR] No terms found! This should not happen.")
            cleanup_ddp()
            return
        
        # 从测试数据集提取ground truth术语（用于分析覆盖率）
        test_gt_terms = extract_all_used_terms(test_dataset)
        test_gt_terms = set(t.lower() for t in test_gt_terms)
        
        # 统计测试集术语在glossary中的覆盖情况
        glossary_set = set(all_glossary_terms)
        covered_terms = test_gt_terms & glossary_set
        uncovered_terms = test_gt_terms - glossary_set
        
        print(f"[INFO] Test dataset contains {len(test_gt_terms)} unique ground truth terms")
        print(f"[INFO] Glossary covers {len(covered_terms)}/{len(test_gt_terms)} ({len(covered_terms)/len(test_gt_terms)*100:.1f}%) of test terms")
        if uncovered_terms:
            print(f"[WARN] {len(uncovered_terms)} test terms not in glossary (will be impossible to retrieve)")
            if len(uncovered_terms) <= 10:
                print(f"[WARN] Uncovered terms: {list(uncovered_terms)}")
        
        retriever = SimpleRetriever(enable_fusion=True, device=device)
        retriever.model = model.module
        retriever.index = faiss.IndexFlatL2(512)
        retriever.term_list = [{'term': t} for t in all_glossary_terms]
        
        print(f"[INFO] Setup complete. Test: {len(test_dataset)}, Index Terms: {len(all_glossary_terms)}")
        
        # 进行评估
        print(f"[INFO] Starting evaluation with full glossary index...")
        model.eval()
        
        recall_results = evaluate_topk_recall(
            model, retriever, test_dataset, device, 
            top_ks=(1, 5, 10), max_eval=args.eval_max_samples,
            train_terms_set=None  # eval_only模式下不区分seen/unseen
        )
        
        print(f"\n[RESULTS] ========== EVALUATION RESULTS ==========")
        print(f"[RESULTS] Model: {eval_model_path if eval_model_path else 'Random Initialization'}")
        print(f"[RESULTS] Test Dataset: {len(test_dataset)} total samples")
        print(f"[RESULTS] Index Size: {len(all_glossary_terms)} terms (full glossary)")
        print(f"[RESULTS] Test GT Terms: {len(test_gt_terms)} unique terms")
        print(f"[RESULTS] Coverage: {len(covered_terms)}/{len(test_gt_terms)} ({len(covered_terms)/len(test_gt_terms)*100:.1f}%)")
        print(f"[RESULTS] Evaluated on: {min(args.eval_max_samples, len(test_dataset))} samples")
        print(f"[RESULTS] " + "="*48)
        
        # 处理新的返回格式
        overall_results = recall_results.get('overall', recall_results)
        
        for top_k in [1, 5, 10, 20, 50, 100]:
            if overall_results.get(top_k) and len(overall_results[top_k]) > 0:
                avg_recall = sum(overall_results[top_k]) / len(overall_results[top_k])
                print(f"[RESULTS] Recall@{top_k:3d}: {avg_recall:.2%} ({len(overall_results[top_k])} samples)")
            else:
                print(f"[RESULTS] Recall@{top_k:3d}: No valid results")
        
        print(f"[RESULTS] " + "="*48)
        print(f"\n[INFO] ✅ Evaluation completed!")
    
    cleanup_ddp()

def train_ddp(rank, world_size, args):
    """DDP训练主函数"""
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    
    if rank == 0:
        print(f"[INFO] Starting DDP training with {world_size} GPUs")
    
    # 初始化模型
    speech_encoder = Qwen2AudioSpeechEncoder(model_name=args.model_name, device=device)
    text_encoder = Qwen2AudioTextEncoder(
        model_name=args.model_name, 
        device=device, 
        shared_model=speech_encoder.get_shared_model()
    )
    
    model = ContrastiveQwen2AudioModel(
        speech_encoder, text_encoder, 
        proj_dim=512,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    ).to(device)
    
    # 强制启用LoRA参数梯度（如果它们被意外禁用）
    if rank == 0:
        print(f"\n🔧 DETAILED MODEL ANALYSIS BEFORE DDP")
        print("-" * 60)
        
        # 检查模型结构
        print(f"[DEBUG] Model type: {type(model)}")
        print(f"[DEBUG] Model training mode: {model.training}")
        
        # 检查各个组件
        print(f"[DEBUG] Speech encoder model type: {type(model.speech_encoder.model)}")
        print(f"[DEBUG] Text encoder model type: {type(model.text_encoder.model)}")
        print(f"[DEBUG] Models are shared: {model.speech_encoder.model is model.text_encoder.model}")
        
        # 强制启用LoRA梯度
        model.force_enable_lora_gradients()
        
        # 检查LoRA状态
        model.check_lora_gradients(step=0)
    
    # 加载预训练权重
    if args.best_model_path and os.path.exists(args.best_model_path):
        if rank == 0:
            print(f"[INFO] Loading weights from {args.best_model_path}")
        try:
            checkpoint = torch.load(args.best_model_path, map_location=device)
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
            
            model.load_state_dict(state_dict)
            if rank == 0:
                print("[INFO] Weights loaded successfully")
        except Exception as e:
            if rank == 0:
                print(f"[ERROR] Failed to load weights: {e}")
    
    # 包装为DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    # DDP包装后运行完整的LoRA诊断（这很重要！）
    if rank == 0:
        print(f"\n🔧 POST-DDP LoRA DIAGNOSIS")
        print("-" * 60)
        raw_model = model.module
        print(f"[DEBUG] Post-DDP model type: {type(raw_model)}")
        print(f"[DEBUG] Post-DDP training mode: {raw_model.training}")
        
        # 运行完整的诊断流程
        raw_model.diagnose_lora_step_by_step()
    
    # 设置优化器（仅包含需要训练的参数，如LoRA和投影层）
    trainable_params = [p for p in model.module.parameters() if p.requires_grad]
    if rank == 0:
        print(f"[INFO] Optimizer will update {len(trainable_params)} parameter tensors")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=(rank == 0)
    )

    # 如果可训练参数仍为 FP16，则禁用 GradScaler 避免 unscale 报错
    contains_fp16_params = any(param.dtype == torch.float16 for param in trainable_params)
    if rank == 0 and contains_fp16_params:
        print("[WARN] Trainable parameters include FP16 tensors; disabling GradScaler to avoid unscale errors")
    
    # 加载数据集
    mmap_shard_dir = getattr(args, 'mmap_shard_dir', None)
    hard_neg_jsonl = getattr(args, 'hard_neg_jsonl', None)
    
    if mmap_shard_dir and os.path.exists(mmap_shard_dir):
        # 使用 mmap 数据集
        if rank == 0:
            print(f"[INFO] Using mmap dataset from: {mmap_shard_dir}")
        
        if args.test_samples_path:
            train_dataset = TermLevelDatasetMMap(args.train_samples_path, mmap_shard_dir, split="train", train_ratio=1.0, hard_neg_jsonl=hard_neg_jsonl)
            test_dataset = TermLevelDatasetMMap(None, mmap_shard_dir, split="test", test_path=args.test_samples_path, hard_neg_jsonl=None)
        else:
            # 如果启用测试集重构，需要更大的测试集来筛选样本
            effective_train_ratio = args.train_ratio
            if args.rebuild_test_set:
                # 动态调整train_ratio以确保有足够的测试样本
                # 目标：至少需要target_test_size * 2的测试样本来筛选
                min_test_ratio = max(0.05, args.target_test_size * 2 / 50000)  # 假设大约5万样本，至少5%测试集
                if (1.0 - args.train_ratio) < min_test_ratio:
                    effective_train_ratio = 1.0 - min_test_ratio
                    print(f"[INFO] Adjusting train_ratio from {args.train_ratio:.3f} to {effective_train_ratio:.3f} for test set rebuilding")
                    print(f"[INFO] This ensures at least {min_test_ratio:.1%} of data for test set to achieve target unseen ratio")
                else:
                    print(f"[INFO] Current train_ratio {args.train_ratio:.3f} provides sufficient test samples")
            
            train_dataset = TermLevelDatasetMMap(args.train_samples_path, mmap_shard_dir, split="train", train_ratio=effective_train_ratio, hard_neg_jsonl=hard_neg_jsonl)
            test_dataset = TermLevelDatasetMMap(args.train_samples_path, mmap_shard_dir, split="test", train_ratio=effective_train_ratio, hard_neg_jsonl=None)
    else:
        # 使用传统数据集
        if rank == 0:
            print("[INFO] Using traditional dataset (file-based audio loading)")
        
        if args.test_samples_path:
            train_dataset = TermLevelDataset(args.train_samples_path, split="train", train_ratio=1.0, hard_neg_jsonl=hard_neg_jsonl)
            test_dataset = TermLevelDataset(None, split="test", test_path=args.test_samples_path, hard_neg_jsonl=None)
        else:
            # 如果启用测试集重构，需要更大的测试集来筛选样本
            effective_train_ratio = args.train_ratio
            if args.rebuild_test_set:
                # 动态调整train_ratio以确保有足够的测试样本
                min_test_ratio = max(0.05, args.target_test_size * 2 / 50000)  # 假设大约5万样本，至少5%测试集
                if (1.0 - args.train_ratio) < min_test_ratio:
                    effective_train_ratio = 1.0 - min_test_ratio
                    print(f"[INFO] Adjusting train_ratio from {args.train_ratio:.3f} to {effective_train_ratio:.3f} for test set rebuilding")
                    print(f"[INFO] This ensures at least {min_test_ratio:.1%} of data for test set to achieve target unseen ratio")
                else:
                    print(f"[INFO] Current train_ratio {args.train_ratio:.3f} provides sufficient test samples")
            
            train_dataset = TermLevelDataset(args.train_samples_path, split="train", train_ratio=effective_train_ratio, hard_neg_jsonl=hard_neg_jsonl)
            test_dataset = TermLevelDataset(args.train_samples_path, split="test", train_ratio=effective_train_ratio, hard_neg_jsonl=None)
    
    # 数据加载器
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size // world_size,
        sampler=train_sampler,
        collate_fn=collate_keep,
        num_workers=16,  # 提高到 16 以充分利用 64 vCPU
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,  # 增加预取因子
        worker_init_fn=worker_init_fn,
        drop_last=True  # 避免最后一个batch大小不一致导致的DDP问题
    )
    
    # 重新构建测试集以确保unseen术语比例（仅主进程）
    if rank == 0:
        # 分别提取训练集和测试集的术语
        used_terms_train = extract_all_used_terms(train_dataset)
        used_terms_test = extract_all_used_terms(test_dataset)
        
        # 转换为小写并去重
        train_terms_set = set(t.lower() for t in used_terms_train)
        test_terms_set = set(t.lower() for t in used_terms_test)
        
        print(f"[DEBUG] Original - Train terms: {len(train_terms_set)}, Test terms: {len(test_terms_set)}")
        
        # 计算unseen terms（在测试集中但不在训练集中的术语）
        unseen_terms = test_terms_set - train_terms_set
        seen_terms = test_terms_set & train_terms_set
        
        initial_unseen_ratio = len(unseen_terms) / len(test_terms_set) if len(test_terms_set) > 0 else 0.0
        print(f"[INFO] Original test set unseen terms ratio: {initial_unseen_ratio:.2%} ({len(unseen_terms)}/{len(test_terms_set)})")
        
        # 重新构建测试集以确保unseen术语比例达到目标值
        if args.rebuild_test_set:
            print(f"[INFO] Rebuilding test set to ensure {args.target_unseen_ratio:.0%} unseen terms ratio...")
            
            # 按术语类型分类测试样本
            seen_samples = []    # 包含seen术语的样本
            unseen_samples = []  # 包含unseen术语的样本
            mixed_samples = []   # 同时包含seen和unseen术语的样本
            
            for i, sample in enumerate(test_dataset.samples):
                ground_truth_terms = sample.get('term_chunk_audio_ground_truth_terms', [])
                sample_terms = set(t.lower() for t in ground_truth_terms if isinstance(t, str))
                
                sample_seen_terms = sample_terms & train_terms_set
                sample_unseen_terms = sample_terms - train_terms_set
                
                if sample_unseen_terms and sample_seen_terms:
                    mixed_samples.append((i, sample, sample_seen_terms, sample_unseen_terms))
                elif sample_unseen_terms:
                    unseen_samples.append((i, sample, set(), sample_unseen_terms))
                elif sample_seen_terms:
                    seen_samples.append((i, sample, sample_seen_terms, set()))
            
            print(f"[DEBUG] Sample classification: seen={len(seen_samples)}, unseen={len(unseen_samples)}, mixed={len(mixed_samples)}")
            
            # 检查是否有足够的unseen样本
            total_unseen_contributing = len(unseen_samples) + len(mixed_samples)
            if total_unseen_contributing < int(args.target_test_size * args.target_unseen_ratio * 0.5):  # 至少要有目标的50%
                print(f"[WARN] Insufficient unseen samples ({total_unseen_contributing}) to achieve target ratio. Consider adjusting train_ratio.")
                print(f"[WARN] Current test set size: {len(test_dataset)}, unseen-contributing samples: {total_unseen_contributing}")
                print(f"[WARN] Proceeding with available samples...")
            
            # 优先选择包含unseen术语的样本
            selected_samples = []
            import random
            random.seed(42)  # 确保可复现
            
            # 1. 优先选择mixed样本（它们贡献unseen术语）
            random.shuffle(mixed_samples)
            for idx, sample, seen_terms, unseen_terms in mixed_samples:
                if len(selected_samples) < args.target_test_size:
                    selected_samples.append(sample)
            
            # 2. 添加pure unseen样本
            random.shuffle(unseen_samples)
            for idx, sample, seen_terms, unseen_terms in unseen_samples:
                if len(selected_samples) < args.target_test_size:
                    selected_samples.append(sample)
            
            # 3. 用seen样本填充剩余位置
            random.shuffle(seen_samples)
            for idx, sample, seen_terms, unseen_terms in seen_samples:
                if len(selected_samples) < args.target_test_size:
                    selected_samples.append(sample)
            
            # 如果样本不足目标数量，使用所有可用样本
            final_test_size = min(args.target_test_size, len(selected_samples))
            test_dataset.samples = selected_samples[:final_test_size]
            
            # 重新计算术语统计
            used_terms_test_new = extract_all_used_terms(test_dataset)
            test_terms_set_new = set(t.lower() for t in used_terms_test_new)
            unseen_terms_new = test_terms_set_new - train_terms_set
            final_unseen_ratio = len(unseen_terms_new) / len(test_terms_set_new) if len(test_terms_set_new) > 0 else 0.0
            
            print(f"[INFO] Rebuilt test set: {len(test_dataset)} samples (target: {args.target_test_size})")
            print(f"[INFO] New test terms: {len(test_terms_set_new)} (unseen: {len(unseen_terms_new)}, ratio: {final_unseen_ratio:.2%})")
            
            if final_unseen_ratio < args.target_unseen_ratio * 0.8:  # 如果达不到目标的80%
                print(f"[WARN] Final unseen ratio ({final_unseen_ratio:.2%}) is significantly lower than target ({args.target_unseen_ratio:.2%})")
                print(f"[WARN] Consider using a smaller train_ratio to get more test samples")
            
            # 更新变量
            used_terms_test = used_terms_test_new
            test_terms_set = test_terms_set_new
            unseen_terms = unseen_terms_new
        else:
            print(f"[INFO] Test set rebuilding disabled, using original test set ({len(test_dataset)} samples)")
    
    # 准备全局随机术语库（用于 hard negative 训练中的随机负例采样）
    global_random_terms = list(train_terms_set) if rank == 0 else []
    
    # 设置评估用的retriever（仅主进程）
    retriever = None
    if rank == 0:
        
        # 使用全量术语建立检索索引（训练集+测试集的所有术语）
        # 这样才能正确召回所有可能的术语，包括unseen terms
        all_used_terms = list(train_terms_set | test_terms_set)
        
        print(f"[DEBUG] Using {len(all_used_terms)} terms for retriever (train+test all terms)")
        if len(all_used_terms) == 0:
            print("[ERROR] No terms found in datasets! This should not happen.")
            print(f"[DEBUG] Train dataset size: {len(train_dataset)}")
            print(f"[DEBUG] Test dataset size: {len(test_dataset)}")
            # 检查前几个样本
            if len(train_dataset) > 0:
                sample = train_dataset[0]
                print(f"[DEBUG] Sample structure: {type(sample)}")
                print(f"[DEBUG] Sample content: {sample}")
        
        retriever = SimpleRetriever(enable_fusion=True, device=device)
        retriever.model = model.module
        retriever.index = faiss.IndexFlatL2(512)
        retriever.term_list = [{'term': t} for t in all_used_terms]
        
        print(f"[INFO] Setup complete. Training: {len(train_dataset)}, Test: {len(test_dataset)}")
        print(f"[INFO] Retriever terms: {len(all_used_terms)} (full vocabulary for proper evaluation)")
        print(f"[INFO] Unseen terms in retriever: {len(unseen_terms)} ({len(unseen_terms)/len(all_used_terms)*100:.1f}% of retriever vocabulary)")
        print(f"[INFO] Batch configuration: {args.batch_size} physical × {args.gradient_accumulation_steps} accumulation = {args.batch_size * args.gradient_accumulation_steps} effective")
        print(f"[INFO] Per-GPU batch size: {args.batch_size // world_size}")
    
    # 训练循环
    best_recall = 0.0
    no_improve_epochs = 0
    scaler = torch.cuda.amp.GradScaler(enabled=not contains_fp16_params)
    if rank == 0:
        scaler_status = "enabled" if scaler.is_enabled() else "disabled"
        print(f"[INFO] GradScaler is {scaler_status} (trainable_params dtype check)")
        if args.check_lora_every > 0:
            print(f"[INFO] LoRA gradient inspection scheduled every {args.check_lora_every} optimizer steps")
        else:
            print("[INFO] LoRA gradient inspection disabled (--check_lora_every 0)")
    global_step = 0
    
    for epoch in range(args.epochs):
        if rank == 0:
            print(f"\n[INFO] Epoch {epoch+1}/{args.epochs}")
        
        train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        
        # 训练
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}") if rank == 0 else train_dataloader
        
        valid_batches = 0
        empty_batches = 0
        failed_batches = 0
        accumulated_loss = 0.0
        accumulation_steps = 0
        
        for batch_idx, batch in enumerate(pbar):
            # 在第一个 epoch 的第一个 batch 进行性能测试
            if epoch == 0 and batch_idx == 0 and batch and len(batch) > 0:
                quick_performance_test(model, batch, device, rank)
            
            # 标记本步是否进行了 backward
            did_backward = torch.tensor([0], device=device, dtype=torch.int)

            # 1) 判空：空 batch 直接跳过（不调用 scaler.update）
            if not batch or len(batch) == 0:
                empty_batches += 1
                if rank == 0 and empty_batches <= 5:  # 只打印前5个空batch
                    print(f"[DEBUG] Empty batch at index {batch_idx}, total empty so far: {empty_batches}")
                # 同步一下，保持各 rank 在相近步长，但不做 update/step
                dist.all_reduce(did_backward, op=dist.ReduceOp.MIN)
                optimizer.zero_grad(set_to_none=True)
                continue

            # 检查batch内容
            if rank == 0 and batch_idx < 3:  # 打印前3个batch的信息
                print(f"[DEBUG] Batch {batch_idx} size: {len(batch)}")
                if len(batch) > 0:
                    sample = batch[0]
                    print(f"[DEBUG] Sample structure: {type(sample)}, length: {len(sample) if hasattr(sample, '__len__') else 'N/A'}")

            # 2) 计算 loss（AMP）
            loss = None
            with torch.cuda.amp.autocast():
                try:
                    loss = train_step(model, batch, device, args, global_random_terms=global_random_terms)
                    valid_batches += 1
                except Exception as e:
                    failed_batches += 1
                    if rank == 0 and failed_batches <= 5:
                        print(f"[WARN] Batch {batch_idx} processing failed: {e}")
                    # 不调用 scaler.update；只做同步门控
                    dist.all_reduce(did_backward, op=dist.ReduceOp.MIN)
                    optimizer.zero_grad(set_to_none=True)
                    continue

            # 3) 只有有效 loss 才 backward（使用梯度累积）
            if (loss is not None) and loss.requires_grad and torch.isfinite(loss):
                # Scale loss by accumulation steps for proper averaging
                scaled_loss = loss / args.gradient_accumulation_steps
                scaler.scale(scaled_loss).backward()
                did_backward.fill_(1)
                accumulated_loss += float(loss.detach().item())
                accumulation_steps += 1

            # 4) 跨 rank 对齐：只有当所有 rank 都 did_backward==1 才可能 step()
            dist.all_reduce(did_backward, op=dist.ReduceOp.MIN)
            
            # 5) 检查是否达到累积步数或最后一个batch
            should_step = (accumulation_steps >= args.gradient_accumulation_steps) or (batch_idx == len(train_dataloader) - 1)
            
            if int(did_backward.item()) == 1 and should_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

                global_step += 1
                if rank == 0 and args.check_lora_every > 0 and global_step % args.check_lora_every == 0:
                    check_lora_training_status(model, global_step, rank)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                total_loss += accumulated_loss
                accumulated_loss = 0.0
                accumulation_steps = 0
            elif int(did_backward.item()) == 0:
                # 有 rank 未 backward：清零但不 step
                optimizer.zero_grad(set_to_none=True)
                accumulated_loss = 0.0
                accumulation_steps = 0
        
        # 同步损失
        avg_loss = total_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0.0
        loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / world_size
        
        if rank == 0:
            print(f"[INFO] Epoch {epoch+1} avg loss: {avg_loss:.4f}")
            print(f"[INFO] Batch statistics: valid={valid_batches}, empty={empty_batches}, failed={failed_batches}")
        
        # 评估（仅主进程）
        if rank == 0:
            recall_results = evaluate_topk_recall(
                model, retriever, test_dataset, device, 
                top_ks=(5, 10), max_eval=min(1000, len(test_dataset)),
                train_terms_set=train_terms_set  # 传递训练术语集合用于seen/unseen区分
            )
            
            # 处理新的返回格式
            overall_results = recall_results.get('overall', recall_results)
            current_recall = sum(overall_results[10]) / len(overall_results[10]) if overall_results.get(10) else 0.0
            scheduler.step(current_recall)
            
            if current_recall > best_recall:
                best_recall = current_recall
                no_improve_epochs = 0
                best_model_path = args.save_path.replace('.pt', '_best.pt')
                torch.save(model.state_dict(), best_model_path)
                print(f"[INFO] New best model saved (Overall Recall@10: {best_recall:.2%})")
            else:
                no_improve_epochs += 1
                print(f"[INFO] No improvement for {no_improve_epochs} epochs")
                
                if no_improve_epochs >= args.patience:
                    print(f"[INFO] Early stopping. Best Overall Recall@10: {best_recall:.2%}")
                    break
        
        dist.barrier()
    
    # 保存最终模型
    if rank == 0:
        torch.save(model.state_dict(), args.save_path)
        print(f"[INFO] Training completed. Best Recall@10: {best_recall:.2%}")
    
    cleanup_ddp()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=512)  # Total batch size across 8 GPUs (64 per GPU)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--train_samples_path', type=str, required=True)
    parser.add_argument('--test_samples_path', type=str, default=None)
    parser.add_argument('--train_ratio', type=float, default=0.998)
    parser.add_argument('--glossary_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default="qwen2_audio_term_level.pt")
    parser.add_argument('--best_model_path', type=str, default=None)
    parser.add_argument('--audio_text_loss_ratio', type=float, default=0.3)
    parser.add_argument('--audio_term_loss_ratio', type=float, default=0.7)
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--mmap_shard_dir', type=str, default=None, help='Directory containing mmap audio shards')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help='Number of steps to accumulate gradients (effective batch = batch_size * accumulation_steps)')
    parser.add_argument('--max_batch_per_gpu', type=int, default=None, help='Maximum batch size per GPU (will auto-adjust total batch_size if needed)')
    parser.add_argument('--check_lora_every', type=int, default=0, help='Inspect LoRA gradients every N optimizer steps (0 to disable)')
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode (minimal setup for debugging)')
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluation on the original model without training')
    parser.add_argument('--eval_max_samples', type=int, default=1000, help='Maximum number of samples to evaluate (for eval_only mode)')
    parser.add_argument('--eval_model_path', type=str, default=None, help='Path to trained model for evaluation (for eval_only mode)')
    parser.add_argument('--rebuild_test_set', action='store_true', help='Rebuild test set to ensure 20% unseen terms ratio')
    parser.add_argument('--target_test_size', type=int, default=1000, help='Target size for rebuilt test set')
    parser.add_argument('--target_unseen_ratio', type=float, default=0.20, help='Target ratio of unseen terms in test set')
    
    # Hard Negative Mining 参数
    parser.add_argument('--hard_neg_jsonl', type=str, default=None, help='Path to hard negatives JSONL file')
    parser.add_argument('--max_hn_per_sample', type=int, default=15, help='Maximum hard negatives per sample')
    parser.add_argument('--rand_neg_per_sample', type=int, default=5, help='Random negatives per sample (for diversity)')
    parser.add_argument('--hard_neg_loss_ratio', type=float, default=0.5, help='Weight for hard negative loss')
    
    args = parser.parse_args()
    
    # 获取world size
    world_size = int(os.environ.get('WORLD_SIZE', torch.cuda.device_count()))
    
    if world_size == 0:
        print("[ERROR] No CUDA devices available!")
        return 1
    
    # 检查是否为评估模式
    if args.eval_only:
        print(f"[INFO] Starting evaluation mode with {world_size} GPUs")
        print(f"[INFO] Model path: {args.eval_model_path if args.eval_model_path else 'Random Initialization'}")
        print(f"[INFO] Glossary path: {args.glossary_path}")
        print(f"[INFO] Max evaluation samples: {args.eval_max_samples}")
        
        # 使用torchrun启动时，LOCAL_RANK会自动设置
        if 'LOCAL_RANK' in os.environ:
            # torchrun模式
            rank = int(os.environ['LOCAL_RANK'])
            eval_only(rank, world_size, args)
        else:
            # 手动多进程模式
            mp.set_start_method('spawn', force=True)
            mp.spawn(eval_only, args=(world_size, args), nprocs=world_size, join=True)
        
        print("[INFO] Evaluation completed")
        return 0
    
    # 训练模式
    # 自动调整batch size以防止OOM
    if args.max_batch_per_gpu:
        max_total_batch = args.max_batch_per_gpu * world_size
        if args.batch_size > max_total_batch:
            print(f"[WARN] Batch size {args.batch_size} exceeds max ({max_total_batch}), adjusting...")
            args.batch_size = max_total_batch
    
    print(f"[INFO] Starting DDP training with {world_size} GPUs")
    print(f"[INFO] Physical batch size: {args.batch_size} (per GPU: {args.batch_size // world_size})")
    print(f"[INFO] Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"[INFO] Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    
    # 使用torchrun启动时，LOCAL_RANK会自动设置
    if 'LOCAL_RANK' in os.environ:
        # torchrun模式
        rank = int(os.environ['LOCAL_RANK'])
        train_ddp(rank, world_size, args)
    else:
        # 手动多进程模式
        mp.set_start_method('spawn', force=True)
        mp.spawn(train_ddp, args=(world_size, args), nprocs=world_size, join=True)
    
    print("[INFO] Training completed")
    return 0


if __name__ == "__main__":
    main()
