import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
from tqdm import tqdm
import argparse
import os
import sys
from torch.optim.lr_scheduler import ReduceLROnPlateau
import faiss
import mmap
from new_retrieve import Retriever
import soundfile as sf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 导入新的Qwen2-Audio模型结构
from Qwen2_Audio_train import (
    Qwen2AudioSpeechEncoder, 
    Qwen2AudioTextEncoder, 
    ContrastiveQwen2AudioModel,
    load_glossary_terms, 
    encode_texts_in_batches, 
    encode_audios_in_batches
)


# === Hard Negative Mining Context Helper ===
class HardNegContext:
    """
    Dual-mode hard negative context:
      1) In-memory tensor mode (small bank): provide emb_tensor [N, D] (L2-normalized) and term2idx dict.
      2) FAISS index mode (large bank): provide faiss_index (IVF/HNSW/Flat etc.) and term2idx dict.
    """
    def __init__(self, terms=None, term2idx=None, emb_tensor=None, faiss_index=None, metric='ip'):
        self.terms = terms or []
        self.term2idx = term2idx or {}
        self.emb_tensor = emb_tensor  # torch.FloatTensor [N, D] on device (normalized)
        self.faiss_index = faiss_index  # faiss.Index or None
        # metric: 'ip' (inner product) or 'l2'
        self.metric = metric


# === Utility to load term2idx JSON ===
def load_term2idx_json(path):
    if path is None:
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load term2idx from {path}: {e}")
        return {}


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
    def __init__(self, path="data/xl_term_level_chunks_merged.json", split="train", train_ratio=0.99, test_path=None, enable_no_term=False, filter_no_term=True):
        self.enable_no_term = enable_no_term
        self.filter_no_term = filter_no_term
        print(f"[INFO] No-term samples enabled: {enable_no_term}")
        print(f"[INFO] Filter no-term samples: {filter_no_term}")
        
        if split == "test" and test_path is not None:
            # 使用独立的测试数据集
            print(f"[INFO] Loading test samples from separate file: {test_path}")
            with open(test_path, "r") as f:
                all_samples = json.load(f)
            # 对于独立测试集，不需要train_ratio分割，直接使用所有样本
            use_split_logic = False
        else:
            # 使用原有的分割逻辑
            print(f"[INFO] Loading term-level chunk samples from {path}")
            with open(path, "r") as f:
                all_samples = json.load(f)
            use_split_logic = True
        
        # 过滤有效样本：包括有术语和无术语的样本
        valid_samples = []
        invalid_audio_count = 0
        term_samples_count = 0
        no_term_samples_count = 0
        
        for i, s in enumerate(all_samples):
            terms = s.get('term_chunk_audio_ground_truth_terms')
            if not isinstance(terms, list):
                terms = []
            
            # 过滤术语（如果有的话）
            filtered_terms = [
                t for t in terms
                if isinstance(t, str)
                and len(t) >= 3
                and sum(c.isdigit() for c in t) <= len(t) // 2
            ]

            # 过滤前后缀
            black_words = ['yeah','this ']
            black_suffixes = ['years']
            filtered_terms = [
                t for t in filtered_terms 
                if not any(t.lower().startswith(prefix.lower()) for prefix in black_words)
                and not any(t.lower().endswith(suffix.lower()) for suffix in black_suffixes)
            ]
            
            # 替换原列表为过滤后的术语（允许为空列表）
            s = dict(s)  # 避免直接修改原始数据
            s['term_chunk_audio_ground_truth_terms'] = filtered_terms
            
            # 检查基本条件
            if not (s.get('term_chunk_text', '').strip() and s.get('term_chunk_audio', '')):
                continue
            
            # 检查音频文件有效性
            audio_path = s.get("term_chunk_audio", "")
            is_valid, reason = is_audio_valid(audio_path)
            
            if is_valid:
                # 根据filter_no_term配置决定是否包含无术语样本
                if filtered_terms:
                    # 有术语的样本总是包含
                    valid_samples.append(s)
                    term_samples_count += 1
                elif not self.filter_no_term:
                    # 只有在不过滤no-term时才包含无术语样本
                    valid_samples.append(s)
                    no_term_samples_count += 1
                # 如果filter_no_term=True且无术语，则跳过该样本
            else:
                invalid_audio_count += 1
                # 只打印前10个无效音频的详细信息
                if invalid_audio_count <= 10:
                    print(f"[WARN] Skipping sample {i}: {audio_path} - {reason}")
        
        if invalid_audio_count > 10:
            print(f"[WARN] ... and {invalid_audio_count - 10} more samples with invalid audio")
            
        print(f"[INFO] Audio validation: {len(valid_samples)} valid, {invalid_audio_count} invalid")
        print(f"[INFO] Dataset composition: {term_samples_count} term samples + {no_term_samples_count} no-term samples = {len(valid_samples)} total")
        if len(valid_samples) > 0:
            print(f"[INFO] No-term ratio: {no_term_samples_count/len(valid_samples):.1%}")
        
        if self.filter_no_term and no_term_samples_count == 0:
            print(f"[INFO] No-term samples filtered out (filter_no_term=True)")
        
        print(f"[INFO] Filtered {len(valid_samples)} valid term-level samples from {len(all_samples)} total samples")
        
        if use_split_logic:
            # 数据分割：99%训练，1%测试
            import random
            random.seed(42)  # 固定随机种子确保可复现
            random.shuffle(valid_samples)
            
            split_idx = int(len(valid_samples) * train_ratio)
            
            if split == "train":
                self.samples = valid_samples[:split_idx]
                # 统计训练集中的no-term样本
                train_no_term_count = sum(1 for s in self.samples if not s.get('term_chunk_audio_ground_truth_terms'))
                print(f"[INFO] Training split: {len(self.samples)} samples ({train_no_term_count} no-term, {len(self.samples)-train_no_term_count} term)")
            elif split == "test":
                self.samples = valid_samples[split_idx:]
                # 统计测试集中的no-term样本
                test_no_term_count = sum(1 for s in self.samples if not s.get('term_chunk_audio_ground_truth_terms'))
                print(f"[INFO] Test split: {len(self.samples)} samples ({test_no_term_count} no-term, {len(self.samples)-test_no_term_count} term)")
            else:
                raise ValueError(f"Invalid split: {split}. Must be 'train' or 'test'")
        else:
            # 独立测试集，直接使用所有有效样本
            self.samples = valid_samples
            test_no_term_count = sum(1 for s in self.samples if not s.get('term_chunk_audio_ground_truth_terms'))
            print(f"[INFO] Using separate test dataset: {len(self.samples)} samples ({test_no_term_count} no-term, {len(self.samples)-test_no_term_count} term)")
        
        print(f"[INFO] Loaded {len(self.samples)} term-level samples for {split} split")

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_path = sample["term_chunk_audio"]  # 使用term chunk音频
        chunk_text = sample["term_chunk_text"]   # 使用term chunk文本
        ground_truth_terms = sample.get('term_chunk_audio_ground_truth_terms', [])
        has_target = bool(ground_truth_terms and len(ground_truth_terms) > 0)
        
        return ground_truth_terms, audio_path, chunk_text, has_target

    def __len__(self):
        return len(self.samples)


def train_step(model, batch, device, args, hn_ctx=None, temperature=0.07):
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model

    if len(batch) < 2:
        print("Batch has less than 2 non-None items, skipping...")
        return torch.tensor(0.0, requires_grad=True).to(device)

    # 拆分batch数据：ground_truth_terms, audio_path, chunk_text, has_target
    ground_truth_terms_list, audio_paths, chunk_texts, has_targets = zip(*batch)
    
    # 全小写处理
    ground_truth_terms_list = [[t.lower() for t in terms if isinstance(t, str)] for terms in ground_truth_terms_list]
    chunk_texts = [text.lower() if isinstance(text, str) else "" for text in chunk_texts]

    # === 编码音频和文本 ===
    try:
        # 先验证音频文件批次
        valid_audio_paths, valid_audio_indices = validate_audio_batch(audio_paths, verbose=True)
        
        if len(valid_audio_paths) == 0:
            print(f"[ERROR] No valid audio files in batch, skipping")
            return torch.tensor(0.0, requires_grad=True).to(device)
        
        if len(valid_audio_paths) != len(audio_paths):
            print(f"[WARN] Only {len(valid_audio_paths)}/{len(audio_paths)} audio files are valid")
            # 重新组织batch，只保留有效的样本
            valid_batch_data = []
            for idx in valid_audio_indices:
                valid_batch_data.append((
                    ground_truth_terms_list[idx],
                    audio_paths[idx], 
                    chunk_texts[idx],
                    has_targets[idx]
                ))
            
            # 如果有效样本太少，跳过这个batch
            if len(valid_batch_data) < 2:
                print(f"[ERROR] Too few valid samples ({len(valid_batch_data)}), skipping batch")
                return torch.tensor(0.0, requires_grad=True).to(device)
            
            # 重新提取数据
            ground_truth_terms_list, audio_paths, chunk_texts, has_targets = zip(*valid_batch_data)
            ground_truth_terms_list = list(ground_truth_terms_list)
            audio_paths = list(audio_paths)
            chunk_texts = list(chunk_texts)
            has_targets = list(has_targets)
        
        # 编码音频
        audio_emb = raw_model.encode_audio(audio_paths)  # [B, proj_dim]
        
        # 检查音频embedding
        if torch.isnan(audio_emb).any() or torch.isinf(audio_emb).any():
            print(f"[ERROR] NaN/Inf detected in audio embeddings after encoding!")
            return torch.tensor(0.0, requires_grad=True).to(device)
        
        if args.audio_text_loss_ratio > 0:
            text_emb = raw_model.encode_text(chunk_texts)    # [B, proj_dim]
            
            # 检查文本embedding
            if torch.isnan(text_emb).any() or torch.isinf(text_emb).any():
                print(f"[ERROR] NaN/Inf detected in text embeddings!")
                print(f"[DEBUG] Text samples: {chunk_texts[:3]}...")
                return torch.tensor(0.0, requires_grad=True).to(device)
        else:
            text_emb = torch.zeros_like(audio_emb)
        
    except Exception as e:
        print(f"[ERROR] Failed to encode audio/text: {e}")
        import traceback
        traceback.print_exc()
        return torch.tensor(0.0, requires_grad=True).to(device)

    # === 计算音频-文本对比损失 ===
    # 定义batch_size，确保在所有代码路径中都可用
    batch_size = len(audio_paths)  # 使用实际的batch size（可能已经过滤）
    
    if args.audio_text_loss_ratio > 0:
        sim_matrix = (audio_emb @ text_emb.T) / temperature  # [B, B]
        # 数值稳定性检查
        if torch.isnan(sim_matrix).any() or torch.isinf(sim_matrix).any():
            print(f"[ERROR] NaN/Inf in contrastive sim_matrix, skipping batch")
            return torch.tensor(0.0, requires_grad=True).to(device)
        
        # 创建正样本mask（对角线为1，表示音频i和文本i是正样本对）
        labels = torch.arange(batch_size).to(device)
        
        # 计算对称的对比损失
        try:
            loss_audio_to_text = F.cross_entropy(sim_matrix, labels)
            loss_text_to_audio = F.cross_entropy(sim_matrix.T, labels)
            
            if torch.isnan(loss_audio_to_text) or torch.isnan(loss_text_to_audio):
                print(f"[ERROR] NaN in contrastive cross_entropy, skipping batch")
                return torch.tensor(0.0, requires_grad=True).to(device)
            
            contrastive_loss = (loss_audio_to_text + loss_text_to_audio) / 2
        except Exception as e:
            print(f"[ERROR] Failed to compute contrastive loss: {e}")
            return torch.tensor(0.0, requires_grad=True).to(device)
    else:
        contrastive_loss = torch.tensor(0.0, device=device)

    # === 计算音频-术语对比损失 ===
    all_gt_terms = []
    audio_term_pairs = []  # (audio_idx, term_idx) 正样本对
    
    for i, terms in enumerate(ground_truth_terms_list):
        for term in terms:
            if term and len(term.strip()) > 0:
                term_idx = len(all_gt_terms)
                all_gt_terms.append(term.strip())
                audio_term_pairs.append((i, term_idx))
    
    if len(all_gt_terms) > 0 and len(audio_term_pairs) > 0:
        # 编码所有的ground truth terms
        terms_emb = raw_model.encode_text(all_gt_terms)  # [N_terms, proj_dim]
        
        # 计算音频-术语相似度矩阵
        audio_term_sim = (audio_emb @ terms_emb.T) / temperature  # [B, N_terms]
        
        # 数值稳定性检查
        if torch.isnan(audio_term_sim).any() or torch.isinf(audio_term_sim).any():
            print(f"[ERROR] NaN/Inf detected in audio_term_sim, skipping batch")
            return torch.tensor(0.0, requires_grad=True).to(device)
        
        # 构建正样本标签
        audio_term_labels = []
        for i in range(batch_size):
            # 找到audio i对应的所有positive term indices
            positive_terms = [term_idx for audio_idx, term_idx in audio_term_pairs if audio_idx == i]
            if positive_terms:
                # 如果有多个正样本，随机选择一个作为主要目标
                import random
                audio_term_labels.append(random.choice(positive_terms))
            else:
                # 如果没有正样本，跳过这个样本
                audio_term_labels.append(-1)
        
        # 计算损失，只对有正样本的音频样本计算
        valid_indices = [i for i, label in enumerate(audio_term_labels) if label >= 0]
        
        if len(valid_indices) > 0:
            valid_audio_term_sim = audio_term_sim[valid_indices]  # [valid_B, N_terms]
            valid_labels = torch.tensor([audio_term_labels[i] for i in valid_indices], device=device)
            
            # 音频到术语的损失
            audio_to_term_loss = F.cross_entropy(valid_audio_term_sim, valid_labels)
            
            # 术语到音频的损失 - 为了对称性
            term_to_audio_sim = valid_audio_term_sim.T  # [N_terms, valid_B]
            # 创建反向标签：对于每个术语，找到对应的音频索引
            term_audio_labels = []
            for term_idx in range(len(all_gt_terms)):
                # 找到term_idx对应的音频在valid_indices中的位置
                corresponding_audios = [j for j, orig_i in enumerate(valid_indices) 
                                      if (orig_i, term_idx) in audio_term_pairs]
                if corresponding_audios:
                    term_audio_labels.append(corresponding_audios[0])  # 选择第一个
                else:
                    term_audio_labels.append(-1)
            
            # 只对有对应音频的术语计算损失
            valid_term_indices = [i for i, label in enumerate(term_audio_labels) if label >= 0]
            if len(valid_term_indices) > 0:
                valid_term_audio_sim = term_to_audio_sim[valid_term_indices]
                valid_term_labels = torch.tensor([term_audio_labels[i] for i in valid_term_indices], device=device)
                term_to_audio_loss = F.cross_entropy(valid_term_audio_sim, valid_term_labels)
            else:
                term_to_audio_loss = torch.tensor(0.0, device=device)
            
            # 组合音频-术语损失
            audio_term_loss = (audio_to_term_loss + term_to_audio_loss) / 2
        else:
            audio_term_loss = torch.tensor(0.0, device=device)
    else:
        audio_term_loss = torch.tensor(0.0, device=device)

    # === No-term margin loss (拒答能力) ===
    no_term_loss = torch.tensor(0.0, device=device)
    no_term_stats = {
        'no_term_count': 0,
        's_max_values': [],
        'margin_violations': 0,
        'avg_s_max': 0.0
    }

    if getattr(args, "use_no_term_loss", False) and getattr(args, "enable_no_term", True):
        # 构造 no-term 掩码：has_targets 来自 batch
        has_term_tensor = torch.tensor([bool(x) for x in has_targets], device=device)
        no_term_mask = ~has_term_tensor
        no_term_count = no_term_mask.sum().item()
        no_term_stats['no_term_count'] = no_term_count

        if no_term_mask.any():
            # 获取无术语样本的音频embeddings
            no_term_audio_emb = audio_emb[no_term_mask]  # [B_no_term, D]
            no_term_audio_emb_norm = F.normalize(no_term_audio_emb, p=2, dim=1)
            
            # 使用FAISS全库检索计算s_max（如果有FAISS索引）
            if hn_ctx is not None and getattr(hn_ctx, "faiss_index", None) is not None:
                try:
                    # 使用FAISS索引进行全库检索
                    top_m = int(getattr(args, "no_term_top_m", 100))  # 检索top-M候选
                    queries = no_term_audio_emb_norm.detach().to("cpu").float().numpy()
                    D, I = hn_ctx.faiss_index.search(queries, top_m)  # D: similarity for IP / distance for L2
                    
                    # 转换为相似度分数
                    if hn_ctx.metric == 'l2':
                        # L2距离转换为相似度（距离越小相似度越高）
                        sim_scores = -torch.tensor(D, device=device, dtype=torch.float32)
                    else:
                        # IP分数直接作为相似度
                        sim_scores = torch.tensor(D, device=device, dtype=torch.float32)
                    
                    # 取每个no-term样本的最大相似度
                    s_max = sim_scores.max(dim=1).values  # [B_no_term]
                    no_term_stats['s_max_values'] = s_max.detach().cpu().tolist()
                    no_term_stats['avg_s_max'] = s_max.mean().item()
                    
                    # 计算margin loss
                    margin = float(getattr(args, "no_term_margin", 0.15))
                    margin_violations = (s_max > margin).sum().item()
                    no_term_stats['margin_violations'] = margin_violations
                    no_term_loss = F.relu(s_max - margin).mean()
                    
                except Exception as e:
                    print(f"[WARN] FAISS no-term loss failed, falling back to batch terms: {e}")
                    # 回退到batch内术语的方式
                    if 'terms_emb' in locals() and terms_emb is not None and terms_emb.numel() > 0:
                        t_norm = F.normalize(terms_emb, p=2, dim=1)
                        sim_all = no_term_audio_emb_norm @ t_norm.T  # [B_no_term, N_terms_in_batch]
                        s_max = sim_all.max(dim=1).values
                        no_term_stats['s_max_values'] = s_max.detach().cpu().tolist()
                        no_term_stats['avg_s_max'] = s_max.mean().item()
                        margin = float(getattr(args, "no_term_margin", 0.15))
                        margin_violations = (s_max > margin).sum().item()
                        no_term_stats['margin_violations'] = margin_violations
                        no_term_loss = F.relu(s_max - margin).mean()
            
            # 如果没有FAISS索引，使用batch内术语（原有逻辑）
            elif 'terms_emb' in locals() and terms_emb is not None and terms_emb.numel() > 0:
                t_norm = F.normalize(terms_emb, p=2, dim=1)
                sim_all = no_term_audio_emb_norm @ t_norm.T  # [B_no_term, N_terms_in_batch]
                s_max = sim_all.max(dim=1).values
                no_term_stats['s_max_values'] = s_max.detach().cpu().tolist()
                no_term_stats['avg_s_max'] = s_max.mean().item()
                margin = float(getattr(args, "no_term_margin", 0.15))
                margin_violations = (s_max > margin).sum().item()
                no_term_stats['margin_violations'] = margin_violations
                no_term_loss = F.relu(s_max - margin).mean()

    # === Hard Negative Mining Loss ===
    hard_neg_loss = torch.tensor(0.0, device=device)
    if getattr(args, "enable_hard_neg", False) and hn_ctx is not None and len(all_gt_terms) > 0 and len(audio_term_pairs) > 0:
        try:
            # Normalize audio embeddings for cosine/IP stability
            audio_emb_norm = torch.nn.functional.normalize(audio_emb, p=2, dim=1)

            # Build one positive text embedding per sample (detach & normalize)
            sample_pos_emb = [None] * batch_size
            seen_pos = set()
            for (a_i, t_idx) in audio_term_pairs:
                if a_i not in seen_pos:
                    pos_e = terms_emb[t_idx].detach()
                    pos_e = torch.nn.functional.normalize(pos_e, p=2, dim=0)
                    sample_pos_emb[a_i] = pos_e
                    seen_pos.add(a_i)

            k = int(getattr(args, "hard_neg_k", 10))
            cand = int(getattr(args, "hard_neg_candidates", max(50, 5 * k)))
            margin = float(getattr(args, "hard_neg_margin", 0.1))

            losses = []

            # Case A: FAISS index mode (preferred for large glossary)
            if getattr(hn_ctx, "faiss_index", None) is not None:
                # Prepare query matrix on CPU float32 (FAISS expects np.float32)
                queries = audio_emb_norm.detach().to("cpu").float().numpy()
                # Perform ANN search
                D, I = hn_ctx.faiss_index.search(queries, cand)  # D: similarity for IP / distance for L2

                # Per-sample hinge over top-k after filtering GT terms
                for i in range(batch_size):
                    pos_emb_i = sample_pos_emb[i]
                    if pos_emb_i is None:
                        continue

                    # Filter out GT indices (mapped through term2idx)
                    gt_terms_i = set(t for t in ground_truth_terms_list[i] if isinstance(t, str))
                    gt_idx_in_ctx = set(hn_ctx.term2idx[t] for t in gt_terms_i if t in hn_ctx.term2idx)

                    if I.shape[0] == 0:
                        continue
                    cand_idx = I[i].tolist()
                    cand_scores = D[i].tolist()

                    # Keep only non-GT candidates, then take top-k
                    filtered = [(idx, score) for idx, score in zip(cand_idx, cand_scores) if idx not in gt_idx_in_ctx and idx >= 0]
                    if not filtered:
                        continue
                    filtered = filtered[:k] if len(filtered) > k else filtered

                    # Compute sim_pos on torch
                    sim_pos = torch.sum(audio_emb_norm[i] * pos_emb_i)

                    # sim_neg comes directly from FAISS results:
                    if hn_ctx.metric == 'l2':
                        sim_negs_vals = [-float(score) for _, score in filtered]
                    else:
                        sim_negs_vals = [float(score) for _, score in filtered]

                    sim_negs = torch.tensor(sim_negs_vals, device=device, dtype=sim_pos.dtype)
                    loss_i = torch.relu(margin + sim_negs - sim_pos).mean()
                    losses.append(loss_i)

                if len(losses) > 0:
                    hard_neg_loss = torch.stack(losses).mean()

            # Case B: In-memory tensor mode (small bank fallback)
            elif getattr(hn_ctx, "emb_tensor", None) is not None:
                hn_emb_tensor = hn_ctx.emb_tensor.to(device)
                sim_full = audio_emb_norm @ hn_emb_tensor.T
                if k > 0 and sim_full.shape[1] > 0:
                    for i in range(batch_size):
                        pos_emb_i = sample_pos_emb[i]
                        if pos_emb_i is None:
                            continue
                        gt_terms_i = set(t for t in ground_truth_terms_list[i] if isinstance(t, str))
                        gt_idx_in_ctx = [hn_ctx.term2idx[t] for t in gt_terms_i if t in hn_ctx.term2idx]
                        take_n = min(sim_full.shape[1], max(k, cand) + max(0, len(gt_idx_in_ctx)))
                        if take_n == 0:
                            continue
                        top_vals, top_idx = torch.topk(sim_full[i], k=take_n, largest=True)
                        if len(gt_idx_in_ctx) > 0:
                            mask = ~torch.isin(top_idx, torch.tensor(gt_idx_in_ctx, device=top_idx.device))
                            top_idx = top_idx[mask]
                            top_vals = top_vals[mask]
                        if top_idx.numel() == 0:
                            continue
                        if top_idx.numel() > k:
                            top_vals = top_vals[:k]
                        sim_pos = torch.sum(audio_emb_norm[i] * pos_emb_i)
                        loss_i = torch.relu(margin + top_vals - sim_pos).mean()
                        losses.append(loss_i)
                    if len(losses) > 0:
                        hard_neg_loss = torch.stack(losses).mean()
        except Exception as e:
            print(f"[WARN] Hard-negative mining failed: {e}")
            hard_neg_loss = torch.tensor(0.0, device=device)

    # === 组合总损失 ===
    total_loss = args.audio_text_loss_ratio * contrastive_loss + args.audio_term_loss_ratio * audio_term_loss
    
    # 添加hard negative损失
    if getattr(args, "enable_hard_neg", False):
        hn_weight = float(getattr(args, "hard_neg_weight", 0.2))
        total_loss = total_loss + hn_weight * hard_neg_loss
    
    # 添加无术语margin损失
    if getattr(args, "use_no_term_loss", False) and getattr(args, "enable_no_term", True):
        lambda_no_term = float(getattr(args, "lambda_no_term", 0.5))
        total_loss = total_loss + lambda_no_term * no_term_loss
    
    # 最终数值稳定性检查
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print(f"[ERROR] NaN/Inf total loss detected, skipping batch")
        return torch.tensor(0.0, requires_grad=True).to(device)
    
    # 返回损失和统计信息
    return total_loss, no_term_stats


# === Helper to build hard-neg context ===
def build_hardneg_ctx(raw_model, source_terms, device, batch_size=2048):
    """
    Legacy helper retained for small-bank mode: encodes `source_terms` into an in-memory
    normalized tensor. For large glossary, prefer FAISS index mode (see epoch setup).
    """
    if not source_terms:
        return None

    cleaned = []
    seen = set()
    for t in source_terms:
        if not isinstance(t, str):
            continue
        tl = t.strip().lower()
        if len(tl) < 3:
            continue
        if tl in seen:
            continue
        seen.add(tl)
        cleaned.append(tl)

    if len(cleaned) == 0:
        return None

    text_emb = encode_texts_in_batches(raw_model, cleaned, device=device)
    if text_emb is None or text_emb.numel() == 0:
        return None

    text_emb = text_emb.to(device)
    text_emb = torch.nn.functional.normalize(text_emb, p=2, dim=1)
    term2idx = {t: i for i, t in enumerate(cleaned)}
    return HardNegContext(terms=cleaned, term2idx=term2idx, emb_tensor=text_emb, faiss_index=None, metric='ip')


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


def evaluate_topk_recall(model, retriever, dataset, device, top_ks=(5, 10, 20), max_eval=1000, field="term", train_terms=None, show_missed_terms=True, no_term_margin=0.15, enable_no_term=False, filter_no_term=True):
    """评估top-k召回率，包括no-term样本的拒答能力评估"""
    model.eval()
    print(f"[INFO] Evaluation no-term samples enabled: {enable_no_term}")
    
    # 用于存储sample-level召回率
    recall_dict = {k: [] for k in top_ks}
    
    # 用于存储所有GT术语和对应的检索结果（用于分析未命中术语）
    all_gt_terms_with_retrieval = {k: [] for k in top_ks}  # 每个元素是 (gt_term, is_retrieved, sample_info)
    sample_info_for_debug = []  # 用于调试输出
    
    # 用于存储no-term样本的拒答能力评估
    no_term_stats = {k: {'total': 0, 'correct_rejections': 0, 'max_sims': [], 'violations': 0} for k in top_ks}

    # === 重建索引 ===
    text_terms = [term['term'] for term in retriever.term_list]
    print(f'[DEBUG] Building index with {len(text_terms)} terms')
    
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    text_emb = encode_texts_in_batches(raw_model, text_terms, device=device)
    
    retriever.index.reset()
    # 确保tensor在CPU上并转换为numpy
    if isinstance(text_emb, torch.Tensor):
        text_emb_numpy = text_emb.detach().cpu().float().numpy()
    else:
        text_emb_numpy = text_emb.astype(np.float32)
    retriever.index.add(text_emb_numpy)
    print(f'[DEBUG] Index built with {retriever.index.ntotal} vectors')

    print(f"[INFO] Dataset size: {len(dataset)}")
    import random
    random.seed(42)  # 固定随机种子确保可复现
    eval_indices = random.sample(range(len(dataset)), min(max_eval, len(dataset)))
    
    # 分离有术语和无术语的样本
    term_samples = []
    term_indices = []
    no_term_samples = []
    no_term_indices = []
    
    for i in eval_indices:
        sample = dataset[i]
        if sample is not None:
            ground_truth_terms, audio_path, chunk_text, has_target = sample
            if has_target and ground_truth_terms:  # 有术语的样本
                term_samples.append(sample)
                term_indices.append(i)
            elif (not has_target or not ground_truth_terms) and enable_no_term and not filter_no_term:  # 无术语的样本（仅在启用且不过滤时评估）
                no_term_samples.append(sample)
                no_term_indices.append(i)

    print(f"[INFO] Selected {len(eval_indices)} samples randomly:")
    print(f"[INFO]   - {len(term_samples)} term samples (for recall evaluation)")
    print(f"[INFO]   - {len(no_term_samples)} no-term samples (for rejection evaluation)")
    
    # === 处理有术语的样本 ===
    if len(term_samples) > 0:
        # 使用term chunk音频进行编码（分批处理）
        term_audio_paths = [sample[1] for sample in term_samples]  # term_chunk_audio paths
        
        # 验证音频文件
        print(f"[DEBUG] Validating {len(term_audio_paths)} term audio files for evaluation...")
        valid_term_audio_paths, valid_term_audio_indices = validate_audio_batch(term_audio_paths, verbose=False)
        
        if len(valid_term_audio_paths) != len(term_audio_paths):
            print(f"[WARN] Term evaluation: Only {len(valid_term_audio_paths)}/{len(term_audio_paths)} audio files are valid")
            # 过滤掉无效的样本
            term_samples = [term_samples[i] for i in valid_term_audio_indices]
            term_indices = [term_indices[i] for i in valid_term_audio_indices]
            term_audio_paths = valid_term_audio_paths
        
        if len(term_audio_paths) > 0:
            print(f"[DEBUG] Encoding {len(term_audio_paths)} valid term audio files...")
            term_audio_embs_tensor = encode_audios_in_batches(raw_model, term_audio_paths, batch_size=1000, device=device)
            # 确保tensor在CPU上并转换为numpy
            if isinstance(term_audio_embs_tensor, torch.Tensor):
                term_audio_embs = term_audio_embs_tensor.detach().cpu().float().numpy()
            else:
                term_audio_embs = term_audio_embs_tensor.astype(np.float32)
        else:
            term_audio_embs = np.array([])
    else:
        term_audio_embs = np.array([])
    
    # === 评估有术语的样本 ===
    if len(term_samples) > 0 and term_audio_embs.size > 0:
        print(f"[INFO] Evaluating {len(term_samples)} term samples for recall...")
        for j, (i, sample) in enumerate(zip(term_indices, term_samples)):
            ground_truth_terms, audio_path, chunk_text, has_target = sample
            audio_emb = term_audio_embs[j:j+1]  # shape: [1, 512]
            gt_terms = [t.lower() for t in ground_truth_terms]  # 使用term_chunk_audio_ground_truth_terms

            # 对每个top_k进行检索
            retrieval_results = {}
            for top_k in top_ks:
                D, I = retriever.index.search(audio_emb, top_k)
                retrieved_terms = [retriever.term_list[idx][field].lower() for idx in I[0]]
                retrieval_results[top_k] = (D[0], I[0], retrieved_terms)
                
                # 计算sample-level召回率
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
    
    # 计算sample-level和term-level召回率
    for top_k in top_ks:
        print(f"\n=== Evaluation Results for Top-{top_k} ===")
        
        # === Term样本召回率评估 ===
        if len(recall_dict[top_k]) > 0:
            # Sample-level平均召回率
            avg_recall = sum(recall_dict[top_k]) / len(recall_dict[top_k])
            print(f"[EVAL] Term Samples - Sample-level Average Recall@{top_k}: {avg_recall:.2%} ({len(recall_dict[top_k])} samples)")
            
            # Term-level微平均召回率
            term_retrieval_pairs = all_gt_terms_with_retrieval[top_k]
            total_terms = len(term_retrieval_pairs)
            hit_terms = sum(1 for _, is_retrieved, _ in term_retrieval_pairs if is_retrieved)
            term_micro_avg_recall = hit_terms / total_terms if total_terms > 0 else 0.0
            print(f"[EVAL] Term Samples - Term-level Micro-Average Recall@{top_k}: {term_micro_avg_recall:.2%} ({hit_terms}/{total_terms} terms)")
        else:
            print(f"[EVAL] Term Samples - No term samples evaluated for Recall@{top_k}")

    model.train()
    return recall_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)  # 可能需要适当调整
    parser.add_argument('--lr', type=float, default=5e-5)  
    parser.add_argument('--patience', type=int, default=2)  
    parser.add_argument('--unfreeze_layers', type=int, default=0, 
                       help="Number of last layers to unfreeze in both encoders (default: 0, all frozen)")
    parser.add_argument('--train_samples_path', type=str, 
                       default="data/xl_term_level_chunks_merged.json",
                       help="Path to term-level chunk samples")
    parser.add_argument('--test_samples_path', type=str, default=None,
                       help="Path to separate test samples. If not provided, will use train_ratio to split training data")
    parser.add_argument('--train_ratio', type=float, default=0.99,
                       help="Ratio of samples to use for training (default: 0.99, only used when test_samples_path is not provided)")
    parser.add_argument('--glossary_path', type=str, default="data/terms/glossary_filtered.json")
    parser.add_argument('--save_path', type=str, default="data/qwen2_audio_term_level.pt")
    parser.add_argument('--best_model_path', type=str, default=None,
                       help="Path to best model checkpoint (.pt file) to continue training from")
    parser.add_argument('--enable_full_eval', action='store_true', 
                       help="Enable full evaluation with complete glossary at the end of training")
    parser.add_argument('--full_eval_every_n_epochs', type=int, default=5,
                       help="Run full evaluation every N epochs (requires --enable_full_eval)")
    parser.add_argument('--audio_text_loss_ratio', type=float, default=0.3,
                       help="Weight for audio-text contrastive loss (default: 0.3)")
    parser.add_argument('--audio_term_loss_ratio', type=float, default=0.7,
                       help="Weight for audio-term contrastive loss (default: 0.7)")

    # 拒答相关参数
    parser.add_argument('--enable_no_term', action='store_true', default=False,
                        help="Enable no-term samples in dataset and evaluation (default: False)")
    parser.add_argument('--filter_no_term', action='store_true', default=True,
                        help="Filter out no-term samples from dataset (default: True)")
    parser.add_argument('--use_no_term_loss', action='store_true', default=False,
                        help="Enable max-sim margin loss for no-term samples (default: False)")
    parser.add_argument('--no_term_margin', type=float, default=0.15,
                        help="Margin m for max-sim loss: relu(s_max - m)")
    parser.add_argument('--lambda_no_term', type=float, default=0.5,
                        help="Weight for no-term margin loss")
    parser.add_argument('--no_term_top_m', type=int, default=100,
                        help="Top-M candidates to retrieve from FAISS for no-term loss computation")

    # Hard negative mining相关参数
    parser.add_argument('--enable_hard_neg', action='store_true', default=False,
                        help="Enable hard negative mining against top-k retrieved non-GT terms (default: False)")
    parser.add_argument('--hard_neg_source', type=str, default='used', choices=['used', 'glossary'],
                        help="Source corpus for mining hard negatives: 'used' (train+test used terms) or 'glossary' (default: used)")
    parser.add_argument('--enable_glossary_hard_neg', action='store_true', default=False,
                        help="Enable glossary-based hard negative mining with FAISS index (default: False, use used terms only)")
    parser.add_argument('--hard_neg_k', type=int, default=10,
                        help="Number of hard negatives per sample (top-k)")
    parser.add_argument('--hard_neg_weight', type=float, default=0.2,
                        help="Weight for hard negative hinge loss")
    parser.add_argument('--hard_neg_margin', type=float, default=0.1,
                        help="Margin for hinge loss: max(0, margin + sim_neg - sim_pos)")

    parser.add_argument('--hard_neg_index_path', type=str, default=None,
                        help="Path to FAISS ANN index for the full glossary (IVF/HNSW/Flat). If set, enables large-bank hard negatives.")
    parser.add_argument('--hard_neg_term2idx_path', type=str, default=None,
                        help="Path to JSON mapping term_string -> int_index that matches the FAISS index order.")
    parser.add_argument('--hard_neg_candidates', type=int, default=100,
                        help="Number of ANN candidates to fetch before filtering GT (then take top-k).")
    parser.add_argument('--hard_neg_nprobe', type=int, default=16,
                        help="FAISS nprobe / efSearch parameter for IVF/HNSW-like indices.")
    parser.add_argument('--hard_neg_metric', type=str, default='ip', choices=['ip', 'l2'],
                        help="Similarity metric of the FAISS index: 'ip' for inner product (recommended with normalized vectors) or 'l2'.")

    # GPU设备选择
    parser.add_argument('--gpu_ids', type=str, default=None,
                        help="GPU IDs to use (e.g., '0,1' or '2'). If not specified, use all available GPUs.")
    
    # Qwen2-Audio模型参数
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2-Audio-7B-Instruct",
                        help="Qwen2-Audio model name or path")
    parser.add_argument('--force_single_gpu', action='store_true', default=False,
                        help="Force single GPU usage even when multiple GPUs are available (default: False)")
    parser.add_argument('--batch_size_per_gpu', type=int, default=None,
                        help="Batch size per GPU (will auto-adjust total batch size based on GPU count)")
    
    # LoRA微调参数
    parser.add_argument('--lora_r', type=int, default=16,
                        help="LoRA rank (default: 16)")
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help="LoRA alpha scaling parameter (default: 32)")
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                        help="LoRA dropout rate (default: 0.1)")

    args = parser.parse_args()

    # 处理no-term配置逻辑
    # 如果enable_no_term=False，则自动设置filter_no_term=True
    if not args.enable_no_term:
        args.filter_no_term = True
    
    print(f"[DEBUG] audio_text_loss_ratio={args.audio_text_loss_ratio}, audio_term_loss_ratio={args.audio_term_loss_ratio}")
    print(f"[DEBUG] enable_no_term={args.enable_no_term}, filter_no_term={args.filter_no_term}, use_no_term_loss={args.use_no_term_loss}")
    print(f"[DEBUG] enable_hard_neg={args.enable_hard_neg}, enable_glossary_hard_neg={args.enable_glossary_hard_neg}")
    print(f"[DEBUG] LoRA parameters: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")

    # GPU设备设置
    if args.gpu_ids is not None:
        # 设置CUDA_VISIBLE_DEVICES环境变量
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        gpu_list = [int(x.strip()) for x in args.gpu_ids.split(',') if x.strip().isdigit()]
        print(f"[INFO] Setting CUDA_VISIBLE_DEVICES={args.gpu_ids}")
        print(f"[INFO] Will use GPUs: {gpu_list}")
        
        # 重新检查CUDA设备
        if torch.cuda.is_available():
            available_gpus = torch.cuda.device_count()
            print(f"[INFO] Available GPUs after setting CUDA_VISIBLE_DEVICES: {available_gpus}")
            device = torch.device("cuda")
        else:
            print("[WARNING] CUDA not available after setting CUDA_VISIBLE_DEVICES, falling back to CPU")
            device = torch.device("cpu")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            available_gpus = torch.cuda.device_count()
            print(f"[INFO] Using all available GPUs: {available_gpus}")
        
    print(f"[INFO] Using device: {device}")

    # === 模型初始化 ===
    print(f"[INFO] Initializing Qwen2-Audio model: {args.model_name}")
    
    # 首先初始化speech encoder
    speech_encoder = Qwen2AudioSpeechEncoder(
        model_name=args.model_name, device=device
    )

    # 让text encoder重用speech encoder的模型，避免重复加载
    text_encoder = Qwen2AudioTextEncoder(
        model_name=args.model_name, device=device, 
        shared_model=speech_encoder.get_shared_model()
    )

    model = ContrastiveQwen2AudioModel(
        speech_encoder, text_encoder, 
        hidden_dim=4096,  # Qwen2-Audio typical hidden size
        proj_dim=512,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    ).to(device)
    
    # 如果提供了best model路径，加载预训练权重
    if args.best_model_path and os.path.exists(args.best_model_path):
        print(f"[INFO] Loading pre-trained weights from {args.best_model_path}")
        try:
            state_dict = torch.load(args.best_model_path, map_location=device)
            
            # 处理 DataParallel 的情况
            if list(state_dict.keys())[0].startswith('module.'):
                # 移除 'module.' 前缀
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_state_dict[k[7:]] = v  # 移除 'module.' (7个字符)
                state_dict = new_state_dict
            
            model.load_state_dict(state_dict, strict=False)  # 使用strict=False以防模型结构稍有不同
            print(f"[INFO] Successfully loaded pre-trained weights")
        except Exception as e:
            print(f"[WARNING] Failed to load pre-trained weights: {e}")
            print(f"[INFO] Continuing with random initialization")
    else:
        print(f"[INFO] No pre-trained weights provided, using random initialization")
    
    # 为不同的参数组设置不同的学习率
    encoder_params = []
    projection_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'proj_' in name:
                projection_params.append(param)
            else:
                encoder_params.append(param)
    
    # 投影层使用更高的学习率，编码器使用较低的学习率
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': args.lr},
        {'params': projection_params, 'lr': args.lr * 10}  # 投影层学习率更高
    ])
    
    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    # 强制单GPU训练模式（LoRA + 7B模型适合单卡）
    if torch.cuda.is_available():
        # 清理显存
        torch.cuda.empty_cache()
        # 获取显存信息
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        print(f"[INFO] GPU 0 total memory: {total_memory:.1f} GB")
        
        # 对于LoRA微调的7B模型，强制使用单GPU以确保稳定性
        print(f"[INFO] Using single GPU mode for LoRA fine-tuning (recommended for stability)")
        device = torch.device("cuda:0")
        model = model.to(device)
        
        # 启用混合精度训练以节省显存
        print(f"[INFO] Enabling mixed precision training (AMP) for memory efficiency")
    else:
        print(f"[INFO] CPU mode (not recommended for large models)")
        device = torch.device("cpu")

    # === 加载数据集 ===
    print(f"[INFO] Loading term-level dataset from {args.train_samples_path}")
    if args.test_samples_path:
        print(f"[INFO] Using separate test dataset: {args.test_samples_path}")
        train_dataset = TermLevelDataset(args.train_samples_path, split="train", train_ratio=1.0, enable_no_term=args.enable_no_term, filter_no_term=args.filter_no_term)  # 使用全部训练数据
        test_dataset = TermLevelDataset(None, split="test", train_ratio=args.train_ratio, test_path=args.test_samples_path, enable_no_term=args.enable_no_term, filter_no_term=args.filter_no_term)
    else:
        print(f"[INFO] Using train ratio: {args.train_ratio:.1%} train, {1-args.train_ratio:.1%} test")
        train_dataset = TermLevelDataset(args.train_samples_path, split="train", train_ratio=args.train_ratio, enable_no_term=args.enable_no_term, filter_no_term=args.filter_no_term)
        test_dataset = TermLevelDataset(args.train_samples_path, split="test", train_ratio=args.train_ratio, enable_no_term=args.enable_no_term, filter_no_term=args.filter_no_term)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: x)
    
    print(f"[INFO] Training samples: {len(train_dataset)}")
    print(f"[INFO] Test samples: {len(test_dataset)}")
    
    # === 构建术语词表用于评估 ===
    print(f"[INFO] Building term vocabulary from training + test data...")
    used_terms_train = extract_all_used_terms(train_dataset)
    used_terms_test = extract_all_used_terms(test_dataset)

    # 合并、去重并小写
    used_terms = list(set(t.lower() for t in (used_terms_train + used_terms_test)))
    print(f"[INFO] Found {len(used_terms)} unique terms")
    
    # === 初始化 retriever 用于评估 ===
    retriever = Retriever(enable_fusion=True, device=device)
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    retriever.model = raw_model
    retriever.index = faiss.IndexFlatL2(512)  # 初始化空索引
    retriever.term_list = [{'term': t} for t in used_terms]

    # 打印模型参数信息
    print("[DEBUG] Trainable parameters:")
    trainable_params = 0
    encoder_params_count = 0
    projection_params_count = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'proj_' in name:
                projection_params_count += param.numel()
                print(f" - [PROJ] {name}: {param.shape}")
            else:
                encoder_params_count += param.numel()
                print(f" - [ENC] {name}: {param.shape}")
            trainable_params += param.numel()

    # 使用新的参数统计方法
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    param_stats = raw_model.get_trainable_parameters()
    
    # 计算总参数数（包括冻结的）
    total_model_params = 0
    for name, param in raw_model.speech_encoder.model.named_parameters():
        total_model_params += param.numel()
    
    # 如果不是共享模型，需要计算text encoder参数
    if raw_model.text_encoder.model is not raw_model.speech_encoder.model:
        for name, param in raw_model.text_encoder.model.named_parameters():
            total_model_params += param.numel()
        print(f"[DEBUG] Speech and text encoders are separate models")
    else:
        print(f"[DEBUG] Speech and text encoders share the same model (Qwen2-Audio) - no double counting")
    
    # 添加投影层参数到总数
    total_model_params += param_stats['proj_params']
    
    print(f"[INFO] Total model parameters: {total_model_params:,}")
    print(f"[INFO] Trainable parameters: {param_stats['total_trainable']:,} ({param_stats['total_trainable']/total_model_params:.2%})")
    print(f"[INFO]   - LoRA parameters: {param_stats['lora_params']:,} ({param_stats['lora_params']/param_stats['total_trainable']:.1%})")
    print(f"[INFO]   - Projection parameters: {param_stats['proj_params']:,} ({param_stats['proj_params']/param_stats['total_trainable']:.1%})")
    print(f"[INFO] Frozen parameters: {total_model_params - param_stats['total_trainable']:,} ({(total_model_params - param_stats['total_trainable'])/total_model_params:.2%})")
    print(f"[INFO] Training with {len(train_dataset)} term-level chunk samples using LoRA fine-tuning")

    # === 准备hard negative mining上下文 ===
    hardneg_source_terms = None
    faiss_index = None
    term2idx_map = {}
    
    if args.enable_hard_neg:
        if args.enable_glossary_hard_neg and args.hard_neg_source == 'glossary':
            try:
                hardneg_source_terms = load_glossary_terms(args.glossary_path)
                print(f"[INFO] Hard-neg source: glossary with {len(hardneg_source_terms)} terms")
            except Exception as e:
                print(f"[WARN] Failed to load glossary for hard negs: {e}. Falling back to used terms.")
                hardneg_source_terms = used_terms
        else:
            hardneg_source_terms = used_terms
            print(f"[INFO] Hard-neg source: used terms ({len(hardneg_source_terms)} terms)")
        
        # 加载FAISS索引（仅当启用glossary hard neg时）
        if args.enable_glossary_hard_neg and args.hard_neg_index_path:
            try:
                print(f"[INFO] Loading FAISS index from: {args.hard_neg_index_path}")
                faiss_index = faiss.read_index(args.hard_neg_index_path)
                # Try to set nprobe/efSearch if available
                try:
                    if hasattr(faiss_index, 'nprobe'):
                        faiss_index.nprobe = int(args.hard_neg_nprobe)
                        print(f"[INFO] Set index.nprobe = {faiss_index.nprobe}")
                except Exception as e:
                    print(f"[WARN] Could not set nprobe: {e}")
                term2idx_map = load_term2idx_json(args.hard_neg_term2idx_path)
                print(f"[INFO] term2idx loaded: {len(term2idx_map)} entries")
            except Exception as e:
                print(f"[WARN] Failed to load FAISS index ({args.hard_neg_index_path}): {e}")
                faiss_index = None
    else:
        print(f"[INFO] Hard negative mining disabled")

    best_recall = 0.0
    no_improve_epochs = 0
    
    # 初始化混合精度训练的scaler
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    for epoch in range(args.epochs):
        # 构建hard negative上下文（每个epoch刷新）
        hn_ctx = None
        if args.enable_hard_neg:
            # 优先使用FAISS索引模式（仅当启用glossary hard neg时）
            if args.enable_glossary_hard_neg and faiss_index is not None and term2idx_map:
                hn_ctx = HardNegContext(terms=None, term2idx=term2idx_map, emb_tensor=None,
                                        faiss_index=faiss_index, metric=getattr(args, "hard_neg_metric", "ip"))
                print(f"[INFO] Hard-neg (FAISS) ready: {len(term2idx_map)} term ids, metric={hn_ctx.metric}, nprobe={getattr(faiss_index, 'nprobe', 'N/A')}")
            elif hardneg_source_terms:
                raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
                hn_ctx = build_hardneg_ctx(raw_model, hardneg_source_terms, device=device)
                if hn_ctx is not None and hn_ctx.emb_tensor is not None:
                    print(f"[INFO] Hard-neg (in-memory) built: {len(hn_ctx.terms)} terms, emb_tensor: {tuple(hn_ctx.emb_tensor.shape)}")
                else:
                    print(f"[WARN] Hard-neg context not available this epoch")
        
        model.train()
        total_loss = 0.0

        # 训练循环
        epoch_no_term_stats = {
            'total_no_term_samples': 0,
            'total_violations': 0,
            'avg_s_max_sum': 0.0,
            'batch_count': 0
        }
        
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"[Epoch {epoch+1}/{args.epochs}]")):
            # 使用混合精度训练
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    result = train_step(model, batch, device, args, hn_ctx=hn_ctx)
            else:
                result = train_step(model, batch, device, args, hn_ctx=hn_ctx)
            
            # 处理返回结果（可能是单个loss或(loss, stats)元组）
            if isinstance(result, tuple):
                loss, no_term_batch_stats = result
                # 累积no-term统计信息
                if no_term_batch_stats['no_term_count'] > 0:
                    epoch_no_term_stats['total_no_term_samples'] += no_term_batch_stats['no_term_count']
                    epoch_no_term_stats['total_violations'] += no_term_batch_stats['margin_violations']
                    epoch_no_term_stats['avg_s_max_sum'] += no_term_batch_stats['avg_s_max'] * no_term_batch_stats['no_term_count']
                    epoch_no_term_stats['batch_count'] += 1
            else:
                loss = result
            
            if loss.requires_grad and not torch.isnan(loss) and not torch.isinf(loss):
                if scaler is not None:
                    # 使用混合精度的反向传播
                    scaler.scale(loss).backward()
                    
                    # 梯度裁剪（需要先unscale）
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # 更新参数
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # 常规反向传播
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                optimizer.zero_grad()
                total_loss += loss.item()
            elif torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARNING] Skipping batch due to NaN/Inf loss: {loss.item()}")
                optimizer.zero_grad()  # 清理梯度

        avg_loss = total_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0.0
        print(f"[INFO] Epoch {epoch+1} avg loss: {avg_loss:.4f}")
        
        # 打印no-term loss统计信息
        if args.use_no_term_loss and args.enable_no_term:
            print(f"[INFO] No-term loss settings: enabled=True, margin={args.no_term_margin:.3f}, weight={args.lambda_no_term:.3f}, top_m={args.no_term_top_m}")
            
            if epoch_no_term_stats['total_no_term_samples'] > 0:
                epoch_avg_s_max = epoch_no_term_stats['avg_s_max_sum'] / epoch_no_term_stats['total_no_term_samples']
                violation_rate = epoch_no_term_stats['total_violations'] / epoch_no_term_stats['total_no_term_samples']
                print(f"[INFO] No-term epoch stats: {epoch_no_term_stats['total_no_term_samples']} samples, "
                      f"avg_s_max={epoch_avg_s_max:.4f}, violation_rate={violation_rate:.2%} "
                      f"({epoch_no_term_stats['total_violations']}/{epoch_no_term_stats['total_no_term_samples']})")
            else:
                print(f"[WARN] No-term: 0 samples processed in this epoch")
        elif not args.enable_no_term:
            print(f"[INFO] No-term processing disabled")
        
        if args.enable_hard_neg:
            mode = "FAISS" if (args.enable_glossary_hard_neg and faiss_index is not None and term2idx_map) else ("in-memory" if hardneg_source_terms else "disabled")
            print(f"[INFO] Hard-neg settings: mode={mode}, k={args.hard_neg_k}, candidates={args.hard_neg_candidates}, weight={args.hard_neg_weight:.3f}, margin={args.hard_neg_margin:.3f}, source={args.hard_neg_source}, glossary_enabled={args.enable_glossary_hard_neg}, metric={args.hard_neg_metric}, nprobe={args.hard_neg_nprobe}")
        else:
            print(f"[INFO] Hard negative mining disabled")

        # === 保存检查点 ===
        ckpt_path = f"data/qwen2_audio_term_level_epoch{epoch+1}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"[INFO] Model saved to {ckpt_path}")

        # === 评估 ===
        print(f"\n[INFO] Epoch {epoch+1} - Evaluation with training-seen terms:")
        recall_results = evaluate_topk_recall(
            model, retriever, test_dataset, device, 
            top_ks=(5, 10), max_eval=min(1000, len(test_dataset)),  # 最多评估1000个样本
            train_terms=used_terms_train,  # 传入仅来自训练集的术语
            show_missed_terms=(epoch + 1) % 2 == 0 or epoch == args.epochs - 1,  # 每2个epoch或最后一个epoch显示详细信息
            no_term_margin=args.no_term_margin,  # 传入no-term阈值
            enable_no_term=args.enable_no_term,  # 传入no-term启用状态
            filter_no_term=args.filter_no_term  # 传入no-term过滤状态
        )
        
        # 使用 Recall@10 作为早停指标
        current_recall = sum(recall_results[10]) / len(recall_results[10]) if recall_results[10] else 0.0
        
        # 更新学习率调度器
        scheduler.step(current_recall)
        
        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        current_proj_lr = optimizer.param_groups[1]['lr']
        print(f"[INFO] Current LR - Encoder: {current_lr:.2e}, Projection: {current_proj_lr:.2e}")
        
        if current_recall > best_recall:
            best_recall = current_recall
            no_improve_epochs = 0
            # 保存最佳模型
            best_model_path = args.save_path.replace('.pt', '_best.pt')
            torch.save(model.state_dict(), best_model_path)
            print(f"[INFO] New best model saved to {best_model_path} (Recall@10: {best_recall:.2%})")
        else:
            no_improve_epochs += 1
            print(f"[INFO] No improvement for {no_improve_epochs} epochs (best: {best_recall:.2%})")
            
            if no_improve_epochs >= args.patience:
                print(f"[EARLY STOPPING] No improvement in {args.patience} epochs. Best Recall@10: {best_recall:.2%}")
                break

    # === 最终保存 ===
    torch.save(model.state_dict(), args.save_path)
    print(f"[INFO] Final model saved to {args.save_path}")
    print(f"[INFO] Training completed. Best Recall@10: {best_recall:.2%}")


if __name__ == "__main__":
    main()
