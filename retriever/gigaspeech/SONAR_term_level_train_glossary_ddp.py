# === GPU设备设置 - 必须在任何CUDA操作之前进行 ===
# ===== 放在文件最开始 =====
import os
import sys
# 限制 OpenMP/BLAS 线程，避免与 DataLoader 进程抢核
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# DDP相关导入
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 提前解析命令行的 --gpu_ids 和 --local_rank
gpu_ids_arg = None
local_rank = None
for i, tok in enumerate(sys.argv):
    if tok.startswith("--gpu_ids="):
        gpu_ids_arg = tok.split("=", 1)[1].strip()
    elif tok == "--gpu_ids" and i + 1 < len(sys.argv):
        gpu_ids_arg = sys.argv[i + 1].strip()
    elif tok.startswith("--local_rank="):
        local_rank = int(tok.split("=", 1)[1])
    elif tok == "--local_rank" and i + 1 < len(sys.argv):
        local_rank = int(sys.argv[i + 1])

# 设置CUDA环境
if gpu_ids_arg:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_arg
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# 设置CUDA路径（必须在导入torch之前）
if "CUDA_HOME" not in os.environ:
    os.environ["CUDA_HOME"] = "/usr/local/cuda"
if "/usr/local/cuda/bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
if "/usr/local/cuda/lib64" not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

# ===== 再导入 torch =====
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

import torch.nn as nn
import torch.nn.functional as F
from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
import time
import psutil

# 导入原有的模型结构和一些函数
from SONAR_train import ContrastiveSpeechTextModel, load_glossary_terms, encode_texts_in_batches, encode_audios_in_batches

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ===== 顶层定义collate函数和worker初始化函数（避免pickle错误）=====
def collate_keep(batch):
    """
    保留样本原样（跳过 None），返回 list[tuple]
    避免使用default_collate，因为样本包含字符串和list
    """
    batch = [b for b in batch if b is not None]
    return batch


def worker_init_fn(worker_id):
    """Worker进程初始化函数，设置随机种子"""
    import random
    import numpy as np
    random.seed(42 + worker_id)
    np.random.seed(42 + worker_id)


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
    def __init__(self, path="data/xl_term_level_chunks_merged.json", split="train", train_ratio=0.998, test_path=None, enable_no_term=False, filter_no_term=True):
        self.enable_no_term = enable_no_term
        self.filter_no_term = filter_no_term
        if dist.get_rank() == 0:  # 只在主进程打印
            print(f"[INFO] No-term samples enabled: {enable_no_term}")
            print(f"[INFO] Filter no-term samples: {filter_no_term}")
        
        if split == "test" and test_path is not None:
            # 使用独立的测试数据集
            if dist.get_rank() == 0:
                print(f"[INFO] Loading test samples from separate file: {test_path}")
            with open(test_path, "r") as f:
                all_samples = json.load(f)
            # 对于独立测试集，不需要train_ratio分割，直接使用所有样本
            use_split_logic = False
        else:
            # 使用原有的分割逻辑
            if dist.get_rank() == 0:
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
                if invalid_audio_count <= 10 and dist.get_rank() == 0:
                    print(f"[WARN] Skipping sample {i}: {audio_path} - {reason}")
        
        if dist.get_rank() == 0:
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
                if dist.get_rank() == 0:
                    print(f"[INFO] Training split: {len(self.samples)} samples ({train_no_term_count} no-term, {len(self.samples)-train_no_term_count} term)")
            elif split == "test":
                self.samples = valid_samples[split_idx:]
                # 统计测试集中的no-term样本
                test_no_term_count = sum(1 for s in self.samples if not s.get('term_chunk_audio_ground_truth_terms'))
                if dist.get_rank() == 0:
                    print(f"[INFO] Test split: {len(self.samples)} samples ({test_no_term_count} no-term, {len(self.samples)-test_no_term_count} term)")
            else:
                raise ValueError(f"Invalid split: {split}. Must be 'train' or 'test'")
        else:
            # 独立测试集，直接使用所有有效样本
            self.samples = valid_samples
            test_no_term_count = sum(1 for s in self.samples if not s.get('term_chunk_audio_ground_truth_terms'))
            if dist.get_rank() == 0:
                print(f"[INFO] Using separate test dataset: {len(self.samples)} samples ({test_no_term_count} no-term, {len(self.samples)-test_no_term_count} term)")
        
        if dist.get_rank() == 0:
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
    """训练步骤，与原版本保持一致但移除了DataParallel相关的代码"""
    raw_model = model.module if isinstance(model, DDP) else model

    if len(batch) < 2:
        if dist.get_rank() == 0:
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
        valid_audio_paths, valid_audio_indices = validate_audio_batch(audio_paths, verbose=False)
        
        if len(valid_audio_paths) == 0:
            if dist.get_rank() == 0:
                print(f"[ERROR] No valid audio files in batch, skipping")
            return torch.tensor(0.0, requires_grad=True).to(device)
        
        if len(valid_audio_paths) != len(audio_paths):
            if dist.get_rank() == 0:
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
                if dist.get_rank() == 0:
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
            if dist.get_rank() == 0:
                print(f"[ERROR] NaN/Inf detected in audio embeddings after encoding!")
            return torch.tensor(0.0, requires_grad=True).to(device)
        
        if args.audio_text_loss_ratio > 0:
            text_emb = raw_model.encode_text(chunk_texts)    # [B, proj_dim]
            
            # 检查文本embedding
            if torch.isnan(text_emb).any() or torch.isinf(text_emb).any():
                if dist.get_rank() == 0:
                    print(f"[ERROR] NaN/Inf detected in text embeddings!")
                return torch.tensor(0.0, requires_grad=True).to(device)
        else:
            text_emb = torch.zeros_like(audio_emb)
        
    except Exception as e:
        if dist.get_rank() == 0:
            print(f"[ERROR] Failed to encode audio/text: {e}")
        return torch.tensor(0.0, requires_grad=True).to(device)

    # === 计算音频-文本对比损失 ===
    batch_size = len(audio_paths)
    
    if args.audio_text_loss_ratio > 0:
        sim_matrix = (audio_emb @ text_emb.T) / temperature  # [B, B]
        if torch.isnan(sim_matrix).any() or torch.isinf(sim_matrix).any():
            if dist.get_rank() == 0:
                print(f"[ERROR] NaN/Inf in contrastive sim_matrix, skipping batch")
            return torch.tensor(0.0, requires_grad=True).to(device)
        
        labels = torch.arange(batch_size).to(device)
        
        try:
            loss_audio_to_text = F.cross_entropy(sim_matrix, labels)
            loss_text_to_audio = F.cross_entropy(sim_matrix.T, labels)
            
            if torch.isnan(loss_audio_to_text) or torch.isnan(loss_text_to_audio):
                if dist.get_rank() == 0:
                    print(f"[ERROR] NaN in contrastive cross_entropy, skipping batch")
                return torch.tensor(0.0, requires_grad=True).to(device)
            
            contrastive_loss = (loss_audio_to_text + loss_text_to_audio) / 2
        except Exception as e:
            if dist.get_rank() == 0:
                print(f"[ERROR] Failed to compute contrastive loss: {e}")
            return torch.tensor(0.0, requires_grad=True).to(device)
    else:
        contrastive_loss = torch.tensor(0.0, device=device)

    # === 计算音频-术语对比损失 ===
    all_gt_terms = []
    audio_term_pairs = []
    
    for i, terms in enumerate(ground_truth_terms_list):
        for term in terms:
            if term and len(term.strip()) > 0:
                term_idx = len(all_gt_terms)
                all_gt_terms.append(term.strip())
                audio_term_pairs.append((i, term_idx))
    
    if len(all_gt_terms) > 0 and len(audio_term_pairs) > 0:
        terms_emb = raw_model.encode_text(all_gt_terms)
        audio_term_sim = (audio_emb @ terms_emb.T) / temperature
        
        if torch.isnan(audio_term_sim).any() or torch.isinf(audio_term_sim).any():
            if dist.get_rank() == 0:
                print(f"[ERROR] NaN/Inf detected in audio_term_sim, skipping batch")
            return torch.tensor(0.0, requires_grad=True).to(device)
        
        # 构建正样本标签
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
            valid_audio_term_sim = audio_term_sim[valid_indices]
            valid_labels = torch.tensor([audio_term_labels[i] for i in valid_indices], device=device)
            
            audio_to_term_loss = F.cross_entropy(valid_audio_term_sim, valid_labels)
            
            # 术语到音频的损失
            term_to_audio_sim = valid_audio_term_sim.T
            term_audio_labels = []
            for term_idx in range(len(all_gt_terms)):
                corresponding_audios = [j for j, orig_i in enumerate(valid_indices) 
                                      if (orig_i, term_idx) in audio_term_pairs]
                if corresponding_audios:
                    term_audio_labels.append(corresponding_audios[0])
                else:
                    term_audio_labels.append(-1)
            
            valid_term_indices = [i for i, label in enumerate(term_audio_labels) if label >= 0]
            if len(valid_term_indices) > 0:
                valid_term_audio_sim = term_to_audio_sim[valid_term_indices]
                valid_term_labels = torch.tensor([term_audio_labels[i] for i in valid_term_indices], device=device)
                term_to_audio_loss = F.cross_entropy(valid_term_audio_sim, valid_term_labels)
            else:
                term_to_audio_loss = torch.tensor(0.0, device=device)
            
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

    if getattr(args, "use_no_term_loss", False) and not getattr(args, "filter_no_term", True):
        has_term_tensor = torch.tensor([bool(x) for x in has_targets], device=device)
        no_term_mask = ~has_term_tensor
        no_term_count = no_term_mask.sum().item()
        no_term_stats['no_term_count'] = no_term_count

        if no_term_mask.any():
            no_term_audio_emb = audio_emb[no_term_mask]
            no_term_audio_emb_norm = F.normalize(no_term_audio_emb, p=2, dim=1)
            
            if hn_ctx is not None and getattr(hn_ctx, "faiss_index", None) is not None:
                try:
                    top_m = int(getattr(args, "no_term_top_m", 100))
                    queries = no_term_audio_emb_norm.detach().to("cpu").float().numpy()
                    D, I = hn_ctx.faiss_index.search(queries, top_m)
                    
                    if hn_ctx.metric == 'l2':
                        sim_scores = -torch.tensor(D, device=device, dtype=torch.float32)
                    else:
                        sim_scores = torch.tensor(D, device=device, dtype=torch.float32)
                    
                    s_max = sim_scores.max(dim=1).values
                    no_term_stats['s_max_values'] = s_max.detach().cpu().tolist()
                    no_term_stats['avg_s_max'] = s_max.mean().item()
                    
                    margin = float(getattr(args, "no_term_margin", 0.15))
                    margin_violations = (s_max > margin).sum().item()
                    no_term_stats['margin_violations'] = margin_violations
                    no_term_loss = F.relu(s_max - margin).mean()
                    
                    if dist.get_rank() == 0:
                        print(f"[DEBUG] No-term FAISS: {no_term_count} samples, avg_s_max={no_term_stats['avg_s_max']:.4f}, violations={margin_violations}/{no_term_count}, loss={no_term_loss.item():.4f}")
                    
                except Exception as e:
                    if dist.get_rank() == 0:
                        print(f"[WARN] FAISS no-term loss failed, falling back to batch terms: {e}")
                    if 'terms_emb' in locals() and terms_emb is not None and terms_emb.numel() > 0:
                        t_norm = F.normalize(terms_emb, p=2, dim=1)
                        sim_all = no_term_audio_emb_norm @ t_norm.T
                        s_max = sim_all.max(dim=1).values
                        no_term_stats['s_max_values'] = s_max.detach().cpu().tolist()
                        no_term_stats['avg_s_max'] = s_max.mean().item()
                        margin = float(getattr(args, "no_term_margin", 0.15))
                        margin_violations = (s_max > margin).sum().item()
                        no_term_stats['margin_violations'] = margin_violations
                        no_term_loss = F.relu(s_max - margin).mean()
                        if dist.get_rank() == 0:
                            print(f"[DEBUG] No-term batch fallback: {no_term_count} samples, avg_s_max={no_term_stats['avg_s_max']:.4f}, violations={margin_violations}/{no_term_count}, loss={no_term_loss.item():.4f}")
            
            elif 'terms_emb' in locals() and terms_emb is not None and terms_emb.numel() > 0:
                t_norm = F.normalize(terms_emb, p=2, dim=1)
                sim_all = no_term_audio_emb_norm @ t_norm.T
                s_max = sim_all.max(dim=1).values
                no_term_stats['s_max_values'] = s_max.detach().cpu().tolist()
                no_term_stats['avg_s_max'] = s_max.mean().item()
                margin = float(getattr(args, "no_term_margin", 0.15))
                margin_violations = (s_max > margin).sum().item()
                no_term_stats['margin_violations'] = margin_violations
                no_term_loss = F.relu(s_max - margin).mean()
                if dist.get_rank() == 0:
                    print(f"[DEBUG] No-term batch only: {no_term_count} samples, avg_s_max={no_term_stats['avg_s_max']:.4f}, violations={margin_violations}/{no_term_count}, loss={no_term_loss.item():.4f}")
            else:
                if dist.get_rank() == 0:
                    print(f"[DEBUG] No-term: {no_term_count} samples, but no terms available for comparison, loss=0.0")

    # === Hard Negative Mining ===
    hard_neg_loss = torch.tensor(0.0, device=device)
    if getattr(args, "enable_hard_neg", False) and hn_ctx is not None:
        try:
            audio_emb_norm = torch.nn.functional.normalize(audio_emb, p=2, dim=1)
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

            if getattr(hn_ctx, "faiss_index", None) is not None:
                queries = audio_emb_norm.detach().to("cpu").float().numpy()
                D, I = hn_ctx.faiss_index.search(queries, cand)

                for i in range(batch_size):
                    pos_emb_i = sample_pos_emb[i]
                    if pos_emb_i is None:
                        continue

                    gt_terms_i = set(t for t in ground_truth_terms_list[i] if isinstance(t, str))
                    gt_idx_in_ctx = set(hn_ctx.term2idx[t] for t in gt_terms_i if t in hn_ctx.term2idx)

                    if I.shape[0] == 0:
                        continue
                    cand_idx = I[i].tolist()
                    cand_scores = D[i].tolist()

                    filtered = [(idx, score) for idx, score in zip(cand_idx, cand_scores) if idx not in gt_idx_in_ctx and idx >= 0]
                    if not filtered:
                        continue
                    filtered = filtered[:k] if len(filtered) > k else filtered

                    sim_pos = torch.sum(audio_emb_norm[i] * pos_emb_i)

                    if hn_ctx.metric == 'l2':
                        sim_negs_vals = [-float(score) for _, score in filtered]
                    else:
                        sim_negs_vals = [float(score) for _, score in filtered]

                    sim_negs = torch.tensor(sim_negs_vals, device=device, dtype=sim_pos.dtype)
                    loss_i = torch.relu(margin + sim_negs - sim_pos).mean()
                    losses.append(loss_i)

                if len(losses) > 0:
                    hard_neg_loss = torch.stack(losses).mean()

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
            if dist.get_rank() == 0:
                print(f"[WARN] Hard-negative mining failed: {e}")
            hard_neg_loss = torch.tensor(0.0, device=device)

    # === 组合总损失 ===
    total_loss = args.audio_text_loss_ratio * contrastive_loss + args.audio_term_loss_ratio * audio_term_loss
    if getattr(args, "enable_hard_neg", False):
        hn_weight = float(getattr(args, "hard_neg_weight", 0.2))
        total_loss = total_loss + hn_weight * hard_neg_loss
    
    if getattr(args, "use_no_term_loss", False) and not getattr(args, "filter_no_term", True):
        lambda_no_term = float(getattr(args, "lambda_no_term", 0.5))
        total_loss = total_loss + lambda_no_term * no_term_loss
    
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        if dist.get_rank() == 0:
            print(f"[ERROR] NaN/Inf total loss detected, skipping batch")
        return torch.tensor(0.0, requires_grad=True).to(device)
    
    return total_loss, no_term_stats


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
            if i < 5 and dist.get_rank() == 0:
                print(f"[DEBUG] extract_all_used_terms - Sample {i}: ground_truth_terms={ground_truth_terms}, chunk_text='{chunk_text}'")
    
    if dist.get_rank() == 0:
        print(f"[DEBUG] extract_all_used_terms - Processed {processed_samples} samples, {valid_samples} valid samples, {len(used_terms)} unique terms")
    return list(used_terms)


def build_hardneg_ctx(raw_model, source_terms, device, batch_size=2048):
    """构建hard negative context"""
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


def evaluate_topk_recall(model, retriever, dataset, device, top_ks=(5, 10, 20), max_eval=1000, field="term", train_terms=None, show_missed_terms=True, no_term_margin=0.15, enable_no_term=False, filter_no_term=True):
    """评估函数 - 只在主进程执行"""
    if dist.get_rank() != 0:
        return {}  # 非主进程直接返回空字典
    
    model.eval()
    
    # 用于存储sample-level召回率
    recall_dict = {k: [] for k in top_ks}
    
    # 用于存储所有GT术语和对应的检索结果
    all_gt_terms_with_retrieval = {k: [] for k in top_ks}
    sample_info_for_debug = []
    
    # 用于存储no-term样本的拒答能力评估
    no_term_stats = {k: {'total': 0, 'correct_rejections': 0, 'max_sims': [], 'violations': 0} for k in top_ks}

    # === 重建索引 ===
    text_terms = [term['term'] for term in retriever.term_list]
    print(f'[DEBUG] Building index with {len(text_terms)} terms')
    
    raw_model = model.module if isinstance(model, DDP) else model
    text_emb = encode_texts_in_batches(raw_model, text_terms, device=device)
    
    retriever.index.reset()
    retriever.index.add(text_emb)
    print(f'[DEBUG] Index built with {retriever.index.ntotal} vectors')

    print(f"[INFO] Dataset size: {len(dataset)}")
    import random
    random.seed(42)
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
            if has_target and ground_truth_terms:
                term_samples.append(sample)
                term_indices.append(i)
            elif (not has_target or not ground_truth_terms) and enable_no_term and not filter_no_term:
                no_term_samples.append(sample)
                no_term_indices.append(i)

    print(f"[INFO] Selected {len(eval_indices)} samples randomly:")
    print(f"[INFO]   - {len(term_samples)} term samples (for recall evaluation)")
    print(f"[INFO]   - {len(no_term_samples)} no-term samples (for rejection evaluation)")
    
    # === 处理有术语的样本 ===
    if len(term_samples) > 0:
        term_audio_paths = [sample[1] for sample in term_samples]
        
        print(f"[DEBUG] Validating {len(term_audio_paths)} term audio files for evaluation...")
        valid_term_audio_paths, valid_term_audio_indices = validate_audio_batch(term_audio_paths, verbose=False)
        
        if len(valid_term_audio_paths) != len(term_audio_paths):
            print(f"[WARN] Term evaluation: Only {len(valid_term_audio_paths)}/{len(term_audio_paths)} audio files are valid")
            term_samples = [term_samples[i] for i in valid_term_audio_indices]
            term_indices = [term_indices[i] for i in valid_term_audio_indices]
            term_audio_paths = valid_term_audio_paths
        
        if len(term_audio_paths) > 0:
            print(f"[DEBUG] Encoding {len(term_audio_paths)} valid term audio files...")
            term_audio_embs = encode_audios_in_batches(raw_model, term_audio_paths, batch_size=1000, device=device).numpy()
        else:
            import numpy as np
            term_audio_embs = np.array([])
    else:
        import numpy as np
        term_audio_embs = np.array([])
    
    # === 处理无术语的样本 ===
    if len(no_term_samples) > 0:
        no_term_audio_paths = [sample[1] for sample in no_term_samples]
        
        print(f"[DEBUG] Validating {len(no_term_audio_paths)} no-term audio files for evaluation...")
        valid_no_term_audio_paths, valid_no_term_audio_indices = validate_audio_batch(no_term_audio_paths, verbose=False)
        
        if len(valid_no_term_audio_paths) != len(no_term_audio_paths):
            print(f"[WARN] No-term evaluation: Only {len(valid_no_term_audio_paths)}/{len(no_term_audio_paths)} audio files are valid")
            no_term_samples = [no_term_samples[i] for i in valid_no_term_audio_indices]
            no_term_indices = [no_term_indices[i] for i in valid_no_term_audio_indices]
            no_term_audio_paths = valid_no_term_audio_paths
        
        if len(no_term_audio_paths) > 0:
            print(f"[DEBUG] Encoding {len(no_term_audio_paths)} valid no-term audio files...")
            no_term_audio_embs_tensor = encode_audios_in_batches(raw_model, no_term_audio_paths, batch_size=1000, device=device)
            if isinstance(no_term_audio_embs_tensor, torch.Tensor):
                no_term_audio_embs = no_term_audio_embs_tensor.detach().cpu().float().numpy()
            else:
                import numpy as np
                no_term_audio_embs = no_term_audio_embs_tensor.astype(np.float32)
        else:
            import numpy as np
            no_term_audio_embs = np.array([])
    else:
        import numpy as np
        no_term_audio_embs = np.array([])

    # === 评估有术语的样本 ===
    if len(term_samples) > 0 and term_audio_embs.size > 0:
        print(f"[INFO] Evaluating {len(term_samples)} term samples for recall...")
        for j, (i, sample) in enumerate(zip(term_indices, term_samples)):
            ground_truth_terms, audio_path, chunk_text, has_target = sample
            audio_emb = term_audio_embs[j:j+1]
            gt_terms = [t.lower() for t in ground_truth_terms]

            retrieval_results = {}
            for top_k in top_ks:
                D, I = retriever.index.search(audio_emb, top_k)
                retrieved_terms = [retriever.term_list[idx][field].lower() for idx in I[0]]
                retrieval_results[top_k] = (D[0], I[0], retrieved_terms)
                
                matched = sum(gt_term in retrieved_terms for gt_term in gt_terms)
                sample_recall = matched / len(gt_terms) if gt_terms else 0.0
                recall_dict[top_k].append(sample_recall)
                
                for gt_term in gt_terms:
                    is_retrieved = gt_term in retrieved_terms
                    sample_info = {
                        'sample_idx': i,
                        'audio_path': audio_path,
                        'chunk_text': chunk_text,
                        'all_gt_terms': gt_terms,
                        'retrieved_terms': retrieved_terms
                    }
                    all_gt_terms_with_retrieval[top_k].append((gt_term, is_retrieved, sample_info))

            if j < 3:
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
                    'total_gt_count': len(gt_terms),
                    'sample_type': 'term'
                })
    
    # === 评估无术语样本的拒答能力 ===
    if len(no_term_samples) > 0 and no_term_audio_embs.size > 0:
        print(f"[INFO] Evaluating {len(no_term_samples)} no-term samples for rejection ability...")
        for j, (i, sample) in enumerate(zip(no_term_indices, no_term_samples)):
            ground_truth_terms, audio_path, chunk_text, has_target = sample
            audio_emb = no_term_audio_embs[j:j+1]
            
            for top_k in top_ks:
                D, I = retriever.index.search(audio_emb, top_k)
                retrieved_terms = [retriever.term_list[idx][field].lower() for idx in I[0]]
                
                max_sim = -D[0][0] if len(D[0]) > 0 else -float('inf')
                
                no_term_stats[top_k]['total'] += 1
                no_term_stats[top_k]['max_sims'].append(max_sim)
                
                if max_sim < -no_term_margin:
                    no_term_stats[top_k]['correct_rejections'] += 1
                else:
                    no_term_stats[top_k]['violations'] += 1
                
                if j < 3 and top_k == top_ks[0]:
                    sample_info_for_debug.append({
                        'sample_idx': i,
                        'audio_path': audio_path,
                        'chunk_text': chunk_text,
                        'gt_terms': [],
                        'audio_emb': audio_emb,
                        'retrieved_indices': I,
                        'retrieved_distances': D,
                        'retrieved_terms': retrieved_terms,
                        'max_sim': max_sim,
                        'should_reject': max_sim < -no_term_margin,
                        'sample_type': 'no_term'
                    })

    # 打印调试信息
    for debug_info in sample_info_for_debug:
        print(f"[DEBUG] Sample {debug_info['sample_idx']} ({debug_info['sample_type']}):")
        print(f"[DEBUG] Audio path: {debug_info['audio_path']}")
        print(f"[DEBUG] Chunk text: {debug_info['chunk_text']}")
        print(f"[DEBUG] Audio embedding stats: mean={debug_info['audio_emb'].mean():.4f}, std={debug_info['audio_emb'].std():.4f}")
        print(f"[DEBUG] Retrieved indices: {debug_info['retrieved_indices']}")
        print(f"[DEBUG] Retrieved distances: {debug_info['retrieved_distances']}")
        print(f"[DEBUG] Retrieved terms: {debug_info['retrieved_terms']}")
        
        if debug_info['sample_type'] == 'term':
            print(f"[DEBUG] GT terms: {debug_info['gt_terms']}")
            print(f"[DEBUG] Match count: {debug_info['matched_count']}/{debug_info['total_gt_count']}")
        else:
            print(f"[DEBUG] GT terms: [] (no-term sample)")
            print(f"[DEBUG] Max similarity: {debug_info['max_sim']:.4f}")
            print(f"[DEBUG] Should reject: {debug_info['should_reject']} (threshold: {-no_term_margin:.4f})")
        
        if len(debug_info['retrieved_distances']) > 0:
            print(f"[DEBUG] Closest term distance: {debug_info['retrieved_distances'][0]:.4f}")
            if len(set(debug_info['retrieved_terms'])) == 1:
                print(f"[ERROR] All retrieved terms are identical: '{debug_info['retrieved_terms'][0]}'")
        print(f"[DEBUG] ---")

    # 计算并打印结果
    for top_k in top_ks:
        print(f"\n=== Evaluation Results for Top-{top_k} ===")
        
        if len(recall_dict[top_k]) > 0:
            avg_recall = sum(recall_dict[top_k]) / len(recall_dict[top_k])
            print(f"[EVAL] Term Samples - Sample-level Average Recall@{top_k}: {avg_recall:.2%} ({len(recall_dict[top_k])} samples)")
            
            term_retrieval_pairs = all_gt_terms_with_retrieval[top_k]
            total_terms = len(term_retrieval_pairs)
            hit_terms = sum(1 for _, is_retrieved, _ in term_retrieval_pairs if is_retrieved)
            term_micro_avg_recall = hit_terms / total_terms if total_terms > 0 else 0.0
            print(f"[EVAL] Term Samples - Term-level Micro-Average Recall@{top_k}: {term_micro_avg_recall:.2%} ({hit_terms}/{total_terms} terms)")
            
            diff = avg_recall - term_micro_avg_recall
            if diff > 0:
                print(f"[EVAL] Multi-term sample penalty: -{diff:.2%} (sample-level higher)")
            elif diff < 0:
                print(f"[EVAL] Multi-term sample benefit: +{abs(diff):.2%} (term-level higher)")
            else:
                print(f"[EVAL] No difference between sample-level and term-level recall")
        else:
            print(f"[EVAL] Term Samples - No term samples evaluated for Recall@{top_k}")
        
        no_term_stat = no_term_stats[top_k]
        if no_term_stat['total'] > 0:
            rejection_rate = no_term_stat['correct_rejections'] / no_term_stat['total']
            violation_rate = no_term_stat['violations'] / no_term_stat['total']
            avg_max_sim = sum(no_term_stat['max_sims']) / len(no_term_stat['max_sims'])
            
            print(f"[EVAL] No-term Samples - Rejection Rate@{top_k}: {rejection_rate:.2%} ({no_term_stat['correct_rejections']}/{no_term_stat['total']} samples)")
            print(f"[EVAL] No-term Samples - Violation Rate@{top_k}: {violation_rate:.2%} ({no_term_stat['violations']}/{no_term_stat['total']} samples)")
            print(f"[EVAL] No-term Samples - Average Max Similarity: {avg_max_sim:.4f} (threshold: {-no_term_margin:.4f})")
            
            if len(no_term_stat['max_sims']) > 0:
                import numpy as np
                print(f"[EVAL] No-term Samples - Max Similarity Stats: min={min(no_term_stat['max_sims']):.4f}, max={max(no_term_stat['max_sims']):.4f}, std={np.std(no_term_stat['max_sims']):.4f}")
        else:
            print(f"[EVAL] No-term Samples - No no-term samples evaluated for Top-{top_k}")
        
        print()

    # 打印未命中术语信息
    if show_missed_terms:
        for top_k in top_ks:
            term_retrieval_pairs = all_gt_terms_with_retrieval[top_k]
            missed_terms_info = []
            for gt_term, is_retrieved, sample_info in term_retrieval_pairs:
                if not is_retrieved:
                    missed_terms_info.append((gt_term, sample_info))
            
            print(f"[EVAL] Missed {len(missed_terms_info)} terms for Recall@{top_k}:")
            
            missed_terms_count = {}
            for gt_term, sample_info in missed_terms_info:
                if gt_term not in missed_terms_count:
                    missed_terms_count[gt_term] = []
                missed_terms_count[gt_term].append(sample_info)
            
            max_terms_to_show = 20
            sorted_missed_terms = sorted(missed_terms_count.items(), key=lambda x: len(x[1]), reverse=True)
            
            for i, (missed_term, sample_infos) in enumerate(sorted_missed_terms):
                if i >= max_terms_to_show:
                    remaining_terms = len(sorted_missed_terms) - max_terms_to_show
                    print(f"[EVAL]   ... and {remaining_terms} more missed terms")
                    break
                    
                print(f"[EVAL]   '{missed_term}' (missed {len(sample_infos)} times):")
                
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
            
            print()

    # 计算seen/unseen recall
    if train_terms is not None:
        for top_k in top_ks:
            seen_terms_set = set(t.lower() for t in train_terms)
            
            seen_recalls, unseen_recalls = [], []
            for recall_val, sample in zip(recall_dict[top_k], term_samples):
                gt_terms = [t.lower() for t in sample[0]]
                if any(gt in seen_terms_set for gt in gt_terms):
                    seen_recalls.append(recall_val)
                else:
                    unseen_recalls.append(recall_val)

            avg_seen = sum(seen_recalls) / len(seen_recalls) if seen_recalls else 0.0
            avg_unseen = sum(unseen_recalls) / len(unseen_recalls) if unseen_recalls else 0.0
            total_samples = len(seen_recalls) + len(unseen_recalls)
            print(f"[EVAL] Sample-level - Seen Recall@{top_k}: {avg_seen:.2%} ({len(seen_recalls)}/{total_samples} samples), Unseen Recall@{top_k}: {avg_unseen:.2%} ({len(unseen_recalls)}/{total_samples} samples)")
            
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


def setup_ddp(rank, world_size, args):
    """设置DDP环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 设置NCCL环境变量
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_P2P_DISABLE'] = '1'  # 先禁用P2P避免通信问题
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    
    # 增加超时时间
    import datetime
    timeout = datetime.timedelta(minutes=30)  # 30分钟超时
    
    # 初始化进程组
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timeout)
        if rank == 0:
            print(f"[INFO] Process group initialized successfully with port {os.environ['MASTER_PORT']}, timeout: {timeout}")
    except Exception as e:
        if rank == 0:
            print(f"[ERROR] Failed to initialize process group on port {os.environ['MASTER_PORT']}: {e}")
        raise
    
    # 设置当前进程的GPU
    torch.cuda.set_device(rank)
    
    # 验证GPU设备
    if rank == 0:
        print(f"[INFO] Using device: cuda:{rank}")
        print(f"[INFO] Device name: {torch.cuda.get_device_name(rank)}")
        print(f"[INFO] Memory: {torch.cuda.get_device_properties(rank).total_memory / 1024**3:.1f} GB")


def cleanup_ddp():
    """清理DDP环境"""
    dist.destroy_process_group()


def warmup_model_download(rank, world_size):
    """Rank0 预热下载模型权重，避免多进程并发写缓存冲突"""
    if rank == 0:
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Rank 0 warming up model download...")
        warmup_start = time.time()
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 使用CPU预热下载，避免GPU内存占用
                print(f"[INFO] [{time.strftime('%H:%M:%S')}] Downloading speech encoder (attempt {attempt+1}/{max_retries})...")
                
                # 设置环境变量避免并发写入冲突
                import os
                os.environ['FAIRSEQ2_CACHE_DIR'] = '/mnt/data2/jiaxuanluo/.cache/fairseq2'
                # 为预热阶段设置专用缓存目录
                warmup_cache_dir = f'/mnt/data2/jiaxuanluo/.cache/fairseq2_warmup'
                os.makedirs(warmup_cache_dir, exist_ok=True)
                os.environ['FAIRSEQ2_CACHE_DIR'] = warmup_cache_dir
                
                speech_encoder_warmup = SpeechToEmbeddingModelPipeline(
                    encoder="sonar_speech_encoder_eng", device="cpu"
                )
                print(f"[INFO] [{time.strftime('%H:%M:%S')}] Speech encoder downloaded successfully")
                
                print(f"[INFO] [{time.strftime('%H:%M:%S')}] Downloading text encoder...")
                text_encoder_warmup = TextToEmbeddingModelPipeline(
                    encoder="text_sonar_basic_encoder",
                    tokenizer="text_sonar_basic_encoder",
                    device="cpu",
                    dtype=torch.float32,
                )
                print(f"[INFO] [{time.strftime('%H:%M:%S')}] Text encoder downloaded successfully")
                
                # 清理预热模型以释放内存
                del speech_encoder_warmup, text_encoder_warmup
                torch.cuda.empty_cache()
                
                # 恢复正常缓存目录
                os.environ['FAIRSEQ2_CACHE_DIR'] = '/mnt/data2/jiaxuanluo/.cache/fairseq2'
                
                warmup_time = time.time() - warmup_start
                print(f"[INFO] [{time.strftime('%H:%M:%S')}] Model warmup completed in {warmup_time:.2f}s")
                break
                
            except Exception as e:
                print(f"[WARN] [{time.strftime('%H:%M:%S')}] Model warmup attempt {attempt+1} failed: {e}")
                
                if attempt < max_retries - 1:
                    # 清理可能损坏的缓存
                    import glob, shutil
                    # 检查两种可能的缓存路径结构
                    cache_patterns = [
                        '/mnt/data2/jiaxuanluo/.cache/fairseq2/assets/*',
                        '/mnt/data2/jiaxuanluo/.cache/fairseq2/*'
                    ]
                    for pattern in cache_patterns:
                        for cache_dir in glob.glob(pattern):
                            if os.path.isdir(cache_dir) and not cache_dir.endswith('/assets'):
                                # 检查是否包含临时文件
                                temp_files = glob.glob(os.path.join(cache_dir, 'tmp*'))
                                if temp_files:
                                    print(f"[INFO] [{time.strftime('%H:%M:%S')}] Cleaning corrupted cache: {cache_dir}")
                                    shutil.rmtree(cache_dir, ignore_errors=True)
                    
                    print(f"[INFO] [{time.strftime('%H:%M:%S')}] Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print(f"[WARN] [{time.strftime('%H:%M:%S')}] All warmup attempts failed. Will proceed with normal initialization")
    
    # 等待rank0完成下载
    dist.barrier()
    
    if rank != 0:
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Rank {rank} proceeding after warmup barrier")


def train_ddp(rank, world_size, args):
    """DDP训练主函数"""
    start_time = time.time()
    
    if rank == 0:
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] DDP Training started with {world_size} GPUs")
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Process {rank} using device: cuda:{rank}")
    
    # === 设置DDP环境 ===
    setup_start = time.time()
    setup_ddp(rank, world_size, args)
    device = torch.device(f"cuda:{rank}")
    
    if rank == 0:
        setup_time = time.time() - setup_start
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] DDP setup completed in {setup_time:.2f}s")
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Device: {device}")
    
    # === Rank0 预热模型下载 ===
    warmup_model_download(rank, world_size)
    
    # === 为每个进程设置独立缓存目录 ===
    import os
    process_cache_dir = f'/mnt/data2/jiaxuanluo/.cache/fairseq2_rank{rank}'
    os.makedirs(process_cache_dir, exist_ok=True)
    os.environ['FAIRSEQ2_CACHE_DIR'] = process_cache_dir
    
    if rank == 0:
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Set process-specific cache dir: {process_cache_dir}")
    
    # === 模型初始化 ===
    if rank == 0:
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Starting model initialization...")
    
    model_init_start = time.time()
    
    # 语音编码器初始化，带重试机制
    speech_encoder = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            speech_start = time.time()
            if rank == 0:
                print(f"[INFO] [{time.strftime('%H:%M:%S')}] Initializing speech encoder (attempt {attempt+1}/{max_retries})...")
            
            speech_encoder = SpeechToEmbeddingModelPipeline(
                encoder="sonar_speech_encoder_eng", device=device
            )
            speech_time = time.time() - speech_start
            if rank == 0:
                print(f"[INFO] [{time.strftime('%H:%M:%S')}] Speech encoder initialized successfully in {speech_time:.2f}s")
            break
            
        except Exception as e:
            if rank == 0:
                print(f"[ERROR] [{time.strftime('%H:%M:%S')}] Speech encoder attempt {attempt+1} failed: {e}")
            
            if attempt < max_retries - 1:
                # 清理可能损坏的缓存
                if rank == 0:
                    import glob, shutil
                    # 检查两种可能的缓存路径结构
                    cache_patterns = [
                        '/mnt/data2/jiaxuanluo/.cache/fairseq2/assets/*',
                        '/mnt/data2/jiaxuanluo/.cache/fairseq2/*'
                    ]
                    for pattern in cache_patterns:
                        for cache_dir in glob.glob(pattern):
                            if os.path.isdir(cache_dir) and not cache_dir.endswith('/assets'):
                                temp_files = glob.glob(os.path.join(cache_dir, 'tmp*'))
                                if temp_files:
                                    print(f"[INFO] [{time.strftime('%H:%M:%S')}] Cleaning corrupted cache: {cache_dir}")
                                    shutil.rmtree(cache_dir, ignore_errors=True)
                
                # 等待一段时间再重试
                if rank == 0:
                    print(f"[INFO] [{time.strftime('%H:%M:%S')}] Retrying in 10 seconds...")
                time.sleep(10)
            else:
                # 最后尝试CPU初始化然后移动到GPU
                if rank == 0:
                    print(f"[INFO] [{time.strftime('%H:%M:%S')}] Trying CPU initialization as fallback...")
                speech_start = time.time()
                speech_encoder = SpeechToEmbeddingModelPipeline(
                    encoder="sonar_speech_encoder_eng"
                )
                if hasattr(speech_encoder, 'model'):
                    speech_encoder.model = speech_encoder.model.to(device)
                    for module in speech_encoder.model.modules():
                        if hasattr(module, 'to'):
                            module.to(device)
                speech_time = time.time() - speech_start
                if rank == 0:
                    print(f"[INFO] [{time.strftime('%H:%M:%S')}] Speech encoder initialized with CPU fallback in {speech_time:.2f}s")
                break

    # 文本编码器初始化，带重试机制
    text_encoder = None
    for attempt in range(max_retries):
        try:
            text_start = time.time()
            if rank == 0:
                print(f"[INFO] [{time.strftime('%H:%M:%S')}] Initializing text encoder (attempt {attempt+1}/{max_retries})...")
            
            text_encoder = TextToEmbeddingModelPipeline(
                encoder="text_sonar_basic_encoder",
                tokenizer="text_sonar_basic_encoder",
                device=device,
                dtype=torch.float32,
            )
            text_time = time.time() - text_start
            if rank == 0:
                print(f"[INFO] [{time.strftime('%H:%M:%S')}] Text encoder initialized successfully in {text_time:.2f}s")
            break
            
        except Exception as e:
            if rank == 0:
                print(f"[ERROR] [{time.strftime('%H:%M:%S')}] Text encoder attempt {attempt+1} failed: {e}")
            
            if attempt < max_retries - 1:
                # 清理可能损坏的缓存
                if rank == 0:
                    import glob, shutil
                    # 检查两种可能的缓存路径结构
                    cache_patterns = [
                        '/mnt/data2/jiaxuanluo/.cache/fairseq2/assets/*',
                        '/mnt/data2/jiaxuanluo/.cache/fairseq2/*'
                    ]
                    for pattern in cache_patterns:
                        for cache_dir in glob.glob(pattern):
                            if os.path.isdir(cache_dir) and not cache_dir.endswith('/assets'):
                                temp_files = glob.glob(os.path.join(cache_dir, 'tmp*'))
                                if temp_files:
                                    print(f"[INFO] [{time.strftime('%H:%M:%S')}] Cleaning corrupted cache: {cache_dir}")
                                    shutil.rmtree(cache_dir, ignore_errors=True)
                
                # 等待一段时间再重试
                if rank == 0:
                    print(f"[INFO] [{time.strftime('%H:%M:%S')}] Retrying in 10 seconds...")
                time.sleep(10)
            else:
                # 最后尝试CPU初始化然后移动到GPU
                if rank == 0:
                    print(f"[INFO] [{time.strftime('%H:%M:%S')}] Trying CPU initialization as fallback...")
                text_start = time.time()
                text_encoder = TextToEmbeddingModelPipeline(
                    encoder="text_sonar_basic_encoder",
                    tokenizer="text_sonar_basic_encoder",
                    dtype=torch.float32,
                )
                if hasattr(text_encoder, 'model'):
                    text_encoder.model = text_encoder.model.to(device)
                    for module in text_encoder.model.modules():
                        if hasattr(module, 'to'):
                            module.to(device)
                text_time = time.time() - text_start
                if rank == 0:
                    print(f"[INFO] [{time.strftime('%H:%M:%S')}] Text encoder initialized with CPU fallback in {text_time:.2f}s")
                break

    # 创建对比模型
    contrastive_start = time.time()
    model = ContrastiveSpeechTextModel(
        speech_encoder, text_encoder, 
        unfreeze_layers=args.unfreeze_layers
    ).to(device)
    contrastive_time = time.time() - contrastive_start
    if rank == 0:
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Contrastive model created in {contrastive_time:.2f}s")
    
    # 加载预训练权重
    best_recall_from_checkpoint = 0.0  # 从checkpoint中读取的best_recall
    if args.best_model_path and os.path.exists(args.best_model_path):
        if rank == 0:
            print(f"[INFO] [{time.strftime('%H:%M:%S')}] Loading pre-trained weights from {args.best_model_path}...")
        load_start = time.time()
        try:
            checkpoint = torch.load(args.best_model_path, map_location=device)
            
            # 检查是否是完整的checkpoint（包含优化器状态等）还是只有state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # 完整checkpoint格式
                state_dict = checkpoint['model_state_dict']
                if rank == 0:
                    print(f"[INFO] [{time.strftime('%H:%M:%S')}] Found complete checkpoint with keys: {list(checkpoint.keys())}")
                    if 'epoch' in checkpoint:
                        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Checkpoint from epoch: {checkpoint['epoch']}")
                    if 'best_recall' in checkpoint:
                        best_recall_from_checkpoint = checkpoint['best_recall']
                        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Checkpoint best recall: {best_recall_from_checkpoint:.2%}")
            else:
                # 只有state_dict的格式
                state_dict = checkpoint
                if rank == 0:
                    print(f"[INFO] [{time.strftime('%H:%M:%S')}] Found state_dict checkpoint")
            
            # 处理DDP前缀
            if list(state_dict.keys())[0].startswith('module.'):
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_state_dict[k[7:]] = v
                state_dict = new_state_dict
                if rank == 0:
                    print(f"[INFO] [{time.strftime('%H:%M:%S')}] Removed 'module.' prefix from state_dict keys")
            
            model.load_state_dict(state_dict)
            load_time = time.time() - load_start
            if rank == 0:
                print(f"[INFO] [{time.strftime('%H:%M:%S')}] Successfully loaded weights in {load_time:.2f}s")
                
        except Exception as e:
            load_time = time.time() - load_start
            if rank == 0:
                print(f"[ERROR] [{time.strftime('%H:%M:%S')}] Failed to load weights in {load_time:.2f}s: {e}")
                print(f"[INFO] [{time.strftime('%H:%M:%S')}] Continuing with random initialization for training")
    else:
        if rank == 0:
            print(f"[INFO] [{time.strftime('%H:%M:%S')}] No pre-trained weights provided, using random initialization")
    
    # 包装模型为DDP
    if rank == 0:
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Wrapping model with DDP...")
    ddp_start = time.time()
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    ddp_time = time.time() - ddp_start
    if rank == 0:
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] DDP wrapper completed in {ddp_time:.2f}s")
    
    # 设置优化器
    if rank == 0:
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Setting up optimizer and scheduler...")
    optimizer_start = time.time()
    
    encoder_params = []
    projection_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'proj_' in name:
                projection_params.append(param)
            else:
                encoder_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': args.lr},
        {'params': projection_params, 'lr': args.lr * 10}
    ])
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=(rank == 0)
    )
    
    optimizer_time = time.time() - optimizer_start
    if rank == 0:
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Optimizer setup completed in {optimizer_time:.2f}s")
    
    # === 加载数据集 ===
    if rank == 0:
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Loading term-level dataset from {args.train_samples_path}")
    
    dataset_start = time.time()
    if args.test_samples_path:
        train_dataset = TermLevelDataset(args.train_samples_path, split="train", train_ratio=1.0, enable_no_term=args.enable_no_term, filter_no_term=args.filter_no_term)
        test_dataset = TermLevelDataset(None, split="test", train_ratio=args.train_ratio, test_path=args.test_samples_path, enable_no_term=args.enable_no_term, filter_no_term=args.filter_no_term)
    else:
        # 如果启用测试集重构，需要更大的测试集来筛选样本
        effective_train_ratio = args.train_ratio
        if args.rebuild_test_set:
            # 动态调整train_ratio以确保有足够的测试样本
            min_test_ratio = max(0.05, args.target_test_size * 2 / 50000)  # 假设大约5万样本，至少5%测试集
            if (1.0 - args.train_ratio) < min_test_ratio:
                effective_train_ratio = 1.0 - min_test_ratio
                if rank == 0:
                    print(f"[INFO] [{time.strftime('%H:%M:%S')}] Adjusting train_ratio from {args.train_ratio:.3f} to {effective_train_ratio:.3f} for test set rebuilding")
                    print(f"[INFO] [{time.strftime('%H:%M:%S')}] This ensures at least {min_test_ratio:.1%} of data for test set to achieve target unseen ratio")
            else:
                if rank == 0:
                    print(f"[INFO] [{time.strftime('%H:%M:%S')}] Current train_ratio {args.train_ratio:.3f} provides sufficient test samples")
        
        train_dataset = TermLevelDataset(args.train_samples_path, split="train", train_ratio=effective_train_ratio, enable_no_term=args.enable_no_term, filter_no_term=args.filter_no_term)
        test_dataset = TermLevelDataset(args.train_samples_path, split="test", train_ratio=effective_train_ratio, enable_no_term=args.enable_no_term, filter_no_term=args.filter_no_term)
    
    dataset_time = time.time() - dataset_start
    if rank == 0:
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Dataset loading completed in {dataset_time:.2f}s")
    
    # 使用DistributedSampler
    if rank == 0:
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Setting up DataLoader...")
    dataloader_start = time.time()
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size // world_size,  # 每个进程的batch size
        sampler=train_sampler,
        collate_fn=collate_keep,          # 使用顶层函数，可pickle
        num_workers=12,                    # 开启多进程加载
        pin_memory=True,
        persistent_workers=True,          # 复用worker，减少反复spawn
        prefetch_factor=4,                # 每worker预取2个batch
        worker_init_fn=worker_init_fn     # 顶层函数，可pickle
    )
    
    dataloader_time = time.time() - dataloader_start
    if rank == 0:
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] DataLoader setup completed in {dataloader_time:.2f}s")
    
    if rank == 0:
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Training samples: {len(train_dataset)}")
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Test samples: {len(test_dataset)}")
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Effective batch size per GPU: {args.batch_size // world_size}")
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Total effective batch size: {args.batch_size}")
    
    # === 构建术语词表用于评估 ===
    if rank == 0:
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Extracting used terms from datasets...")
    terms_start = time.time()
    
    used_terms_train = extract_all_used_terms(train_dataset)
    used_terms_test = extract_all_used_terms(test_dataset)
    
    # 计算unseen terms比例
    train_terms_set = set(t.lower() for t in used_terms_train)
    test_terms_set = set(t.lower() for t in used_terms_test)
    unseen_terms = test_terms_set - train_terms_set
    total_test_terms = len(test_terms_set)
    unseen_ratio = len(unseen_terms) / total_test_terms if total_test_terms > 0 else 0.0
    
    if rank == 0:
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Initial unseen terms ratio: {unseen_ratio:.2%} ({len(unseen_terms)}/{total_test_terms})")
    
    # 根据配置选择不同的unseen术语比例控制策略
    if args.rebuild_test_set and rank == 0:
        # 新策略：重构测试集以确保unseen术语比例
        if rank == 0:
            print(f"[INFO] [{time.strftime('%H:%M:%S')}] Rebuilding test set to ensure {args.target_unseen_ratio:.0%} unseen terms ratio...")
        
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
        
        if rank == 0:
            print(f"[DEBUG] [{time.strftime('%H:%M:%S')}] Sample classification: seen={len(seen_samples)}, unseen={len(unseen_samples)}, mixed={len(mixed_samples)}")
        
        # 检查是否有足够的unseen样本
        total_unseen_contributing = len(unseen_samples) + len(mixed_samples)
        if total_unseen_contributing < int(args.target_test_size * args.target_unseen_ratio * 0.5):  # 至少要有目标的50%
            if rank == 0:
                print(f"[WARN] [{time.strftime('%H:%M:%S')}] Insufficient unseen samples ({total_unseen_contributing}) to achieve target ratio. Consider adjusting train_ratio.")
                print(f"[WARN] [{time.strftime('%H:%M:%S')}] Current test set size: {len(test_dataset)}, unseen-contributing samples: {total_unseen_contributing}")
                print(f"[WARN] [{time.strftime('%H:%M:%S')}] Proceeding with available samples...")
        
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
        
        if rank == 0:
            print(f"[INFO] [{time.strftime('%H:%M:%S')}] Rebuilt test set: {len(test_dataset)} samples (target: {args.target_test_size})")
            print(f"[INFO] [{time.strftime('%H:%M:%S')}] New test terms: {len(test_terms_set_new)} (unseen: {len(unseen_terms_new)}, ratio: {final_unseen_ratio:.2%})")
        
        if final_unseen_ratio < args.target_unseen_ratio * 0.8 and rank == 0:  # 如果达不到目标的80%
            print(f"[WARN] [{time.strftime('%H:%M:%S')}] Final unseen ratio ({final_unseen_ratio:.2%}) is significantly lower than target ({args.target_unseen_ratio:.2%})")
            print(f"[WARN] [{time.strftime('%H:%M:%S')}] Consider using a smaller train_ratio to get more test samples")
        
        # 更新变量
        used_terms_test = used_terms_test_new
        test_terms_set = test_terms_set_new
        unseen_terms = unseen_terms_new
        unseen_ratio = final_unseen_ratio
        total_test_terms = len(test_terms_set_new)
        
    elif (unseen_ratio < args.min_unseen_ratio or args.force_unseen_ratio) and total_test_terms > 0:
        # 旧策略：从训练集中移除术语（保持向后兼容）
        if rank == 0:
            print(f"[INFO] [{time.strftime('%H:%M:%S')}] Using legacy strategy: removing terms from training set")
            print(f"[INFO] [{time.strftime('%H:%M:%S')}] Unseen ratio {unseen_ratio:.2%} < {args.min_unseen_ratio:.1%}, adjusting term distribution...")
        
        # 从训练集中移除一些术语，使它们变成unseen
        all_terms = list(train_terms_set | test_terms_set)
        target_unseen_count = max(int(total_test_terms * args.min_unseen_ratio), len(unseen_terms))
        
        # 从训练集中随机选择一些术语作为"unseen"
        import random
        random.seed(42)  # 确保可复现
        train_only_terms = train_terms_set - test_terms_set
        if len(train_only_terms) > 0:
            # 从只在训练集中出现的术语中选择一部分作为unseen
            terms_to_make_unseen = random.sample(
                list(train_only_terms), 
                min(len(train_only_terms), target_unseen_count - len(unseen_terms))
            )
            
            # 更新术语集合
            used_terms_train = [t for t in used_terms_train if t.lower() not in terms_to_make_unseen]
            used_terms_test = used_terms_test + terms_to_make_unseen
            
            # 重新计算比例
            train_terms_set = set(t.lower() for t in used_terms_train)
            test_terms_set = set(t.lower() for t in used_terms_test)
            unseen_terms = test_terms_set - train_terms_set
            unseen_ratio = len(unseen_terms) / len(test_terms_set) if len(test_terms_set) > 0 else 0.0
            
            if rank == 0:
                print(f"[INFO] [{time.strftime('%H:%M:%S')}] Adjusted unseen terms ratio: {unseen_ratio:.2%} ({len(unseen_terms)}/{len(test_terms_set)})")
                print(f"[INFO] [{time.strftime('%H:%M:%S')}] Moved {len(terms_to_make_unseen)} terms from train to test")
    else:
        if rank == 0:
            print(f"[INFO] [{time.strftime('%H:%M:%S')}] Using original test set without modifications ({len(test_dataset)} samples)")
    
    used_terms = list(set(t.lower() for t in (used_terms_train + used_terms_test)))
    
    terms_time = time.time() - terms_start
    if rank == 0:
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Term extraction completed in {terms_time:.2f}s")
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Found {len(used_terms)} unique terms")
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Training-only terms: {len(used_terms_train)}")
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Test-only terms: {len(used_terms_test)}")
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Final unseen terms ratio: {unseen_ratio:.2%} ({len(unseen_terms)} unseen terms)")
    
    # === 初始化retriever用于评估（仅主进程） ===
    retriever = None
    if rank == 0:
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Initializing retriever for evaluation...")
        retriever_start = time.time()
        
        retriever = Retriever(enable_fusion=True, device=device)
        raw_model = model.module
        retriever.model = raw_model
        retriever.index = faiss.IndexFlatL2(512)
        retriever.term_list = [{'term': t} for t in used_terms]
        
        retriever_time = time.time() - retriever_start
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Retriever initialization completed in {retriever_time:.2f}s")
    
    # === 准备hard-negative source terms ===
    hardneg_source_terms = None
    if args.enable_hard_neg:
        if rank == 0:
            print(f"[INFO] [{time.strftime('%H:%M:%S')}] Preparing hard-negative source terms...")
        hardneg_start = time.time()
        
        if args.hard_neg_source == 'glossary':
            try:
                hardneg_source_terms = load_glossary_terms(args.glossary_path)
                hardneg_time = time.time() - hardneg_start
                if rank == 0:
                    print(f"[INFO] [{time.strftime('%H:%M:%S')}] Hard-neg source: glossary with {len(hardneg_source_terms)} terms (loaded in {hardneg_time:.2f}s)")
            except Exception as e:
                hardneg_time = time.time() - hardneg_start
                if rank == 0:
                    print(f"[WARN] [{time.strftime('%H:%M:%S')}] Failed to load glossary for hard negs in {hardneg_time:.2f}s: {e}. Falling back to used terms.")
                hardneg_source_terms = used_terms
        else:
            hardneg_source_terms = used_terms
            hardneg_time = time.time() - hardneg_start
            if rank == 0:
                print(f"[INFO] [{time.strftime('%H:%M:%S')}] Hard-neg source: used terms ({len(hardneg_source_terms)} terms) (prepared in {hardneg_time:.2f}s)")

    # === Optional: Prepare FAISS index / term2idx for large-bank hard-neg ===
    faiss_index = None
    term2idx_map = {}
    if args.enable_hard_neg and args.hard_neg_source == 'glossary' and args.hard_neg_index_path:
        if rank == 0:
            print(f"[INFO] [{time.strftime('%H:%M:%S')}] Loading FAISS index from: {args.hard_neg_index_path}")
        faiss_start = time.time()
        try:
            faiss_index = faiss.read_index(args.hard_neg_index_path)
            try:
                if hasattr(faiss_index, 'nprobe'):
                    faiss_index.nprobe = int(args.hard_neg_nprobe)
                    if rank == 0:
                        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Set index.nprobe = {faiss_index.nprobe}")
            except Exception as e:
                if rank == 0:
                    print(f"[WARN] [{time.strftime('%H:%M:%S')}] Could not set nprobe: {e}")
            term2idx_map = load_term2idx_json(args.hard_neg_term2idx_path)
            faiss_time = time.time() - faiss_start
            if rank == 0:
                print(f"[INFO] [{time.strftime('%H:%M:%S')}] FAISS index and term2idx loaded in {faiss_time:.2f}s: {len(term2idx_map)} entries")
        except Exception as e:
            faiss_time = time.time() - faiss_start
            if rank == 0:
                print(f"[WARN] [{time.strftime('%H:%M:%S')}] Failed to load FAISS index in {faiss_time:.2f}s ({args.hard_neg_index_path}): {e}")
            faiss_index = None

    if rank == 0:
        # 打印模型参数信息
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Model parameter summary:")
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

        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Frozen parameters: {frozen:,} / {total:,} ({frozen / total:.2%})")
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Trainable parameters: {trainable_params:,}")
        print(f"[INFO] [{time.strftime('%H:%M:%S')}]   - Encoder parameters: {encoder_params_count:,} ({encoder_params_count/trainable_params:.1%})")
        print(f"[INFO] [{time.strftime('%H:%M:%S')}]   - Projection parameters: {projection_params_count:,} ({projection_params_count/trainable_params:.1%})")
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Training with {len(train_dataset)} term-level chunk samples")
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Unfrozen layers: {args.unfreeze_layers}")

    # 计算总初始化时间
    total_init_time = time.time() - start_time
    if rank == 0:
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] ===== Initialization completed in {total_init_time:.2f}s =====")
        # 显示GPU内存使用情况
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_allocated = torch.cuda.memory_allocated(0) / 1024**3
            gpu_cached = torch.cuda.memory_reserved(0) / 1024**3
            print(f"[INFO] [{time.strftime('%H:%M:%S')}] GPU Memory: {gpu_allocated:.2f}GB allocated, {gpu_cached:.2f}GB cached, {gpu_memory:.2f}GB total")
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Starting training loop...")

    # 使用从checkpoint中读取的best_recall，如果没有则使用0.0
    best_recall = best_recall_from_checkpoint
    no_improve_epochs = 0
    
    if rank == 0:
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Starting training with best_recall: {best_recall:.2%}")
    
    # === 训练循环 ===
    for epoch in range(args.epochs):
        epoch_start = time.time()
        if rank == 0:
            print(f"\n[INFO] [{time.strftime('%H:%M:%S')}] ===== Epoch {epoch+1}/{args.epochs} =====")
        # 设置sampler的epoch（重要！）
        train_sampler.set_epoch(epoch)
        
        # Refresh hard-negative context at the start of each epoch
        hn_ctx = None
        if args.enable_hard_neg:
            if rank == 0:
                print(f"[INFO] [{time.strftime('%H:%M:%S')}] Building hard-negative context...")
            hn_build_start = time.time()
            
            if faiss_index is not None and term2idx_map:
                hn_ctx = HardNegContext(terms=None, term2idx=term2idx_map, emb_tensor=None,
                                        faiss_index=faiss_index, metric=getattr(args, "hard_neg_metric", "ip"))
                hn_build_time = time.time() - hn_build_start
                if rank == 0:
                    print(f"[INFO] [{time.strftime('%H:%M:%S')}] Hard-neg (FAISS) ready in {hn_build_time:.2f}s: {len(term2idx_map)} term ids, metric={hn_ctx.metric}, nprobe={getattr(faiss_index, 'nprobe', 'N/A')}")
            elif hardneg_source_terms:
                raw_model = model.module
                hn_ctx = build_hardneg_ctx(raw_model, hardneg_source_terms, device=device)
                hn_build_time = time.time() - hn_build_start
                if hn_ctx is not None and hn_ctx.emb_tensor is not None:
                    if rank == 0:
                        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Hard-neg (in-memory) built in {hn_build_time:.2f}s: {len(hn_ctx.terms)} terms, emb_tensor: {tuple(hn_ctx.emb_tensor.shape)}")
                else:
                    if rank == 0:
                        print(f"[WARN] [{time.strftime('%H:%M:%S')}] Hard-neg context not available this epoch (build time: {hn_build_time:.2f}s)")

        model.train()
        total_loss = 0.0
        
        # 训练循环
        epoch_no_term_stats = {
            'total_no_term_samples': 0,
            'total_violations': 0,
            'avg_s_max_sum': 0.0,
            'batch_count': 0
        }
        
        # 创建进度条（仅主进程显示）
        if rank == 0:
            print(f"[INFO] [{time.strftime('%H:%M:%S')}] Starting training loop...")
            pbar = tqdm(train_dataloader, desc=f"[Epoch {epoch+1}/{args.epochs}]")
        else:
            pbar = train_dataloader
        
        training_start = time.time()
        for batch_idx, batch in enumerate(pbar):
            result = train_step(model, batch, device, args, hn_ctx=hn_ctx)
            
            # 处理返回结果
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
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
            elif torch.isnan(loss) or torch.isinf(loss):
                if rank == 0:
                    print(f"[WARNING] Skipping batch due to NaN/Inf loss: {loss.item()}")
                optimizer.zero_grad()

        # 同步所有进程的损失
        training_time = time.time() - training_start
        avg_loss = total_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0.0
        
        # 收集所有进程的损失进行平均
        loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / world_size
        
        if rank == 0:
            print(f"[INFO] [{time.strftime('%H:%M:%S')}] Epoch {epoch+1} training completed in {training_time:.2f}s")
            print(f"[INFO] [{time.strftime('%H:%M:%S')}] Epoch {epoch+1} avg loss: {avg_loss:.4f}")
            # 显示训练后的GPU内存使用情况
            if torch.cuda.is_available():
                gpu_allocated = torch.cuda.memory_allocated(0) / 1024**3
                gpu_cached = torch.cuda.memory_reserved(0) / 1024**3
                print(f"[INFO] [{time.strftime('%H:%M:%S')}] GPU Memory after training: {gpu_allocated:.2f}GB allocated, {gpu_cached:.2f}GB cached")
            
            # 打印no-term loss统计信息
            if args.use_no_term_loss:
                print(f"[INFO] No-term loss settings: enabled=True, margin={args.no_term_margin:.3f}, weight={args.lambda_no_term:.3f}, top_m={args.no_term_top_m}")
                
                if epoch_no_term_stats['total_no_term_samples'] > 0:
                    epoch_avg_s_max = epoch_no_term_stats['avg_s_max_sum'] / epoch_no_term_stats['total_no_term_samples']
                    violation_rate = epoch_no_term_stats['total_violations'] / epoch_no_term_stats['total_no_term_samples']
                    print(f"[INFO] No-term epoch stats: {epoch_no_term_stats['total_no_term_samples']} samples, "
                          f"avg_s_max={epoch_avg_s_max:.4f}, violation_rate={violation_rate:.2%} "
                          f"({epoch_no_term_stats['total_violations']}/{epoch_no_term_stats['total_no_term_samples']})")
                else:
                    print(f"[WARN] No-term: 0 samples processed in this epoch")
            
            if args.enable_hard_neg:
                mode = "FAISS" if (faiss_index is not None and term2idx_map) else ("in-memory" if hardneg_source_terms else "disabled")
                print(f"[INFO] Hard-neg settings: mode={mode}, k={args.hard_neg_k}, candidates={args.hard_neg_candidates}, weight={args.hard_neg_weight:.3f}, margin={args.hard_neg_margin:.3f}, source={args.hard_neg_source}, metric={args.hard_neg_metric}, nprobe={args.hard_neg_nprobe}")

        # === 保存检查点 ===（仅主进程）
        if rank == 0:
            print(f"[INFO] [{time.strftime('%H:%M:%S')}] Saving checkpoint...")
            save_start = time.time()
            ckpt_path = f"data/clap_term_level_epoch{epoch+1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            save_time = time.time() - save_start
            print(f"[INFO] [{time.strftime('%H:%M:%S')}] Model saved to {ckpt_path} in {save_time:.2f}s")

        # === 评估 ===（仅主进程）
        if rank == 0:
            print(f"\n[INFO] [{time.strftime('%H:%M:%S')}] Epoch {epoch+1} - Starting evaluation...")
            eval_start = time.time()
            recall_results = evaluate_topk_recall(
                model, retriever, test_dataset, device, 
                top_ks=(5, 10), max_eval=min(1000, len(test_dataset)),
                train_terms=used_terms_train,
                show_missed_terms=(epoch + 1) % 2 == 0 or epoch == args.epochs - 1,
                no_term_margin=args.no_term_margin,
                enable_no_term=args.enable_no_term,
                filter_no_term=args.filter_no_term
            )
            eval_time = time.time() - eval_start
            print(f"[INFO] [{time.strftime('%H:%M:%S')}] Evaluation completed in {eval_time:.2f}s")
            
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
                # 保存最佳模型（包含best_recall信息）
                best_model_path = args.save_path.replace('.pt', '_best.pt')
                best_checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'best_recall': best_recall,
                    'epoch': epoch + 1,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }
                torch.save(best_checkpoint, best_model_path)
                print(f"[INFO] New best model saved to {best_model_path} (Recall@10: {best_recall:.2%})")
                # 有改进，不需要早停
                early_stop_flag = torch.tensor(0, device=device)
            else:
                no_improve_epochs += 1
                print(f"[INFO] No improvement for {no_improve_epochs} epochs (best: {best_recall:.2%})")
                
                if no_improve_epochs >= args.patience:
                    print(f"[EARLY STOPPING] No improvement in {args.patience} epochs. Best Recall@10: {best_recall:.2%}")
                    # 设置早停标志，通知所有进程
                    early_stop_flag = torch.tensor(1, device=device)
                else:
                    early_stop_flag = torch.tensor(0, device=device)
        else:
            # 非主进程，不需要早停
            early_stop_flag = torch.tensor(0, device=device)
        
        # 广播早停标志到所有进程
        dist.broadcast(early_stop_flag, src=0)
        
        # 同步所有进程，确保主进程完成评估后再继续
        dist.barrier()
        
        # 检查是否早停
        if early_stop_flag.item() == 1:
            if rank == 0:
                print(f"[INFO] [{time.strftime('%H:%M:%S')}] Early stopping triggered, exiting training loop")
            break
        
        # 计算epoch总时间
        epoch_time = time.time() - epoch_start
        if rank == 0:
            print(f"[INFO] [{time.strftime('%H:%M:%S')}] Epoch {epoch+1} completed in {epoch_time:.2f}s")

    # === 最终保存 ===（仅主进程）
    if rank == 0:
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Saving final model...")
        final_save_start = time.time()
        # 保存完整的checkpoint，包含best_recall信息
        final_checkpoint = {
            'model_state_dict': model.state_dict(),
            'best_recall': best_recall,
            'epoch': args.epochs,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        torch.save(final_checkpoint, args.save_path)
        final_save_time = time.time() - final_save_start
        total_training_time = time.time() - start_time
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Final model saved to {args.save_path} in {final_save_time:.2f}s")
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Training completed in {total_training_time:.2f}s. Best Recall@10: {best_recall:.2%}")

    cleanup_ddp()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=5e-5)  
    parser.add_argument('--patience', type=int, default=3)  
    parser.add_argument('--unfreeze_layers', type=int, default=10, 
                       help="Number of last layers to unfreeze in both encoders (default: 10)")
    parser.add_argument('--train_samples_path', type=str, 
                       default="data/xl_term_level_chunks_merged.json",
                       help="Path to term-level chunk samples")
    parser.add_argument('--test_samples_path', type=str, default=None,
                       help="Path to separate test samples. If not provided, will use train_ratio to split training data")
    parser.add_argument('--train_ratio', type=float, default=0.998,
                       help="Ratio of samples to use for training (default: 0.998, only used when test_samples_path is not provided)")
    parser.add_argument('--glossary_path', type=str, default="data/terms/glossary_filtered.json")
    parser.add_argument('--save_path', type=str, default="data/clap_term_level.pt")
    parser.add_argument('--best_model_path', type=str, default=None,
                       help="Path to best model checkpoint (.pt file) to continue training from")
    parser.add_argument('--audio_text_loss_ratio', type=float, default=0.3,
                       help="Weight for audio-text contrastive loss (default: 0.3)")
    parser.add_argument('--audio_term_loss_ratio', type=float, default=0.7,
                       help="Weight for audio-term contrastive loss (default: 0.7)")
    
    # Hard negative mining参数
    parser.add_argument('--enable_hard_neg', action='store_true',
                        help="Enable hard negative mining against top-k retrieved non-GT terms")
    parser.add_argument('--hard_neg_source', type=str, default='used', choices=['used', 'glossary'],
                        help="Source corpus for mining hard negatives: 'used' (train+test used terms) or 'glossary'")
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

    # GPU设备选择
    parser.add_argument('--gpu_ids', type=str, default=None,
                        help="GPU IDs to use (e.g., '0,1,2,3,4,5,6,7' or '2'). If not specified, use all available GPUs.")
    
    # Unseen terms比例控制
    parser.add_argument('--min_unseen_ratio', type=float, default=0.20,
                        help="Minimum ratio of unseen terms in test set (default: 0.20)")
    parser.add_argument('--force_unseen_ratio', action='store_true',
                        help="Force adjustment of term distribution to meet min_unseen_ratio")
    parser.add_argument('--rebuild_test_set', action='store_true', 
                        help='Rebuild test set to ensure unseen terms ratio instead of removing terms from training')
    parser.add_argument('--target_test_size', type=int, default=1000, 
                        help='Target size for rebuilt test set')
    parser.add_argument('--target_unseen_ratio', type=float, default=0.20, 
                        help='Target ratio of unseen terms in rebuilt test set')

    args = parser.parse_args()

    # 处理no-term配置逻辑
    if not args.enable_no_term:
        args.filter_no_term = True

    # 检查GPU数量
    if args.gpu_ids:
        gpu_list = [int(x) for x in args.gpu_ids.split(',')]
        world_size = len(gpu_list)
    else:
        world_size = torch.cuda.device_count()
    
    if world_size == 0:
        print("[ERROR] No CUDA devices available!")
        return
    
    print(f"[INFO] Starting DDP training with {world_size} GPUs")
    print(f"[INFO] GPU IDs: {args.gpu_ids if args.gpu_ids else 'all available'}")
    print(f"[INFO] Total batch size: {args.batch_size}, per-GPU batch size: {args.batch_size // world_size}")
    
    # 启动多进程训练
    mp.spawn(train_ddp, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
