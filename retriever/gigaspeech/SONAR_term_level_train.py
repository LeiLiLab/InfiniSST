import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
from tqdm import tqdm
import argparse, os, sys
from torch.optim.lr_scheduler import ReduceLROnPlateau
import faiss
from new_retrieve import Retriever
import soundfile as sf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

# 导入原有的模型结构和一些函数
from SONAR_train import ContrastiveSpeechTextModel, load_glossary_terms, encode_texts_in_batches, encode_audios_in_batches


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
    def __init__(self, path="data/xl_term_level_chunks_merged.json", split="train", train_ratio=0.99):
        print(f"[INFO] Loading term-level chunk samples from {path}")
        with open(path, "r") as f:
            all_samples = json.load(f)
        
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
        
        print(f"[INFO] Loaded {len(self.samples)} term-level samples for {split} split")

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_path = sample["term_chunk_audio"]  # 使用term chunk音频
        chunk_text = sample["term_chunk_text"]   # 使用term chunk文本
        ground_truth_terms = sample.get('term_chunk_audio_ground_truth_terms', [])
        
        return ground_truth_terms, audio_path, chunk_text, True

    def __len__(self):
        return len(self.samples)


def train_step(model, batch, device, temperature=0.07):
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
        print(f"[DEBUG] Processing batch with {len(audio_paths)} audio files")
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
        print(f"[DEBUG] Encoding {len(audio_paths)} audio files...")
        audio_emb = raw_model.encode_audio(audio_paths)  # [B, proj_dim]
        
        # 检查音频embedding
        if torch.isnan(audio_emb).any() or torch.isinf(audio_emb).any():
            print(f"[ERROR] NaN/Inf detected in audio embeddings after encoding!")
            # 逐个检查音频文件
            for i, path in enumerate(audio_paths):
                try:
                    single_emb = raw_model.encode_audio([path])
                    if torch.isnan(single_emb).any() or torch.isinf(single_emb).any():
                        print(f"[ERROR] Bad audio embedding from: {path}")
                        # 检查音频文件详细信息
                        try:
                            data, sr = sf.read(path)
                            print(f"[DEBUG] Audio stats - Duration: {len(data)/sr:.3f}s, "
                                  f"Shape: {data.shape}, Mean: {np.mean(data):.6f}, "
                                  f"Std: {np.std(data):.6f}, Min: {np.min(data):.6f}, Max: {np.max(data):.6f}")
                        except Exception as ae:
                            print(f"[ERROR] Failed to read audio file: {ae}")
                except Exception as ee:
                    print(f"[ERROR] Failed to encode single audio {path}: {ee}")
            return torch.tensor(0.0, requires_grad=True).to(device)
        
        # 编码文本
        print(f"[DEBUG] Encoding {len(chunk_texts)} text chunks...")
        text_emb = raw_model.encode_text(chunk_texts)    # [B, proj_dim]
        
        # 检查文本embedding
        if torch.isnan(text_emb).any() or torch.isinf(text_emb).any():
            print(f"[ERROR] NaN/Inf detected in text embeddings!")
            print(f"[DEBUG] Text samples: {chunk_texts[:3]}...")
            return torch.tensor(0.0, requires_grad=True).to(device)
        
        print(f"[DEBUG] Embeddings OK - Audio: {audio_emb.shape}, Text: {text_emb.shape}")
        
    except Exception as e:
        print(f"[ERROR] Failed to encode audio/text: {e}")
        import traceback
        traceback.print_exc()
        return torch.tensor(0.0, requires_grad=True).to(device)

    # === 计算音频-文本对比损失 ===
    # 对于term-level数据，音频和对应的term文本应该高度相似
    sim_matrix = (audio_emb @ text_emb.T) / temperature  # [B, B]
    
    # 数值稳定性检查
    if torch.isnan(sim_matrix).any() or torch.isinf(sim_matrix).any():
        print(f"[ERROR] NaN/Inf in contrastive sim_matrix, skipping batch")
        return torch.tensor(0.0, requires_grad=True).to(device)
    
    # 创建正样本mask（对角线为1，表示音频i和文本i是正样本对）
    batch_size = len(audio_paths)  # 使用实际的batch size（可能已经过滤）
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

    # === 计算音频-术语对比损失 ===
    # 对于term-level数据，每个样本的ground_truth_terms通常只有一个term
    # 但仍然使用相同的逻辑以保持一致性
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
            print(f"[DEBUG] audio_emb stats: mean={audio_emb.mean().item():.4f}, std={audio_emb.std().item():.4f}")
            print(f"[DEBUG] terms_emb stats: mean={terms_emb.mean().item():.4f}, std={terms_emb.std().item():.4f}")
            print(f"[DEBUG] temperature: {temperature}")
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

    # === 组合总损失 ===
    # 与MFA chunk训练保持一致的权重：弱化audio-text loss，强化audio-term loss
    # 音频-术语检索是主要任务
    total_loss = 0.3 * contrastive_loss + 0.7 * audio_term_loss
    
    # 最终数值稳定性检查
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print(f"[ERROR] NaN/Inf total loss detected, skipping batch")
        print(f"[DEBUG] contrastive_loss: {contrastive_loss.item():.4f}")
        print(f"[DEBUG] audio_term_loss: {audio_term_loss.item():.4f}")
        return torch.tensor(0.0, requires_grad=True).to(device)
    
    return total_loss


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


def evaluate_topk_recall(model, retriever, dataset, device, top_ks=(5, 10, 20), max_eval=1000, field="term", train_terms=None):
    """评估top-k召回率，适配term-level数据格式"""
    model.eval()
    recall_dict = {k: [] for k in top_ks}

    # === 重建索引 ===
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
    
    print(f'[DEBUG] Text embeddings shape: {text_emb.shape}')
    print(f'[DEBUG] Text embeddings stats: mean={text_emb.mean().item():.4f}, std={text_emb.std().item():.4f}')
    
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
    
    # 使用term chunk音频进行编码（分批处理）
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
    audio_embs = encode_audios_in_batches(raw_model, audio_paths, batch_size=1000, device=device).numpy()

    for j, (i, sample) in enumerate(zip(valid_indices, valid_samples)):
        ground_truth_terms, audio_path, chunk_text, has_target = sample
        audio_emb = audio_embs[j:j+1]  # shape: [1, 512]

        for top_k in top_ks:
            D, I = retriever.index.search(audio_emb, top_k)
            retrieved_terms = [retriever.term_list[idx][field].lower() for idx in I[0]]
            gt_terms = [t.lower() for t in ground_truth_terms]  # 使用term_chunk_audio_ground_truth_terms

            matched = sum(gt in retrieved_terms for gt in gt_terms)
            recall = matched / len(gt_terms) if gt_terms else 0.0
            recall_dict[top_k].append(recall)

            if j < 3 and top_k == top_ks[0]:  # 只打印前3个样本的详细信息
                print(f"[DEBUG] Sample {i}:")
                print(f"[DEBUG] Audio path: {audio_path}")
                print(f"[DEBUG] Chunk text: {chunk_text}")
                print(f"[DEBUG] GT terms: {gt_terms}")
                print(f"[DEBUG] Audio embedding stats: mean={audio_emb.mean():.4f}, std={audio_emb.std():.4f}")
                print(f"[DEBUG] Retrieved indices: {I[0]}")
                print(f"[DEBUG] Retrieved distances: {D[0]}")
                print(f"[DEBUG] Retrieved terms: {retrieved_terms}")
                print(f"[DEBUG] Match count: {matched}/{len(gt_terms)}")
                print(f"[DEBUG] Recall: {recall:.2%}")
                
                # 额外检查：看看距离最近的几个terms
                if len(D[0]) > 0:
                    print(f"[DEBUG] Closest term distance: {D[0][0]:.4f}")
                    if len(set(retrieved_terms)) == 1:
                        print(f"[ERROR] All retrieved terms are identical: '{retrieved_terms[0]}'")
                print(f"[DEBUG] ---")

    # 打印统计结果
    for top_k in top_ks:
        avg_recall = sum(recall_dict[top_k]) / len(recall_dict[top_k]) if recall_dict[top_k] else 0.0
        print(f"[EVAL] Average Recall@{top_k}: {avg_recall:.2%}")

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
        else:
            print(f"[WARN] train_terms not provided, skipping seen/unseen analysis")

    model.train()
    return recall_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)  # 可能需要适当调整
    parser.add_argument('--lr', type=float, default=5e-5)  
    parser.add_argument('--patience', type=int, default=3)  
    parser.add_argument('--unfreeze_layers', type=int, default=10, 
                       help="Number of last layers to unfreeze in both encoders (default: 10)")
    parser.add_argument('--train_samples_path', type=str, 
                       default="data/xl_term_level_chunks_merged.json",
                       help="Path to term-level chunk samples")
    parser.add_argument('--train_ratio', type=float, default=0.99,
                       help="Ratio of samples to use for training (default: 0.99)")
    parser.add_argument('--glossary_path', type=str, default="data/terms/glossary_filtered.json")
    parser.add_argument('--save_path', type=str, default="data/clap_term_level.pt")
    parser.add_argument('--enable_full_eval', action='store_true', 
                       help="Enable full evaluation with complete glossary at the end of training")
    parser.add_argument('--full_eval_every_n_epochs', type=int, default=5,
                       help="Run full evaluation every N epochs (requires --enable_full_eval)")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # === 模型初始化 ===
    device = torch.device("cuda")
    speech_encoder = SpeechToEmbeddingModelPipeline(
        encoder="sonar_speech_encoder_eng", device=device
    )

    text_encoder = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=device,
        dtype=torch.float32,
    )

    model = ContrastiveSpeechTextModel(
        speech_encoder, text_encoder, 
        unfreeze_layers=args.unfreeze_layers
    ).to(device)
    
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
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    
    # 自动多GPU包装
    if torch.cuda.device_count() > 1:
        print(f"[INFO] 🚀 Detected {torch.cuda.device_count()} GPUs, wrapping with DataParallel")
        available_gpus = list(range(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model, device_ids=available_gpus)
        print(f"[INFO] ✅ DataParallel enabled on GPUs: {available_gpus}")
    else:
        print(f"[INFO] Single GPU mode")

    # === 加载数据集 ===
    print(f"[INFO] Loading term-level dataset from {args.train_samples_path}")
    print(f"[INFO] Using train ratio: {args.train_ratio:.1%} train, {1-args.train_ratio:.1%} test")
    train_dataset = TermLevelDataset(args.train_samples_path, split="train", train_ratio=args.train_ratio)
    test_dataset = TermLevelDataset(args.train_samples_path, split="test", train_ratio=args.train_ratio)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: x)
    
    print(f"[INFO] Training samples: {len(train_dataset)}")
    print(f"[INFO] Test samples: {len(test_dataset)}")
    
    # === 构建术语词表用于评估（使用训练时的used_terms） ===
    print(f"[INFO] Building term vocabulary from training + test data...")
    used_terms_train = extract_all_used_terms(train_dataset)
    used_terms_test = extract_all_used_terms(test_dataset)

    print(f"[DEBUG] Raw training terms count: {len(used_terms_train)}")
    print(f"[DEBUG] Raw test terms count: {len(used_terms_test)}")
    print(f"[DEBUG] First 10 training terms: {used_terms_train[:10] if used_terms_train else []}")
    print(f"[DEBUG] First 10 test terms: {used_terms_test[:10] if used_terms_test else []}")

    # 合并、去重并小写
    used_terms = list(set(t.lower() for t in (used_terms_train + used_terms_test)))
    print(f"[INFO] Found {len(used_terms)} unique terms")
    print(f"[INFO] Training-only terms: {len(used_terms_train)}")
    print(f"[INFO] Test-only terms: {len(used_terms_test)}")
    
    print(f"[DEBUG] Final unique terms sample: {used_terms[:20] if len(used_terms) >= 20 else used_terms}")
    
    # 分析train/test术语重叠
    train_set = set(used_terms_train)
    test_set = set(used_terms_test)
    overlap = train_set.intersection(test_set)
    print(f"[INFO] Terms overlap between train/test: {len(overlap)} terms")
    print(f"[INFO] Test terms that are unseen in training: {len(test_set - train_set)} terms")
    
    # 检查是否有异常的术语重复
    term_counts = {}
    for t in (used_terms_train + used_terms_test):
        term_counts[t.lower()] = term_counts.get(t.lower(), 0) + 1
    
    frequent_terms = [(term, count) for term, count in term_counts.items() if count > 10]
    if frequent_terms:
        print(f"[DEBUG] Most frequent terms: {sorted(frequent_terms, key=lambda x: x[1], reverse=True)[:10]}")

    # === 初始化 retriever 用于评估（使用训练时的used_terms） ===
    retriever = Retriever(enable_fusion=True, device=device)
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    retriever.model = raw_model
    retriever.index = faiss.IndexFlatL2(512)  # 初始化空索引
    retriever.term_list = [{'term': t} for t in used_terms]

    # === 准备full evaluation（如果启用） ===
    full_retriever = None
    if args.enable_full_eval:
        print(f"[INFO] Preparing full evaluation with complete glossary...")
        glossary_terms = load_glossary_terms(args.glossary_path)
        full_retriever = Retriever(enable_fusion=True, device=device)
        full_retriever.model = raw_model
        full_retriever.index = faiss.IndexFlatL2(512)
        full_retriever.term_list = [{'term': t} for t in glossary_terms]
        print(f"[INFO] Full evaluation will use {len(glossary_terms)} terms from glossary")

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

    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Frozen parameters: {frozen:,} / {total:,} ({frozen / total:.2%})")
    print(f"[INFO] Trainable parameters: {trainable_params:,}")
    print(f"[INFO]   - Encoder parameters: {encoder_params_count:,} ({encoder_params_count/trainable_params:.1%})")
    print(f"[INFO]   - Projection parameters: {projection_params_count:,} ({projection_params_count/trainable_params:.1%})")
    print(f"[INFO] Training with {len(train_dataset)} term-level chunk samples")
    print(f"[INFO] Unfrozen layers: {args.unfreeze_layers}")

    best_recall = 0.0
    no_improve_epochs = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        
        # 训练循环
        for batch in tqdm(train_dataloader, desc=f"[Epoch {epoch+1}/{args.epochs}]"):
            loss = train_step(model, batch, device)
            if loss.requires_grad and not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                
                # 梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
            elif torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARNING] Skipping batch due to NaN/Inf loss: {loss.item()}")
                optimizer.zero_grad()  # 清理梯度

        avg_loss = total_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0.0
        print(f"[INFO] Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        # === 保存检查点 ===
        ckpt_path = f"data/clap_term_level_epoch{epoch+1}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"[INFO] Model saved to {ckpt_path}")

        # === 评估 ===
        # 使用内部分割的测试集进行评估
        print(f"\n[INFO] Epoch {epoch+1} - Evaluation with training-seen terms:")
        recall_results = evaluate_topk_recall(
            model, retriever, test_dataset, device, 
            top_ks=(5, 10), max_eval=min(1000, len(test_dataset)),  # 最多评估1000个样本
            train_terms=used_terms_train  # 传入仅来自训练集的术语
        )
        
        # 使用 Recall@10 作为早停指标
        current_recall = sum(recall_results[10]) / len(recall_results[10]) if recall_results[10] else 0.0

        # === Full Evaluation（如果启用且满足频率） ===
        if args.enable_full_eval and full_retriever is not None:
            if (epoch + 1) % args.full_eval_every_n_epochs == 0 or epoch == args.epochs - 1:
                print(f"\n[INFO] Epoch {epoch+1} - Full evaluation with complete glossary:")
                full_recall_results = evaluate_topk_recall(
                    model, full_retriever, test_dataset, device,
                    top_ks=(5, 10), max_eval=min(1000, len(test_dataset)),
                    train_terms=used_terms_train  # 传入仅来自训练集的术语
                )
        
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

    # === 最终Full Evaluation ===
    if args.enable_full_eval and full_retriever is not None:
        print(f"\n" + "="*60)
        print("FINAL FULL EVALUATION WITH COMPLETE GLOSSARY")
        print("="*60)
        final_full_recall = evaluate_topk_recall(
            model, full_retriever, test_dataset, device,
            top_ks=(1, 5, 10), max_eval=min(1000, len(test_dataset)),
            train_terms=used_terms_train  # 传入仅来自训练集的术语
        )
        print(f"[INFO] Final full evaluation completed")
        print(f"[INFO] To run full evaluation separately, use:")
        print(f"[INFO] python SONAR_full_evaluate.py --model_path {args.save_path} --glossary_path {args.glossary_path}")


if __name__ == "__main__":
    main() 