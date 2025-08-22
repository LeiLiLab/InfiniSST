import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
from tqdm import tqdm
from new_giga_speech import load_preprocessed_samples
import argparse, os, sys
from torch.optim.lr_scheduler import ReduceLROnPlateau
import faiss
from new_retrieve import Retriever

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline


class ContrastiveSpeechTextModel(nn.Module):
    def __init__(self, speech_encoder, text_encoder, hidden_dim=1024, proj_dim=512, unfreeze_layers=10):
        super().__init__()
        self.speech_encoder = speech_encoder
        self.text_encoder = text_encoder

        # projection layers
        self.proj_speech = nn.Linear(hidden_dim, proj_dim)
        self.proj_text = nn.Linear(hidden_dim, proj_dim)

        # 首先冻结所有参数
        for param in self.speech_encoder.model.parameters():
            param.requires_grad = False
        for param in self.text_encoder.model.parameters():
            param.requires_grad = False
        
        # 解冻语音编码器的后几层
        self._unfreeze_last_layers(self.speech_encoder.model, unfreeze_layers, "Speech")
        
        # 解冻文本编码器的后几层  
        self._unfreeze_last_layers(self.text_encoder.model, unfreeze_layers, "Text")
    
    def _unfreeze_last_layers(self, model, num_layers, model_type):
        """解冻模型的后几层参数"""
        # 获取所有可训练的层
        layers = []
        for name, module in model.named_modules():
            if any(layer_type in name.lower() for layer_type in ['layer', 'block', 'transformer', 'encoder']):
                if hasattr(module, 'weight') or any(hasattr(module, param) for param in ['weight', 'bias']):
                    layers.append((name, module))
        
        # 如果找不到标准的层结构，尝试按参数组解冻
        if not layers:
            all_params = list(model.named_parameters())
            # 解冻最后 num_layers * 10 个参数（粗略估计）
            unfreeze_count = min(num_layers * 10, len(all_params))
            for name, param in all_params[-unfreeze_count:]:
                param.requires_grad = True
                print(f"[INFO] {model_type} - Unfrozen parameter: {name}")
            return
        
        # 解冻后几层
        unfreeze_count = min(num_layers, len(layers))
        unfrozen_layers = layers[-unfreeze_count:]
        
        print(f"[INFO] {model_type} encoder - Unfreezing last {unfreeze_count} layers:")
        for name, module in unfrozen_layers:
            for param_name, param in module.named_parameters():
                param.requires_grad = True
                print(f"[INFO] {model_type} - Unfrozen: {name}.{param_name}")
        
        print(f"[INFO] {model_type} encoder - Total unfrozen layers: {unfreeze_count}/{len(layers)}")

    def encode_audio(self, audio_paths):
        speech_embeddings = self.speech_encoder.predict(audio_paths)  # [B, 1024]
        if isinstance(speech_embeddings, np.ndarray):
            speech_embeddings = torch.from_numpy(speech_embeddings)
        speech_embeddings = speech_embeddings.clone().detach().to(self.proj_speech.weight.device).requires_grad_(True)
        return F.normalize(self.proj_speech(speech_embeddings), dim=-1)

    def encode_text(self, texts, source_lang="eng_Latn"):
        with torch.no_grad():
            text_embeddings = self.text_encoder.predict(texts, source_lang=source_lang)  # numpy 或 tensor

        if isinstance(text_embeddings, np.ndarray):
            text_embeddings = torch.from_numpy(text_embeddings)

        text_embeddings = text_embeddings.clone().detach().to(self.proj_text.weight.device).requires_grad_(True)
        return F.normalize(self.proj_text(text_embeddings), dim=-1)


class InBatchDataset(Dataset):
    def __init__(self, path="data/samples/xl/test_mfa_3chunks_samples_0_500000.json", split="train", train_ratio=0.99):
        print(f"[INFO] Loading MFA chunk samples from {path}")
        with open(path, "r") as f:
            all_samples = json.load(f)

        # 过滤有效样本：必须有音频文件、chunk文本和ground truth terms
        valid_samples = []
        for s in all_samples:
            terms = s.get('n_chunk_audio_ground_truth_terms')
            if not (terms and isinstance(terms, list)):
                continue
            # 过滤术语
            filtered_terms = [
                t for t in terms
                if isinstance(t, str)
                and len(t) >= 3
                and sum(c.isdigit() for c in t) <= len(t) // 2
                and len(t.strip().split()) <= 5 # 过滤长术语
            ]
            if not filtered_terms:
                continue
            # 替换原列表为过滤后的术语
            s = dict(s)  # 避免直接修改原始数据
            s['n_chunk_audio_ground_truth_terms'] = filtered_terms
            # 检查其他条件
            if (
                s.get('n_chunk_text', '').strip()
                and s.get('n_chunk_audio', '')
                and os.path.exists(s.get("n_chunk_audio", ""))
            ):
                valid_samples.append(s)

        print(f"[INFO] Filtered {len(valid_samples)} valid samples from {len(all_samples)} total samples")

        # 数据分割：99%训练，1%测试
        import random
        random.seed(42)  # 固定随机种子确保可复现
        random.shuffle(valid_samples)

        split_idx = int(len(valid_samples) * train_ratio)

        if split == "train":
            self.samples = valid_samples[:split_idx]
            print(f"[INFO] Training split: {len(self.samples)} samples")
        elif split == "test":
            self.samples = valid_samples[split_idx:]
            print(f"[INFO] Test split: {len(self.samples)} samples")
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'test'")

        print(f"[INFO] Loaded {len(self.samples)} samples for {split} split")

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_path = sample["n_chunk_audio"]  # 使用chunk音频
        chunk_text = sample["n_chunk_text"]   # 使用chunk文本
        ground_truth_terms = sample.get('n_chunk_audio_ground_truth_terms', [])
        
        return ground_truth_terms, audio_path, chunk_text, True

    def __len__(self):
        return len(self.samples)


def train_step(model, batch, device, args, temperature=0.07):
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
    audio_emb = raw_model.encode_audio(audio_paths)  # [B, proj_dim]
    if args.audio_text_loss_ratio>0:
        text_emb = raw_model.encode_text(chunk_texts)    # [B, proj_dim]
        # === 计算音频-文本对比损失 ===
        # 音频和对应的chunk文本应该相似
        sim_matrix = (audio_emb @ text_emb.T) / temperature  # [B, B]
        
        # 创建正样本mask（对角线为1，表示音频i和文本i是正样本对）
        batch_size = len(audio_paths)
        labels = torch.arange(batch_size).to(device)
        
        # 计算对称的对比损失
        loss_audio_to_text = F.cross_entropy(sim_matrix, labels)
        loss_text_to_audio = F.cross_entropy(sim_matrix.T, labels)
        
        contrastive_loss = (loss_audio_to_text + loss_text_to_audio) / 2
    else:
        contrastive_loss = torch.tensor(0.0, device=device)

    # === 计算音频-术语对比损失 ===
    # 收集batch中所有的ground truth terms
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
        
        # 构建正样本标签
        # 为每个音频样本，找到它对应的正样本术语
        audio_term_labels = []
        for i in range(batch_size):
            # 找到audio i对应的所有positive term indices
            positive_terms = [term_idx for audio_idx, term_idx in audio_term_pairs if audio_idx == i]
            if positive_terms:
                # 如果有多个正样本，随机选择一个作为主要目标
                import random
                audio_term_labels.append(random.choice(positive_terms))
            else:
                # 如果没有正样本，跳过这个样本（在损失计算中会被mask掉）
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
    # 使用可配置的损失权重
    total_loss = args.audio_text_loss_ratio * contrastive_loss + args.audio_term_loss_ratio * audio_term_loss
    
    return total_loss


def extract_all_used_terms(dataset):
    """提取数据集中所有使用的术语"""
    used_terms = set()
    for sample in dataset:
        if sample is None:
            continue
        ground_truth_terms, audio_path, chunk_text, has_target = sample
        if has_target and ground_truth_terms:
            used_terms.update(t.lower() for t in ground_truth_terms if isinstance(t, str))
    return list(used_terms)


def load_glossary_terms(glossary_path):
    """加载完整的术语表"""
    print(f"[INFO] Loading glossary from {glossary_path}")
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
    return terms


def encode_texts_in_batches(model, texts, batch_size=1000, device="cuda"):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        with torch.no_grad():
            emb = model.encode_text(batch).cpu()
            all_embeddings.append(emb)
    return torch.cat(all_embeddings, dim=0)


def encode_audios_in_batches(model, audio_paths, batch_size=32, device="cuda"):
    """分批编码音频"""
    all_embeddings = []
    print(f"[INFO] Encoding {len(audio_paths)} audio files in batches of {batch_size}")
    
    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i:i + batch_size]
        print(f"[INFO] Processing audio batch {i//batch_size + 1}/{(len(audio_paths) + batch_size - 1)//batch_size}")
        
        with torch.no_grad():
            try:
                emb = model.encode_audio(batch_paths).cpu()
                all_embeddings.append(emb)
            except Exception as e:
                print(f"[ERROR] Failed to encode audio batch {i//batch_size + 1}: {e}")
                print(f"[INFO] Trying smaller batch size for this batch...")
                # 如果batch失败，尝试单个处理
                for single_path in batch_paths:
                    try:
                        single_emb = model.encode_audio([single_path]).cpu()
                        all_embeddings.append(single_emb)
                    except Exception as e2:
                        print(f"[ERROR] Failed to encode single audio {single_path}: {e2}")
                        # 跳过这个音频文件
                        continue
    
    if not all_embeddings:
        raise RuntimeError("No audio files were successfully encoded")
    
    return torch.cat(all_embeddings, dim=0)


def evaluate_topk_recall(model, retriever, dataset, device, top_ks=(5, 10, 20), max_eval=1000, field="term", train_terms=None):
    """评估top-k召回率，使用n_chunk_audio_ground_truth_terms作为目标
    
    Args:
        train_terms: 仅来自训练集的术语列表，用于区分seen/unseen terms
    """
    model.eval()
    recall_dict = {k: [] for k in top_ks}

    # === 重建索引 ===
    text_terms = [term['term'] for term in retriever.term_list]
    print(f'[DEBUG] text_terms: {len(text_terms)}')
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    text_emb = encode_texts_in_batches(raw_model, text_terms, device=device)

    retriever.index.reset()
    retriever.index.add(text_emb)

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

    # 使用chunk音频进行编码（分批处理）
    audio_paths = [sample[1] for sample in valid_samples]  # n_chunk_audio paths
    audio_embs = encode_audios_in_batches(raw_model, audio_paths, batch_size=1000, device=device).numpy()

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
    parser.add_argument('--batch_size', type=int, default=256)  # 增大batch size适应大数据集
    parser.add_argument('--lr', type=float, default=5e-5)  # 降低学习率，适合微调
    parser.add_argument('--patience', type=int, default=3)  # 减少patience，大数据集下更快收敛
    parser.add_argument('--unfreeze_layers', type=int, default=10, 
                       help="Number of last layers to unfreeze in both encoders (default: 10)")
    parser.add_argument('--train_samples_path', type=str, 
                       default="data/samples/xl/test_mfa_3chunks_samples_0_500000.json",
                       help="Path to MFA chunk samples (will be split into 99% train, 1% test)")
    parser.add_argument('--train_ratio', type=float, default=0.99,
                       help="Ratio of samples to use for training (default: 0.99)")
    parser.add_argument('--glossary_path', type=str, default="data/terms/glossary_filtered.json")
    parser.add_argument('--save_path', type=str, default="data/clap_mfa_chunks.pt")
    parser.add_argument('--enable_full_eval', action='store_true', 
                       help="Enable full evaluation with complete glossary at the end of training")
    parser.add_argument('--full_eval_every_n_epochs', type=int, default=5,
                       help="Run full evaluation every N epochs (requires --enable_full_eval)")
    parser.add_argument('--audio_text_loss_ratio', type=float, default=0.3,
                       help="Weight for audio-text contrastive loss (default: 0.3)")
    parser.add_argument('--audio_term_loss_ratio', type=float, default=0.7,
                       help="Weight for audio-term contrastive loss (default: 0.7)")

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
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # === 加载数据集 ===
    print(f"[INFO] Loading dataset from {args.train_samples_path}")
    print(f"[INFO] Using train ratio: {args.train_ratio:.1%} train, {1-args.train_ratio:.1%} test")
    train_dataset = InBatchDataset(args.train_samples_path, split="train", train_ratio=args.train_ratio)
    test_dataset = InBatchDataset(args.train_samples_path, split="test", train_ratio=args.train_ratio)
    
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
    print(f"[INFO] Training-only terms: {len(used_terms_train)}")
    print(f"[INFO] Test-only terms: {len(used_terms_test)}")
    
    # 分析train/test术语重叠
    train_set = set(used_terms_train)
    test_set = set(used_terms_test)
    overlap = train_set.intersection(test_set)
    print(f"[INFO] Terms overlap between train/test: {len(overlap)} terms")
    print(f"[INFO] Test terms that are unseen in training: {len(test_set - train_set)} terms")

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
    print(f"[INFO] Training with {len(train_dataset)} MFA chunk samples")
    print(f"[INFO] Unfrozen layers: {args.unfreeze_layers}")
    print(f"[INFO] Loss ratios - Audio-Text: {args.audio_text_loss_ratio:.1f}, Audio-Term: {args.audio_term_loss_ratio:.1f}")

    best_recall = 0.0
    no_improve_epochs = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        
        # 训练循环
        for batch in tqdm(train_dataloader, desc=f"[Epoch {epoch+1}/{args.epochs}]"):
            loss = train_step(model, batch, device, args)
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0.0
        print(f"[INFO] Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        # === 保存检查点 ===
        ckpt_path = f"data/clap_mfa_epoch{epoch+1}.pt"
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
                    top_ks=(5, 10, 20), max_eval=min(1000, len(test_dataset)),
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