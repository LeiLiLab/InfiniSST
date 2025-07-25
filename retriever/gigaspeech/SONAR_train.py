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
    def __init__(self, path="data/samples/xl/test_mfa_3chunks_samples_0_500000.json", split="train", train_ratio=0.999):
        print(f"[INFO] Loading MFA chunk samples from {path}")
        with open(path, "r") as f:
            all_samples = json.load(f)
        
        # 过滤有效样本：必须有音频文件、chunk文本和ground truth terms
        valid_samples = [
            s for s in all_samples
            if s.get('n_chunk_audio_ground_truth_terms')  # 必须有ground truth terms
               and s.get('n_chunk_text', '').strip()  # 必须有chunk文本
               and s.get('n_chunk_audio', '')  # 必须有音频路径
               and os.path.exists(s.get("n_chunk_audio", ""))  # 音频文件必须存在
               and any(  # 过滤掉过短或数字过多的术语
                   len(t) >= 3 and sum(c.isdigit() for c in t) <= len(t) // 2
                   for t in s["n_chunk_audio_ground_truth_terms"]
               )
        ]
        
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
    audio_emb = raw_model.encode_audio(audio_paths)  # [B, proj_dim]
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

    return contrastive_loss


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


def encode_texts_in_batches(model, texts, batch_size=512, device="cuda"):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        with torch.no_grad():
            emb = model.encode_text(batch).cpu()
            all_embeddings.append(emb)
    return torch.cat(all_embeddings, dim=0)


def evaluate_topk_recall(model, retriever, dataset, device, top_ks=(5, 10, 20), max_eval=1000, field="term"):
    """评估top-k召回率，使用n_chunk_audio_ground_truth_terms作为目标"""
    model.eval()
    recall_dict = {k: [] for k in top_ks}

    # === 重建索引 ===
    text_terms = [term['term'] for term in retriever.term_list]
    print(f'[DEBUG] text_terms: {len(text_terms)}')
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    text_emb = encode_texts_in_batches(raw_model, text_terms, device=device)

    retriever.index.reset()
    retriever.index.add(text_emb)

    print(f"len dataset: {len(dataset)}")
    import random
    eval_indices = random.sample(range(len(dataset)), min(max_eval, len(dataset)))
    valid_samples = []
    valid_indices = []
    
    for i in eval_indices:
        sample = dataset[i]
        if sample is not None and sample[3] and sample[0]:  # has_target=True and has ground_truth_terms
            valid_samples.append(sample)
            valid_indices.append(i)

    # 使用chunk音频进行编码
    audio_paths = [sample[1] for sample in valid_samples]  # n_chunk_audio paths
    with torch.no_grad():
        audio_embs = raw_model.encode_audio(audio_paths).cpu().numpy()

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
    print(f"[INFO] Building term vocabulary from training data...")
    used_terms = extract_all_used_terms(train_dataset)
    used_terms = list(set(t.lower() for t in used_terms))  # 去重并小写
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

    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Frozen parameters: {frozen:,} / {total:,} ({frozen / total:.2%})")
    print(f"[INFO] Trainable parameters: {trainable_params:,}")
    print(f"[INFO]   - Encoder parameters: {encoder_params_count:,} ({encoder_params_count/trainable_params:.1%})")
    print(f"[INFO]   - Projection parameters: {projection_params_count:,} ({projection_params_count/trainable_params:.1%})")
    print(f"[INFO] Training with {len(train_dataset)} MFA chunk samples")
    print(f"[INFO] Unfrozen layers: {args.unfreeze_layers}")

    best_recall = 0.0
    no_improve_epochs = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        
        # 训练循环
        for batch in tqdm(train_dataloader, desc=f"[Epoch {epoch+1}/{args.epochs}]"):
            loss = train_step(model, batch, device)
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
        recall_results = evaluate_topk_recall(
            model, retriever, test_dataset, device, 
            top_ks=(5, 10, 20), max_eval=min(1000, len(test_dataset))  # 最多评估1000个样本
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