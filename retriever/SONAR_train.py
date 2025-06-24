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
    def __init__(self, speech_encoder, text_encoder, hidden_dim=1024, proj_dim=512):
        super().__init__()
        self.speech_encoder = speech_encoder
        self.text_encoder = text_encoder

        # projection layers
        self.proj_speech = nn.Linear(hidden_dim, proj_dim)
        self.proj_text = nn.Linear(hidden_dim, proj_dim)

        # freeze encoder weights
        for param in self.speech_encoder.model.parameters():
            param.requires_grad = False
        for param in self.text_encoder.model.parameters():
            param.requires_grad = False

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
    def __init__(self, path="data/preprocessed_samples_merged.json"):
        print(path)
        with open(path, "r") as f:
            self.samples = json.load(f)
        self.samples = [
            s for s in self.samples
            if s.get('ground_truth_term')
               and any(
                   len(t) >= 5 and sum(c.isdigit() for c in t) <= 4
                   for t in s["ground_truth_term"]
               )
               and os.path.exists(s.get("audio", ""))
        ]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_path = sample["audio"]
        term_list = sample.get('ground_truth_term', None)
        return term_list, audio_path, True  # 此时已确认 has_target == True

    def __len__(self):
        return len(self.samples)


def train_step(model, batch, device, temperature=0.07):
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model

    if len(batch) < 2:
        print("Batch has less than 2 non-None items, skipping...")
        return torch.tensor(0.0, requires_grad=True).to(device)

    # 拆分成 term 列表和音频
    term_lists, audio_paths, has_targets = zip(*batch)
    # 全小写
    term_lists = [[t.lower() for t in terms if isinstance(t, str)] for terms in term_lists]

    # === 去重术语构建 ===
    term2index = dict()
    all_terms = []
    for terms in term_lists:
        for t in terms:
            if isinstance(t, str) and t.strip() and t not in term2index:
                term2index[t] = len(all_terms)
                all_terms.append(t)

    if not all_terms:
        print("No valid terms in batch, skipping...")
        return torch.tensor(0.0, requires_grad=True).to(device)

    # 每个 audio 对应的 positive term 索引
    pos_mask = []
    for terms in term_lists:
        indices = []
        for t in terms:
            if t in term2index:
                indices.append(term2index[t])
        pos_mask.append(indices)
    # === 处理 audio ===
    audio_emb = raw_model.encode_audio(audio_paths)
    # === 编码 text, 一个audio对应多个terms，去重后得到all_terms再embeddings ===
    text_emb = raw_model.encode_text(all_terms)
    # === 计算相似度 ===
    sim = (audio_emb @ text_emb.T) / temperature  # shape: [B, T]

    # === 构造 multi-positive mask ===
    pos_mask_tensor = torch.zeros_like(sim)
    for i, indices in enumerate(pos_mask):
        for j in indices:
            pos_mask_tensor[i, j] = 1.0

    if pos_mask_tensor.sum() == 0:
        print("All audios have no valid term matches, skipping...")
        return torch.tensor(0.0, requires_grad=True).to(device)

    # === 计算 InfoNCE loss ===
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    loss = - (log_prob * pos_mask_tensor).sum(dim=1) / pos_mask_tensor.sum(dim=1).clamp(min=1)

    return loss.mean()

def extract_all_used_terms(dataset):
    used_terms = set()
    for sample in dataset:
        if sample is None:
            continue
        term_list, audio_tensor, has_target = sample
        if has_target and term_list:
            used_terms.update(t.lower() for t in term_list if isinstance(t, str))
    return list(used_terms)


def rebuild_index_from_terms(model, retriever, term_subset, device, glossary_path=None, text_field="term"):
    model.eval()

    # === term -> text embedding 输入 ===
    if text_field=="short_description" and glossary_path and os.path.exists(glossary_path):
        # with open(glossary_path, "r", encoding="utf-8") as f:
        #     glossary = json.load(f)

        def get_term_text(term):
            return term.lower()
            # entry = glossary.get(term)
            # if not entry:
            #     return term
            # desc = entry.get("short_description", "")
            # desc_first = desc.split(",", 1)[1].strip()
            # res =  f"{term}, {desc_first}" if desc_first and desc_first.lower() != term.lower() else term
            # return res

        texts = [get_term_text(t) for t in term_subset]
        print(f"text samples:{texts[:5]}")
    else:
        texts = term_subset

    with torch.no_grad():
        text_emb = model.encode_text(texts).cpu().numpy()

    retriever.term_list = [{'term': t} for t in term_subset]
    retriever.index.reset()
    retriever.index.add(text_emb)

def encode_texts_in_batches(model, texts, batch_size=512, device="cuda"):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        with torch.no_grad():
            emb = model.encode_text(batch).cpu()
            all_embeddings.append(emb)
    return torch.cat(all_embeddings, dim=0)

def evaluate_topk_recall(model, retriever, dataset, device, top_ks=(5,10, 20), max_eval=1000, field="term"):
    model.eval()
    recall_dict = {k: [] for k in top_ks}

    # === 重建索引 ===
    text_terms = [term['term'] for term in retriever.term_list]
    print(f'[DEBUG] text_terms: {len(text_terms)}')
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    text_emb = encode_texts_in_batches(raw_model, text_terms, device=device)

    retriever.index.reset()
    retriever.index.add(text_emb)

    print(f"len dataset:{len(dataset)}")
    import random
    eval_indices = random.sample(range(len(dataset)), min(max_eval, len(dataset)))
    valid_samples = []
    valid_indices = []
    for i in eval_indices:
        sample = dataset[i]
        if sample is not None and sample[2] and sample[0]:
            valid_samples.append(sample)
            valid_indices.append(i)

    audio_paths = [sample[1] for sample in valid_samples]
    with torch.no_grad():
        audio_embs = raw_model.encode_audio(audio_paths).cpu().numpy()

    for j, (i, sample) in enumerate(zip(valid_indices, valid_samples)):
        term_list, audio_path, has_target = sample
        audio_emb = audio_embs[j:j+1]  # shape: [1, 512]

        for top_k in top_ks:
            D, I = retriever.index.search(audio_emb, top_k)
            retrieved_terms = [retriever.term_list[idx][field].lower() for idx in I[0]]
            gt_terms = [t.lower() for t in term_list]

            matched = sum(gt in retrieved_terms for gt in gt_terms)
            recall = matched / len(gt_terms)
            recall_dict[top_k].append(recall)

            if j < 3 and top_k == top_ks[0]:  # 只打印一次
                print(f"[DEBUG] GT terms: {gt_terms}")
                print(f"[DEBUG] Retrieved terms: {retrieved_terms}")
                print(f"[DEBUG] Match count: {matched}")

    # 打印统计结果
    for top_k in top_ks:
        avg_recall = sum(recall_dict[top_k]) / len(recall_dict[top_k]) if recall_dict[top_k] else 0.0
        print(f"[EVAL] Average Recall@{top_k}: {avg_recall:.2%}")

    model.train()
    return recall_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--samples_path', type=str, default="data/preprocessed_samples_merged.json")
    parser.add_argument('--glossary_path', type=str, default="data/terms/glossary_filtered.json")
    parser.add_argument('--save_path', type=str, default="data/clap_inbatch.pt")
    parser.add_argument(
        '--text_field', type=str, default="short_description", choices=["term", "short_description"],
        help="Which field to use as input text (term: comma-split title, short_description: full description)"
    )


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

    model = ContrastiveSpeechTextModel(speech_encoder, text_encoder).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    dataset = InBatchDataset(args.samples_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: x)
    # === 评估 recall（构建子集索引）===
    print(f"[INFO] Rebuilding FAISS index from used terms...")
    used_terms = extract_all_used_terms(dataset)
    used_terms = [t.lower() for t in used_terms]


    # === 初始化 retriever，用于每轮动态构建索引 + recall 评估 ===
    retriever = Retriever(enable_fusion=True, device=device)
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    retriever.model = raw_model
    # 初始化空 index（避免后续 reset() 报错）
    retriever.index = faiss.IndexFlatL2(512)

    # 测试下是否冻结成功
    print("[DEBUG] Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f" - {name}: {param.shape}")

    # 打印被冻结的参数数量
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Frozen parameters: {frozen} / {total} ({frozen / total:.2%})")
    print(f"[INFO] Training with {len(dataset)} samples")

    best_recall = 0.0
    no_improve_epochs = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        # 在每轮训练开始前记录一组参数值
        print("[DEBUG] proj_speech.weight norm (before):", raw_model.proj_speech.weight.norm().item())

        for batch in tqdm(dataloader, desc=f"[Epoch {epoch}]"):
            loss = train_step(model, batch, device)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        print("[DEBUG] proj_speech.weight norm (after):", raw_model.proj_speech.weight.norm().item())

        avg_loss = total_loss / len(dataloader)
        print(f"[INFO] Epoch {epoch} avg loss: {avg_loss:.4f}")

        # === 保存模型 ===
        ckpt_path = f"data/clap_epoch{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"[INFO] Model saved to {ckpt_path}")

        # 全小写
        # rebuild_index_from_terms(
        #     retriever.model,
        #     retriever,
        #     used_terms,
        #     device,
        #     glossary_path=args.glossary_path,
        #     text_field=args.text_field
        # )
        retriever.term_list = [{'term': t} for t in used_terms]


        # 使用测试集评估 recall
        if args.text_field == "term":
            test_dataset = InBatchDataset(f"data/{args.text_field}_test_preprocessed_samples_merged.json")
        else:
            test_dataset = InBatchDataset(f"data/test_preprocessed_samples_merged.json")
        recall_results = evaluate_topk_recall(model, retriever, test_dataset, device, top_ks=(5, 10, 20), max_eval=1000)
        recall = recall_results[10] and sum(recall_results[10]) / len(recall_results[10])  # 用 Recall@10 继续控制 Early Stop
        if recall > best_recall:
            best_recall = recall
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= args.patience:
                print(f"[EARLY STOPPING] No improvement in {args.patience} evals. Best Recall@10: {best_recall:.2%}")
                break

    # === 最终保存 ===
    torch.save(model.state_dict(), args.save_path)
    print(f"[INFO] Final model saved to {args.save_path}")

if __name__ == "__main__":
    main()