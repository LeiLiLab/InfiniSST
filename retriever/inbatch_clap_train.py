import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
from tqdm import tqdm
from new_giga_speech import load_preprocessed_samples
import laion_clap
import torch
import torch.nn.functional as F
import argparse, os, sys
from torch.optim.lr_scheduler import ReduceLROnPlateau
import faiss
from new_retrieve import Retriever


class InBatchDataset(Dataset):
    def __init__(self,path = "data/preprocessed_samples_merged.json"):
        self.samples = load_preprocessed_samples(path)
        self.samples = [
            s for s in self.samples
            if s.get("has_target", True)
               and any (len(t)>=5 for t in s["ground_truth_term"])
               and isinstance(s.get("audio_tensor"), torch.Tensor)
               and s["audio_tensor"].numel() > 0
               and 48000 <= s["audio_tensor"].shape[-1] <= 480000
        ]


    # def __getitem__(self, idx):
    #     sample = self.samples[idx]
    #     audio_tensor = sample.get('audio_tensor', None)
    #     term_list = sample.get('ground_truth_term', None)
    #     if audio_tensor is None or audio_tensor.numel() == 0:
    #         return None
    #     return term_list, audio_tensor, sample.get("has_target", True)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_tensor = sample.get('audio_tensor', None)
        term_list = sample.get('ground_truth_term', None)
        return term_list, audio_tensor, True  # 此时已确认 has_target == True

    def __len__(self):
        return len(self.samples)


def freeze_clap_layers(model, freeze_text_encoder=True, freeze_audio_encoder_until=None):
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    clap = raw_model.model  # ✅ 获取 CLAP 主体

    if freeze_text_encoder:
        for name, param in clap.text_branch.named_parameters():
            param.requires_grad = False

    if freeze_audio_encoder_until is not None:
        for name, param in clap.audio_branch.named_parameters():
            if "layers." in name:
                layer_num = int(name.split("layers.")[1].split(".")[0])
                param.requires_grad = layer_num >= freeze_audio_encoder_until
            else:
                param.requires_grad = True  # 非 encoder 层不冻结

    # 不冻结 audio/text projection
    for param in clap.audio_projection.parameters():
        param.requires_grad = True
    for param in clap.text_projection.parameters():
        param.requires_grad = True

def train_step(model, batch, device, temperature=0.07):
    if len(batch) < 2:
        print("Batch has less than 2 non-None items, skipping...")
        return torch.tensor(0.0, requires_grad=True).to(device)

    # 拆分成 term 列表和音频
    term_lists, audios, has_targets = zip(*batch)
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
    audio_list = [a.squeeze().cpu() for a in audios]
    max_len = max([a.shape[0] for a in audio_list])
    padded_audio = torch.stack([F.pad(a, (0, max_len - a.shape[0])) for a in audio_list]).to(device)

    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    audio_emb = raw_model.get_audio_embedding_from_data(x=padded_audio, use_tensor=True)
    audio_emb = F.normalize(audio_emb, dim=-1)

    # === 编码 text ===
    text_emb = raw_model.get_text_embedding(all_terms, use_tensor=True)
    text_emb = text_emb.to(device)
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

    # 加权策略（可选）
    #positive_weight = 1.0 + 4.0 * (1.0 - has_target_tensor.float())  # has_target=False => weight higher
    #loss = (loss * positive_weight).mean()
    #num_pos_terms = int(pos_mask_tensor.sum().item())
    #return loss.mean(), num_pos_terms

def safe_collate(batch):
    return [
        item for item in batch
        if item is not None
        and isinstance(item, tuple)
        and len(item) == 3
        and isinstance(item[1], torch.Tensor)
        and item[1].numel() > 0
    ]
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
        with open(glossary_path, "r", encoding="utf-8") as f:
            glossary = json.load(f)

        def get_term_text(term):
            entry = glossary.get(term)
            if not entry:
                return term
            desc = entry.get("short_description", "")
            desc_first = desc.split(",", 1)[0].strip()
            return f"{term}, {desc_first}" if desc_first and desc_first.lower() != term.lower() else term

        texts = [get_term_text(t) for t in term_subset]
    else:
        texts = term_subset

    with torch.no_grad():
        text_emb = model.get_text_embedding(texts, use_tensor=True)
        text_emb = F.normalize(text_emb, dim=-1).cpu().numpy()

    retriever.term_list = [{'term': t} for t in term_subset]
    retriever.index.reset()
    retriever.index.add(text_emb)

def encode_texts_in_batches(model, texts, batch_size=512, device="cuda"):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        with torch.no_grad():
            emb = model.get_text_embedding(batch, use_tensor=True).to(device)
            emb = F.normalize(emb, dim=-1).cpu()
            all_embeddings.append(emb)
    return torch.cat(all_embeddings, dim=0)

def evaluate_topk_recall(model, retriever, dataset, device, top_ks=(5,10, 50), max_eval=1000, field="term"):
    model.eval()
    recall_dict = {k: [] for k in top_ks}

    # === 重建索引 ===
    text_terms = [term['term'] for term in retriever.term_list]
    print(f'[DEBUG] text_terms: {len(text_terms)}')
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    text_emb = encode_texts_in_batches(raw_model, text_terms, device=device)

    retriever.index.reset()
    retriever.index.add(text_emb)

    import random
    eval_indices = random.sample(range(len(dataset)), min(max_eval, len(dataset)))
    for i in eval_indices:
        sample = dataset[i]
        if sample is None:
            continue
        term_list, audio_tensor, has_target = sample
        if not has_target or not term_list:
            continue

        audio = audio_tensor.squeeze().unsqueeze(0).to(device)
        with torch.no_grad():
            audio_emb = raw_model.get_audio_embedding_from_data(x=audio, use_tensor=True)
            audio_emb = F.normalize(audio_emb, dim=-1).cpu().numpy()

        for top_k in top_ks:
            D, I = retriever.index.search(audio_emb, top_k)
            retrieved_terms = [retriever.term_list[idx][field].lower() for idx in I[0]]
            gt_terms = [t.lower() for t in term_list]

            matched = sum(gt in retrieved_terms for gt in gt_terms)
            recall = matched / len(gt_terms)
            recall_dict[top_k].append(recall)

            if i < 3 and top_k == top_ks[0]:  # 只打印一次
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
        '--text_field', type=str, default="term", choices=["term", "short_description"],
        help="Which field to use as input text (term: comma-split title, short_description: full description)"
    )


    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # === 模型初始化 ===
    model = laion_clap.CLAP_Module(enable_fusion=True)
    model.load_ckpt()
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    dataset = InBatchDataset(args.samples_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=safe_collate)
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
    #先不冻结
    freeze_clap_layers(raw_model, freeze_text_encoder=True, freeze_audio_encoder_until=10)
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

        for batch in tqdm(dataloader, desc=f"[Epoch {epoch}]"):
            loss = train_step(model, batch, device)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[INFO] Epoch {epoch} avg loss: {avg_loss:.4f}")

        # === 保存模型 ===
        ckpt_path = f"data/clap_epoch{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"[INFO] Model saved to {ckpt_path}")

        # 全小写
        rebuild_index_from_terms(
            retriever.model,
            retriever,
            used_terms,
            device,
            glossary_path=args.glossary_path,
            text_field=args.text_field
        )

        recall_results = evaluate_topk_recall(model, retriever, dataset, device, top_ks=(5, 10, 50), max_eval=1000)
        recall = recall_results[50] and sum(recall_results[50]) / len(recall_results[50])  # 用 Recall@50 继续控制 Early Stop
        scheduler.step(recall)
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