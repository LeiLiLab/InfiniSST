import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import laion_clap
import numpy as np
import json
from tqdm import tqdm
from new_giga_speech import handle_giga_speech_train_samples

class InBatchDataset(Dataset):
    def __init__(self):
        self.samples = handle_giga_speech_train_samples()

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_tensor = sample.get('audio_tensor', None)
        term_list = sample.get('ground_truth_term', None)
        if audio_tensor is None or audio_tensor.numel() == 0 or not term_list:
            raise ValueError(f"Invalid item at index {idx}")
        return term_list, audio_tensor

    def __len__(self):
        return len(self.samples)

def train_step(model, batch, device, temperature=0.07):
    batch = [item for item in batch if item != (None, None)]
    if len(batch) < 2:
        print("Batch has less than 2 non-None items, skipping...")
        return torch.tensor(0.0, requires_grad=True).to(device)

    # 拆分成 term 列表和音频
    term_lists, audios = zip(*batch)

    min_len = 48000  # 1 second
    filtered = [
        (terms, a) for terms, a in zip(term_lists, audios)
        if isinstance(terms, list) and len(terms) > 0 and a is not None and a.shape[-1] >= min_len
    ]
    # print(f"[DEBUG] Filtered batch size: {len(filtered)}")
    if len(filtered) < 2:
        print("Filtered batch too small, skipping.")
        return torch.tensor(0.0, requires_grad=True).to(device)

    term_lists, audios = zip(*filtered)

    # 展平所有 terms 并记录每个 audio 对应的 positive 索引
    all_terms = []
    pos_mask = []
    offset = 0
    for terms in term_lists:
        indices = []
        for t in terms:
            if isinstance(t, str) and t.strip():
                all_terms.append(t)
                indices.append(offset)
                offset += 1
        pos_mask.append(indices)

    # 检查是否有 audio 没有任何 valid term（应避免除0）
    if any(len(p) == 0 for p in pos_mask):
        print("Some audio has no valid terms after filtering, skipping.")
        return torch.tensor(0.0, requires_grad=True).to(device)

    # text/audio 编码
    text_data = [t for t in all_terms]
    # print(f"[DEBUG] Audio tensor shapes: {[a.shape for a in audios]}")

    audio_list = [a.squeeze().cpu() for a in audios]
    valid_audio_list = []
    for i, audio in enumerate(audio_list):
        if not isinstance(audio, torch.Tensor):
            print(f"[SKIP] audio {i} is not a tensor")
            continue
        if audio.ndim != 1:
            print(f"[SKIP] audio {i} has invalid ndim: {audio.ndim}")
            continue
        if not torch.isfinite(audio).all():
            print(f"[SKIP] audio {i} has NaN or Inf values")
            continue
        if audio.shape[0] < 16000:
            print(f"[SKIP] audio {i} too short: {audio.shape[0]} samples")
            continue
        valid_audio_list.append(audio)

    if len(valid_audio_list) < 2:
        print(f"[ERROR] Not enough valid audio inputs after filtering, skipping batch.")
        return torch.tensor(0.0, requires_grad=True).to(device)

    # Pad all valid audio tensors to the same length
    max_len = max([a.shape[0] for a in valid_audio_list])
    padded_audio = torch.stack([F.pad(a, (0, max_len - a.shape[0])) for a in valid_audio_list]).to(device)

    with torch.no_grad():
        test_emb = model.get_audio_embedding_from_data(x=padded_audio, use_tensor=True)
        test_emb = F.normalize(test_emb, dim=-1)
        # print(f"[DEBUG] Output audio embedding shape: {test_emb.shape}")

    text_emb = model.get_text_embedding(text_data, use_tensor=True)
    text_emb = text_emb.to(device)
    audio_emb = test_emb

    sim = (audio_emb @ text_emb.T) / temperature  # (num_audio, num_terms)

    # 构造 multi-positive mask
    pos_mask_tensor = torch.zeros_like(sim)
    for i, pos_indices in enumerate(pos_mask):
        for j in pos_indices:
            pos_mask_tensor[i, j] = 1.0

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    loss = - (log_prob * pos_mask_tensor).sum(dim=1) / pos_mask_tensor.sum(dim=1).clamp(min=1)
    return loss.mean()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argume nt('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--save_path', type=str, default="data/clap_inbatch.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()

    dataset = InBatchDataset()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: x, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            loss = train_step(model, batch, device)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1} loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

if __name__ == "__main__":
    main()