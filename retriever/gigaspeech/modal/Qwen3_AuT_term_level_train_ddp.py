#!/usr/bin/env python3
"""
DDP training/evaluation with Qwen3-Omni AuT as drop-in speech encoder.
Reuse existing datasets and evaluation pipeline. Train only projection heads by default;
optionally enable lightweight LoRA on AuT q/k/v modules.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from Qwen3_AuT_speech_encoder import Qwen3AuTSpeechEncoder
from Qwen2_Audio_train import Qwen2AudioTextEncoder, SimpleRetriever, encode_texts_in_batches

# Reuse dataset and evaluation utilities from the simplified Qwen2 script
# NOTE: On Modal volume this file is uploaded as 'train_ddp_simplified.py'
from train_ddp_simplified import (
    TermLevelDatasetMMap,
    TermLevelDataset,
    evaluate_topk_recall,
    extract_all_used_terms,
)

try:
    from peft import LoraConfig, get_peft_model, TaskType
except Exception:
    LoraConfig = None
    get_peft_model = None
    TaskType = None


def setup_ddp(rank, world_size):
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29500')
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    if rank == 0:
        print(f"[INFO] Using device: cuda:{rank} ({torch.cuda.get_device_name(rank)})")


def cleanup_ddp():
    dist.destroy_process_group()


class ContrastiveAuTModel(nn.Module):
    def __init__(
        self,
        speech_encoder: Qwen3AuTSpeechEncoder,
        text_encoder: Qwen2AudioTextEncoder,
        proj_dim: int = 512,
        enable_speech_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.speech_encoder = speech_encoder
        self.text_encoder = text_encoder

        # Freeze base encoders
        if hasattr(self.speech_encoder, 'model') and isinstance(self.speech_encoder.model, nn.Module):
            for p in self.speech_encoder.model.parameters():
                p.requires_grad = False
        if hasattr(self.text_encoder, 'model') and isinstance(self.text_encoder.model, nn.Module):
            for p in self.text_encoder.model.parameters():
                p.requires_grad = False

        # Optional: LoRA on AuT q/k/v only
        if enable_speech_lora:
            if LoraConfig is None or get_peft_model is None:
                print("[WARN] peft not available; disabling speech LoRA")
            else:
                print("[INFO] Applying LoRA to AuT speech encoder (q/k/v only)")
                lora_cfg = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=max(0.0, lora_dropout),
                    target_modules=["q_proj", "k_proj", "v_proj"],
                    bias="none",
                )
                self.speech_encoder.model = get_peft_model(self.speech_encoder.model, lora_cfg)

        # Projection heads
        speech_hidden = self.speech_encoder.get_hidden_size()
        text_hidden = self.text_encoder.get_hidden_size()
        print(f"[INFO] AuT hidden size: {speech_hidden}; Text hidden size: {text_hidden}")
        self.proj_speech = nn.Linear(speech_hidden, proj_dim)
        self.proj_text = nn.Linear(text_hidden, proj_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep encoders in the requested mode
        if hasattr(self.speech_encoder, 'model'):
            self.speech_encoder.model.train(mode)
        if hasattr(self.text_encoder, 'model'):
            self.text_encoder.model.train(mode)
        return self

    def encode_audio(self, audio_inputs):
        if self.training:
            emb = self.speech_encoder.predict(audio_inputs)
        else:
            with torch.no_grad():
                emb = self.speech_encoder.predict(audio_inputs)
        if not isinstance(emb, torch.Tensor):
            emb = torch.as_tensor(emb)
        emb = emb.float().to(self.proj_speech.weight.device)
        if emb.dim() == 3:
            emb = emb.mean(dim=1)
        return F.normalize(self.proj_speech(emb), dim=-1)

    def encode_text(self, texts):
        if self.training:
            emb = self.text_encoder.predict(texts)
        else:
            with torch.no_grad():
                emb = self.text_encoder.predict(texts)
        if not isinstance(emb, torch.Tensor):
            emb = torch.as_tensor(emb)
        emb = emb.float().to(self.proj_text.weight.device)
        return F.normalize(self.proj_text(emb), dim=-1)


def collate_keep(batch):
    batch = [b for b in batch if b is not None]
    return batch


def train_step(model: ContrastiveAuTModel, batch, device, args, temperature=0.07):
    if len(batch) < 2:
        return None
    gt_terms_list, audio_items, chunk_texts = zip(*batch)
    gt_terms_list = [[t.lower() for t in terms if isinstance(t, str)] for terms in gt_terms_list]
    chunk_texts = [text.lower() if isinstance(text, str) else "" for text in chunk_texts]

    # Encode
    audio_emb = model.encode_audio(audio_items)
    text_emb = model.encode_text(chunk_texts) if args.audio_text_loss_ratio > 0 else torch.zeros_like(audio_emb)

    # Audio-Text contrastive loss
    contrastive = torch.tensor(0.0, device=device)
    if args.audio_text_loss_ratio > 0:
        sim = (audio_emb @ text_emb.T) / temperature
        labels = torch.arange(len(batch), device=device)
        loss_a2t = F.cross_entropy(sim, labels)
        loss_t2a = F.cross_entropy(sim.T, labels)
        contrastive = (loss_a2t + loss_t2a) * 0.5

    # Audio-Term loss (sample one positive term per audio)
    audio_term = torch.tensor(0.0, device=device)
    all_terms = []
    pairs = []
    for i, terms in enumerate(gt_terms_list):
        for term in terms:
            if term:
                idx = len(all_terms)
                all_terms.append(term)
                pairs.append((i, idx))
    if len(all_terms) > 0:
        with torch.no_grad():
            terms_emb = model.encode_text(all_terms).detach()
        sim = (audio_emb @ terms_emb.T) / temperature
        labels = []
        for i in range(len(batch)):
            pos = [ti for ai, ti in pairs if ai == i]
            if pos:
                labels.append(pos[0])
            else:
                labels.append(-1)
        valid_idx = [i for i, y in enumerate(labels) if y >= 0]
        if valid_idx:
            sim_v = sim[valid_idx]
            y_v = torch.tensor([labels[i] for i in valid_idx], device=device)
            audio_term = F.cross_entropy(sim_v, y_v)

    total = args.audio_text_loss_ratio * contrastive + args.audio_term_loss_ratio * audio_term
    if not torch.isfinite(total):
        return None
    return total


def train_ddp(rank, world_size, args):
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"[INFO] Starting AuT DDP training with {world_size} GPUs")

    # Encoders
    speech = Qwen3AuTSpeechEncoder(model_name=args.aut_model_name, device=device)
    text = Qwen2AudioTextEncoder(model_name=args.text_model_name, device=device)

    model = ContrastiveAuTModel(
        speech, text, proj_dim=512,
        enable_speech_lora=args.enable_speech_lora,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
    ).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # Dataset
    mmap_dir = getattr(args, 'mmap_shard_dir', None)
    if mmap_dir and os.path.exists(mmap_dir):
        if args.test_samples_path:
            train_set = TermLevelDatasetMMap(args.train_samples_path, mmap_dir, split="train", train_ratio=1.0)
            test_set = TermLevelDatasetMMap(None, mmap_dir, split="test", test_path=args.test_samples_path)
        else:
            train_set = TermLevelDatasetMMap(args.train_samples_path, mmap_dir, split="train", train_ratio=args.train_ratio)
            test_set = TermLevelDatasetMMap(args.train_samples_path, mmap_dir, split="test", train_ratio=args.train_ratio)
    else:
        if args.test_samples_path:
            train_set = TermLevelDataset(args.train_samples_path, split="train", train_ratio=1.0)
            test_set = TermLevelDataset(None, split="test", test_path=args.test_samples_path)
        else:
            train_set = TermLevelDataset(args.train_samples_path, split="train", train_ratio=args.train_ratio)
            test_set = TermLevelDataset(args.train_samples_path, split="test", train_ratio=args.train_ratio)

    sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size // world_size,
        sampler=sampler,
        collate_fn=collate_keep,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )

    # Retriever (rank 0)
    train_terms = extract_all_used_terms(train_set)
    test_terms = extract_all_used_terms(test_set)
    train_terms_set = set(t.lower() for t in train_terms)
    test_terms_set = set(t.lower() for t in test_terms)
    if rank == 0:
        all_terms = list(train_terms_set | test_terms_set)
        retriever = SimpleRetriever(enable_fusion=True, device=device)
        retriever.model = model.module
        import faiss
        retriever.index = faiss.IndexFlatL2(512)
        retriever.term_list = [{"term": t} for t in all_terms]
        print(f"[INFO] Retriever vocabulary: {len(all_terms)} terms")
    else:
        retriever = None

    # Optimizer
    trainable_params = [p for p in model.module.parameters() if p.requires_grad]
    if rank == 0:
        print(f"[INFO] Optimizer will update {len(trainable_params)} parameter tensors")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()

    best_recall = 0.0
    global_step = 0
    for epoch in range(args.epochs):
        if rank == 0:
            print(f"\n[INFO] Epoch {epoch+1}/{args.epochs}")
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        valid_batches = 0

        it = loader if rank != 0 else __import__('tqdm').tqdm(loader, desc=f"Epoch {epoch+1}")
        for batch in it:
            if not batch:
                optimizer.zero_grad(set_to_none=True)
                continue
            with torch.cuda.amp.autocast():
                loss = train_step(model, batch, device, args)
            if loss is None or not loss.requires_grad or not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach().item())
            valid_batches += 1
            global_step += 1

        # Epoch summary
        avg_loss = total_loss / max(1, valid_batches)
        loss_t = torch.tensor(avg_loss, device=device)
        dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
        avg_loss = loss_t.item() / world_size
        if rank == 0:
            print(f"[INFO] Epoch {epoch+1} avg loss: {avg_loss:.4f}")

            # Evaluate
            results = evaluate_topk_recall(
                model, retriever, test_set, device,
                top_ks=(5, 10), max_eval=min(1000, len(test_set)),
                train_terms_set=train_terms_set,
            )
            overall = results.get('overall', results)
            recall10 = sum(overall[10]) / len(overall[10]) if overall.get(10) else 0.0
            if recall10 > best_recall:
                best_recall = recall10
                torch.save(model.state_dict(), args.save_path.replace('.pt', '_best.pt'))
                print(f"[INFO] New best model saved (Recall@10: {best_recall:.2%})")

    if rank == 0:
        torch.save(model.state_dict(), args.save_path)
        print(f"[INFO] Training completed. Best Recall@10: {best_recall:.2%}")
    cleanup_ddp()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--train_samples_path', type=str, required=True)
    p.add_argument('--test_samples_path', type=str, default=None)
    p.add_argument('--train_ratio', type=float, default=0.998)
    p.add_argument('--save_path', type=str, default="qwen3_aut_term_level.pt")
    p.add_argument('--audio_text_loss_ratio', type=float, default=0.3)
    p.add_argument('--audio_term_loss_ratio', type=float, default=0.7)
    p.add_argument('--mmap_shard_dir', type=str, default=None)
    p.add_argument('--aut_model_name', type=str, default="Qwen/Qwen3-Omni-Audio")
    p.add_argument('--text_model_name', type=str, default="Qwen/Qwen2-Audio-7B-Instruct")
    p.add_argument('--enable_speech_lora', action='store_true')
    p.add_argument('--lora_r', type=int, default=8)
    p.add_argument('--lora_alpha', type=int, default=16)
    p.add_argument('--lora_dropout', type=float, default=0.0)
    args = p.parse_args()

    world_size = int(os.environ.get('WORLD_SIZE', torch.cuda.device_count()))
    if world_size == 0:
        print("[ERROR] No CUDA devices available!")
        return 1

    if 'LOCAL_RANK' in os.environ:
        rank = int(os.environ['LOCAL_RANK'])
        train_ddp(rank, world_size, args)
    else:
        mp.set_start_method('spawn', force=True)
        mp.spawn(train_ddp, args=(world_size, args), nprocs=world_size, join=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())



