#!/usr/bin/env python3
"""
Single-process training with Qwen3-Omni AuT sharded across multiple GPUs.
Uses device_map="auto" for tensor parallelism instead of DDP.
Optimized for 4x A6000 (48GB each) to fit 30B model.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from Qwen3_AuT_speech_encoder import Qwen3AuTSpeechEncoder
from Qwen2_Audio_train import Qwen2AudioTextEncoder, SimpleRetriever

# Reuse dataset and evaluation utilities
from train_ddp_simplified import (
    TermLevelDatasetMMap,
    TermLevelDataset,
    evaluate_topk_recall,
    extract_all_used_terms,
)

try:
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
except Exception:
    LoraConfig = None
    get_peft_model = None
    TaskType = None
    prepare_model_for_kbit_training = None


class ContrastiveAuTModelSharded(nn.Module):
    """
    Contrastive model with AuT sharded across GPUs.
    Only speech encoder gradients are enabled; text encoder frozen.
    """
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

        # Freeze text encoder completely (no gradients)
        if hasattr(self.text_encoder, 'model') and isinstance(self.text_encoder.model, nn.Module):
            for p in self.text_encoder.model.parameters():
                p.requires_grad = False
            self.text_encoder.model.eval()  # Keep in eval mode
            print("[INFO] Text encoder frozen (no gradients)")

        # Speech encoder: freeze base, optionally add LoRA
        if hasattr(self.speech_encoder, 'model') and isinstance(self.speech_encoder.model, nn.Module):
            # Freeze all parameters first
            for p in self.speech_encoder.model.parameters():
                p.requires_grad = False
            
            # Optional: LoRA on AuT q/k/v only
            if enable_speech_lora:
                if LoraConfig is None or get_peft_model is None:
                    print("[WARN] peft not available; disabling speech LoRA")
                else:
                    print("[INFO] Applying LoRA to AuT speech encoder (q/k/v only)")
                    
                    # Prepare for quantized training if needed
                    if prepare_model_for_kbit_training is not None:
                        try:
                            self.speech_encoder.model = prepare_model_for_kbit_training(
                                self.speech_encoder.model,
                                use_gradient_checkpointing=True
                            )
                            print("[INFO] Prepared model for k-bit training")
                        except Exception as e:
                            print(f"[WARN] prepare_model_for_kbit_training failed: {e}")
                    
                    lora_cfg = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        r=lora_r,
                        lora_alpha=lora_alpha,
                        lora_dropout=max(0.0, lora_dropout),
                        target_modules=["q_proj", "k_proj", "v_proj"],
                        bias="none",
                    )
                    self.speech_encoder.model = get_peft_model(self.speech_encoder.model, lora_cfg)
                    self.speech_encoder.model.print_trainable_parameters()

        # Projection heads (trainable)
        speech_hidden = self.speech_encoder.get_hidden_size()
        text_hidden = self.text_encoder.get_hidden_size()
        print(f"[INFO] AuT hidden size: {speech_hidden}; Text hidden size: {text_hidden}")
        
        # Place projection heads on first GPU for simplicity
        self.proj_device = torch.device("cuda:0")
        self.proj_speech = nn.Linear(speech_hidden, proj_dim).to(self.proj_device)
        self.proj_text = nn.Linear(text_hidden, proj_dim).to(self.proj_device)

    def encode_audio(self, audio_inputs):
        # Speech encoder may have gradients (if LoRA enabled)
        emb = self.speech_encoder.predict(audio_inputs)
        if not isinstance(emb, torch.Tensor):
            emb = torch.as_tensor(emb)
        emb = emb.float().to(self.proj_device)
        if emb.dim() == 3:
            emb = emb.mean(dim=1)
        return F.normalize(self.proj_speech(emb), dim=-1)

    def encode_text(self, texts):
        # Text encoder always no_grad
        with torch.no_grad():
            emb = self.text_encoder.predict(texts)
        if not isinstance(emb, torch.Tensor):
            emb = torch.as_tensor(emb)
        emb = emb.float().to(self.proj_device)
        return F.normalize(self.proj_text(emb), dim=-1)


def collate_keep(batch):
    batch = [b for b in batch if b is not None]
    return batch


def train_step(model: ContrastiveAuTModelSharded, batch, args, temperature=0.07):
    if len(batch) < 2:
        return None
    gt_terms_list, audio_items, chunk_texts = zip(*batch)
    gt_terms_list = [[t.lower() for t in terms if isinstance(t, str)] for terms in gt_terms_list]
    chunk_texts = [text.lower() if isinstance(text, str) else "" for text in chunk_texts]

    # Encode (text is always no_grad internally)
    audio_emb = model.encode_audio(audio_items)
    text_emb = model.encode_text(chunk_texts) if args.audio_text_loss_ratio > 0 else torch.zeros_like(audio_emb)

    # Audio-Text contrastive loss
    device = audio_emb.device
    contrastive = torch.tensor(0.0, device=device)
    if args.audio_text_loss_ratio > 0:
        sim = (audio_emb @ text_emb.T) / temperature
        labels = torch.arange(len(batch), device=device)
        loss_a2t = F.cross_entropy(sim, labels)
        loss_t2a = F.cross_entropy(sim.T, labels)
        contrastive = (loss_a2t + loss_t2a) * 0.5

    # Audio-Term loss
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


def train_sharded(args):
    """Single-process training with model sharded across GPUs"""
    
    print(f"[INFO] Starting sharded training on {torch.cuda.device_count()} GPUs")
    print(f"[INFO] Available devices: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    
    # Set environment variables for sharded loading
    os.environ["AUT_DEVICE_MAP"] = "auto"  # Enable auto sharding
    os.environ["AUT_LOAD_IN_4BIT"] = "1"   # Use 4-bit quantization
    os.environ["AUT_MAX_MEMORY"] = "46GiB" # Per-GPU memory limit
    os.environ["AUT_NO_FLASH_ATTENTION"] = "1"  # Disable FA2 to save memory
    os.environ["AUT_DTYPE"] = "bfloat16"   # Use bfloat16
    
    # Optionally set offload folder for CPU/disk spill
    if hasattr(args, 'offload_folder') and args.offload_folder:
        os.environ["AUT_OFFLOAD_FOLDER"] = args.offload_folder
        os.makedirs(args.offload_folder, exist_ok=True)
        os.environ["AUT_MAX_CPU_MEMORY"] = "200GiB"
        print(f"[INFO] Offload folder: {args.offload_folder}")
    
    # Load encoders
    print("\n[INFO] Loading speech encoder (will be sharded)...")
    speech = Qwen3AuTSpeechEncoder(model_name=args.aut_model_name, device="cuda:0")
    
    print("\n[INFO] Loading text encoder (single GPU)...")
    text = Qwen2AudioTextEncoder(model_name=args.text_model_name, device="cuda:0")

    # Build model
    model = ContrastiveAuTModelSharded(
        speech, text, proj_dim=512,
        enable_speech_lora=args.enable_speech_lora,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
    )

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

    # DataLoader (no distributed sampler needed)
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_keep,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )

    # Retriever for evaluation
    train_terms = extract_all_used_terms(train_set)
    test_terms = extract_all_used_terms(test_set)
    train_terms_set = set(t.lower() for t in train_terms)
    test_terms_set = set(t.lower() for t in test_terms)
    all_terms = list(train_terms_set | test_terms_set)
    
    retriever = SimpleRetriever(enable_fusion=True, device=model.proj_device)
    retriever.model = model
    import faiss
    retriever.index = faiss.IndexFlatL2(512)
    retriever.term_list = [{"term": t} for t in all_terms]
    print(f"[INFO] Retriever vocabulary: {len(all_terms)} terms")

    # Optimizer (only trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[INFO] Trainable parameters: {len(trainable_params)} tensors")
    print(f"[INFO] Total trainable params: {sum(p.numel() for p in trainable_params):,}")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    
    # Gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    best_recall = 0.0
    global_step = 0
    
    for epoch in range(args.epochs):
        print(f"\n{'='*80}")
        print(f"[INFO] Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*80}")
        
        model.train()
        # Keep text encoder in eval
        if hasattr(model.text_encoder, 'model'):
            model.text_encoder.model.eval()
        
        total_loss = 0.0
        valid_batches = 0
        
        progress = tqdm(loader, desc=f"Epoch {epoch+1}")
        for batch_idx, batch in enumerate(progress):
            if not batch:
                continue
            
            # Gradient accumulation
            is_accumulating = (batch_idx + 1) % args.gradient_accumulation_steps != 0
            
            with torch.cuda.amp.autocast():
                loss = train_step(model, batch, args)
            
            if loss is None or not torch.isfinite(loss):
                continue
            
            # Scale loss by accumulation steps
            loss = loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            
            if not is_accumulating:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                global_step += 1
            
            total_loss += float(loss.detach().item()) * args.gradient_accumulation_steps
            valid_batches += 1
            
            # Update progress bar
            if valid_batches > 0:
                progress.set_postfix({"loss": f"{total_loss/valid_batches:.4f}"})

        # Epoch summary
        avg_loss = total_loss / max(1, valid_batches)
        print(f"[INFO] Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        # Evaluate
        print("[INFO] Evaluating...")
        results = evaluate_topk_recall(
            model, retriever, test_set, model.proj_device,
            top_ks=(5, 10), max_eval=min(1000, len(test_set)),
            train_terms_set=train_terms_set,
        )
        overall = results.get('overall', results)
        recall5 = sum(overall[5]) / len(overall[5]) if overall.get(5) else 0.0
        recall10 = sum(overall[10]) / len(overall[10]) if overall.get(10) else 0.0
        
        print(f"[INFO] Recall@5: {recall5:.2%}, Recall@10: {recall10:.2%}")
        
        if recall10 > best_recall:
            best_recall = recall10
            save_path_best = args.save_path.replace('.pt', '_best.pt')
            torch.save(model.state_dict(), save_path_best)
            print(f"[INFO] New best model saved (Recall@10: {best_recall:.2%}) -> {save_path_best}")

    # Save final model
    torch.save(model.state_dict(), args.save_path)
    print(f"\n[INFO] Training completed!")
    print(f"[INFO] Best Recall@10: {best_recall:.2%}")
    print(f"[INFO] Final model saved to: {args.save_path}")
    print(f"[INFO] Best model saved to: {args.save_path.replace('.pt', '_best.pt')}")


def main():
    p = argparse.ArgumentParser(description="Sharded training for Qwen3-Omni AuT on multiple GPUs")
    
    # Training params
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=8, help="Effective batch size per step (small for memory)")
    p.add_argument('--gradient_accumulation_steps', type=int, default=16, help="Accumulate gradients to simulate larger batch")
    p.add_argument('--lr', type=float, default=1e-4)
    
    # Data paths
    p.add_argument('--train_samples_path', type=str, required=True)
    p.add_argument('--test_samples_path', type=str, default=None)
    p.add_argument('--train_ratio', type=float, default=0.998)
    p.add_argument('--mmap_shard_dir', type=str, default=None)
    
    # Model params
    p.add_argument('--aut_model_name', type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    p.add_argument('--text_model_name', type=str, default="Qwen/Qwen2-Audio-7B-Instruct")
    p.add_argument('--enable_speech_lora', action='store_true')
    p.add_argument('--lora_r', type=int, default=8)
    p.add_argument('--lora_alpha', type=int, default=16)
    p.add_argument('--lora_dropout', type=float, default=0.0)
    
    # Loss params
    p.add_argument('--audio_text_loss_ratio', type=float, default=0.3)
    p.add_argument('--audio_term_loss_ratio', type=float, default=0.7)
    
    # Save paths
    p.add_argument('--save_path', type=str, default="models/qwen3_aut_sharded.pt")
    p.add_argument('--offload_folder', type=str, default=None, help="Folder for CPU/disk offloading")
    
    # Compatibility args (ignored)
    p.add_argument('--glossary_path', type=str, default=None)
    p.add_argument('--best_model_path', type=str, default=None)
    p.add_argument('--model_name', type=str, default=None)
    p.add_argument('--patience', type=int, default=4)
    
    args = p.parse_args()
    
    if torch.cuda.device_count() == 0:
        print("[ERROR] No CUDA devices available!")
        return 1
    
    # Create save directory
    os.makedirs(os.path.dirname(args.save_path) if os.path.dirname(args.save_path) else ".", exist_ok=True)
    
    train_sharded(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

