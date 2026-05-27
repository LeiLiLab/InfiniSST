#!/usr/bin/env python3
"""
Local DDP Training Script for Term-Level RAG
Based on Qwen2-Audio with LoRA fine-tuning

Usage:
    # Single GPU
    python train_term_rag_local.py --train_jsonl /path/to/data.jsonl

    # Multi-GPU DDP
    torchrun --nproc_per_node=8 train_term_rag_local.py --train_jsonl /path/to/data.jsonl
"""

import os
import sys
import time
import argparse
import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import soundfile as sf
from tqdm import tqdm

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Enable TF32 for better performance on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Import model components
from modal.Qwen2_Audio_train import (
    Qwen2AudioSpeechEncoder,
    Qwen2AudioTextEncoder,
    ContrastiveQwen2AudioModel,
    encode_texts_in_batches,
)


class TermRAGDataset(Dataset):
    """
    Dataset for Term-Level RAG training.
    Supports both positive samples (with terms) and negative samples (no terms).
    
    Data format (JSONL):
    {
        "term": "popular tales",  # Can be empty string for no-term samples
        "translation": "民间故事",
        "chunk_src_text": "nature, popular tales are",
        "chunk_tgt_text": "自然一样， 民间故事",
        "chunk_audio_path": "/path/to/audio.wav",
        "utter_id": "AUD0000000003_1",
        "chunk_idx": 2
    }
    """
    
    def __init__(
        self,
        jsonl_path: str,
        split: str = "train",
        train_ratio: float = 0.99,
        max_audio_duration: float = 10.0,
        min_audio_duration: float = 0.1,
        seed: int = 42,
        limit: Optional[int] = None,
    ):
        self.max_audio_samples = int(max_audio_duration * 16000)
        self.min_audio_samples = int(min_audio_duration * 16000)
        
        # Load all samples
        logger.info(f"Loading data from {jsonl_path}...")
        all_samples = []
        
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if limit and line_idx >= limit:
                    break
                try:
                    sample = json.loads(line.strip())
                    # Validate required fields
                    if "chunk_audio_path" not in sample or "chunk_src_text" not in sample:
                        continue
                    
                    # Check audio file exists
                    audio_path = sample["chunk_audio_path"]
                    if not os.path.exists(audio_path):
                        continue
                    
                    all_samples.append(sample)
                    
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    if line_idx < 10:
                        logger.warning(f"Error parsing line {line_idx}: {e}")
                    continue
        
        logger.info(f"Loaded {len(all_samples)} valid samples")
        
        # Count positive and negative samples
        pos_count = sum(1 for s in all_samples if s.get("term", "").strip())
        neg_count = len(all_samples) - pos_count
        logger.info(f"Positive samples (with term): {pos_count}")
        logger.info(f"Negative samples (no term): {neg_count} ({neg_count/len(all_samples)*100:.1f}%)")
        
        # Shuffle and split
        random.seed(seed)
        random.shuffle(all_samples)
        
        split_idx = int(len(all_samples) * train_ratio)
        if split == "train":
            self.samples = all_samples[:split_idx]
        else:
            self.samples = all_samples[split_idx:]
        
        logger.info(f"{split.capitalize()} set: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Optional[Dict]:
        sample = self.samples[idx]
        
        try:
            # Load audio
            audio_path = sample["chunk_audio_path"]
            audio_data, sr = sf.read(audio_path)
            
            # Validate audio
            if len(audio_data) == 0:
                return None
            
            # Convert to mono if stereo
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Resample if needed (Qwen2-Audio expects 16kHz)
            if sr != 16000:
                logger.info(f"Resampling audio from {sr}Hz to 16000Hz")
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
            
            # Clip to max duration
            if len(audio_data) > self.max_audio_samples:
                audio_data = audio_data[:self.max_audio_samples]
            
            # Skip if too short
            if len(audio_data) < self.min_audio_samples:
                return None
            
            # Normalize audio
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data)) * 0.95
            
            audio_tensor = torch.from_numpy(audio_data.astype(np.float32))
            
            return {
                "term": sample.get("term", "").strip().lower(),
                "transcript": sample.get("chunk_src_text", "").strip().lower(),
                "audio": audio_tensor,
                "utter_id": sample.get("utter_id", ""),
                "chunk_idx": sample.get("chunk_idx", 0),
            }
            
        except Exception as e:
            if idx < 10:
                logger.warning(f"Error loading sample {idx}: {e}")
            return None


def collate_fn(batch: List[Optional[Dict]]) -> List[Dict]:
    """Filter None samples and return list of valid samples."""
    return [s for s in batch if s is not None]


def train_step(
    model: nn.Module,
    batch: List[Dict],
    device: torch.device,
    audio_term_ratio: float = 0.7,
    audio_text_ratio: float = 0.3,
    temperature: float = 0.07,
) -> Optional[torch.Tensor]:
    """
    Training step with in-batch contrastive learning.
    
    Loss = audio_term_ratio * Loss(Audio, Term) + audio_text_ratio * Loss(Audio, Transcript)
    
    Trash Bin Strategy:
    - Empty terms are replaced with "[NO_TERM]" to pull no-term audio to a "null" embedding.
    - This also pushes no-term audio away from real terms in the batch.
    """
    raw_model = model.module if isinstance(model, DDP) else model
    NULL_TERM_TOKEN = "[NO_TERM]"
    
    if len(batch) < 2:
        return None
    
    # Extract data
    audio_list = [s["audio"] for s in batch]
    transcripts = [s["transcript"] for s in batch]
    # Replace empty terms with NULL_TERM_TOKEN (Trash Bin)
    batch_terms = [s["term"] if s["term"] else NULL_TERM_TOKEN for s in batch]
    
    try:
        # Encode audio
        audio_emb = raw_model.encode_audio(audio_list)  # [B, D]
        
        if torch.isnan(audio_emb).any() or torch.isinf(audio_emb).any():
            return None
        
    except Exception as e:
        logger.error(f"Audio encoding failed: {e}")
        return None
    
    batch_size = len(batch)
    
    # ==================== Audio-Transcript Contrastive Loss ====================
    contrastive_loss = torch.tensor(0.0, device=device)
    if audio_text_ratio > 0:
        try:
            with torch.no_grad():
                text_emb = raw_model.encode_text(transcripts)  # [B, D]
            text_emb = text_emb.detach()
            
            # Similarity matrix
            sim_matrix = (audio_emb @ text_emb.T) / temperature  # [B, B]
            labels = torch.arange(batch_size, dtype=torch.long, device=device)
            
            loss_a2t = F.cross_entropy(sim_matrix, labels)
            loss_t2a = F.cross_entropy(sim_matrix.T, labels)
            contrastive_loss = (loss_a2t + loss_t2a) / 2
            
        except Exception as e:
            logger.error(f"Transcript contrastive loss failed: {e}")
    
    # ==================== Audio-Term Contrastive Loss (Trash Bin Strategy) ====================
    audio_term_loss = torch.tensor(0.0, device=device)
    
    if audio_term_ratio > 0:
        try:
            # Deduplicate terms in the batch to handle multi-positive samples correctly
            # and reduce text encoding overhead.
            unique_terms = list(set(batch_terms))
            term_to_idx = {t: idx for idx, t in enumerate(unique_terms)}
            
            # Encode unique terms (including [NO_TERM])
            with torch.no_grad():
                terms_emb = raw_model.encode_text(unique_terms)  # [U, D]
            terms_emb = terms_emb.detach()
            
            # Audio-Term similarity matrix: [Batch, UniqueTerms]
            audio_term_sim = (audio_emb @ terms_emb.T) / temperature
            
            # Build labels: which index in unique_terms does each audio point to?
            audio_term_labels = torch.tensor(
                [term_to_idx[t] for t in batch_terms], 
                dtype=torch.long, 
                device=device
            )
            
            # Cross entropy naturally handles pulling audio to its target term 
            # while pushing it away from all other terms in the batch.
            audio_term_loss = F.cross_entropy(audio_term_sim, audio_term_labels)
            
        except Exception as e:
            logger.error(f"Audio-term loss failed: {e}")
    
    # ==================== Combined Loss ====================
    total_loss = audio_term_ratio * audio_term_loss + audio_text_ratio * contrastive_loss
    
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        return None
    
    return {
        "total_loss": total_loss,
        "audio_term_loss": audio_term_loss,
        "audio_text_loss": contrastive_loss
    }


def evaluate(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    max_samples: int = 1000,
    top_ks: Tuple[int, ...] = (5, 10, 20),
    glossary_terms: Optional[List[str]] = None,
) -> Dict[int, float]:
    """
    Evaluate term retrieval recall against a glossary.
    """
    model.eval()
    raw_model = model.module if isinstance(model, DDP) else model
    
    # Sample evaluation indices
    eval_indices = random.sample(range(len(dataset)), min(max_samples, len(dataset)))
    
    # Define the search space (the glossary)
    if glossary_terms:
        all_terms = glossary_terms
        logger.info(f"Evaluating against provided glossary of {len(all_terms)} terms")
    else:
        # Fallback: only terms in eval samples (easier task)
        all_terms_set = set()
        for idx in eval_indices:
            sample = dataset[idx]
            if sample is not None and sample["term"]:
                all_terms_set.add(sample["term"])
        all_terms = list(all_terms_set)
        logger.info(f"Evaluating against {len(all_terms)} terms found in eval samples")

    valid_samples = []
    for idx in eval_indices:
        sample = dataset[idx]
        if sample is not None and sample["term"]:
            # Only evaluate if the term actually exists in our index
            if sample["term"] in all_terms:
                valid_samples.append(sample)
    
    # Encode all terms
    with torch.no_grad():
        term_embs = encode_texts_in_batches(raw_model, all_terms, batch_size=256, device=device)
        term_embs = term_embs.cpu().numpy()
    
    # Build FAISS index
    import faiss
    index = faiss.IndexFlatIP(term_embs.shape[1])
    faiss.normalize_L2(term_embs)
    index.add(term_embs)
    
    # Evaluate
    recall_dict = {k: [] for k in top_ks}
    
    for sample in tqdm(valid_samples, desc="Evaluating"):
        gt_term = sample["term"]
        audio = sample["audio"]
        
        # Encode audio
        with torch.no_grad():
            audio_emb = raw_model.encode_audio([audio])
            audio_emb = audio_emb.cpu().numpy()
        
        faiss.normalize_L2(audio_emb)
        
        # Search
        max_k = max(top_ks)
        D, I = index.search(audio_emb, max_k)
        retrieved_terms = [all_terms[i] for i in I[0]]
        
        # Calculate recall
        for k in top_ks:
            hit = 1.0 if gt_term in retrieved_terms[:k] else 0.0
            recall_dict[k].append(hit)
    
    # Average recall
    avg_recall = {}
    for k in top_ks:
        avg_recall[k] = sum(recall_dict[k]) / len(recall_dict[k]) if recall_dict[k] else 0.0
        logger.info(f"Recall@{k}: {avg_recall[k]:.2%}")
    
    model.train()
    return avg_recall


def setup_ddp(rank: int, world_size: int):
    """Setup DDP environment."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    
    # NCCL settings
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["NCCL_P2P_DISABLE"] = "0"
    os.environ["NCCL_IB_DISABLE"] = "1"
    
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)
    
    if rank == 0:
        logger.info(f"DDP initialized with {world_size} GPUs")


def cleanup_ddp():
    """Cleanup DDP."""
    if dist.is_initialized():
        dist.destroy_process_group()


def build_glossary_from_jsonl(jsonl_path: str) -> List[str]:
    """
    Build a unique list of terms from the training JSONL.
    Merges translations for the same term with commas.
    """
    logger.info(f"Building glossary from {jsonl_path}... this may take a minute.")
    term_map = {} # term -> set of translations
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                term = item.get("term", "").strip().lower()
                if not term:
                    continue
                
                trans = item.get("translation", "").strip()
                if term not in term_map:
                    term_map[term] = set()
                if trans:
                    term_map[term].add(trans)
            except:
                continue
    
    unique_terms = sorted(list(term_map.keys()))
    logger.info(f"Glossary built: {len(unique_terms)} unique terms.")
    return unique_terms


def train(rank: int, world_size: int, args):
    """Main training function."""
    # Setup DDP
    if world_size > 1:
        setup_ddp(rank, world_size)
    
    device = torch.device(f"cuda:{rank}")
    is_main = rank == 0
    
    if is_main:
        logger.info(f"Starting training with {world_size} GPUs")
        logger.info(f"Training data: {args.train_jsonl}")
    
    # ==================== Model Setup ====================
    if is_main:
        logger.info(f"Loading model: {args.model_name}")
    
    speech_encoder = Qwen2AudioSpeechEncoder(model_name=args.model_name, device=device)
    text_encoder = Qwen2AudioTextEncoder(
        model_name=args.model_name,
        device=device,
        shared_model=speech_encoder.get_shared_model(),
    )
    
    model = ContrastiveQwen2AudioModel(
        speech_encoder,
        text_encoder,
        proj_dim=512,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    ).to(device)
    
    # Force enable LoRA gradients
    if is_main:
        model.force_enable_lora_gradients()
    
    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    # ==================== Dataset Setup ====================
    full_glossary = None
    if is_main:
        # Build glossary with caching
        cache_path = args.train_jsonl + ".glossary.pkl"
        if os.path.exists(cache_path):
            logger.info(f"Loading glossary from cache: {cache_path}")
            try:
                import pickle
                with open(cache_path, "rb") as f:
                    full_glossary = pickle.load(f)
                logger.info(f"Loaded {len(full_glossary)} terms from cache")
            except Exception as e:
                logger.error(f"Failed to load glossary cache: {e}, rebuilding...")
                full_glossary = build_glossary_from_jsonl(args.train_jsonl)
        else:
            full_glossary = build_glossary_from_jsonl(args.train_jsonl)
            try:
                import pickle
                with open(cache_path, "wb") as f:
                    pickle.dump(full_glossary, f)
                logger.info(f"Glossary cache saved to: {cache_path}")
            except Exception as e:
                logger.error(f"Failed to save glossary cache: {e}")

    train_dataset = TermRAGDataset(
        args.train_jsonl,
        split="train",
        train_ratio=args.train_ratio,
        limit=args.test_limit,
    )
    
    test_dataset = TermRAGDataset(
        args.train_jsonl,
        split="test",
        train_ratio=args.train_ratio,
        limit=args.test_limit if args.test_limit else None,
    )
    
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        train_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size // max(1, world_size),
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    
    if is_main:
        logger.info(f"Train: {len(train_dataset)} samples, Test: {len(test_dataset)} samples")
        logger.info(f"Batch size: {args.batch_size} (per GPU: {args.batch_size // max(1, world_size)})")
        logger.info(f"Gradient accumulation: {args.gradient_accumulation_steps}")
        logger.info(f"Effective batch: {args.batch_size * args.gradient_accumulation_steps}")
        
        # Initialize WandB
        if args.wandb_project:
            try:
                import wandb
                wandb.init(
                    project=args.wandb_project,
                    name=args.wandb_exp_name,
                    config=vars(args)
                )
            except ImportError:
                logger.error("WandB requested but not installed. Skipping.")
            except Exception as e:
                logger.error(f"WandB init failed: {e}")
    
    # ==================== Optimizer Setup ====================
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if is_main:
        total_params = sum(p.numel() for p in trainable_params)
        logger.info(f"Trainable parameters: {total_params:,}")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader) // args.gradient_accumulation_steps
    )
    scaler = torch.cuda.amp.GradScaler()
    
    # ==================== Training Loop ====================
    best_recall = 0.0
    no_improve_epochs = 0
    global_step = 0
    start_epoch = 0

    # Load checkpoint if provided or auto-resume
    resume_path = args.resume_from
    if not resume_path and args.resume and os.path.exists(args.save_path):
        resume_path = args.save_path

    if resume_path and os.path.exists(resume_path):
        if is_main:
            logger.info(f"Loading checkpoint from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        
        # Load model weights
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            # Load other states if they exist
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"]
            if "global_step" in checkpoint:
                global_step = checkpoint["global_step"]
            if "recall@10" in checkpoint:
                best_recall = checkpoint["recall@10"]
        else:
            state_dict = checkpoint
        
        # Handle DDP prefix
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        if is_main:
            logger.info(f"Successfully resumed from epoch {start_epoch}, step {global_step}")

    for epoch in range(start_epoch, args.epochs):
        if world_size > 1:
            train_sampler.set_epoch(epoch)
        
        model.train()
        total_loss = 0.0
        valid_batches = 0
        accumulated_loss = 0.0
        accumulation_steps = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}") if is_main else train_loader
        
        for batch_idx, batch in enumerate(pbar):
            if not batch or len(batch) < 2:
                continue
            
            # Forward pass with AMP
            with torch.cuda.amp.autocast():
                loss_dict = train_step(
                    model, batch, device,
                    audio_term_ratio=args.audio_term_ratio,
                    audio_text_ratio=args.audio_text_ratio,
                )
            
            if loss_dict is None:
                continue
                
            loss = loss_dict["total_loss"]
            if not loss.requires_grad:
                continue
            
            # Backward with gradient accumulation
            scaled_loss = loss / args.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            accumulated_loss += loss.item()
            accumulation_steps += 1
            
            # Optimizer step
            if accumulation_steps >= args.gradient_accumulation_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                
                total_loss += accumulated_loss
                valid_batches += 1
                global_step += 1
                
                # Log to WandB
                if is_main and args.wandb_project:
                    import wandb
                    wandb.log({
                        "train/loss": accumulated_loss / args.gradient_accumulation_steps,
                        "train/audio_term_loss": loss_dict["audio_term_loss"].item(),
                        "train/audio_text_loss": loss_dict["audio_text_loss"].item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch + 1,
                        "global_step": global_step
                    })
                
                accumulated_loss = 0.0
                accumulation_steps = 0
                
                # Periodic saving (main process only)
                if is_main and global_step % args.save_every_steps == 0:
                    raw_model = model.module if isinstance(model, DDP) else model
                    torch.save({
                        "model_state_dict": raw_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "recall@10": best_recall,
                    }, args.save_path)
                    logger.info(f"Checkpoint saved at step {global_step} to {args.save_path}")

                if is_main and isinstance(pbar, tqdm):
                    pbar.set_postfix({"loss": f"{total_loss/valid_batches:.4f}"})
        
        # Epoch summary
        avg_loss = total_loss / max(1, valid_batches)
        if world_size > 1:
            loss_tensor = torch.tensor(avg_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / world_size
        
        if is_main:
            logger.info(f"Epoch {epoch+1}/{args.epochs} - Avg Loss: {avg_loss:.4f}")
        
        # Evaluation (main process only)
        if is_main and (epoch + 1) % args.eval_every == 0:
            # Every 10 epochs (or if it's the last epoch), evaluate against full glossary
            use_full = (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs
            
            recall_dict = evaluate(
                model, 
                test_dataset, 
                device, 
                max_samples=args.eval_samples,
                glossary_terms=full_glossary if use_full else None
            )
            
            current_recall = recall_dict.get(10, 0.0)
            
            # Log to WandB
            if args.wandb_project:
                import wandb
                wandb.log({
                    f"eval/recall@{k}": v for k, v in recall_dict.items()
                })
                wandb.log({"epoch": epoch + 1})
            
            if current_recall > best_recall:
                best_recall = current_recall
                no_improve_epochs = 0
                
                # Save best model
                save_path = args.save_path.replace(".pt", "_best.pt")
                raw_model = model.module if isinstance(model, DDP) else model
                torch.save({
                    "model_state_dict": raw_model.state_dict(),
                    "epoch": epoch + 1,
                    "recall@10": best_recall,
                }, save_path)
                logger.info(f"New best model saved (Recall@10: {best_recall:.2%})")
            else:
                no_improve_epochs += 1
                logger.info(f"No improvement for {no_improve_epochs} epochs")
            
            if no_improve_epochs >= args.patience:
                logger.info(f"Early stopping. Best Recall@10: {best_recall:.2%}")
                break
        
        if world_size > 1:
            dist.barrier()
    
    # Save final model
    if is_main:
        raw_model = model.module if isinstance(model, DDP) else model
        torch.save({
            "model_state_dict": raw_model.state_dict(),
            "epoch": args.epochs,
            "recall@10": best_recall,
        }, args.save_path)
        logger.info(f"Final model saved to {args.save_path}")
        logger.info(f"Best Recall@10: {best_recall:.2%}")
    
    cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(description="Local Term-Level RAG Training")
    
    # Data
    parser.add_argument("--train_jsonl", type=str, required=True, help="Path to training JSONL file")
    parser.add_argument("--train_ratio", type=float, default=0.99, help="Train/test split ratio")
    
    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    
    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128, help="Total batch size across all GPUs")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    
    # Loss weights
    parser.add_argument("--audio_term_ratio", type=float, default=0.7)
    parser.add_argument("--audio_text_ratio", type=float, default=0.3)
    
    # Evaluation
    parser.add_argument("--eval_every", type=int, default=1, help="Evaluate every N epochs")
    parser.add_argument("--eval_samples", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=3)
    
    # Checkpoints
    parser.add_argument("--save_path", type=str, default="term_rag_model.pt")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--save_every_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--resume", action="store_true", help="Auto resume from save_path if exists")
    
    # Debug / Sanity Check
    parser.add_argument("--test_limit", type=int, default=None, help="Limit number of samples for sanity check")
    
    # WandB
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_exp_name", type=str, default="rag")
    
    args = parser.parse_args()
    
    # Determine world size
    if "LOCAL_RANK" in os.environ:
        # Launched with torchrun
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    else:
        # Single GPU
        rank = 0
        world_size = 1
    
    train(rank, world_size, args)


if __name__ == "__main__":
    main()

