#!/usr/bin/env python3
"""
Qwen3-Omni + BGE-M3 (Unfrozen Text Encoder) Training Script
Audio Encoder: Qwen3OmniMoeAudioEncoder (LoRA)
Text Encoder: BGE-M3 (LoRA, Unfrozen)
Data: JSONL with local audio files
"""

import os
import sys
import time
import argparse
import json
import random
import logging
import pickle
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
from transformers import WhisperFeatureExtractor, AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder
from peft import LoraConfig, get_peft_model, TaskType

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Enable TF32/BF16 optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==================== Model Components ====================

class AttentivePooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, x, mask=None):
        scores = self.attention(x) # [B, T, 1]
        if mask is not None:
            mask = mask.unsqueeze(-1)
            scores = scores.masked_fill(~mask, -1e9)
        weights = F.softmax(scores, dim=1)
        pooled = torch.sum(x * weights, dim=1)
        return pooled

class GatherLayer(torch.autograd.Function):
    """用于在 DDP 模式下收集 Tensor 并保持梯度传播"""
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out

def all_gather_with_grad(tensor):
    world_size = dist.get_world_size()
    if world_size <= 1:
        return tensor
    tensors_gather = GatherLayer.apply(tensor)
    return torch.cat(tensors_gather, dim=0)

class BgeM3TextEncoder(nn.Module):
    def __init__(self, model_id="BAAI/bge-m3", lora_rank=16, lora_alpha=32, target_modules=None):
        super().__init__()
        # 显式使用 BF16 加载以节省显存
        self.encoder = AutoModel.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16,
            add_pooling_layer=False
        )
        
        if target_modules is None:
            target_modules = ["query", "key", "value"]

        # BGE-M3 (XLM-Roberta) target modules
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=None
        )
        self.encoder = get_peft_model(self.encoder, lora_config)
        self.encoder.print_trainable_parameters()
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # 取 CLS Token 并归一化
        embeddings = outputs.last_hidden_state[:, 0, :]
        return F.normalize(embeddings, p=2, dim=-1)

class Qwen3OmniRetriever(nn.Module):
    def __init__(self, model_id="Atotti/Qwen3-Omni-AudioTransformer", target_dim=1024, 
                 use_lora=True, lora_rank=32, lora_alpha=64, lora_target_modules=None,
                 temperature=0.07):
        super().__init__()
        
        # 1. 加载 Encoder (BF16)
        self.audio_encoder = Qwen3OmniMoeAudioEncoder.from_pretrained(
            model_id, 
            dtype=torch.bfloat16
        )
        
        if hasattr(self.audio_encoder, "conv2d1"):
            self.audio_encoder.get_input_embeddings = lambda: self.audio_encoder.conv2d1
            
        self.audio_encoder.gradient_checkpointing_enable()
        
        # 2. 应用 LoRA
        if use_lora:
            if lora_target_modules is None:
                lora_target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "proj1", "proj2"]
            
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type=None
            )
            self.audio_encoder = get_peft_model(self.audio_encoder, lora_config)
            self.audio_encoder.print_trainable_parameters()
        else:
            for param in self.audio_encoder.parameters():
                param.requires_grad = False

        # 3. 投影层和 Pooler
        self.pooler = AttentivePooling(2048) 
        self.projector = nn.Linear(2048, target_dim)
        
        # Logit Scale
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

    def forward(self, input_features, feature_lens):
        # 🟢 核心修复：在各 GPU 内部进行打包，确保 DDP 切分的是 Batch 维度
        if input_features.ndim == 3:
            # [B, C, T] -> [C, B*T]
            input_features = input_features.transpose(0, 1).reshape(input_features.shape[1], -1)

        outputs = self.audio_encoder(input_features, feature_lens)
        hidden_states = outputs.last_hidden_state
        
        if hidden_states.ndim == 2:
            output_lens = []
            for l in feature_lens.tolist():
                curr_l = l
                for _ in range(3):
                    curr_l = (curr_l + 1) // 2
                output_lens.append(curr_l)
            
            if sum(output_lens) != hidden_states.shape[0]:
                ratio = input_features.shape[1] / hidden_states.shape[0]
                output_lens = [max(1, round(l / ratio)) for l in feature_lens.tolist()]
                output_lens[-1] = hidden_states.shape[0] - sum(output_lens[:-1])

            hidden_states_list = torch.split(hidden_states, output_lens, dim=0)
            from torch.nn.utils.rnn import pad_sequence
            hidden_states = pad_sequence(hidden_states_list, batch_first=True)
            feature_lens = torch.tensor(output_lens, device=hidden_states.device)
            
        batch_size, max_len, _ = hidden_states.shape
        mask = torch.arange(max_len, device=hidden_states.device).expand(batch_size, max_len) < feature_lens.unsqueeze(1)
        
        pooled_audio = self.pooler(hidden_states, mask)
        projected = self.projector(pooled_audio)
        return F.normalize(projected, p=2, dim=-1)

# ==================== Dataset ====================

class TermRAGDataset(Dataset):
    def __init__(self, samples: List[Dict]):
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_path = sample["chunk_audio_path"]
        try:
            audio_data, sr = sf.read(audio_path)
            if sr != 16000:
                logger.warning(f"[WRONG SR] {audio_path} has SR {sr}, expected 16000")
            
            if audio_data.ndim > 1: audio_data = audio_data.mean(axis=1)
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            res = {k: v for k, v in sample.items()}
            res["audio"] = audio_data.astype(np.float32)
            return res
        except Exception as e:
            logger.warning(f"[SKIP] Audio load error: {audio_path} | Error: {e}")
            return {"audio": None, "chunk_audio_path": audio_path}

def collate_fn(batch, feature_extractor):
    valid_samples = []
    for s in batch:
        if s is None:
            logger.warning("[COLLATE] Received None sample from dataset.")
            continue
        
        audio = s.get("audio")
        if audio is None:
            logger.warning(f"[COLLATE] Sample has no audio data. Path: {s.get('chunk_audio_path')}")
            continue
            
        if len(audio) <= 3000:
            logger.warning(f"[COLLATE] Audio too short ({len(audio)} samples). Path: {s.get('chunk_audio_path')}")
            continue
            
        valid_samples.append(s)
    
    # 🟢 核心修复：DDP 模式下绝不能返回 None，否则进程步调不一致会崩溃
    if not valid_samples:
        logger.error("[COLLATE] CRITICAL: Entire batch was filtered out! Using dummy sample.")
        dummy_audio = np.zeros(30720, dtype=np.float32)
        valid_samples = [{"audio": dummy_audio, "term": "dummy", "translation": "dummy", "gt_term": "dummy", "chunk_audio_path": "DUMMY"}]

    target_len = 30720 
    audios = []
    for s in valid_samples:
        audio = s["audio"]
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')
        elif len(audio) > target_len:
            audio = audio[:target_len]
        audios.append(audio)

    try:
        inputs = feature_extractor(audios, sampling_rate=16000, return_tensors="pt", padding=False)
        input_features = inputs.input_features # [B, C, T_mel]
        B, C, T_mel = input_features.shape
        feature_lens = torch.full((B,), T_mel, dtype=torch.long)
    except Exception as e:
        logger.error(f"[CRITICAL] Batch extraction failed: {e}")
        return None
    
    return {
        "input_features": input_features,
        "feature_lens": feature_lens,
        "samples": valid_samples 
    }

# ==================== Evaluation Helpers ====================

def get_glossary_info(jsonl_paths, cache_dir):
    cache_path = os.path.join(cache_dir, "glossary_info_merged_unfrozen.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    if isinstance(jsonl_paths, str):
        jsonl_paths = [jsonl_paths]

    term_to_info = {}
    for jsonl_path in jsonl_paths:
        if not os.path.exists(jsonl_path):
            logger.warning(f"Glossary path not found: {jsonl_path}")
            continue
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(tqdm(f, desc=f"Scanning {os.path.basename(jsonl_path)}")):
                try:
                    item = json.loads(line)
                    term = item.get("term", "").strip().lower()
                    if not term: continue
                    trans = item.get("translation", "").strip()
                    if term not in term_to_info:
                        term_to_info[term] = {"translations": set()}
                    if trans:
                        term_to_info[term]["translations"].add(trans)
                except: continue
            
    unique_terms = []
    raw_terms = []
    for term, info in term_to_info.items():
        merged_trans = ", ".join(sorted(list(info["translations"])))
        if merged_trans:
            unique_terms.append(f"{term} ({merged_trans})")
        else:
            unique_terms.append(term)
        raw_terms.append(term)
        
    result = {"terms": unique_terms, "raw_terms": raw_terms}
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    return result

def get_dev_glossary_info(jsonl_path, cache_dir):
    cache_path = os.path.join(cache_dir, "dev_glossary_info_unfrozen.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    dev_term_to_idx = {}
    NULL_TOKEN = "[NO_TERM]"
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            try:
                t = json.loads(line).get("term", "").strip().lower()
                if not t: t = NULL_TOKEN
                if t not in dev_term_to_idx:
                    dev_term_to_idx[t] = line_idx
            except: continue
            
    result = {"unique_terms": list(dev_term_to_idx.keys())}
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    return result

# ==================== Training Logic ====================

def train(rank, world_size, args):
    if world_size > 1:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    is_main = rank == 0

    # 1. Models
    retriever = Qwen3OmniRetriever(
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        temperature=args.temperature
    ).to(device)

    text_encoder = BgeM3TextEncoder(
        lora_rank=args.text_lora_rank,
        lora_alpha=args.text_lora_alpha,
        target_modules=args.text_lora_target_modules
    ).to(device)
    text_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    
    if is_main:
        logger.info("======= Model Structure =======")
        print(retriever)
        print(text_encoder)
        logger.info("===============================")
    
    if world_size > 1:
        retriever = DDP(retriever, device_ids=[rank])
        text_encoder = DDP(text_encoder, device_ids=[rank])
    
    # 2. Metadata & Glossary
    # 🟢 修复：合并训练集和开发集的术语，确保全量评测库完整
    glossary_info = get_glossary_info([args.train_jsonl, args.dev_jsonl], args.precomputed_dir)
    dev_info = get_dev_glossary_info(args.dev_jsonl, args.precomputed_dev_dir)
    dev_unique_terms = dev_info["unique_terms"]

    # 3. Optimizer & Parameters
    raw_model = retriever.module if world_size > 1 else retriever
    raw_text_model = text_encoder.module if world_size > 1 else text_encoder

    audio_lora_params = [p for p in raw_model.audio_encoder.parameters() if p.requires_grad]
    text_lora_params = [p for p in raw_text_model.encoder.parameters() if p.requires_grad]
    head_params = list(raw_model.pooler.parameters()) + \
                  list(raw_model.projector.parameters()) + \
                  [raw_model.logit_scale]

    optimizer_grouped_parameters = []
    if len(audio_lora_params) > 0:
        optimizer_grouped_parameters.append({"params": audio_lora_params, "lr": args.lr, "name": "audio_lora"})
    if len(text_lora_params) > 0:
        optimizer_grouped_parameters.append({"params": text_lora_params, "lr": args.lr, "name": "text_lora"})
    optimizer_grouped_parameters.append({"params": head_params, "lr": args.lr * 10, "name": "head"})

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")

    # 4. Resume
    start_epoch = 0
    global_step = 0
    best_recall5_sampled = 0.0
    best_recall5_full = 0.0
    pending_scheduler_state = None
    pending_scaler_state = None
    
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)

        # Always load into the *raw* (unwrapped) modules to avoid DDP prefix mismatch.
        raw_retriever = retriever.module if world_size > 1 else retriever
        raw_text_encoder = text_encoder.module if world_size > 1 else text_encoder

        def _strip_module_prefix(state_dict):
            if not isinstance(state_dict, dict) or not state_dict:
                return state_dict
            # If keys are like "module.xxx", strip the first "module.".
            if any(k.startswith("module.") for k in state_dict.keys()):
                return {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}
            return state_dict

        # Model
        model_sd = _strip_module_prefix(checkpoint.get("model_state_dict", {}))
        incompat = raw_retriever.load_state_dict(model_sd, strict=False)

        # Text model (optional)
        if "text_model_state_dict" in checkpoint:
            text_sd = _strip_module_prefix(checkpoint.get("text_model_state_dict", {}))
            text_incompat = raw_text_encoder.load_state_dict(text_sd, strict=False)
        else:
            text_incompat = None

        # Optimizer (optional; can fail if param groups changed)
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                logger.warning(f"[RESUME] Failed to load optimizer_state_dict: {e}")

        # Scheduler / scaler are created later; stash and load after creation.
        pending_scheduler_state = checkpoint.get("scheduler_state_dict")
        pending_scaler_state = checkpoint.get("scaler_state_dict")

        start_epoch = checkpoint.get("epoch", -1) + 1
        global_step = checkpoint.get("global_step", 0)
        best_recall5_sampled = checkpoint.get("best_recall5_sampled", 0.0)
        best_recall5_full = checkpoint.get("best_recall5_full", 0.0)

        if is_main:
            # Print a short summary so we can verify resume actually restored weights.
            mk = getattr(incompat, "missing_keys", [])
            uk = getattr(incompat, "unexpected_keys", [])
            logger.info(f"[RESUME] Loaded model_state_dict from: {args.resume}")
            logger.info(f"[RESUME] start_epoch={start_epoch}, global_step={global_step}")
            logger.info(f"[RESUME] model missing_keys={len(mk)}, unexpected_keys={len(uk)}")
            if len(mk) > 0:
                logger.info(f"[RESUME] model missing_keys (head): {mk[:10]}")
            if len(uk) > 0:
                logger.info(f"[RESUME] model unexpected_keys (head): {uk[:10]}")

            if text_incompat is not None:
                tmk = getattr(text_incompat, "missing_keys", [])
                tuk = getattr(text_incompat, "unexpected_keys", [])
                logger.info(f"[RESUME] text missing_keys={len(tmk)}, unexpected_keys={len(tuk)}")
                if len(tmk) > 0:
                    logger.info(f"[RESUME] text missing_keys (head): {tmk[:10]}")
                if len(tuk) > 0:
                    logger.info(f"[RESUME] text unexpected_keys (head): {tuk[:10]}")

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")

    # 5. Data Loaders
    train_samples = []
    with open(args.train_jsonl, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if args.test_limit and line_idx >= args.test_limit: break
            try:
                sample = json.loads(line.strip())
                train_samples.append(sample)
            except: continue
            
    test_samples = []
    with open(args.dev_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                sample = json.loads(line.strip())
                sample["gt_term"] = sample.get("term", "").lower()
                test_samples.append(sample)
            except: continue
    
    train_dataset = TermRAGDataset(train_samples)
    test_dataset = TermRAGDataset(test_samples)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size // world_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        collate_fn=lambda b: collate_fn(b, feature_extractor),
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size // world_size,
        sampler=DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None,
        shuffle=False, 
        collate_fn=lambda b: collate_fn(b, feature_extractor),
        num_workers=args.num_workers, pin_memory=True
    )

    # 6. Scheduler
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(num_training_steps * 0.1)
    for group in optimizer.param_groups:
        if "initial_lr" not in group:
            group["initial_lr"] = args.lr * 10 if group["name"] == "head" else args.lr

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps,
        last_epoch=global_step - 1 if global_step > 0 else -1
    )

    # Load scheduler/scaler states if we resumed (optional).
    if pending_scheduler_state is not None:
        try:
            scheduler.load_state_dict(pending_scheduler_state)
            if is_main:
                logger.info("[RESUME] Loaded scheduler_state_dict.")
        except Exception as e:
            logger.warning(f"[RESUME] Failed to load scheduler_state_dict: {e}")

    if pending_scaler_state is not None:
        try:
            scaler.load_state_dict(pending_scaler_state)
            if is_main:
                logger.info("[RESUME] Loaded scaler_state_dict.")
        except Exception as e:
            logger.warning(f"[RESUME] Failed to load scaler_state_dict: {e}")

    if is_main:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_exp_name, config=vars(args))

    def run_eval(is_full_eval=False):
        # 🟢 关键修复：评测时获取裸模型，避免 DDP 包装器的同步冲突
        raw_retriever = retriever.module if world_size > 1 else retriever
        raw_text_encoder = text_encoder.module if world_size > 1 else text_encoder
        
        nonlocal best_recall5_sampled, best_recall5_full
        logger.info(f"Starting evaluation (Full: {is_full_eval}, Step: {global_step})...")
        import faiss
        raw_retriever.eval()
        raw_text_encoder.eval()
        
        # 🟢 关键修复：统一使用 raw_terms 进行检索，避免翻译括号对 Embedding 的干扰
        eval_texts = glossary_info['raw_terms'] if is_full_eval else dev_unique_terms
        all_text_embs = []
        eval_batch_size = 256
        with torch.no_grad():
            for i in range(0, len(eval_texts), eval_batch_size):
                batch_texts = eval_texts[i:i+eval_batch_size]
                inputs = text_tokenizer(batch_texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
                # 🟢 使用 raw_text_encoder
                embs = raw_text_encoder(inputs.input_ids, inputs.attention_mask)
                all_text_embs.append(embs.cpu().float().numpy())
        
        search_embs = np.concatenate(all_text_embs, axis=0)
        faiss.normalize_L2(search_embs)
        search_index = faiss.IndexFlatIP(search_embs.shape[1])
        search_index.add(search_embs)

        recall_results_hits = {5: 0.0, 10: 0.0, 20: 0.0}
        total_valid_samples = 0
        eval_samples_count = 0
        max_eval_samples = 2000 // world_size # 每张卡跑一部分
        
        with torch.no_grad():
            for eval_batch in tqdm(test_loader, desc=f"Eval {'Full' if is_full_eval else 'Sampled'}", disable=not is_main):
                if eval_batch is None: continue
                if eval_samples_count >= max_eval_samples: break
                input_features = eval_batch["input_features"].to(device).to(torch.bfloat16)
                feature_lens = eval_batch["feature_lens"].to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    audio_embs = raw_retriever(input_features, feature_lens)
                audio_embs_np = audio_embs.cpu().float().numpy()
                faiss.normalize_L2(audio_embs_np)
                D, I = search_index.search(audio_embs_np, 20)
                
                for i, sample in enumerate(eval_batch["samples"]):
                    eval_samples_count += 1
                    gt_term = sample.get("gt_term", "").strip().lower()
                    if not gt_term: continue
                    total_valid_samples += 1
                    retrieved_terms = [eval_texts[idx] for idx in I[i]]
                    for k in recall_results_hits.keys():
                        if gt_term in [t.lower() for t in retrieved_terms[:k]]:
                            recall_results_hits[k] += 1.0
        
        # 汇总所有卡的命中数和样本数
        if world_size > 1:
            stats_tensor = torch.tensor([total_valid_samples] + list(recall_results_hits.values()), device=device)
            dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
            total_valid_samples = stats_tensor[0].item()
            for idx, k in enumerate(recall_results_hits.keys()):
                recall_results_hits[k] = stats_tensor[idx+1].item()

        if is_main:
            final_metrics = {}
            current_epoch_recall5 = 0.0
            for k, hits_sum in recall_results_hits.items():
                avg_recall = hits_sum / total_valid_samples if total_valid_samples > 0 else 0
                metric_name = f"eval/recall@{k}" + ("_full" if is_full_eval else "_sampled")
                final_metrics[metric_name] = avg_recall
                if k == 5: current_epoch_recall5 = avg_recall
            
            wandb.log(final_metrics)
            improved = False
            if is_full_eval:
                if current_epoch_recall5 > best_recall5_full:
                    best_recall5_full = current_epoch_recall5
                    improved = True
            else:
                if current_epoch_recall5 > best_recall5_sampled:
                    best_recall5_sampled = current_epoch_recall5
                    improved = True

            if improved:
                suffix = "full" if is_full_eval else "sampled"
                actual_save_path = args.save_path.replace(".pt", f"_{suffix}_best.pt")
                save_data = {
                    "model_state_dict": raw_retriever.state_dict(),
                    "text_model_state_dict": raw_text_encoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_recall5_sampled": best_recall5_sampled,
                    "best_recall5_full": best_recall5_full,
                    "args": vars(args)
                }
                torch.save(save_data, actual_save_path)
                logger.info(f"Model saved to {actual_save_path}")
        
        # 同步 best_recall 状态给所有 rank，防止后续改进判定逻辑不一致
        if world_size > 1:
            best_tensor = torch.tensor([best_recall5_sampled, best_recall5_full], device=device)
            dist.broadcast(best_tensor, src=0)
            best_recall5_sampled, best_recall5_full = best_tensor[0].item(), best_tensor[1].item()

        retriever.train()
        text_encoder.train()

    # 7. Training Loop
    for epoch in range(start_epoch, args.epochs):
        retriever.train()
        text_encoder.train()
        if train_sampler: train_sampler.set_epoch(epoch)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}") if is_main else train_loader
        
        for batch in pbar:
            if batch is None: 
                logger.error("Batch is None! This should not happen with the new collate_fn.")
                continue
            global_step += 1
            
            # 每 N 步做一次全量评测
            if args.eval_steps_full > 0 and global_step % args.eval_steps_full == 0:
                run_eval(is_full_eval=True)
            
            # 每 N 步做一次采样评测 (如果同时到了全量步数，优先跑全量，此处跳过采样)
            elif args.eval_steps_sample > 0 and global_step % args.eval_steps_sample == 0:
                run_eval(is_full_eval=False)

            if args.save_steps > 0 and global_step % args.save_steps == 0 and is_main:
                step_save_path = args.save_path.replace(".pt", f"_step_{global_step}.pt")
                torch.save({
                    'epoch': epoch, 'global_step': global_step,
                    'model_state_dict': retriever.state_dict(),
                    'text_model_state_dict': text_encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, step_save_path)
            
            input_features = batch["input_features"].to(device).to(torch.bfloat16)
            feature_lens = batch["feature_lens"].to(device)
            samples = batch["samples"]
            
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                audio_embs = retriever(input_features, feature_lens)
                
                def encode_text(texts):
                    inputs = text_tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
                    return text_encoder(inputs.input_ids, inputs.attention_mask)
                
                logit_scale = retriever.module.logit_scale.exp() if world_size > 1 else retriever.logit_scale.exp()
                
                NULL_TOKEN = "[NO_TERM]"
                total_loss = 0
                loss_term_val = 0
                loss_trans_val = 0
                
                if args.term_weight > 0:
                    # [改动 3] [NO_TERM] 显式对齐：通过 or NULL_TOKEN 统一空术语
                    local_terms = [s.get("term", "").strip().lower() or NULL_TOKEN for s in samples]
                    local_term_embs = encode_text(local_terms) # [B, D]
                    
                    if world_size > 1:
                        # [改动 1] 全局负采样：收集所有卡的 Embeddings 并保持梯度
                        global_audio_embs = all_gather_with_grad(audio_embs) # [B*world, D]
                        global_term_embs = all_gather_with_grad(local_term_embs) # [B*world, D]
                        
                        # 收集文本标签用于构造 mask (字符串收集不带梯度)
                        global_terms_list = [None] * world_size
                        dist.all_gather_object(global_terms_list, local_terms)
                        global_terms = [t for sublist in global_terms_list for t in sublist]
                    else:
                        global_audio_embs = audio_embs
                        global_term_embs = local_term_embs
                        global_terms = local_terms

                    # [改动 4] 用 Cosine Similarity 计算相似度
                    # 由于 audio_embs 和 global_term_embs 已经经过 F.normalize，
                    # 矩阵乘法 (@) 的结果即为 Cosine Similarity。
                    sim_matrix = (audio_embs @ global_term_embs.T) * logit_scale # [Local_B, Global_B]
                    
                    # [改动 2] pos_mask 实现多正例：基于文本匹配构造 mask
                    pos_mask = torch.zeros_like(sim_matrix)
                    for i, l_term in enumerate(local_terms):
                        for j, g_term in enumerate(global_terms):
                            if l_term == g_term: # 字符串完全匹配即视为正例
                                pos_mask[i, j] = 1.0
                    
                    # 多正例 InfoNCE Loss 计算 (MIL)
                    log_prob = sim_matrix - torch.logsumexp(sim_matrix, dim=1, keepdim=True)
                    # 对每个音频的所有正例进行均值 Loss 计算
                    loss_term = - (log_prob * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
                    loss_term = loss_term.mean()
                    
                    total_loss += loss_term * args.term_weight
                    loss_term_val = loss_term.item()
                
                if args.trans_weight > 0:
                    local_trans = [s.get("translation", "").strip().lower() or (s.get("term", "").strip().lower() or NULL_TOKEN) for s in samples]
                    local_trans_embs = encode_text(local_trans)
                    
                    if world_size > 1:
                        global_trans_embs = all_gather_with_grad(local_trans_embs)
                        global_trans_list = [None] * world_size
                        dist.all_gather_object(global_trans_list, local_trans)
                        global_trans = [t for sublist in global_trans_list for t in sublist]
                    else:
                        global_trans_embs = local_trans_embs
                        global_trans = local_trans

                    sim_trans = (audio_embs @ global_trans_embs.T) * logit_scale
                    
                    pos_mask_trans = torch.zeros_like(sim_trans)
                    for i, l_t in enumerate(local_trans):
                        for j, g_t in enumerate(global_trans):
                            if l_t == g_t:
                                pos_mask_trans[i, j] = 1.0
                                
                    log_prob_trans = sim_trans - torch.logsumexp(sim_trans, dim=1, keepdim=True)
                    loss_trans = - (log_prob_trans * pos_mask_trans).sum(dim=1) / pos_mask_trans.sum(dim=1).clamp(min=1)
                    loss_trans = loss_trans.mean()
                    
                    total_loss += loss_trans * args.trans_weight
                    loss_trans_val = loss_trans.item()

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(retriever.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if is_main:
                pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})
                wandb.log({
                    "train/loss": total_loss.item(), 
                    "train/loss_term": loss_term_val,
                    "train/loss_trans": loss_trans_val,
                    "train/lr": optimizer.param_groups[0]["lr"]
                })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--dev_jsonl", type=str, required=True)
    parser.add_argument("--precomputed_dir", type=str, required=True)
    parser.add_argument("--precomputed_dev_dir", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="qwen3_retriever.pt")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use_lora", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--text_lora_rank", type=int, default=16)
    parser.add_argument("--text_lora_alpha", type=int, default=32)
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=None)
    parser.add_argument("--text_lora_target_modules", type=str, nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--test_limit", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps_sample", type=int, default=200)
    parser.add_argument("--eval_steps_full", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--term_weight", type=float, default=1.0)
    parser.add_argument("--trans_weight", type=float, default=0.0)
    parser.add_argument("--wandb_project", type=str, default="qwen3_rag")
    parser.add_argument("--wandb_exp_name", type=str, default="unfrozen_text_encoder")
    args = parser.parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("LOCAL_RANK", 0))
    train(rank, world_size, args)

if __name__ == "__main__":
    main()

