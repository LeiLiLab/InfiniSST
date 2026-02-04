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
import datetime
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
                 temperature=0.07, learn_temp=False):
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
        if learn_temp:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        else:
            self.register_buffer("logit_scale", torch.tensor(np.log(1 / temperature)))

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
    def __init__(self, samples: List[Dict], force_dummy_audio: bool = False):
        self.samples = samples
        self._remap_src = os.environ.get("AUDIO_PATH_REMAP_SRC", "").strip()
        self._remap_dst = os.environ.get("AUDIO_PATH_REMAP_DST", "").strip()
        self._force_dummy_audio = force_dummy_audio
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_path = sample["chunk_audio_path"]
        if self._force_dummy_audio:
            # Fully bypass IO to debug DDP hangs (use fixed-length dummy audio).
            dummy_audio = np.zeros(30720, dtype=np.float32)  # 1.92s @ 16kHz
            res = {k: v for k, v in sample.items()}
            res["audio"] = dummy_audio
            res["chunk_audio_path"] = "DUMMY"
            res["is_dummy"] = True
            return res
        # Remap data1 -> data2 if the file exists there. This avoids IO stalls when data1 is full/unhealthy.
        if self._remap_src and self._remap_dst and audio_path.startswith(self._remap_src):
            candidate = self._remap_dst + audio_path[len(self._remap_src):]
            if os.path.exists(candidate):
                audio_path = candidate
        try:
            audio_data, sr = sf.read(audio_path)
            if sr != 16000:
                logger.warning(f"[WRONG SR] {audio_path} has SR {sr}, expected 16000")
            
            if audio_data.ndim > 1: audio_data = audio_data.mean(axis=1)
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            res = {k: v for k, v in sample.items()}
            res["audio"] = audio_data.astype(np.float32)
            res["chunk_audio_path"] = audio_path
            return res
        except Exception as e:
            logger.warning(f"[SKIP] Audio load error: {audio_path} | Error: {e}")
            return {"audio": None, "chunk_audio_path": audio_path}

def collate_fn(batch, feature_extractor):
    """
    DDP safety requirements:
    - Must ALWAYS return a batch dict (never None), otherwise ranks will desync and hang on collectives.
    - Must keep a consistent local batch size across ranks. Do NOT drop samples; replace invalid ones with dummy.
    """
    target_len = 30720
    dummy_audio = np.zeros(target_len, dtype=np.float32)

    fixed_samples = []
    audios = []

    for s in batch:
        if s is None:
            logger.warning("[BAD_AUDIO] reason=dataset_returned_none action=use_dummy path=<unknown>")
            s = {"term": "dummy", "translation": "dummy", "gt_term": "dummy", "chunk_audio_path": "DUMMY"}
            audio = dummy_audio
            s["is_dummy"] = True
        else:
            audio = s.get("audio")
            if audio is None:
                logger.warning(f"[BAD_AUDIO] reason=audio_missing action=use_dummy path={s.get('chunk_audio_path')}")
                audio = dummy_audio
                s["is_dummy"] = True
            elif len(audio) <= 3000:
                logger.warning(
                    f"[BAD_AUDIO] reason=audio_too_short action=use_dummy "
                    f"num_samples={len(audio)} path={s.get('chunk_audio_path')}"
                )
                audio = dummy_audio
                s["is_dummy"] = True
            else:
                s["is_dummy"] = False

        # Pad / truncate to fixed length for WhisperFeatureExtractor
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)), mode="constant")
        elif len(audio) > target_len:
            audio = audio[:target_len]

        fixed_samples.append(s)
        audios.append(audio)

    # Feature extraction must never fail; if it does, fallback to all-dummy.
    try:
        inputs = feature_extractor(audios, sampling_rate=16000, return_tensors="pt", padding=False)
    except Exception as e:
        logger.error(f"[BAD_AUDIO] reason=feature_extractor_failed action=fallback_to_dummy_batch error={e}")
        fixed_samples = [{"term": "dummy", "translation": "dummy", "gt_term": "dummy", "chunk_audio_path": "DUMMY", "is_dummy": True} for _ in fixed_samples]
        inputs = feature_extractor([dummy_audio for _ in fixed_samples], sampling_rate=16000, return_tensors="pt", padding=False)

    input_features = inputs.input_features  # [B, C, T_mel]
    B, C, T_mel = input_features.shape
    feature_lens = torch.full((B,), T_mel, dtype=torch.long)

    return {
        "input_features": input_features,
        "feature_lens": feature_lens,
        "samples": fixed_samples,
    }

# ==================== Evaluation Helpers ====================

def get_glossary_info(jsonl_paths, cache_dir):
    cache_path = os.path.join(cache_dir, "glossary_info_merged_unfrozen.pkl")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            # Guard against stale/empty cache created by a previous bad run.
            if isinstance(cached, dict) and cached.get("raw_terms"):
                return cached
            logger.warning(
                f"[GLOSSARY_CACHE] reason=empty_or_invalid action=rebuild path={cache_path} "
                f"keys={list(cached.keys()) if isinstance(cached, dict) else type(cached)}"
            )
        except Exception as e:
            logger.warning(f"[GLOSSARY_CACHE] reason=load_failed action=rebuild path={cache_path} error={e}")

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
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if isinstance(cached, dict) and cached.get("unique_terms"):
                return cached
            logger.warning(
                f"[DEV_GLOSSARY_CACHE] reason=empty_or_invalid action=rebuild path={cache_path} "
                f"keys={list(cached.keys()) if isinstance(cached, dict) else type(cached)}"
            )
        except Exception as e:
            logger.warning(f"[DEV_GLOSSARY_CACHE] reason=load_failed action=rebuild path={cache_path} error={e}")

    dev_term_to_idx = {}
    NULL_TOKEN = "[NO_TERM]"
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            try:
                t = json.loads(line).get("term", "").strip().lower()
                if not t: continue
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
        # 增加超时时间到 2 小时，防止 IO 波动导致 NCCL timeout
        dist.init_process_group(
            backend="nccl", 
            rank=rank, 
            world_size=world_size,
            timeout=datetime.timedelta(seconds=7200)
        )
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    is_main = rank == 0

    # 1. Models
    retriever = Qwen3OmniRetriever(
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        temperature=args.temperature,
        learn_temp=args.learn_temp
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
    # DDP note: building glossary cache can be heavy (scanning large JSONL).
    # Let rank0 build/write the cache once; other ranks wait and then load it.
    if world_size > 1:
        if is_main:
            glossary_info = get_glossary_info([args.train_jsonl, args.dev_jsonl], args.precomputed_dir)
            dev_info = get_dev_glossary_info(args.dev_jsonl, args.precomputed_dev_dir)
        dist.barrier()
        if not is_main:
            glossary_info = get_glossary_info([args.train_jsonl, args.dev_jsonl], args.precomputed_dir)
            dev_info = get_dev_glossary_info(args.dev_jsonl, args.precomputed_dev_dir)
        dist.barrier()
    else:
        glossary_info = get_glossary_info([args.train_jsonl, args.dev_jsonl], args.precomputed_dir)
        dev_info = get_dev_glossary_info(args.dev_jsonl, args.precomputed_dev_dir)

    dev_unique_terms = dev_info["unique_terms"]
    if is_main:
        logger.info(
            f"[GLOSSARY] train_jsonl={args.train_jsonl} dev_jsonl={args.dev_jsonl} "
            f"full_terms={len(glossary_info.get('raw_terms', []))} dev_terms={len(dev_unique_terms)}"
        )

    # 3. Optimizer & Parameters
    raw_model = retriever.module if world_size > 1 else retriever
    raw_text_model = text_encoder.module if world_size > 1 else text_encoder

    audio_lora_params = [p for p in raw_model.audio_encoder.parameters() if p.requires_grad]
    text_lora_params = [p for p in raw_text_model.encoder.parameters() if p.requires_grad]
    head_params = list(raw_model.pooler.parameters()) + \
                  list(raw_model.projector.parameters())
    
    if args.learn_temp:
        head_params.append(raw_model.logit_scale)

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

        # Optimizer (optional)
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
            mk = getattr(incompat, "missing_keys", [])
            uk = getattr(incompat, "unexpected_keys", [])
            logger.info(f"[RESUME] Loaded model_state_dict from: {args.resume}")
            logger.info(f"[RESUME] start_epoch={start_epoch}, global_step={global_step}")
            logger.info(f"[RESUME] model missing_keys={len(mk)}, unexpected_keys={len(uk)}")
            if len(mk) > 0:
                logger.info(f"[RESUME] model missing_keys (head): {mk[:10]}")
            if text_incompat is not None:
                tmk = getattr(text_incompat, "missing_keys", [])
                tuk = getattr(text_incompat, "unexpected_keys", [])
                logger.info(f"[RESUME] text missing_keys={len(tmk)}, unexpected_keys={len(tuk)}")

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
    
    train_dataset = TermRAGDataset(train_samples, force_dummy_audio=args.force_dummy_audio)
    test_dataset = TermRAGDataset(test_samples, force_dummy_audio=args.force_dummy_audio)
    
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
        if not eval_texts:
            # Keep DDP control flow identical across ranks to avoid training deadlocks.
            if is_main:
                logger.error(
                    f"[EVAL_SKIP] reason=empty_eval_texts is_full_eval={is_full_eval} "
                    f"full_terms={len(glossary_info.get('raw_terms', []))} dev_terms={len(dev_unique_terms)} "
                    f"cache_dir_full={args.precomputed_dir} cache_dir_dev={args.precomputed_dev_dir}"
                )
            if world_size > 1:
                dist.barrier()
            retriever.train()
            text_encoder.train()
            return
        term_to_idx = {t.lower().strip(): i for i, t in enumerate(eval_texts)}

        all_text_embs = []
        eval_batch_size = 256
        with torch.no_grad():
            for i in range(0, len(eval_texts), eval_batch_size):
                batch_texts = eval_texts[i:i+eval_batch_size]
                inputs = text_tokenizer(batch_texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
                # 🟢 使用 raw_text_encoder
                embs = raw_text_encoder(inputs.input_ids, inputs.attention_mask)
                all_text_embs.append(embs.cpu().float().numpy())

        if not all_text_embs:
            # Should not happen because eval_texts is non-empty, but keep it safe anyway.
            if is_main:
                logger.error(f"[EVAL_SKIP] reason=no_text_embs_collected is_full_eval={is_full_eval}")
            if world_size > 1:
                dist.barrier()
            retriever.train()
            text_encoder.train()
            return

        search_embs_np = np.concatenate(all_text_embs, axis=0)
        faiss.normalize_L2(search_embs_np)
        search_index = faiss.IndexFlatIP(search_embs_np.shape[1])
        search_index.add(search_embs_np)

        # 定义评测维度
        K_list = [1, 5, 10]
        tau_list = [0.02, 0.05]
        
        # Strict multi-positive recall: count how many positive terms are retrieved in Top-K.
        # Denominator is the total number of positive terms (unique terms) across chunks.
        pos_hits = {k: 0 for k in K_list}
        margin_lists = {k: [] for k in K_list} # 仅记录命中的样本的 Δ_outK
        # Confident recall is also computed on a per-positive basis:
        # only positives that are actually retrieved in Top-K contribute to confident_hits.
        confident_hits = {k: {tau: 0 for tau in tau_list} for k in K_list}
        
        total_valid_pos = 0
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
                # 搜索 top 100 以获取足够的负样本 (多正例场景)
                D, I = search_index.search(audio_embs_np, 100)
                
                # 按 chunk_id 分组，处理多正例场景
                from collections import defaultdict
                chunk_to_indices = defaultdict(list)
                for i, sample in enumerate(eval_batch["samples"]):
                    uid = str(sample.get("utter_id", ""))
                    cidx = str(sample.get("chunk_idx", ""))
                    chunk_id = f"{uid}::{cidx}"
                    chunk_to_indices[chunk_id].append(i)

                for cid, idxs_in_batch in chunk_to_indices.items():
                    eval_samples_count += 1
                    
                    # 收集该 chunk 的所有正例索引
                    pos_indices = set()
                    for idx in idxs_in_batch:
                        t = eval_batch["samples"][idx].get("gt_term", "").strip().lower()
                        if t and t in term_to_idx:
                            pos_indices.add(term_to_idx[t])
                    
                    if not pos_indices: continue
                    # Count total positives (unique terms) for strict recall denominator
                    total_valid_pos += len(pos_indices)
                    
                    first_idx = idxs_in_batch[0]
                    a_emb_np = audio_embs_np[first_idx]
                    
                    # 1. 获取 s_pos = max(sim of positives)
                    s_pos = -1.0
                    for rank, idx in enumerate(I[first_idx]):
                        if idx in pos_indices:
                            s_pos = max(s_pos, float(D[first_idx][rank]))
                    
                    # 补充检查：如果最高分不在 Top-100，手动算一遍
                    if s_pos < D[first_idx][-1]:
                        for p_idx in pos_indices:
                            s_pos = max(s_pos, float(a_emb_np @ search_embs_np[p_idx]))
                    
                    # 2. 获取纯负例分数列表 (剔除所有正例)
                    neg_scores = []
                    for rank, idx in enumerate(I[first_idx]):
                        if idx not in pos_indices:
                            neg_scores.append(float(D[first_idx][rank]))
                    
                    # 3. 计算各 K 档位指标
                    for k in K_list:
                        # Strict recall@K: count how many positives appear in Top-K
                        topk_set = set(int(x) for x in I[first_idx][:k])
                        pos_in_topk = len(topk_set.intersection(pos_indices))
                        if pos_in_topk > 0:
                            pos_hits[k] += pos_in_topk
                        
                        # Margin_out@K = s_pos - s_neg_(K+1)
                        # neg_scores[k] 即为第 k+1 个负例
                        if len(neg_scores) > k:
                            s_neg_k_plus_1 = neg_scores[k]
                            delta_out_k = s_pos - s_neg_k_plus_1
                            
                            if pos_in_topk > 0:
                                margin_lists[k].append(delta_out_k)
                                # Confident Hit Rate
                                for tau in tau_list:
                                    if delta_out_k >= tau:
                                        confident_hits[k][tau] += pos_in_topk
        
        # 汇总所有卡的命中数、样本数和 Margin 列表
        if world_size > 1:
            # 汇总标量
            stats = [total_valid_pos]
            for k in K_list:
                stats.append(pos_hits[k])
                for tau in tau_list:
                    stats.append(confident_hits[k][tau])
            
            stats_tensor = torch.tensor(stats, device=device, dtype=torch.float32)
            dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
            
            total_valid_pos = int(stats_tensor[0].item())
            cursor = 1
            for k in K_list:
                pos_hits[k] = int(stats_tensor[cursor].item())
                cursor += 1
                for tau in tau_list:
                    confident_hits[k][tau] = int(stats_tensor[cursor].item())
                    cursor += 1
            
            # 汇总 Margin 列表用于百分位数计算
            for k in K_list:
                gathered_margins = [None] * world_size
                dist.all_gather_object(gathered_margins, margin_lists[k])
                margin_lists[k] = [m for sublist in gathered_margins for m in sublist]

        if is_main:
            final_metrics = {}
            current_epoch_recall5 = 0.0
            suffix = "_full" if is_full_eval else "_sampled"
            
            for k in K_list:
                recall = pos_hits[k] / total_valid_pos if total_valid_pos > 0 else 0
                final_metrics[f"eval/recall@{k}{suffix}"] = recall
                if k == 5: current_epoch_recall5 = recall
                
                # Margin 统计 (仅在命中的样本上计算)
                if margin_lists[k]:
                    m_arr = np.array(margin_lists[k])
                    final_metrics[f"eval/margin_p50@{k}{suffix}"] = np.median(m_arr)
                    final_metrics[f"eval/margin_p10@{k}{suffix}"] = np.percentile(m_arr, 10)
                
                # Confident Recall
                for tau in tau_list:
                    c_recall = confident_hits[k][tau] / total_valid_pos if total_valid_pos > 0 else 0
                    final_metrics[f"eval/conf_recall@{k}_tau{tau}{suffix}"] = c_recall
            
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
                s_suffix = "full" if is_full_eval else "sampled"
                actual_save_path = args.save_path.replace(".pt", f"_{s_suffix}_best.pt")
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
                    NULL_TOKEN = "[NO_TERM]"

                    # -------- 1) build local anchors (unique chunks) --------
                    # chunk_id -> set(term)
                    from collections import defaultdict

                    chunk_to_terms = defaultdict(set)
                    chunk_to_first_i = {}  # first occurrence index in local batch

                    for i, s in enumerate(samples):
                        uid = str(s.get("utter_id", ""))
                        cidx = str(s.get("chunk_idx", ""))
                        chunk_id = f"{uid}::{cidx}"

                        t = (s.get("term", "") or "").strip().lower()
                        if not t:
                            t = NULL_TOKEN

                        chunk_to_terms[chunk_id].add(t)
                        if chunk_id not in chunk_to_first_i:
                            chunk_to_first_i[chunk_id] = i

                    local_chunk_ids = list(chunk_to_first_i.keys())
                    anchor_indices = [chunk_to_first_i[cid] for cid in local_chunk_ids]

                    # one audio embedding per chunk
                    anchor_audio_embs = audio_embs[anchor_indices]  # [A, D]
                    A_local = anchor_audio_embs.size(0)

                    # flatten local (chunk, term) pairs for text encoding
                    local_pos_terms = []
                    local_pos_owner = []  # owner anchor index [0..A_local-1]
                    for a, cid in enumerate(local_chunk_ids):
                        for t in sorted(chunk_to_terms[cid]):
                            if t == NULL_TOKEN:
                                continue
                            local_pos_terms.append(t)
                            local_pos_owner.append(a)

                    # IMPORTANT (DDP): all ranks must execute collectives in the same order.
                    # Previously, only rank0 called all_reduce here -> deadlock at step multiples of 50.
                    if global_step % 50 == 0:
                        local_max_terms_per_chunk = max(len(v) for v in chunk_to_terms.values()) if chunk_to_terms else 0
                        global_max_terms_per_chunk = local_max_terms_per_chunk
                        if world_size > 1:
                            t_max = torch.tensor(local_max_terms_per_chunk, device=device, dtype=torch.int32)
                            dist.all_reduce(t_max, op=dist.ReduceOp.MAX)
                            global_max_terms_per_chunk = int(t_max.item())

                        if is_main:
                            logger.info(
                                f"A_local={A_local}, T_local={len(local_pos_terms)}, "
                                f"unique_chunks={len(local_chunk_ids)}, "
                                f"max_terms_per_chunk_local={local_max_terms_per_chunk}, "
                                f"max_terms_per_chunk_global={global_max_terms_per_chunk}"
                            )

                    if not local_pos_terms:
                        # Handle case where rank has no positive terms
                        local_term_embs = torch.zeros((0, 1024), device=device, dtype=torch.bfloat16)
                    else:
                        local_term_embs = encode_text(local_pos_terms)  # [T_local, D]

                    # -------- 2) gather global negatives + metadata with Padding for DDP --------
                    if world_size > 1:
                        # a) 收集每张卡的术语数量，计算全局最大长度
                        t_local = local_term_embs.shape[0]
                        t_locals = [None] * world_size
                        dist.all_gather_object(t_locals, t_local)
                        t_max = max(t_locals)

                        # b) 对 Embedding 进行 Padding 以适配 all_gather [t_max, D]
                        if t_max > t_local:
                            pad_len = t_max - t_local
                            local_term_embs_padded = F.pad(local_term_embs, (0, 0, 0, pad_len))
                        else:
                            local_term_embs_padded = local_term_embs
                        
                        # c) Gather 补齐后的 Embedding [world_size * t_max, D]
                        global_term_embs_padded = all_gather_with_grad(local_term_embs_padded)

                        # d) 收集元数据用于构造 pos_mask
                        owners_list = [None] * world_size
                        dist.all_gather_object(owners_list, local_pos_owner)

                        A_counts = [None] * world_size
                        dist.all_gather_object(A_counts, A_local)
                        prefix = [0]
                        for c in A_counts[:-1]:
                            prefix.append(prefix[-1] + c)

                        rank_id = dist.get_rank()
                        local_global_start = prefix[rank_id]
                        local_global_to_i = {local_global_start + i: i for i in range(A_local)}

                        # e) 构造 pos_mask [A_local, world_size * t_max] 和 col_mask
                        pos_mask = torch.zeros((A_local, world_size * t_max), device=device)
                        col_mask = torch.zeros(world_size * t_max, device=device, dtype=torch.bool)
                        
                        for r, (r_owners, r_t_len) in enumerate(zip(owners_list, t_locals)):
                            start_col = r * t_max
                            # 标记该卡贡献的有效列（非 padding 区域）
                            col_mask[start_col : start_col + r_t_len] = True
                            # 填充正样本索引
                            for k, o in enumerate(r_owners):
                                owner_g = o + prefix[r]
                                if owner_g in local_global_to_i:
                                    pos_mask[local_global_to_i[owner_g], start_col + k] = 1.0
                        
                        global_term_embs = global_term_embs_padded
                    else:
                        global_term_embs = local_term_embs
                        pos_mask = torch.zeros((A_local, local_term_embs.shape[0]), device=device)
                        for k, o in enumerate(local_pos_owner):
                            pos_mask[o, k] = 1.0
                        col_mask = torch.ones(local_term_embs.shape[0], device=device, dtype=torch.bool)

                    # -------- 3) sim + masking --------
                    if global_term_embs.size(0) == 0:
                        continue

                    sim_matrix = (anchor_audio_embs @ global_term_embs.T) * logit_scale  # [A_local, world_size * t_max]

                    # 屏蔽 Padding 列，使其不参与 logsumexp
                    sim_matrix.masked_fill_(~col_mask.unsqueeze(0), -1e9)

                    # -------- 4) Symmetric multi-positive InfoNCE --------

                    # Audio-to-Text (only for anchors that have at least one term)
                    log_prob_a2t = sim_matrix - torch.logsumexp(sim_matrix, dim=1, keepdim=True)
                    row_mask = (pos_mask.sum(dim=1) > 0).float()
                    loss_a2t = - (log_prob_a2t * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
                    loss_a2t = (loss_a2t * row_mask).sum() / row_mask.sum().clamp(min=1)

                    # Text-to-Audio (all audio anchors are negatives, including NO_TERM ones)
                    log_prob_t2a = sim_matrix.T - torch.logsumexp(sim_matrix.T, dim=1, keepdim=True)
                    # 只有有效列（非 padding 文本）计算 T2A loss
                    loss_t2a_all = - (log_prob_t2a * pos_mask.T).sum(dim=1) / pos_mask.T.sum(dim=1).clamp(min=1)
                    loss_t2a = (loss_t2a_all * col_mask.float()).sum() / col_mask.float().sum().clamp(min=1)

                    loss_term = (loss_a2t + loss_t2a) / 2
                    total_loss += loss_term * args.term_weight
                    loss_term_val = loss_term.item()

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(retriever.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if is_main:
                pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})
                
                # 计算当前真实的 logit_scale 和 temperature
                current_logit_scale = logit_scale.item()
                current_temp = 1.0 / current_logit_scale if current_logit_scale != 0 else 1.0
                
                wandb.log({
                    "train/loss": total_loss.item(), 
                    "train/loss_term": loss_term_val,
                    "train/loss_trans": loss_trans_val,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/logit_scale": current_logit_scale,
                    "train/temperature": current_temp
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
    parser.add_argument("--term_weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--learn_temp", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="qwen3_rag")
    parser.add_argument("--wandb_exp_name", type=str, default="unfrozen_text_encoder")
    parser.add_argument("--force_dummy_audio", action="store_true", default=False, help="Debug mode: bypass audio IO and use dummy audio for all samples.")
    args = parser.parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("LOCAL_RANK", 0))
    train(rank, world_size, args)

if __name__ == "__main__":
    main()

