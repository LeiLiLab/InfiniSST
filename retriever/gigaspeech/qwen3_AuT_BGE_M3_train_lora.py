#!/usr/bin/env python3
"""
Qwen3-Omni + BGE-M3 (Precomputed) Training Script
Audio Encoder: Qwen3OmniMoeAudioEncoder (Frozen)
Text Anchor: BGE-M3 (Precomputed Memmap)
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
from transformers import WhisperFeatureExtractor, get_cosine_schedule_with_warmup
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

class Qwen3OmniRetriever(nn.Module):
    def __init__(self, model_id="Atotti/Qwen3-Omni-AudioTransformer", target_dim=1024, 
                 use_lora=True, lora_rank=32, lora_alpha=64, lora_target_modules=None):
        super().__init__()
        
        # 1. 加载 Encoder (BF16)
        self.audio_encoder = Qwen3OmniMoeAudioEncoder.from_pretrained(
            model_id, 
            dtype=torch.bfloat16 # 修复弃用警告: torch_dtype -> dtype
        )
        
        if hasattr(self.audio_encoder, "conv2d1"):
            self.audio_encoder.get_input_embeddings = lambda: self.audio_encoder.conv2d1
            
        self.audio_encoder.gradient_checkpointing_enable()
        
        # 2. 关键：应用 LoRA
        if use_lora:
            # 如果没有传入 modules，则使用默认的全量配置
            if lora_target_modules is None:
                lora_target_modules = [
                    # 1. Attention 部分
                    "q_proj", "k_proj", "v_proj", "out_proj", 
                    
                    # 2. FFN 部分 (你的模型用的是 fc1/fc2，不是 gate/up/down)
                    "fc1", "fc2",
                    
                    # 3. 关键：输出投影层 (Feature Adapter)
                    # 这两层直接决定了 2048 维输出的语义，加上 LoRA 收益极高
                    "proj1", "proj2" 
                ]
            
            # 定义 LoRA 配置
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type=None
            )
            
            # 这一步会自动冻结原模型参数，只把 LoRA 设为 trainable
            self.audio_encoder = get_peft_model(self.audio_encoder, lora_config)
            
            # 打印可训练参数量，确认配置生效
            self.audio_encoder.print_trainable_parameters()
            
        else:
            # 如果不用的 LoRA，则冻结全部 (Linear Probe 模式)
            for param in self.audio_encoder.parameters():
                param.requires_grad = False

        # 3. 投影层和 Pooler (始终全量训练)
        self.pooler = AttentivePooling(2048) 
        self.projector = nn.Linear(2048, target_dim)
        
        # Logit Scale
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, input_features, feature_lens):
        # 注意：Qwen3OmniMoeAudioEncoder 期待的是 "packed" 格式 [128, Total_T]
        outputs = self.audio_encoder(input_features, feature_lens)
        hidden_states = outputs.last_hidden_state # [Total_Output_T, 2048]
        
        # 关键修复：将 Packed 2D 输出还原为 Batch 3D
        if hidden_states.ndim == 2:
            # 1. 计算每个样本经过下采样后的输出长度
            # Qwen3-Omni 经过 3 层 stride=2 卷积，公式为 ceil(L / 8)
            output_lens = []
            for l in feature_lens.tolist():
                curr_l = l
                for _ in range(3):
                    curr_l = (curr_l + 1) // 2 # 相当于 ceil(curr_l / 2)
                output_lens.append(curr_l)
            
            # 2. 验证总长度是否对齐，若不对齐则回退到比例计算（容错逻辑）
            if sum(output_lens) != hidden_states.shape[0]:
                ratio = input_features.shape[1] / hidden_states.shape[0]
                output_lens = [max(1, round(l / ratio)) for l in feature_lens.tolist()]
                # 强行对齐最后一个 chunk 的长度
                output_lens[-1] = hidden_states.shape[0] - sum(output_lens[:-1])

            # 3. 切分并还原为 [B, max_len, 2048]
            hidden_states_list = torch.split(hidden_states, output_lens, dim=0)
            from torch.nn.utils.rnn import pad_sequence
            hidden_states = pad_sequence(hidden_states_list, batch_first=True)
            
            # 4. 更新 feature_lens 为下采样后的真实长度，用于后续 Mask
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
            # 基础归一化
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # 将原始 sample 信息透传给 collate_fn
            res = {k: v for k, v in sample.items()}
            res["audio"] = audio_data.astype(np.float32)
            return res
            
        except Exception as e:
            logger.warning(f"[SKIP] Audio load error: {audio_path} | Error: {e}")
            return {"audio": None, "chunk_audio_path": audio_path}

def collate_fn(batch, feature_extractor):
    # 1. 过滤掉损坏的数据
    valid_samples = []
    for s in batch:
        if s is not None and s.get("audio") is not None and len(s["audio"]) > 3000:
            valid_samples.append(s)
    
    if not valid_samples: return None

    # 2. 统一填充音频到标准长度 (30720 samples = 1.92s)
    target_len = 30720 
    audios = []
    for s in valid_samples:
        audio = s["audio"]
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')
        elif len(audio) > target_len:
            audio = audio[:target_len]
        audios.append(audio)

    # 3. 批量提取特征
    try:
        inputs = feature_extractor(audios, sampling_rate=16000, return_tensors="pt", padding=False)
        # input_features 形状是 [B, 80, T_mel]
        features = inputs.input_features
        B, C, T_mel = features.shape
        
        # 转换为 Packed 格式 [C, B*T_mel]
        input_features = features.transpose(0, 1).reshape(C, -1)
        
        # 关键修正：在音频层面我们已经手动 padding 过了，且 Qwen3 的 Packed 模式
        # 要求 split_sizes 的总和必须完全等于输入 Tensor 的长度。
        # 因此这里我们直接传固定的 T_mel，不再手动计算变长，避免浮点数精度或步长导致的 split 错误。
        feature_lens = torch.full((B,), T_mel, dtype=torch.long)
        
    except Exception as e:
        logger.error(f"[CRITICAL] Batch extraction failed after padding: {e}")
        return None
    
    return {
        "input_features": input_features,
        "feature_lens": feature_lens,
        "line_indices": [s["line_idx"] for s in valid_samples],
        "samples": valid_samples 
    }

# ==================== Evaluation Helpers ====================

def get_glossary_info(jsonl_path, cache_dir):
    """
    构造唯一 Term 列表及其对应的 line_idx，用于评测时从 mmap 提取向量。
    """
    cache_path = os.path.join(cache_dir, "glossary_info.pkl")
    if os.path.exists(cache_path):
        logger.info(f"Loading glossary info from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    logger.info(f"Building glossary info from {jsonl_path}...")
    term_to_info = {} # term -> {"line_idx": int, "translations": set}
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(tqdm(f, desc="Scanning for glossary")):
            try:
                item = json.loads(line)
                term = item.get("term", "").strip().lower()
                if not term: continue
                
                trans = item.get("translation", "").strip()
                if term not in term_to_info:
                    term_to_info[term] = {"line_idx": line_idx, "translations": set()}
                if trans:
                    term_to_info[term]["translations"].add(trans)
            except: continue
            
    unique_terms = []
    unique_indices = []
    for term, info in term_to_info.items():
        # 合并翻译
        merged_trans = ", ".join(sorted(list(info["translations"])))
        unique_terms.append(f"{term} ({merged_trans})")
        unique_indices.append(info["line_idx"])
        
    result = {
        "terms": unique_terms,
        "indices": unique_indices,
        "raw_terms": list(term_to_info.keys())
    }
    
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    logger.info(f"Glossary info saved. Unique terms: {len(unique_terms)}")
    return result

def get_dev_glossary_info(jsonl_path, cache_dir):
    """
    构造测试集唯一词库信息并缓存
    """
    cache_path = os.path.join(cache_dir, "dev_glossary_info.pkl")
    if os.path.exists(cache_path):
        logger.info(f"Loading dev glossary info from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    logger.info(f"Building dev glossary from {jsonl_path}...")
    dev_term_to_idx = {} # term -> first_line_idx_in_dev
    NULL_TOKEN = "[NO_TERM]"
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            try:
                t = json.loads(line).get("term", "").strip().lower()
                if not t: t = NULL_TOKEN
                if t not in dev_term_to_idx:
                    dev_term_to_idx[t] = line_idx
            except: continue
            
    result = {
        "unique_terms": list(dev_term_to_idx.keys()),
        "indices": list(dev_term_to_idx.values())
    }
    
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    logger.info(f"Dev glossary info saved. Unique terms: {len(result['unique_terms'])}")
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
        lora_target_modules=args.lora_target_modules
    ).to(device)
    
    if is_main:
        logger.info("======= Model Structure =======")
        print(retriever)
        logger.info("===============================")
    
    if world_size > 1:
        retriever = DDP(retriever, device_ids=[rank])
    
    # 2. 加载预计算的 memmaps (直接读，不占内存)
    logger.info(f"Mapping precomputed embeddings from {args.precomputed_dir}...")
    train_meta = json.load(open(os.path.join(args.precomputed_dir, "meta.json")))
    train_total = train_meta["num_samples"]
    term_mmap = np.memmap(os.path.join(args.precomputed_dir, "terms.mmap"), dtype='float32', mode='r', shape=(train_total, 1024))
    trans_mmap = np.memmap(os.path.join(args.precomputed_dir, "trans.mmap"), dtype='float32', mode='r', shape=(train_total, 1024))
    
    # 2.1 加载 Dev 集预计算 Embeddings
    logger.info(f"Mapping precomputed dev embeddings from {args.precomputed_dev_dir}...")
    dev_meta = json.load(open(os.path.join(args.precomputed_dev_dir, "meta.json")))
    dev_total = dev_meta["num_samples"]
    dev_term_mmap = np.memmap(os.path.join(args.precomputed_dev_dir, "terms.mmap"), dtype='float32', mode='r', shape=(dev_total, 1024))

    # 3. 准备 Glossary 用于评测 (仅 main process)
    glossary_info = None
    dev_glossary_index = None
    dev_unique_terms = []
    
    if is_main:
        # 3.1 训练集全量词库 (50万级)
        glossary_info = get_glossary_info(args.train_jsonl, args.precomputed_dir)
        
        # 3.2 测试集独立词库 (数千级，用于快速 Sample Eval)
        dev_info = get_dev_glossary_info(args.dev_jsonl, args.precomputed_dev_dir)
        dev_unique_terms = dev_info["unique_terms"]
        dev_indices = dev_info["indices"]
        
        # 构建一个常驻内存的小型 FAISS 索引
        import faiss
        dev_glossary_embs = dev_term_mmap[dev_indices].copy().astype('float32')
        faiss.normalize_L2(dev_glossary_embs)
        dev_glossary_index = faiss.IndexFlatIP(dev_glossary_embs.shape[1])
        dev_glossary_index.add(dev_glossary_embs)
        logger.info(f"Dev glossary index built.")

    # 3. Optimizer & Data
    # 1) 获取裸模型引用 (处理 DDP .module 包装)
    raw_model = retriever.module if world_size > 1 else retriever

    # 2) 收集参数分组
    # Group 1: LoRA 参数 (Audio Encoder 中所有需要梯度的参数)
    # 注意: get_peft_model 已经自动冻结了非 LoRA 参数，这里再用 requires_grad 过滤一道双重保险
    lora_params = [p for p in raw_model.audio_encoder.parameters() if p.requires_grad]

    # Group 2: Head 参数 (Projector, Pooler, Logit Scale) - 这些是随机初始化的
    head_params = list(raw_model.pooler.parameters()) + \
                  list(raw_model.projector.parameters()) + \
                  [raw_model.logit_scale]

    # 3) 定义优化器分组
    optimizer_grouped_parameters = []
    
    # 只有当 LoRA 开启且有可训练参数时才加入该组
    if len(lora_params) > 0:
        optimizer_grouped_parameters.append({
            "params": lora_params, 
            "lr": args.lr,
            "name": "lora_params"
        })
    
    # Head 参数组始终加入
    optimizer_grouped_parameters.append({
        "params": head_params, 
        "lr": args.lr * 10,
        "name": "head_params"
    })

    # 5. Optimizer & Scaler
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")

    # 6. Resume Logic
    start_epoch = 0
    global_step = 0
    best_recall5_sampled = 0.0
    best_recall5_full = 0.0
    
    if args.resume and os.path.exists(args.resume):
        if is_main:
            logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        state_dict = checkpoint["model_state_dict"]
        # 自动处理 DDP 前缀不匹配问题
        new_state_dict = {}
        has_module = any(k.startswith("module.") for k in state_dict.keys())
        is_ddp = world_size > 1
        
        for k, v in state_dict.items():
            if is_ddp and not has_module:
                new_state_dict[f"module.{k}"] = v
            elif not is_ddp and has_module:
                new_state_dict[k.replace("module.", "")] = v
            else:
                new_state_dict[k] = v
                
        retriever.load_state_dict(new_state_dict, strict=False)
        
        # 尝试加载优化器和调度器状态
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if is_main: logger.info("Loaded optimizer_state_dict")
            except Exception as e:
                if is_main: logger.warning(f"Could not load optimizer_state_dict: {e}")

        start_epoch = checkpoint.get("epoch", -1) + 1
        global_step = checkpoint.get("global_step", 0)
        best_recall5_sampled = checkpoint.get("best_recall5_sampled", 0.0)
        best_recall5_full = checkpoint.get("best_recall5_full", 0.0)
        
        if is_main:
            logger.info(f"Resumed from epoch {start_epoch}, step {global_step}")
            logger.info(f"Previous Best - Sampled: {best_recall5_sampled:.2%}, Full: {best_recall5_full:.2%}")

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")

    # 7. 分别加载训练集和测试集 (不再使用愚蠢的内存切分)
    logger.info(f"Loading train samples from {args.train_jsonl}...")
    # ==================== Efficient Data Loading ====================
    import time
    time.sleep(rank * 1.5) # 错峰加载，避免 8 个进程抢占磁盘 IO
    
    if is_main: logger.info(f"Loading samples...")
    train_samples = []
    with open(args.train_jsonl, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if args.test_limit and line_idx >= args.test_limit: break
            try:
                sample = json.loads(line.strip())
                # 移除 os.path.exists 检查，由 DataLoader 后续按需捕获
                sample["line_idx"] = line_idx
                train_samples.append(sample)
            except: continue
            
    test_samples = []
    with open(args.dev_jsonl, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            # dev 集一般较小，且不受限于 test_limit
            try:
                sample = json.loads(line.strip())
                sample["line_idx"] = line_idx
                sample["gt_term"] = sample.get("term", "").lower()
                test_samples.append(sample)
            except: continue
    
    if is_main:
        logger.info(f"Train samples: {len(train_samples)}")
        logger.info(f"Dev samples: {len(test_samples)}")

    train_dataset = TermRAGDataset(train_samples)
    test_dataset = TermRAGDataset(test_samples)
    
    train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size // world_size,
        sampler=train_sampler, 
        shuffle=(train_sampler is None),
        collate_fn=lambda b: collate_fn(b, feature_extractor),
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size // world_size,
        shuffle=False, collate_fn=lambda b: collate_fn(b, feature_extractor),
        num_workers=args.num_workers, pin_memory=True
    )

    # 8. Scheduler (Cosine with Warmup)
    # 计算总步数
    num_training_steps = len(train_loader) * args.epochs
    # 预热步数
    num_warmup_steps = int(num_training_steps * 0.1)
    
    # 关键修复：补全 initial_lr，必须尊重不同参数组的倍数关系
    # 当 last_epoch > -1 时，调度器要求 param_groups 必须有 initial_lr
    for group in optimizer.param_groups:
        if "initial_lr" not in group:
            name = group.get("name", "")
            if name == "head_params":
                group["initial_lr"] = args.lr * 10
            else:
                # 默认为基础学习率 (针对 lora_params 或未命名的组)
                group["initial_lr"] = args.lr
            
            if is_main:
                logger.info(f"Supplemented initial_lr for group '{name}': {group['initial_lr']}")

    # 初始化 scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        last_epoch=global_step - 1 if global_step > 0 else -1
    )

    if is_main:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_exp_name, config=vars(args))

    for epoch in range(start_epoch, args.epochs):
        retriever.train()
        if train_sampler: train_sampler.set_epoch(epoch)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}") if is_main else train_loader
        for batch in pbar:
            if batch is None: continue
            global_step += 1
            
            # 每 N 步保存一次 pt (与 best 隔离)
            if args.save_steps > 0 and global_step % args.save_steps == 0 and is_main:
                step_save_path = args.save_path.replace(".pt", f"_step_{global_step}.pt")
                logger.info(f"Saving checkpoint at step {global_step} to {step_save_path}")
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': retriever.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_recall5_sampled': best_recall5_sampled,
                    'best_recall5_full': best_recall5_full,
                }, step_save_path)
            
            # Packed 格式：input_features 是 [128, Total_T]
            input_features = batch["input_features"].to(device).to(torch.bfloat16)
            feature_lens = batch["feature_lens"].to(device)
            line_indices = batch["line_indices"]
            
            # 从 memmap 极速读取预计算好的向量
            # copy() 是为了将内存数据转为普通 numpy array，避免多进程共享冲突
            term_embs = torch.from_numpy(term_mmap[line_indices].copy()).to(device).to(torch.bfloat16)
            trans_embs = torch.from_numpy(trans_mmap[line_indices].copy()).to(device).to(torch.bfloat16)
            
            # 修复弃用警告: torch.cuda.amp.autocast -> torch.amp.autocast
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                audio_embs = retriever(input_features, feature_lens)
                
                # Contrastive Loss
                logit_scale = retriever.module.logit_scale.exp() if world_size > 1 else retriever.logit_scale.exp()
                labels = torch.arange(len(audio_embs), device=device)
                
                total_loss = 0
                loss_term_val = 0
                loss_trans_val = 0

                # Audio-Term Loss
                if args.term_weight > 0:
                    sim_term = (audio_embs @ term_embs.T) * logit_scale
                    loss_term = F.cross_entropy(sim_term, labels)
                    total_loss += loss_term * args.term_weight
                    loss_term_val = loss_term.item()
                
                # Audio-Transcript Loss
                if args.trans_weight > 0:
                    sim_trans = (audio_embs @ trans_embs.T) * logit_scale
                    loss_trans = F.cross_entropy(sim_trans, labels)
                    total_loss += loss_trans * args.trans_weight
                    loss_trans_val = loss_trans.item()

                # Anomaly Detection (Research mode)
                if total_loss.item() > args.anomaly_threshold and is_main:
                    with torch.no_grad():
                        # 计算单样本 Loss 来定位“毒样本”
                        per_sample_loss = F.cross_entropy(
                            (audio_embs @ term_embs.T) * logit_scale if args.term_weight > 0 else (audio_embs @ trans_embs.T) * logit_scale,
                            labels, reduction='none'
                        )
                        max_loss, max_idx = torch.max(per_sample_loss, dim=0)
                        bad_sample = batch["samples"][max_idx.item()]
                        
                        logger.warning(f"⚠️ [ANOMALY] High Loss Detect: {total_loss.item():.2f} | "
                                       f"Max Sample Loss: {max_loss.item():.2f}")
                        logger.warning(f"Details: Path={bad_sample.get('chunk_audio_path')}, "
                                       f"Term='{bad_sample.get('term')}', Trans='{bad_sample.get('translation')}'")
                        
                        # 写入文件供后续数据清洗
                        with open("training_anomalies.jsonl", "a") as af:
                            # bad_sample contains "audio" (np.ndarray) which is not JSON-serializable.
                            # Keep only JSON-safe metadata (drop raw audio).
                            clean_metadata = {k: v for k, v in bad_sample.items() if k != "audio"}
                            anomaly_record = {
                                "global_step": global_step,
                                "batch_avg_loss": total_loss.item(),
                                "sample_loss": max_loss.item(),
                                "metadata": clean_metadata
                            }
                            af.write(json.dumps(anomaly_record, ensure_ascii=False) + "\n")

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            
            # Gradient Clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(retriever.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if is_main:
                pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})
                
                # 构造学习率日志，支持多参数组监控
                lr_logs = {}
                for group in optimizer.param_groups:
                    name = group.get("name", "unknown")
                    lr_logs[f"train/lr_{name}"] = group["lr"]

                wandb.log({
                    "train/loss": total_loss.item(), 
                    "train/audio_term_loss": loss_term_val,
                    "train/audio_trans_loss": loss_trans_val,
                    **lr_logs
                })

        # ==================== Evaluation ====================
        if is_main:
            # 每 1 个 epoch 做一次采样评测 (In-batch 逻辑)，每 10 个 epoch 做一次全量 Glossary 评测
            is_full_eval = (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs
            logger.info(f"Starting evaluation (Full Glossary: {is_full_eval})...")
            
            import faiss
            retriever.eval()
            
            # 准备搜索库
            if is_full_eval:
                # 全量 50万 Term 搜索
                glossary_indices = glossary_info['indices']
                search_embs = term_mmap[glossary_indices].copy().astype('float32')
                faiss.normalize_L2(search_embs)
                search_index = faiss.IndexFlatIP(search_embs.shape[1])
                search_index.add(search_embs)
            else:
                # 简单逻辑：仅针对测试集采样出的 Batch 进行对比
                # 我们会在下面的循环中动态处理
                search_index = None

            recall_results = {5: [], 10: [], 20: []}
            eval_samples_count = 0
            max_eval_samples = 1000 # 评测采样 1000 条音频
            
            with torch.no_grad():
                for eval_batch in tqdm(test_loader, desc="Evaluating Recall"):
                    if eval_batch is None: continue
                    if eval_samples_count >= max_eval_samples: break
                    
                    input_features = eval_batch["input_features"].to(device).to(torch.bfloat16)
                    feature_lens = eval_batch["feature_lens"].to(device)
                    batch_samples = eval_batch["samples"]
                    
                    # 获取音频特征
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        audio_embs = retriever(input_features, feature_lens)
                    
                    audio_embs_np = audio_embs.cpu().float().numpy()
                    faiss.normalize_L2(audio_embs_np)
                    
                    for i, sample in enumerate(batch_samples):
                        gt_term = sample.get("gt_term", "").strip().lower()
                        if not gt_term: gt_term = "[NO_TERM]" # 与 NULL_TOKEN 对齐
                        
                        if is_full_eval:
                            # 1. 全量 50万 词库搜索
                            D, I = search_index.search(audio_embs_np[i:i+1], 20)
                            retrieved_terms = [glossary_info['raw_terms'][idx] for idx in I[0]]
                            for k in recall_results.keys():
                                recall_results[k].append(1.0 if gt_term in [t.lower() for t in retrieved_terms[:k]] else 0.0)
                        else:
                            # 2. 采样评测：针对 Dev 集特有的词库搜索 (数千个词)
                            # 这比 In-batch 难得多，但比 Full 快得多
                            D, I = dev_glossary_index.search(audio_embs_np[i:i+1], 20)
                            retrieved_terms = [dev_unique_terms[idx] for idx in I[0]]
                            for k in recall_results.keys():
                                recall_results[k].append(1.0 if gt_term in retrieved_terms[:k] else 0.0)

                        eval_samples_count += 1
            
            # 计算并记录平均 Recall
            final_metrics = {}
            current_epoch_recall5 = 0.0
            for k, hits in recall_results.items():
                avg_recall = sum(hits) / len(hits) if hits else 0
                metric_name = f"eval/recall@{k}" + ("_full" if is_full_eval else "_sampled")
                final_metrics[metric_name] = avg_recall
                logger.info(f"{metric_name}: {avg_recall:.2%}")
                if k == 5:
                    current_epoch_recall5 = avg_recall
            
            wandb.log(final_metrics)

            # 只有在各自分类的 Recall@5 提升时才保存
            improved = False
            best_ref = ""
            
            if is_full_eval:
                if current_epoch_recall5 > best_recall5_full:
                    best_recall5_full = current_epoch_recall5
                    improved = True
                    best_ref = "full"
            else:
                if current_epoch_recall5 > best_recall5_sampled:
                    best_recall5_sampled = current_epoch_recall5
                    improved = True
                    best_ref = "sampled"

            if improved:
                logger.info(f"New best Recall@5 ({best_ref}): {current_epoch_recall5:.2%}, saving checkpoint...")
                
                # 构造文件名：直接追加 best 类型后缀
                actual_save_path = args.save_path.replace(".pt", f"_{best_ref}_best.pt")
                
                save_data = {
                    "model_state_dict": retriever.module.state_dict() if world_size > 1 else retriever.state_dict(),
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--dev_jsonl", type=str, required=True, help="Path to dev JSONL file")
    parser.add_argument("--precomputed_dir", type=str, required=True, help="Directory with .mmap files for train")
    parser.add_argument("--precomputed_dev_dir", type=str, required=True, help="Directory with .mmap files for dev")
    parser.add_argument("--save_path", type=str, default="qwen3_retriever.pt")
    parser.add_argument("--resume", type=str, default=None, help="Path to .pt file to resume from")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use_lora", action="store_true", default=False, help="Whether to use LoRA for audio encoder")
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=None, 
                        help="List of target modules for LoRA. e.g. q_proj v_proj proj1 proj2")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--test_limit", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--term_weight", type=float, default=1.0, help="Weight for Audio-Term loss")
    parser.add_argument("--trans_weight", type=float, default=0.0, help="Weight for Audio-Transcript loss")
    parser.add_argument("--wandb_project", type=str, default="qwen3_rag")
    parser.add_argument("--wandb_exp_name", type=str, default="precomputed_linear_probe")
    parser.add_argument("--anomaly_threshold", type=float, default=7.0, help="Log samples if loss exceeds this value")
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("LOCAL_RANK", 0))
    train(rank, world_size, args)

if __name__ == "__main__":
    main()
