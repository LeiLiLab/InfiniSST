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
import webdataset as wds
from torch.utils.data.distributed import DistributedSampler
import soundfile as sf
from tqdm import tqdm
from transformers import WhisperFeatureExtractor, AutoTokenizer, AutoModel
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

class BgeM3TextEncoder(nn.Module):
    def __init__(self, model_id="BAAI/bge-m3", lora_rank=16):
        super().__init__()
        # 🟢 极其重要：BGE-M3 在 Dense Retrieval 时不使用预训练的 Pooler 层，
        # 而是直接取 last_hidden_state 的 CLS Token。这里通过 add_pooling_layer=False 显式移除它。
        self.encoder = AutoModel.from_pretrained(
            model_id, 
            dtype=torch.bfloat16, # 修复弃用警告: torch_dtype -> dtype
            add_pooling_layer=False,
            trust_remote_code=True
        )
        
        # BGE-M3 (XLM-Roberta) target modules
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            target_modules=["query", "key", "value"],
            lora_dropout=0.05,
            bias="none",
            task_type=None
        )
        self.encoder = get_peft_model(self.encoder, lora_config)
        self.encoder.print_trainable_parameters()
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # 🟢 核心：始终只取 last_hidden_state 的第 0 位 (CLS Token) 做为特征向量
        # 不要取 outputs.pooler_output (即 Pooler 层输出)，那会导致检索效果大幅下降
        embeddings = outputs.last_hidden_state[:, 0, :]
        return F.normalize(embeddings, p=2, dim=-1)

class Qwen3OmniRetriever(nn.Module):
    def __init__(self, model_id="Atotti/Qwen3-Omni-AudioTransformer", target_dim=1024, use_lora=True):
        super().__init__()
        
        # 1. 加载 Encoder (BF16)
        # 🟢 关键修复：显式指定 attn_implementation="eager" 或 "flash_attention_2"
        # 防止不同卡由于自动探测结果不同（如卡 A 命中 FA2，卡 B 没命中）导致模型结构不一致
        self.audio_encoder = Qwen3OmniMoeAudioEncoder.from_pretrained(
            model_id, 
            dtype=torch.bfloat16, 
            trust_remote_code=True,
            attn_implementation="eager" # 强制使用 eager 以确保各 rank 结构绝对一致
        )
        
        if hasattr(self.audio_encoder, "conv2d1"):
            self.audio_encoder.get_input_embeddings = lambda: self.audio_encoder.conv2d1
            
        self.audio_encoder.gradient_checkpointing_enable()
        
        # 2. 关键：应用 LoRA
        if use_lora:
            # 定义 LoRA 配置
            lora_config = LoraConfig(
                r=32,
                lora_alpha=64,
                # 核心修改：根据你的日志匹配层名称
                target_modules=[
                    # 1. Attention 部分
                    "q_proj", "k_proj", "v_proj", "out_proj", 
                    
                    # 2. FFN 部分 (你的模型用的是 fc1/fc2，不是 gate/up/down)
                    "fc1", "fc2",
                    
                    # 3. 关键：输出投影层 (Feature Adapter)
                    # 这两层直接决定了 2048 维输出的语义，加上 LoRA 收益极高
                    "proj1", "proj2" 
                ],
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
    # Support two modes:
    # - Online mode: sample["audio"] exists (np.ndarray)
    # - Precomputed mode: sample["fbank"] exists (np.ndarray, shape [80, T])
    valid_samples = []
    use_fbank = False
    for s in batch:
        if s is None:
            continue
        if s.get("fbank") is not None:
            valid_samples.append(s)
            use_fbank = True
        elif s.get("audio") is not None and len(s["audio"]) > 3000:
            valid_samples.append(s)

    if not valid_samples:
        return None

    if use_fbank:
        # Precomputed mel/features path: stack -> packed
        fbanks = []
        for s in valid_samples:
            f = s["fbank"]
            if f is None:
                continue
            # expected [80, T]
            if f.ndim == 3 and f.shape[0] == 1:
                f = f[0]
            if f.ndim != 2:
                continue
            fbanks.append(f.astype(np.float32, copy=False))
        if not fbanks:
            return None

        features = torch.from_numpy(np.stack(fbanks, axis=0))  # [B, 80, T]
        B, C, T_mel = features.shape

        # Packed format [C, B*T]
        input_features = features.transpose(0, 1).reshape(C, -1)
        feature_lens = torch.full((B,), T_mel, dtype=torch.long)

        return {
            "input_features": input_features,
            "feature_lens": feature_lens,
            "line_indices": [s["line_idx"] for s in valid_samples],
            "samples": valid_samples,
        }

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

# ==================== WebDataset Loader ====================

def get_wds_loader(shards_path, batch_size, feature_extractor, is_train=True, num_workers=8):
    """
    构造 WebDataset 加载器，修复 DataPipeline 属性错误
    """
    
    def transform_sample(sample):
        # Expect either:
        # - fbank+json (new packing)
        # - wav+json (legacy)
        fbank = sample.get("fbank")
        wav = sample.get("wav")
        meta = sample.get("json")

        if isinstance(meta, bytes):
            meta = json.loads(meta.decode("utf-8", errors="ignore"))
        if meta is None:
            meta = {}
        
        res = {k: v for k, v in meta.items()} if isinstance(meta, dict) else {}
        
        # 2. 核心修复：将 global_idx 映射为 line_idx，供 collate_fn 使用
        if "global_idx" in res:
            res["line_idx"] = res["global_idx"]
        elif "line_idx" not in res:
            # 兜底逻辑：如果是旧数据没有这个字段，尝试从 WebDataset 自动生成的 __key__ 获取
            try:
                res["line_idx"] = int(sample["__key__"])
            except:
                pass

        # Prefer precomputed fbank if present
        if isinstance(fbank, bytes):
            import io
            fb = np.load(io.BytesIO(fbank), allow_pickle=False)
            res["fbank"] = fb
            res["audio"] = None
            return res

        # Legacy wav path (bytes or tensor)
        audio = wav
        if isinstance(audio, bytes):
            import io
            audio, sr = sf.read(io.BytesIO(audio))
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        if isinstance(audio, np.ndarray):
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            res["audio"] = audio.astype(np.float32)
        else:
            res["audio"] = None

        res["fbank"] = None
        return res

    if is_train:
        # 训练模式：ResampledShards -> tar -> shuffle -> decode -> map
        dataset = wds.DataPipeline(
            wds.ResampledShards(shards_path),
            wds.tarfile_to_samples(),
            wds.shuffle(5000),
            wds.map(transform_sample)
        )
    else:
        # 验证模式：SimpleShardList -> split -> tar -> decode -> map
        dataset = wds.DataPipeline(
            wds.SimpleShardList(shards_path),
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.map(transform_sample)
        )

    loader = wds.WebLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, feature_extractor)
    )
    
    if is_train:
        # 设置每个 epoch 的虚拟迭代次数 (830万 / 总 batch_size)
        steps_per_epoch = max(1, 8300000 // batch_size)
        loader = loader.with_epoch(steps_per_epoch)
        # Store for tqdm(total=...) since WebLoader is an Iterable and usually has no __len__
        loader.steps_per_epoch = steps_per_epoch
        
    return loader

# ==================== Training Logic ====================

def train(rank, world_size, args):
    if world_size > 1:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    is_main = rank == 0

    # 1. Models
    logger.info(f"[Rank {rank}] Initializing retriever...")
    retriever = Qwen3OmniRetriever(use_lora=args.use_lora).to(device)
    logger.info(f"[Rank {rank}] Retriever initialized.")
    
    logger.info(f"[Rank {rank}] Initializing text_encoder...")
    text_encoder = BgeM3TextEncoder(lora_rank=args.text_lora_rank).to(device)
    logger.info(f"[Rank {rank}] Text_encoder initialized.")
    
    logger.info(f"[Rank {rank}] Loading tokenizer...")
    text_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    logger.info(f"[Rank {rank}] Tokenizer loaded.")
    
    if is_main:
        logger.info("======= Model Structure =======")
        print(retriever)
        print(text_encoder)
        logger.info("===============================")
    
    # 🟢 调试：在 DDP 之前检查所有可训练参数的形状，防止 Rank 间不一致导致挂死
    logger.info(f"[Rank {rank}] Checking trainable parameters...")
    retriever_trainable = [(n, p.shape) for n, p in retriever.named_parameters() if p.requires_grad]
    for i, (name, shape) in enumerate(retriever_trainable):
        logger.info(f"[Rank {rank}] Retriever Param[{i}]: {name} | {list(shape)}")
    
    text_trainable = [(n, p.shape) for n, p in text_encoder.named_parameters() if p.requires_grad]
    for i, (name, shape) in enumerate(text_trainable):
        logger.info(f"[Rank {rank}] TextEncoder Param[{i}]: {name} | {list(shape)}")

    if world_size > 1:
        logger.info(f"[Rank {rank}] Syncing before DDP...")
        dist.barrier()
        logger.info(f"[Rank {rank}] Wrapping retriever in DDP...")
        try:
            retriever = DDP(retriever, device_ids=[rank])
        except Exception as e:
            logger.error(f"[Rank {rank}] DDP Wrapping failed for retriever: {e}")
            raise e
        logger.info(f"[Rank {rank}] Retriever DDP wrapped.")
        
        dist.barrier()
        logger.info(f"[Rank {rank}] Wrapping text_encoder in DDP...")
        try:
            text_encoder = DDP(text_encoder, device_ids=[rank])
        except Exception as e:
            logger.error(f"[Rank {rank}] DDP Wrapping failed for text_encoder: {e}")
            raise e
        logger.info(f"[Rank {rank}] Text_encoder DDP wrapped.")
    
    # 2. 加载预计算的元数据 (不再需要 memmaps 进行训练，但可能需要 meta.json 获取样本数)
    logger.info(f"[Rank {rank}] Loading metadata from {args.precomputed_dir}...")
    train_meta = json.load(open(os.path.join(args.precomputed_dir, "meta.json")))
    train_total = train_meta["num_samples"]
    logger.info(f"[Rank {rank}] Metadata loaded. Samples: {train_total}")
    # term_mmap 和 trans_mmap 仅在构建初始索引或离线评测时有用，在线训练不再直接依赖它们
    
    # 3. 准备 Glossary 用于评测 (所有进程都需要加载，以便同步执行 Text Encoding)
    # 注意：由于 Text Encoder 现在是可训练的，评测时需要重新计算 Glossary Embeddings
    logger.info(f"[Rank {rank}] Loading glossary metadata...")
    glossary_info = get_glossary_info(args.train_jsonl, args.precomputed_dir)
    dev_info = get_dev_glossary_info(args.dev_jsonl, args.precomputed_dev_dir)
    dev_unique_terms = dev_info["unique_terms"]
    dev_indices = dev_info["indices"]
    
    if is_main:
        logger.info(f"Glossary metadata loaded. Total terms: {len(glossary_info['terms'])}")
    
    if world_size > 1:
        dist.barrier() # 确保所有进程都加载完元数据再继续

    # 4. Optimizer & Data
    # 1) 获取裸模型引用 (处理 DDP .module 包装)
    raw_model = retriever.module if world_size > 1 else retriever
    raw_text_model = text_encoder.module if world_size > 1 else text_encoder

    # 2) 收集参数分组
    # Group 1: Audio LoRA 参数
    audio_lora_params = [p for p in raw_model.audio_encoder.parameters() if p.requires_grad]
    
    # Group 2: Text LoRA 参数
    text_lora_params = [p for p in raw_text_model.encoder.parameters() if p.requires_grad]

    # Group 3: Head 参数 (Projector, Pooler, Logit Scale)
    head_params = list(raw_model.pooler.parameters()) + \
                  list(raw_model.projector.parameters()) + \
                  [raw_model.logit_scale]

    # 3) 定义优化器分组
    optimizer_grouped_parameters = []
    
    if len(audio_lora_params) > 0:
        optimizer_grouped_parameters.append({
            "params": audio_lora_params, 
            "lr": args.lr,
            "name": "audio_lora_params"
        })
    
    if len(text_lora_params) > 0:
        optimizer_grouped_parameters.append({
            "params": text_lora_params, 
            "lr": args.lr,
            "name": "text_lora_params"
        })
    
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

        if "text_model_state_dict" in checkpoint:
            text_state_dict = checkpoint["text_model_state_dict"]
            new_text_state_dict = {}
            has_module_text = any(k.startswith("module.") for k in text_state_dict.keys())
            for k, v in text_state_dict.items():
                if is_ddp and not has_module_text:
                    new_text_state_dict[f"module.{k}"] = v
                elif not is_ddp and has_module_text:
                    new_text_state_dict[k.replace("module.", "")] = v
                else:
                    new_text_state_dict[k] = v
            text_encoder.load_state_dict(new_text_state_dict, strict=False)
            if is_main: logger.info("Loaded text_model_state_dict from checkpoint")

        start_epoch = checkpoint.get("epoch", -1) + 1
        global_step = checkpoint.get("global_step", 0)
        best_recall5_sampled = checkpoint.get("best_recall5_sampled", 0.0)
        best_recall5_full = checkpoint.get("best_recall5_full", 0.0)
        
        if is_main:
            logger.info(f"Resumed from epoch {start_epoch}")
            logger.info(f"Previous Best - Sampled: {best_recall5_sampled:.2%}, Full: {best_recall5_full:.2%}")

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")

    # 7. 数据加载器 (使用 WebDataset)
    if is_main:
        logger.info(f"Using WebDataset shards from {args.train_shards}")
    
    train_loader = get_wds_loader(
        args.train_shards, 
        batch_size=args.batch_size // world_size,
        feature_extractor=feature_extractor,
        is_train=True,
        num_workers=args.num_workers
    )
    
    test_loader = get_wds_loader(
        args.dev_shards, 
        batch_size=args.batch_size // world_size,
        feature_extractor=feature_extractor,
        is_train=False,
        num_workers=args.num_workers
    )

    if is_main:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_exp_name, config=vars(args))

    def run_eval(is_full_eval=False):
        nonlocal best_recall5_sampled, best_recall5_full
        if is_main:
            logger.info(f"Starting evaluation (Full Glossary: {is_full_eval}, Step: {global_step})...")
        
        import faiss
        retriever.eval()
        text_encoder.eval()
        
        # 1. 动态编码搜索库
        if is_full_eval:
            eval_texts = glossary_info['terms']
        else:
            eval_texts = dev_unique_terms
            
        all_text_embs = []
        eval_batch_size = 256 # 调小一点避免 OOM
        with torch.no_grad():
            #  tqdm 只在主进程显示
            pbar_text = tqdm(range(0, len(eval_texts), eval_batch_size), desc="Encoding Glossary") if is_main else range(0, len(eval_texts), eval_batch_size)
            for i in pbar_text:
                batch_texts = eval_texts[i:i+eval_batch_size]
                inputs = text_tokenizer(
                    batch_texts, padding=True, truncation=True, 
                    max_length=64, return_tensors="pt"
                ).to(device)
                # DDP forward: 必须所有 rank 同时调用
                embs = text_encoder(inputs.input_ids, inputs.attention_mask)
                all_text_embs.append(embs.cpu().float().numpy())
        
        search_embs = np.concatenate(all_text_embs, axis=0)
        faiss.normalize_L2(search_embs)
        
        # 本地构建索引
        search_index = faiss.IndexFlatIP(search_embs.shape[1])
        search_index.add(search_embs)
        if is_main:
            logger.info(f"Search index built with {len(eval_texts)} terms.")

        recall_results = {5: [], 10: [], 20: []}
        pos_scores = []
        neg_scores = []
        eval_samples_count = 0
        max_eval_samples = 2000 // world_size # 分布式评测
        
        with torch.no_grad():
            pbar_audio = tqdm(test_loader, desc=f"Eval {'Full' if is_full_eval else 'Sampled'}") if is_main else test_loader
            for eval_batch in pbar_audio:
                if eval_batch is None: continue
                if eval_samples_count >= max_eval_samples: break
                
                input_features = eval_batch["input_features"].to(device).to(torch.bfloat16)
                feature_lens = eval_batch["feature_lens"].to(device)
                batch_samples = eval_batch["samples"]
                
                # 获取音频特征 (DDP forward)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    audio_embs = retriever(input_features, feature_lens)
                
                audio_embs_np = audio_embs.cpu().float().numpy()
                faiss.normalize_L2(audio_embs_np)
                
                D, I = search_index.search(audio_embs_np, 20)
                
                for i, sample in enumerate(batch_samples):
                    eval_samples_count += 1
                    gt_term = sample.get("term", "").strip().lower()
                    top1_score = float(D[i][0])

                    if not gt_term:
                        neg_scores.append(top1_score)
                        continue
                    
                    pos_scores.append(top1_score)

                    if is_full_eval:
                        retrieved_terms = [glossary_info['raw_terms'][idx] for idx in I[i]]
                    else:
                        retrieved_terms = [dev_unique_terms[idx] for idx in I[i]]
                        
                    for k in recall_results.keys():
                        recall_results[k].append(1.0 if gt_term in [t.lower() for t in retrieved_terms[:k]] else 0.0)
        
        # 分布式汇总结果
        final_recall = {}
        avg_pos, avg_neg = 0, 0
        if world_size > 1:
            for k in recall_results.keys():
                hits = torch.tensor([sum(recall_results[k])], device=device, dtype=torch.float32)
                counts = torch.tensor([len(recall_results[k])], device=device, dtype=torch.float32)
                dist.all_reduce(hits, op=dist.ReduceOp.SUM)
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
                final_recall[k] = (hits / counts).item() if counts > 0 else 0
            
            sum_pos = torch.tensor([sum(pos_scores)], device=device, dtype=torch.float32)
            num_pos = torch.tensor([len(pos_scores)], device=device, dtype=torch.float32)
            sum_neg = torch.tensor([sum(neg_scores)], device=device, dtype=torch.float32)
            num_neg = torch.tensor([len(neg_scores)], device=device, dtype=torch.float32)
            dist.all_reduce(sum_pos)
            dist.all_reduce(num_pos)
            dist.all_reduce(sum_neg)
            dist.all_reduce(num_neg)
            avg_pos = (sum_pos / num_pos).item() if num_pos > 0 else 0
            avg_neg = (sum_neg / num_neg).item() if num_neg > 0 else 0
        else:
            final_recall = {k: (sum(v)/len(v) if v else 0) for k, v in recall_results.items()}
            avg_pos = sum(pos_scores)/len(pos_scores) if pos_scores else 0
            avg_neg = sum(neg_scores)/len(neg_scores) if neg_scores else 0

        # 只在主进程记录和保存
        if is_main:
            final_metrics = {}
            current_epoch_recall5 = final_recall[5]
            suffix = "_full" if is_full_eval else "_sampled"
            
            for k, val in final_recall.items():
                metric_name = f"eval/recall@{k}{suffix}"
                final_metrics[metric_name] = val
                logger.info(f"{metric_name}: {val:.2%}")
            
            final_metrics[f"eval/avg_pos_score{suffix}"] = avg_pos
            final_metrics[f"eval/avg_neg_score{suffix}"] = avg_neg
            logger.info(f"📊 Positive Avg Score: {avg_pos:.4f}, Negative: {avg_neg:.4f}")
            
            wandb.log(final_metrics)

            # 保存逻辑
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
                actual_save_path = args.save_path.replace(".pt", f"_{best_ref}_best.pt")
                save_data = {
                    "model_state_dict": retriever.module.state_dict() if world_size > 1 else retriever.state_dict(),
                    "text_model_state_dict": text_encoder.module.state_dict() if world_size > 1 else text_encoder.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_recall5_sampled": best_recall5_sampled,
                    "best_recall5_full": best_recall5_full,
                    "args": vars(args)
                }
                torch.save(save_data, actual_save_path)
                logger.info(f"Model saved to {actual_save_path}")
        
        retriever.train()
        text_encoder.train()
        if world_size > 1:
            dist.barrier() # 确保主进程保存完再继续

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"[Rank {rank}] Starting epoch {epoch}")
        retriever.train()
        text_encoder.train()

        steps_per_epoch = getattr(train_loader, "steps_per_epoch", None) if is_main else None
        pbar = tqdm(
            total=steps_per_epoch,
            desc=f"Epoch {epoch}",
            dynamic_ncols=True,
            mininterval=5.0,
        ) if is_main else None

        for step, batch in enumerate(train_loader):
            if is_main and steps_per_epoch is not None and step >= steps_per_epoch:
                break
            if batch is None: 
                logger.warning(f"[Rank {rank}] Received None batch at step {step}")
                continue
                
            global_step += 1
            
            # 每 N 步做一次全量评测
            if args.eval_steps_full > 0 and global_step % args.eval_steps_full == 0:
                logger.info(f"[Rank {rank}] Entering full evaluation at step {global_step}...")
                run_eval(is_full_eval=True)
                logger.info(f"[Rank {rank}] Full evaluation completed.")
            
            # 每 N 步做一次采样评测 (如果同时到了全量步数，优先跑全量，此处跳过采样)
            elif args.eval_steps_sample > 0 and global_step % args.eval_steps_sample == 0:
                logger.info(f"[Rank {rank}] Entering sampled evaluation at step {global_step}...")
                run_eval(is_full_eval=False)
                logger.info(f"[Rank {rank}] Sampled evaluation completed.")

            # 每 N 步保存一次 pt (与 best 隔离)
            if args.save_steps > 0 and global_step % args.save_steps == 0 and is_main:
                step_save_path = args.save_path.replace(".pt", f"_step_{global_step}.pt")
                logger.info(f"Saving checkpoint at step {global_step} to {step_save_path}")
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': retriever.state_dict(),
                    'text_model_state_dict': text_encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_recall5_sampled': best_recall5_sampled,
                    'best_recall5_full': best_recall5_full,
                }, step_save_path)
            
            # Packed 格式：input_features 是 [128, Total_T]
            input_features = batch["input_features"].to(device).to(torch.bfloat16)
            feature_lens = batch["feature_lens"].to(device)
            samples = batch["samples"]
            
            if step % 10 == 0:
                logger.info(f"[Rank {rank}] Step {step} (Global {global_step}): batch_size={len(samples)}")
            
            # 提取文本并进行 Tokenize
            # 修复弃用警告: torch.cuda.amp.autocast -> torch.amp.autocast
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                audio_embs = retriever(input_features, feature_lens)
                
                # 在线计算 Text Embeddings (开启 LoRA 梯度)
                def encode_text_online(texts):
                    inputs = text_tokenizer(
                        texts, padding=True, truncation=True, 
                        max_length=64, return_tensors="pt"
                    ).to(device)
                    return text_encoder(inputs.input_ids, inputs.attention_mask)
                
                # Contrastive Loss
                logit_scale = retriever.module.logit_scale.exp() if world_size > 1 else retriever.logit_scale.exp()
                labels = torch.arange(len(audio_embs), device=device)
                
                total_loss = 0
                loss_term_val = 0
                loss_trans_val = 0
                term_embs = None
                trans_embs = None

                # Audio-Term Loss
                if args.term_weight > 0:
                    terms = [s.get("term", "").strip() for s in samples]
                    term_embs = encode_text_online(terms)
                    sim_term = (audio_embs @ term_embs.T) * logit_scale
                    loss_term = F.cross_entropy(sim_term, labels)
                    total_loss += loss_term * args.term_weight
                    loss_term_val = loss_term.item()
                
                # Audio-Transcript Loss
                if args.trans_weight > 0:
                    translations = [s.get("translation", "").strip() or s.get("term", "").strip() for s in samples]
                    trans_embs = encode_text_online(translations)
                    sim_trans = (audio_embs @ trans_embs.T) * logit_scale
                    loss_trans = F.cross_entropy(sim_trans, labels)
                    total_loss += loss_trans * args.trans_weight
                    loss_trans_val = loss_trans.item()

                # Anomaly Detection (Research mode)
                # Note: use tqdm.write to avoid breaking the progress bar rendering.
                if total_loss.item() > args.anomaly_threshold and is_main:
                    with torch.no_grad():
                        # 计算单样本 Loss 来定位“毒样本”
                        per_sample_loss = F.cross_entropy(
                            (audio_embs @ term_embs.T) * logit_scale if args.term_weight > 0 else (audio_embs @ trans_embs.T) * logit_scale,
                            labels, reduction='none'
                        )
                        max_loss, max_idx = torch.max(per_sample_loss, dim=0)
                        bad_sample = batch["samples"][max_idx.item()]
                        tqdm.write(
                            f"[ANOMALY] High loss detected: {total_loss.item():.2f} | "
                            f"Max sample loss: {max_loss.item():.2f}"
                        )
                        tqdm.write(
                            f"[ANOMALY] Details: Path={bad_sample.get('chunk_audio_path')}, "
                            f"Term='{bad_sample.get('term')}', Trans='{bad_sample.get('translation')}'"
                        )
                        
                        # 写入文件供后续数据清洗
                        with open("training_anomalies.jsonl", "a") as af:
                            # Use a recursive helper to convert any numpy arrays to lists for JSON serialization
                            def to_json_compatible(obj):
                                if isinstance(obj, np.ndarray):
                                    return obj.tolist()
                                if isinstance(obj, dict):
                                    return {k: to_json_compatible(v) for k, v in obj.items()}
                                if isinstance(obj, (list, tuple)):
                                    return [to_json_compatible(i) for i in obj]
                                if isinstance(obj, (np.float32, np.float64)):
                                    return float(obj)
                                if isinstance(obj, (np.int32, np.int64)):
                                    return int(obj)
                                return obj

                            # 核心修复：排除掉极其占用空间的 audio 数据并转换其余 numpy 类型
                            clean_metadata = {k: v for k, v in bad_sample.items() if k != "audio"}
                            anomaly_record = {
                                "global_step": global_step,
                                "batch_avg_loss": total_loss.item(),
                                "sample_loss": max_loss.item(),
                                "metadata": to_json_compatible(clean_metadata)
                            }
                            af.write(json.dumps(anomaly_record, ensure_ascii=False) + "\n")

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            
            # Gradient Clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(retriever.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            if is_main:
                pbar.update(1)
                pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})
                wandb.log({
                    "train/loss": total_loss.item(), 
                    "train/audio_term_loss": loss_term_val,
                    "train/audio_trans_loss": loss_trans_val,
                    "train/lr": optimizer.param_groups[0]["lr"]
                })

        if is_main and pbar is not None:
            pbar.close()

        # ==================== Evaluation (End of Epoch) ====================
        if is_main:
            # 每个 epoch 结束做一次全量词库评测
            run_eval(is_full_eval=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--dev_jsonl", type=str, required=True, help="Path to dev JSONL file")
    parser.add_argument("--train_shards", type=str, required=True, help="WebDataset shards pattern for train")
    parser.add_argument("--dev_shards", type=str, required=True, help="WebDataset shards pattern for dev")
    parser.add_argument("--precomputed_dir", type=str, required=True, help="Directory with .mmap files for train")
    parser.add_argument("--precomputed_dev_dir", type=str, required=True, help="Directory with .mmap files for dev")
    parser.add_argument("--save_path", type=str, default="qwen3_retriever.pt")
    parser.add_argument("--resume", type=str, default=None, help="Path to .pt file to resume from")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use_lora", action="store_true", default=False, help="Whether to use LoRA for audio encoder")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--text_lora_rank", type=int, default=16, help="LoRA rank for BGE-M3 text encoder")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--test_limit", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps_sample", type=int, default=200, help="Sampled evaluation every N steps")
    parser.add_argument("--eval_steps_full", type=int, default=500, help="Full evaluation every N steps")
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
