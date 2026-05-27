#!/usr/bin/env python3
"""
Qwen3-Omni + BGE-M3 (Unfrozen Text Encoder) Training Script with Hard Negatives
Upgraded with dynamic ACL evaluation and separation diagnostics.
"""

import os
import sys
import time
import argparse
import json
import random
import logging
import pickle
import re
import glob
from collections import OrderedDict
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterable

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
import faiss

# Import ACL evaluation utilities
from retriever.gigaspeech.acl_eval_utils import (
    build_dynamic_index,
    run_acl_simulation
)

# Optional: fast keyword extraction (same as offline eval script)
try:
    from flashtext import KeywordProcessor  # type: ignore
except Exception:  # pragma: no cover
    KeywordProcessor = None

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Enable TF32/BF16 optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==================== Model Components ====================

class TextNegCache:
    """
    Stale negative cache:
    - Stores negative term embeddings computed from a (recent) text encoder snapshot.
    - Positives are always encoded online.
    - Cache can be refreshed (cleared) every N steps.

    Notes:
    - This cache is per-rank and does not synchronize across ranks.
    - Embeddings are stored on CPU in float16 to reduce GPU memory pressure.
    """
    def __init__(self, max_size: int, refresh_steps: int, device: torch.device):
        self.max_size = int(max_size)
        self.refresh_steps = int(refresh_steps)
        self.device = device
        self._store: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._last_refresh_step = 0

    def maybe_refresh(self, step: int) -> None:
        if self.refresh_steps <= 0:
            return
        if step - self._last_refresh_step >= self.refresh_steps:
            self._store.clear()
            self._last_refresh_step = step

    def _put(self, key: str, emb: np.ndarray) -> None:
        if key in self._store:
            self._store.pop(key, None)
        self._store[key] = emb
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def get_embeddings(self, terms: List[str], text_encoder, tokenizer) -> torch.Tensor:
        """
        Return embeddings for `terms` in the same order.
        - Missing terms are encoded online and added to cache.
        - Already cached terms are retrieved.
        - Ensures consistency even if cache eviction happens during the call.
        """
        # 1. Identify missing terms (ignoring duplicates within this batch)
        missing = []
        missing_set = set()
        for t in terms:
            if t not in self._store and t not in missing_set:
                missing.append(t)
                missing_set.add(t)

        # 2. Encode missing terms and store them temporarily for this batch
        batch_new_embs = {}
        if missing:
            inputs = tokenizer(missing, padding=True, truncation=True, max_length=64, return_tensors="pt").to(self.device)
            with torch.no_grad():
                embs = text_encoder(inputs.input_ids, inputs.attention_mask)  # [U, D]
                embs_cpu = embs.detach().to(torch.float16).cpu().numpy()
            
            for t, e in zip(missing, embs_cpu):
                batch_new_embs[t] = e

        # 3. Consolidate results for the entire request BEFORE updating the persistent store
        # This prevents items needed for the current request from being evicted while we're processing it.
        res_list = []
        for t in terms:
            if t in batch_new_embs:
                res_list.append(batch_new_embs[t])
            elif t in self._store:
                res_list.append(self._store[t])
            else:
                # This should only happen if there's an extremely rare race or logic error.
                logger.error(f"[TextNegCache] Unexpected missing term in consolidation: '{t}'. Using zero fallback.")
                res_list.append(np.zeros(1024, dtype=np.float16))

        # 4. Now update the persistent store (and trigger LRU evictions)
        for t, e in batch_new_embs.items():
            self._put(t, e)

        out = np.stack(res_list, axis=0)
        return torch.from_numpy(out).to(self.device).to(torch.bfloat16)

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
    def __init__(self, model_id="BAAI/bge-m3", lora_rank=16, lora_alpha=32, target_modules=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16,
            add_pooling_layer=False
        )
        
        if target_modules is None:
            target_modules = ["query", "key", "value"]

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=None
        )
        self.encoder = get_peft_model(self.encoder, lora_config)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]
        return F.normalize(embeddings, p=2, dim=-1)

class Qwen3OmniRetriever(nn.Module):
    def __init__(self, model_id="Atotti/Qwen3-Omni-AudioTransformer", target_dim=1024, 
                 use_lora=True, lora_rank=32, lora_alpha=64, lora_target_modules=None):
        super().__init__()
        
        self.audio_encoder = Qwen3OmniMoeAudioEncoder.from_pretrained(
            model_id, 
            dtype=torch.bfloat16
        )
        
        if hasattr(self.audio_encoder, "conv2d1"):
            self.audio_encoder.get_input_embeddings = lambda: self.audio_encoder.conv2d1
            
        self.audio_encoder.gradient_checkpointing_enable()
        
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
        
        self.pooler = AttentivePooling(2048) 
        self.projector = nn.Linear(2048, target_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, input_features, feature_lens):
        if input_features.ndim == 3:
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
    def __init__(
        self,
        samples: List[Dict],
        hard_negatives: Dict[str, List[str]] = None,
        num_negatives: int = 32,
        fallback_terms: Optional[List[str]] = None,
        hn_fallback_mode: str = "random",  # random | dummy | error
        hn_dummy_negative: str = "dummy_negative",
        hn_select_mode: str = "top",  # top | random
        hn_random_pool_size: int = 32,  # when hn_select_mode=random, sample from top-N candidates
    ):
        self.samples = samples
        self.hard_negatives = hard_negatives or {}
        self.num_negatives = num_negatives
        self.fallback_terms = fallback_terms or list(self.hard_negatives.keys())
        self.hn_fallback_mode = hn_fallback_mode
        self.hn_dummy_negative = hn_dummy_negative
        self.hn_select_mode = hn_select_mode
        self.hn_random_pool_size = int(hn_random_pool_size)
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_path = sample.get("chunk_audio_path")
        try:
            res = {k: v for k, v in sample.items()}
            # If using precomputed fbank, we don't load audio here.
            if audio_path:
                audio_data, sr = sf.read(audio_path)
                if sr != 16000:
                    audio_data, sr = sf.read(audio_path)
                if audio_data.ndim > 1:
                    audio_data = audio_data.mean(axis=1)
                if np.max(np.abs(audio_data)) > 0:
                    audio_data = audio_data / np.max(np.abs(audio_data))
                res["audio"] = audio_data.astype(np.float32)
            else:
                res["audio"] = None
            
            # Hard Negatives - FIX: added .strip() to match mining script
            term = sample.get("term", "").strip().lower()
            hns = self.hard_negatives.get(term, [])
            selected_hns: List[str] = []
            res["_hn_fallback"] = False
            res["_hn_fallback_mode"] = ""
            if hns:
                # If mining script saved HNs in similarity order, "top" keeps the hardest ones deterministic.
                # "random" adds diversity (but can be non-deterministic).
                if self.hn_select_mode == "random":
                    pool_n = self.hn_random_pool_size if self.hn_random_pool_size > 0 else len(hns)
                    pool = hns[: min(pool_n, len(hns))]
                    if len(pool) >= self.num_negatives:
                        selected_hns = random.sample(pool, self.num_negatives)
                    else:
                        # Sample with replacement to enforce fixed M
                        if not pool:
                            selected_hns = [self.hn_dummy_negative] * self.num_negatives
                        else:
                            selected_hns = [random.choice(pool) for _ in range(self.num_negatives)]
                else:
                    # default: top
                    if len(hns) >= self.num_negatives:
                        selected_hns = hns[: self.num_negatives]
                    else:
                        # Pad by cycling to keep fixed M deterministically
                        reps = (self.num_negatives + len(hns) - 1) // len(hns)
                        selected_hns = (hns * reps)[: self.num_negatives]
            else:
                # Fallback: some terms may have no mined HN entry (e.g., key mismatch or mining coverage).
                # We keep M fixed for stable shapes, but MUST make this behavior explicit and measurable.
                res["_hn_fallback"] = True
                res["_hn_fallback_mode"] = self.hn_fallback_mode
                if self.hn_fallback_mode == "error":
                    raise KeyError(f"No hard negatives found for term='{term}'")
                elif self.hn_fallback_mode == "dummy":
                    selected_hns = [self.hn_dummy_negative] * self.num_negatives
                else:
                    # default: random
                    if not self.fallback_terms:
                        selected_hns = [self.hn_dummy_negative] * self.num_negatives
                    else:
                        for _ in range(self.num_negatives):
                            cand = random.choice(self.fallback_terms)
                            # avoid self-term if possible
                            if cand == term and len(self.fallback_terms) > 1:
                                for _try in range(3):
                                    cand = random.choice(self.fallback_terms)
                                    if cand != term:
                                        break
                            selected_hns.append(cand)

            res["hard_negatives"] = selected_hns
                
            return res
        except Exception as e:
            # If user wants strict behavior, don't silently swallow exceptions.
            if getattr(self, "hn_fallback_mode", "") == "error":
                raise
            return {"audio": None, "chunk_audio_path": audio_path}

def collate_fn_hn(batch, feature_extractor):
    valid_samples = []
    for s in batch:
        if s is None or s.get("audio") is None or len(s["audio"]) <= 3000:
            continue
        valid_samples.append(s)
    
    if not valid_samples:
        dummy_audio = np.zeros(30720, dtype=np.float32)
        valid_samples = [{"audio": dummy_audio, "term": "dummy", "translation": "dummy", "hard_negatives": []}]

    target_len = 30720 
    audios = []
    for s in valid_samples:
        audio = s["audio"]
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')
        elif len(audio) > target_len:
            audio = audio[:target_len]
        audios.append(audio)

    inputs = feature_extractor(audios, sampling_rate=16000, return_tensors="pt", padding=False)
    input_features = inputs.input_features
    B, C, T_mel = input_features.shape
    feature_lens = torch.full((B,), T_mel, dtype=torch.long)
    
    return {
        "input_features": input_features,
        "feature_lens": feature_lens,
        "samples": valid_samples 
    }

# ==================== Evaluation & Diagnostics ====================

def run_dev_eval(retriever, text_encoder, tokenizer, dev_loader, device, dev_unique_terms, is_main):
    """
    Upgraded Dev Evaluation:
    - Runs only on Rank 0.
    - Global retrieval against all dev unique terms.
    - Safety checks for index mapping and dummy labels.
    """
    if not is_main:
        return {}

    retriever.eval()
    text_encoder.eval()
    logger.info(f"[Dev Eval] corpus_size={len(dev_unique_terms)}")
    
    # 1. Encode all unique dev terms
    all_text_embs = []
    eval_batch_size = 256
    with torch.no_grad():
        for i in range(0, len(dev_unique_terms), eval_batch_size):
            batch_texts = dev_unique_terms[i:i+eval_batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                embs = text_encoder(inputs.input_ids, inputs.attention_mask)
            all_text_embs.append(embs.cpu().float().numpy())
    
    if not all_text_embs:
        return {}
        
    search_embs = np.concatenate(all_text_embs, axis=0)
    faiss.normalize_L2(search_embs)
    search_index = faiss.IndexFlatIP(search_embs.shape[1])
    search_index.add(search_embs)

    # 2. Retrieval loop
    recall_results_hits = {1: 0.0, 5: 0.0, 10: 0.0}
    total_samples = 0
    # kept for future debugging (e.g., skip reasons); currently unused
    skipped = 0
    
    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Dev Eval", leave=False):
            input_features = batch["input_features"].to(device).to(torch.bfloat16)
            feature_lens = batch["feature_lens"].to(device)
            samples = batch["samples"]
            
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                audio_embs = retriever(input_features, feature_lens)
            
            audio_embs_np = audio_embs.cpu().float().numpy()
            faiss.normalize_L2(audio_embs_np)
            
            D, I = search_index.search(audio_embs_np, 10)
            
            for i, sample in enumerate(samples):
                # Match old script behavior: prefer gt_term if present, fallback to term
                gt_term = sample.get("gt_term", sample.get("term", "")).strip().lower()
                if not gt_term or gt_term == "dummy": continue
                
                total_samples += 1
                retrieved_indices = I[i]
                retrieved_terms = []
                for idx in retrieved_indices:
                    if 0 <= idx < len(dev_unique_terms):
                        retrieved_terms.append(dev_unique_terms[idx].lower())
                
                for k in recall_results_hits.keys():
                    if gt_term in retrieved_terms[:k]:
                        recall_results_hits[k] += 1.0

    metrics = {}
    if total_samples > 0:
        for k, hits in recall_results_hits.items():
            metrics[f"dev/recall@{k}"] = hits / total_samples
    logger.info(f"[Dev Eval] evaluated={total_samples} (skipped={skipped})")
    
    return metrics

def run_acl_eval(retriever, text_encoder, tokenizer, args, device, is_main, glossary_entries):
    """
    Upgraded ACL Offline Evaluation:
    - Uses unified simulation logic from acl_eval_utils.
    """
    if not is_main:
        return {}
    
    logger.info("Running ACL Offline Evaluation (aligned to v4 offline eval)...")
    glossary_terms = [e["key"] for e in glossary_entries]
    faiss_index = build_dynamic_index(text_encoder, tokenizer, glossary_entries, device, batch_size=args.acl_index_batch_size)
    
    # Load ACL metadata
    wav_files = sorted(
        glob.glob(os.path.join(args.wav_dir, "*.wav")),
        key=lambda x: int(re.search(r"sent_(\d+)", x).group(1)) if re.search(r"sent_(\d+)", x) else 0,
    )
    if not wav_files:
        logger.warning(f"No wav files found in {args.wav_dir}")
        return {}
        
    with open(args.txt_path, "r", encoding="utf-8") as f:
        text_lines = [line.strip() for line in f]
    
    retriever.eval()
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")

    sim_results = run_acl_simulation(
        retriever=retriever,
        faiss_index=faiss_index,
        glossary_terms=glossary_terms,
        wav_files=wav_files,
        text_lines=text_lines,
        args=args,
        device=device,
        feature_extractor=feature_extractor,
        limit=args.acl_eval_limit
    )

    logger.info(f"[ACL Eval] used_samples={sim_results['used_samples']} skipped_no_gt={sim_results['skipped_no_gt']} total_gt={sim_results['total_gt']}")

    return {
        "acl/recall@5": sim_results["recall"],
        "acl/precision": sim_results["precision"],
        "acl/pos_score_mean": sim_results["pos_score_mean"],
        "acl/neg_score_mean": sim_results["neg_score_mean"],
        "acl/margin": sim_results["margin"],
        "acl/gap_top1_top5_mean": sim_results["gap_top1_top5_mean"],
        "acl/gt_minus_mean_top5_mean": sim_results["gt_minus_mean_top5_mean"],
    }

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

    text_encoder = BgeM3TextEncoder(
        lora_rank=args.text_lora_rank,
        lora_alpha=args.text_lora_alpha,
        target_modules=args.text_lora_target_modules
    ).to(device)
    text_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

    # 1.5 Load Checkpoint if provided (IMPORTANT: load BEFORE DDP wrapping)
    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            if is_main:
                logger.warning(f"Checkpoint path {args.checkpoint} not found! Starting from scratch.")
        else:
            if is_main:
                logger.info(f"Loading checkpoint from {args.checkpoint}...")
            ckpt = torch.load(args.checkpoint, map_location="cpu")

            def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
                if not sd:
                    return sd
                if any(k.startswith("module.") for k in sd.keys()):
                    return {k[len("module."):]: v for k, v in sd.items()}
                return sd

            def _load_component(model: nn.Module, state_dict: Dict[str, torch.Tensor], name: str) -> None:
                sd = _strip_module_prefix(state_dict)
                model_keys = set(model.state_dict().keys())
                sd_keys = set(sd.keys())
                matched = len(model_keys & sd_keys)
                incompatible = model.load_state_dict(sd, strict=False)
                if is_main:
                    logger.info(
                        f"[CKPT] {name}: matched_keys={matched}/{len(model_keys)} "
                        f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}"
                    )
                    # If we effectively loaded nothing, scream loudly.
                    if matched < 50:
                        logger.warning(f"[CKPT] {name}: suspiciously few matched keys ({matched}). Check LoRA ranks/targets and checkpoint format.")

            if "model_state_dict" in ckpt:
                _load_component(retriever, ckpt["model_state_dict"], "retriever")
            else:
                if is_main:
                    logger.warning("[CKPT] missing key 'model_state_dict' in checkpoint")

            if "text_model_state_dict" in ckpt:
                _load_component(text_encoder, ckpt["text_model_state_dict"], "text_encoder")
            else:
                if is_main:
                    logger.warning("[CKPT] missing key 'text_model_state_dict' in checkpoint")

            if is_main:
                logger.info("Checkpoint load done.")

    # Wrap with DDP after loading weights
    if world_size > 1:
        retriever = DDP(retriever, device_ids=[rank])
        text_encoder = DDP(text_encoder, device_ids=[rank])

    # 2. Hard Negatives & Glossary Pre-loading
    hard_negatives = {}
    if args.hn_path and os.path.exists(args.hn_path):
        with open(args.hn_path, "r", encoding="utf-8") as f:
            hard_negatives = json.load(f)
        if is_main: logger.info(f"Loaded hard negatives from {args.hn_path}")
    fallback_terms = list(hard_negatives.keys())

    # Pre-parse glossary for ACL eval to avoid repeated JSON parsing
    glossary_entries = []
    if is_main:
        logger.info(f"Pre-loading glossary from {args.glossary_path}...")
        with open(args.glossary_path, "r", encoding="utf-8") as f:
            glossary = json.load(f)
        seen_keys = set()
        for term, payload in glossary.items():
            canonical_key = term.strip().lower()
            if not canonical_key or canonical_key in seen_keys: continue
            seen_keys.add(canonical_key)
            glossary_entries.append({"key": canonical_key, "term": term.strip()})
        logger.info(f"Pre-loaded {len(glossary_entries)} glossary entries.")

    # 3. Data Loaders
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
    
    train_samples = []
    with open(args.train_jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.max_train_samples > 0 and i >= args.max_train_samples:
                break
            train_samples.append(json.loads(line))
    
    dev_samples = []
    with open(args.dev_jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.max_dev_samples > 0 and i >= args.max_dev_samples:
                break
            dev_samples.append(json.loads(line))
    
    # Match the old training script: dev term pool is lowercased (important for consistent retrieval/eval)
    dev_unique_terms = sorted(list(set([s.get("term", "").strip().lower() for s in dev_samples if s.get("term")])))
    if is_main: logger.info(f"Dev unique terms: {len(dev_unique_terms)}")

    # --- Train Loader ---
    train_dataset = TermRAGDataset(
        train_samples,
        hard_negatives=hard_negatives,
        num_negatives=args.num_hard_negs,
        fallback_terms=fallback_terms,
        hn_fallback_mode=args.hn_fallback_mode,
        hn_select_mode=args.hn_select_mode,
        hn_random_pool_size=args.hn_random_pool_size,
    )
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size // world_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=lambda b: collate_fn_hn(b, feature_extractor),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # --- Dev Loader ---
    dev_dataset = TermRAGDataset(dev_samples, num_negatives=0) 
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size // world_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn_hn(b, feature_extractor),
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 4. Optimizer
    raw_retriever = retriever.module if world_size > 1 else retriever
    raw_text_encoder = text_encoder.module if world_size > 1 else text_encoder
    
    optimizer = torch.optim.AdamW([
        {"params": [p for p in raw_retriever.audio_encoder.parameters() if p.requires_grad], "lr": args.lr},
        {"params": [p for p in raw_text_encoder.encoder.parameters() if p.requires_grad], "lr": args.lr},
        {"params": list(raw_retriever.pooler.parameters()) + list(raw_retriever.projector.parameters()) + [raw_retriever.logit_scale], "lr": args.lr * 2}
    ], weight_decay=0.01)
    
    scaler = torch.amp.GradScaler("cuda")
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=len(train_loader) * args.epochs)

    if is_main:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_exp_name, config=vars(args))

    global_step = 0
    best_recall5_dev = 0.0
    best_recall5_acl = 0.0
    
    neg_cache: Optional["TextNegCache"] = None
    if args.neg_cache_max_size > 0:
        neg_cache = TextNegCache(
            max_size=args.neg_cache_max_size,
            refresh_steps=args.neg_cache_refresh_steps,
            device=device,
        )

    for epoch in range(args.epochs):
        retriever.train()
        text_encoder.train()
        if train_sampler: train_sampler.set_epoch(epoch)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}") if is_main else train_loader
        step_in_epoch = 0
        for batch in pbar:
            step_in_epoch += 1
            global_step += 1

            if args.max_steps_per_epoch > 0 and step_in_epoch > args.max_steps_per_epoch:
                break
            
            # Dynamic Eval
            if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                # 1. Dev Eval (JSONL based Global Retrieval)
                dev_results = run_dev_eval(raw_retriever, raw_text_encoder, text_tokenizer, dev_loader, device, dev_unique_terms, is_main)
                
                # 2. ACL Eval (Sliding Window + Max Pooling)
                acl_results = run_acl_eval(raw_retriever, raw_text_encoder, text_tokenizer, args, device, is_main, glossary_entries)
                
                if is_main:
                    combined_results = {**dev_results, **acl_results}
                    wandb.log(combined_results, step=global_step)
                    
                    r5_dev = dev_results.get("dev/recall@5", 0)
                    r5_acl = acl_results.get("acl/recall@5", 0)
                    logger.info(f"Step {global_step} | Dev R@5: {r5_dev:.4f} | ACL R@5: {r5_acl:.4f}")
                    
                    # Save best models
                    if r5_dev > best_recall5_dev:
                        best_recall5_dev = r5_dev
                        torch.save({
                            "model_state_dict": raw_retriever.state_dict(),
                            "text_model_state_dict": raw_text_encoder.state_dict(),
                            "global_step": global_step,
                            "recall5_dev": r5_dev
                        }, args.save_path.replace(".pt", "_best_dev.pt"))
                        logger.info(f"New Best Dev Model saved with R@5={r5_dev:.4f}")

                    if r5_acl > best_recall5_acl:
                        best_recall5_acl = r5_acl
                        torch.save({
                            "model_state_dict": raw_retriever.state_dict(),
                            "text_model_state_dict": raw_text_encoder.state_dict(),
                            "global_step": global_step,
                            "recall5_acl": r5_acl
                        }, args.save_path.replace(".pt", "_best_acl.pt"))
                        logger.info(f"New Best ACL Model saved with R@5={r5_acl:.4f}")
                
                # Sync best stats to other ranks
                if world_size > 1:
                    dist.barrier() # Ensure main rank finished saving
                    best_t = torch.tensor([best_recall5_dev, best_recall5_acl], device=device)
                    dist.broadcast(best_t, src=0)
                    best_recall5_dev, best_recall5_acl = best_t[0].item(), best_t[1].item()

                retriever.train()
                text_encoder.train()

            input_features = batch["input_features"].to(device).to(torch.bfloat16)
            feature_lens = batch["feature_lens"].to(device)
            samples = batch["samples"]

            # 🟢 HN fallback diagnostics (explicitly surfaced; no hidden rules)
            if is_main:
                fb_cnt = sum(1 for s in samples if s.get("_hn_fallback"))
                if (args.hn_fallback_log_steps > 0) and (global_step % args.hn_fallback_log_steps == 0):
                    fb_frac = fb_cnt / max(1, len(samples))
                    # log a few examples for debugging key mismatches
                    ex_terms = [s.get("term", "") for s in samples if s.get("_hn_fallback")][:5]
                    if fb_cnt > 0:
                        logger.warning(f"[HN-Fallback] step={global_step} fallback_frac={fb_frac:.2%} ({fb_cnt}/{len(samples)}) mode={args.hn_fallback_mode} examples={ex_terms}")
                    else:
                        logger.info(f"[HN-Fallback] step={global_step} fallback_frac=0.00% (0/{len(samples)}) mode={args.hn_fallback_mode}")
                    wandb.log(
                        {
                            "train/hn_fallback_frac": fb_frac,
                            "train/hn_fallback_cnt": fb_cnt,
                        },
                        step=global_step,
                    )
            
            # Prepare texts: positives online, negatives potentially cached (stale)
            M = args.num_hard_negs
            pos_texts: List[str] = []
            neg_texts_flat: List[str] = []
            for s in samples:
                pos_text = s.get("term", "").strip()
                pos_texts.append(pos_text)
                hns = s.get("hard_negatives", [])
                if len(hns) != M:
                    # Should not happen with dataset enforcement; keep safe fallback.
                    if len(hns) == 0:
                        hns = ["dummy_negative"] * M
                    elif len(hns) < M:
                        hns = hns + [hns[-1]] * (M - len(hns))
                    else:
                        hns = hns[:M]
                neg_texts_flat.extend(hns)
            
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                audio_embs = retriever(input_features, feature_lens)

                logit_scale = raw_retriever.logit_scale.exp()
                
                B = audio_embs.shape[0]
                # 1) Positive embeddings: always online (text encoder is training)
                pos_inputs = text_tokenizer(pos_texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
                pos_embs = text_encoder(pos_inputs.input_ids, pos_inputs.attention_mask)  # [B, D]

                # 2) Negative embeddings: stale cache (optionally refreshed every N steps)
                if neg_cache is None:
                    # No caching: encode negatives online
                    neg_inputs = text_tokenizer(neg_texts_flat, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
                    neg_embs = text_encoder(neg_inputs.input_ids, neg_inputs.attention_mask).view(B, M, -1)
                else:
                    neg_cache.maybe_refresh(global_step)
                    # FIX: Use raw_text_encoder instead of DDP-wrapped text_encoder to avoid sync issues
                    neg_embs = neg_cache.get_embeddings(neg_texts_flat, text_encoder=raw_text_encoder, tokenizer=text_tokenizer).view(B, M, -1)

                pos_logits = torch.sum(audio_embs * pos_embs, dim=-1, keepdim=True)          # [B, 1]
                neg_logits = torch.sum(audio_embs.unsqueeze(1) * neg_embs, dim=-1)           # [B, M]
                logits = torch.cat([pos_logits, neg_logits], dim=1) * logit_scale            # [B, 1+M]

                labels = torch.zeros(B, dtype=torch.long, device=device)
                total_loss = F.cross_entropy(logits, labels)

                # Diagnostics (raw cosine sim, before logit_scale)
                batch_pos_scores = pos_logits.squeeze(1).detach().float().cpu().numpy().tolist()
                batch_neg_scores = neg_logits.detach().float().mean(dim=1).cpu().numpy().tolist()

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            if is_main:
                pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})
                wandb.log({
                    "train/loss": total_loss.item(),
                    "train/pos_score": np.mean(batch_pos_scores) if batch_pos_scores else 0,
                    "train/neg_score": np.mean(batch_neg_scores) if batch_neg_scores else 0,
                    "train/margin": (np.mean(batch_pos_scores) - np.mean(batch_neg_scores)) if batch_pos_scores else 0,
                }, step=global_step)

        if is_main:
            save_path = args.save_path.replace(".pt", f"_epoch_{epoch}.pt")
            torch.save({
                "model_state_dict": raw_retriever.state_dict(),
                "text_model_state_dict": raw_text_encoder.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
            }, save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--dev_jsonl", type=str, required=True)
    parser.add_argument("--hn_path", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to pre-trained checkpoint")
    parser.add_argument("--num_hard_negs", type=int, default=32)
    parser.add_argument("--glossary_path", type=str, required=True)
    parser.add_argument("--wav_dir", type=str, required=True)
    parser.add_argument("--txt_path", type=str, required=True)
    parser.add_argument("--acl_eval_limit", type=int, default=100)
    # ACL eval parameters (aligned to run_eval_rag_offline_aries_v4.sh)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--rag_voting_k", type=int, default=20)
    parser.add_argument("--rag_voting_min_votes", type=int, default=2)
    parser.add_argument("--rag_chunk_size", type=float, default=1.92)
    parser.add_argument("--rag_hop_size", type=float, default=0.96)
    parser.add_argument("--vllm_interval", type=float, default=1.92)
    parser.add_argument("--rag_strategy", type=str, default="max_pool", choices=["voting", "max_pool"])
    parser.add_argument("--score_threshold", type=float, default=0.45)
    parser.add_argument("--acl_audio_batch_size", type=int, default=32, help="Batch size for ACL eval window encoding")
    parser.add_argument("--acl_index_batch_size", type=int, default=512, help="Batch size for encoding glossary when building dynamic index")
    parser.add_argument("--save_path", type=str, default="qwen3_retriever_hn.pt")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--text_lora_rank", type=int, default=16)
    parser.add_argument("--text_lora_alpha", type=int, default=32)
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=None)
    parser.add_argument("--text_lora_target_modules", type=str, nargs="+", default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_train_samples", type=int, default=0, help="0 means use all training samples")
    parser.add_argument("--max_dev_samples", type=int, default=0, help="0 means use all dev samples")
    parser.add_argument("--max_steps_per_epoch", type=int, default=0, help="0 means run full epoch")
    parser.add_argument("--neg_cache_max_size", type=int, default=200000, help="0 disables stale negative cache")
    parser.add_argument("--neg_cache_refresh_steps", type=int, default=1000, help="Refresh negative cache every N steps")
    parser.add_argument("--hn_fallback_mode", type=str, default="error", choices=["random", "dummy", "error"], help="What to do if a term has no mined hard negatives")
    parser.add_argument("--hn_fallback_log_steps", type=int, default=100, help="Log HN fallback stats every N steps (0 disables)")
    parser.add_argument("--hn_select_mode", type=str, default="random", choices=["top", "random"], help="How to select HNs when a term has >= M candidates (top keeps mined order)")
    parser.add_argument("--hn_random_pool_size", type=int, default=32, help="When --hn_select_mode=random, sample negatives from top-N mined candidates (0 means use full list)")
    parser.add_argument("--wandb_project", type=str, default="qwen3_rag_hn")
    parser.add_argument("--wandb_exp_name", type=str, default="hn_training")
    args = parser.parse_args()
    
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("LOCAL_RANK", 0))
    train(rank, world_size, args)

if __name__ == "__main__":
    main()

