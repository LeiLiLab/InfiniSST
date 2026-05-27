#!/usr/bin/env python3
"""
Qwen3-Omni + BGE-M3 Training with Glossary-Scale Negative Columns

Key features:
1. In-batch contrastive learning with multi-positive grouping by chunk_id
2. False-negative masking: same-term pairs across different chunks are masked
   out of the denominator (not treated as negatives)
3. Glossary negatives: full wiki glossary (~10k terms) appended as permanent
   negative columns in the similarity matrix every step, directly optimising
   the model to rank GT terms above all glossary distractors
4. Text encoder full-finetune option (skip LoRA, unfreeze all parameters)
"""

import os
import json
import time
import argparse
import datetime
import hashlib
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    WhisperFeatureExtractor,
    get_cosine_schedule_with_warmup,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
)
from peft import LoraConfig, get_peft_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ======Configuration=====
DEFAULT_QWEN_AUDIO_MODEL_ID = "Atotti/Qwen3-Omni-AudioTransformer"
DEFAULT_TEXT_MODEL_ID = "BAAI/bge-m3"

DEFAULT_AUDIO_SAMPLE_RATE = 16000
DEFAULT_MIN_AUDIO_SAMPLES = 3000
DEFAULT_FIXED_AUDIO_SAMPLES = 30720
DEFAULT_TEXT_MAX_LENGTH = 64
DEFAULT_TARGET_DIM = 1024
DEFAULT_AUDIO_HIDDEN_DIM = 2048

DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_HEAD_LR_SCALE = 10.0
DEFAULT_GRAD_CLIP_MAX_NORM = 1.0
DEFAULT_WARMUP_RATIO = 0.1

DEFAULT_LOG_INTERVAL = 20
DEFAULT_SAVE_INTERVAL = 1000
DEFAULT_KEEP_CHECKPOINTS = 3
DEFAULT_DDP_TIMEOUT_SECONDS = 7200
DEFAULT_WANDB_LOG_INTERVAL = 20
DEFAULT_EVAL_STEPS_SAMPLE = 200
DEFAULT_EVAL_BATCH_SIZE = 256
DEFAULT_EVAL_TOPK = 5
DEFAULT_EVAL_TOPK_EXTRA = 10

DEFAULT_NEG_BANK_SIZE = 0
DEFAULT_NEG_BANK_REFRESH_STEPS = 500
DEFAULT_NEG_BANK_ENCODE_BATCH = 512
DEFAULT_HARD_NEG_K = 0

DEFAULT_EVAL_GLOSSARY_SIZES: List[int] = []
DEFAULT_EVAL_WIKI_GLOSSARY = ""
DEFAULT_ACL_DEV_JSONL = ""
DEFAULT_BEST_METRIC = ""
DEFAULT_EVAL_TERM_ENCODE_BATCH = 512

DEFAULT_GLOSSARY_NEG_PATH = ""
DEFAULT_GLOSSARY_NEG_REFRESH_STEPS = 200
DEFAULT_GLOSSARY_NEG_ENCODE_BATCH = 512
DEFAULT_TEXT_LR_SCALE = 0.1

SYNTH_UTTER_PREFIX = "wiki_synth_"
AUGMENT_SNR_MIN_DB = 10.0
AUGMENT_SNR_MAX_DB = 30.0
AUGMENT_SPEED_MIN = 0.9
AUGMENT_SPEED_MAX = 1.1
AUGMENT_REVERB_PROB = 0.3
AUGMENT_REVERB_DECAY_MIN = 0.1
AUGMENT_REVERB_DECAY_MAX = 0.4
AUGMENT_REVERB_DELAY_MIN_MS = 10
AUGMENT_REVERB_DELAY_MAX_MS = 40

INVALID_ID_SENTINEL = 0
SIGNED_INT64_MASK = (1 << 63) - 1
# ======Configuration=====


# ==================== Hashing helpers ====================


def stable_term_id(term_text: str) -> int:
    if not term_text:
        return INVALID_ID_SENTINEL
    digest = hashlib.blake2b(term_text.encode("utf-8"), digest_size=8).digest()
    tid = int.from_bytes(digest, byteorder="little", signed=False) & SIGNED_INT64_MASK
    return tid if tid != INVALID_ID_SENTINEL else 1


def stable_group_id(group_key: str) -> int:
    if not group_key:
        return INVALID_ID_SENTINEL
    digest = hashlib.blake2b(group_key.encode("utf-8"), digest_size=8).digest()
    gid = int.from_bytes(digest, byteorder="little", signed=False) & SIGNED_INT64_MASK
    return gid if gid != INVALID_ID_SENTINEL else 1


def build_speech_group_key(sample: Dict[str, Any]) -> str:
    utter_id = str(sample.get("utter_id", "") or "").strip()
    chunk_idx = str(sample.get("chunk_idx", "") or "").strip()
    if utter_id and chunk_idx:
        return f"{utter_id}::{chunk_idx}"
    path = str(sample.get("chunk_audio_path", "") or "").strip()
    if path:
        return f"path::{path}"
    return ""


# ==================== Model Components ====================


class AttentivePooling(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        scores = self.attention(x)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), -1e9)
        weights = F.softmax(scores, dim=1)
        return torch.sum(x * weights, dim=1)


class GatherLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor):
        ctx.save_for_backward(input_tensor)
        outputs = [torch.zeros_like(input_tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(outputs, input_tensor)
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grads):
        (input_tensor,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input_tensor)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


def all_gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        gathered = GatherLayer.apply(tensor)
        return torch.cat(gathered, dim=0)
    return tensor


class BgeM3TextEncoder(nn.Module):
    def __init__(
        self,
        model_id: str,
        lora_rank: int,
        lora_alpha: int,
        target_modules: Optional[List[str]],
        full_finetune: bool = False,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, add_pooling_layer=False
        )
        self.full_finetune = full_finetune

        if full_finetune:
            for p in self.encoder.parameters():
                p.requires_grad = True
            self.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            total = sum(p.numel() for p in self.encoder.parameters())
            trainable = sum(
                p.numel() for p in self.encoder.parameters() if p.requires_grad
            )
            logger.info(
                f"[TEXT_ENCODER] Full finetune + gradient checkpointing: "
                f"trainable={trainable:,} / total={total:,} "
                f"({100.0 * trainable / total:.2f}%)"
            )
        else:
            if target_modules is None:
                target_modules = ["query", "key", "value"]
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=DEFAULT_LORA_DROPOUT,
                bias="none",
                task_type=None,
            )
            self.encoder = get_peft_model(self.encoder, lora_config)
            self.encoder.print_trainable_parameters()

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return F.normalize(outputs.last_hidden_state[:, 0, :], p=2, dim=-1)


class Qwen3OmniRetriever(nn.Module):
    def __init__(
        self,
        model_id: str,
        target_dim: int,
        use_lora: bool,
        lora_rank: int,
        lora_alpha: int,
        lora_target_modules: Optional[List[str]],
        temperature: float,
        learn_temp: bool,
    ):
        super().__init__()
        self.audio_encoder = Qwen3OmniMoeAudioEncoder.from_pretrained(
            model_id, dtype=torch.bfloat16
        )
        if hasattr(self.audio_encoder, "conv2d1"):
            self.audio_encoder.get_input_embeddings = (
                lambda: self.audio_encoder.conv2d1
            )
        self.audio_encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        if use_lora:
            if lora_target_modules is None:
                lora_target_modules = [
                    "q_proj", "k_proj", "v_proj", "out_proj",
                    "fc1", "fc2", "proj1", "proj2",
                ]
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=DEFAULT_LORA_DROPOUT,
                bias="none",
                task_type=None,
            )
            self.audio_encoder = get_peft_model(self.audio_encoder, lora_config)
            self.audio_encoder.print_trainable_parameters()
        else:
            for p in self.audio_encoder.parameters():
                p.requires_grad = False

        self.pooler = AttentivePooling(DEFAULT_AUDIO_HIDDEN_DIM)
        self.projector = nn.Linear(DEFAULT_AUDIO_HIDDEN_DIM, target_dim)

        if learn_temp:
            self.logit_scale = nn.Parameter(
                torch.ones([]) * np.log(1.0 / temperature)
            )
        else:
            self.register_buffer(
                "logit_scale", torch.tensor(np.log(1.0 / temperature))
            )

    def forward(
        self, input_features: torch.Tensor, feature_lens: torch.Tensor
    ) -> torch.Tensor:
        if input_features.ndim == 3:
            input_features = input_features.transpose(0, 1).reshape(
                input_features.shape[1], -1
            )
        outputs = self.audio_encoder(input_features, feature_lens)
        hidden_states = outputs.last_hidden_state

        if hidden_states.ndim == 2:
            output_lens: List[int] = []
            for cur in feature_lens.tolist():
                reduced = cur
                for _ in range(3):
                    reduced = (reduced + 1) // 2
                output_lens.append(reduced)
            if sum(output_lens) != hidden_states.shape[0]:
                ratio = input_features.shape[1] / hidden_states.shape[0]
                output_lens = [max(1, round(x / ratio)) for x in feature_lens.tolist()]
                output_lens[-1] = hidden_states.shape[0] - sum(output_lens[:-1])

            from torch.nn.utils.rnn import pad_sequence

            parts = torch.split(hidden_states, output_lens, dim=0)
            hidden_states = pad_sequence(parts, batch_first=True)
            feature_lens = torch.tensor(output_lens, device=hidden_states.device)

        batch_size, max_len, _ = hidden_states.shape
        mask = (
            torch.arange(max_len, device=hidden_states.device).expand(
                batch_size, max_len
            )
            < feature_lens.unsqueeze(1)
        )
        pooled = self.pooler(hidden_states, mask)
        projected = self.projector(pooled)
        return F.normalize(projected, p=2, dim=-1)


# ==================== Audio Augmentation ====================


def _add_gaussian_noise(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise at a random SNR between AUGMENT_SNR_MIN_DB and AUGMENT_SNR_MAX_DB."""
    snr_db = rng.uniform(AUGMENT_SNR_MIN_DB, AUGMENT_SNR_MAX_DB)
    signal_power = np.mean(audio ** 2)
    if signal_power < 1e-10:
        return audio
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0, np.sqrt(noise_power), size=audio.shape).astype(np.float32)
    return audio + noise


def _speed_perturb(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simple speed perturbation via linear interpolation (no librosa dependency)."""
    speed = rng.uniform(AUGMENT_SPEED_MIN, AUGMENT_SPEED_MAX)
    if abs(speed - 1.0) < 0.01:
        return audio
    orig_len = len(audio)
    new_len = int(orig_len / speed)
    if new_len < 1:
        return audio
    indices = np.linspace(0, orig_len - 1, new_len).astype(np.float32)
    idx_floor = np.floor(indices).astype(np.int64)
    idx_ceil = np.minimum(idx_floor + 1, orig_len - 1)
    frac = indices - idx_floor
    resampled = audio[idx_floor] * (1.0 - frac) + audio[idx_ceil] * frac
    return resampled.astype(np.float32)


def _simple_reverb(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate simple early reflections via a single decayed delay tap."""
    delay_ms = rng.uniform(AUGMENT_REVERB_DELAY_MIN_MS, AUGMENT_REVERB_DELAY_MAX_MS)
    decay = rng.uniform(AUGMENT_REVERB_DECAY_MIN, AUGMENT_REVERB_DECAY_MAX)
    delay_samples = int(delay_ms * DEFAULT_AUDIO_SAMPLE_RATE / 1000.0)
    if delay_samples >= len(audio) or delay_samples < 1:
        return audio
    reverbed = audio.copy()
    reverbed[delay_samples:] += decay * audio[:-delay_samples]
    return reverbed


def augment_synth_audio(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply augmentation chain to synthetic TTS audio to close domain gap with real speech."""
    audio = _speed_perturb(audio, rng)
    audio = _add_gaussian_noise(audio, rng)
    if rng.random() < AUGMENT_REVERB_PROB:
        audio = _simple_reverb(audio, rng)
    return audio


# ==================== Dataset ====================


class TermRAGDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict],
        force_dummy_audio: bool = False,
        augment_synth: bool = False,
    ):
        self.samples = samples
        self._remap_src = os.environ.get("AUDIO_PATH_REMAP_SRC", "").strip()
        self._remap_dst = os.environ.get("AUDIO_PATH_REMAP_DST", "").strip()
        self._force_dummy = force_dummy_audio
        self._augment_synth = augment_synth
        self._rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = dict(self.samples[idx])
        term_text = (sample.get("term_key", "") or "").strip().lower()

        if self._force_dummy:
            dummy = np.zeros(DEFAULT_FIXED_AUDIO_SAMPLES, dtype=np.float32)
            return {
                "audio": dummy,
                "term_text": term_text,
                "skip_sample": True,
                "chunk_audio_path": "DUMMY",
                "utter_id": str(sample.get("utter_id", "")),
                "chunk_idx": str(sample.get("chunk_idx", "")),
            }

        audio_path = sample.get("chunk_audio_path", "")
        if (
            self._remap_src
            and self._remap_dst
            and audio_path.startswith(self._remap_src)
        ):
            candidate = self._remap_dst + audio_path[len(self._remap_src) :]
            if os.path.exists(candidate):
                audio_path = candidate

        try:
            audio_data, sr = sf.read(audio_path)
            assert sr == DEFAULT_AUDIO_SAMPLE_RATE, (
                f"Expected {DEFAULT_AUDIO_SAMPLE_RATE}Hz, got {sr}Hz: {audio_path}"
            )
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
            is_synth = str(sample.get("utter_id", "")).startswith(SYNTH_UTTER_PREFIX)
            if self._augment_synth and is_synth:
                audio_data = augment_synth_audio(audio_data.astype(np.float32), self._rng)
            max_abs = np.max(np.abs(audio_data))
            if max_abs > 0:
                audio_data = audio_data / max_abs
            return {
                "audio": audio_data.astype(np.float32),
                "term_text": term_text,
                "skip_sample": False,
                "chunk_audio_path": audio_path,
                "utter_id": str(sample.get("utter_id", "")),
                "chunk_idx": str(sample.get("chunk_idx", "")),
            }
        except Exception as exc:
            logger.warning(f"[AUDIO_LOAD_FAIL] path={audio_path} error={exc}")
            return {
                "audio": None,
                "term_text": "",
                "skip_sample": True,
                "chunk_audio_path": audio_path,
                "utter_id": str(sample.get("utter_id", "")),
                "chunk_idx": str(sample.get("chunk_idx", "")),
            }


def collate_fn(batch: List[Dict], feature_extractor: WhisperFeatureExtractor) -> Dict:
    dummy_audio = np.zeros(DEFAULT_FIXED_AUDIO_SAMPLES, dtype=np.float32)
    audio_list: List[np.ndarray] = []
    text_list: List[str] = []
    valid_list: List[bool] = []
    samples: List[Dict] = []

    for s in batch:
        audio = s.get("audio")
        skip = bool(s.get("skip_sample", False))
        if audio is None or len(audio) <= DEFAULT_MIN_AUDIO_SAMPLES:
            audio = dummy_audio
            skip = True

        if len(audio) < DEFAULT_FIXED_AUDIO_SAMPLES:
            audio = np.pad(
                audio, (0, DEFAULT_FIXED_AUDIO_SAMPLES - len(audio)), mode="constant"
            )
        elif len(audio) > DEFAULT_FIXED_AUDIO_SAMPLES:
            audio = audio[:DEFAULT_FIXED_AUDIO_SAMPLES]

        audio_list.append(audio)
        text_list.append(s.get("term_text", ""))
        valid_list.append(bool(s.get("term_text")) and (not skip))
        samples.append(s)

    inputs = feature_extractor(
        audio_list,
        sampling_rate=DEFAULT_AUDIO_SAMPLE_RATE,
        return_tensors="pt",
        padding=False,
    )
    feats = inputs.input_features
    feat_lens = torch.full((feats.size(0),), feats.size(-1), dtype=torch.long)

    return {
        "input_features": feats,
        "feature_lens": feat_lens,
        "term_texts": text_list,
        "valid_mask": torch.tensor(valid_list, dtype=torch.bool),
        "samples": samples,
    }


# ==================== Negative Term Bank ====================


class NegativeTermBank:
    """
    Maintains a detached embedding cache of the full glossary.
    Periodically refreshed with the current text encoder weights.
    """

    def __init__(self, unique_terms: List[str], device: torch.device):
        assert len(unique_terms) > 0, "NegativeTermBank requires at least one term"
        self.terms = unique_terms
        self.term_ids = torch.tensor(
            [stable_term_id(t) for t in unique_terms], dtype=torch.long, device=device
        )
        self.embeddings: Optional[torch.Tensor] = None
        self._device = device
        self._last_refresh_step = -1

    @property
    def size(self) -> int:
        return len(self.terms)

    @torch.no_grad()
    def refresh(
        self,
        text_encoder: nn.Module,
        text_tokenizer,
        device: torch.device,
        batch_size: int = DEFAULT_NEG_BANK_ENCODE_BATCH,
    ) -> None:
        text_encoder.eval()
        all_embs: List[torch.Tensor] = []
        for i in range(0, len(self.terms), batch_size):
            chunk = self.terms[i : i + batch_size]
            tok = text_tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=DEFAULT_TEXT_MAX_LENGTH,
                return_tensors="pt",
            ).to(device)
            embs = text_encoder(tok.input_ids, tok.attention_mask)
            all_embs.append(embs.float().cpu())
        self.embeddings = torch.cat(all_embs, dim=0)
        text_encoder.train()

    def sample(
        self, k: int, rng: random.Random
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.embeddings is not None, (
            "NegativeTermBank.refresh() must be called before sample()"
        )
        n = self.size
        k = min(k, n)
        indices = rng.sample(range(n), k)
        idx_t = torch.tensor(indices, dtype=torch.long)
        embs = self.embeddings[idx_t].to(self._device)
        tids = self.term_ids[idx_t]
        return embs, tids

    @torch.no_grad()
    def mine_hard_negatives(
        self,
        speech_embs: torch.Tensor,
        local_term_ids: torch.Tensor,
        local_valid_mask: torch.Tensor,
        k_per_sample: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        For each valid speech embedding, find the top-k bank terms with highest
        similarity that are NOT the GT term.  Returns deduplicated
        (embeddings, term_ids, num_unique) across the whole batch.
        """
        assert self.embeddings is not None, "refresh() must be called first"

        speech_f32 = speech_embs.detach().float()
        bank_embs = self.embeddings.to(speech_f32.device)
        bank_tids = self.term_ids.to(local_term_ids.device)

        sims = speech_f32 @ bank_embs.T

        gt_match = local_term_ids.unsqueeze(1) == bank_tids.unsqueeze(0)
        sims.masked_fill_(gt_match, -1e9)
        sims.masked_fill_(~local_valid_mask.unsqueeze(1), -1e9)

        actual_k = min(k_per_sample, self.size)
        _, topk_idx = sims.topk(actual_k, dim=1)

        unique_idx = topk_idx.reshape(-1).unique().cpu()
        return (
            self.embeddings[unique_idx].to(self._device),
            self.term_ids[unique_idx],
            unique_idx.numel(),
        )


# ==================== Loss ====================


def compute_masked_contrastive_loss(
    speech_embs: torch.Tensor,
    text_embs: torch.Tensor,
    logit_scale: torch.Tensor,
    local_group_ids: torch.Tensor,
    local_term_ids: torch.Tensor,
    local_valid_mask: torch.Tensor,
    neg_bank_embs: Optional[torch.Tensor] = None,
    neg_bank_term_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Masked multi-positive InfoNCE with optional global negative bank.

    Positive:  same group_id (same chunk) AND both valid.
    Masked:    same term_id but different group_id (false negative) → excluded from denom.
    Negative:  everything else (different chunk, different term).
    Bank:      appended as extra columns; false-neg-masked if term_id matches any anchor.
    """
    world_size = (
        dist.get_world_size()
        if dist.is_available() and dist.is_initialized()
        else 1
    )
    device = speech_embs.device

    # 1. Gather in-batch text embeddings + metadata across all GPUs
    global_text_embs = all_gather_with_grad(text_embs)

    if world_size > 1:
        gathered_gids = [torch.zeros_like(local_group_ids) for _ in range(world_size)]
        gathered_tids = [torch.zeros_like(local_term_ids) for _ in range(world_size)]
        gathered_valid = [torch.zeros_like(local_valid_mask) for _ in range(world_size)]
        dist.all_gather(gathered_gids, local_group_ids)
        dist.all_gather(gathered_tids, local_term_ids)
        dist.all_gather(gathered_valid, local_valid_mask)
        global_group_ids = torch.cat(gathered_gids, dim=0)
        global_term_ids = torch.cat(gathered_tids, dim=0)
        global_valid_mask = torch.cat(gathered_valid, dim=0)
    else:
        global_group_ids = local_group_ids
        global_term_ids = local_term_ids
        global_valid_mask = local_valid_mask

    # 2. Append global negative bank (detached, no gradient)
    if neg_bank_embs is not None and neg_bank_term_ids is not None:
        k = neg_bank_embs.size(0)
        global_text_embs = torch.cat(
            [global_text_embs, neg_bank_embs.detach().to(device)], dim=0
        )
        global_group_ids = torch.cat(
            [global_group_ids, torch.zeros(k, dtype=torch.long, device=device)]
        )
        global_term_ids = torch.cat(
            [global_term_ids, neg_bank_term_ids.to(device)]
        )
        global_valid_mask = torch.cat(
            [global_valid_mask, torch.ones(k, dtype=torch.bool, device=device)]
        )

    # 3. Similarity matrix  [B_local, N_global]
    logits = (speech_embs @ global_text_embs.T) * logit_scale

    # 4. Positive mask: same chunk (group_id) AND both sides valid
    pos_mask = local_group_ids.unsqueeze(1) == global_group_ids.unsqueeze(0)
    pos_mask = pos_mask & local_valid_mask.unsqueeze(1) & global_valid_mask.unsqueeze(0)

    # 5. False-negative mask: same term content (term_id) but different chunk (group_id)
    same_term = local_term_ids.unsqueeze(1) == global_term_ids.unsqueeze(0)
    fn_mask = same_term & ~pos_mask

    # 6. Apply masks: invalid columns and false negatives → -inf
    logits = logits.masked_fill(~global_valid_mask.unsqueeze(0), -1e9)
    logits = logits.masked_fill(fn_mask, -1e9)

    # 7. Multi-positive InfoNCE
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_count = pos_mask.sum(dim=1)
    row_valid = (local_valid_mask & (pos_count > 0)).float()

    loss_per_row = -(
        (log_prob * pos_mask.float()).sum(dim=1) / pos_count.clamp(min=1).float()
    )
    loss = (loss_per_row * row_valid).sum() / row_valid.sum().clamp(min=1.0)
    return loss


# ==================== Evaluation ====================


@torch.no_grad()
def _encode_terms_batch(
    text_encoder: nn.Module,
    text_tokenizer,
    terms: List[str],
    device: torch.device,
    batch_size: int = DEFAULT_EVAL_TERM_ENCODE_BATCH,
) -> torch.Tensor:
    """Encode a list of term strings through text encoder. Returns [N, D] float32 cpu."""
    text_encoder.eval()
    all_embs: List[torch.Tensor] = []
    for i in range(0, len(terms), batch_size):
        chunk = terms[i : i + batch_size]
        tok = text_tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=DEFAULT_TEXT_MAX_LENGTH,
            return_tensors="pt",
        ).to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            embs = text_encoder(tok.input_ids, tok.attention_mask)
        all_embs.append(embs.float().cpu())
    return torch.cat(all_embs, dim=0)


def _load_eval_wiki_terms(wiki_path: str) -> List[str]:
    """Load wiki glossary JSON for eval glossary-scale expansion."""
    assert os.path.isfile(wiki_path), f"Eval wiki glossary not found: {wiki_path}"
    with open(wiki_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    assert isinstance(entries, list)
    seen: set = set()
    terms: List[str] = []
    for e in entries:
        t = e["term"].strip().lower()
        if t and t not in seen:
            seen.add(t)
            terms.append(t)
    return terms


def run_sample_eval(
    retriever: nn.Module,
    text_encoder: nn.Module,
    text_tokenizer,
    eval_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    global_step: int,
    epoch: int,
    wandb_run,
    eval_name: str = "dev",
    wiki_terms: Optional[List[str]] = None,
    glossary_sizes: Optional[List[int]] = None,
) -> Dict[str, float]:
    retriever.eval()
    text_encoder.eval()
    t0 = time.time()

    speech_emb_list: List[torch.Tensor] = []
    text_emb_list: List[torch.Tensor] = []
    valid_list: List[torch.Tensor] = []
    term_text_list: List[str] = []
    group_id_list: List[torch.Tensor] = []
    term_id_list: List[torch.Tensor] = []
    sample_list: List[Dict] = []

    with torch.no_grad():
        for batch in eval_loader:
            feats = batch["input_features"].to(device).to(torch.bfloat16)
            flens = batch["feature_lens"].to(device)
            texts = batch["term_texts"]
            valid = batch["valid_mask"].to(device)
            samples = batch["samples"]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                s_embs = retriever(feats, flens)
                tok = text_tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=DEFAULT_TEXT_MAX_LENGTH,
                    return_tensors="pt",
                ).to(device)
                t_embs = text_encoder(tok.input_ids, tok.attention_mask)

            speech_emb_list.append(s_embs.float().cpu())
            text_emb_list.append(t_embs.float().cpu())
            valid_list.append(valid.bool().cpu())
            term_text_list.extend([t.strip().lower() for t in texts])
            group_id_list.append(
                torch.tensor(
                    [stable_group_id(build_speech_group_key(s)) for s in samples],
                    dtype=torch.long,
                )
            )
            term_id_list.append(
                torch.tensor(
                    [stable_term_id(s.get("term_text", "") or "") for s in samples],
                    dtype=torch.long,
                )
            )
            sample_list.extend(samples)

    if not speech_emb_list:
        retriever.train()
        text_encoder.train()
        return {}

    speech_embs = torch.cat(speech_emb_list, dim=0)
    text_embs = torch.cat(text_emb_list, dim=0)
    valid_mask = torch.cat(valid_list, dim=0)
    group_ids = torch.cat(group_id_list, dim=0)
    term_ids = torch.cat(term_id_list, dim=0)

    valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1).tolist()
    if not valid_indices:
        retriever.train()
        text_encoder.train()
        return {}

    # ---- Eval loss: masked InfoNCE over the full dev set (no gradient) ----
    # Uses fixed temperature=1/args.temperature (no logit_scale parameter lookup)
    logit_scale_eval = float(np.log(1.0 / args.temperature))
    logit_scale_t = torch.tensor(np.exp(logit_scale_eval), dtype=torch.float32)

    sim = (speech_embs @ text_embs.T) * logit_scale_t  # [N, N]

    pos_mask_eval = group_ids.unsqueeze(1) == group_ids.unsqueeze(0)
    pos_mask_eval = pos_mask_eval & valid_mask.unsqueeze(1) & valid_mask.unsqueeze(0)

    fn_mask_eval = (
        (term_ids.unsqueeze(1) == term_ids.unsqueeze(0)) & ~pos_mask_eval
    )
    # Mask invalid and false-negative columns from softmax denominator
    sim = sim.masked_fill(~valid_mask.unsqueeze(0), -1e9)
    sim = sim.masked_fill(fn_mask_eval, -1e9)

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_count = pos_mask_eval.sum(dim=1)
    row_valid = (valid_mask & (pos_count > 0)).float()
    loss_per_row = -(
        (log_prob * pos_mask_eval.float()).sum(dim=1)
        / pos_count.clamp(min=1).float()
    )
    eval_loss = (
        (loss_per_row * row_valid).sum() / row_valid.sum().clamp(min=1.0)
    ).item()

    # ---- Recall metrics: deduplicated term bank ----
    term_to_bank: Dict[str, int] = {}
    bank_rows: List[torch.Tensor] = []
    for idx in valid_indices:
        if idx >= len(term_text_list):
            continue
        t = term_text_list[idx]
        if not t or t in term_to_bank:
            continue
        term_to_bank[t] = len(bank_rows)
        bank_rows.append(text_embs[idx])

    if not bank_rows:
        retriever.train()
        text_encoder.train()
        return {}

    bank_embs = torch.stack(bank_rows, dim=0)
    speech_valid = speech_embs[valid_indices]
    recall_logits = speech_valid @ bank_embs.t()

    targets: List[int] = []
    row_keep: List[int] = []
    for row_idx, sample_idx in enumerate(valid_indices):
        if sample_idx >= len(term_text_list):
            continue
        t = term_text_list[sample_idx]
        target = term_to_bank.get(t)
        if target is None:
            continue
        targets.append(target)
        row_keep.append(row_idx)

    if not targets:
        retriever.train()
        text_encoder.train()
        return {}

    recall_logits = recall_logits[row_keep]
    targets_t = torch.tensor(targets, dtype=torch.long)

    top1 = (recall_logits.argmax(dim=1) == targets_t).float().mean().item()
    k_primary = min(args.eval_topk, recall_logits.size(1))
    k_extra = min(args.eval_topk_extra, recall_logits.size(1))
    recall_primary = (
        torch.topk(recall_logits, k=k_primary, dim=1)
        .indices.eq(targets_t.unsqueeze(1))
        .any(dim=1)
        .float()
        .mean()
        .item()
    )
    recall_extra = (
        torch.topk(recall_logits, k=k_extra, dim=1)
        .indices.eq(targets_t.unsqueeze(1))
        .any(dim=1)
        .float()
        .mean()
        .item()
    )

    prefix = f"eval_{eval_name}"
    metrics: Dict[str, float] = {
        f"{prefix}/loss": eval_loss,
        f"{prefix}/top1": top1,
        f"{prefix}/recall@{k_primary}": recall_primary,
        f"{prefix}/recall@{k_extra}": recall_extra,
    }

    elapsed = time.time() - t0
    log_parts = [
        f"[EVAL_{eval_name.upper()}] step={global_step} epoch={epoch}",
        f"samples={len(valid_indices)} bank_terms={len(bank_rows)}",
        f"loss={eval_loss:.6f}",
        f"top1={top1:.4f}",
        f"recall@{k_primary}={recall_primary:.4f}",
        f"recall@{k_extra}={recall_extra:.4f}",
    ]

    # ---- Glossary-scale recall ----
    gt_bank_size = len(bank_rows)
    gt_terms_set = set(term_to_bank.keys())
    effective_glossary_sizes = glossary_sizes or []

    if wiki_terms and effective_glossary_sizes:
        wiki_filtered = [t for t in wiki_terms if t not in gt_terms_set]
        wiki_embs = _encode_terms_batch(
            text_encoder, text_tokenizer, wiki_filtered, device
        )
        for gs in effective_glossary_sizes:
            n_extra = gs - gt_bank_size
            if n_extra <= 0:
                logger.info(
                    f"[EVAL_{eval_name.upper()}] gs{gs} skipped: "
                    f"GT bank ({gt_bank_size}) already >= {gs}"
                )
                continue
            n_wiki_add = min(n_extra, len(wiki_filtered))
            expanded_bank = torch.cat(
                [bank_embs, wiki_embs[:n_wiki_add]], dim=0
            )
            expanded_logits = speech_valid @ expanded_bank.T
            gs_kp = min(k_primary, expanded_logits.size(1))
            gs_ke = min(k_extra, expanded_logits.size(1))
            gs_recall_p = (
                torch.topk(expanded_logits, k=gs_kp, dim=1)
                .indices.eq(targets_t.unsqueeze(1))
                .any(dim=1)
                .float()
                .mean()
                .item()
            )
            gs_recall_e = (
                torch.topk(expanded_logits, k=gs_ke, dim=1)
                .indices.eq(targets_t.unsqueeze(1))
                .any(dim=1)
                .float()
                .mean()
                .item()
            )
            metrics[f"{prefix}/recall@{gs_kp}_gs{gs}"] = gs_recall_p
            metrics[f"{prefix}/recall@{gs_ke}_gs{gs}"] = gs_recall_e
            actual_bank = expanded_bank.size(0)
            log_parts.append(
                f"gs{gs}(bank={actual_bank}): "
                f"r@{gs_kp}={gs_recall_p:.4f} r@{gs_ke}={gs_recall_e:.4f}"
            )

    log_parts.append(f"elapsed={elapsed:.2f}s")
    logger.info("  ".join(log_parts))

    if wandb_run is not None:
        wandb_payload = {k: v for k, v in metrics.items()}
        wandb_payload[f"{prefix}/bank_terms"] = gt_bank_size
        wandb_payload[f"{prefix}/step"] = global_step
        wandb_run.log(wandb_payload, step=global_step)

    retriever.train()
    text_encoder.train()
    return metrics


# ==================== Training ====================


def train(rank: int, world_size: int, args: argparse.Namespace) -> None:
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=DEFAULT_DDP_TIMEOUT_SECONDS),
        )
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    is_main = rank == 0

    # ---- Models ----
    retriever = Qwen3OmniRetriever(
        model_id=args.audio_model_id,
        target_dim=args.target_dim,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        temperature=args.temperature,
        learn_temp=args.learn_temp,
    ).to(device)

    text_encoder = BgeM3TextEncoder(
        model_id=args.text_model_id,
        lora_rank=args.text_lora_rank,
        lora_alpha=args.text_lora_alpha,
        target_modules=args.text_lora_target_modules,
        full_finetune=args.text_full_finetune,
    ).to(device)
    text_tokenizer = AutoTokenizer.from_pretrained(args.text_model_id)
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")

    if world_size > 1:
        retriever = DDP(retriever, device_ids=[rank])
        text_encoder = DDP(text_encoder, device_ids=[rank])

    raw_retriever = retriever.module if world_size > 1 else retriever
    raw_text_encoder = text_encoder.module if world_size > 1 else text_encoder

    # ---- Optimizer ----
    audio_lora_params = [
        p for p in raw_retriever.audio_encoder.parameters() if p.requires_grad
    ]
    text_trainable_params = [
        p for p in raw_text_encoder.encoder.parameters() if p.requires_grad
    ]
    text_lr = args.text_lr if args.text_lr > 0 else (
        args.lr * DEFAULT_TEXT_LR_SCALE if args.text_full_finetune else args.lr
    )
    head_params = list(raw_retriever.pooler.parameters()) + list(
        raw_retriever.projector.parameters()
    )
    if args.learn_temp:
        head_params.append(raw_retriever.logit_scale)

    opt_groups = []
    if audio_lora_params:
        opt_groups.append(
            {"params": audio_lora_params, "lr": args.lr, "name": "audio_lora"}
        )
    text_group_name = "text_full" if args.text_full_finetune else "text_lora"
    if text_trainable_params:
        opt_groups.append(
            {"params": text_trainable_params, "lr": text_lr, "name": text_group_name}
        )
    opt_groups.append(
        {
            "params": head_params,
            "lr": args.lr * DEFAULT_HEAD_LR_SCALE,
            "name": "head",
        }
    )

    optimizer = torch.optim.AdamW(opt_groups, weight_decay=DEFAULT_WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    # ---- Resume ----
    start_epoch = 0
    global_step = 0
    pending_scheduler_state = None
    pending_scaler_state = None

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)

        def _strip(sd):
            if any(k.startswith("module.") for k in sd):
                return {
                    (k[len("module.") :] if k.startswith("module.") else k): v
                    for k, v in sd.items()
                }
            return sd

        raw_retriever.load_state_dict(_strip(ckpt.get("model_state_dict", {})), strict=False)
        if "text_model_state_dict" in ckpt:
            raw_text_encoder.load_state_dict(
                _strip(ckpt["text_model_state_dict"]), strict=False
            )
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as exc:
                logger.warning(f"[RESUME] optimizer load failed: {exc}")

        pending_scheduler_state = ckpt.get("scheduler_state_dict")
        pending_scaler_state = ckpt.get("scaler_state_dict")
        start_epoch = ckpt.get("epoch", -1) + 1
        global_step = ckpt.get("global_step", 0)
        if is_main:
            logger.info(
                f"[RESUME] {args.resume} epoch={start_epoch} step={global_step}"
            )

    # ---- Data ----
    train_samples: List[Dict] = []
    wiki_rank_skipped = 0
    noisy_kept = 0
    clean_kept = 0
    noisy_dropped = 0
    clean_dropped = 0
    _nr_rng = random.Random(42)
    with open(args.train_jsonl, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if args.train_limit and idx >= args.train_limit:
                break
            try:
                sample = json.loads(line)
            except Exception:
                continue
            if args.wiki_rank > 0:
                p31_rank = sample.get("p31_rank", -1)
                if p31_rank >= 0 and p31_rank >= args.wiki_rank:
                    wiki_rank_skipped += 1
                    continue
            audio_type = sample.get("audio_type", "")
            if audio_type in ("clean", "noisy") and args.noisy_ratio >= 0:
                if audio_type == "noisy":
                    if _nr_rng.random() >= args.noisy_ratio:
                        noisy_dropped += 1
                        continue
                    noisy_kept += 1
                else:
                    if _nr_rng.random() >= (1.0 - args.noisy_ratio):
                        clean_dropped += 1
                        continue
                    clean_kept += 1
            elif audio_type in ("clean", "noisy"):
                if audio_type == "noisy":
                    noisy_kept += 1
                else:
                    clean_kept += 1
            train_samples.append(sample)
    if args.wiki_rank > 0 and rank == 0:
        logger.info(
            f"[DATA] wiki_rank={args.wiki_rank}: kept {len(train_samples):,}, "
            f"skipped {wiki_rank_skipped:,} wiki entries beyond rank cutoff"
        )
    if rank == 0 and (noisy_kept + clean_kept + noisy_dropped + clean_dropped > 0):
        logger.info(
            f"[DATA] noisy_ratio={args.noisy_ratio}: "
            f"noisy={noisy_kept:,} kept / {noisy_dropped:,} dropped, "
            f"clean={clean_kept:,} kept / {clean_dropped:,} dropped"
        )

    dev_samples: List[Dict] = []
    if args.dev_jsonl:
        with open(args.dev_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    dev_samples.append(json.loads(line))
                except Exception:
                    continue

    dataset = TermRAGDataset(
        train_samples,
        force_dummy_audio=args.force_dummy_audio,
        augment_synth=args.augment_synth,
    )
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1
        else None
    )
    per_rank_bs = args.batch_size // world_size
    train_loader = DataLoader(
        dataset,
        batch_size=per_rank_bs,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=lambda b: collate_fn(b, feature_extractor),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    eval_loader = None
    if dev_samples:
        eval_dataset = TermRAGDataset(dev_samples, force_dummy_audio=args.force_dummy_audio)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, feature_extractor),
            num_workers=4,
            pin_memory=True,
        )

    # ---- ACL6060 dev data (cross-domain eval) ----
    acl_dev_samples: List[Dict] = []
    if args.acl_dev_jsonl and os.path.isfile(args.acl_dev_jsonl):
        with open(args.acl_dev_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    acl_dev_samples.append(json.loads(line))
                except Exception:
                    continue

    acl_eval_loader: Optional[DataLoader] = None
    if acl_dev_samples:
        acl_eval_dataset = TermRAGDataset(
            acl_dev_samples, force_dummy_audio=args.force_dummy_audio
        )
        acl_eval_loader = DataLoader(
            acl_eval_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, feature_extractor),
            num_workers=4,
            pin_memory=True,
        )

    # ---- Wiki terms for eval glossary-scale ----
    eval_wiki_terms: Optional[List[str]] = None
    eval_glossary_sizes: List[int] = args.eval_glossary_sizes or []
    if args.eval_wiki_glossary and eval_glossary_sizes:
        eval_wiki_terms = _load_eval_wiki_terms(args.eval_wiki_glossary)
        if is_main:
            logger.info(
                f"[EVAL] Wiki glossary loaded: {len(eval_wiki_terms)} terms, "
                f"glossary_sizes={eval_glossary_sizes}"
            )

    # ---- Negative bank ----
    neg_bank: Optional[NegativeTermBank] = None
    neg_bank_rng = random.Random(42)
    use_neg_bank = args.neg_bank_size > 0 or args.hard_neg_k > 0
    if use_neg_bank:
        train_terms = sorted(
            {(s.get("term_key", "") or "").strip().lower() for s in train_samples}
            - {""}
        )
        assert len(train_terms) > 0, "No valid terms found for negative bank"

        wiki_terms: List[str] = []
        if args.hard_neg_glossary:
            assert os.path.isfile(args.hard_neg_glossary), (
                f"hard_neg_glossary not found: {args.hard_neg_glossary}"
            )
            with open(args.hard_neg_glossary, "r", encoding="utf-8") as f:
                wiki_entries = json.load(f)
            assert isinstance(wiki_entries, list), f"Expected list, got {type(wiki_entries)}"
            wiki_terms = [e["term"].strip().lower() for e in wiki_entries]

        train_set = set(train_terms)
        wiki_new = [t for t in wiki_terms if t and t not in train_set]
        all_unique_terms = train_terms + sorted(set(wiki_new))

        neg_bank = NegativeTermBank(all_unique_terms, device)
        bank_mode = "hard_neg" if args.hard_neg_k > 0 else "random"
        if is_main:
            logger.info(
                f"[NEG_BANK] mode={bank_mode} "
                f"train_terms={len(train_terms)} wiki_terms={len(wiki_new)} "
                f"total_unique={neg_bank.size} "
                f"hard_neg_k={args.hard_neg_k} random_sample={args.neg_bank_size} "
                f"refresh_every={args.neg_bank_refresh_steps} steps"
            )

    # ---- Glossary negatives (always-on, full glossary in similarity matrix) ----
    glossary_neg_terms: List[str] = []
    glossary_neg_term_ids: Optional[torch.Tensor] = None
    glossary_neg_embs: Optional[torch.Tensor] = None
    if args.glossary_neg_path:
        assert os.path.isfile(args.glossary_neg_path), (
            f"glossary_neg_path not found: {args.glossary_neg_path}"
        )
        glossary_neg_terms = _load_eval_wiki_terms(args.glossary_neg_path)
        assert len(glossary_neg_terms) > 0, "Glossary neg file loaded 0 terms"
        glossary_neg_term_ids = torch.tensor(
            [stable_term_id(t) for t in glossary_neg_terms], dtype=torch.long
        )
        if is_main:
            logger.info(
                f"[GLOSSARY_NEG] Loaded {len(glossary_neg_terms)} terms, "
                f"refresh every {args.glossary_neg_refresh_steps} steps"
            )

    # ---- Scheduler ----
    total_steps = len(train_loader) * args.epochs
    if args.lr_finder_steps > 0:
        warmup_steps = 0
        lr_finder_total = args.lr_finder_steps
        if is_main:
            logger.info(
                f"[LR_FINDER] Exponential sweep from {args.lr_finder_start} to "
                f"{args.lr_finder_end} over {lr_finder_total} steps"
            )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=(args.lr_finder_end / args.lr_finder_start) ** (1.0 / lr_finder_total),
        )
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr_finder_start
    else:
        if args.warmup_steps >= 0:
            warmup_steps = args.warmup_steps
        else:
            warmup_steps = int(total_steps * DEFAULT_WARMUP_RATIO)
        if is_main:
            logger.info(f"[SCHEDULER] total_steps={total_steps} warmup_steps={warmup_steps}")
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    if pending_scheduler_state:
        try:
            scheduler.load_state_dict(pending_scheduler_state)
        except Exception as exc:
            logger.warning(f"[RESUME] scheduler load failed, stepping to global_step: {exc}")
            for _ in range(global_step):
                scheduler.step()
    if pending_scaler_state:
        try:
            scaler.load_state_dict(pending_scaler_state)
        except Exception as exc:
            logger.warning(f"[RESUME] scaler load failed: {exc}")

    # ---- WandB ----
    wandb_run = None
    if is_main and args.enable_wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_exp_name,
                config=vars(args),
            )
            wandb.define_metric("train/step")
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("eval_dev/step")
            wandb.define_metric("eval_dev/*", step_metric="eval_dev/step")
            wandb.define_metric("eval_acl6060/step")
            wandb.define_metric("eval_acl6060/*", step_metric="eval_acl6060/step")
        except Exception as exc:
            logger.warning(f"[WANDB] init failed: {exc}")

    recent_ckpts: List[str] = []
    best_metric_value = float("-inf")
    best_metric_key = args.best_metric or ""
    best_metric_secondary_value = float("-inf")
    best_metric_secondary_key = args.best_metric_secondary or ""

    if is_main:
        logger.info(
            f"[SETUP] train={len(train_samples)} dev={len(dev_samples)} "
            f"acl_dev={len(acl_dev_samples)} "
            f"world_size={world_size} per_rank_bs={per_rank_bs} "
            f"total_steps={total_steps} neg_bank={'ON' if neg_bank else 'OFF'}"
        )
        logger.info(
            f"[SETUP] text_encoder: "
            f"full_finetune={args.text_full_finetune} text_lr={text_lr:.2e} "
            f"glossary_neg={len(glossary_neg_terms)} terms "
            f"augment_synth={args.augment_synth}"
        )
        if args.augment_synth:
            synth_count = sum(
                1 for s in train_samples
                if str(s.get("utter_id", "")).startswith(SYNTH_UTTER_PREFIX)
            )
            logger.info(
                f"[SETUP] Synth augmentation ON: {synth_count}/{len(train_samples)} "
                f"entries will receive noise/speed/reverb augmentation"
            )
        if best_metric_key:
            logger.info(f"[SETUP] Best checkpoint metric: {best_metric_key}")
        if best_metric_secondary_key:
            logger.info(
                f"[SETUP] Secondary best checkpoint metric: "
                f"{best_metric_secondary_key}"
            )
        if eval_glossary_sizes:
            logger.info(f"[SETUP] Eval glossary sizes: {eval_glossary_sizes}")

    # ==================== Training loop ====================
    t_train_start = time.time()
    time_budget_exceeded = False
    lr_finder_active = args.lr_finder_steps > 0
    lr_finder_best_loss = float("inf")
    lr_finder_results: list[tuple[float, float]] = []

    for epoch in range(start_epoch, args.epochs):
        retriever.train()
        text_encoder.train()
        if sampler is not None:
            sampler.set_epoch(epoch)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not is_main)
        for batch in pbar:
            global_step += 1

            # Refresh neg bank if needed (all ranks, deterministic)
            if (
                neg_bank is not None
                and args.neg_bank_refresh_steps > 0
                and (
                    neg_bank.embeddings is None
                    or global_step % args.neg_bank_refresh_steps == 1
                )
            ):
                if is_main:
                    logger.info(
                        f"[NEG_BANK] Refreshing at step {global_step} ..."
                    )
                neg_bank.refresh(raw_text_encoder, text_tokenizer, device)
                if world_size > 1:
                    dist.barrier()

            # Refresh glossary negatives (all ranks, deterministic)
            if (
                glossary_neg_terms
                and args.glossary_neg_refresh_steps > 0
                and (
                    glossary_neg_embs is None
                    or global_step % args.glossary_neg_refresh_steps == 1
                )
            ):
                if is_main:
                    logger.info(
                        f"[GLOSSARY_NEG] Encoding {len(glossary_neg_terms)} terms "
                        f"at step {global_step} ..."
                    )
                raw_text_encoder.eval()
                glossary_neg_embs = _encode_terms_batch(
                    raw_text_encoder, text_tokenizer, glossary_neg_terms,
                    device, batch_size=DEFAULT_GLOSSARY_NEG_ENCODE_BATCH,
                )
                raw_text_encoder.train()
                if is_main:
                    logger.info(
                        f"[GLOSSARY_NEG] Done: shape={glossary_neg_embs.shape}"
                    )
                if world_size > 1:
                    dist.barrier()

            # Eval
            if (
                args.eval_steps_sample > 0
                and global_step % args.eval_steps_sample == 0
                and eval_loader is not None
            ):
                if world_size > 1:
                    dist.barrier()
                all_eval_metrics: Dict[str, float] = {}
                if is_main:
                    dev_metrics = run_sample_eval(
                        raw_retriever,
                        raw_text_encoder,
                        text_tokenizer,
                        eval_loader,
                        device,
                        args,
                        global_step,
                        epoch,
                        wandb_run,
                        eval_name="dev",
                        wiki_terms=eval_wiki_terms,
                        glossary_sizes=eval_glossary_sizes,
                    )
                    all_eval_metrics.update(dev_metrics)

                    if acl_eval_loader is not None:
                        acl_metrics = run_sample_eval(
                            raw_retriever,
                            raw_text_encoder,
                            text_tokenizer,
                            acl_eval_loader,
                            device,
                            args,
                            global_step,
                            epoch,
                            wandb_run,
                            eval_name="acl6060",
                            wiki_terms=eval_wiki_terms,
                            glossary_sizes=eval_glossary_sizes,
                        )
                        all_eval_metrics.update(acl_metrics)

                    # Best checkpoint tracking
                    if (
                        best_metric_key
                        and best_metric_key in all_eval_metrics
                        and all_eval_metrics[best_metric_key] > best_metric_value
                    ):
                        best_metric_value = all_eval_metrics[best_metric_key]
                        best_path = args.save_path.replace(".pt", "_best.pt")
                        torch.save(
                            {
                                "model_state_dict": raw_retriever.state_dict(),
                                "text_model_state_dict": raw_text_encoder.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),
                                "scaler_state_dict": scaler.state_dict(),
                                "global_step": global_step,
                                "epoch": epoch,
                                "args": vars(args),
                                "best_metric_key": best_metric_key,
                                "best_metric_value": best_metric_value,
                            },
                            best_path,
                        )
                        logger.info(
                            f"[BEST] {best_metric_key}={best_metric_value:.4f} "
                            f"step={global_step} -> {best_path}"
                        )
                        if wandb_run is not None:
                            wandb_run.log(
                                {
                                    "best/metric_value": best_metric_value,
                                    "best/step": global_step,
                                },
                                step=global_step,
                            )

                    # Secondary best checkpoint (e.g. ACL6060 recall@k on gs10000)
                    if (
                        best_metric_secondary_key
                        and best_metric_secondary_key in all_eval_metrics
                        and all_eval_metrics[best_metric_secondary_key]
                        > best_metric_secondary_value
                    ):
                        best_metric_secondary_value = all_eval_metrics[
                            best_metric_secondary_key
                        ]
                        best_secondary_path = args.save_path.replace(
                            ".pt", "_best_acl6060_gs10000.pt"
                        )
                        torch.save(
                            {
                                "model_state_dict": raw_retriever.state_dict(),
                                "text_model_state_dict": raw_text_encoder.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),
                                "scaler_state_dict": scaler.state_dict(),
                                "global_step": global_step,
                                "epoch": epoch,
                                "args": vars(args),
                                "best_metric_key": best_metric_secondary_key,
                                "best_metric_value": best_metric_secondary_value,
                            },
                            best_secondary_path,
                        )
                        logger.info(
                            f"[BEST_SECONDARY] {best_metric_secondary_key}="
                            f"{best_metric_secondary_value:.4f} "
                            f"step={global_step} -> {best_secondary_path}"
                        )
                        if wandb_run is not None:
                            wandb_run.log(
                                {
                                    "best_secondary/metric_value": (
                                        best_metric_secondary_value
                                    ),
                                    "best_secondary/step": global_step,
                                },
                                step=global_step,
                            )

                if world_size > 1:
                    dist.barrier()

            # Checkpoint
            if is_main and global_step % args.save_steps == 0:
                ckpt_path = args.save_path.replace(".pt", f"_step_{global_step}.pt")
                torch.save(
                    {
                        "model_state_dict": raw_retriever.state_dict(),
                        "text_model_state_dict": raw_text_encoder.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "global_step": global_step,
                        "epoch": epoch,
                        "args": vars(args),
                    },
                    ckpt_path,
                )
                recent_ckpts.append(ckpt_path)
                logger.info(f"[CHECKPOINT] saved={ckpt_path}")
                while len(recent_ckpts) > args.keep_checkpoints:
                    old = recent_ckpts.pop(0)
                    if os.path.exists(old):
                        os.remove(old)
                        logger.info(f"[CHECKPOINT] removed_old={old}")

            # ---- Forward ----
            feats = batch["input_features"].to(device).to(torch.bfloat16)
            flens = batch["feature_lens"].to(device)
            texts = batch["term_texts"]
            valid = batch["valid_mask"].to(device)
            samples = batch["samples"]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                speech_embs = retriever(feats, flens)
                tok = text_tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=DEFAULT_TEXT_MAX_LENGTH,
                    return_tensors="pt",
                ).to(device)
                text_embs = text_encoder(tok.input_ids, tok.attention_mask)

                logit_scale = (
                    retriever.module.logit_scale.exp()
                    if world_size > 1
                    else retriever.logit_scale.exp()
                )

                group_ids = torch.tensor(
                    [stable_group_id(build_speech_group_key(s)) for s in samples],
                    dtype=torch.long,
                    device=device,
                )
                term_ids = torch.tensor(
                    [stable_term_id((s.get("term_text", "") or "")) for s in samples],
                    dtype=torch.long,
                    device=device,
                )

                # Negative bank: hard mining or random sampling
                nb_embs, nb_tids = None, None
                hard_neg_count = 0
                if neg_bank is not None and neg_bank.embeddings is not None:
                    if args.hard_neg_k > 0:
                        nb_embs, nb_tids, hard_neg_count = neg_bank.mine_hard_negatives(
                            speech_embs, term_ids, valid, args.hard_neg_k
                        )
                    elif args.neg_bank_size > 0:
                        nb_embs, nb_tids = neg_bank.sample(
                            args.neg_bank_size, neg_bank_rng
                        )
                    if nb_embs is not None:
                        nb_embs = nb_embs.to(torch.bfloat16)

                # Glossary negatives: append full glossary embeddings
                if glossary_neg_embs is not None:
                    gn_embs = glossary_neg_embs.to(device).to(torch.bfloat16)
                    gn_tids = glossary_neg_term_ids.to(device)
                    if nb_embs is not None:
                        nb_embs = torch.cat([nb_embs, gn_embs], dim=0)
                        nb_tids = torch.cat([nb_tids, gn_tids], dim=0)
                    else:
                        nb_embs = gn_embs
                        nb_tids = gn_tids

                total_loss = compute_masked_contrastive_loss(
                    speech_embs=speech_embs,
                    text_embs=text_embs,
                    logit_scale=logit_scale,
                    local_group_ids=group_ids,
                    local_term_ids=term_ids,
                    local_valid_mask=valid,
                    neg_bank_embs=nb_embs,
                    neg_bank_term_ids=nb_tids,
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            all_trainable = list(retriever.parameters()) + list(text_encoder.parameters())
            torch.nn.utils.clip_grad_norm_(
                all_trainable, max_norm=DEFAULT_GRAD_CLIP_MAX_NORM
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # ---- Logging ----
            if is_main and global_step % DEFAULT_LOG_INTERVAL == 0:
                ls_val = float(logit_scale.item())
                temp = 1.0 / ls_val if ls_val != 0 else 1.0
                pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})
                extra_parts = []
                if hard_neg_count > 0:
                    extra_parts.append(f"hard_negs={hard_neg_count}")
                if glossary_neg_embs is not None:
                    extra_parts.append(f"glossary_neg={glossary_neg_embs.size(0)}")
                extra_str = " " + " ".join(extra_parts) if extra_parts else ""
                logger.info(
                    f"[TRAIN] step={global_step} loss={total_loss.item():.6f} "
                    f"logit_scale={ls_val:.4f} temperature={temp:.6f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}{extra_str}"
                )
                if wandb_run is not None and global_step % args.wandb_log_interval == 0:
                    log_dict = {
                        "train/loss": total_loss.item(),
                        "train/logit_scale": ls_val,
                        "train/temperature": temp,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/step": global_step,
                        "train/epoch": epoch,
                    }
                    if hard_neg_count > 0:
                        log_dict["train/hard_neg_count"] = hard_neg_count
                    wandb_run.log(log_dict, step=global_step)

            # ---- LR Finder check ----
            if lr_finder_active and is_main:
                cur_lr = optimizer.param_groups[0]["lr"]
                cur_loss = total_loss.item()
                lr_finder_results.append((cur_lr, cur_loss))
                if cur_loss < lr_finder_best_loss:
                    lr_finder_best_loss = cur_loss
                if wandb_run is not None:
                    wandb_run.log({"lr_finder/lr": cur_lr, "lr_finder/loss": cur_loss}, step=global_step)
                if global_step >= args.lr_finder_steps or cur_loss > lr_finder_best_loss * 4:
                    logger.info(
                        f"[LR_FINDER] Stopping at step {global_step} "
                        f"lr={cur_lr:.2e} loss={cur_loss:.4f} best_loss={lr_finder_best_loss:.4f}"
                    )
                    logger.info("[LR_FINDER] Results (lr, loss):")
                    for lr_val, loss_val in lr_finder_results:
                        logger.info(f"  lr={lr_val:.6e}  loss={loss_val:.6f}")
                    time_budget_exceeded = True
                    break

            # ---- Time budget check (broadcast from rank 0 to avoid race) ----
            if args.time_budget > 0:
                budget_flag = torch.tensor(
                    [1 if is_main and (time.time() - t_train_start) >= args.time_budget else 0],
                    dtype=torch.long, device=device,
                )
                if world_size > 1:
                    dist.broadcast(budget_flag, src=0)
            else:
                budget_flag = torch.zeros(1, dtype=torch.long, device=device)
            if budget_flag.item() == 1:
                if is_main:
                    logger.info(
                        f"[TIME_BUDGET] {args.time_budget}s reached at step "
                        f"{global_step}, running final eval and exiting"
                    )
                if world_size > 1:
                    dist.barrier()
                if is_main:
                    final_metrics: Dict[str, float] = {}
                    if eval_loader is not None:
                        dev_m = run_sample_eval(
                            raw_retriever, raw_text_encoder, text_tokenizer,
                            eval_loader, device, args, global_step, epoch,
                            wandb_run, eval_name="dev",
                            wiki_terms=eval_wiki_terms,
                            glossary_sizes=eval_glossary_sizes,
                        )
                        final_metrics.update(dev_m)
                    if acl_eval_loader is not None:
                        acl_m = run_sample_eval(
                            raw_retriever, raw_text_encoder, text_tokenizer,
                            acl_eval_loader, device, args, global_step, epoch,
                            wandb_run, eval_name="acl6060",
                            wiki_terms=eval_wiki_terms,
                            glossary_sizes=eval_glossary_sizes,
                        )
                        final_metrics.update(acl_m)
                    # Track best
                    if (
                        best_metric_key
                        and best_metric_key in final_metrics
                        and final_metrics[best_metric_key] > best_metric_value
                    ):
                        best_metric_value = final_metrics[best_metric_key]
                    if (
                        best_metric_secondary_key
                        and best_metric_secondary_key in final_metrics
                        and final_metrics[best_metric_secondary_key]
                        > best_metric_secondary_value
                    ):
                        best_metric_secondary_value = final_metrics[
                            best_metric_secondary_key
                        ]
                if world_size > 1:
                    dist.barrier()
                time_budget_exceeded = True
                break  # exit inner loop

        if time_budget_exceeded:
            break  # exit outer epoch loop

        # Epoch save
        if is_main:
            ep_path = args.save_path.replace(".pt", f"_epoch_{epoch}.pt")
            torch.save(
                {
                    "model_state_dict": raw_retriever.state_dict(),
                    "text_model_state_dict": raw_text_encoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                    "args": vars(args),
                },
                ep_path,
            )
            logger.info(f"[EPOCH_SAVE] {ep_path}")

    # Final save
    if is_main:
        torch.save(
            {
                "model_state_dict": raw_retriever.state_dict(),
                "text_model_state_dict": raw_text_encoder.state_dict(),
                "global_step": global_step,
                "args": vars(args),
            },
            args.save_path,
        )
        logger.info(f"[FINAL_SAVE] {args.save_path}")

    # ---- Autoresearch summary (parseable by experiment.py) ----
    if is_main:
        t_total = time.time() - t_train_start
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print("---")
        print(f"best_acl6060_recall10_gs10000: {best_metric_secondary_value:.6f}")
        print(f"best_dev_recall10_gs10000:     {best_metric_value:.6f}")
        print(f"training_seconds:              {t_total:.1f}")
        print(f"peak_vram_mb:                  {peak_vram_mb:.1f}")
        print(f"total_steps:                   {global_step}")
        print(f"wiki_rank:                     {args.wiki_rank}")
        print(f"noisy_ratio:                   {args.noisy_ratio}")

    if is_main and wandb_run is not None:
        import signal

        def _wandb_timeout_handler(signum, frame):
            raise TimeoutError("wandb.finish() timed out")

        old_handler = signal.signal(signal.SIGALRM, _wandb_timeout_handler)
        signal.alarm(120)
        try:
            wandb_run.finish(quiet=True)
        except (TimeoutError, Exception) as e:
            logger.warning(f"[WANDB] finish() failed or timed out: {e}")
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


# ==================== CLI ====================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Masked FN InfoNCE + Glossary-Scale Negative training"
    )
    # Data
    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--dev_jsonl", type=str, default="")
    p.add_argument("--save_path", type=str, default="qwen3_masked_neg.pt")
    p.add_argument("--resume", type=str, default="")
    p.add_argument(
        "--wiki_rank", type=int, default=0,
        help="Include wiki_synth entries with p31_rank < this value. "
             "0 = include all (no filtering). Gigaspeech (p31_rank=-1) always included.",
    )
    p.add_argument(
        "--time_budget", type=int, default=0,
        help="Training time budget in seconds. 0 = no limit (train full epochs). "
             "When exceeded, run final eval, save checkpoint, print summary, and exit.",
    )
    p.add_argument(
        "--warmup_steps", type=int, default=-1,
        help="Explicit warmup steps. -1 = use DEFAULT_WARMUP_RATIO * total_steps.",
    )
    p.add_argument(
        "--lr_finder_steps", type=int, default=0,
        help="If > 0, run LR Finder mode: exponential LR sweep over this many steps.",
    )
    p.add_argument("--lr_finder_start", type=float, default=1e-7)
    p.add_argument("--lr_finder_end", type=float, default=1e-1)
    p.add_argument(
        "--noisy_ratio", type=float, default=1.0,
        help="Proportion of noisy wiki_synth samples to keep (0-1). "
             "1.0 = keep all noisy / drop all clean, "
             "0.0 = keep all clean / drop all noisy, "
             "0.5 = keep ~50%% of each. "
             "GigaSpeech (no audio_type) is always kept.",
    )

    # Model
    p.add_argument("--audio_model_id", type=str, default=DEFAULT_QWEN_AUDIO_MODEL_ID)
    p.add_argument("--text_model_id", type=str, default=DEFAULT_TEXT_MODEL_ID)
    p.add_argument("--target_dim", type=int, default=DEFAULT_TARGET_DIM)
    p.add_argument("--use_lora", action="store_true", default=False)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--text_lora_rank", type=int, default=16)
    p.add_argument("--text_lora_alpha", type=int, default=32)
    p.add_argument("--lora_target_modules", type=str, nargs="+", default=None)
    p.add_argument("--text_lora_target_modules", type=str, nargs="+", default=None)
    p.add_argument(
        "--text_full_finetune",
        action="store_true",
        default=False,
        help="Full finetune text encoder (skip LoRA, unfreeze all parameters)",
    )
    p.add_argument(
        "--text_lr",
        type=float,
        default=0.0,
        help="Learning rate for text encoder (0 = auto: lr*0.1 for full_finetune, lr for LoRA)",
    )

    # Training
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--train_limit", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.03)
    p.add_argument("--learn_temp", action="store_true", default=False)

    # Negative bank
    p.add_argument(
        "--neg_bank_size",
        type=int,
        default=DEFAULT_NEG_BANK_SIZE,
        help="Number of random global negatives per step (0 = disabled, ignored when hard_neg_k > 0)",
    )
    p.add_argument(
        "--neg_bank_refresh_steps",
        type=int,
        default=DEFAULT_NEG_BANK_REFRESH_STEPS,
        help="Re-encode the full glossary every N steps",
    )
    p.add_argument(
        "--hard_neg_k",
        type=int,
        default=DEFAULT_HARD_NEG_K,
        help="Hard negatives per sample mined from bank (0 = disabled, uses random bank)",
    )
    p.add_argument(
        "--hard_neg_glossary",
        type=str,
        default="",
        help="Path to external glossary JSON to mix into neg bank (e.g. wiki_glossary_nlp_ai_cs.json)",
    )

    # Glossary negatives (always-on negative columns)
    p.add_argument(
        "--glossary_neg_path",
        type=str,
        default=DEFAULT_GLOSSARY_NEG_PATH,
        help="Path to wiki glossary JSON for permanent negative columns in loss",
    )
    p.add_argument(
        "--glossary_neg_refresh_steps",
        type=int,
        default=DEFAULT_GLOSSARY_NEG_REFRESH_STEPS,
        help="Re-encode glossary negatives every N steps",
    )

    # Eval & checkpointing
    p.add_argument("--eval_steps_sample", type=int, default=DEFAULT_EVAL_STEPS_SAMPLE)
    p.add_argument("--eval_batch_size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    p.add_argument("--eval_topk", type=int, default=DEFAULT_EVAL_TOPK)
    p.add_argument("--eval_topk_extra", type=int, default=DEFAULT_EVAL_TOPK_EXTRA)
    p.add_argument("--save_steps", type=int, default=DEFAULT_SAVE_INTERVAL)
    p.add_argument("--keep_checkpoints", type=int, default=DEFAULT_KEEP_CHECKPOINTS)
    p.add_argument("--force_dummy_audio", action="store_true", default=False)
    p.add_argument(
        "--augment_synth", action="store_true", default=False,
        help="Apply noise/speed/reverb augmentation to synthetic TTS audio (wiki_synth_* entries)",
    )

    # Multi-domain / glossary-scale eval
    p.add_argument(
        "--acl_dev_jsonl",
        type=str,
        default=DEFAULT_ACL_DEV_JSONL,
        help="Path to ACL6060 dev JSONL for cross-domain eval",
    )
    p.add_argument(
        "--eval_wiki_glossary",
        type=str,
        default=DEFAULT_EVAL_WIKI_GLOSSARY,
        help="Path to wiki glossary JSON for eval glossary-scale expansion",
    )
    p.add_argument(
        "--eval_glossary_sizes",
        type=int,
        nargs="+",
        default=DEFAULT_EVAL_GLOSSARY_SIZES,
        help="Glossary sizes to evaluate (e.g. 1000 10000)",
    )
    p.add_argument(
        "--best_metric",
        type=str,
        default=DEFAULT_BEST_METRIC,
        help="Metric key for best checkpoint tracking (e.g. eval_acl6060/recall@10_gs1000)",
    )
    p.add_argument(
        "--best_metric_secondary",
        type=str,
        default="",
        help=(
            "Optional second metric for another best checkpoint "
            "(e.g. eval_acl6060/recall@10_gs10000); "
            "saved as <save_stem>_best_acl6060_gs10000.pt"
        ),
    )

    # WandB
    p.add_argument("--enable_wandb", action="store_true", default=False)
    p.add_argument("--wandb_project", type=str, default="qwen3_rag")
    p.add_argument("--wandb_exp_name", type=str, default="masked_neg_bank")
    p.add_argument(
        "--wandb_log_interval", type=int, default=DEFAULT_WANDB_LOG_INTERVAL
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    t0 = time.time()
    train(rank=local_rank, world_size=world_size, args=args)
    if local_rank == 0:
        logger.info(f"[DONE] elapsed={time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
