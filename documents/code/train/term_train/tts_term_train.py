#!/usr/bin/env python3
"""
Qwen3-Omni + BGE-M3 TTS Term Training Script

Triplet:
- speech_utt: source chunk audio
- term_text: term_key text
- term_tts: TTS audio of the same term

Loss:
L = lambda * L(speech -> term_tts) + (1 - lambda) * L(speech -> term_text)
"""

import os
import json
import time
import argparse
import datetime
import logging
import hashlib
from typing import Dict, List, Optional, Tuple

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
from transformers import AutoModel, AutoTokenizer, WhisperFeatureExtractor, get_cosine_schedule_with_warmup
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder
from peft import LoraConfig, get_peft_model


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
DEFAULT_EVAL_STEPS_SAMPLE = 1000
DEFAULT_EVAL_BATCH_SIZE = 256
DEFAULT_EVAL_TOPK = 5
DEFAULT_EVAL_TOPK_EXTRA = 10
DEFAULT_INVALID_TERM_ID = 0

# =======================


os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AttentivePooling(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            add_pooling_layer=False,
        )
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

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]
        return F.normalize(embeddings, p=2, dim=-1)


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
            model_id,
            dtype=torch.bfloat16,
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
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1.0 / temperature))
        else:
            self.register_buffer("logit_scale", torch.tensor(np.log(1.0 / temperature)))

    def forward(self, input_features: torch.Tensor, feature_lens: torch.Tensor) -> torch.Tensor:
        if input_features.ndim == 3:
            input_features = input_features.transpose(0, 1).reshape(input_features.shape[1], -1)

        outputs = self.audio_encoder(input_features, feature_lens)
        hidden_states = outputs.last_hidden_state

        if hidden_states.ndim == 2:
            output_lens = []
            for current_len in feature_lens.tolist():
                reduced = current_len
                for _ in range(3):
                    reduced = (reduced + 1) // 2
                output_lens.append(reduced)
            if sum(output_lens) != hidden_states.shape[0]:
                ratio = input_features.shape[1] / hidden_states.shape[0]
                output_lens = [max(1, round(x / ratio)) for x in feature_lens.tolist()]
                output_lens[-1] = hidden_states.shape[0] - sum(output_lens[:-1])

            from torch.nn.utils.rnn import pad_sequence

            hidden_states_list = torch.split(hidden_states, output_lens, dim=0)
            hidden_states = pad_sequence(hidden_states_list, batch_first=True)
            feature_lens = torch.tensor(output_lens, device=hidden_states.device)

        batch_size, max_len, _ = hidden_states.shape
        mask = torch.arange(max_len, device=hidden_states.device).expand(batch_size, max_len) < feature_lens.unsqueeze(1)
        pooled = self.pooler(hidden_states, mask)
        projected = self.projector(pooled)
        return F.normalize(projected, p=2, dim=-1)


def build_tts_audio_path(tts_root_dir: str, utter_id: str, chunk_idx: int) -> Optional[str]:
    if not utter_id:
        return None
    parts = utter_id.rsplit("_", 1)
    if len(parts) != 2:
        return None
    speaker_utt, segment_id = parts[0], parts[1]
    tts_path = os.path.join(tts_root_dir, speaker_utt, segment_id, f"chunk_{chunk_idx}.wav")
    return tts_path


class TtsTermDataset(Dataset):
    def __init__(self, samples: List[Dict], tts_root_dir: str, force_dummy_audio: bool = False):
        self.samples = samples
        self.tts_root_dir = tts_root_dir
        self.force_dummy_audio = force_dummy_audio
        self.path_remap_src = os.environ.get("AUDIO_PATH_REMAP_SRC", "").strip()
        self.path_remap_dst = os.environ.get("AUDIO_PATH_REMAP_DST", "").strip()

    def __len__(self) -> int:
        return len(self.samples)

    def _read_audio(self, path: str) -> Optional[np.ndarray]:
        try:
            audio_data, sample_rate = sf.read(path)
            if sample_rate != DEFAULT_AUDIO_SAMPLE_RATE:
                logger.warning(f"[WRONG_SR] path={path} sample_rate={sample_rate} expected={DEFAULT_AUDIO_SAMPLE_RATE}")
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
            max_abs = np.max(np.abs(audio_data))
            if max_abs > 0:
                audio_data = audio_data / max_abs
            return audio_data.astype(np.float32)
        except Exception as exc:
            logger.warning(f"[AUDIO_LOAD_FAIL] path={path} error={exc}")
            return None

    def __getitem__(self, idx: int) -> Dict:
        sample = dict(self.samples[idx])
        term_text = (sample.get("term_key", "") or "").strip().lower()

        if self.force_dummy_audio:
            dummy = np.zeros(DEFAULT_FIXED_AUDIO_SAMPLES, dtype=np.float32)
            return {
                "speech_audio": dummy,
                "term_tts_audio": dummy,
                "term_text": term_text,
                "has_tts_audio": False,
                "chunk_audio_path": "DUMMY",
                "term_tts_path": "DUMMY",
            }

        speech_audio_path = sample.get("chunk_audio_path", "")
        if self.path_remap_src and self.path_remap_dst and speech_audio_path.startswith(self.path_remap_src):
            candidate = self.path_remap_dst + speech_audio_path[len(self.path_remap_src):]
            if os.path.exists(candidate):
                speech_audio_path = candidate

        speech_audio = self._read_audio(speech_audio_path)
        utter_id = str(sample.get("utter_id", "") or "")
        chunk_idx_raw = sample.get("chunk_idx", None)
        chunk_idx = int(chunk_idx_raw) if chunk_idx_raw is not None else -1
        tts_audio_path = build_tts_audio_path(self.tts_root_dir, utter_id, chunk_idx) if chunk_idx >= 0 else None
        term_tts_audio = self._read_audio(tts_audio_path) if tts_audio_path and os.path.exists(tts_audio_path) else None

        return {
            "speech_audio": speech_audio,
            "term_tts_audio": term_tts_audio,
            "term_text": term_text,
            "has_tts_audio": term_tts_audio is not None,
            "chunk_audio_path": speech_audio_path,
            "term_tts_path": tts_audio_path if tts_audio_path else "",
        }


def normalize_length(audio: np.ndarray, fixed_len: int) -> np.ndarray:
    if len(audio) < fixed_len:
        return np.pad(audio, (0, fixed_len - len(audio)), mode="constant")
    if len(audio) > fixed_len:
        return audio[:fixed_len]
    return audio


def collate_fn(batch: List[Dict], feature_extractor: WhisperFeatureExtractor) -> Dict:
    dummy_audio = np.zeros(DEFAULT_FIXED_AUDIO_SAMPLES, dtype=np.float32)
    speech_list: List[np.ndarray] = []
    tts_list: List[np.ndarray] = []
    text_list: List[str] = []
    has_tts_list: List[bool] = []
    has_text_list: List[bool] = []
    samples: List[Dict] = []

    for sample in batch:
        speech_audio = sample.get("speech_audio")
        term_tts_audio = sample.get("term_tts_audio")
        term_text = sample.get("term_text", "")

        if speech_audio is None or len(speech_audio) <= DEFAULT_MIN_AUDIO_SAMPLES:
            speech_audio = dummy_audio
        if term_tts_audio is None or len(term_tts_audio) <= DEFAULT_MIN_AUDIO_SAMPLES:
            has_tts = False
            term_tts_audio = dummy_audio
        else:
            has_tts = bool(sample.get("has_tts_audio", False))

        speech_audio = normalize_length(speech_audio, DEFAULT_FIXED_AUDIO_SAMPLES)
        term_tts_audio = normalize_length(term_tts_audio, DEFAULT_FIXED_AUDIO_SAMPLES)

        speech_list.append(speech_audio)
        tts_list.append(term_tts_audio)
        text_list.append(term_text)
        has_tts_list.append(has_tts)
        has_text_list.append(bool(term_text))
        samples.append(sample)

    speech_inputs = feature_extractor(
        speech_list,
        sampling_rate=DEFAULT_AUDIO_SAMPLE_RATE,
        return_tensors="pt",
        padding=False,
    )
    tts_inputs = feature_extractor(
        tts_list,
        sampling_rate=DEFAULT_AUDIO_SAMPLE_RATE,
        return_tensors="pt",
        padding=False,
    )

    speech_features = speech_inputs.input_features
    tts_features = tts_inputs.input_features
    speech_lens = torch.full((speech_features.size(0),), speech_features.size(-1), dtype=torch.long)
    tts_lens = torch.full((tts_features.size(0),), tts_features.size(-1), dtype=torch.long)

    return {
        "speech_input_features": speech_features,
        "speech_feature_lens": speech_lens,
        "tts_input_features": tts_features,
        "tts_feature_lens": tts_lens,
        "term_texts": text_list,
        "has_tts_audio": torch.tensor(has_tts_list, dtype=torch.bool),
        "has_term_text": torch.tensor(has_text_list, dtype=torch.bool),
        "samples": samples,
    }


def compute_single_direction_loss(
    speech_embs: torch.Tensor,
    key_embs: torch.Tensor,
    logit_scale: torch.Tensor,
    positive_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    local_batch_size = speech_embs.size(0)

    global_keys = all_gather_with_grad(key_embs)
    logits = (speech_embs @ global_keys.T) * logit_scale

    if world_size > 1:
        rank_id = dist.get_rank()
        targets = torch.arange(local_batch_size, device=speech_embs.device) + rank_id * local_batch_size
    else:
        targets = torch.arange(local_batch_size, device=speech_embs.device)

    per_sample_loss = F.cross_entropy(logits, targets, reduction="none")

    if positive_mask is None:
        return per_sample_loss.mean()

    valid_mask = positive_mask.float()
    denom = valid_mask.sum().clamp(min=1.0)
    return (per_sample_loss * valid_mask).sum() / denom


def stable_term_id(term_text: str) -> int:
    if not term_text:
        return DEFAULT_INVALID_TERM_ID
    digest = hashlib.blake2b(term_text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def compute_multi_positive_text_loss(
    speech_embs: torch.Tensor,
    text_embs: torch.Tensor,
    logit_scale: torch.Tensor,
    local_term_ids: torch.Tensor,
    local_has_text: torch.Tensor,
) -> torch.Tensor:
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    global_text_embs = all_gather_with_grad(text_embs)

    if world_size > 1:
        gathered_term_ids = [torch.zeros_like(local_term_ids) for _ in range(world_size)]
        gathered_has_text = [torch.zeros_like(local_has_text) for _ in range(world_size)]
        dist.all_gather(gathered_term_ids, local_term_ids)
        dist.all_gather(gathered_has_text, local_has_text)
        global_term_ids = torch.cat(gathered_term_ids, dim=0)
        global_has_text = torch.cat(gathered_has_text, dim=0)
    else:
        global_term_ids = local_term_ids
        global_has_text = local_has_text

    logits = (speech_embs @ global_text_embs.T) * logit_scale
    logits = logits.masked_fill(~global_has_text.unsqueeze(0), -1e9)

    pos_mask = (local_term_ids.unsqueeze(1) == global_term_ids.unsqueeze(0))
    pos_mask = pos_mask & local_has_text.unsqueeze(1) & global_has_text.unsqueeze(0)

    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_count = pos_mask.sum(dim=1)
    row_valid = (local_has_text & (pos_count > 0)).float()

    loss_per_row = -((log_prob * pos_mask.float()).sum(dim=1) / pos_count.clamp(min=1).float())
    loss = (loss_per_row * row_valid).sum() / row_valid.sum().clamp(min=1.0)
    return loss


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
    ).to(device)
    text_tokenizer = AutoTokenizer.from_pretrained(args.text_model_id)
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")

    if world_size > 1:
        retriever = DDP(retriever, device_ids=[rank])
        text_encoder = DDP(text_encoder, device_ids=[rank])

    raw_retriever = retriever.module if world_size > 1 else retriever
    raw_text_encoder = text_encoder.module if world_size > 1 else text_encoder

    audio_lora_params = [p for p in raw_retriever.audio_encoder.parameters() if p.requires_grad]
    text_lora_params = [p for p in raw_text_encoder.encoder.parameters() if p.requires_grad]
    head_params = list(raw_retriever.pooler.parameters()) + list(raw_retriever.projector.parameters())
    if args.learn_temp:
        head_params.append(raw_retriever.logit_scale)

    optimizer_groups = []
    if audio_lora_params:
        optimizer_groups.append({"params": audio_lora_params, "lr": args.lr, "name": "audio_lora"})
    if text_lora_params:
        optimizer_groups.append({"params": text_lora_params, "lr": args.lr, "name": "text_lora"})
    optimizer_groups.append({"params": head_params, "lr": args.lr * DEFAULT_HEAD_LR_SCALE, "name": "head"})

    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=DEFAULT_WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda")

    train_samples: List[Dict] = []
    with open(args.train_jsonl, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if args.train_limit and line_idx >= args.train_limit:
                break
            try:
                train_samples.append(json.loads(line))
            except Exception:
                continue

    dev_samples: List[Dict] = []
    if args.dev_jsonl:
        with open(args.dev_jsonl, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if args.dev_limit and line_idx >= args.dev_limit:
                    break
                try:
                    dev_samples.append(json.loads(line))
                except Exception:
                    continue

    dataset = TtsTermDataset(
        samples=train_samples,
        tts_root_dir=args.tts_root_dir,
        force_dummy_audio=args.force_dummy_audio,
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    per_rank_batch_size = args.batch_size // world_size
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=per_rank_batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=lambda b: collate_fn(b, feature_extractor),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    eval_loader = None
    if dev_samples:
        eval_dataset = TtsTermDataset(
            samples=dev_samples,
            tts_root_dir=args.tts_root_dir,
            force_dummy_audio=args.force_dummy_audio,
        )
        eval_loader = DataLoader(
            dataset=eval_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, feature_extractor),
            num_workers=args.eval_num_workers,
            pin_memory=True,
            drop_last=False,
        )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * DEFAULT_WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    global_step = 0
    recent_checkpoints: List[str] = []
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
            wandb.define_metric("eval_sample/step")
            wandb.define_metric("eval_sample/*", step_metric="eval_sample/step")
            logger.info(f"[WANDB] enabled project={args.wandb_project} name={args.wandb_exp_name}")
        except Exception as exc:
            wandb_run = None
            logger.warning(f"[WANDB] disabled reason=init_failed error={exc}")

    if is_main:
        logger.info(
            f"[TRAIN_SETUP] samples={len(train_samples)} world_size={world_size} "
            f"per_rank_batch={per_rank_batch_size} total_steps={total_steps}"
        )
        if args.dev_jsonl:
            logger.info(
                f"[EVAL_SETUP] dev_samples={len(dev_samples)} eval_steps_sample={args.eval_steps_sample} "
                f"eval_batch_size={args.eval_batch_size}"
            )

    def run_sample_eval(current_step: int, current_epoch: int) -> None:
        if eval_loader is None or not dev_samples:
            return

        raw_retriever.eval()
        raw_text_encoder.eval()
        start_time = time.time()

        speech_emb_list: List[torch.Tensor] = []
        text_emb_list: List[torch.Tensor] = []
        tts_emb_list: List[torch.Tensor] = []
        has_tts_list: List[torch.Tensor] = []
        has_text_list: List[torch.Tensor] = []

        with torch.no_grad():
            for eval_batch in eval_loader:
                speech_features = eval_batch["speech_input_features"].to(device).to(torch.bfloat16)
                speech_lens = eval_batch["speech_feature_lens"].to(device)
                tts_features = eval_batch["tts_input_features"].to(device).to(torch.bfloat16)
                tts_lens = eval_batch["tts_feature_lens"].to(device)
                term_texts = eval_batch["term_texts"]
                has_tts_audio = eval_batch["has_tts_audio"].to(device)
                has_term_text = eval_batch["has_term_text"].to(device)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    combined_features = torch.cat([speech_features, tts_features], dim=0)
                    combined_lens = torch.cat([speech_lens, tts_lens], dim=0)
                    combined_embs = raw_retriever(combined_features, combined_lens)
                    eval_batch_size = speech_features.size(0)
                    speech_embs, term_tts_embs = torch.split(combined_embs, [eval_batch_size, eval_batch_size], dim=0)

                    text_inputs = text_tokenizer(
                        term_texts,
                        padding=True,
                        truncation=True,
                        max_length=DEFAULT_TEXT_MAX_LENGTH,
                        return_tensors="pt",
                    ).to(device)
                    term_text_embs = raw_text_encoder(text_inputs.input_ids, text_inputs.attention_mask)

                speech_emb_list.append(speech_embs.float().cpu())
                text_emb_list.append(term_text_embs.float().cpu())
                tts_emb_list.append(term_tts_embs.float().cpu())
                has_tts_list.append(has_tts_audio.bool().cpu())
                has_text_list.append(has_term_text.bool().cpu())

        if not speech_emb_list:
            raw_retriever.train()
            raw_text_encoder.train()
            return

        speech_embs = torch.cat(speech_emb_list, dim=0)
        text_embs = torch.cat(text_emb_list, dim=0)
        tts_embs = torch.cat(tts_emb_list, dim=0)
        has_tts_mask = torch.cat(has_tts_list, dim=0)
        has_text_mask = torch.cat(has_text_list, dim=0)

        sample_count = speech_embs.size(0)
        logit_scale_eval = float(raw_retriever.logit_scale.exp().detach().cpu().item())
        valid_text_indices = torch.nonzero(has_text_mask, as_tuple=False).squeeze(1)
        valid_text_count = int(valid_text_indices.numel())
        text_top1 = 0.0
        text_recall_topk_primary = 0.0
        text_recall_topk_extra = 0.0
        text_loss_eval = 0.0
        if valid_text_count > 0:
            speech_text_valid = speech_embs[valid_text_indices]
            text_valid = text_embs[valid_text_indices]
            text_logits = speech_text_valid @ text_valid.t()
            text_targets = torch.arange(valid_text_count)
            recall_k_primary = min(args.eval_topk, valid_text_count)
            recall_k_extra = min(args.eval_topk_extra, valid_text_count)
            text_top1 = (text_logits.argmax(dim=1) == text_targets).float().mean().item()
            text_topk_hits_primary = torch.topk(text_logits, k=recall_k_primary, dim=1).indices.eq(text_targets.unsqueeze(1)).any(dim=1)
            text_recall_topk_primary = text_topk_hits_primary.float().mean().item()
            text_topk_hits_extra = torch.topk(text_logits, k=recall_k_extra, dim=1).indices.eq(text_targets.unsqueeze(1)).any(dim=1)
            text_recall_topk_extra = text_topk_hits_extra.float().mean().item()
            text_loss_eval = F.cross_entropy(text_logits * logit_scale_eval, text_targets).item()

        valid_indices = torch.nonzero(has_tts_mask, as_tuple=False).squeeze(1)
        valid_tts_count = int(valid_indices.numel())
        tts_top1 = 0.0
        tts_recall_topk_primary = 0.0
        tts_recall_topk_extra = 0.0
        tts_loss_eval = 0.0
        if valid_tts_count > 0:
            speech_valid = speech_embs[valid_indices]
            tts_valid = tts_embs[valid_indices]
            tts_logits = speech_valid @ tts_valid.t()
            tts_targets = torch.arange(valid_tts_count)
            tts_top1 = (tts_logits.argmax(dim=1) == tts_targets).float().mean().item()
            tts_topk_primary = min(args.eval_topk, valid_tts_count)
            tts_topk_hits_primary = torch.topk(tts_logits, k=tts_topk_primary, dim=1).indices.eq(tts_targets.unsqueeze(1)).any(dim=1)
            tts_recall_topk_primary = tts_topk_hits_primary.float().mean().item()
            tts_topk_extra = min(args.eval_topk_extra, valid_tts_count)
            tts_topk_hits_extra = torch.topk(tts_logits, k=tts_topk_extra, dim=1).indices.eq(tts_targets.unsqueeze(1)).any(dim=1)
            tts_recall_topk_extra = tts_topk_hits_extra.float().mean().item()
            tts_loss_eval = F.cross_entropy(tts_logits * logit_scale_eval, tts_targets).item()

        lambda_weight = args.tts_loss_weight
        total_eval_loss = lambda_weight * tts_loss_eval + (1.0 - lambda_weight) * text_loss_eval
        eval_elapsed = time.time() - start_time

        logger.info(
            f"[EVAL_SAMPLE] step={current_step} epoch={current_epoch} "
            f"samples={sample_count} valid_text={valid_text_count} valid_tts={valid_tts_count} "
            f"loss={total_eval_loss:.6f} text_top1={text_top1:.4f} "
            f"text_recall@{args.eval_topk}={text_recall_topk_primary:.4f} "
            f"text_recall@{args.eval_topk_extra}={text_recall_topk_extra:.4f} "
            f"tts_top1={tts_top1:.4f} "
            f"tts_recall@{args.eval_topk}={tts_recall_topk_primary:.4f} "
            f"tts_recall@{args.eval_topk_extra}={tts_recall_topk_extra:.4f} "
            f"elapsed_seconds={eval_elapsed:.2f}"
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "eval_sample/loss": total_eval_loss,
                    "eval_sample/loss_speech_to_text": text_loss_eval,
                    "eval_sample/loss_speech_to_tts": tts_loss_eval,
                    "eval_sample/text_top1": text_top1,
                    f"eval_sample/text_recall@{args.eval_topk}": text_recall_topk_primary,
                    f"eval_sample/text_recall@{args.eval_topk_extra}": text_recall_topk_extra,
                    "eval_sample/tts_top1": tts_top1,
                    f"eval_sample/tts_recall@{args.eval_topk}": tts_recall_topk_primary,
                    f"eval_sample/tts_recall@{args.eval_topk_extra}": tts_recall_topk_extra,
                    "eval_sample/valid_text_count": valid_text_count,
                    "eval_sample/valid_tts_count": valid_tts_count,
                    "eval_sample/sample_count": sample_count,
                    "eval_sample/step": current_step,
                    "eval_sample/epoch": current_epoch,
                },
                step=current_step,
            )

        raw_retriever.train()
        raw_text_encoder.train()

    for epoch in range(args.epochs):
        retriever.train()
        text_encoder.train()
        if sampler is not None:
            sampler.set_epoch(epoch)

        iterator = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not is_main)
        for batch in iterator:
            global_step += 1

            speech_features = batch["speech_input_features"].to(device).to(torch.bfloat16)
            speech_lens = batch["speech_feature_lens"].to(device)
            tts_features = batch["tts_input_features"].to(device).to(torch.bfloat16)
            tts_lens = batch["tts_feature_lens"].to(device)
            term_texts = batch["term_texts"]
            has_tts_audio = batch["has_tts_audio"].to(device)
            has_term_text = batch["has_term_text"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # Run a single retriever forward per iteration to avoid
                # DDP + gradient-checkpointing reentrant-backward conflicts.
                combined_features = torch.cat([speech_features, tts_features], dim=0)
                combined_lens = torch.cat([speech_lens, tts_lens], dim=0)
                combined_embs = retriever(combined_features, combined_lens)
                speech_batch_size = speech_features.size(0)
                speech_embs, term_tts_embs = torch.split(combined_embs, [speech_batch_size, speech_batch_size], dim=0)
                text_inputs = text_tokenizer(
                    term_texts,
                    padding=True,
                    truncation=True,
                    max_length=DEFAULT_TEXT_MAX_LENGTH,
                    return_tensors="pt",
                ).to(device)
                term_text_embs = text_encoder(text_inputs.input_ids, text_inputs.attention_mask)

                logit_scale = retriever.module.logit_scale.exp() if world_size > 1 else retriever.logit_scale.exp()
                local_term_ids_list = [stable_term_id(t) for t in term_texts]
                local_term_ids = torch.tensor(local_term_ids_list, device=device, dtype=torch.long)

                loss_speech_to_tts = compute_single_direction_loss(
                    speech_embs=speech_embs,
                    key_embs=term_tts_embs,
                    logit_scale=logit_scale,
                    positive_mask=has_tts_audio,
                )
                loss_speech_to_text = compute_multi_positive_text_loss(
                    speech_embs=speech_embs,
                    text_embs=term_text_embs,
                    logit_scale=logit_scale,
                    local_term_ids=local_term_ids,
                    local_has_text=has_term_text,
                )

                lambda_weight = args.tts_loss_weight
                total_loss = (
                    lambda_weight * loss_speech_to_tts
                    + (1.0 - lambda_weight) * loss_speech_to_text
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(retriever.parameters(), max_norm=DEFAULT_GRAD_CLIP_MAX_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if args.eval_steps_sample > 0 and global_step % args.eval_steps_sample == 0 and args.dev_jsonl:
                if world_size > 1:
                    dist.barrier()
                if is_main:
                    run_sample_eval(current_step=global_step, current_epoch=epoch)
                if world_size > 1:
                    dist.barrier()

            if is_main and global_step % DEFAULT_LOG_INTERVAL == 0:
                current_logit_scale = float(logit_scale.item())
                current_temperature = 1.0 / current_logit_scale if current_logit_scale != 0 else 1.0
                valid_tts_ratio = float(has_tts_audio.float().mean().item())
                valid_text_ratio = float(has_term_text.float().mean().item())
                iterator.set_postfix(
                    {
                        "loss": f"{float(total_loss.item()):.4f}",
                        "l_s2a": f"{float(loss_speech_to_tts.item()):.4f}",
                        "l_s2t": f"{float(loss_speech_to_text.item()):.4f}",
                        "valid_tts": f"{valid_tts_ratio:.2f}",
                        "valid_txt": f"{valid_text_ratio:.2f}",
                    }
                )
                logger.info(
                    f"[TRAIN] step={global_step} "
                    f"loss={float(total_loss.item()):.6f} "
                    f"loss_speech_to_tts={float(loss_speech_to_tts.item()):.6f} "
                    f"loss_speech_to_text={float(loss_speech_to_text.item()):.6f} "
                    f"valid_tts_ratio={valid_tts_ratio:.4f} "
                    f"valid_text_ratio={valid_text_ratio:.4f} "
                    f"logit_scale={current_logit_scale:.4f} "
                    f"temperature={current_temperature:.6f}"
                )

                if wandb_run is not None and global_step % args.wandb_log_interval == 0:
                    wandb_run.log(
                        {
                            "train/loss": float(total_loss.item()),
                            "train/loss_speech_to_tts": float(loss_speech_to_tts.item()),
                            "train/loss_speech_to_text": float(loss_speech_to_text.item()),
                            "train/valid_tts_ratio": valid_tts_ratio,
                            "train/valid_text_ratio": valid_text_ratio,
                            "train/logit_scale": current_logit_scale,
                            "train/temperature": current_temperature,
                            "train/lr": float(optimizer.param_groups[0]["lr"]),
                            "train/step": global_step,
                            "train/epoch": epoch,
                        },
                        step=global_step,
                    )

            if is_main and global_step % args.save_steps == 0:
                save_payload = {
                    "model_state_dict": raw_retriever.state_dict(),
                    "text_model_state_dict": raw_text_encoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                    "args": vars(args),
                }
                ckpt_path = args.save_path.replace(".pt", f"_step_{global_step}.pt")
                torch.save(save_payload, ckpt_path)
                recent_checkpoints.append(ckpt_path)
                logger.info(f"[CHECKPOINT] saved={ckpt_path}")

                while len(recent_checkpoints) > args.keep_checkpoints:
                    old_ckpt = recent_checkpoints.pop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)
                        logger.info(f"[CHECKPOINT] removed_old={old_ckpt}")

        if is_main:
            epoch_ckpt = args.save_path.replace(".pt", f"_epoch_{epoch}.pt")
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
                epoch_ckpt,
            )
            logger.info(f"[EPOCH_SAVE] saved={epoch_ckpt}")

    if is_main:
        torch.save(
            {
                "model_state_dict": raw_retriever.state_dict(),
                "text_model_state_dict": raw_text_encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "global_step": global_step,
                "args": vars(args),
            },
            args.save_path,
        )
        logger.info(f"[FINAL_SAVE] saved={args.save_path}")

    if is_main and wandb_run is not None:
        wandb_run.finish()

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--dev_jsonl", type=str, default="")
    parser.add_argument("--tts_root_dir", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="qwen3_retriever_tts.pt")
    parser.add_argument("--audio_model_id", type=str, default=DEFAULT_QWEN_AUDIO_MODEL_ID)
    parser.add_argument("--text_model_id", type=str, default=DEFAULT_TEXT_MODEL_ID)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--dev_limit", type=int, default=None)
    parser.add_argument("--eval_steps_sample", type=int, default=DEFAULT_EVAL_STEPS_SAMPLE)
    parser.add_argument("--eval_batch_size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--eval_num_workers", type=int, default=4)
    parser.add_argument("--eval_topk", type=int, default=DEFAULT_EVAL_TOPK)
    parser.add_argument("--eval_topk_extra", type=int, default=DEFAULT_EVAL_TOPK_EXTRA)

    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--learn_temp", action="store_true", default=False)
    parser.add_argument("--tts_loss_weight", type=float, default=0.5)

    parser.add_argument("--use_lora", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--text_lora_rank", type=int, default=16)
    parser.add_argument("--text_lora_alpha", type=int, default=32)
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=None)
    parser.add_argument("--text_lora_target_modules", type=str, nargs="+", default=None)

    parser.add_argument("--target_dim", type=int, default=DEFAULT_TARGET_DIM)
    parser.add_argument("--save_steps", type=int, default=DEFAULT_SAVE_INTERVAL)
    parser.add_argument("--keep_checkpoints", type=int, default=DEFAULT_KEEP_CHECKPOINTS)
    parser.add_argument("--force_dummy_audio", action="store_true", default=False)
    parser.add_argument("--enable_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="qwen3_rag")
    parser.add_argument("--wandb_exp_name", type=str, default="tts_term_train")
    parser.add_argument("--wandb_log_interval", type=int, default=DEFAULT_WANDB_LOG_INTERVAL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.tts_loss_weight <= 1.0):
        raise ValueError("--tts_loss_weight must be in [0, 1].")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    start = time.time()
    train(rank=local_rank, world_size=world_size, args=args)
    if local_rank == 0:
        logger.info(f"[DONE] elapsed_seconds={time.time() - start:.2f}")


if __name__ == "__main__":
    main()
