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

import contextlib
import math
import os
import json
import time
import argparse
import datetime
import hashlib
import logging
import random
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import record_function as _record_function
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    AutoTokenizer,
    WhisperModel,
    get_cosine_schedule_with_warmup,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
)
from peft import LoraConfig, get_peft_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

_GENERAL_CODE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "general")
)
if _GENERAL_CODE_DIR not in sys.path:
    sys.path.append(_GENERAL_CODE_DIR)
from wandb_tags import prepare_wandb_tags


# ==================== Experiment tracking helpers ====================
# See .cursor/rules/experiment_tracking.mdc for the full convention.

_REQUIRED_NOTES_SECTIONS = (
    "## Hypothesis",
    "## Background / Motivation",
    "## What changed vs baseline",
    "## Expected metrics",
    "## Verdict",
)


def load_and_validate_run_notes(notes_path: str) -> str:
    """Read notes markdown, ensure required sections exist and are non-empty.

    Fails loudly (no silent fallback) per repo rules. `## Verdict` may contain
    placeholder copy that gets replaced after the run finishes, but it still
    must exist as a header.
    """
    if not notes_path:
        raise ValueError(
            "experiment_tracking: --notes_file is required when --enable_wandb "
            "is on. See documents/code/_templates/run_notes_template.md."
        )
    if not os.path.isfile(notes_path):
        raise FileNotFoundError(
            f"experiment_tracking: --notes_file not found: {notes_path}"
        )
    with open(notes_path, "r", encoding="utf-8") as f:
        text = f.read()
    missing = [s for s in _REQUIRED_NOTES_SECTIONS if s not in text]
    if missing:
        raise ValueError(
            f"experiment_tracking: notes file {notes_path} is missing required "
            f"sections: {missing}. Copy documents/code/_templates/run_notes_template.md."
        )
    # Enforce non-empty bodies for the four up-front sections (Verdict is filled later).
    for i, section in enumerate(_REQUIRED_NOTES_SECTIONS[:-1]):
        start = text.index(section) + len(section)
        next_section_start = len(text)
        for later in _REQUIRED_NOTES_SECTIONS[i + 1 :]:
            idx = text.find(later, start)
            if idx != -1:
                next_section_start = min(next_section_start, idx)
        body = text[start:next_section_start]
        # Strip HTML comments (template placeholders) before emptiness check.
        body_stripped = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL).strip()
        if not body_stripped:
            raise ValueError(
                f"experiment_tracking: section '{section}' in {notes_path} is empty. "
                "Fill it in (HTML comments do not count)."
            )
    return text


def build_wandb_tags(args: argparse.Namespace) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Assemble the mandatory structured tags for this run."""
    tags: List[str] = []
    if not getattr(args, "experiment_family", ""):
        raise ValueError(
            "experiment_tracking: --experiment_family is required "
            "(e.g. retriever_3variant, sst_density_ablation)."
        )
    if not getattr(args, "data_tag", ""):
        raise ValueError(
            "experiment_tracking: --data_tag is required "
            "(e.g. 3variant_1m_mfa, adversarial_varlen_d5)."
        )
    task_tag = getattr(args, "task_tag", "train") or "train"
    tags.append(f"family:{args.experiment_family}")
    tags.append(f"task:{task_tag}")
    tags.append(f"data:{args.data_tag}")
    tags.append("status:running")
    for extra in getattr(args, "extra_wandb_tags", []) or []:
        if extra and extra not in tags:
            tags.append(extra)
    return prepare_wandb_tags(tags)


def finalize_wandb_run(
    wandb_run,
    success: bool,
    verdict: str = "",
) -> None:
    """Flip status tag and write summary.verdict before finishing the run."""
    if wandb_run is None:
        return
    try:
        new_status = "status:success" if success else "status:failed"
        current_tags = [t for t in (wandb_run.tags or []) if not t.startswith("status:")]
        current_tags.append(new_status)
        wandb_run.tags = tuple(current_tags)
        if verdict:
            wandb_run.summary["verdict"] = verdict
        else:
            if wandb_run.summary.get("verdict") is None:
                wandb_run.summary[
                    "verdict"
                ] = "pending - awaiting agent fill (see run notes)"
    except Exception as exc:  # pragma: no cover - best effort before finish()
        logging.getLogger(__name__).warning(
            f"[WANDB] finalize_wandb_run failed: {exc}"
        )


def _parse_env_overrides(items: Optional[Sequence[str]]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(
                f"auto_full_eval_extra_env entry must be KEY=VALUE, got: {item}"
            )
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"auto_full_eval_extra_env has empty key: {item}")
        overrides[key] = value
    return overrides


def _safe_run_name_fragment(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return text.strip("_") or "run"


def _submit_auto_full_eval(
    args: argparse.Namespace,
    checkpoint_path: str,
    global_step: int,
    metric_key: str,
    metric_value: float,
    wandb_run,
    logger: logging.Logger,
) -> Optional[str]:
    launcher = getattr(args, "auto_full_eval_launcher", "") or ""
    if not launcher:
        logger.warning("[AUTO_FULL_EVAL] enabled but no launcher was provided")
        return None
    if not os.path.isfile(launcher):
        logger.warning(f"[AUTO_FULL_EVAL] launcher not found: {launcher}")
        return None
    if not os.path.isfile(checkpoint_path):
        logger.warning(f"[AUTO_FULL_EVAL] checkpoint not found: {checkpoint_path}")
        return None

    immutable_checkpoint_path = checkpoint_path
    if checkpoint_path.endswith(".pt"):
        immutable_checkpoint_path = checkpoint_path.replace(
            ".pt", f"_auto_eval_step_{global_step}.pt"
        )
        try:
            shutil.copy2(checkpoint_path, immutable_checkpoint_path)
        except Exception as exc:
            logger.warning(
                f"[AUTO_FULL_EVAL] failed to snapshot checkpoint for "
                f"step={global_step}: {exc}"
            )
            return None
        logger.info(
            f"[AUTO_FULL_EVAL] snapshotted checkpoint for step={global_step}: "
            f"{immutable_checkpoint_path}"
        )

    source_run_id = getattr(wandb_run, "id", "") if wandb_run is not None else ""
    source_name = _safe_run_name_fragment(args.wandb_exp_name)
    eval_suffix = f"eval1m_step{global_step}"
    eval_name = f"{source_name}_{eval_suffix}"
    eval_version = _safe_run_name_fragment(f"{source_name}_{eval_suffix}")
    baseline_ids = list(getattr(args, "baseline_run_ids", []) or [])
    if source_run_id and source_run_id not in baseline_ids:
        baseline_ids = [source_run_id] + baseline_ids

    # Use a deliberately small sbatch environment. Inheriting the training
    # process environment leaks SLURM/TMPDIR/WandB settings into the child job.
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", ""),
        "USER": os.environ.get("USER", ""),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
    }
    env.update(
        {
            "RESUME": immutable_checkpoint_path,
            "WANDB_EXP_NAME": eval_name,
            "VERSION": eval_version,
            "BASELINE_RUN_IDS": " ".join(baseline_ids),
            "RUN_VERDICT": (
                "Auto 1M eval for primary best checkpoint at step "
                f"{global_step} ({metric_key}={metric_value:.4f})."
            ),
            "AUTO_FULL_EVAL_SOURCE_RUN_ID": source_run_id,
            "AUTO_FULL_EVAL_SOURCE_STEP": str(global_step),
            "AUTO_FULL_EVAL_SOURCE_METRIC": metric_key,
            "AUTO_FULL_EVAL_SOURCE_VALUE": f"{metric_value:.8f}",
        }
    )
    try:
        env.update(_parse_env_overrides(getattr(args, "auto_full_eval_extra_env", [])))
    except Exception as exc:
        logger.warning(f"[AUTO_FULL_EVAL] invalid extra env overrides: {exc}")
        return None

    cmd = ["sbatch"]
    partition = getattr(args, "auto_full_eval_partition", "") or ""
    if partition:
        cmd.append(f"--partition={partition}")
    cmd.append(launcher)

    try:
        proc = subprocess.run(
            cmd,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
    except Exception as exc:
        stderr = getattr(exc, "stderr", "") or ""
        logger.warning(
            f"[AUTO_FULL_EVAL] sbatch failed for step={global_step}: {exc} {stderr}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "auto_full_eval/submit_failed": 1,
                    "auto_full_eval/source_step": global_step,
                },
                step=global_step,
            )
        return None

    output = (proc.stdout or "").strip()
    match = re.search(r"Submitted batch job\s+(\d+)", output)
    job_id = match.group(1) if match else output
    logger.info(
        f"[AUTO_FULL_EVAL] submitted job={job_id} step={global_step} "
        f"checkpoint={immutable_checkpoint_path}"
    )
    if wandb_run is not None:
        wandb_run.log(
            {
                "auto_full_eval/source_step": global_step,
                "auto_full_eval/source_value": metric_value,
                "auto_full_eval/submitted": 1,
                "auto_full_eval/slurm_job_id": int(job_id) if job_id.isdigit() else 0,
            },
            step=global_step,
        )
    return job_id


# ==================== Phoneme Append ====================

PHONEME_SEP = " [SEP] PHONEMES: "

_G2P_INSTANCE = None


def _get_g2p():
    """Lazy-init g2p_en for eval/neg term encoding (main process only, small sets)."""
    global _G2P_INSTANCE
    if _G2P_INSTANCE is None:
        from g2p_en import G2p
        _G2P_INSTANCE = G2p()
    return _G2P_INSTANCE


def g2p_phoneme_str(term_text: str) -> str:
    """Compute ARPAbet phoneme string on-the-fly via g2p_en. For small eval sets only."""
    g2p = _get_g2p()
    phones = [p for p in g2p(term_text) if p.strip()]
    return " ".join(phones)


def append_phoneme_str(term_text: str, phonemes: str) -> str:
    """Append pre-computed phoneme string to term text for text encoder input.
    The 'phonemes' field is expected to be pre-computed ARPAbet in the JSONL."""
    if not term_text or not phonemes:
        return term_text
    return term_text + PHONEME_SEP + phonemes


def _apply_text_input_prefix(texts: Sequence[str], prefix: str) -> List[str]:
    if not prefix:
        return list(texts)
    return [f"{prefix}{t}" for t in texts]


def _tokenize_texts(
    text_tokenizer,
    texts: Sequence[str],
    device: torch.device,
    *,
    text_input_prefix: str = "",
    padding=True,
):
    return text_tokenizer(
        _apply_text_input_prefix(texts, text_input_prefix),
        padding=padding,
        truncation=True,
        max_length=DEFAULT_TEXT_MAX_LENGTH,
        return_tensors="pt",
    ).to(device)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ======Configuration=====
DEFAULT_QWEN_AUDIO_MODEL_ID = "Atotti/Qwen3-Omni-AudioTransformer"
DEFAULT_TEXT_MODEL_ID = "BAAI/bge-m3"
DEFAULT_QWEN_AUDIO_FEATURE_EXTRACTOR_ID = "openai/whisper-large-v3"

DEFAULT_AUDIO_SAMPLE_RATE = 16000
DEFAULT_MIN_AUDIO_SAMPLES = 3000
DEFAULT_FIXED_AUDIO_SAMPLES = 30720
DEFAULT_FIXED_AUDIO_SECONDS = DEFAULT_FIXED_AUDIO_SAMPLES / DEFAULT_AUDIO_SAMPLE_RATE
DEFAULT_TEXT_MAX_LENGTH = 64
DEFAULT_TARGET_DIM = 1024
DEFAULT_AUDIO_HIDDEN_DIM = 2048
DEFAULT_BGE_M3_HIDDEN_DIM = 1024

TEXT_ENCODER_PRESETS: Dict[str, Dict[str, str]] = {
    "custom": {},
    "bge-m3": {"model_id": DEFAULT_TEXT_MODEL_ID, "input_prefix": ""},
    "bge-large-en-v1.5": {
        "model_id": "BAAI/bge-large-en-v1.5",
        "input_prefix": "",
    },
    "multilingual-e5-large": {
        "model_id": "intfloat/multilingual-e5-large",
        "input_prefix": "query: ",
    },
}

AUDIO_ENCODER_TYPES = ("qwen3_omni", "whisper", "wavlm")
AUDIO_INPUT_DTYPES = ("auto", "bf16", "fp32")
AUDIO_ENCODER_PRESETS: Dict[str, Dict[str, str]] = {
    "custom": {},
    "qwen3-omni": {
        "type": "qwen3_omni",
        "model_id": DEFAULT_QWEN_AUDIO_MODEL_ID,
        "feature_extractor_id": DEFAULT_QWEN_AUDIO_FEATURE_EXTRACTOR_ID,
    },
    "whisper-medium-en": {
        "type": "whisper",
        "model_id": "openai/whisper-medium.en",
        "feature_extractor_id": "openai/whisper-medium.en",
    },
    "whisper-mid-en": {
        "type": "whisper",
        "model_id": "openai/whisper-medium.en",
        "feature_extractor_id": "openai/whisper-medium.en",
    },
    "wavlm-large": {
        "type": "wavlm",
        "model_id": "microsoft/wavlm-large",
        "feature_extractor_id": "microsoft/wavlm-large",
    },
    "wavlm-large-plus": {
        "type": "wavlm",
        "model_id": "microsoft/wavlm-large",
        "feature_extractor_id": "microsoft/wavlm-large",
    },
}

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
# Chunk size for bank-side matmul in per-sample HN mining.
# Tuned so the peak [B, W, chunk] activation at B=1536, W~180 stays under
# ~18 GB at bf16 on a 48 GB A6000 while keeping chunk count low enough that
# per-mine wall-clock is fractions of a second at full-bank scale (1.3M).
DEFAULT_HARD_NEG_MINE_CHUNK = 32768

DEFAULT_EVAL_GLOSSARY_SIZES: List[int] = []
DEFAULT_EVAL_WIKI_GLOSSARY = ""
DEFAULT_ACL_DEV_JSONL = ""
DEFAULT_TAGGED_ACL_DEV_JSONL = ""
DEFAULT_MEDICINE_DEV_JSONL = ""
DEFAULT_BEST_METRIC = ""
DEFAULT_EVAL_TERM_ENCODE_BATCH = 512
DEFAULT_EVAL_SCORE_DEVICE = "cuda"
DEFAULT_EVAL_SCORE_QUERY_CHUNK = 256
DEFAULT_EVAL_SCORE_TEXT_CHUNK = 1024

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


def _limit_eval_samples(
    samples: List[Dict],
    limit: int,
    seed: int,
    eval_name: str,
    *,
    is_main: bool,
) -> List[Dict]:
    """Return a deterministic random subset for cheap inline eval smoke checks."""
    total = len(samples)
    limit = max(0, int(limit or 0))
    if limit <= 0 or total <= limit:
        if is_main and limit > 0:
            logger.info(
                f"[EVAL_SAMPLE_LIMIT] {eval_name}: using all {total:,} samples "
                f"(limit={limit:,})"
            )
        return samples
    rng = random.Random(int(seed))
    keep_indices = sorted(rng.sample(range(total), limit))
    if is_main:
        logger.info(
            f"[EVAL_SAMPLE_LIMIT] {eval_name}: sampled {limit:,}/{total:,} "
            f"samples with seed={seed}"
        )
    return [samples[i] for i in keep_indices]


SIGNED_INT64_MASK = (1 << 63) - 1

# ---- Threshold-Consistent Margin (TCM) auxiliary loss ----
# Zhang et al., "Learning Threshold-Consistent Margin Loss", ICLR 2024.
# Absolute-threshold penalty added on top of InfoNCE to calibrate the
# retriever's cosine-similarity operating point across domains:
#   L_TCM_pos = mean_{i,j in pos}  relu(T_beta  - cos_sim)^p
#   L_TCM_neg = mean_{i,j in neg}  relu(cos_sim - T_alpha)^p
# Defaults come from the Config C sim-distribution diagnostic
# (see documents/code/offline_evaluation/retriever_sim_distribution.py).
DEFAULT_TCM_LOSS_WEIGHT = 0.0      # 0 disables TCM (additive auxiliary loss).
DEFAULT_TCM_POS_THRESHOLD = 0.7    # push positives above this cos-sim.
DEFAULT_TCM_NEG_THRESHOLD = 0.4    # push negatives below this cos-sim.
DEFAULT_TCM_LOSS_FORM = "squared_hinge"  # one of TCM_LOSS_FORMS below.
TCM_LOSS_FORMS = ("squared_hinge", "hinge")
# Reduction for TCM pos/neg penalty:
#   "mean_all":  L = sum(relu^p) / total_pair_count.  Matches paper formulation
#                when batch is small; with O(10k) negatives per anchor the
#                mean is dominated by non-violating pairs and the gradient
#                signal per violating pair collapses.
#   "mean_viol": L = sum(relu^p) / num_violating_pairs. Keeps the average
#                violation magnitude meaningful under very large negative
#                banks; recommended default for this codebase.
DEFAULT_TCM_REDUCTION = "mean_viol"
TCM_REDUCTIONS = ("mean_all", "mean_viol")
DEFAULT_TCM_NEG_SCOPE = "all"
TCM_NEG_SCOPES = ("all", "topk")

# ---- Hard Contrastive Loss (HCL) importance-sample reweighting ----
# Robinson et al., "Contrastive Learning with Hard Negative Samples",
# ICLR 2021.  Each negative receives an importance weight
# w_j ∝ exp(beta * s_j) normalized so the mean weight per row is 1.
# beta=0 disables (collapses to uniform InfoNCE weighting);
# beta→∞ concentrates entirely on the single hardest negative.
DEFAULT_HCL_BETA = 0.0
# ======Configuration=====


# ==================== Hashing helpers ====================


# ---- Term-id normalization (near-variant HN false-positive suppression) ----
# Collision analysis (analyze_hn_variant_collision.py) found that under
# strict blake2b(term_text) the HN miner surfaces plural/punctuation/
# substring variants of the anchor GT as "hard negatives": at n=200 sampled
# train anchors, 50.5% have an aggressive-normalization-equal variant in
# the bank, and the average top-64 contains ~7.7 bank entries with SM-ratio
# >= 0.80 vs the GT. InfoNCE at tau=0.07 pushes these variants away from GT,
# which is a destructive signal that grows with K. The fix is to normalize
# ONLY the hash input — the text encoder still sees the original surface
# form, but term_id("propositions") == term_id("proposition"), so the
# existing gt_match / fn_mask paths correctly recognize them as positives
# (or drops them from the denominator).
#
# Modes:
#   "none"        -> legacy behavior, bit-for-bit compatible with pre-fix ckpts
#   "lower_strip" -> .lower().strip() (almost no-op since term_key already is)
#   "aggressive"  -> lower+strip + punctuation strip + naive plural strip
# Runtime knob, set by parse_args()->main() at startup.  Never changed mid-run.
TERM_ID_NORMALIZE_MODES = ("none", "lower_strip", "aggressive")
DEFAULT_TERM_ID_NORMALIZE = "none"
_TERM_ID_NORMALIZE_MODE = DEFAULT_TERM_ID_NORMALIZE

_TERM_ID_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_TERM_ID_MULTISPACE_RE = re.compile(r"\s+")


def set_term_id_normalize_mode(mode: str) -> None:
    """Set the module-level term_id normalization mode (call once at startup)."""
    assert mode in TERM_ID_NORMALIZE_MODES, (
        f"term_id_normalize must be in {TERM_ID_NORMALIZE_MODES}, got {mode!r}"
    )
    global _TERM_ID_NORMALIZE_MODE
    _TERM_ID_NORMALIZE_MODE = mode


def _normalize_term_for_id(term_text: str) -> str:
    """Normalize a term's surface form for term_id hashing only.

    IMPORTANT: this does NOT change the text that reaches the encoder —
    it only folds near-variants into a common hash so the existing
    false-negative / positive masks can catch them.
    """
    if _TERM_ID_NORMALIZE_MODE == "none":
        return term_text
    t = term_text.strip().lower()
    if _TERM_ID_NORMALIZE_MODE == "lower_strip":
        return t
    # aggressive
    t = _TERM_ID_PUNCT_RE.sub(" ", t)
    t = _TERM_ID_MULTISPACE_RE.sub(" ", t).strip()
    toks = []
    for w in t.split(" "):
        if len(w) > 4 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 3 and w.endswith("es") and not w.endswith(("ses", "xes")):
            w = w[:-2]
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        toks.append(w)
    return " ".join(toks)


def stable_term_id(term_text: str) -> int:
    if not term_text:
        return INVALID_ID_SENTINEL
    norm = _normalize_term_for_id(term_text)
    if not norm:
        return INVALID_ID_SENTINEL
    digest = hashlib.blake2b(norm.encode("utf-8"), digest_size=8).digest()
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


def _sample_term_key(sample: Dict[str, Any]) -> str:
    return (sample.get("term_key", "") or sample.get("term", "") or "").strip().lower()


def _dedupe_ints(values: Sequence[int]) -> List[int]:
    out: List[int] = []
    seen: set[int] = set()
    for value in values:
        ivalue = int(value)
        if ivalue == INVALID_ID_SENTINEL or ivalue in seen:
            continue
        seen.add(ivalue)
        out.append(ivalue)
    return out


def attach_chunk_positive_term_ids(samples: List[Dict]) -> Dict[str, float]:
    """Attach all known GT term_ids for each speech chunk to every sample row.

    Training JSONL is one row per (chunk, term).  For hard-negative masking we
    need the row to know every term that is valid for its speech chunk, not just
    the current row's term.
    """
    group_to_tids: Dict[str, set[int]] = defaultdict(set)
    for sample in samples:
        group_key = build_speech_group_key(sample)
        term_id = stable_term_id(_sample_term_key(sample))
        if group_key and term_id != INVALID_ID_SENTINEL:
            group_to_tids[group_key].add(term_id)

    multi_term_groups = 0
    max_terms_per_group = 0
    for tids in group_to_tids.values():
        max_terms_per_group = max(max_terms_per_group, len(tids))
        if len(tids) > 1:
            multi_term_groups += 1

    rows_with_multi_term_group = 0
    for sample in samples:
        group_key = build_speech_group_key(sample)
        own_tid = stable_term_id(_sample_term_key(sample))
        tids = list(group_to_tids.get(group_key, set()))
        if own_tid != INVALID_ID_SENTINEL:
            tids.append(own_tid)
        deduped = _dedupe_ints(tids)
        sample["_chunk_positive_term_ids"] = deduped
        if len(deduped) > 1:
            rows_with_multi_term_group += 1

    n_groups = max(len(group_to_tids), 1)
    n_rows = max(len(samples), 1)
    return {
        "groups": float(len(group_to_tids)),
        "multi_term_groups": float(multi_term_groups),
        "multi_term_group_rate": float(multi_term_groups / n_groups),
        "rows_with_multi_term_group": float(rows_with_multi_term_group),
        "rows_with_multi_term_group_rate": float(rows_with_multi_term_group / n_rows),
        "max_terms_per_group": float(max_terms_per_group),
    }


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


class TransformerPooling(nn.Module):
    """1-layer cross-attention pooling with a learnable query token.

    A single learnable query attends to variable-length audio frames
    via standard multi-head attention, producing a fixed-size vector.
    Much more expressive than scalar attentive pooling: the query can
    learn to focus on term-relevant temporal patterns.
    """

    POOL_NUM_HEADS = 8
    POOL_FFN_MULT = 4
    POOL_DROPOUT = 0.1

    def __init__(self, input_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=self.POOL_NUM_HEADS,
            dropout=self.POOL_DROPOUT,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * self.POOL_FFN_MULT),
            nn.GELU(),
            nn.Dropout(self.POOL_DROPOUT),
            nn.Linear(input_dim * self.POOL_FFN_MULT, input_dim),
            nn.Dropout(self.POOL_DROPOUT),
        )
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] audio frame features
            mask: [B, T] bool, True = valid frame
        Returns:
            [B, D] pooled representation
        """
        B = x.size(0)
        query = self.query.expand(B, -1, -1)

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask

        attn_out, _ = self.cross_attn(
            query=query, key=x, value=x,
            key_padding_mask=key_padding_mask,
        )
        h = self.norm1(query + attn_out)
        h = self.norm2(h + self.ffn(h))
        return h.squeeze(1)


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


TEXT_POOLING_MODES = {"cls", "mean", "max", "cls_mean", "cls_max", "gated"}


class BgeM3TextEncoder(nn.Module):
    def __init__(
        self,
        model_id: str,
        lora_rank: int,
        lora_alpha: int,
        target_modules: Optional[List[str]],
        full_finetune: bool = False,
        sparse_weight: float = 0.0,
        text_pooling: str = "cls",
        use_colbert: bool = False,
    ):
        super().__init__()
        self.use_colbert = use_colbert
        if use_colbert:
            assert sparse_weight == 0.0, (
                "ColBERT multi-vector mode is incompatible with sparse_weight > 0"
            )
            assert text_pooling == "cls", (
                "ColBERT multi-vector mode ignores text_pooling; set text_pooling='cls'"
            )
        assert text_pooling in TEXT_POOLING_MODES, (
            f"Unknown text_pooling '{text_pooling}', expected one of {TEXT_POOLING_MODES}"
        )
        base_encoder = AutoModel.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, add_pooling_layer=False
        )
        self.hidden_dim = int(
            getattr(base_encoder.config, "hidden_size", DEFAULT_BGE_M3_HIDDEN_DIM)
        )
        self.encoder = base_encoder
        self.full_finetune = full_finetune
        self.sparse_weight = sparse_weight
        self.text_pooling = text_pooling

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

        if self.sparse_weight > 0.0:
            self.sparse_linear = nn.Linear(
                self.hidden_dim, 1, dtype=torch.bfloat16
            )
            self._load_pretrained_sparse_linear(model_id)
            logger.info(
                f"[TEXT_ENCODER] Hybrid dense+sparse mode: "
                f"sparse_weight={self.sparse_weight:.2f}"
            )

        D = self.hidden_dim
        if text_pooling in ("cls_mean", "cls_max"):
            self.pool_gate = nn.Linear(D * 2, D, dtype=torch.bfloat16)
        elif text_pooling == "gated":
            self.pool_gate = nn.Linear(D * 3, D, dtype=torch.bfloat16)

        if use_colbert:
            self.colbert_linear = nn.Linear(D, D, dtype=torch.bfloat16)
            self._load_pretrained_colbert_linear(model_id)
            logger.info("[TEXT_ENCODER] ColBERT multi-vector mode enabled")
        else:
            logger.info(f"[TEXT_ENCODER] text_pooling={text_pooling}")

    def _load_pretrained_sparse_linear(self, model_id: str) -> None:
        """Load BGE-M3's pretrained sparse_linear.pt if available."""
        try:
            from huggingface_hub import hf_hub_download
            pt_path = hf_hub_download(model_id, "sparse_linear.pt")
            state = torch.load(pt_path, map_location="cpu")
            self.sparse_linear.load_state_dict(state)
            logger.info(f"[TEXT_ENCODER] Loaded pretrained sparse_linear from {pt_path}")
        except Exception as exc:
            logger.warning(
                f"[TEXT_ENCODER] Could not load pretrained sparse_linear: {exc}. "
                f"Using random init."
            )

    def _load_pretrained_colbert_linear(self, model_id: str) -> None:
        """Load BGE-M3's pretrained colbert_linear.pt if available."""
        try:
            from huggingface_hub import hf_hub_download
            pt_path = hf_hub_download(model_id, "colbert_linear.pt")
            state = torch.load(pt_path, map_location="cpu")
            self.colbert_linear.load_state_dict(state)
            logger.info(f"[TEXT_ENCODER] Loaded pretrained colbert_linear from {pt_path}")
        except Exception as exc:
            logger.warning(
                f"[TEXT_ENCODER] Could not load pretrained colbert_linear: {exc}. "
                f"Using random init."
            )

    @staticmethod
    def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).float()
        return (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-9)

    @staticmethod
    def _masked_max(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1)
        hidden = hidden.masked_fill(~mask_f.bool(), -1e9)
        return hidden.max(dim=1).values

    def _pool(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        cls_emb = hidden[:, 0, :]
        if self.text_pooling == "cls":
            return cls_emb
        if self.text_pooling == "mean":
            return self._masked_mean(hidden, attention_mask)
        if self.text_pooling == "max":
            return self._masked_max(hidden, attention_mask)
        if self.text_pooling == "cls_mean":
            mean_emb = self._masked_mean(hidden, attention_mask)
            return self.pool_gate(torch.cat([cls_emb, mean_emb], dim=-1))
        if self.text_pooling == "cls_max":
            max_emb = self._masked_max(hidden, attention_mask)
            return self.pool_gate(torch.cat([cls_emb, max_emb], dim=-1))
        assert self.text_pooling == "gated"
        mean_emb = self._masked_mean(hidden, attention_mask)
        max_emb = self._masked_max(hidden, attention_mask)
        return self.pool_gate(torch.cat([cls_emb, mean_emb, max_emb], dim=-1))

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state

        if self.use_colbert:
            # ColBERT multi-vector: skip CLS (index 0), project all tokens
            token_hidden = hidden[:, 1:]  # [B, T-1, D]
            token_mask = attention_mask[:, 1:]  # [B, T-1]
            colbert_vecs = self.colbert_linear(token_hidden)  # [B, T-1, D]
            colbert_vecs = colbert_vecs * token_mask.unsqueeze(-1).float()
            return F.normalize(colbert_vecs, p=2, dim=-1)

        pooled = self._pool(hidden, attention_mask)

        if self.sparse_weight > 0.0:
            token_weights = torch.relu(self.sparse_linear(hidden)).squeeze(-1)
            token_weights = token_weights * attention_mask.float()
            weight_sum = token_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            token_weights_norm = token_weights / weight_sum
            sparse_emb = (token_weights_norm.unsqueeze(-1) * hidden).sum(dim=1)
            pooled = (
                (1.0 - self.sparse_weight) * pooled
                + self.sparse_weight * sparse_emb
            )

        return F.normalize(pooled, p=2, dim=-1)


POOLING_TYPES = {"attentive", "transformer"}

# Multi-Scale Max-Sim: window sizes in encoder frames (~12.5 fps for 1.92s chunks)
# Derived from ACL6060 term duration analysis:
#   1-word mean=0.55s(7fr), 2-word mean=1.01s(13fr), 3-word mean=1.50s(19fr)
# Windows sized slightly larger than means to fully wrap pronunciations:
#   6fr=0.48s (fast single words), 10fr=0.80s (most single words, P80),
#   16fr=1.28s (two-word terms), 24fr=1.92s (full chunk safety net)
MAXSIM_DEFAULT_WINDOWS = [6, 10, 16, 24]
MAXSIM_DEFAULT_STRIDE = 2
ENCODER_FPS = 12.5
FRAME_SEC = 1.0 / ENCODER_FPS


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
        pooling_type: str = "attentive",
        use_maxsim: bool = False,
        maxsim_windows: Optional[List[int]] = None,
        maxsim_stride: int = MAXSIM_DEFAULT_STRIDE,
        audio_encoder_type: str = "qwen3_omni",
        audio_hidden_dim: int = 0,
    ):
        super().__init__()
        assert audio_encoder_type in AUDIO_ENCODER_TYPES, (
            f"Unknown audio_encoder_type '{audio_encoder_type}', "
            f"expected one of {AUDIO_ENCODER_TYPES}"
        )
        assert pooling_type in POOLING_TYPES, (
            f"Unknown pooling_type '{pooling_type}', expected one of {POOLING_TYPES}"
        )
        self.audio_encoder_type = audio_encoder_type
        self.pooling_type = pooling_type
        self.use_maxsim = use_maxsim
        self.maxsim_windows = maxsim_windows or MAXSIM_DEFAULT_WINDOWS
        self.maxsim_stride = maxsim_stride

        if audio_encoder_type == "qwen3_omni":
            self.audio_encoder = Qwen3OmniMoeAudioEncoder.from_pretrained(
                model_id, dtype=torch.bfloat16
            )
            inferred_hidden_dim = DEFAULT_AUDIO_HIDDEN_DIM
            if hasattr(self.audio_encoder, "conv2d1"):
                self.audio_encoder.get_input_embeddings = (
                    lambda: self.audio_encoder.conv2d1
                )
            default_lora_targets = [
                "q_proj", "k_proj", "v_proj", "out_proj",
                "fc1", "fc2", "proj1", "proj2",
            ]
        elif audio_encoder_type == "whisper":
            whisper = WhisperModel.from_pretrained(
                model_id, torch_dtype=torch.bfloat16
            )
            self.audio_encoder = whisper.encoder
            inferred_hidden_dim = int(
                getattr(self.audio_encoder.config, "d_model", DEFAULT_AUDIO_HIDDEN_DIM)
            )
            default_lora_targets = [
                "q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2",
            ]
        else:
            self.audio_encoder = AutoModel.from_pretrained(
                model_id, torch_dtype=torch.bfloat16
            )
            if hasattr(getattr(self.audio_encoder, "config", None), "layerdrop"):
                old_layerdrop = float(getattr(self.audio_encoder.config, "layerdrop", 0.0))
                self.audio_encoder.config.layerdrop = 0.0
                encoder_config = getattr(getattr(self.audio_encoder, "encoder", None), "config", None)
                if encoder_config is not None and hasattr(encoder_config, "layerdrop"):
                    encoder_config.layerdrop = 0.0
                if old_layerdrop != 0.0:
                    logger.info(
                        "[AUDIO_ENCODER] disabled WavLM layerdrop "
                        f"({old_layerdrop} -> 0.0) for DDP GradCache stability."
                    )
            wavlm_conv_layers = getattr(
                getattr(self.audio_encoder, "feature_extractor", None),
                "conv_layers",
                None,
            )
            if wavlm_conv_layers:
                first_wavlm_conv = wavlm_conv_layers[0].conv
                self.audio_encoder.get_input_embeddings = (
                    lambda first_wavlm_conv=first_wavlm_conv: first_wavlm_conv
                )
                logger.info(
                    "[GRAD_CKPT] patched WavLM get_input_embeddings() to the "
                    "first feature-extractor conv layer for PEFT checkpointing."
                )
            inferred_hidden_dim = int(
                getattr(self.audio_encoder.config, "hidden_size", DEFAULT_AUDIO_HIDDEN_DIM)
            )
            default_lora_targets = [
                "q_proj", "k_proj", "v_proj", "out_proj",
                "intermediate_dense", "output_dense",
            ]
        self.audio_hidden_dim = int(audio_hidden_dim or inferred_hidden_dim)

        if hasattr(self.audio_encoder, "gradient_checkpointing_enable"):
            self.audio_encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        if use_lora:
            if lora_target_modules is None:
                lora_target_modules = default_lora_targets
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

        if not use_maxsim:
            if pooling_type == "transformer":
                self.pooler = TransformerPooling(self.audio_hidden_dim)
            else:
                self.pooler = AttentivePooling(self.audio_hidden_dim)
        self.projector = nn.Linear(self.audio_hidden_dim, target_dim)

        if learn_temp:
            self.logit_scale = nn.Parameter(
                torch.ones([]) * np.log(1.0 / temperature)
            )
        else:
            self.register_buffer(
                "logit_scale", torch.tensor(np.log(1.0 / temperature))
            )

        if use_maxsim:
            logger.info(
                f"[MAXSIM] Multi-Scale Max-Sim enabled: "
                f"windows={self.maxsim_windows} stride={self.maxsim_stride}"
            )
        logger.info(
            f"[AUDIO_ENCODER] type={audio_encoder_type} model_id={model_id} "
            f"hidden_dim={self.audio_hidden_dim}"
        )

    def _multiscale_pool(
        self, projected_seq: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply multi-scale average pooling with dense stride to produce window embeddings.

        Uses a fixed small stride (e.g. 2 frames = 160ms) across all window sizes
        to guarantee at least one window perfectly aligns with any term pronunciation.

        Args:
            projected_seq: [B, T, D] projected hidden states (already masked)
            mask: [B, T] boolean validity mask

        Returns:
            [B, W_total, D] window embeddings (NOT L2-normalized)
        """
        B, T, D = projected_seq.shape
        max_valid_len = int(mask.sum(dim=1).max().item()) if mask.numel() else T
        max_valid_len = max(1, min(T, max_valid_len))
        if max_valid_len < T:
            projected_seq = projected_seq[:, :max_valid_len, :]
            mask = mask[:, :max_valid_len]
            T = max_valid_len

        x_t = projected_seq.transpose(1, 2)  # [B, D, T]
        mask_1d = mask.float().unsqueeze(1)   # [B, 1, T]
        stride = self.maxsim_stride

        all_windows = []
        for w in self.maxsim_windows:
            if w >= T:
                mask_f = mask.unsqueeze(-1).float()  # [B, T, 1]
                pooled = (
                    (projected_seq * mask_f).sum(dim=1, keepdim=True)
                    / mask_f.sum(dim=1, keepdim=True).clamp(min=1e-9)
                )  # [B, 1, D]
                all_windows.append(pooled)
            else:
                sum_pool = F.avg_pool1d(x_t, kernel_size=w, stride=stride) * w
                cnt_pool = F.avg_pool1d(mask_1d, kernel_size=w, stride=stride) * w
                cnt_pool = cnt_pool.clamp(min=1e-9)
                pooled = (sum_pool / cnt_pool).transpose(1, 2)  # [B, T_w, D]
                all_windows.append(pooled)

        return torch.cat(all_windows, dim=1)  # [B, W_total, D]

    def forward(
        self, input_features: torch.Tensor, feature_lens: torch.Tensor
    ) -> torch.Tensor:
        if self.audio_encoder_type == "qwen3_omni":
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
                    output_lens = [
                        max(1, round(x / ratio)) for x in feature_lens.tolist()
                    ]
                    output_lens[-1] = hidden_states.shape[0] - sum(output_lens[:-1])

                from torch.nn.utils.rnn import pad_sequence

                parts = torch.split(hidden_states, output_lens, dim=0)
                hidden_states = pad_sequence(parts, batch_first=True)
                feature_lens = torch.tensor(output_lens, device=hidden_states.device)
        elif self.audio_encoder_type == "whisper":
            orig_feature_lens = feature_lens
            expected_mel_len = int(
                getattr(self.audio_encoder.config, "max_source_positions", 1500) * 2
            )
            cur_mel_len = int(input_features.size(-1))
            if cur_mel_len < expected_mel_len:
                input_features = F.pad(
                    input_features, (0, expected_mel_len - cur_mel_len)
                )
            elif cur_mel_len > expected_mel_len:
                input_features = input_features[..., :expected_mel_len]
                if orig_feature_lens is not None:
                    orig_feature_lens = orig_feature_lens.clamp(max=expected_mel_len)
            outputs = self.audio_encoder(input_features=input_features)
            hidden_states = outputs.last_hidden_state
            if orig_feature_lens is None:
                feature_lens = torch.full(
                    (hidden_states.size(0),),
                    hidden_states.size(1),
                    dtype=torch.long,
                    device=hidden_states.device,
                )
            else:
                downsample = input_features.size(-1) / max(1, hidden_states.size(1))
                feature_lens = torch.ceil(
                    orig_feature_lens.to(hidden_states.device).float() / downsample
                ).long()
                feature_lens = feature_lens.clamp(min=1, max=hidden_states.size(1))
        else:
            attention_mask = None
            if feature_lens is not None:
                attention_mask = (
                    torch.arange(input_features.size(1), device=input_features.device)
                    .expand(input_features.size(0), input_features.size(1))
                    < feature_lens.unsqueeze(1)
                ).long()
            outputs = self.audio_encoder(
                input_values=input_features,
                attention_mask=attention_mask,
            )
            hidden_states = outputs.last_hidden_state
            feature_lens = torch.full(
                (hidden_states.size(0),),
                hidden_states.size(1),
                dtype=torch.long,
                device=hidden_states.device,
            )

        batch_size, max_len, _ = hidden_states.shape
        mask = (
            torch.arange(max_len, device=hidden_states.device).expand(
                batch_size, max_len
            )
            < feature_lens.unsqueeze(1)
        )

        if self.use_maxsim:
            projected_seq = self.projector(hidden_states)  # [B, T, D]
            projected_seq = projected_seq * mask.unsqueeze(-1).float()
            window_embs = self._multiscale_pool(projected_seq, mask)  # [B, W, D]
            return F.normalize(window_embs, p=2, dim=-1)

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
        fixed_audio_samples: int = DEFAULT_FIXED_AUDIO_SAMPLES,
    ):
        self.samples = samples
        self._remap_src = os.environ.get("AUDIO_PATH_REMAP_SRC", "").strip()
        self._remap_dst = os.environ.get("AUDIO_PATH_REMAP_DST", "").strip()
        self._force_dummy = force_dummy_audio
        self._augment_synth = augment_synth
        self._fixed_audio_samples = fixed_audio_samples
        self._rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = dict(self.samples[idx])
        term_text = _sample_term_key(sample)
        phonemes = (sample.get("phonemes", "") or "").strip()
        passthrough_meta = {
            "term": str(sample.get("term", "") or ""),
            "term_key": str(sample.get("term_key", "") or ""),
            "chunk_src_text": str(sample.get("chunk_src_text", "") or ""),
            "domain": str(sample.get("domain", "") or ""),
            "sample_id": str(sample.get("sample_id", "") or ""),
            "context_duration_tag": str(sample.get("context_duration_tag", "") or ""),
            "chunk_duration_sec": sample.get("chunk_duration_sec", ""),
            "context_duration_sec": sample.get("context_duration_sec", ""),
            "mfa_term_start_in_chunk": sample.get("mfa_term_start_in_chunk", ""),
            "mfa_term_end_in_chunk": sample.get("mfa_term_end_in_chunk", ""),
            "mfa_term_duration": sample.get("mfa_term_duration", ""),
            "mfa_locate_method": str(sample.get("mfa_locate_method", "") or ""),
            "source_seg_id": str(sample.get("source_seg_id", "") or ""),
            "source_audio": str(sample.get("source_audio", "") or ""),
            "source_start_sample": str(sample.get("source_start_sample", "") or ""),
        }
        positive_term_ids = _dedupe_ints(
            sample.get("_chunk_positive_term_ids", []) or [stable_term_id(term_text)]
        )

        mfa_start_in_chunk = sample.get("mfa_term_start_in_chunk")
        mfa_end_in_chunk = sample.get("mfa_term_end_in_chunk")
        if mfa_start_in_chunk is not None and mfa_end_in_chunk is not None:
            mfa_term_start = float(mfa_start_in_chunk)
            mfa_term_end = float(mfa_end_in_chunk)
        else:
            mfa_term_start = -1.0
            mfa_term_end = -1.0

        if self._force_dummy:
            dummy = np.zeros(self._fixed_audio_samples, dtype=np.float32)
            return {
                "audio": dummy,
                "term_text": term_text,
                "phonemes": phonemes,
                "skip_sample": True,
                "chunk_audio_path": "DUMMY",
                "utter_id": str(sample.get("utter_id", "")),
                "chunk_idx": str(sample.get("chunk_idx", "")),
                "mfa_term_start": -1.0,
                "mfa_term_end": -1.0,
                "positive_term_ids": positive_term_ids,
                **passthrough_meta,
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
                "phonemes": phonemes,
                "skip_sample": False,
                "chunk_audio_path": audio_path,
                "utter_id": str(sample.get("utter_id", "")),
                "chunk_idx": str(sample.get("chunk_idx", "")),
                "mfa_term_start": mfa_term_start,
                "mfa_term_end": mfa_term_end,
                "positive_term_ids": positive_term_ids,
                **passthrough_meta,
            }
        except Exception as exc:
            logger.warning(f"[AUDIO_LOAD_FAIL] path={audio_path} error={exc}")
            return {
                "audio": None,
                "term_text": "",
                "phonemes": "",
                "skip_sample": True,
                "chunk_audio_path": audio_path,
                "utter_id": str(sample.get("utter_id", "")),
                "chunk_idx": str(sample.get("chunk_idx", "")),
                "mfa_term_start": -1.0,
                "mfa_term_end": -1.0,
                "positive_term_ids": [],
                **passthrough_meta,
            }


def collate_fn(
    batch: List[Dict],
    feature_extractor,
    use_phoneme_append: bool = False,
    fixed_audio_samples: int = DEFAULT_FIXED_AUDIO_SAMPLES,
    audio_encoder_type: str = "qwen3_omni",
) -> Dict:
    dummy_audio = np.zeros(fixed_audio_samples, dtype=np.float32)
    audio_list: List[np.ndarray] = []
    text_list: List[str] = []
    valid_list: List[bool] = []
    mfa_start_list: List[float] = []
    mfa_end_list: List[float] = []
    positive_term_id_rows: List[List[int]] = []
    samples: List[Dict] = []

    for s in batch:
        audio = s.get("audio")
        skip = bool(s.get("skip_sample", False))
        if audio is None or len(audio) <= DEFAULT_MIN_AUDIO_SAMPLES:
            audio = dummy_audio
            skip = True

        if len(audio) < fixed_audio_samples:
            audio = np.pad(
                audio, (0, fixed_audio_samples - len(audio)), mode="constant"
            )
        elif len(audio) > fixed_audio_samples:
            audio = audio[:fixed_audio_samples]

        audio_list.append(audio)
        raw_text = s.get("term_text", "")
        if use_phoneme_append:
            phonemes = s.get("phonemes", "")
            text_list.append(append_phoneme_str(raw_text, phonemes))
        else:
            text_list.append(raw_text)
        valid_list.append(bool(s.get("term_text")) and (not skip))
        mfa_start_list.append(float(s.get("mfa_term_start", -1.0)))
        mfa_end_list.append(float(s.get("mfa_term_end", -1.0)))
        positive_term_id_rows.append(_dedupe_ints(s.get("positive_term_ids", [])))
        samples.append(s)

    if audio_encoder_type == "wavlm":
        inputs = feature_extractor(
            audio_list,
            sampling_rate=DEFAULT_AUDIO_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        feats = inputs.input_values
        if hasattr(inputs, "attention_mask"):
            feat_lens = inputs.attention_mask.sum(dim=-1).long()
        else:
            feat_lens = torch.tensor([len(a) for a in audio_list], dtype=torch.long)
    else:
        inputs = feature_extractor(
            audio_list,
            sampling_rate=DEFAULT_AUDIO_SAMPLE_RATE,
            return_tensors="pt",
            padding=False,
        )
        feats = inputs.input_features
        feat_lens = torch.full((feats.size(0),), feats.size(-1), dtype=torch.long)
    max_pos_terms = max((len(row) for row in positive_term_id_rows), default=0)
    max_pos_terms = max(1, max_pos_terms)
    positive_term_ids = torch.full(
        (len(positive_term_id_rows), max_pos_terms),
        INVALID_ID_SENTINEL,
        dtype=torch.long,
    )
    positive_term_mask = torch.zeros(
        (len(positive_term_id_rows), max_pos_terms),
        dtype=torch.bool,
    )
    for row_idx, ids in enumerate(positive_term_id_rows):
        if not ids:
            continue
        n_ids = min(len(ids), max_pos_terms)
        positive_term_ids[row_idx, :n_ids] = torch.tensor(ids[:n_ids], dtype=torch.long)
        positive_term_mask[row_idx, :n_ids] = True

    return {
        "input_features": feats,
        "feature_lens": feat_lens,
        "term_texts": text_list,
        "valid_mask": torch.tensor(valid_list, dtype=torch.bool),
        "mfa_term_starts": torch.tensor(mfa_start_list, dtype=torch.float32),
        "mfa_term_ends": torch.tensor(mfa_end_list, dtype=torch.float32),
        "positive_term_ids": positive_term_ids,
        "positive_term_mask": positive_term_mask,
        "samples": samples,
    }


def _resolve_audio_sample_count(
    *,
    samples_arg: int,
    seconds_arg: float,
    default_samples: int,
    label: str,
) -> int:
    if samples_arg > 0 and seconds_arg > 0:
        raise ValueError(f"Use only one of --{label}_samples or --{label}_seconds")
    if seconds_arg > 0:
        samples_arg = int(round(seconds_arg * DEFAULT_AUDIO_SAMPLE_RATE))
    if samples_arg <= 0:
        samples_arg = default_samples
    if samples_arg <= DEFAULT_MIN_AUDIO_SAMPLES:
        raise ValueError(
            f"{label} length too short: {samples_arg} samples "
            f"(min>{DEFAULT_MIN_AUDIO_SAMPLES})"
        )
    return samples_arg


def resolve_fixed_audio_samples(args: argparse.Namespace) -> Tuple[int, int]:
    train_samples = _resolve_audio_sample_count(
        samples_arg=int(getattr(args, "fixed_audio_samples", 0) or 0),
        seconds_arg=float(getattr(args, "fixed_audio_seconds", 0.0) or 0.0),
        default_samples=DEFAULT_FIXED_AUDIO_SAMPLES,
        label="fixed_audio",
    )
    eval_samples = _resolve_audio_sample_count(
        samples_arg=int(getattr(args, "eval_fixed_audio_samples", 0) or 0),
        seconds_arg=float(getattr(args, "eval_fixed_audio_seconds", 0.0) or 0.0),
        default_samples=train_samples,
        label="eval_fixed_audio",
    )
    args.fixed_audio_samples = train_samples
    args.fixed_audio_seconds = train_samples / DEFAULT_AUDIO_SAMPLE_RATE
    args.eval_fixed_audio_samples = eval_samples
    args.eval_fixed_audio_seconds = eval_samples / DEFAULT_AUDIO_SAMPLE_RATE
    return train_samples, eval_samples


def _resolve_audio_input_dtype(args: argparse.Namespace) -> torch.dtype:
    dtype_arg = getattr(args, "audio_input_dtype", "auto")
    if dtype_arg == "bf16":
        return torch.bfloat16
    if dtype_arg == "fp32":
        return torch.float32
    # WavLM consumes raw waveform input_values; keep those fp32 by default.
    if getattr(args, "audio_encoder_type", "qwen3_omni") == "wavlm":
        return torch.float32
    return torch.bfloat16


def _move_audio_batch_to_device(
    batch: Dict,
    device: torch.device,
    args: argparse.Namespace,
    *,
    non_blocking: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    feats = batch["input_features"].to(device, non_blocking=non_blocking)
    feats = feats.to(_resolve_audio_input_dtype(args))
    flens = batch["feature_lens"].to(device, non_blocking=non_blocking)
    return feats, flens


def _term_ids_match_any(
    candidate_term_ids: torch.Tensor,
    positive_term_ids: torch.Tensor,
    positive_term_mask: torch.Tensor,
) -> torch.Tensor:
    """Return candidate-term membership in each row's positive term-id set.

    candidate_term_ids can be [N] for shared columns or [B, K] for per-row
    hard negatives.  The returned mask is [B, N] or [B, K].
    """
    positive_term_ids = positive_term_ids.to(candidate_term_ids.device)
    positive_term_mask = positive_term_mask.to(candidate_term_ids.device)
    if candidate_term_ids.ndim == 1:
        return (
            (candidate_term_ids.view(1, -1, 1) == positive_term_ids.unsqueeze(1))
            & positive_term_mask.unsqueeze(1)
        ).any(dim=2)
    if candidate_term_ids.ndim == 2:
        return (
            (candidate_term_ids.unsqueeze(2) == positive_term_ids.unsqueeze(1))
            & positive_term_mask.unsqueeze(1)
        ).any(dim=2)
    raise ValueError(f"candidate_term_ids must be 1D or 2D, got {candidate_term_ids.shape}")


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
        use_phoneme_append: bool = False,
        text_input_prefix: str = "",
    ) -> None:
        """
        Refresh the bank embeddings under the current text encoder weights.

        When DDP is initialized, the workload is sharded across all ranks with
        a ceil-shard pattern (rank r owns terms[r*shard_size : (r+1)*shard_size]).
        After each rank encodes its shard, ``torch.distributed.all_gather`` is
        used with zero-padding so every rank ends with an identical CPU copy of
        the full bank.  This is the critical optimization that turns a ~7 min
        serial refresh of 1.4M terms into a ~50 s parallel one on 8 GPUs.
        """
        text_encoder.eval()
        if use_phoneme_append:
            enc_terms = [append_phoneme_str(t, g2p_phoneme_str(t)) for t in self.terms]
        else:
            enc_terms = self.terms

        n_total = len(enc_terms)
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        if world_size > n_total:
            raise ValueError(
                f"NegativeTermBank.refresh: world_size={world_size} exceeds "
                f"bank size {n_total}; shrink world or grow the bank."
            )

        shard_size = (n_total + world_size - 1) // world_size
        start = rank * shard_size
        end = min(start + shard_size, n_total)
        local_terms = enc_terms[start:end]

        local_chunks: List[torch.Tensor] = []
        for i in range(0, len(local_terms), batch_size):
            chunk = local_terms[i : i + batch_size]
            tok = _tokenize_texts(
                text_tokenizer,
                chunk,
                device,
                text_input_prefix=text_input_prefix,
            )
            embs = text_encoder(tok.input_ids, tok.attention_mask).float()
            local_chunks.append(embs)

        if not local_chunks:
            raise RuntimeError(
                f"NegativeTermBank.refresh: rank={rank} got empty shard "
                f"(start={start}, end={end}, n_total={n_total})"
            )
        local_embs = torch.cat(local_chunks, dim=0)

        if world_size > 1:
            emb_dim = local_embs.shape[-1]
            pad_len = shard_size - local_embs.shape[0]
            if pad_len > 0:
                pad = torch.zeros(
                    pad_len, emb_dim, dtype=local_embs.dtype, device=device
                )
                local_padded = torch.cat([local_embs, pad], dim=0)
            else:
                local_padded = local_embs

            gathered = [
                torch.empty(
                    shard_size, emb_dim, dtype=local_embs.dtype, device=device
                )
                for _ in range(world_size)
            ]
            dist.all_gather(gathered, local_padded.contiguous())

            pieces = []
            for r in range(world_size):
                s = r * shard_size
                e = min(s + shard_size, n_total)
                pieces.append(gathered[r][: e - s])
            full_embs = torch.cat(pieces, dim=0)
        else:
            full_embs = local_embs

        if full_embs.shape[0] != n_total:
            raise RuntimeError(
                f"NegativeTermBank.refresh: expected {n_total} embeddings, "
                f"got {full_embs.shape[0]}"
            )
        self.embeddings = full_embs.cpu()
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
    def mine_hard_negatives_per_sample(
        self,
        speech_embs: torch.Tensor,
        local_term_ids: torch.Tensor,
        local_valid_mask: torch.Tensor,
        local_positive_term_ids: torch.Tensor,
        local_positive_term_mask: torch.Tensor,
        k_per_sample: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Each row gets its OWN top-K bank terms, without cross-batch
        deduplication.  The returned tensors
        have a leading batch dimension so that downstream loss code can treat
        them as per-row extra negative columns.

        Supports both [B, D] single-vector and [B, W, D] multi-scale Max-Sim
        speech embeddings.

        Returns:
            hn_embs: [B, K, D] on self._device, fp32 master precision
            hn_tids: [B, K] on local_term_ids.device
            count:   B * K (total hard-negative slots, for logging parity)
        """
        assert self.embeddings is not None, "refresh() must be called first"

        device = speech_embs.device
        speech_mine = speech_embs.detach().to(dtype=torch.bfloat16)
        bank_full = self.embeddings.to(device=device, dtype=torch.bfloat16, non_blocking=True)
        bank_tids = self.term_ids.to(local_term_ids.device)

        N = self.size
        B_local = local_term_ids.shape[0]
        actual_k = min(k_per_sample, N)
        chunk_size = min(DEFAULT_HARD_NEG_MINE_CHUNK, N)

        running_vals: Optional[torch.Tensor] = None
        running_idx: Optional[torch.Tensor] = None

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk_embs = bank_full[start:end]
            chunk_tids = bank_tids[start:end]

            if speech_mine.ndim == 3:
                chunk_sim = _maxsim_score(speech_mine, chunk_embs)
            else:
                chunk_sim = speech_mine @ chunk_embs.T

            gt_match = _term_ids_match_any(
                chunk_tids,
                local_positive_term_ids,
                local_positive_term_mask,
            )
            chunk_sim = chunk_sim.masked_fill(gt_match, float("-inf"))
            chunk_sim = chunk_sim.masked_fill(~local_valid_mask.unsqueeze(1), float("-inf"))

            chunk_k = min(actual_k, end - start)
            c_vals, c_pos = chunk_sim.topk(chunk_k, dim=1)
            c_global_idx = c_pos + start

            if running_vals is None:
                running_vals = c_vals
                running_idx = c_global_idx
            else:
                merged_v = torch.cat([running_vals, c_vals], dim=1)
                merged_i = torch.cat([running_idx, c_global_idx], dim=1)
                merged_topk = merged_v.topk(actual_k, dim=1)
                running_vals = merged_topk.values
                running_idx = merged_i.gather(1, merged_topk.indices)

        assert running_idx is not None, (
            f"NegativeTermBank.mine_hard_negatives_per_sample: empty bank (N={N})"
        )
        del bank_full

        flat_idx = running_idx.reshape(-1).cpu()
        hn_embs = self.embeddings[flat_idx].view(B_local, actual_k, -1).to(self._device)
        hn_tids = self.term_ids[flat_idx].view(B_local, actual_k).to(local_term_ids.device)
        return hn_embs, hn_tids, B_local * actual_k


# ==================== Loss ====================


def _resolve_tcm_branch_weights(
    *,
    tcm_loss_weight: float,
    tcm_pos_loss_weight: Optional[float],
    tcm_neg_loss_weight: Optional[float],
) -> Tuple[float, float]:
    pos_weight = (
        tcm_loss_weight
        if tcm_pos_loss_weight is None
        else float(tcm_pos_loss_weight)
    )
    neg_weight = (
        tcm_loss_weight
        if tcm_neg_loss_weight is None
        else float(tcm_neg_loss_weight)
    )
    return float(pos_weight), float(neg_weight)


def _resolve_tcm_branch_weights_from_args(args) -> Tuple[float, float]:
    return _resolve_tcm_branch_weights(
        tcm_loss_weight=getattr(args, "tcm_loss_weight", 0.0),
        tcm_pos_loss_weight=getattr(args, "tcm_pos_loss_weight", None),
        tcm_neg_loss_weight=getattr(args, "tcm_neg_loss_weight", None),
    )


def gradcache_train_step(
    batch: Dict,
    retriever: nn.Module,
    text_encoder: nn.Module,
    text_tokenizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    optimizer: torch.optim.Optimizer,
    chunk_size: int,
    world_size: int,
    neg_bank,
    neg_bank_rng,
    glossary_neg_embs: Optional[torch.Tensor],
    glossary_neg_term_ids: Optional[torch.Tensor],
    args,
    easy_neg_weight: float = 0.0,
    effective_tcm_pos_weight: Optional[float] = None,
    effective_tcm_neg_weight: Optional[float] = None,
) -> Tuple[torch.Tensor, float]:
    """
    GradCache training step: decouple batch size from GPU memory.

    Phase 1: no_grad forward on sub-batches, collect embeddings.
    Phase 2: compute contrastive loss on full embeddings, cache grad.
    Phase 3: re-forward sub-batches with grad, propagate cached grad.

    ~1/3 more FLOPs (extra forward pass) but memory = chunk_size, not batch_size.
    """
    with _record_function("gc/prep_h2d"):
        # Pinned-memory H2D transfers happen with non_blocking=True so that
        # the copies overlap the Phase 1 retriever forward kicked off below.
        # The DataLoader collate already calls pin_memory() (see L3740+),
        # which is a precondition for non_blocking to be actually async.
        feats, flens = _move_audio_batch_to_device(batch, device, args, non_blocking=True)
        texts = batch["term_texts"]
        valid = batch["valid_mask"].to(device, non_blocking=True)
        positive_term_ids = batch["positive_term_ids"].to(device, non_blocking=True)
        positive_term_mask = batch["positive_term_mask"].to(device, non_blocking=True)
        mfa_starts = batch.get("mfa_term_starts")
        mfa_ends = batch.get("mfa_term_ends")
        if mfa_starts is not None:
            mfa_starts = mfa_starts.to(device, non_blocking=True)
        if mfa_ends is not None:
            mfa_ends = mfa_ends.to(device, non_blocking=True)
        samples = batch["samples"]
        N = feats.size(0)

        raw_retriever = retriever.module if world_size > 1 else retriever
        chunk_ranges = [(i, min(i + chunk_size, N)) for i in range(0, N, chunk_size)]

        # ---- Phase 1: collect embeddings (no model gradients) ----
        raw_text_encoder = text_encoder.module if world_size > 1 else text_encoder
        is_colbert = getattr(raw_text_encoder, "use_colbert", False)

        # For ColBERT, tokenize to fixed max_length so T is uniform across all GPUs
        if is_colbert:
            all_tok = _tokenize_texts(
                text_tokenizer,
                texts,
                device,
                text_input_prefix=args.text_input_prefix,
                padding="max_length",
            )

    with _record_function("gc/phase1_no_grad_fwd"):
        speech_chunks: List[torch.Tensor] = []
        text_chunks: List[torch.Tensor] = []
        with torch.no_grad():
            for start, end in chunk_ranges:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    with _record_function("gc/p1_retriever_fwd"):
                        speech_chunks.append(retriever(feats[start:end], flens[start:end]))
                    with _record_function("gc/p1_text_fwd"):
                        if is_colbert:
                            text_chunks.append(text_encoder(
                                all_tok.input_ids[start:end],
                                all_tok.attention_mask[start:end],
                            ))
                        else:
                            tok = _tokenize_texts(
                                text_tokenizer,
                                texts[start:end],
                                device,
                                text_input_prefix=args.text_input_prefix,
                            )
                            text_chunks.append(text_encoder(tok.input_ids, tok.attention_mask))

        speech_embs = torch.cat(speech_chunks, dim=0).requires_grad_(True)
        text_embs = torch.cat(text_chunks, dim=0).requires_grad_(True)

    # ---- Phase 2: full-batch loss + embedding gradients ----
    _p2_cm = _record_function("gc/phase2_loss_plus_emb_bwd")
    _p2_cm.__enter__()
    logit_scale = raw_retriever.logit_scale.exp()

    group_ids = torch.tensor(
        [stable_group_id(build_speech_group_key(s)) for s in samples],
        dtype=torch.long, device=device,
    )
    term_ids = torch.tensor(
        [stable_term_id((s.get("term_text", "") or "")) for s in samples],
        dtype=torch.long, device=device,
    )

    nb_embs, nb_tids = None, None
    ps_embs, ps_tids = None, None
    hard_neg_count = 0
    if neg_bank is not None and neg_bank.embeddings is not None:
        if getattr(args, "hard_neg_k_per_sample", 0) > 0:
            ps_embs, ps_tids, hard_neg_count = neg_bank.mine_hard_negatives_per_sample(
                speech_embs.detach(),
                term_ids,
                valid,
                positive_term_ids,
                positive_term_mask,
                args.hard_neg_k_per_sample,
            )
            ps_embs = ps_embs.to(torch.bfloat16)
        elif args.neg_bank_size > 0:
            nb_embs, nb_tids = neg_bank.sample(args.neg_bank_size, neg_bank_rng)
        if nb_embs is not None:
            nb_embs = nb_embs.to(torch.bfloat16)

    if glossary_neg_embs is not None:
        gn_embs = glossary_neg_embs.to(device, non_blocking=True).to(torch.bfloat16)
        gn_tids = glossary_neg_term_ids.to(device, non_blocking=True)
        if nb_embs is not None:
            nb_embs = torch.cat([nb_embs, gn_embs], dim=0)
            nb_tids = torch.cat([nb_tids, gn_tids], dim=0)
        else:
            nb_embs = gn_embs
            nb_tids = gn_tids

    optimizer.zero_grad(set_to_none=True)

    loss_win_starts = None
    loss_win_ends = None
    loss_mfa_starts = None
    loss_mfa_ends = None
    if getattr(args, "mfa_supervised_maxsim", False) and mfa_starts is not None:
        W = speech_embs.shape[1] if speech_embs.ndim == 3 else 0
        if W > 0:
            T_enc = _infer_encoder_frames(
                raw_retriever.maxsim_windows, raw_retriever.maxsim_stride, W,
            )
            # Cached on-device build; avoids re-allocating a new CPU tensor
            # from a Python list + H2D copy every step.
            loss_win_starts, loss_win_ends = _get_window_time_ranges_on(
                raw_retriever.maxsim_windows,
                raw_retriever.maxsim_stride,
                T_enc,
                device,
                frame_sec=float(args.fixed_audio_seconds) / float(T_enc),
            )
            assert loss_win_starts.shape[0] == W, (
                f"Window count mismatch: expected {W}, got {loss_win_starts.shape[0]}"
            )
            loss_mfa_starts = mfa_starts
            loss_mfa_ends = mfa_ends

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss_outputs = compute_masked_contrastive_loss(
            speech_embs=speech_embs,
            text_embs=text_embs,
            logit_scale=logit_scale,
            local_group_ids=group_ids,
            local_term_ids=term_ids,
            local_positive_term_ids=positive_term_ids,
            local_positive_term_mask=positive_term_mask,
            local_valid_mask=valid,
            neg_bank_embs=nb_embs,
            neg_bank_term_ids=nb_tids,
            per_sample_neg_embs=ps_embs,
            per_sample_neg_term_ids=ps_tids,
            margin=args.margin,
            online_hard_neg_k=args.online_hard_neg_k,
            easy_neg_weight=easy_neg_weight,
            maxsim_agg=getattr(args, "maxsim_agg", "hard_max"),
            maxsim_softmax_tau=getattr(args, "maxsim_softmax_tau", 1.0),
            mfa_term_starts=loss_mfa_starts,
            mfa_term_ends=loss_mfa_ends,
            win_starts=loss_win_starts,
            win_ends=loss_win_ends,
            mfa_window_selection=getattr(
                args, "mfa_window_selection", DEFAULT_MFA_WINDOW_SELECTION
            ),
            mfa_lse_temperature=getattr(
                args, "mfa_lse_temperature", DEFAULT_MFA_LSE_TEMPERATURE
            ),
            mfa_positive_scope=getattr(
                args, "mfa_positive_scope", DEFAULT_MFA_POSITIVE_SCOPE
            ),
            tcm_loss_weight=getattr(args, "tcm_loss_weight", 0.0),
            tcm_pos_loss_weight=effective_tcm_pos_weight,
            tcm_neg_loss_weight=effective_tcm_neg_weight,
            tcm_pos_threshold=getattr(args, "tcm_pos_threshold", DEFAULT_TCM_POS_THRESHOLD),
            tcm_neg_threshold=getattr(args, "tcm_neg_threshold", DEFAULT_TCM_NEG_THRESHOLD),
            tcm_loss_form=getattr(args, "tcm_loss_form", DEFAULT_TCM_LOSS_FORM),
            tcm_reduction=getattr(args, "tcm_reduction", DEFAULT_TCM_REDUCTION),
            tcm_neg_scope=getattr(args, "tcm_neg_scope", DEFAULT_TCM_NEG_SCOPE),
            tcm_neg_topk=getattr(args, "tcm_neg_topk", 0),
            hcl_beta=getattr(args, "hcl_beta", DEFAULT_HCL_BETA),
        )
        total_loss = loss_outputs["total"]

    with _record_function("gc/phase2_bwd_to_embs"):
        scaler.scale(total_loss).backward()

        speech_grad = speech_embs.grad.detach().clone()
        text_grad = text_embs.grad.detach().clone()
    _p2_cm.__exit__(None, None, None)

    # ---- Phase 3: re-forward sub-batches, propagate cached gradients ----
    _p3_cm = _record_function("gc/phase3_refwd_bwd_to_weights")
    _p3_cm.__enter__()
    retriever_has_no_sync = hasattr(retriever, "no_sync")
    text_encoder_has_no_sync = hasattr(text_encoder, "no_sync")
    for idx, (start, end) in enumerate(chunk_ranges):
        is_last_chunk = idx == len(chunk_ranges) - 1
        speech_ctx = (
            contextlib.nullcontext()
            if is_last_chunk or not retriever_has_no_sync
            else retriever.no_sync()
        )
        text_ctx = (
            contextlib.nullcontext()
            if is_last_chunk or not text_encoder_has_no_sync
            else text_encoder.no_sync()
        )

        with speech_ctx, text_ctx:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                with _record_function("gc/p3_retriever_fwd"):
                    s_emb = retriever(feats[start:end], flens[start:end])
                with _record_function("gc/p3_text_fwd"):
                    if is_colbert:
                        t_emb = text_encoder(
                            all_tok.input_ids[start:end],
                            all_tok.attention_mask[start:end],
                        )
                    else:
                        tok = _tokenize_texts(
                            text_tokenizer,
                            texts[start:end],
                            device,
                            text_input_prefix=args.text_input_prefix,
                        )
                        t_emb = text_encoder(tok.input_ids, tok.attention_mask)

            with _record_function("gc/p3_bwd"):
                s_emb.backward(speech_grad[start:end])
                t_emb.backward(text_grad[start:end])
    _p3_cm.__exit__(None, None, None)

    with _record_function("gc/optimizer_step"):
        scaler.unscale_(optimizer)
        all_trainable = list(retriever.parameters()) + list(text_encoder.parameters())
        torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=DEFAULT_GRAD_CLIP_MAX_NORM)
        scaler.step(optimizer)
        scaler.update()

    return loss_outputs, hard_neg_count


def _pad_and_cat_3d(tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """Concatenate 3D tensors along `dim`, padding the T dimension (dim=1) to match."""
    if not tensors or tensors[0].ndim != 3:
        return torch.cat(tensors, dim=dim)
    max_t = max(t.size(1) for t in tensors)
    padded = []
    for t in tensors:
        if t.size(1) < max_t:
            pad = torch.zeros(
                t.size(0), max_t - t.size(1), t.size(2),
                dtype=t.dtype, device=t.device,
            )
            padded.append(torch.cat([t, pad], dim=1))
        else:
            padded.append(t)
    return torch.cat(padded, dim=dim)


MAXSIM_AGG_MODES = {"hard_max", "softmax"}
MAXSIM_SOFTMAX_TAU_INIT = 1.0
MAXSIM_SOFTMAX_TAU_MIN = 0.05

# MFA window selection modes (A2 ablation axis):
#   - "hard_max":  argmax-by-similarity among windows fully covering the term (baseline).
#   - "smallest":  deterministic pick: the shortest-duration window covering the term
#                  (tight crop, eliminates context shortcut, only that window gets grad).
#   - "logsumexp": all covering windows participate via LSE aggregation with temperature
#                  MFA_LSE_TAU (gradient distributed softmax-weighted over covering windows).
MFA_WINDOW_SELECTION_MODES = {"hard_max", "smallest", "logsumexp"}
DEFAULT_MFA_WINDOW_SELECTION = "hard_max"
DEFAULT_MFA_LSE_TEMPERATURE = 1.0
MFA_POSITIVE_SCOPE_MODES = {"auto", "chunk", "term"}
DEFAULT_MFA_POSITIVE_SCOPE = "auto"


def _maxsim_score(
    speech_embs: torch.Tensor, text_embs: torch.Tensor,
    agg_mode: str = "hard_max", softmax_tau: float = 1.0,
    text_chunk_size: int = 1024,
) -> torch.Tensor:
    """Compute Max-Sim between multi-scale speech and (optionally multi-vector) text.

    Args:
        speech_embs: [B, W, D] multi-scale window embeddings (L2-normalized)
        text_embs: [N, D] single-vector text OR [N, T, D] ColBERT multi-vector
        agg_mode: "hard_max" (standard) or "softmax" (weighted sum over windows)
        softmax_tau: temperature for softmax aggregation (lower = sharper, approaches hard_max)

    Returns:
        [B, N] similarity matrix
    """
    if text_embs.ndim == 2:
        # Standard: audio multi-window vs text single-vector. Chunk over the
        # text bank so large glossary evals do not materialize [B, W, N].
        TEXT_CHUNK = max(1, int(text_chunk_size or 1024))
        result_chunks = []
        for j in range(0, text_embs.size(0), TEXT_CHUNK):
            text_chunk = text_embs[j : j + TEXT_CHUNK]
            # [B, W, D] @ [D, chunk] -> [B, W, chunk] -> [B, chunk]
            sim_all = torch.matmul(speech_embs, text_chunk.T)
            if agg_mode == "hard_max":
                chunk_score = sim_all.max(dim=1).values
            else:
                assert agg_mode == "softmax", f"Unknown agg_mode: {agg_mode}"
                weights = F.softmax(sim_all / softmax_tau, dim=1)
                chunk_score = (weights * sim_all).sum(dim=1)
            result_chunks.append(chunk_score)
        return torch.cat(result_chunks, dim=1)

    assert text_embs.ndim == 3, f"Expected text_embs to be 2D or 3D, got {text_embs.ndim}D"
    # ColBERT late interaction: audio [B, W, D] vs text [N, T, D]
    # For each (b, n): for each text token t, find best audio window w, then avg over t
    # Memory-efficient: process text in chunks to avoid [B, N, W, T] materialization
    B, W, D = speech_embs.shape
    N, T, _ = text_embs.shape

    token_norms = text_embs.norm(dim=-1)  # [N, T]
    active_mask = (token_norms > 1e-6).float()  # [N, T]
    active_count = active_mask.sum(dim=-1).clamp(min=1.0)  # [N]

    COLBERT_CHUNK = 256
    result_chunks = []
    for j in range(0, N, COLBERT_CHUNK):
        j_end = min(j + COLBERT_CHUNK, N)
        # [B, W, D] @ [chunk, T, D]^T -> [B, W, chunk, T]
        chunk_sim = torch.einsum("bwd,ntd->bnwt", speech_embs, text_embs[j:j_end])
        # max over W -> [B, chunk, T]
        chunk_max_w = chunk_sim.max(dim=2).values
        # masked mean over T -> [B, chunk]
        chunk_masked = chunk_max_w * active_mask[j:j_end].unsqueeze(0)
        chunk_score = chunk_masked.sum(dim=-1) / active_count[j:j_end].unsqueeze(0)
        result_chunks.append(chunk_score)
    return torch.cat(result_chunks, dim=1)  # [B, N]


def _resolve_eval_score_device(
    args: argparse.Namespace,
    train_device: torch.device,
) -> torch.device:
    requested = str(
        getattr(args, "eval_score_device", DEFAULT_EVAL_SCORE_DEVICE) or "cuda"
    ).strip().lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda":
        if torch.cuda.is_available():
            return train_device if train_device.type == "cuda" else torch.device("cuda")
        logger.warning("[EVAL_SCORE] requested cuda but CUDA is unavailable; falling back to CPU")
    return torch.device("cpu")


def _score_eval_logits(
    speech_embs: torch.Tensor,
    text_embs: torch.Tensor,
    args: argparse.Namespace,
    train_device: torch.device,
    *,
    score_device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Score eval logits with query/text chunking; output stays on CPU."""
    score_device = score_device or _resolve_eval_score_device(args, train_device)
    query_chunk = max(
        1,
        int(getattr(args, "eval_score_query_chunk", DEFAULT_EVAL_SCORE_QUERY_CHUNK) or 1),
    )
    text_chunk = max(
        1,
        int(getattr(args, "eval_score_text_chunk", DEFAULT_EVAL_SCORE_TEXT_CHUNK) or 1),
    )

    if score_device.type != "cuda":
        speech_cpu = speech_embs.cpu()
        text_cpu = text_embs.cpu()
        if speech_cpu.ndim == 3:
            return _maxsim_score(speech_cpu, text_cpu)
        if text_cpu.ndim == 3:
            return _maxsim_score(speech_cpu.unsqueeze(1), text_cpu)
        return speech_cpu @ text_cpu.T

    out_rows: List[torch.Tensor] = []
    with torch.no_grad():
        for q0 in range(0, speech_embs.size(0), query_chunk):
            q = speech_embs[q0 : q0 + query_chunk].to(
                score_device, non_blocking=True
            )
            row_chunks: List[torch.Tensor] = []
            for t0 in range(0, text_embs.size(0), text_chunk):
                t = text_embs[t0 : t0 + text_chunk].to(
                    score_device, non_blocking=True
                )
                if q.ndim == 3:
                    scores = _maxsim_score(q, t, text_chunk_size=text_chunk)
                elif t.ndim == 3:
                    scores = _maxsim_score(
                        q.unsqueeze(1), t, text_chunk_size=text_chunk
                    )
                else:
                    scores = q @ t.T
                row_chunks.append(scores.float().cpu())
                del t, scores
            out_rows.append(torch.cat(row_chunks, dim=1))
            del q, row_chunks
    return torch.cat(out_rows, dim=0)


def _count_windows(maxsim_windows: List[int], maxsim_stride: int, T: int) -> int:
    """Count total windows produced by _multiscale_pool for given T."""
    total = 0
    for w in maxsim_windows:
        if w >= T:
            total += 1
        else:
            total += (T - w) // maxsim_stride + 1
    return total


def _infer_encoder_frames(
    maxsim_windows: List[int], maxsim_stride: int, W_total: int,
) -> int:
    """Given total window count W_total, infer the encoder output frame count T."""
    for T in range(1, 500):
        if _count_windows(maxsim_windows, maxsim_stride, T) == W_total:
            return T
    raise ValueError(
        f"Cannot infer T for W_total={W_total} with windows={maxsim_windows} "
        f"stride={maxsim_stride}"
    )


def _build_window_time_ranges(
    maxsim_windows: List[int],
    maxsim_stride: int,
    T: int,
    frame_sec: float = FRAME_SEC,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build [W_total] tensors of (start_sec, end_sec) for each window position.

    Must match the concatenation order in _multiscale_pool exactly.

    Returns: (win_starts [W], win_ends [W]) in seconds relative to chunk start.
    """
    starts = []
    ends = []
    for w in maxsim_windows:
        if w >= T:
            starts.append(0.0)
            ends.append(T * frame_sec)
        else:
            n_out = (T - w) // maxsim_stride + 1
            for p in range(n_out):
                frame_start = p * maxsim_stride
                frame_end = frame_start + w
                starts.append(frame_start * frame_sec)
                ends.append(frame_end * frame_sec)
    return (
        torch.tensor(starts, dtype=torch.float32),
        torch.tensor(ends, dtype=torch.float32),
    )


# Cache of _build_window_time_ranges results on device so the per-step
# `torch.tensor([...]).to(device)` pattern collapses to a cached GPU tensor
# reference.  Keyed on `(tuple(maxsim_windows), maxsim_stride, T, device_index)`.
_WINDOW_RANGES_CACHE: Dict[
    Tuple[Tuple[int, ...], int, int, float, Optional[int]],
    Tuple[torch.Tensor, torch.Tensor],
] = {}


def _get_window_time_ranges_on(
    maxsim_windows: List[int],
    maxsim_stride: int,
    T: int,
    device: torch.device,
    frame_sec: float = FRAME_SEC,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cached, on-device version of `_build_window_time_ranges`.

    First call for a given (windows, stride, T, device) tuple builds the CPU
    tensors and uploads them once; subsequent calls return the cached GPU
    tensors directly, skipping the per-step `torch.tensor([...]).to(device)`
    hot path identified in `perf_notes.md`.
    """
    frame_sec_key = round(float(frame_sec), 9)
    key = (
        tuple(maxsim_windows),
        int(maxsim_stride),
        int(T),
        frame_sec_key,
        device.index,
    )
    hit = _WINDOW_RANGES_CACHE.get(key)
    if hit is not None:
        return hit
    ws_cpu, we_cpu = _build_window_time_ranges(
        maxsim_windows,
        maxsim_stride,
        T,
        frame_sec=frame_sec_key,
    )
    ws = ws_cpu.to(device, non_blocking=True)
    we = we_cpu.to(device, non_blocking=True)
    _WINDOW_RANGES_CACHE[key] = (ws, we)
    return ws, we


def _maxsim_score_mfa(
    speech_embs: torch.Tensor,
    text_embs: torch.Tensor,
    mfa_term_starts: Optional[torch.Tensor],
    mfa_term_ends: Optional[torch.Tensor],
    win_starts: Optional[torch.Tensor],
    win_ends: Optional[torch.Tensor],
    selection_mode: str = DEFAULT_MFA_WINDOW_SELECTION,
    lse_temperature: float = DEFAULT_MFA_LSE_TEMPERATURE,
    return_debug: bool = False,
) -> torch.Tensor:
    """MFA-supervised MaxSim: only windows that fully cover the term participate.

    A window covers the term iff: win_start <= term_start AND win_end >= term_end.
    Samples with term_start < 0 (no MFA data) use all windows (standard hard_max).

    Selection modes (A2 ablation):
        hard_max:  argmax by similarity among covering windows (single window's grad).
        smallest:  pick the shortest-duration covering window (argmin by duration,
                   single deterministic window regardless of similarity).
        logsumexp: aggregate all covering windows via LSE with temperature tau:
                   sim[b,n] = tau * logsumexp(sim_all[b,w,n] / tau, dim=w over covering).
                   Gradient is distributed softmax-weighted across covering windows.

    Fallback: if NO window fully covers the term (e.g. term longer than the
    longest window or alignment crosses chunk boundaries), fall back to standard
    MaxSim over all windows instead of forcing a deterministic low-signal window.

    Args:
        speech_embs: [B, W, D]
        text_embs:   [N, D]
        mfa_term_starts: [B] term start in chunk (seconds); <0 means no MFA
        mfa_term_ends:   [B] term end in chunk (seconds); <0 means no MFA
        win_starts: [W] window start times (seconds)
        win_ends:   [W] window end times (seconds)
        selection_mode: one of MFA_WINDOW_SELECTION_MODES.
        lse_temperature: LSE temperature (only used when selection_mode="logsumexp").
        return_debug: if True, also return a dict with selection-mode-specific
            diagnostics (covering counts, selected window idx / softmax weights).

    Returns:
        [B, N] similarity matrix, or (sim, debug_dict) if return_debug=True.
    """
    assert text_embs.ndim == 2, "MFA-supervised MaxSim only supports single-vector text"
    assert selection_mode in MFA_WINDOW_SELECTION_MODES, (
        f"Invalid selection_mode={selection_mode}; expected one of "
        f"{sorted(MFA_WINDOW_SELECTION_MODES)}"
    )
    B, W, D = speech_embs.shape
    no_mfa_inputs = (
        mfa_term_starts is None
        or mfa_term_ends is None
        or win_starts is None
        or win_ends is None
        or (mfa_term_starts < 0).all()
    )
    if no_mfa_inputs:
        sim_all = torch.matmul(speech_embs, text_embs.T)  # [B, W, N]
        sim_out = sim_all.max(dim=1).values
        if return_debug:
            return sim_out, {"mode": "no_mfa_fallback_global_max"}
        return sim_out

    device = speech_embs.device
    # non_blocking=True is a no-op when tensors already live on `device`
    # (common in the GradCache path), and lets the H2D copy overlap when
    # they don't (fallback non-cache path).
    ws = win_starts.to(device, non_blocking=True)    # [W]
    we = win_ends.to(device, non_blocking=True)      # [W]
    ts = mfa_term_starts.to(device, non_blocking=True)  # [B]
    te = mfa_term_ends.to(device, non_blocking=True)    # [B]

    has_mfa = ts >= 0  # [B]

    # window_ok[b, w] = (ws[w] <= ts[b]) AND (we[w] >= te[b])
    covers_start = ws.unsqueeze(0) <= ts.unsqueeze(1)  # [B, W]
    covers_end = we.unsqueeze(0) >= te.unsqueeze(1)      # [B, W]
    window_ok = covers_start & covers_end                 # [B, W]
    # Samples without MFA data: fall back to "all windows" (standard hard_max).
    window_ok = window_ok | (~has_mfa).unsqueeze(1)

    # Rows with valid MFA but no covering window fall back to standard MaxSim.
    # Keep `window_ok` as the true covering mask; mode-specific branches mix in
    # the all-window max only for fallback rows.
    any_valid = window_ok.any(dim=1)              # [B]
    needs_fallback = has_mfa & (~any_valid)       # [B]

    if selection_mode == "smallest":
        win_dur = (we - ws).to(device)  # [W]
        win_dur_b = win_dur.unsqueeze(0).expand(B, -1).clone()  # [B, W]
        win_dur_b = win_dur_b.masked_fill(~window_ok, float("inf"))
        min_idx = win_dur_b.argmin(dim=1)  # [B]
        batch_idx = torch.arange(B, device=device)
        selected_speech = speech_embs[batch_idx, min_idx]  # [B, D]
        sim_out = torch.matmul(selected_speech, text_embs.T)  # [B, N]
        if needs_fallback.any():
            fallback_idx = needs_fallback.nonzero(as_tuple=False).flatten()
            fallback_sim = torch.matmul(
                speech_embs[fallback_idx], text_embs.T
            ).max(dim=1).values
            sim_out = sim_out.clone()
            sim_out[fallback_idx] = fallback_sim
        if return_debug:
            selected_idx = min_idx.masked_fill(needs_fallback, -1)
            debug = {
                "mode": "smallest",
                "covering_counts": window_ok.sum(dim=1).detach().cpu(),
                "selected_win_idx": selected_idx.detach().cpu(),
                "selected_win_dur": win_dur[min_idx].masked_fill(
                    needs_fallback, float("nan")
                ).detach().cpu(),
                "fallback_rows": needs_fallback.detach().cpu(),
            }
            return sim_out, debug
        return sim_out

    sim_all = torch.matmul(speech_embs, text_embs.T)  # [B, W, N]

    if selection_mode == "hard_max":
        max_window_ok = window_ok | needs_fallback.unsqueeze(1)
        window_mask = max_window_ok.unsqueeze(2).expand_as(sim_all)  # [B, W, N]
        sim_masked = sim_all.masked_fill(~window_mask, -1e9)
        sim_out = sim_masked.max(dim=1).values  # [B, N]
        if return_debug:
            # For each row, record how many windows were valid.
            debug = {
                "mode": "hard_max",
                "covering_counts": window_ok.sum(dim=1).detach().cpu(),
                "fallback_rows": needs_fallback.detach().cpu(),
            }
            return sim_out, debug
        return sim_out

    # selection_mode == "logsumexp"
    assert lse_temperature > 0.0, (
        f"MFA logsumexp requires lse_temperature > 0, got {lse_temperature}"
    )
    window_mask = window_ok.unsqueeze(2).expand_as(sim_all)  # [B, W, N]
    sim_scaled = (sim_all / lse_temperature).masked_fill(~window_mask, float("-inf"))
    sim_out = lse_temperature * torch.logsumexp(sim_scaled, dim=1)  # [B, N]
    if needs_fallback.any():
        fallback_sim = sim_all.max(dim=1).values
        sim_out = torch.where(needs_fallback.unsqueeze(1), fallback_sim, sim_out)
    if return_debug:
        # Softmax weights (detached) over covering windows for first term column.
        with torch.no_grad():
            first_col = sim_scaled[:, :, 0]
            weights = torch.softmax(first_col, dim=1)
        debug = {
            "mode": "logsumexp",
            "covering_counts": window_ok.sum(dim=1).detach().cpu(),
            "softmax_weights_col0": weights.detach().cpu(),
            "lse_temperature": lse_temperature,
            "fallback_rows": needs_fallback.detach().cpu(),
        }
        return sim_out, debug
    return sim_out


def _maxsim_score_per_sample(
    speech_embs: torch.Tensor,
    hn_embs: torch.Tensor,
    agg_mode: str = "hard_max",
    softmax_tau: float = 1.0,
) -> torch.Tensor:
    """Per-sample Max-Sim for independent hard negatives (no cross-row sharing).

    Args:
        speech_embs: [B, W, D] multi-scale speech windows (L2-normalized)
        hn_embs:     [B, K, D] per-row hard-negative text embeddings
        agg_mode:    "hard_max" or "softmax"
        softmax_tau: softmax temperature (used only for agg_mode="softmax")

    Returns:
        [B, K] similarity matrix, row i scored against its own K negatives only.
    """
    assert speech_embs.ndim == 3 and hn_embs.ndim == 3, (
        f"_maxsim_score_per_sample expects 3D tensors, "
        f"got speech={speech_embs.shape} hn={hn_embs.shape}"
    )
    assert speech_embs.shape[0] == hn_embs.shape[0], (
        f"Batch mismatch: speech B={speech_embs.shape[0]} vs hn B={hn_embs.shape[0]}"
    )
    sim_all = torch.einsum("bwd,bkd->bwk", speech_embs, hn_embs)  # [B, W, K]
    if agg_mode == "hard_max":
        return sim_all.max(dim=1).values  # [B, K]
    assert agg_mode == "softmax", f"Unknown agg_mode: {agg_mode}"
    weights = F.softmax(sim_all / softmax_tau, dim=1)  # [B, W, K]
    return (weights * sim_all).sum(dim=1)  # [B, K]


def _maxsim_score_mfa_per_sample(
    speech_embs: torch.Tensor,
    hn_embs: torch.Tensor,
    mfa_term_starts: Optional[torch.Tensor],
    mfa_term_ends: Optional[torch.Tensor],
    win_starts: Optional[torch.Tensor],
    win_ends: Optional[torch.Tensor],
    selection_mode: str = DEFAULT_MFA_WINDOW_SELECTION,
    lse_temperature: float = DEFAULT_MFA_LSE_TEMPERATURE,
) -> torch.Tensor:
    """Per-sample MFA-supervised MaxSim for hard negatives.

    Mirrors `_maxsim_score_mfa` but the text side is per-row [B, K, D] instead
    of shared [N, D].  Similarity is computed via einsum ``bwd,bkd->bwk`` and
    covering-window selection is applied row-wise using ``window_ok [B, W]``.

    Args:
        speech_embs: [B, W, D]
        hn_embs:     [B, K, D]
        mfa_term_starts: [B] term start in chunk (seconds); <0 means no MFA
        mfa_term_ends:   [B] term end in chunk (seconds); <0 means no MFA
        win_starts:  [W] window start times (seconds)
        win_ends:    [W] window end times (seconds)
        selection_mode: one of MFA_WINDOW_SELECTION_MODES.
        lse_temperature: LSE temperature (logsumexp mode only).

    Returns:
        [B, K] similarity matrix.
    """
    assert speech_embs.ndim == 3 and hn_embs.ndim == 3, (
        f"_maxsim_score_mfa_per_sample expects 3D tensors, "
        f"got speech={speech_embs.shape} hn={hn_embs.shape}"
    )
    assert speech_embs.shape[0] == hn_embs.shape[0], (
        f"Batch mismatch: speech B={speech_embs.shape[0]} vs hn B={hn_embs.shape[0]}"
    )
    assert selection_mode in MFA_WINDOW_SELECTION_MODES, (
        f"Invalid selection_mode={selection_mode}; expected one of "
        f"{sorted(MFA_WINDOW_SELECTION_MODES)}"
    )
    B, W, D = speech_embs.shape
    _, K, _ = hn_embs.shape

    no_mfa_inputs = (
        mfa_term_starts is None
        or mfa_term_ends is None
        or win_starts is None
        or win_ends is None
        or (mfa_term_starts < 0).all()
    )
    if no_mfa_inputs:
        sim_all = torch.einsum("bwd,bkd->bwk", speech_embs, hn_embs)  # [B, W, K]
        return sim_all.max(dim=1).values  # [B, K]

    device = speech_embs.device
    ws = win_starts.to(device, non_blocking=True)    # [W]
    we = win_ends.to(device, non_blocking=True)      # [W]
    ts = mfa_term_starts.to(device, non_blocking=True)  # [B]
    te = mfa_term_ends.to(device, non_blocking=True)    # [B]

    has_mfa = ts >= 0  # [B]
    covers_start = ws.unsqueeze(0) <= ts.unsqueeze(1)  # [B, W]
    covers_end = we.unsqueeze(0) >= te.unsqueeze(1)      # [B, W]
    window_ok = covers_start & covers_end                 # [B, W]
    window_ok = window_ok | (~has_mfa).unsqueeze(1)

    any_valid = window_ok.any(dim=1)              # [B]
    needs_fallback = has_mfa & (~any_valid)       # [B]

    if selection_mode == "smallest":
        win_dur = (we - ws).to(device)  # [W]
        win_dur_b = win_dur.unsqueeze(0).expand(B, -1).clone()  # [B, W]
        win_dur_b = win_dur_b.masked_fill(~window_ok, float("inf"))
        min_idx = win_dur_b.argmin(dim=1)  # [B]
        batch_idx = torch.arange(B, device=device)
        selected_speech = speech_embs[batch_idx, min_idx]  # [B, D]
        sim_out = torch.einsum("bd,bkd->bk", selected_speech, hn_embs)  # [B, K]
        if needs_fallback.any():
            fallback_idx = needs_fallback.nonzero(as_tuple=False).flatten()
            fallback_sim = torch.einsum(
                "bwd,bkd->bwk",
                speech_embs[fallback_idx],
                hn_embs[fallback_idx],
            ).max(dim=1).values
            sim_out = sim_out.clone()
            sim_out[fallback_idx] = fallback_sim
        return sim_out

    sim_all = torch.einsum("bwd,bkd->bwk", speech_embs, hn_embs)  # [B, W, K]

    if selection_mode == "hard_max":
        # [B, W, 1] broadcast across K.
        max_window_ok = window_ok | needs_fallback.unsqueeze(1)
        window_mask = max_window_ok.unsqueeze(2).expand_as(sim_all)  # [B, W, K]
        sim_masked = sim_all.masked_fill(~window_mask, -1e9)
        return sim_masked.max(dim=1).values  # [B, K]

    # selection_mode == "logsumexp"
    assert lse_temperature > 0.0, (
        f"MFA logsumexp requires lse_temperature > 0, got {lse_temperature}"
    )
    window_mask = window_ok.unsqueeze(2).expand_as(sim_all)  # [B, W, K]
    sim_scaled = (sim_all / lse_temperature).masked_fill(~window_mask, float("-inf"))
    sim_out = lse_temperature * torch.logsumexp(sim_scaled, dim=1)  # [B, K]
    if needs_fallback.any():
        fallback_sim = sim_all.max(dim=1).values
        sim_out = torch.where(needs_fallback.unsqueeze(1), fallback_sim, sim_out)
    return sim_out


def compute_masked_contrastive_loss(
    speech_embs: torch.Tensor,
    text_embs: torch.Tensor,
    logit_scale: torch.Tensor,
    local_group_ids: torch.Tensor,
    local_term_ids: torch.Tensor,
    local_positive_term_ids: torch.Tensor,
    local_positive_term_mask: torch.Tensor,
    local_valid_mask: torch.Tensor,
    neg_bank_embs: Optional[torch.Tensor] = None,
    neg_bank_term_ids: Optional[torch.Tensor] = None,
    per_sample_neg_embs: Optional[torch.Tensor] = None,
    per_sample_neg_term_ids: Optional[torch.Tensor] = None,
    margin: float = 0.0,
    online_hard_neg_k: int = 0,
    easy_neg_weight: float = 0.0,
    maxsim_agg: str = "hard_max",
    maxsim_softmax_tau: float = 1.0,
    mfa_term_starts: Optional[torch.Tensor] = None,
    mfa_term_ends: Optional[torch.Tensor] = None,
    win_starts: Optional[torch.Tensor] = None,
    win_ends: Optional[torch.Tensor] = None,
    mfa_window_selection: str = DEFAULT_MFA_WINDOW_SELECTION,
    mfa_lse_temperature: float = DEFAULT_MFA_LSE_TEMPERATURE,
    mfa_positive_scope: str = DEFAULT_MFA_POSITIVE_SCOPE,
    tcm_loss_weight: float = 0.0,
    tcm_pos_loss_weight: Optional[float] = None,
    tcm_neg_loss_weight: Optional[float] = None,
    tcm_pos_threshold: float = DEFAULT_TCM_POS_THRESHOLD,
    tcm_neg_threshold: float = DEFAULT_TCM_NEG_THRESHOLD,
    tcm_loss_form: str = DEFAULT_TCM_LOSS_FORM,
    tcm_reduction: str = DEFAULT_TCM_REDUCTION,
    tcm_neg_scope: str = DEFAULT_TCM_NEG_SCOPE,
    tcm_neg_topk: int = 0,
    hcl_beta: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    Masked multi-positive InfoNCE with optional CosFace margin and global negative bank.

    Supports both standard [B, D] and multi-scale Max-Sim [B, W, D] speech embeddings.

    Positive:  same group_id (same chunk) AND both valid.
    Masked:    same term_id but different group_id (false negative) → excluded from denom.
    Negative:  everything else (different chunk, different term).
    Bank:      appended as extra columns; false-neg-masked if term_id matches any anchor.

    With MFA-supervised MaxSim, the default positive scope is term-level:
    other terms from the same chunk are masked as neutral because this row's
    MFA span belongs to the anchor term, not every co-occurring term.

    When margin > 0, positive logits are penalized by margin * logit_scale (CosFace):
        exp((sim_pos - m) / T)  /  (exp((sim_pos - m) / T) + sum exp(sim_neg / T))

    When online_hard_neg_k > 0, only the top-K hardest negatives are fully weighted.
    Remaining easy negatives are scaled by easy_neg_weight in the softmax denominator:
        easy_neg_weight=0 → hard mask (original behavior)
        easy_neg_weight in (0,1) → soft weighting via logit += log(weight)
        easy_neg_weight>=1 → no O-HNM effect

    When tcm_loss_weight > 0, an absolute-threshold auxiliary loss (TCM,
    Zhang et al., ICLR 2024) is added: positives are pushed above
    tcm_pos_threshold and negatives are pushed below tcm_neg_threshold,
    calibrating the retriever's operating point across domains.

    When hcl_beta > 0, negatives are importance-sample reweighted per
    Robinson et al. "Contrastive Learning with Hard Negative Samples"
    (ICLR 2021, eq 3-4): each negative j receives an additive log-weight
    log(w_j) = beta * s_j - logsumexp(beta * s) + log(N_neg), yielding
    a soft "taunt" on hard negatives without discarding easy ones.
    Mutually exclusive with online_hard_neg_k.

    Returns a dict with the following tensors (scalars):
        total, infonce, tcm_pos, tcm_neg,
        tcm_pos_viol_rate, tcm_neg_viol_rate,
        pos_sim_mean, neg_sim_mean,
        hcl_neg_sim_weighted_mean, hcl_log_weight_max.
    """
    assert tcm_loss_form in TCM_LOSS_FORMS, (
        f"Unknown tcm_loss_form={tcm_loss_form}, expected one of {TCM_LOSS_FORMS}"
    )
    assert tcm_reduction in TCM_REDUCTIONS, (
        f"Unknown tcm_reduction={tcm_reduction}, expected one of {TCM_REDUCTIONS}"
    )
    assert tcm_neg_scope in TCM_NEG_SCOPES, (
        f"Unknown tcm_neg_scope={tcm_neg_scope}, expected one of {TCM_NEG_SCOPES}"
    )
    assert mfa_positive_scope in MFA_POSITIVE_SCOPE_MODES, (
        f"Unknown mfa_positive_scope={mfa_positive_scope}, "
        f"expected one of {MFA_POSITIVE_SCOPE_MODES}"
    )
    world_size = (
        dist.get_world_size()
        if dist.is_available() and dist.is_initialized()
        else 1
    )
    device = speech_embs.device
    is_maxsim = speech_embs.ndim == 3
    use_mfa = False

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
        global_text_embs = _pad_and_cat_3d(
            [global_text_embs, neg_bank_embs.detach().to(device, non_blocking=True)], dim=0
        )
        global_group_ids = torch.cat(
            [global_group_ids, torch.zeros(k, dtype=torch.long, device=device)]
        )
        global_term_ids = torch.cat(
            [global_term_ids, neg_bank_term_ids.to(device, non_blocking=True)]
        )
        global_valid_mask = torch.cat(
            [global_valid_mask, torch.ones(k, dtype=torch.bool, device=device)]
        )

    # 3. Similarity matrix  [B_local, N_global]
    if is_maxsim:
        use_mfa = (
            mfa_term_starts is not None
            and mfa_term_ends is not None
            and win_starts is not None
            and win_ends is not None
            and global_text_embs.ndim == 2
        )
        if use_mfa:
            raw_sim = _maxsim_score_mfa(
                speech_embs, global_text_embs,
                mfa_term_starts, mfa_term_ends,
                win_starts, win_ends,
                selection_mode=mfa_window_selection,
                lse_temperature=mfa_lse_temperature,
            )
        else:
            raw_sim = _maxsim_score(
                speech_embs, global_text_embs,
                agg_mode=maxsim_agg, softmax_tau=maxsim_softmax_tau,
            )
    else:
        raw_sim = speech_embs @ global_text_embs.T
    logits = raw_sim * logit_scale

    local_positive_term_ids = local_positive_term_ids.to(device, non_blocking=True)
    local_positive_term_mask = local_positive_term_mask.to(device, non_blocking=True)
    effective_positive_scope = (
        "term"
        if mfa_positive_scope == "auto" and use_mfa
        else "chunk"
        if mfa_positive_scope == "auto"
        else mfa_positive_scope
    )

    # 4. Positive mask.  Under MFA each row's speech score is restricted to
    # the row term's time span, so same-chunk different-term columns are neutral
    # rather than positives trained on the wrong window.
    same_chunk = (
        (local_group_ids.unsqueeze(1) == global_group_ids.unsqueeze(0))
        & local_group_ids.unsqueeze(1).ne(INVALID_ID_SENTINEL)
        & global_group_ids.unsqueeze(0).ne(INVALID_ID_SENTINEL)
    )
    same_anchor_term = local_term_ids.unsqueeze(1) == global_term_ids.unsqueeze(0)
    row_positive_term_match = _term_ids_match_any(
        global_term_ids,
        local_positive_term_ids,
        local_positive_term_mask,
    )
    if effective_positive_scope == "term":
        pos_mask = same_chunk & same_anchor_term
    else:
        pos_mask = same_chunk
    pos_mask = pos_mask & local_valid_mask.unsqueeze(1) & global_valid_mask.unsqueeze(0)

    # 4.5 CosFace margin: subtract m/T from positive logits
    if margin > 0:
        logits = logits - pos_mask.float() * (margin * logit_scale)

    # 5. False-negative / neutral mask: any term known to be valid for this
    # row's speech chunk is removed from the denominator unless it is a scoped
    # positive.  This covers co-chunk GT terms, bank/glossary duplicates, and
    # same-term occurrences from other chunks.
    fn_mask = row_positive_term_match & ~pos_mask

    # 5.5 Valid-negative mask, shared by HCL reweighting and TCM below.
    neg_mask = (~pos_mask) & (~fn_mask) & global_valid_mask.unsqueeze(0)
    cochunk_neutral_count = (
        (same_chunk & row_positive_term_match & ~pos_mask)
        & local_valid_mask.unsqueeze(1)
        & global_valid_mask.unsqueeze(0)
    ).float().sum()
    positive_term_mask_count = (
        row_positive_term_match
        & ~pos_mask
        & global_valid_mask.unsqueeze(0)
    ).float().sum()
    hn_false_positive_masked_count = raw_sim.new_zeros(())

    # 5.75 Per-sample hard negatives: concatenate each row's own K private
    # negative columns (no cross-row sharing).  Expands raw_sim / logits /
    # {pos,fn,neg}_mask on dim=1 by K, and promotes the 1D global_valid_mask
    # to a per-row 2D valid_row_mask used by step 6 below.
    has_per_sample_hn = (
        per_sample_neg_embs is not None and per_sample_neg_term_ids is not None
    )
    if has_per_sample_hn:
        B_local, K_ps, _D = per_sample_neg_embs.shape
        assert per_sample_neg_term_ids.shape == (B_local, K_ps), (
            f"per_sample_neg_term_ids shape mismatch: "
            f"expected ({B_local}, {K_ps}), got {tuple(per_sample_neg_term_ids.shape)}"
        )
        hn_embs_cast = per_sample_neg_embs.detach().to(device=device, dtype=speech_embs.dtype)
        if is_maxsim:
            if use_mfa:
                hn_sim = _maxsim_score_mfa_per_sample(
                    speech_embs, hn_embs_cast,
                    mfa_term_starts, mfa_term_ends,
                    win_starts, win_ends,
                    selection_mode=mfa_window_selection,
                    lse_temperature=mfa_lse_temperature,
                )
            else:
                hn_sim = _maxsim_score_per_sample(
                    speech_embs, hn_embs_cast,
                    agg_mode=maxsim_agg, softmax_tau=maxsim_softmax_tau,
                )
        else:
            hn_sim = torch.einsum("bd,bkd->bk", speech_embs, hn_embs_cast)
        hn_logits = hn_sim * logit_scale

        raw_sim = torch.cat([raw_sim, hn_sim], dim=1)
        logits = torch.cat([logits, hn_logits], dim=1)

        # HN columns: pos=False always; fn=True for any term known to be valid
        # for this speech chunk (kept defensive; the miner should already
        # exclude these terms).
        hn_term_ids_local = per_sample_neg_term_ids.to(device, non_blocking=True)
        fn_hn = _term_ids_match_any(
            hn_term_ids_local,
            local_positive_term_ids,
            local_positive_term_mask,
        )
        pos_hn = torch.zeros_like(fn_hn)
        neg_hn = (~fn_hn)  # always valid; not a false-negative -> a real negative
        hn_false_positive_masked_count = fn_hn.float().sum()
        pos_mask = torch.cat([pos_mask, pos_hn], dim=1)
        fn_mask = torch.cat([fn_mask, fn_hn], dim=1)
        neg_mask = torch.cat([neg_mask, neg_hn], dim=1)

        valid_row_mask = torch.cat(
            [
                global_valid_mask.unsqueeze(0).expand(B_local, -1),
                torch.ones(B_local, K_ps, dtype=torch.bool, device=device),
            ],
            dim=1,
        )
    else:
        valid_row_mask = global_valid_mask.unsqueeze(0).expand_as(logits)

    # 6. Apply masks: invalid columns and false negatives → -inf
    logits = logits.masked_fill(~valid_row_mask, -1e9)
    logits = logits.masked_fill(fn_mask, -1e9)

    # 6.3 HCL (Robinson et al., ICLR 2021) importance-sample reweighting of
    # negatives.  Adds log(w_j) as an additive shift on each negative's logit,
    # where w_j ∝ exp(beta * s_j) and weights are normalized so that their
    # mean across the row equals 1 (so β=0 collapses to standard InfoNCE).
    # Weights are detached: the reweighting is a sampling mechanism, not a
    # learning target.  Mutually exclusive with online_hard_neg_k (HCL is the
    # soft version; pick one).
    hcl_neg_sim_weighted_mean = raw_sim.new_zeros(())
    hcl_log_weight_max = raw_sim.new_zeros(())
    if hcl_beta > 0.0:
        assert online_hard_neg_k == 0, (
            "hcl_beta > 0 is incompatible with online_hard_neg_k > 0 "
            "(HCL is the soft version of online hard negative mining)."
        )
        beta_s = hcl_beta * raw_sim
        beta_s_masked = beta_s.masked_fill(~neg_mask, float("-inf"))
        lse_beta = torch.logsumexp(beta_s_masked, dim=1, keepdim=True)
        n_neg_per_row = neg_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
        log_w = beta_s - lse_beta + torch.log(n_neg_per_row)
        log_w = (log_w * neg_mask.float()).detach()
        logits = logits + log_w
        with torch.no_grad():
            w = torch.exp(log_w) * neg_mask.float()
            w_sum_per_row = w.sum(dim=1, keepdim=True).clamp(min=1.0)
            hcl_neg_sim_weighted_mean = (
                (w * raw_sim * neg_mask.float()).sum() / w.sum().clamp(min=1.0)
            )
            hcl_log_weight_max = log_w.masked_fill(~neg_mask, float("-inf")).amax(dim=1).mean()

    # 6.5 Online Hard Negative Mining with soft easy-negative weighting
    if online_hard_neg_k > 0:
        is_neg = neg_mask
        neg_only = logits.masked_fill(~is_neg, -1e9)
        avail_neg = int(is_neg.sum(dim=1).min().item())
        k = min(online_hard_neg_k, avail_neg)
        assert k > 0, f"online_hard_neg_k={online_hard_neg_k} but no negatives available"
        _, topk_idx = torch.topk(neg_only, k, dim=1)

        hard_neg_mask = torch.zeros_like(pos_mask)
        hard_neg_mask.scatter_(1, topk_idx, True)
        easy_neg_mask = is_neg & ~hard_neg_mask

        if easy_neg_weight <= 0:
            logits = logits.masked_fill(easy_neg_mask, -1e9)
        elif easy_neg_weight < 1.0:
            logits = logits + easy_neg_mask.float() * math.log(easy_neg_weight)

    # 7. Multi-positive InfoNCE
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_count = pos_mask.sum(dim=1)
    row_valid = (local_valid_mask & (pos_count > 0)).float()

    loss_per_row = -(
        (log_prob * pos_mask.float()).sum(dim=1) / pos_count.clamp(min=1).float()
    )
    infonce_loss = (loss_per_row * row_valid).sum() / row_valid.sum().clamp(min=1.0)

    # 8. TCM auxiliary loss on absolute cos-sim (raw_sim in [-1, 1]).
    # neg_mask computed in step 5.5 (shared with HCL above).
    tcm_pos_weight, tcm_neg_weight = _resolve_tcm_branch_weights(
        tcm_loss_weight=tcm_loss_weight,
        tcm_pos_loss_weight=tcm_pos_loss_weight,
        tcm_neg_loss_weight=tcm_neg_loss_weight,
    )
    tcm_neg_mask = neg_mask
    if tcm_neg_scope == "topk" and tcm_neg_topk > 0:
        neg_only = raw_sim.masked_fill(~neg_mask, float("-inf"))
        k = min(int(tcm_neg_topk), neg_only.size(1))
        if k > 0:
            _, topk_idx = torch.topk(neg_only, k=k, dim=1)
            tcm_neg_mask = torch.zeros_like(neg_mask)
            tcm_neg_mask.scatter_(1, topk_idx, True)
            tcm_neg_mask = tcm_neg_mask & neg_mask
        else:
            tcm_neg_mask = torch.zeros_like(neg_mask)
    pos_mask_f = pos_mask.float()
    neg_mask_f = neg_mask.float()
    tcm_neg_mask_f = tcm_neg_mask.float()
    pos_count_total = pos_mask_f.sum()
    neg_count_total = tcm_neg_mask_f.sum()
    pos_count_safe = pos_count_total.clamp(min=1.0)
    neg_count_safe = neg_count_total.clamp(min=1.0)

    tcm_zero = infonce_loss.new_zeros(())
    if tcm_pos_weight > 0.0 or tcm_neg_weight > 0.0:
        pos_viol = F.relu(tcm_pos_threshold - raw_sim)
        neg_viol = F.relu(raw_sim - tcm_neg_threshold)
        if tcm_loss_form == "squared_hinge":
            pos_viol = pos_viol * pos_viol
            neg_viol = neg_viol * neg_viol

        pos_viol_count = (
            ((raw_sim < tcm_pos_threshold) & pos_mask).float().sum()
        )
        neg_viol_count = (
            ((raw_sim > tcm_neg_threshold) & tcm_neg_mask).float().sum()
        )

        if tcm_reduction == "mean_viol":
            tcm_pos_loss = (pos_viol * pos_mask_f).sum() / pos_viol_count.clamp(min=1.0)
            tcm_neg_loss = (neg_viol * tcm_neg_mask_f).sum() / neg_viol_count.clamp(min=1.0)
        else:
            tcm_pos_loss = (pos_viol * pos_mask_f).sum() / pos_count_safe
            tcm_neg_loss = (neg_viol * tcm_neg_mask_f).sum() / neg_count_safe

        total_loss = (
            infonce_loss
            + tcm_pos_weight * tcm_pos_loss
            + tcm_neg_weight * tcm_neg_loss
        )
    else:
        tcm_pos_loss = tcm_zero
        tcm_neg_loss = tcm_zero
        total_loss = infonce_loss

    with torch.no_grad():
        pos_viol_rate = (
            ((raw_sim < tcm_pos_threshold) & pos_mask).float().sum() / pos_count_safe
        )
        neg_viol_rate = (
            ((raw_sim > tcm_neg_threshold) & tcm_neg_mask).float().sum() / neg_count_safe
        )
        pos_sim_mean = (raw_sim * pos_mask_f).sum() / pos_count_safe
        neg_sim_mean = (raw_sim * neg_mask_f).sum() / neg_count_safe

    return {
        "total": total_loss,
        "infonce": infonce_loss,
        "tcm_pos": tcm_pos_loss,
        "tcm_neg": tcm_neg_loss,
        "tcm_pos_viol_rate": pos_viol_rate,
        "tcm_neg_viol_rate": neg_viol_rate,
        "pos_sim_mean": pos_sim_mean,
        "neg_sim_mean": neg_sim_mean,
        "hcl_neg_sim_weighted_mean": hcl_neg_sim_weighted_mean,
        "hcl_log_weight_max": hcl_log_weight_max,
        "pos_count_mean": pos_count.float().mean(),
        "cochunk_neutral_count": cochunk_neutral_count,
        "positive_term_mask_count": positive_term_mask_count,
        "hn_false_positive_masked_count": hn_false_positive_masked_count,
    }


# ==================== Evaluation ====================


@torch.no_grad()
def _encode_terms_batch(
    text_encoder: nn.Module,
    text_tokenizer,
    terms: List[str],
    device: torch.device,
    batch_size: int = DEFAULT_EVAL_TERM_ENCODE_BATCH,
    use_phoneme_append: bool = False,
    text_input_prefix: str = "",
) -> torch.Tensor:
    """Encode a list of term strings through text encoder.
    Returns [N, D] float32 cpu for single-vector, or [N, T, D] for ColBERT."""
    text_encoder.eval()
    if use_phoneme_append:
        enc_terms = [append_phoneme_str(t, g2p_phoneme_str(t)) for t in terms]
    else:
        enc_terms = terms
    all_embs: List[torch.Tensor] = []
    for i in range(0, len(enc_terms), batch_size):
        chunk = enc_terms[i : i + batch_size]
        tok = _tokenize_texts(
            text_tokenizer,
            chunk,
            device,
            text_input_prefix=text_input_prefix,
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            embs = text_encoder(tok.input_ids, tok.attention_mask)
        all_embs.append(embs.float().cpu())
    if all_embs and all_embs[0].ndim == 3:
        return _pad_and_cat_3d(all_embs, dim=0)
    return torch.cat(all_embs, dim=0)


def _load_eval_wiki_terms(wiki_path: str, max_terms: int = 0) -> List[str]:
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
            if max_terms > 0 and len(terms) >= max_terms:
                break
    return terms


def _load_term_key_set_from_glossary(path: str) -> Tuple[set, int]:
    """Load normalized term keys from a JSON/JSONL glossary for train leakage filtering."""
    assert os.path.isfile(path), f"Train exclude glossary not found: {path}"
    terms: set = set()
    raw_count = 0

    def add_entry(entry: Any) -> None:
        nonlocal raw_count
        raw_count += 1
        if isinstance(entry, dict):
            term = (
                entry.get("term")
                or entry.get("term_key")
                or entry.get("text")
                or entry.get("source_term")
                or ""
            )
        else:
            term = entry
        term_key = str(term or "").strip().lower()
        if term_key:
            terms.add(term_key)

    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                try:
                    add_entry(json.loads(line))
                except Exception:
                    continue
    else:
        with open(path, "r", encoding="utf-8") as fin:
            payload = json.load(fin)
        if isinstance(payload, list):
            for entry in payload:
                add_entry(entry)
        elif isinstance(payload, dict):
            entries = payload.get("terms") or payload.get("glossary") or payload.get("data")
            if isinstance(entries, list):
                for entry in entries:
                    add_entry(entry)
            else:
                add_entry(payload)
        else:
            add_entry(payload)

    return terms, raw_count


def load_train_exclude_term_keys(paths: Sequence[str]) -> Tuple[set, List[Tuple[str, int, int]]]:
    """Return exact term_key exclusion set plus per-path load stats."""
    merged: set = set()
    stats: List[Tuple[str, int, int]] = []
    for path in paths or []:
        if not path:
            continue
        terms, raw_count = _load_term_key_set_from_glossary(path)
        before = len(merged)
        merged.update(terms)
        stats.append((path, raw_count, len(merged) - before))
    return merged, stats


def _metric_key_to_checkpoint_suffix(metric_key: str) -> str:
    """Convert a WandB metric key into a stable checkpoint filename suffix."""
    suffix = metric_key.strip().replace("@", "at").replace(".", "p")
    suffix = re.sub(r"[^A-Za-z0-9]+", "_", suffix).strip("_")
    return suffix or "secondary"


def _atomic_torch_save(payload: Dict[str, Any], path: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


# ======Configuration=====
THRESHOLD_SWEEP_STEPS = 200
F_BETA_SQUARED = 4  # beta=2 => beta^2=4, for F2-score
DETECTION_FPR_AT_TPR_TARGETS = (0.95, 0.80)
DETECTION_DETECTORS = ("max_sim", "top1_zscore", "softmax_prob", "neg_entropy")
GLOSSARY_MATCH_MAX_TOKENS = 8
# ======Configuration=====


_GLOSSARY_MATCH_PUNCT_RE = re.compile(r"[^a-z0-9']+")


def _normalize_glossary_match_word(word: str) -> str:
    word = word.strip().lower().replace("\u2019", "'")
    if word.endswith("'s"):
        word = word[:-2]
    word = _GLOSSARY_MATCH_PUNCT_RE.sub("", word)
    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) > 3 and word.endswith("es") and not word.endswith(("ses", "xes")):
        word = word[:-2]
    elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    return word


def _glossary_match_tokens(text: str, max_tokens: int = GLOSSARY_MATCH_MAX_TOKENS) -> Tuple[str, ...]:
    toks = tuple(
        tok for tok in (_normalize_glossary_match_word(w) for w in str(text or "").split())
        if tok
    )
    if not toks or len(toks) > max_tokens:
        return ()
    return toks


def _glossary_match_norm_char_count(text: str) -> int:
    return sum(len(tok) for tok in _glossary_match_tokens(text))


def _sample_eval_text(sample: Dict) -> str:
    return str(sample.get("chunk_src_text", "") or "")


def _build_glossary_positive_indices(
    samples: Sequence[Dict],
    bank_terms: Sequence[str],
    max_tokens: int = GLOSSARY_MATCH_MAX_TOKENS,
    min_norm_chars: int = 1,
) -> Tuple[List[List[int]], List[List[str]], Dict[str, float]]:
    """Map each sample to all active-bank terms present in its metadata/text.

    This makes no-term labels conditional on the actual inference glossary.
    Rows whose transcript contains an active glossary term are treated as
    with-term positives even when their original JSONL `term_key` is empty.
    """
    term_to_indices: Dict[str, List[int]] = {}
    tuple_to_indices: Dict[Tuple[str, ...], List[int]] = {}
    text_match_terms_skipped_short = 0
    for idx, term in enumerate(bank_terms):
        t = str(term or "").strip().lower()
        if not t:
            continue
        term_to_indices.setdefault(t, []).append(idx)
        toks = _glossary_match_tokens(t, max_tokens=max_tokens)
        if toks and sum(len(tok) for tok in toks) >= int(min_norm_chars):
            tuple_to_indices.setdefault(toks, []).append(idx)
        elif toks:
            text_match_terms_skipped_short += 1

    all_indices: List[List[int]] = []
    all_terms: List[List[str]] = []
    stats = {
        "n_original_term_positive": 0.0,
        "n_text_match_positive": 0.0,
        "n_any_positive": 0.0,
        "n_text_match_terms_skipped_short": float(text_match_terms_skipped_short),
    }
    for sample in samples:
        pos: set[int] = set()
        original_positive = False
        text_positive = False
        original_term = str(sample.get("term_text", "") or "").strip().lower()
        if original_term:
            for idx in term_to_indices.get(original_term, []):
                pos.add(idx)
            original_positive = bool(pos)

        text = _sample_eval_text(sample)
        text_toks = [
            tok for tok in (_normalize_glossary_match_word(w) for w in text.split())
            if tok
        ]
        for n in range(1, min(max_tokens, len(text_toks)) + 1):
            for start in range(0, len(text_toks) - n + 1):
                key = tuple(text_toks[start:start + n])
                for idx in tuple_to_indices.get(key, []):
                    if idx not in pos:
                        text_positive = True
                    pos.add(idx)

        ordered = sorted(pos)
        all_indices.append(ordered)
        all_terms.append([str(bank_terms[i]) for i in ordered])
        if original_positive:
            stats["n_original_term_positive"] += 1.0
        if text_positive:
            stats["n_text_match_positive"] += 1.0
        if ordered:
            stats["n_any_positive"] += 1.0
    return all_indices, all_terms, stats


def _positive_indices_to_mask(
    positive_indices: Sequence[Sequence[int]],
    bank_size: int,
) -> torch.Tensor:
    mask = torch.zeros((len(positive_indices), bank_size), dtype=torch.bool)
    for row_idx, indices in enumerate(positive_indices):
        if indices:
            mask[row_idx, torch.tensor(list(indices), dtype=torch.long)] = True
    return mask


def _map_positive_terms_to_bank_indices(
    positive_terms: Sequence[Sequence[str]],
    bank_terms: Sequence[str],
) -> List[List[int]]:
    """Map fixed-denominator positive term strings into a retriever bank.

    The metrics denominator is allowed to be fixed while the retriever bank
    changes with glossary size.  This helper keeps the positive universe fixed
    by first deciding positives as term strings, then asking which of those
    strings are present in the current candidate bank.
    """
    term_to_indices: Dict[str, List[int]] = {}
    for idx, term in enumerate(bank_terms):
        key = str(term or "").strip().lower()
        if key:
            term_to_indices.setdefault(key, []).append(idx)

    mapped: List[List[int]] = []
    for row_terms in positive_terms:
        row: set[int] = set()
        for term in row_terms:
            key = str(term or "").strip().lower()
            for idx in term_to_indices.get(key, []):
                row.add(idx)
        mapped.append(sorted(row))
    return mapped


def _topk_recall_from_positive_mask(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    k: int,
) -> float:
    K = min(int(k), logits.size(1))
    top_idx = torch.topk(logits, k=K, dim=1).indices
    hits = positive_mask.gather(1, top_idx).any(dim=1)
    return float(hits.float().mean().item())


def _roc_auc_from_ranks(
    scores: torch.Tensor, labels: torch.Tensor
) -> float:
    """Compute binary ROC AUC via Mann-Whitney U statistic with tie-averaged ranks.

    Args:
        scores: [N] float tensor; higher values should correspond to positive class.
        labels: [N] bool tensor; True marks positives.

    Returns:
        AUC in [0, 1] or ``float('nan')`` when either class is empty.
    """
    assert scores.ndim == 1 and labels.ndim == 1, (
        f"Expected 1D scores/labels, got {scores.shape} / {labels.shape}"
    )
    assert scores.numel() == labels.numel(), "scores/labels must have equal length"

    scores_f = scores.detach().float().cpu()
    labels_b = labels.detach().bool().cpu()
    n = scores_f.numel()
    n_pos = int(labels_b.sum().item())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    sorted_vals, sorted_idx = torch.sort(scores_f)
    ranks = torch.empty(n, dtype=torch.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and float(sorted_vals[j + 1].item()) == float(sorted_vals[i].item()):
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-indexed ranks, averaged over tie group
        ranks[sorted_idx[i : j + 1]] = avg_rank
        i = j + 1
    sum_ranks_pos = float(ranks[labels_b].sum().item())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _average_precision(
    scores: torch.Tensor, labels: torch.Tensor
) -> float:
    """Compute average precision (area under PR curve, step-integration form).

    Args:
        scores: [N] float tensor; higher = more likely positive.
        labels: [N] bool tensor.
    """
    assert scores.ndim == 1 and labels.ndim == 1, (
        f"Expected 1D inputs, got {scores.shape} / {labels.shape}"
    )
    assert scores.numel() == labels.numel(), "scores/labels must have equal length"

    scores_f = scores.detach().float().cpu()
    labels_b = labels.detach().bool().cpu()
    n = scores_f.numel()
    n_pos = int(labels_b.sum().item())
    if n_pos == 0 or n_pos == n:
        return float("nan")

    idx = torch.argsort(scores_f, descending=True)
    labels_sorted = labels_b[idx].float()
    tp = torch.cumsum(labels_sorted, dim=0)
    fp = torch.cumsum(1.0 - labels_sorted, dim=0)
    precision = tp / (tp + fp).clamp(min=1.0)
    recall = tp / float(n_pos)
    prev_recall = torch.cat(
        [torch.zeros(1, dtype=recall.dtype), recall[:-1]]
    )
    ap = float(((recall - prev_recall) * precision).sum().item())
    return ap


def _fpr_at_tpr(
    scores: torch.Tensor, labels: torch.Tensor, target_tpr: float
) -> float:
    """Return FPR at the smallest threshold that achieves TPR >= target_tpr.

    Returns ``1.0`` if no threshold can reach *target_tpr* (i.e. positives are dominated).
    """
    assert 0.0 < target_tpr <= 1.0, f"target_tpr must be in (0, 1], got {target_tpr}"

    scores_f = scores.detach().float().cpu()
    labels_b = labels.detach().bool().cpu()
    n_pos = int(labels_b.sum().item())
    n_neg = labels_b.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    idx = torch.argsort(scores_f, descending=True)
    labels_sorted = labels_b[idx].float()
    tp = torch.cumsum(labels_sorted, dim=0)
    fp = torch.cumsum(1.0 - labels_sorted, dim=0)
    tpr = tp / float(n_pos)
    mask = tpr >= target_tpr
    if not bool(mask.any().item()):
        return 1.0
    first_idx = int(mask.nonzero(as_tuple=False)[0].item())
    return float((fp[first_idx] / n_neg).item())


def _compute_detection_metrics(
    all_logits: torch.Tensor,
    labels: torch.Tensor,
    softmax_temp: float,
    target_tau: Optional[float] = None,
) -> Dict[str, float]:
    """Compute chunk-level ``has-term`` detection metrics.

    The goal is to measure how well the retriever distinguishes chunks that
    contain a glossary term from chunks that do not, BEFORE any threshold is
    applied.  Four detectors are evaluated:

    - ``max_sim``:      raw max cos-similarity over the bank (glossary-dependent).
    - ``top1_zscore``:  (max - mean) / std of the chunk's scores over the bank.
    - ``softmax_prob``: softmax probability of the top-1 term (temperature = retriever).
    - ``neg_entropy``:  negative entropy of the softmax distribution (higher = more peaked).

    Args:
        all_logits: [N, G] cos-sim logits for every chunk against the bank.
        labels:     [N] bool; True = has term, False = no-term.
        softmax_temp: softmax temperature; must be > 0.
        target_tau: if given, report FPR / TPR on the ``max_sim`` detector at this
                    threshold (used by ACL to apply the dev-swept tau).

    Returns:
        Dict with keys ``det/<detector>_rocauc``, ``_prauc``, ``_fpr_at_<tpr>_tpr``;
        plus ``det/max_sim_fpr_at_target_tau`` / ``_tpr_at_target_tau`` when
        ``target_tau`` is provided.  ``det/n_with_term`` and ``det/n_no_term``
        record the class counts.
    """
    assert all_logits.ndim == 2, (
        f"Expected 2D logits, got shape {tuple(all_logits.shape)}"
    )
    assert softmax_temp > 0, f"softmax_temp must be > 0, got {softmax_temp}"
    assert all_logits.size(0) == labels.size(0), (
        f"logits/labels size mismatch: {all_logits.size(0)} vs {labels.size(0)}"
    )

    labels_bool = labels.bool()
    n_pos = int(labels_bool.sum().item())
    n_neg = labels_bool.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return {
            "det/n_with_term": float(n_pos),
            "det/n_no_term": float(n_neg),
        }

    all_logits_f = all_logits.detach().float().cpu()

    # Detector 1: raw max similarity
    max_sim = all_logits_f.max(dim=1).values

    # Detector 2: top-1 z-score within this chunk's glossary distribution
    gloss_mean = all_logits_f.mean(dim=1)
    gloss_std = all_logits_f.std(dim=1).clamp(min=1e-6)
    top1_zscore = (max_sim - gloss_mean) / gloss_std

    # Detector 3 + 4: softmax-based scores
    logits_scaled = all_logits_f / softmax_temp
    log_probs = logits_scaled - torch.logsumexp(logits_scaled, dim=1, keepdim=True)
    probs = log_probs.exp()
    softmax_prob_top1 = probs.max(dim=1).values
    entropy = -(probs * log_probs).sum(dim=1)
    neg_entropy = -entropy

    metrics: Dict[str, float] = {
        "det/n_with_term": float(n_pos),
        "det/n_no_term": float(n_neg),
    }

    detector_scores = {
        "max_sim": max_sim,
        "top1_zscore": top1_zscore,
        "softmax_prob": softmax_prob_top1,
        "neg_entropy": neg_entropy,
    }
    for name in DETECTION_DETECTORS:
        s = detector_scores[name]
        metrics[f"det/{name}_rocauc"] = _roc_auc_from_ranks(s, labels_bool)
        metrics[f"det/{name}_prauc"] = _average_precision(s, labels_bool)
        for tpr_tgt in DETECTION_FPR_AT_TPR_TARGETS:
            key = f"det/{name}_fpr_at_{int(tpr_tgt * 100)}_tpr"
            metrics[key] = _fpr_at_tpr(s, labels_bool, tpr_tgt)

    if target_tau is not None:
        pos_scores = max_sim[labels_bool]
        neg_scores = max_sim[~labels_bool]
        metrics["det/max_sim_fpr_at_target_tau"] = float(
            (neg_scores >= target_tau).float().mean().item()
        )
        metrics["det/max_sim_tpr_at_target_tau"] = float(
            (pos_scores >= target_tau).float().mean().item()
        )
        metrics["det/max_sim_target_tau"] = float(target_tau)

    return metrics


def _compute_threshold_metrics(
    recall_logits: torch.Tensor,
    targets_t: torch.Tensor,
    threshold: Optional[float] = None,
    positive_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute threshold-based precision/recall/F2 and score distribution metrics.

    If *threshold* is ``None`` (dev mode), sweeps to find the F2-optimal
    threshold.  Otherwise (acl mode), applies the given threshold directly.
    """
    N = recall_logits.size(0)
    assert N > 0, "Need at least 1 query for threshold metrics"

    if positive_mask is None:
        positive_mask = torch.zeros_like(recall_logits, dtype=torch.bool)
        positive_mask[torch.arange(N), targets_t] = True
    else:
        positive_mask = positive_mask.to(dtype=torch.bool)
    pos_scores_masked = recall_logits.masked_fill(~positive_mask, -float("inf"))
    gt_scores = pos_scores_masked.max(dim=1).values

    logits_masked = recall_logits.clone()
    logits_masked = logits_masked.masked_fill(positive_mask, -float("inf"))
    top1_non_gt_scores = logits_masked.max(dim=1).values

    score_gap = gt_scores - top1_non_gt_scores

    metrics: Dict[str, float] = {
        "gt_score_mean": gt_scores.mean().item(),
        "gt_score_std": gt_scores.std().item(),
        "top1_nongt_score_mean": top1_non_gt_scores.mean().item(),
        "top1_nongt_score_std": top1_non_gt_scores.std().item(),
        "score_gap": score_gap.mean().item(),
    }

    beta_sq = F_BETA_SQUARED

    def _pr_f2(tau_val: float):
        tp = (gt_scores >= tau_val).sum().item()
        total_retrieved = (recall_logits >= tau_val).sum().item()
        p = tp / max(1, total_retrieved)
        r = tp / N
        denom = beta_sq * p + r
        f2 = (1 + beta_sq) * p * r / denom if denom > 0 else 0.0
        return p, r, f2

    if threshold is None:
        score_min = recall_logits.min().item()
        score_max = recall_logits.max().item()
        thresholds = torch.linspace(score_min, score_max, THRESHOLD_SWEEP_STEPS)

        best_f2 = -1.0
        best_tau = score_min
        best_p, best_r = 0.0, 0.0

        for tau in thresholds:
            p, r, f2 = _pr_f2(tau.item())
            if f2 > best_f2:
                best_f2, best_tau, best_p, best_r = f2, tau.item(), p, r

        metrics["opt_threshold"] = best_tau
        metrics["precision@tau"] = best_p
        metrics["recall@tau"] = best_r
        metrics["f2@tau"] = best_f2
    else:
        p, r, f2 = _pr_f2(threshold)
        metrics["precision@tau"] = p
        metrics["recall@tau"] = r
        metrics["f2@tau"] = f2

    return metrics


def _compute_tcm_threshold_metrics(
    recall_logits: torch.Tensor,
    targets_t: torch.Tensor,
    tcm_pos_threshold: float,
    tcm_neg_threshold: float,
    extra_thresholds: Optional[List[float]] = None,
    topk_for_filtered: int = 10,
    minimal: bool = False,
    positive_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Precision / recall / F1 / pass-rate at the two TCM calibration thresholds.

    This mirrors the filtering policy TCM enforces during training:

    * ``T_beta`` (pos threshold, strict): accept a retrieval iff its score
      >= T_beta.  ``precision@tbeta`` = TP / total_accepted_cells (how clean
      the accept set is across the entire query x bank score matrix);
      ``recall@tbeta`` = TP / N (fraction of queries whose gt score clears
      T_beta).  Also reports ``pass_rate@tbeta`` = fraction of queries with
      at least one candidate at/above T_beta, which is the natural
      downstream-gating keep-rate.

    * ``T_alpha`` (neg threshold, loose): accept iff score >= T_alpha; same
      metrics computed on the larger accept set.  At this threshold precision
      is typically much lower and recall higher; the gap vs tbeta quantifies
      how much signal lives in the grey zone between the two thresholds.

    When ``extra_thresholds`` is given, the function additionally sweeps
    each tau in that list and reports P / R / pass_rate plus two
    filter-after-retrieval metrics that are the primary deployment signal:

    * ``tcm_filtered_top1@tau_{tau}``: top-1 accuracy restricted to queries
      that pass the filter (max bank score >= tau).  This is the true
      "confidence-gated" accuracy if we only accept retrievals with at
      least one high-enough candidate.

    * ``chunk_any_positive_filtered_recall@tau_{tau}``: recall@k among
      filter-passing queries, where any active-bank term present in the chunk
      text/metadata counts as a positive.  Shows how much headroom the top-1
      mistake has inside the accept zone.

    ``topk_for_filtered`` controls k for the filtered recall metric (default
    10 to match the primary ``recall@10`` we already track).
    """
    N = recall_logits.size(0)
    assert N > 0, "Need at least 1 query for TCM threshold metrics"

    if positive_mask is None:
        positive_mask = torch.zeros_like(recall_logits, dtype=torch.bool)
        positive_mask[torch.arange(N), targets_t] = True
    else:
        positive_mask = positive_mask.to(dtype=torch.bool)
    pos_scores_masked = recall_logits.masked_fill(~positive_mask, -float("inf"))
    gt_scores = pos_scores_masked.max(dim=1).values

    metrics: Dict[str, float] = {}
    if not minimal:
        # Full matrix-level TCM metrics at tbeta / talpha.  In minimal mode these
        # are skipped because they are superseded by the per-candidate sweep
        # metrics below (which match deployment semantics).
        for name, tau in (
            ("tbeta", float(tcm_pos_threshold)),
            ("talpha", float(tcm_neg_threshold)),
        ):
            tp = (gt_scores >= tau).sum().item()
            total_retrieved = (recall_logits >= tau).sum().item()
            precision = tp / max(1, total_retrieved)
            recall = tp / N
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            pass_rate = (recall_logits >= tau).any(dim=1).float().mean().item()

            metrics[f"tcm_precision@{name}"] = precision
            metrics[f"tcm_recall@{name}"] = recall
            metrics[f"tcm_f1@{name}"] = f1
            metrics[f"tcm_pass_rate@{name}"] = pass_rate
            metrics[f"tcm_{name}_value"] = tau

    if extra_thresholds:
        # Per-candidate top-K ∩ τ: emulate the deployment pipeline where the
        # LLM-side is only ever shown the top-K retrievals, then candidates
        # with cosine similarity below τ are dropped.  Metrics are computed
        # over the resulting filtered subsets, not over the full bank × query
        # similarity matrix.
        #
        # For each query q with top-K indices I_q and scores s_q:
        #   kept_q       = {i ∈ I_q : s_q[i] >= τ}      (|kept_q| in [0, K])
        #   gt_in_topk_q = (GT ∈ I_q)
        #   gt_kept_q    = (GT ∈ kept_q)                (= GT ∈ I_q AND s_gt >= τ)
        # Aggregates:
        #   filtered_recall = mean(gt_kept_q)
        #      -> fraction of queries whose GT survives both cuts
        #   precision_topk_tau_micro = sum(gt_kept_q) / sum(|kept_q|)
        #      -> out of all candidates we actually forward to the LLM,
        #         what fraction are correct (micro-avg)
        #   precision_topk_tau_macro_passing = mean over passing q of
        #      1(gt_kept_q) / |kept_q|
        #      -> expected precision of a single passing query
        #   pass_rate = mean(|kept_q| > 0)
        #      -> fraction of queries where at least one top-K candidate
        #         survives τ (complement is the abstention rate)
        #   avg_kept_if_pass = mean over passing q of |kept_q|
        K = min(topk_for_filtered, recall_logits.size(1))
        topk_out = torch.topk(recall_logits, k=K, dim=1)
        top_idx = topk_out.indices
        top_scores = topk_out.values
        gt_in_topk = positive_mask.gather(1, top_idx)

        for tau in extra_thresholds:
            tau_f = float(tau)
            tag = f"tau_{tau_f:.2f}".replace(".", "p")

            keep_mask = top_scores >= tau_f
            kept_count = keep_mask.sum(dim=1)
            pass_row = kept_count > 0
            gt_kept = (keep_mask & gt_in_topk).any(dim=1)

            pass_rate = pass_row.float().mean().item()
            pass_count = int(pass_row.sum().item())

            filtered_recall = gt_kept.float().mean().item()
            total_kept = int(kept_count.sum().item())
            tp = int(gt_kept.sum().item())
            micro_precision = tp / total_kept if total_kept > 0 else 0.0

            if pass_count > 0:
                per_q_prec = torch.zeros(N, dtype=torch.float32)
                per_q_prec[gt_kept] = 1.0 / kept_count[gt_kept].float()
                macro_precision_passing = (
                    per_q_prec[pass_row].mean().item()
                )
                avg_kept_if_pass = (
                    kept_count[pass_row].float().mean().item()
                )
            else:
                macro_precision_passing = 0.0
                avg_kept_if_pass = 0.0

            metrics[f"topk{K}_chunk_any_positive_filtered_recall@{tag}"] = filtered_recall
            metrics[f"topk{K}_filtered_precision_micro@{tag}"] = micro_precision
            metrics[f"topk{K}_filtered_precision_macro@{tag}"] = macro_precision_passing
            metrics[f"topk{K}_avg_kept_if_pass@{tag}"] = avg_kept_if_pass
            if not minimal:
                # Derivable from P_mac and avg_kept; emitted only in full mode.
                metrics[f"topk{K}_pass_rate@{tag}"] = pass_rate
                metrics[f"topk{K}_{tag}_value"] = tau_f

    return metrics


def _compute_noterm_noise(
    full_logits: torch.Tensor,
    has_term_mask: torch.Tensor,
    taus: List[float],
    topk: int = 10,
) -> Dict[str, float]:
    """Average count of top-K candidates passing each tau on no-term chunks.

    Deployment signal: when an audio segment has no GT term (false positive
    territory), how many candidates does a tau-gated retriever still emit?
    Lower is better. Matches `avg_noise_terms` in threshold_sweep_maxsim.py.
    """
    noterm_mask = ~has_term_mask
    if int(noterm_mask.sum().item()) == 0:
        return {}
    noterm_logits = full_logits[noterm_mask]  # [M, bank]
    K = min(topk, noterm_logits.size(1))
    top_vals, _ = torch.topk(noterm_logits, k=K, dim=1)  # [M, K]
    out: Dict[str, float] = {}
    for tau in taus:
        tau_f = float(tau)
        tag = f"tau_{tau_f:.2f}".replace(".", "p")
        avg_kept = (top_vals >= tau_f).sum(dim=1).float().mean().item()
        out[f"noterm_noise@top{K}_{tag}"] = avg_kept
    return out


def _dump_sim_distributions(
    out_path: str,
    logits: torch.Tensor,
    targets: torch.Tensor,
    k_neg: int = 32,
    k_rank: int = 10,
    term_texts: Optional[List[str]] = None,
    positive_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Dump per-sample pos_sim + top-K neg_sim to an NPZ.

    If `positive_mask` is provided, all active-bank positives are excluded from
    negatives and `pos_sim` is the best positive score for the row.
    """
    import numpy as np
    logits_f = logits.detach().float().cpu()
    tgt = targets.detach().long().cpu()
    N, bank = logits_f.shape
    rows = torch.arange(N)
    if positive_mask is None:
        pos_mask = torch.zeros_like(logits_f, dtype=torch.bool)
        pos_mask[rows, tgt] = True
    else:
        pos_mask = positive_mask.detach().bool().cpu()
    pos_sim = logits_f.masked_fill(~pos_mask, -float("inf")).max(dim=1).values
    masked = logits_f.masked_fill(pos_mask, float("-inf"))
    K = min(k_neg, bank - 1)
    if K <= 0:
        neg_top_sim = torch.zeros((N, 0), dtype=torch.float32)
    else:
        neg_top_sim, _ = torch.topk(masked, k=K, dim=1)
    K_rank = min(int(k_rank), bank)
    top_rank_idx = torch.topk(logits_f, k=K_rank, dim=1).indices
    target_hits = pos_mask.gather(1, top_rank_idx)
    target_in_topk = target_hits.any(dim=1)
    pos_rank = torch.full((N,), bank + 1, dtype=torch.int64)
    if int(target_in_topk.sum().item()) > 0:
        hit_rows = torch.nonzero(target_in_topk, as_tuple=False).squeeze(1)
        hit_pos = target_hits[hit_rows].float().argmax(dim=1).long() + 1
        pos_rank[hit_rows] = hit_pos
    # For mean/max over all non-positive negatives.  Rows may have different
    # numbers of positives after glossary-conditioned relabeling.
    neg_count = (~pos_mask).sum(dim=1).clamp(min=1).float()
    neg_sum = logits_f.masked_fill(pos_mask, 0.0).sum(dim=1)
    neg_sim_mean = neg_sum / neg_count
    neg_sim_max = logits_f.masked_fill(pos_mask, -float("inf")).max(dim=1).values
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_kwargs = dict(
        pos_sim=pos_sim.numpy(),
        neg_top_sim=neg_top_sim.numpy(),
        neg_sim_mean=neg_sim_mean.numpy(),
        neg_sim_max=neg_sim_max.numpy(),
        pos_rank_topk=pos_rank.numpy(),
        rank_k=np.array([K_rank], dtype=np.int64),
        bank_size=np.array([bank], dtype=np.int64),
        K_neg=np.array([K], dtype=np.int64),
    )
    if term_texts is not None:
        save_kwargs["term_texts"] = np.array(term_texts, dtype=object)
    np.savez(out_path, **save_kwargs)
    return {
        "pos_mean": float(pos_sim.mean().item()),
        "pos_median": float(pos_sim.median().item()),
        "pos_p10": float(torch.quantile(pos_sim, 0.10).item()),
        "pos_p90": float(torch.quantile(pos_sim, 0.90).item()),
        "neg_max_mean": float(neg_sim_max.mean().item()),
        "neg_max_p90": float(torch.quantile(neg_sim_max, 0.90).item()),
        "neg_mean_mean": float(neg_sim_mean.mean().item()),
        "gap_mean": float((pos_sim - neg_sim_max).mean().item()),
        "gap_p10": float(torch.quantile(pos_sim - neg_sim_max, 0.10).item()),
    }


def _dump_noterm_topk_scores(
    out_path: str,
    logits: torch.Tensor,
    has_term_mask: torch.Tensor,
    topk: int = 10,
    term_names: Optional[List[str]] = None,
    samples: Optional[List[Dict]] = None,
) -> Dict[str, float]:
    """Dump raw top-K scores for audio-ok no-term chunks."""
    import numpy as np

    logits_f = logits.detach().float().cpu()
    labels = has_term_mask.detach().bool().cpu()
    K = min(int(topk), logits_f.size(1))
    top_vals, top_idx = torch.topk(logits_f, k=K, dim=1)
    noterm_topk = top_vals[~labels]
    noterm_top_idx = top_idx[~labels]
    term_topk = top_vals[labels]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_kwargs = dict(
        noterm_topk_sim=noterm_topk.numpy(),
        noterm_topk_idx=noterm_top_idx.numpy(),
        term_topk_sim=term_topk.numpy(),
        has_term_mask=labels.numpy(),
        bank_size=np.array([logits_f.size(1)], dtype=np.int64),
        K_top=np.array([K], dtype=np.int64),
    )
    if term_names is not None:
        names = list(term_names)
        save_kwargs["bank_terms"] = np.array(names, dtype=object)
        top_terms = [
            [names[int(j)] if int(j) < len(names) else "" for j in row]
            for row in noterm_top_idx.tolist()
        ]
        save_kwargs["noterm_topk_terms"] = np.array(top_terms, dtype=object)
    if samples is not None and len(samples) == len(labels):
        noterm_samples = [s for s, has_term in zip(samples, labels.tolist()) if not has_term]

        def field(name: str) -> np.ndarray:
            return np.array([str(s.get(name, "") or "") for s in noterm_samples], dtype=object)

        save_kwargs["noterm_utter_id"] = field("utter_id")
        save_kwargs["noterm_chunk_idx"] = field("chunk_idx")
        save_kwargs["noterm_chunk_audio_path"] = field("chunk_audio_path")
        save_kwargs["noterm_chunk_src_text"] = field("chunk_src_text")
        save_kwargs["noterm_source_seg_id"] = field("source_seg_id")
        save_kwargs["noterm_source_audio"] = field("source_audio")
        save_kwargs["noterm_source_start_sample"] = field("source_start_sample")
    np.savez(out_path, **save_kwargs)
    if noterm_topk.numel() == 0:
        return {"n_no_term": 0.0, "noterm_top1_p95": float("nan")}
    top1 = noterm_topk[:, 0]
    return {
        "n_no_term": float(noterm_topk.size(0)),
        "n_with_term": float(term_topk.size(0)),
        "noterm_top1_mean": float(top1.mean().item()),
        "noterm_top1_p90": float(torch.quantile(top1, 0.90).item()),
        "noterm_top1_p95": float(torch.quantile(top1, 0.95).item()),
        "noterm_top1_p99": float(torch.quantile(top1, 0.99).item()),
    }


def _split_cli_tokens(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    out: List[str] = []
    for value in values:
        for part in str(value).replace(",", " ").split():
            part = part.strip()
            if part:
                out.append(part)
    return out


def _safe_dump_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("_") or "unknown"


def _should_dump_eval_misses(args: argparse.Namespace, eval_name: str, bank_label: str) -> bool:
    out_dir = getattr(args, "dump_eval_misses_dir", "") or ""
    if not out_dir:
        return False
    names = {x.lower() for x in _split_cli_tokens(getattr(args, "dump_eval_misses_eval_names", []))}
    eval_keys = {eval_name.lower(), f"eval_{eval_name.lower()}"}
    if names and not (names & eval_keys):
        return False
    banks = {x.lower() for x in _split_cli_tokens(getattr(args, "dump_eval_misses_banks", []))}
    if banks and bank_label.lower() not in banks:
        return False
    return True


def _sample_text_excerpt(text: Any, max_chars: int = 320) -> str:
    excerpt = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(excerpt) <= max_chars:
        return excerpt
    return excerpt[: max_chars - 3] + "..."


def _dump_eval_miss_cases(
    args: argparse.Namespace,
    eval_name: str,
    bank_label: str,
    logits: torch.Tensor,
    positive_indices: List[List[int]],
    sample_indices: List[int],
    sample_list: List[Dict],
    term_names: List[str],
    topk: int,
    global_step: int,
    epoch: int,
) -> None:
    if not _should_dump_eval_misses(args, eval_name, bank_label):
        return
    if logits.numel() == 0 or logits.size(0) != len(positive_indices):
        logger.warning(
            f"[MISS_DUMP][{eval_name}][{bank_label}] skip: "
            f"logits_shape={tuple(logits.shape)} positives={len(positive_indices)}"
        )
        return

    logits_f = logits.detach().float().cpu()
    bank_size = logits_f.size(1)
    k = min(max(int(topk), 1), bank_size)
    top_vals, top_idx = torch.topk(logits_f, k=k, dim=1)
    term_name_by_idx = list(term_names)
    term_to_idx = {str(term).strip().lower(): i for i, term in enumerate(term_name_by_idx)}

    rows: List[Dict[str, Any]] = []
    for row_idx, pos_raw in enumerate(positive_indices):
        pos_set = {
            int(idx)
            for idx in pos_raw
            if 0 <= int(idx) < bank_size
        }
        if not pos_set:
            continue
        retrieved = [int(idx) for idx in top_idx[row_idx].tolist()]
        if any(idx in pos_set for idx in retrieved):
            continue

        row_scores = logits_f[row_idx]
        pos_infos: List[Dict[str, Any]] = []
        for pos_idx in sorted(pos_set):
            score = float(row_scores[pos_idx].item())
            rank = int((row_scores > score).sum().item()) + 1
            pos_infos.append(
                {
                    "term": term_name_by_idx[pos_idx] if pos_idx < len(term_name_by_idx) else "",
                    "index": pos_idx,
                    "rank": rank,
                    "score": score,
                }
            )
        pos_infos.sort(key=lambda item: (item["rank"], -item["score"]))
        best_pos = pos_infos[0]

        sample_idx = sample_indices[row_idx] if row_idx < len(sample_indices) else -1
        sample = sample_list[sample_idx] if 0 <= sample_idx < len(sample_list) else {}
        target_terms = [
            str(sample.get("term_text", "") or "").strip().lower(),
            str(sample.get("term_key", "") or "").strip().lower(),
            str(sample.get("term", "") or "").strip().lower(),
        ]
        target_terms = [t for t in target_terms if t]
        target_idx = None
        for target_term in target_terms:
            if target_term in term_to_idx:
                target_idx = term_to_idx[target_term]
                break
        target_rank = None
        target_score = None
        if target_idx is not None:
            target_score = float(row_scores[target_idx].item())
            target_rank = int((row_scores > target_score).sum().item()) + 1

        predictions: List[Dict[str, Any]] = []
        for rank, (idx, score) in enumerate(
            zip(top_idx[row_idx].tolist(), top_vals[row_idx].tolist()), 1
        ):
            idx_i = int(idx)
            predictions.append(
                {
                    "rank": rank,
                    "term": term_name_by_idx[idx_i] if idx_i < len(term_name_by_idx) else "",
                    "index": idx_i,
                    "score": float(score),
                    "is_positive": idx_i in pos_set,
                }
            )

        top1_score = predictions[0]["score"] if predictions else float("nan")
        row = {
            "eval_name": eval_name,
            "bank_label": bank_label,
            "global_step": int(global_step),
            "epoch": int(epoch),
            "row_idx": row_idx,
            "sample_idx": int(sample_idx),
            "bank_size": int(bank_size),
            "topk": int(k),
            "utter_id": sample.get("utter_id", ""),
            "sample_id": sample.get("sample_id", ""),
            "domain": sample.get("domain", ""),
            "chunk_idx": sample.get("chunk_idx", ""),
            "context_duration_tag": sample.get("context_duration_tag", ""),
            "chunk_duration_sec": sample.get("chunk_duration_sec", ""),
            "context_duration_sec": sample.get("context_duration_sec", ""),
            "chunk_audio_path": sample.get("chunk_audio_path", ""),
            "term": sample.get("term", sample.get("term_text", "")),
            "term_key": sample.get("term_key", ""),
            "term_text": sample.get("term_text", ""),
            "mfa_term_start_in_chunk": sample.get("mfa_term_start_in_chunk", ""),
            "mfa_term_end_in_chunk": sample.get("mfa_term_end_in_chunk", ""),
            "mfa_term_duration": sample.get("mfa_term_duration", ""),
            "mfa_locate_method": sample.get("mfa_locate_method", ""),
            "chunk_src_text": sample.get("chunk_src_text", ""),
            "chunk_src_text_excerpt": _sample_text_excerpt(sample.get("chunk_src_text", "")),
            "positive_terms": pos_infos,
            "best_positive_term": best_pos["term"],
            "best_positive_rank": best_pos["rank"],
            "best_positive_score": best_pos["score"],
            "target_rank": target_rank,
            "target_score": target_score,
            "top_predictions": predictions,
            "top1_term": predictions[0]["term"] if predictions else "",
            "top1_score": top1_score,
            "top1_minus_best_positive_score": float(top1_score - best_pos["score"]),
        }
        rows.append(row)

    rows.sort(
        key=lambda item: (
            int(item.get("best_positive_rank") or 0),
            float(item.get("top1_minus_best_positive_score") or 0.0),
        ),
        reverse=True,
    )

    out_dir = getattr(args, "dump_eval_misses_dir", "") or ""
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{_safe_dump_name(eval_name)}_{_safe_dump_name(bank_label)}"
    jsonl_path = os.path.join(out_dir, f"{stem}_misses.jsonl")
    md_path = os.path.join(out_dir, f"{stem}_misses.md")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(positive_indices)
    miss_count = len(rows)
    recall = 1.0 - (miss_count / total if total else 0.0)
    topn = max(int(getattr(args, "dump_eval_misses_topn", 80) or 0), 0)
    md_lines = [
        f"# {eval_name} {bank_label} miss cases",
        "",
        f"- step: {global_step}",
        f"- epoch: {epoch}",
        f"- bank_size: {bank_size}",
        f"- recall@{k}: {recall:.6f}",
        f"- misses: {miss_count}/{total}",
        f"- jsonl: `{jsonl_path}`",
        "",
    ]
    for i, row in enumerate(rows[:topn], 1):
        top_terms = ", ".join(
            f"#{p['rank']} {p['term']} ({p['score']:.4f})"
            for p in row["top_predictions"]
        )
        positives = ", ".join(
            f"{p['term']}@{p['rank']} ({p['score']:.4f})"
            for p in row["positive_terms"][:8]
        )
        md_lines.extend(
            [
                f"## {i}. {row.get('utter_id') or row.get('sample_id')}",
                "",
                f"- term: `{row.get('term') or row.get('term_text')}` / key `{row.get('term_key')}`",
                f"- best_positive: {positives}",
                f"- top1_minus_best_positive_score: {row['top1_minus_best_positive_score']:.4f}",
                f"- chunk_idx/context: {row.get('chunk_idx')} / {row.get('context_duration_tag')}",
                f"- mfa: {row.get('mfa_term_start_in_chunk')} - {row.get('mfa_term_end_in_chunk')} ({row.get('mfa_locate_method')})",
                f"- text: {row.get('chunk_src_text_excerpt')}",
                f"- top{len(row['top_predictions'])}: {top_terms}",
                "",
            ]
        )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines).rstrip() + "\n")
    logger.info(
        f"[MISS_DUMP][{eval_name}][{bank_label}] "
        f"misses={miss_count}/{total} recall@{k}={recall:.4f} "
        f"-> {jsonl_path} / {md_path}"
    )


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
    metrics_terms: Optional[List[str]] = None,
    threshold_from_dev: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    retriever.eval()
    text_encoder.eval()
    t0 = time.time()
    score_device = _resolve_eval_score_device(args, device)
    logger.info(
        f"[EVAL_{eval_name.upper()}] score_device={score_device} "
        f"query_chunk={getattr(args, 'eval_score_query_chunk', DEFAULT_EVAL_SCORE_QUERY_CHUNK)} "
        f"text_chunk={getattr(args, 'eval_score_text_chunk', DEFAULT_EVAL_SCORE_TEXT_CHUNK)}"
    )

    speech_emb_list: List[torch.Tensor] = []
    text_emb_list: List[torch.Tensor] = []
    valid_list: List[torch.Tensor] = []
    term_text_list: List[str] = []
    group_id_list: List[torch.Tensor] = []
    term_id_list: List[torch.Tensor] = []
    sample_list: List[Dict] = []

    with torch.no_grad():
        for batch in eval_loader:
            feats, flens = _move_audio_batch_to_device(
                batch, device, args, non_blocking=False
            )
            texts = batch["term_texts"]
            valid = batch["valid_mask"].to(device)
            samples = batch["samples"]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                s_embs = retriever(feats, flens)
                tok = _tokenize_texts(
                    text_tokenizer,
                    texts,
                    device,
                    text_input_prefix=args.text_input_prefix,
                )
                t_embs = text_encoder(tok.input_ids, tok.attention_mask)

            speech_emb_list.append(s_embs.float().cpu())
            text_emb_list.append(t_embs.float().cpu())
            valid_list.append(valid.bool().cpu())
            term_text_list.extend([
                (s.get("term_text", "") or "").strip().lower() for s in samples
            ])
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
    if text_emb_list and text_emb_list[0].ndim == 3:
        text_embs = _pad_and_cat_3d(text_emb_list, dim=0)
    else:
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

    sim = (
        _score_eval_logits(
            speech_embs, text_embs, args, device, score_device=score_device
        )
        * logit_scale_t
    )

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
    bank_term_names = [""] * len(term_to_bank)
    for term, bank_idx in term_to_bank.items():
        if 0 <= bank_idx < len(bank_term_names):
            bank_term_names[bank_idx] = term
    base_positive_indices, _, base_label_stats = _build_glossary_positive_indices(
        sample_list,
        metrics_terms or bank_term_names,
        min_norm_chars=args.eval_glossary_match_min_norm_chars,
    )
    if metrics_terms:
        _, base_positive_terms, base_label_stats = _build_glossary_positive_indices(
            sample_list,
            metrics_terms,
            min_norm_chars=args.eval_glossary_match_min_norm_chars,
        )
        base_positive_indices = _map_positive_terms_to_bank_indices(
            base_positive_terms,
            bank_term_names,
        )
    else:
        _, base_positive_terms, _ = _build_glossary_positive_indices(
            sample_list,
            bank_term_names,
            min_norm_chars=args.eval_glossary_match_min_norm_chars,
        )
    base_has_term_mask = torch.tensor(
        [bool(terms) for terms in base_positive_terms],
        dtype=torch.bool,
    )
    fixed_metric_denominator = (
        str(getattr(args, "eval_metric_denominator", "fixed_raw")).strip().lower()
        == "fixed_raw"
    )
    speech_valid = speech_embs[valid_indices]
    recall_logits = _score_eval_logits(
        speech_valid, bank_embs, args, device, score_device=score_device
    )

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
    recall_sample_indices = [valid_indices[i] for i in row_keep]
    recall_positive_indices = [base_positive_indices[i] for i in recall_sample_indices]
    recall_positive_terms = [base_positive_terms[i] for i in recall_sample_indices]
    recall_positive_mask = _positive_indices_to_mask(
        recall_positive_indices,
        recall_logits.size(1),
    )

    top1_idx = recall_logits.argmax(dim=1, keepdim=True)
    top1 = recall_positive_mask.gather(1, top1_idx).float().mean().item()
    k_primary = min(args.eval_topk, recall_logits.size(1))
    k_extra = min(args.eval_topk_extra, recall_logits.size(1))
    recall_primary = _topk_recall_from_positive_mask(
        recall_logits, recall_positive_mask, k_primary
    )
    recall_extra = _topk_recall_from_positive_mask(
        recall_logits, recall_positive_mask, k_extra
    )

    prefix = f"eval_{eval_name}"
    metrics: Dict[str, float] = {
        f"{prefix}/loss": eval_loss,
        f"{prefix}/top1": top1,
        f"{prefix}/recall@{k_primary}": recall_primary,
        f"{prefix}/recall@{k_extra}": recall_extra,
        f"{prefix}/base_label_text_match_positive": base_label_stats["n_text_match_positive"],
        f"{prefix}/base_label_any_positive": base_label_stats["n_any_positive"],
        f"{prefix}/base_label_text_match_terms_skipped_short": base_label_stats[
            "n_text_match_terms_skipped_short"
        ],
        f"{prefix}/fixed_metric_denominator": 1.0 if fixed_metric_denominator else 0.0,
        f"{prefix}/metrics_bank_terms": float(len(metrics_terms or bank_term_names)),
        f"{prefix}/glossary_match_min_norm_chars": float(
            args.eval_glossary_match_min_norm_chars
        ),
    }

    # ---- Optional: dump pos_sim / neg_top_sim distributions (base bank) ----
    dump_base = getattr(args, "dump_sim_distributions", "") or ""
    if dump_base:
        # term_texts aligned with recall_logits rows: valid_indices[row_keep[i]]
        kept_term_texts = [
            term_text_list[valid_indices[i]] for i in row_keep
        ]
        base_out = os.path.join(dump_base, f"{eval_name}_base.npz")
        summary = _dump_sim_distributions(
            base_out, recall_logits, targets_t,
            term_texts=kept_term_texts,
            positive_mask=recall_positive_mask,
        )
        logger.info(
            f"[SIM_DIST][{eval_name}][base bank={recall_logits.size(1)}] "
            f"pos mean={summary['pos_mean']:.3f} median={summary['pos_median']:.3f} "
            f"p10={summary['pos_p10']:.3f} p90={summary['pos_p90']:.3f}  "
            f"neg_max mean={summary['neg_max_mean']:.3f} p90={summary['neg_max_p90']:.3f} "
            f"neg_mean={summary['neg_mean_mean']:.3f}  "
            f"gap mean={summary['gap_mean']:.3f} p10={summary['gap_p10']:.3f}  "
            f"-> {base_out}"
        )

    _dump_eval_miss_cases(
        args=args,
        eval_name=eval_name,
        bank_label="base",
        logits=recall_logits,
        positive_indices=recall_positive_indices,
        sample_indices=recall_sample_indices,
        sample_list=sample_list,
        term_names=bank_term_names,
        topk=k_primary,
        global_step=global_step,
        epoch=epoch,
    )

    # ---- Threshold-based precision / F2 / score-gap (base bank) ----
    eval_minimal = bool(getattr(args, "eval_minimal_metrics", False))
    base_tau = threshold_from_dev.get("base") if threshold_from_dev else None
    if eval_minimal:
        tm_base: Dict[str, float] = {}
    else:
        tm_base = _compute_threshold_metrics(
            recall_logits,
            targets_t,
            threshold=base_tau,
            positive_mask=recall_positive_mask,
        )
        for k, v in tm_base.items():
            metrics[f"{prefix}/{k}"] = v

    # ---- TCM-gated precision / recall at T_beta and T_alpha (base bank) ----
    tcm_sweep = list(getattr(args, "tcm_sweep_thresholds", []) or [])
    tcm_base = _compute_tcm_threshold_metrics(
        recall_logits,
        targets_t,
        tcm_pos_threshold=args.tcm_pos_threshold,
        tcm_neg_threshold=args.tcm_neg_threshold,
        extra_thresholds=tcm_sweep if tcm_sweep else None,
        topk_for_filtered=k_primary,
        minimal=eval_minimal,
        positive_mask=recall_positive_mask,
    )
    for k, v in tcm_base.items():
        metrics[f"{prefix}/{k}"] = v

    # ---- Chunk-level has-term detection metrics (base bank) ----
    # Include no-term chunks that were previously filtered by valid_mask.  A
    # chunk contributes to detection iff its audio loaded successfully; the
    # label is whether the chunk has a non-empty term_text.  Scoring uses the
    # same bank as recall but is applied to every audio-ok chunk.
    has_term_mask = base_has_term_mask
    audio_ok_mask = torch.tensor(
        [not bool(s.get("skip_sample", False)) for s in sample_list],
        dtype=torch.bool,
    )
    det_select = audio_ok_mask
    det_metrics_base = {}
    full_logits_base = None
    det_labels_base = None
    speech_det = None
    if int(det_select.sum().item()) > 0 and int(has_term_mask[det_select].sum().item()) > 0:
        # Compute full_logits_base only when needed. In minimal mode, a tau
        # sweep needs full audio-ok logits only for no-term noise; all-positive
        # dev sets can skip this duplicate pass over the bank.
        det_labels_candidate = has_term_mask[det_select]
        has_noterm_for_noise = int((~det_labels_candidate).sum().item()) > 0
        need_full_logits = (
            (not eval_minimal)
            or bool(dump_base)
            or (bool(tcm_sweep) and has_noterm_for_noise)
        )
        if need_full_logits:
            speech_det = speech_embs[det_select]
            det_samples = [
                sample_list[i]
                for i, keep in enumerate(det_select.tolist())
                if keep
            ]
            full_logits_base = _score_eval_logits(
                speech_det, bank_embs, args, device, score_device=score_device
            )
            det_labels_base = det_labels_candidate
            if not eval_minimal:
                det_target_tau_base = (
                    base_tau if base_tau is not None else tm_base.get("opt_threshold")
                )
                det_metrics_base = _compute_detection_metrics(
                    all_logits=full_logits_base,
                    labels=det_labels_base,
                    softmax_temp=args.temperature,
                    target_tau=det_target_tau_base,
                )
                for k, v in det_metrics_base.items():
                    metrics[f"{prefix}/{k}"] = v

    if dump_base and full_logits_base is not None and det_labels_base is not None:
        no_term_out = os.path.join(dump_base, f"{eval_name}_base_noterm_topk.npz")
        no_term_summary = _dump_noterm_topk_scores(
            no_term_out,
            full_logits_base,
            det_labels_base,
            topk=k_primary,
            term_names=bank_term_names,
            samples=det_samples,
        )
        logger.info(
            f"[NOTERM_DIST][{eval_name}][base bank={full_logits_base.size(1)}] "
            f"n_no_term={no_term_summary.get('n_no_term', 0):.0f} "
            f"top1_p90={no_term_summary.get('noterm_top1_p90', float('nan')):.3f} "
            f"top1_p95={no_term_summary.get('noterm_top1_p95', float('nan')):.3f} "
            f"top1_p99={no_term_summary.get('noterm_top1_p99', float('nan')):.3f} "
            f"-> {no_term_out}"
        )

    # ---- No-term chunk avg-kept noise (base bank), per sweep tau ----
    noise_base: Dict[str, float] = {}
    if (
        tcm_sweep
        and full_logits_base is not None
        and det_labels_base is not None
        and int((~det_labels_base).sum().item()) > 0
    ):
        noise_base = _compute_noterm_noise(
            full_logits_base, det_labels_base,
            taus=tcm_sweep, topk=k_primary,
        )
        for k, v in noise_base.items():
            metrics[f"{prefix}/{k}"] = v

    if eval_minimal:
        log_parts = [
            f"[EVAL_{eval_name.upper()}] step={global_step} epoch={epoch}",
            f"samples={len(valid_indices)} bank_terms={len(bank_rows)}",
            f"loss={eval_loss:.6f}",
            f"top1={top1:.4f}",
            f"recall@{k_primary}={recall_primary:.4f}",
            f"recall@{k_extra}={recall_extra:.4f}",
        ]
    else:
        log_parts = [
            f"[EVAL_{eval_name.upper()}] step={global_step} epoch={epoch}",
            f"samples={len(valid_indices)} bank_terms={len(bank_rows)}",
            f"loss={eval_loss:.6f}",
            f"top1={top1:.4f}",
            f"recall@{k_primary}={recall_primary:.4f}",
            f"recall@{k_extra}={recall_extra:.4f}",
            f"f2@tau={tm_base.get('f2@tau', 0):.4f}",
            f"P@tau={tm_base.get('precision@tau', 0):.4f}",
            f"score_gap={tm_base.get('score_gap', 0):.4f}",
            f"tcm_P/R@tbeta={tcm_base.get('tcm_precision@tbeta', 0):.4f}"
            f"/{tcm_base.get('tcm_recall@tbeta', 0):.4f}",
            f"tcm_P/R@talpha={tcm_base.get('tcm_precision@talpha', 0):.4f}"
            f"/{tcm_base.get('tcm_recall@talpha', 0):.4f}",
            f"tcm_pass@tbeta/talpha={tcm_base.get('tcm_pass_rate@tbeta', 0):.4f}"
            f"/{tcm_base.get('tcm_pass_rate@talpha', 0):.4f}",
        ]
    if det_metrics_base:
        log_parts.extend(
            [
                f"det_auc(max/z/sm/H)="
                f"{det_metrics_base.get('det/max_sim_rocauc', float('nan')):.3f}/"
                f"{det_metrics_base.get('det/top1_zscore_rocauc', float('nan')):.3f}/"
                f"{det_metrics_base.get('det/softmax_prob_rocauc', float('nan')):.3f}/"
                f"{det_metrics_base.get('det/neg_entropy_rocauc', float('nan')):.3f}",
                f"fpr@95tpr={det_metrics_base.get('det/max_sim_fpr_at_95_tpr', float('nan')):.3f}",
            ]
        )
        if "det/max_sim_fpr_at_target_tau" in det_metrics_base:
            log_parts.append(
                f"fpr@tau={det_metrics_base['det/max_sim_fpr_at_target_tau']:.3f}"
            )

    if tcm_sweep:
        for tau in tcm_sweep:
            tau_f = float(tau)
            tag = f"tau_{tau_f:.2f}".replace(".", "p")
            noise_key = f"noterm_noise@top{k_primary}_{tag}"
            noise_val = noise_base.get(noise_key, float("nan"))
            if eval_minimal:
                log_parts.append(
                    f"sweep@{tau_f:.2f}: "
                    f"R={tcm_base.get(f'topk{k_primary}_chunk_any_positive_filtered_recall@{tag}', 0):.3f} "
                    f"P_mic={tcm_base.get(f'topk{k_primary}_filtered_precision_micro@{tag}', 0):.3f} "
                    f"P_mac={tcm_base.get(f'topk{k_primary}_filtered_precision_macro@{tag}', 0):.3f} "
                    f"kept={tcm_base.get(f'topk{k_primary}_avg_kept_if_pass@{tag}', 0):.2f} "
                    f"noise={noise_val:.2f}"
                )
            else:
                log_parts.append(
                    f"sweep@{tau_f:.2f}: "
                    f"R={tcm_base.get(f'topk{k_primary}_chunk_any_positive_filtered_recall@{tag}', 0):.3f} "
                    f"P_mic={tcm_base.get(f'topk{k_primary}_filtered_precision_micro@{tag}', 0):.3f} "
                    f"P_mac={tcm_base.get(f'topk{k_primary}_filtered_precision_macro@{tag}', 0):.3f} "
                    f"pass={tcm_base.get(f'topk{k_primary}_pass_rate@{tag}', 0):.3f} "
                    f"kept={tcm_base.get(f'topk{k_primary}_avg_kept_if_pass@{tag}', 0):.2f} "
                    f"noise={noise_val:.2f}"
                )
    # ---- Glossary-scale recall ----
    gt_bank_size = len(bank_rows)
    gt_terms_set = set(term_to_bank.keys())
    effective_glossary_sizes = glossary_sizes or []

    largest_gs_logits = None
    largest_gs_term_names = None
    largest_gs = 0

    if wiki_terms and effective_glossary_sizes:
        wiki_filtered = [t for t in wiki_terms if t not in gt_terms_set]
        min_match_chars = int(args.eval_glossary_match_min_norm_chars)
        if min_match_chars > 1:
            before_filter = len(wiki_filtered)
            wiki_filtered = [
                t for t in wiki_filtered
                if _glossary_match_norm_char_count(t) >= min_match_chars
            ]
            skipped_short = before_filter - len(wiki_filtered)
            if skipped_short:
                logger.info(
                    f"[EVAL_{eval_name.upper()}] filtered {skipped_short:,}/"
                    f"{before_filter:,} expansion terms with normalized char count "
                    f"< {min_match_chars}"
                )
        wiki_embs = _encode_terms_batch(
            text_encoder, text_tokenizer, wiki_filtered, device,
            use_phoneme_append=args.use_phoneme_append,
            text_input_prefix=args.text_input_prefix,
        )
        for gs in effective_glossary_sizes:
            n_extra = gs - gt_bank_size
            if n_extra <= 0:
                logger.info(
                    f"[EVAL_{eval_name.upper()}] gs{gs} skipped: "
                    f"GT bank ({gt_bank_size}) already >= {gs}"
                )
                continue
            gs_key = f"gs{gs}"
            n_wiki_add = min(n_extra, len(wiki_filtered))
            expanded_term_names = bank_term_names + wiki_filtered[:n_wiki_add]
            expanded_bank = _pad_and_cat_3d(
                [bank_embs, wiki_embs[:n_wiki_add]], dim=0
            )
            if fixed_metric_denominator:
                expanded_label_stats = base_label_stats
                expanded_positive_indices = _map_positive_terms_to_bank_indices(
                    base_positive_terms,
                    expanded_term_names,
                )
            else:
                expanded_positive_indices, _, expanded_label_stats = _build_glossary_positive_indices(
                    sample_list,
                    expanded_term_names,
                    min_norm_chars=args.eval_glossary_match_min_norm_chars,
                )
            expanded_has_term_mask = torch.tensor(
                [bool(terms) for terms in base_positive_terms]
                if fixed_metric_denominator
                else [bool(indices) for indices in expanded_positive_indices],
                dtype=torch.bool,
            )
            expanded_logits = _score_eval_logits(
                speech_valid, expanded_bank, args, device, score_device=score_device
            )
            expanded_recall_logits = expanded_logits[row_keep]
            if fixed_metric_denominator:
                expanded_recall_positive_indices = _map_positive_terms_to_bank_indices(
                    recall_positive_terms,
                    expanded_term_names,
                )
            else:
                expanded_recall_positive_indices = [
                    expanded_positive_indices[i] for i in recall_sample_indices
                ]
            expanded_recall_positive_mask = _positive_indices_to_mask(
                expanded_recall_positive_indices,
                expanded_recall_logits.size(1),
            )
            gs_kp = min(k_primary, expanded_recall_logits.size(1))
            gs_ke = min(k_extra, expanded_recall_logits.size(1))
            gs_recall_p = _topk_recall_from_positive_mask(
                expanded_recall_logits, expanded_recall_positive_mask, gs_kp
            )
            gs_recall_e = _topk_recall_from_positive_mask(
                expanded_recall_logits, expanded_recall_positive_mask, gs_ke
            )
            metrics[f"{prefix}/recall@{gs_kp}_gs{gs}"] = gs_recall_p
            metrics[f"{prefix}/recall@{gs_ke}_gs{gs}"] = gs_recall_e
            metrics[f"{prefix}/{gs_key}_label_text_match_positive"] = expanded_label_stats[
                "n_text_match_positive"
            ]
            metrics[f"{prefix}/{gs_key}_label_any_positive"] = expanded_label_stats[
                "n_any_positive"
            ]
            metrics[f"{prefix}/{gs_key}_label_text_match_terms_skipped_short"] = (
                expanded_label_stats["n_text_match_terms_skipped_short"]
            )

            _dump_eval_miss_cases(
                args=args,
                eval_name=eval_name,
                bank_label=gs_key,
                logits=expanded_recall_logits,
                positive_indices=expanded_recall_positive_indices,
                sample_indices=recall_sample_indices,
                sample_list=sample_list,
                term_names=expanded_term_names,
                topk=gs_kp,
                global_step=global_step,
                epoch=epoch,
            )

            # ---- Optional: dump pos/neg sim distributions for this glossary size ----
            if dump_base:
                gs_out = os.path.join(dump_base, f"{eval_name}_gs{gs}.npz")
                gs_summary = _dump_sim_distributions(
                    gs_out, expanded_recall_logits, targets_t,
                    term_texts=kept_term_texts,
                    positive_mask=expanded_recall_positive_mask,
                )
                logger.info(
                    f"[SIM_DIST][{eval_name}][gs{gs} bank={expanded_logits.size(1)}] "
                    f"pos mean={gs_summary['pos_mean']:.3f} median={gs_summary['pos_median']:.3f} "
                    f"p10={gs_summary['pos_p10']:.3f} p90={gs_summary['pos_p90']:.3f}  "
                    f"neg_max mean={gs_summary['neg_max_mean']:.3f} p90={gs_summary['neg_max_p90']:.3f} "
                    f"neg_mean={gs_summary['neg_mean_mean']:.3f}  "
                    f"gap mean={gs_summary['gap_mean']:.3f} p10={gs_summary['gap_p10']:.3f}  "
                    f"-> {gs_out}"
                )

            gs_tau = (
                threshold_from_dev.get(gs_key) if threshold_from_dev else None
            )
            if eval_minimal:
                tm_gs: Dict[str, float] = {}
            else:
                tm_gs = _compute_threshold_metrics(
                    expanded_recall_logits,
                    targets_t,
                    threshold=gs_tau,
                    positive_mask=expanded_recall_positive_mask,
                )
                for k, v in tm_gs.items():
                    metrics[f"{prefix}/{k}_{gs_key}"] = v

            tcm_gs = _compute_tcm_threshold_metrics(
                expanded_recall_logits,
                targets_t,
                tcm_pos_threshold=args.tcm_pos_threshold,
                tcm_neg_threshold=args.tcm_neg_threshold,
                extra_thresholds=tcm_sweep if tcm_sweep else None,
                topk_for_filtered=gs_kp,
                minimal=eval_minimal,
                positive_mask=expanded_recall_positive_mask,
            )
            for k, v in tcm_gs.items():
                metrics[f"{prefix}/{k}_{gs_key}"] = v

            # Detection on expanded bank (all audio-ok chunks)
            det_metrics_gs: Dict[str, float] = {}
            full_logits_gs = None
            if eval_minimal:
                # In minimal mode: skip detection AUCs but still compute
                # no-term noise below if a sweep is requested, or raw no-term
                # top-K dumps for offline tau calibration.
                has_noterm_for_noise_gs = int(
                    (~expanded_has_term_mask[det_select]).sum().item()
                ) > 0
                if (
                    ((tcm_sweep and has_noterm_for_noise_gs) or dump_base)
                    and int(det_select.sum().item()) > 0
                    and int(expanded_has_term_mask[det_select].sum().item()) > 0
                    and speech_det is not None
                ):
                    full_logits_gs = _score_eval_logits(
                        speech_det,
                        expanded_bank,
                        args,
                        device,
                        score_device=score_device,
                    )
            elif (
                int(det_select.sum().item()) > 0
                and int(expanded_has_term_mask[det_select].sum().item()) > 0
            ):
                full_logits_gs = _score_eval_logits(
                    speech_det,
                    expanded_bank,
                    args,
                    device,
                    score_device=score_device,
                )
                det_target_tau_gs = (
                    gs_tau if gs_tau is not None else tm_gs.get("opt_threshold")
                )
                det_metrics_gs = _compute_detection_metrics(
                    all_logits=full_logits_gs,
                    labels=expanded_has_term_mask[det_select],
                    softmax_temp=args.temperature,
                    target_tau=det_target_tau_gs,
                )
                for k, v in det_metrics_gs.items():
                    metrics[f"{prefix}/{k}_{gs_key}"] = v

            # No-term noise for this glossary size (runs in both minimal + full
            # modes when a sweep is configured and no-term chunks exist).
            noise_gs: Dict[str, float] = {}
            if (
                tcm_sweep
                and full_logits_gs is not None
                and int((~expanded_has_term_mask[det_select]).sum().item()) > 0
            ):
                noise_gs = _compute_noterm_noise(
                    full_logits_gs,
                    expanded_has_term_mask[det_select],
                    taus=tcm_sweep,
                    topk=gs_kp,
                )
                for k, v in noise_gs.items():
                    metrics[f"{prefix}/{k}_{gs_key}"] = v

            if (
                dump_base
                and full_logits_gs is not None
                and int((~expanded_has_term_mask[det_select]).sum().item()) > 0
            ):
                no_term_gs_out = os.path.join(
                    dump_base, f"{eval_name}_gs{gs}_noterm_topk.npz"
                )
                no_term_gs_summary = _dump_noterm_topk_scores(
                    no_term_gs_out,
                    full_logits_gs,
                    expanded_has_term_mask[det_select],
                    topk=gs_kp,
                    term_names=expanded_term_names,
                    samples=det_samples,
                )
                logger.info(
                    f"[NOTERM_DIST][{eval_name}][gs{gs} bank={full_logits_gs.size(1)}] "
                    f"n_no_term={no_term_gs_summary.get('n_no_term', 0):.0f} "
                    f"top1_p90={no_term_gs_summary.get('noterm_top1_p90', float('nan')):.3f} "
                    f"top1_p95={no_term_gs_summary.get('noterm_top1_p95', float('nan')):.3f} "
                    f"top1_p99={no_term_gs_summary.get('noterm_top1_p99', float('nan')):.3f} "
                    f"-> {no_term_gs_out}"
                )

            actual_bank = expanded_bank.size(0)
            if eval_minimal:
                gs_log = (
                    f"gs{gs}(bank={actual_bank}): "
                    f"r@{gs_kp}={gs_recall_p:.4f} r@{gs_ke}={gs_recall_e:.4f}"
                )
            else:
                gs_log = (
                    f"gs{gs}(bank={actual_bank}): "
                    f"r@{gs_kp}={gs_recall_p:.4f} r@{gs_ke}={gs_recall_e:.4f} "
                    f"f2@tau={tm_gs.get('f2@tau', 0):.4f} "
                    f"P@tau={tm_gs.get('precision@tau', 0):.4f} "
                    f"gap={tm_gs.get('score_gap', 0):.4f} "
                    f"tcm_P/R@tbeta={tcm_gs.get('tcm_precision@tbeta', 0):.4f}"
                    f"/{tcm_gs.get('tcm_recall@tbeta', 0):.4f} "
                    f"tcm_P/R@talpha={tcm_gs.get('tcm_precision@talpha', 0):.4f}"
                    f"/{tcm_gs.get('tcm_recall@talpha', 0):.4f}"
                )
            if det_metrics_gs:
                gs_log += (
                    " det_auc="
                    f"{det_metrics_gs.get('det/max_sim_rocauc', float('nan')):.3f}/"
                    f"{det_metrics_gs.get('det/top1_zscore_rocauc', float('nan')):.3f}/"
                    f"{det_metrics_gs.get('det/softmax_prob_rocauc', float('nan')):.3f}/"
                    f"{det_metrics_gs.get('det/neg_entropy_rocauc', float('nan')):.3f}"
                )
                if "det/max_sim_fpr_at_target_tau" in det_metrics_gs:
                    gs_log += (
                        f" fpr@tau={det_metrics_gs['det/max_sim_fpr_at_target_tau']:.3f}"
                    )
            log_parts.append(gs_log)

            if tcm_sweep:
                for tau in tcm_sweep:
                    tau_f = float(tau)
                    tag = f"tau_{tau_f:.2f}".replace(".", "p")
                    noise_key = f"noterm_noise@top{gs_kp}_{tag}"
                    noise_val = noise_gs.get(noise_key, float("nan"))
                    if eval_minimal:
                        log_parts.append(
                            f"gs{gs}_sweep@{tau_f:.2f}: "
                            f"R={tcm_gs.get(f'topk{gs_kp}_chunk_any_positive_filtered_recall@{tag}', 0):.3f} "
                            f"P_mic={tcm_gs.get(f'topk{gs_kp}_filtered_precision_micro@{tag}', 0):.3f} "
                            f"P_mac={tcm_gs.get(f'topk{gs_kp}_filtered_precision_macro@{tag}', 0):.3f} "
                            f"kept={tcm_gs.get(f'topk{gs_kp}_avg_kept_if_pass@{tag}', 0):.2f} "
                            f"noise={noise_val:.2f}"
                        )
                    else:
                        log_parts.append(
                            f"gs{gs}_sweep@{tau_f:.2f}: "
                            f"R={tcm_gs.get(f'topk{gs_kp}_chunk_any_positive_filtered_recall@{tag}', 0):.3f} "
                            f"P_mic={tcm_gs.get(f'topk{gs_kp}_filtered_precision_micro@{tag}', 0):.3f} "
                            f"P_mac={tcm_gs.get(f'topk{gs_kp}_filtered_precision_macro@{tag}', 0):.3f} "
                            f"pass={tcm_gs.get(f'topk{gs_kp}_pass_rate@{tag}', 0):.3f} "
                            f"kept={tcm_gs.get(f'topk{gs_kp}_avg_kept_if_pass@{tag}', 0):.2f} "
                            f"noise={noise_val:.2f}"
                        )

            if gs > largest_gs:
                largest_gs = gs
                largest_gs_logits = expanded_recall_logits
                bank_idx_to_name = [""] * gt_bank_size
                for t, bidx in term_to_bank.items():
                    bank_idx_to_name[bidx] = t
                largest_gs_term_names = bank_idx_to_name + wiki_filtered[:n_wiki_add]

    # ---- Top-100 qualitative analysis on the largest glossary scale ----
    n_top100 = getattr(args, "eval_top100_samples", 0)
    if (
        n_top100 > 0
        and largest_gs_logits is not None
        and largest_gs_term_names is not None
    ):
        n_queries = len(targets)
        sample_count = min(n_top100, n_queries)
        sampled = random.sample(range(n_queries), sample_count)

        for si in sampled:
            gt_idx = targets_t[si].item()
            gt_term = largest_gs_term_names[gt_idx]
            query_logits = largest_gs_logits[si]
            sample_idx = valid_indices[row_keep[si]]
            sample_meta = sample_list[sample_idx] if sample_idx < len(sample_list) else {}
            utter_id = sample_meta.get("utter_id", "?")

            sorted_indices = query_logits.argsort(descending=True)
            gt_rank = int((sorted_indices == gt_idx).nonzero(as_tuple=False)[0].item()) + 1
            total_bank = len(largest_gs_term_names)

            topk_count = min(100, total_bank)
            top_indices = sorted_indices[:topk_count]
            top_scores = query_logits[top_indices]

            lines = [
                f"[TOP100] utter_id={utter_id} GT='{gt_term}' "
                f"gt_rank={gt_rank}/{total_bank} gs={largest_gs}"
            ]
            for rank, (idx_val, score) in enumerate(
                zip(top_indices.tolist(), top_scores.tolist()), 1
            ):
                term_name = largest_gs_term_names[idx_val]
                marker = ""
                if idx_val == gt_idx:
                    marker = "  <<<< GT >>>>"
                lines.append(f"  #{rank:3d}  sim={score:8.4f}  '{term_name}'{marker}")

            if gt_rank > topk_count:
                gt_score = query_logits[gt_idx].item()
                lines.append(
                    f"  ... GT not in top-{topk_count}, "
                    f"actual rank={gt_rank}, sim={gt_score:.4f}"
                )
            logger.info("\n".join(lines))

    elapsed = time.time() - t0
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
    if is_main:
        logger.info(
            f"[ENCODERS] audio_preset={args.audio_encoder_preset} "
            f"audio_type={args.audio_encoder_type} audio_model={args.audio_model_id} "
            f"audio_feature_extractor={args.audio_feature_extractor_id} "
            f"audio_input_dtype={args.audio_input_dtype} "
            f"text_preset={args.text_encoder_preset} text_model={args.text_model_id} "
            f"text_input_prefix={args.text_input_prefix!r}"
        )
    retriever = Qwen3OmniRetriever(
        model_id=args.audio_model_id,
        target_dim=args.target_dim,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        temperature=args.temperature,
        learn_temp=args.learn_temp,
        pooling_type=args.pooling_type,
        use_maxsim=args.use_maxsim,
        maxsim_windows=args.maxsim_windows,
        maxsim_stride=args.maxsim_stride,
        audio_encoder_type=args.audio_encoder_type,
        audio_hidden_dim=args.audio_hidden_dim,
    ).to(device)

    text_encoder = BgeM3TextEncoder(
        model_id=args.text_model_id,
        lora_rank=args.text_lora_rank,
        lora_alpha=args.text_lora_alpha,
        target_modules=args.text_lora_target_modules,
        full_finetune=args.text_full_finetune,
        sparse_weight=args.sparse_weight,
        text_pooling=args.text_pooling,
        use_colbert=getattr(args, "use_colbert", False),
    ).to(device)
    text_tokenizer = AutoTokenizer.from_pretrained(args.text_model_id)
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        args.audio_feature_extractor_id or args.audio_model_id
    )

    if world_size > 1:
        retriever_ddp_find_unused = args.audio_encoder_type == "wavlm"
        if retriever_ddp_find_unused and is_main:
            logger.info(
                "[DDP] WavLM retriever uses find_unused_parameters=True "
                "for GradCache multi-forward stability."
            )
        retriever = DDP(
            retriever,
            device_ids=[rank],
            find_unused_parameters=retriever_ddp_find_unused,
        )
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
    if hasattr(raw_text_encoder, "sparse_linear"):
        text_trainable_params.extend(
            p for p in raw_text_encoder.sparse_linear.parameters() if p.requires_grad
        )
    if hasattr(raw_text_encoder, "pool_gate"):
        text_trainable_params.extend(
            p for p in raw_text_encoder.pool_gate.parameters() if p.requires_grad
        )
    text_lr = args.text_lr if args.text_lr > 0 else (
        args.lr * DEFAULT_TEXT_LR_SCALE if args.text_full_finetune else args.lr
    )
    head_params = list(raw_retriever.projector.parameters())
    if hasattr(raw_retriever, "pooler"):
        head_params.extend(raw_retriever.pooler.parameters())
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
    pending_best_metric_value = None
    pending_best_metric_secondary_value = None
    pending_best_metric_key = None
    pending_best_metric_secondary_key = None

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
        pending_best_metric_value = ckpt.get("best_metric_value", None)
        pending_best_metric_secondary_value = ckpt.get(
            "best_metric_secondary_value", None
        )
        pending_best_metric_key = ckpt.get("best_metric_key", None)
        pending_best_metric_secondary_key = ckpt.get(
            "best_metric_secondary_key", None
        )
        if is_main:
            logger.info(
                f"[RESUME] {args.resume} epoch={start_epoch} step={global_step}"
            )

    # ---- Data ----
    train_samples: List[Dict] = []
    wiki_rank_skipped = 0
    train_exclude_terms: set = set()
    train_exclude_stats: List[Tuple[str, int, int]] = []
    train_term_key_excluded = 0
    noisy_kept = 0
    clean_kept = 0
    noisy_dropped = 0
    clean_dropped = 0
    noisy_rng = random.Random(42)
    if args.eval_only:
        if is_main:
            logger.info(
                "[EVAL_ONLY] Skipping train data load; using empty train set."
            )
    else:
        assert args.train_jsonl and os.path.isfile(args.train_jsonl), (
            f"--train_jsonl is required for training runs but not found: "
            f"{args.train_jsonl!r}"
        )
        if args.strict_train_eval_term_filter:
            train_exclude_terms, train_exclude_stats = load_train_exclude_term_keys(
                args.train_exclude_term_glossaries
            )
            if is_main and train_exclude_stats:
                logger.info(
                    "[DATA_LEAK_FILTER] strict=true loaded exact term_key exclusions: "
                    f"unique={len(train_exclude_terms):,} "
                    + " ".join(
                        f"path={path} raw={raw_count:,} new={new_count:,}"
                        for path, raw_count, new_count in train_exclude_stats
                    )
                )
        elif is_main and args.train_exclude_term_glossaries:
            logger.info(
                "[DATA_LEAK_FILTER] strict=false; not filtering train rows by eval "
                "glossary term_key. configured_glossaries="
                f"{len(args.train_exclude_term_glossaries)}"
            )
        with open(args.train_jsonl, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if args.train_limit and idx >= args.train_limit:
                    break
                try:
                    sample = json.loads(line)
                except Exception:
                    continue
                if train_exclude_terms:
                    term_key = _sample_term_key(sample)
                    if term_key and term_key in train_exclude_terms:
                        train_term_key_excluded += 1
                        continue
                if args.wiki_rank > 0:
                    p31_rank = sample.get("p31_rank", -1)
                    if p31_rank >= 0 and p31_rank >= args.wiki_rank:
                        wiki_rank_skipped += 1
                        continue
                audio_type = sample.get("audio_type", "")
                if audio_type in ("clean", "noisy") and args.noisy_ratio >= 0:
                    if audio_type == "noisy":
                        if noisy_rng.random() >= args.noisy_ratio:
                            noisy_dropped += 1
                            continue
                        noisy_kept += 1
                    else:
                        if noisy_rng.random() >= (1.0 - args.noisy_ratio):
                            clean_dropped += 1
                            continue
                        clean_kept += 1
                elif audio_type in ("clean", "noisy"):
                    if audio_type == "noisy":
                        noisy_kept += 1
                    else:
                        clean_kept += 1
                train_samples.append(sample)
        chunk_pos_stats = attach_chunk_positive_term_ids(train_samples)
        if is_main:
            logger.info(
                "[DATA] chunk-positive term ids: "
                f"groups={int(chunk_pos_stats['groups']):,} "
                f"multi_term_groups={int(chunk_pos_stats['multi_term_groups']):,} "
                f"({chunk_pos_stats['multi_term_group_rate']:.2%}) "
                f"rows_in_multi_term_groups={int(chunk_pos_stats['rows_with_multi_term_group']):,} "
                f"({chunk_pos_stats['rows_with_multi_term_group_rate']:.2%}) "
                f"max_terms_per_group={int(chunk_pos_stats['max_terms_per_group'])}"
            )
    if rank == 0:
        if args.wiki_rank > 0:
            logger.info(
                f"[DATA] wiki_rank={args.wiki_rank}: kept {len(train_samples):,}, "
                f"skipped {wiki_rank_skipped:,} wiki entries beyond rank cutoff"
            )
        if train_exclude_stats:
            logger.info(
                "[DATA_LEAK_FILTER] exact eval-glossary term_key filter: "
                f"excluded={train_term_key_excluded:,} kept={len(train_samples):,} "
                f"exclude_unique={len(train_exclude_terms):,}"
            )
        if noisy_kept + clean_kept + noisy_dropped + clean_dropped > 0:
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
    dev_samples = _limit_eval_samples(
        dev_samples,
        args.eval_sample_limit,
        args.eval_sample_seed,
        "dev",
        is_main=is_main,
    )

    per_rank_bs = max(1, args.batch_size // world_size)
    dataset: Optional[TermRAGDataset] = None
    sampler = None
    train_loader: Optional[DataLoader] = None
    if not args.eval_only:
        dataset = TermRAGDataset(
            train_samples,
            force_dummy_audio=args.force_dummy_audio,
            augment_synth=args.augment_synth,
            fixed_audio_samples=args.fixed_audio_samples,
        )
        sampler = (
            DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
            if world_size > 1
            else None
        )
        train_loader = DataLoader(
            dataset,
            batch_size=per_rank_bs,
            sampler=sampler,
            shuffle=(sampler is None),
            collate_fn=lambda b: collate_fn(
                b,
                feature_extractor,
                args.use_phoneme_append,
                args.fixed_audio_samples,
                args.audio_encoder_type,
            ),
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
            drop_last=True,
        )

    eval_loader = None
    if dev_samples:
        eval_dataset = TermRAGDataset(
            dev_samples,
            force_dummy_audio=args.force_dummy_audio,
            fixed_audio_samples=args.eval_fixed_audio_samples,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(
                b,
                feature_extractor,
                args.use_phoneme_append,
                args.eval_fixed_audio_samples,
                args.audio_encoder_type,
            ),
            num_workers=args.num_workers,
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
    acl_dev_samples = _limit_eval_samples(
        acl_dev_samples,
        args.acl_eval_sample_limit,
        args.eval_sample_seed + 1009,
        "acl6060",
        is_main=is_main,
    )

    acl_eval_loader: Optional[DataLoader] = None
    if acl_dev_samples:
        acl_eval_dataset = TermRAGDataset(
            acl_dev_samples,
            force_dummy_audio=args.force_dummy_audio,
            fixed_audio_samples=args.eval_fixed_audio_samples,
        )
        acl_eval_loader = DataLoader(
            acl_eval_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(
                b,
                feature_extractor,
                args.use_phoneme_append,
                args.eval_fixed_audio_samples,
                args.audio_encoder_type,
            ),
            num_workers=args.num_workers,
            pin_memory=True,
        )

    # ---- Tagged ACL6060 dev data (cross-domain eval) ----
    tagged_acl_dev_samples: List[Dict] = []
    if args.tagged_acl_dev_jsonl and os.path.isfile(args.tagged_acl_dev_jsonl):
        with open(args.tagged_acl_dev_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    tagged_acl_dev_samples.append(json.loads(line))
                except Exception:
                    continue
    tagged_acl_dev_samples = _limit_eval_samples(
        tagged_acl_dev_samples,
        args.tagged_acl_eval_sample_limit,
        args.eval_sample_seed + 1511,
        "tagged_acl",
        is_main=is_main,
    )

    tagged_acl_eval_loader: Optional[DataLoader] = None
    if tagged_acl_dev_samples:
        tagged_acl_eval_dataset = TermRAGDataset(
            tagged_acl_dev_samples,
            force_dummy_audio=args.force_dummy_audio,
            fixed_audio_samples=args.eval_fixed_audio_samples,
        )
        tagged_acl_eval_loader = DataLoader(
            tagged_acl_eval_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(
                b,
                feature_extractor,
                args.use_phoneme_append,
                args.eval_fixed_audio_samples,
                args.audio_encoder_type,
            ),
            num_workers=args.num_workers,
            pin_memory=True,
        )

    # ---- Medicine dev data (cross-domain eval) ----
    medicine_dev_samples: List[Dict] = []
    if args.medicine_dev_jsonl and os.path.isfile(args.medicine_dev_jsonl):
        with open(args.medicine_dev_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    medicine_dev_samples.append(json.loads(line))
                except Exception:
                    continue
    medicine_dev_samples = _limit_eval_samples(
        medicine_dev_samples,
        args.medicine_eval_sample_limit,
        args.eval_sample_seed + 2003,
        "medicine",
        is_main=is_main,
    )

    medicine_eval_loader: Optional[DataLoader] = None
    if medicine_dev_samples:
        medicine_eval_dataset = TermRAGDataset(
            medicine_dev_samples,
            force_dummy_audio=args.force_dummy_audio,
            fixed_audio_samples=args.eval_fixed_audio_samples,
        )
        medicine_eval_loader = DataLoader(
            medicine_eval_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(
                b,
                feature_extractor,
                args.use_phoneme_append,
                args.eval_fixed_audio_samples,
                args.audio_encoder_type,
            ),
            num_workers=args.num_workers,
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
    eval_metrics_terms: Optional[List[str]] = None
    if args.eval_metrics_glossary:
        eval_metrics_terms = _load_eval_wiki_terms(args.eval_metrics_glossary)
        if is_main:
            logger.info(
                f"[EVAL] Dev fixed metrics glossary loaded: "
                f"{len(eval_metrics_terms)} terms, source={args.eval_metrics_glossary}"
            )
    acl_eval_wiki_terms: Optional[List[str]] = eval_wiki_terms
    acl_eval_glossary_sizes: List[int] = eval_glossary_sizes
    acl_eval_metrics_terms: Optional[List[str]] = eval_metrics_terms
    if args.acl_eval_wiki_glossary:
        acl_eval_wiki_terms = _load_eval_wiki_terms(args.acl_eval_wiki_glossary)
        acl_eval_glossary_sizes = (
            args.acl_eval_glossary_sizes or eval_glossary_sizes
        )
        if is_main:
            logger.info(
                f"[EVAL] ACL wiki glossary loaded: "
                f"{len(acl_eval_wiki_terms)} terms, "
                f"glossary_sizes={acl_eval_glossary_sizes}"
            )
    elif args.acl_eval_glossary_sizes:
        acl_eval_glossary_sizes = args.acl_eval_glossary_sizes
        if is_main:
            logger.info(
                f"[EVAL] ACL glossary sizes override: "
                f"{acl_eval_glossary_sizes}"
            )
    if args.acl_eval_metrics_glossary:
        acl_eval_metrics_terms = _load_eval_wiki_terms(args.acl_eval_metrics_glossary)
        if is_main:
            logger.info(
                f"[EVAL] ACL fixed metrics glossary loaded: "
                f"{len(acl_eval_metrics_terms)} terms, "
                f"source={args.acl_eval_metrics_glossary}"
            )
    tagged_acl_eval_wiki_terms: Optional[List[str]] = acl_eval_wiki_terms
    tagged_acl_eval_glossary_sizes: List[int] = acl_eval_glossary_sizes
    tagged_acl_eval_metrics_terms: Optional[List[str]] = acl_eval_metrics_terms
    if args.tagged_acl_eval_wiki_glossary:
        tagged_acl_eval_wiki_terms = _load_eval_wiki_terms(
            args.tagged_acl_eval_wiki_glossary
        )
        tagged_acl_eval_glossary_sizes = (
            args.tagged_acl_eval_glossary_sizes
            or args.acl_eval_glossary_sizes
            or eval_glossary_sizes
        )
        if is_main:
            logger.info(
                f"[EVAL] Tagged ACL wiki glossary loaded: "
                f"{len(tagged_acl_eval_wiki_terms)} terms, "
                f"glossary_sizes={tagged_acl_eval_glossary_sizes}"
            )
    elif args.tagged_acl_eval_glossary_sizes:
        tagged_acl_eval_glossary_sizes = args.tagged_acl_eval_glossary_sizes
        if is_main:
            logger.info(
                f"[EVAL] Tagged ACL glossary sizes override: "
                f"{tagged_acl_eval_glossary_sizes}"
            )
    if args.tagged_acl_eval_metrics_glossary:
        tagged_acl_eval_metrics_terms = _load_eval_wiki_terms(
            args.tagged_acl_eval_metrics_glossary
        )
        if is_main:
            logger.info(
                f"[EVAL] Tagged ACL fixed metrics glossary loaded: "
                f"{len(tagged_acl_eval_metrics_terms)} terms, "
                f"source={args.tagged_acl_eval_metrics_glossary}"
            )
    medicine_eval_wiki_terms: Optional[List[str]] = eval_wiki_terms
    medicine_eval_glossary_sizes: List[int] = eval_glossary_sizes
    medicine_eval_metrics_terms: Optional[List[str]] = eval_metrics_terms
    if args.medicine_eval_wiki_glossary:
        medicine_eval_wiki_terms = _load_eval_wiki_terms(args.medicine_eval_wiki_glossary)
        medicine_eval_glossary_sizes = (
            args.medicine_eval_glossary_sizes or eval_glossary_sizes
        )
        if is_main:
            logger.info(
                f"[EVAL] Medicine wiki glossary loaded: "
                f"{len(medicine_eval_wiki_terms)} terms, "
                f"glossary_sizes={medicine_eval_glossary_sizes}"
            )
    elif args.medicine_eval_glossary_sizes:
        medicine_eval_glossary_sizes = args.medicine_eval_glossary_sizes
        if is_main:
            logger.info(
                f"[EVAL] Medicine glossary sizes override: "
                f"{medicine_eval_glossary_sizes}"
            )
    if args.medicine_eval_metrics_glossary:
        medicine_eval_metrics_terms = _load_eval_wiki_terms(
            args.medicine_eval_metrics_glossary
        )
        if is_main:
            logger.info(
                f"[EVAL] Medicine fixed metrics glossary loaded: "
                f"{len(medicine_eval_metrics_terms)} terms, "
                f"source={args.medicine_eval_metrics_glossary}"
            )
    full_eval_glossary_sizes: List[int] = args.full_eval_glossary_sizes or []
    full_eval_max_terms = max(full_eval_glossary_sizes) if full_eval_glossary_sizes else 0
    full_eval_enabled = (
        bool(args.full_eval_wiki_glossary)
        and bool(full_eval_glossary_sizes)
        and args.full_eval_every_n_evals > 0
    )

    # ---- Negative bank ----
    neg_bank: Optional[NegativeTermBank] = None
    neg_bank_rng = random.Random(42)
    use_neg_bank = (
        args.neg_bank_size > 0
        or args.hard_neg_k_per_sample > 0
    )
    if use_neg_bank:
        train_terms = sorted(
            {(s.get("term_key", "") or "").strip().lower() for s in train_samples}
            - {""}
        )
        assert len(train_terms) > 0, "No valid terms found for negative bank"

        neg_bank = NegativeTermBank(train_terms, device)
        if args.hard_neg_k_per_sample > 0:
            bank_mode = "hard_neg_per_sample"
        else:
            bank_mode = "random"
        if is_main:
            logger.info(
                f"[NEG_BANK] mode={bank_mode} "
                f"train_terms={len(train_terms)} "
                f"total_unique={neg_bank.size} "
                f"hard_neg_k_per_sample={args.hard_neg_k_per_sample} "
                f"random_sample={args.neg_bank_size} "
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
    scheduler_epochs = args.scheduler_epochs if args.scheduler_epochs > 0 else args.epochs
    total_steps = (len(train_loader) if train_loader is not None else 0) * scheduler_epochs
    warmup_steps = int(total_steps * DEFAULT_WARMUP_RATIO)

    if args.constant_lr > 0.0:
        assert not args.reset_scheduler, (
            "--constant_lr and --reset_scheduler are mutually exclusive"
        )
        constant_lr_map = {
            "audio_lora": args.constant_lr,
            "text_lora": args.constant_lr,
            "text_full": args.constant_lr,
            "head": args.constant_lr * DEFAULT_HEAD_LR_SCALE,
        }
        for pg in optimizer.param_groups:
            name = pg.get("name", "")
            assert name in constant_lr_map, (
                f"Unknown param group '{name}', expected one of {list(constant_lr_map)}"
            )
            pg["lr"] = constant_lr_map[name]
            if "initial_lr" in pg:
                pg["initial_lr"] = constant_lr_map[name]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
        if is_main:
            logger.info(
                f"[RESUME] constant_lr={args.constant_lr:.2e} (head_lr="
                f"{constant_lr_map['head']:.2e}); cosine scheduler disabled"
            )
    elif args.resume_cosine_decay_to_max_steps:
        assert args.resume, "--resume_cosine_decay_to_max_steps requires --resume"
        assert args.max_steps > global_step, (
            "--resume_cosine_decay_to_max_steps requires --max_steps greater "
            f"than the resumed global_step ({global_step})"
        )
        assert not args.reset_scheduler, (
            "--resume_cosine_decay_to_max_steps and --reset_scheduler are mutually exclusive"
        )
        decay_steps = max(1, int(args.max_steps - global_step))
        start_lrs = [float(pg["lr"]) for pg in optimizer.param_groups]
        for pg in optimizer.param_groups:
            # Make LambdaLR's base_lrs equal the checkpoint LR, not the original
            # peak LR stored in optimizer.initial_lr.
            pg["initial_lr"] = float(pg["lr"])

        def _resume_decay_lambda(step_idx: int) -> float:
            rel_step = min(max(int(step_idx), 0), decay_steps)
            return 0.5 * (1.0 + math.cos(math.pi * rel_step / decay_steps))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=_resume_decay_lambda,
        )
        if is_main:
            logger.info(
                f"[RESUME] cosine-decay continuation: start_lrs={start_lrs} "
                f"decay_steps={decay_steps} target_global_step={args.max_steps}; "
                "warmup disabled"
            )
    else:
        if args.reset_scheduler and pending_scheduler_state is not None:
            desired_lr_map = {
                "audio_lora": args.lr,
                "text_lora": text_lr,
                "text_full": text_lr,
                "head": args.lr * DEFAULT_HEAD_LR_SCALE,
            }
            for pg in optimizer.param_groups:
                name = pg.get("name", "")
                assert name in desired_lr_map, (
                    f"Unknown param group '{name}', expected one of {list(desired_lr_map)}"
                )
                pg["lr"] = desired_lr_map[name]
                if "initial_lr" in pg:
                    pg["initial_lr"] = desired_lr_map[name]

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max(1, total_steps),
        )
        if pending_scheduler_state and not args.reset_scheduler:
            try:
                scheduler.load_state_dict(pending_scheduler_state)
            except Exception as exc:
                logger.warning(f"[RESUME] scheduler load failed, stepping to global_step: {exc}")
                for _ in range(global_step):
                    scheduler.step()
        elif pending_scheduler_state and args.reset_scheduler:
            for _ in range(global_step):
                scheduler.step()
            resumed_lr = optimizer.param_groups[0]["lr"]
            if is_main:
                logger.info(
                    f"[RESUME] reset_scheduler=True: fresh cosine schedule "
                    f"(peak_lr={args.lr}, total_steps={total_steps}, warmup={warmup_steps}), "
                    f"stepped to global_step={global_step}, current_lr={resumed_lr:.2e}"
                )
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

            run_tags, tag_changes = build_wandb_tags(args)
            if tag_changes:
                logger.warning(
                    "[WANDB] shortened overlong tags before init: %s",
                    "; ".join(f"{old} -> {new}" for old, new in tag_changes),
                )
            run_notes = load_and_validate_run_notes(args.notes_file)
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_exp_name,
                config=vars(args),
                tags=run_tags,
                notes=run_notes,
                save_code=True,
            )
            if tag_changes:
                wandb_run.config.update(
                    {
                        "wandb_tag_shortening": [
                            {"original": old, "safe": new}
                            for old, new in tag_changes
                        ]
                    },
                    allow_val_change=True,
                )
            if args.baseline_run_ids:
                wandb_run.config.update(
                    {"baseline_run_ids": list(args.baseline_run_ids)},
                    allow_val_change=True,
                )
            wandb.define_metric("train/step")
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("eval_dev/step")
            wandb.define_metric("eval_dev/*", step_metric="eval_dev/step")
            wandb.define_metric("eval_acl6060/step")
            wandb.define_metric("eval_acl6060/*", step_metric="eval_acl6060/step")
            wandb.define_metric("eval_tagged_acl/step")
            wandb.define_metric("eval_tagged_acl/*", step_metric="eval_tagged_acl/step")
            wandb.define_metric("eval_medicine/step")
            wandb.define_metric("eval_medicine/*", step_metric="eval_medicine/step")
        except Exception as exc:
            logger.error(f"[WANDB] init failed: {exc}")
            raise RuntimeError(
                "W&B init failed while --enable_wandb is set; aborting per "
                "experiment tracking rules."
            ) from exc

    recent_ckpts: List[str] = []
    best_metric_value = float("-inf")
    best_metric_key = args.best_metric or ""
    best_metric_secondary_value = float("-inf")
    best_metric_secondary_key = args.best_metric_secondary or ""
    last_auto_full_eval_step = -1
    last_latest_checkpoint_step = -1

    def latest_checkpoint_payload(epoch_value: int, step_value: int) -> Dict:
        return {
            "model_state_dict": raw_retriever.state_dict(),
            "text_model_state_dict": raw_text_encoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "global_step": step_value,
            "epoch": epoch_value,
            "args": vars(args),
            "best_metric_key": best_metric_key,
            "best_metric_secondary_key": best_metric_secondary_key,
            "best_metric_value": best_metric_value,
            "best_metric_secondary_value": best_metric_secondary_value,
        }

    should_restore_best_metric = (
        pending_best_metric_value is not None
        and not args.reset_best_on_resume
        and pending_best_metric_key == best_metric_key
    )
    if should_restore_best_metric:
        best_metric_value = pending_best_metric_value
        if is_main:
            logger.info(
                f"[RESUME] Restored best_metric_value={best_metric_value:.4f} "
                f"for metric={best_metric_key or '<none>'}"
            )
    elif pending_best_metric_value is not None and is_main:
        reason = "reset requested" if args.reset_best_on_resume else (
            f"checkpoint metric={pending_best_metric_key!r} "
            f"!= current metric={best_metric_key!r}"
        )
        logger.info(f"[RESUME] Not restoring primary best value: {reason}")

    should_restore_best_metric_secondary = (
        pending_best_metric_secondary_value is not None
        and not args.reset_best_on_resume
        and pending_best_metric_secondary_key == best_metric_secondary_key
    )
    if should_restore_best_metric_secondary:
        best_metric_secondary_value = pending_best_metric_secondary_value
        if is_main:
            logger.info(
                f"[RESUME] Restored best_metric_secondary_value="
                f"{best_metric_secondary_value:.4f} "
                f"for metric={best_metric_secondary_key or '<none>'}"
            )
    elif pending_best_metric_secondary_value is not None and is_main:
        reason = "reset requested" if args.reset_best_on_resume else (
            f"checkpoint secondary metric={pending_best_metric_secondary_key!r} "
            f"!= current secondary metric={best_metric_secondary_key!r}"
        )
        logger.info(f"[RESUME] Not restoring secondary best value: {reason}")

    hn_start_step = global_step
    train_wall_start = time.time()
    walltime_stop = False
    primary_no_improve_evals = 0

    if is_main:
        ohnm_desc = "OFF"
        if args.online_hard_neg_k > 0:
            ohnm_desc = (
                f"K={args.online_hard_neg_k} "
                f"easy_w={args.easy_neg_weight:.2f}→{args.easy_neg_weight_final:.2f} "
                f"decay={args.hn_decay_steps}steps"
            )
        logger.info(
            f"[SETUP] train={len(train_samples)} dev={len(dev_samples)} "
            f"acl_dev={len(acl_dev_samples)} "
            f"tagged_acl_dev={len(tagged_acl_dev_samples)} "
            f"medicine_dev={len(medicine_dev_samples)} "
            f"world_size={world_size} per_rank_bs={per_rank_bs} "
            f"total_steps={total_steps} warmup_steps={warmup_steps} "
            f"neg_bank={'ON' if neg_bank else 'OFF'} "
            f"grad_cache={'chunk=' + str(args.grad_cache_chunk_size) if args.grad_cache_chunk_size > 0 else 'OFF'} "
            f"pooling={args.pooling_type} "
            f"maxsim={args.use_maxsim} "
            f"{'maxsim_win=' + str(args.maxsim_windows) + '_s' + str(args.maxsim_stride) + ' ' if args.use_maxsim else ''}"
            f"margin={args.margin} "
            f"o_hnm={ohnm_desc}"
        )
        logger.info(
            f"[SETUP] eval_metric_denominator={args.eval_metric_denominator} "
            "(fixed_raw keeps strict raw metrics denominator while retriever "
            "glossary sizes change)"
        )
        # Sanity check for wall-time-capped runs.  The cosine schedule's warmup
        # fraction is tied to total_steps = steps_per_epoch * epochs, NOT to
        # max_train_seconds.  If epochs is large while wall-time is short, the
        # LR will still be in warmup when wall-time expires and the model
        # effectively does not train.  We fail loudly rather than silently
        # producing a near-untrained checkpoint.
        if args.max_train_seconds > 0 and args.epochs > 2:
            raise ValueError(
                f"[SETUP] max_train_seconds={args.max_train_seconds} > 0 with "
                f"epochs={args.epochs} > 2: cosine schedule spans "
                f"total_steps={total_steps} (warmup={warmup_steps}) which almost "
                f"certainly does not fit in the wall-time budget, leaving LR "
                f"stuck in warmup.  In wall-time-capped mode set epochs to the "
                f"number of epochs you actually expect to finish (usually 1)."
            )
        logger.info(
            f"[SETUP] text_encoder: "
            f"full_finetune={args.text_full_finetune} text_lr={text_lr:.2e} "
            f"phoneme_append={args.use_phoneme_append} "
            f"sparse_weight={args.sparse_weight} "
            f"text_pooling={args.text_pooling} "
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
        if args.early_stop_best_patience_evals > 0:
            logger.info(
                "[SETUP] Early stop: stop after "
                f"{args.early_stop_best_patience_evals} consecutive evals "
                f"without improving {best_metric_key or '<none>'}"
            )
        if best_metric_secondary_key:
            logger.info(
                f"[SETUP] Secondary best checkpoint metric: "
                f"{best_metric_secondary_key}"
            )
        if eval_glossary_sizes:
            logger.info(f"[SETUP] Eval glossary sizes: {eval_glossary_sizes}")
        if acl_eval_loader is not None and (
            acl_eval_wiki_terms is not eval_wiki_terms
            or acl_eval_glossary_sizes != eval_glossary_sizes
        ):
            logger.info(
                f"[SETUP] ACL eval glossary sizes: {acl_eval_glossary_sizes}"
            )
        if tagged_acl_eval_loader is not None and (
            tagged_acl_eval_wiki_terms is not acl_eval_wiki_terms
            or tagged_acl_eval_glossary_sizes != acl_eval_glossary_sizes
        ):
            logger.info(
                f"[SETUP] Tagged ACL eval glossary sizes: "
                f"{tagged_acl_eval_glossary_sizes}"
            )
        if medicine_eval_loader is not None and (
            medicine_eval_wiki_terms is not eval_wiki_terms
            or medicine_eval_glossary_sizes != eval_glossary_sizes
        ):
            logger.info(
                f"[SETUP] Medicine eval glossary sizes: {medicine_eval_glossary_sizes}"
            )
        if full_eval_enabled:
            logger.info(
                f"[SETUP] Sparse full eval: name={args.full_eval_name} "
                f"every {args.full_eval_every_n_evals} evals "
                f"glossary_sizes={full_eval_glossary_sizes} "
                f"source={args.full_eval_wiki_glossary}"
            )

    # ==================== Eval-only short-circuit ====================
    if args.eval_only:
        assert (
            eval_loader is not None
            or acl_eval_loader is not None
            or tagged_acl_eval_loader is not None
            or medicine_eval_loader is not None
        ), (
            "--eval_only requires at least one of --dev_jsonl / --acl_dev_jsonl "
            "/ --tagged_acl_dev_jsonl / --medicine_dev_jsonl to be provided"
        )
        if is_main:
            logger.info(
                "[EVAL_ONLY] Running one-shot evaluation on configured dev/ACL/tagged-ACL/medicine sets, then exiting."
            )
            dev_metrics: Dict[str, float] = {}
            if eval_loader is not None:
                dev_metrics = run_sample_eval(
                    raw_retriever,
                    raw_text_encoder,
                    text_tokenizer,
                    eval_loader,
                    device,
                    args,
                    global_step,
                    start_epoch,
                    wandb_run,
                    eval_name="dev",
                    wiki_terms=eval_wiki_terms,
                    glossary_sizes=eval_glossary_sizes,
                    metrics_terms=eval_metrics_terms,
                )

            dev_thresholds: Dict[str, float] = {}
            dev_prefix = "eval_dev"
            if f"{dev_prefix}/opt_threshold" in dev_metrics:
                dev_thresholds["base"] = dev_metrics[
                    f"{dev_prefix}/opt_threshold"
                ]
            for gs in (eval_glossary_sizes or []):
                gs_key = f"gs{gs}"
                tau_key = f"{dev_prefix}/opt_threshold_{gs_key}"
                if tau_key in dev_metrics:
                    dev_thresholds[gs_key] = dev_metrics[tau_key]

            if acl_eval_loader is not None:
                run_sample_eval(
                    raw_retriever,
                    raw_text_encoder,
                    text_tokenizer,
                    acl_eval_loader,
                    device,
                    args,
                    global_step,
                    start_epoch,
                    wandb_run,
                    eval_name="acl6060",
                    wiki_terms=acl_eval_wiki_terms,
                    glossary_sizes=acl_eval_glossary_sizes,
                    metrics_terms=acl_eval_metrics_terms,
                    threshold_from_dev=dev_thresholds or None,
                )

            if tagged_acl_eval_loader is not None:
                run_sample_eval(
                    raw_retriever,
                    raw_text_encoder,
                    text_tokenizer,
                    tagged_acl_eval_loader,
                    device,
                    args,
                    global_step,
                    start_epoch,
                    wandb_run,
                    eval_name="tagged_acl",
                    wiki_terms=tagged_acl_eval_wiki_terms,
                    glossary_sizes=tagged_acl_eval_glossary_sizes,
                    metrics_terms=tagged_acl_eval_metrics_terms,
                    threshold_from_dev=dev_thresholds or None,
                )

            if medicine_eval_loader is not None:
                run_sample_eval(
                    raw_retriever,
                    raw_text_encoder,
                    text_tokenizer,
                    medicine_eval_loader,
                    device,
                    args,
                    global_step,
                    start_epoch,
                    wandb_run,
                    eval_name="medicine",
                    wiki_terms=medicine_eval_wiki_terms,
                    glossary_sizes=medicine_eval_glossary_sizes,
                    metrics_terms=medicine_eval_metrics_terms,
                    threshold_from_dev=dev_thresholds or None,
                )

            if wandb_run is not None:
                finalize_wandb_run(
                    wandb_run,
                    success=True,
                    verdict=args.run_verdict,
                )
                wandb_run.finish()

        if world_size > 1:
            dist.barrier()
            dist.destroy_process_group()
        return

    # ==================== Optional torch.profiler setup ====================
    # Rank-0-only profile. All ranks run the same kernels under DDP so rank 0
    # is representative. Enable with --profile_out_dir <dir>; outputs Chrome
    # trace JSON (tensorboard_trace_handler) + key_averages.txt summary.
    _profiler = None
    if args.profile_out_dir and is_main:
        from torch.profiler import (
            profile as _torch_profile,
            schedule as _torch_prof_schedule,
            tensorboard_trace_handler,
            ProfilerActivity,
        )
        os.makedirs(args.profile_out_dir, exist_ok=True)
        _sch = [int(x) for x in args.profile_schedule.split(",")]
        if len(_sch) != 4:
            raise ValueError(
                f"--profile_schedule must be 'wait,warmup,active,repeat' "
                f"(got {args.profile_schedule!r})"
            )
        _profiler = _torch_profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=_torch_prof_schedule(
                wait=_sch[0], warmup=_sch[1], active=_sch[2], repeat=_sch[3]
            ),
            on_trace_ready=tensorboard_trace_handler(args.profile_out_dir),
            record_shapes=True,
            with_stack=False,
        )
        _profiler.start()
        logger.info(
            f"[PROFILE] torch.profiler enabled schedule=wait{_sch[0]}+"
            f"warmup{_sch[1]}+active{_sch[2]}x{_sch[3]} -> {args.profile_out_dir}"
        )

    # ==================== Training loop ====================
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs):
        last_epoch = epoch
        if walltime_stop:
            break
        retriever.train()
        text_encoder.train()
        if sampler is not None:
            sampler.set_epoch(epoch)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not is_main)
        for batch in pbar:
            if (
                args.max_train_seconds > 0
                and (time.time() - train_wall_start) >= args.max_train_seconds
            ):
                if is_main:
                    elapsed = time.time() - train_wall_start
                    logger.info(
                        f"[WALLTIME] max_train_seconds={args.max_train_seconds} "
                        f"reached at elapsed={elapsed:.0f}s step={global_step}, "
                        f"stopping."
                    )
                walltime_stop = True
                break
            if args.max_steps > 0 and global_step >= args.max_steps:
                if is_main:
                    logger.info(
                        f"[MAX_STEPS] max_steps={args.max_steps} reached at "
                        f"step={global_step}, stopping."
                    )
                walltime_stop = True
                break
            _step_t0 = time.time()
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
                neg_bank.refresh(
                    raw_text_encoder,
                    text_tokenizer,
                    device,
                    use_phoneme_append=args.use_phoneme_append,
                    text_input_prefix=args.text_input_prefix,
                )
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
                    use_phoneme_append=args.use_phoneme_append,
                    text_input_prefix=args.text_input_prefix,
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
                stop_after_eval = False
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
                        metrics_terms=eval_metrics_terms,
                    )
                    all_eval_metrics.update(dev_metrics)

                    eval_index = (
                        global_step // args.eval_steps_sample
                        if args.eval_steps_sample > 0
                        else 0
                    )
                    run_full_eval = (
                        full_eval_enabled
                        and eval_index > 0
                        and eval_index % args.full_eval_every_n_evals == 0
                    )
                    if run_full_eval:
                        logger.info(
                            f"[FULL_EVAL] step={global_step} "
                            f"name={args.full_eval_name} "
                            f"glossary_sizes={full_eval_glossary_sizes}"
                        )
                        full_eval_wiki_terms = _load_eval_wiki_terms(
                            args.full_eval_wiki_glossary,
                            max_terms=full_eval_max_terms,
                        )
                        # Full eval is for recall tracking and checkpointing.
                        # Avoid no-term sweep diagnostics here because they
                        # materialize a large all-query x full-bank matrix.
                        original_tcm_sweep_thresholds = args.tcm_sweep_thresholds
                        args.tcm_sweep_thresholds = []
                        try:
                            full_eval_metrics = run_sample_eval(
                                raw_retriever,
                                raw_text_encoder,
                                text_tokenizer,
                                eval_loader,
                                device,
                                args,
                                global_step,
                                epoch,
                                wandb_run,
                                eval_name=args.full_eval_name,
                                wiki_terms=full_eval_wiki_terms,
                                glossary_sizes=full_eval_glossary_sizes,
                                metrics_terms=eval_metrics_terms,
                            )
                        finally:
                            args.tcm_sweep_thresholds = original_tcm_sweep_thresholds
                        all_eval_metrics.update(full_eval_metrics)
                        del full_eval_wiki_terms
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    dev_thresholds: Dict[str, float] = {}
                    dev_prefix = "eval_dev"
                    if f"{dev_prefix}/opt_threshold" in dev_metrics:
                        dev_thresholds["base"] = dev_metrics[
                            f"{dev_prefix}/opt_threshold"
                        ]
                    for gs in (eval_glossary_sizes or []):
                        gs_key = f"gs{gs}"
                        tau_key = f"{dev_prefix}/opt_threshold_{gs_key}"
                        if tau_key in dev_metrics:
                            dev_thresholds[gs_key] = dev_metrics[tau_key]

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
                            wiki_terms=acl_eval_wiki_terms,
                            glossary_sizes=acl_eval_glossary_sizes,
                            metrics_terms=acl_eval_metrics_terms,
                            threshold_from_dev=dev_thresholds or None,
                        )
                        all_eval_metrics.update(acl_metrics)

                    if tagged_acl_eval_loader is not None:
                        tagged_acl_metrics = run_sample_eval(
                            raw_retriever,
                            raw_text_encoder,
                            text_tokenizer,
                            tagged_acl_eval_loader,
                            device,
                            args,
                            global_step,
                            epoch,
                            wandb_run,
                            eval_name="tagged_acl",
                            wiki_terms=tagged_acl_eval_wiki_terms,
                            glossary_sizes=tagged_acl_eval_glossary_sizes,
                            metrics_terms=tagged_acl_eval_metrics_terms,
                            threshold_from_dev=dev_thresholds or None,
                        )
                        all_eval_metrics.update(tagged_acl_metrics)

                    if medicine_eval_loader is not None:
                        medicine_metrics = run_sample_eval(
                            raw_retriever,
                            raw_text_encoder,
                            text_tokenizer,
                            medicine_eval_loader,
                            device,
                            args,
                            global_step,
                            epoch,
                            wandb_run,
                            eval_name="medicine",
                            wiki_terms=medicine_eval_wiki_terms,
                            glossary_sizes=medicine_eval_glossary_sizes,
                            metrics_terms=medicine_eval_metrics_terms,
                            threshold_from_dev=dev_thresholds or None,
                        )
                        all_eval_metrics.update(medicine_metrics)

                    # Best checkpoint tracking
                    primary_metric_seen = (
                        best_metric_key and best_metric_key in all_eval_metrics
                    )
                    primary_metric_value = (
                        all_eval_metrics[best_metric_key]
                        if primary_metric_seen
                        else None
                    )
                    primary_improved = (
                        primary_metric_value is not None
                        and primary_metric_value > best_metric_value
                    )
                    if primary_improved:
                        best_metric_value = float(primary_metric_value)
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
                                "best_metric_secondary_key": best_metric_secondary_key,
                                "best_metric_value": best_metric_value,
                                "best_metric_secondary_value": best_metric_secondary_value,
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
                        auto_min_delta = max(
                            0, int(getattr(args, "auto_full_eval_min_step_delta", 0))
                        )
                        should_submit_auto_eval = (
                            getattr(args, "auto_full_eval_on_best", False)
                            and (
                                last_auto_full_eval_step < 0
                                or global_step - last_auto_full_eval_step
                                >= auto_min_delta
                            )
                        )
                        if should_submit_auto_eval:
                            job_id = _submit_auto_full_eval(
                                args=args,
                                checkpoint_path=best_path,
                                global_step=global_step,
                                metric_key=best_metric_key,
                                metric_value=best_metric_value,
                                wandb_run=wandb_run,
                                logger=logger,
                            )
                            if job_id is not None:
                                last_auto_full_eval_step = global_step

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
                        secondary_suffix = _metric_key_to_checkpoint_suffix(
                            best_metric_secondary_key
                        )
                        best_secondary_path = args.save_path.replace(
                            ".pt", f"_best_{secondary_suffix}.pt"
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
                                "best_metric_key": best_metric_key,
                                "best_metric_secondary_key": best_metric_secondary_key,
                                "best_metric_value": best_metric_value,
                                "best_metric_secondary_value": best_metric_secondary_value,
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

                    if getattr(args, "save_latest_on_eval", False):
                        latest_path = args.save_path.replace(".pt", "_latest.pt")
                        _atomic_torch_save(
                            latest_checkpoint_payload(epoch, global_step),
                            latest_path,
                        )
                        last_latest_checkpoint_step = global_step
                        logger.info(
                            f"[LATEST] eval checkpoint step={global_step} -> {latest_path}"
                        )
                        if wandb_run is not None:
                            wandb_run.log(
                                {"latest_eval_checkpoint/step": global_step},
                                step=global_step,
                            )

                    if args.early_stop_best_patience_evals > 0:
                        if primary_metric_seen:
                            if primary_improved:
                                primary_no_improve_evals = 0
                                logger.info(
                                    "[EARLY_STOP] primary best improved; "
                                    "no-improve eval counter reset."
                                )
                            else:
                                primary_no_improve_evals += 1
                                logger.info(
                                    "[EARLY_STOP] "
                                    f"{best_metric_key}={primary_metric_value:.4f} "
                                    f"did not improve best={best_metric_value:.4f}; "
                                    f"no_improve_evals={primary_no_improve_evals}/"
                                    f"{args.early_stop_best_patience_evals}"
                                )
                            if (
                                primary_no_improve_evals
                                >= args.early_stop_best_patience_evals
                            ):
                                logger.info(
                                    "[EARLY_STOP] patience reached; stopping after "
                                    f"eval at step={global_step}."
                                )
                                stop_after_eval = True
                        else:
                            logger.warning(
                                "[EARLY_STOP] primary metric "
                                f"{best_metric_key!r} missing from eval metrics; "
                                "not counting this eval toward patience."
                            )

                if world_size > 1:
                    dist.barrier()
                    stop_tensor = torch.tensor(
                        [1 if stop_after_eval else 0],
                        dtype=torch.int32,
                        device=device,
                    )
                    dist.broadcast(stop_tensor, src=0)
                    stop_after_eval = bool(stop_tensor.item())
                if stop_after_eval:
                    walltime_stop = True
                    break

            # Checkpoint
            if is_main and global_step % args.save_steps == 0:
                ckpt_path = args.save_path.replace(".pt", f"_step_{global_step}.pt")
                torch.save(
                    latest_checkpoint_payload(epoch, global_step),
                    ckpt_path,
                )
                recent_ckpts.append(ckpt_path)
                logger.info(f"[CHECKPOINT] saved={ckpt_path}")
                while len(recent_ckpts) > args.keep_checkpoints:
                    old = recent_ckpts.pop(0)
                    if os.path.exists(old):
                        os.remove(old)
                        logger.info(f"[CHECKPOINT] removed_old={old}")

            if (
                is_main
                and args.save_latest_steps > 0
                and global_step % args.save_latest_steps == 0
                and global_step != last_latest_checkpoint_step
            ):
                latest_path = args.save_path.replace(".pt", "_latest.pt")
                _atomic_torch_save(
                    latest_checkpoint_payload(epoch, global_step),
                    latest_path,
                )
                last_latest_checkpoint_step = global_step
                logger.info(
                    f"[LATEST] step checkpoint step={global_step} -> {latest_path}"
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {"latest_checkpoint/step": global_step},
                        step=global_step,
                    )

            # ---- Compute current easy_neg_weight (alpha) for soft O-HNM ----
            if args.online_hard_neg_k > 0 and args.hn_decay_steps > 0:
                hn_progress = min(1.0, (global_step - hn_start_step) / args.hn_decay_steps)
                current_easy_neg_weight = (
                    args.easy_neg_weight
                    + (args.easy_neg_weight_final - args.easy_neg_weight) * hn_progress
                )
            else:
                current_easy_neg_weight = args.easy_neg_weight

            # ---- Anneal softmax tau for MaxSim (from initial -> min over training) ----
            if args.maxsim_agg == "softmax" and total_steps > 0:
                tau_progress = min(1.0, global_step / total_steps)
                args.maxsim_softmax_tau = (
                    MAXSIM_SOFTMAX_TAU_INIT
                    + (MAXSIM_SOFTMAX_TAU_MIN - MAXSIM_SOFTMAX_TAU_INIT) * tau_progress
                )

            # ---- TCM warmup (must be computed BEFORE both grad-cache and
            # non-grad-cache paths so the value is available for logging) ----
            _base_tcm_pos_w, _base_tcm_neg_w = _resolve_tcm_branch_weights_from_args(args)
            _tcm_warmup = getattr(args, "tcm_warmup_steps", 0)
            _tcm_warmup_scale = 1.0
            if (_base_tcm_pos_w > 0.0 or _base_tcm_neg_w > 0.0) and _tcm_warmup > 0 and global_step < _tcm_warmup:
                _tcm_warmup_scale = global_step / _tcm_warmup
            _effective_tcm_pos_weight = _base_tcm_pos_w * _tcm_warmup_scale
            _effective_tcm_neg_weight = _base_tcm_neg_w * _tcm_warmup_scale
            _tcm_enabled = _base_tcm_pos_w > 0.0 or _base_tcm_neg_w > 0.0

            # ---- Forward + Backward ----
            if args.grad_cache_chunk_size > 0:
                with _record_function("train_step/gradcache"):
                    loss_outputs, hard_neg_count = gradcache_train_step(
                        batch=batch,
                        retriever=retriever,
                        text_encoder=text_encoder,
                        text_tokenizer=text_tokenizer,
                        device=device,
                        scaler=scaler,
                        optimizer=optimizer,
                        chunk_size=args.grad_cache_chunk_size,
                        world_size=world_size,
                        neg_bank=neg_bank,
                        neg_bank_rng=neg_bank_rng,
                        glossary_neg_embs=glossary_neg_embs,
                        glossary_neg_term_ids=glossary_neg_term_ids,
                        args=args,
                        easy_neg_weight=current_easy_neg_weight,
                        effective_tcm_pos_weight=_effective_tcm_pos_weight,
                        effective_tcm_neg_weight=_effective_tcm_neg_weight,
                    )
                total_loss = loss_outputs["total"]
                logit_scale = (
                    retriever.module.logit_scale.exp()
                    if world_size > 1
                    else retriever.logit_scale.exp()
                )
                scheduler.step()
            else:
                feats, flens = _move_audio_batch_to_device(
                    batch, device, args, non_blocking=True
                )
                texts = batch["term_texts"]
                valid = batch["valid_mask"].to(device, non_blocking=True)
                positive_term_ids = batch["positive_term_ids"].to(device, non_blocking=True)
                positive_term_mask = batch["positive_term_mask"].to(device, non_blocking=True)
                mfa_starts_nb = batch.get("mfa_term_starts")
                mfa_ends_nb = batch.get("mfa_term_ends")
                if mfa_starts_nb is not None:
                    mfa_starts_nb = mfa_starts_nb.to(device, non_blocking=True)
                if mfa_ends_nb is not None:
                    mfa_ends_nb = mfa_ends_nb.to(device, non_blocking=True)
                samples = batch["samples"]

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    speech_embs = retriever(feats, flens)
                    tok = _tokenize_texts(
                        text_tokenizer,
                        texts,
                        device,
                        text_input_prefix=args.text_input_prefix,
                    )
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
                    ps_embs, ps_tids = None, None
                    hard_neg_count = 0
                    if neg_bank is not None and neg_bank.embeddings is not None:
                        if getattr(args, "hard_neg_k_per_sample", 0) > 0:
                            ps_embs, ps_tids, hard_neg_count = (
                                neg_bank.mine_hard_negatives_per_sample(
                                    speech_embs,
                                    term_ids,
                                    valid,
                                    positive_term_ids,
                                    positive_term_mask,
                                    args.hard_neg_k_per_sample,
                                )
                            )
                            ps_embs = ps_embs.to(torch.bfloat16)
                        elif args.neg_bank_size > 0:
                            nb_embs, nb_tids = neg_bank.sample(
                                args.neg_bank_size, neg_bank_rng
                            )
                        if nb_embs is not None:
                            nb_embs = nb_embs.to(torch.bfloat16)

                    # Glossary negatives: append full glossary embeddings
                    if glossary_neg_embs is not None:
                        gn_embs = glossary_neg_embs.to(device, non_blocking=True).to(torch.bfloat16)
                        gn_tids = glossary_neg_term_ids.to(device, non_blocking=True)
                        if nb_embs is not None:
                            nb_embs = torch.cat([nb_embs, gn_embs], dim=0)
                            nb_tids = torch.cat([nb_tids, gn_tids], dim=0)
                        else:
                            nb_embs = gn_embs
                            nb_tids = gn_tids

                    raw_ret = retriever.module if world_size > 1 else retriever
                    loss_ws_nb = None
                    loss_we_nb = None
                    loss_ms_nb = None
                    loss_me_nb = None
                    if getattr(args, "mfa_supervised_maxsim", False) and mfa_starts_nb is not None:
                        W_nb = speech_embs.shape[1] if speech_embs.ndim == 3 else 0
                        if W_nb > 0:
                            T_enc_nb = _infer_encoder_frames(
                                raw_ret.maxsim_windows, raw_ret.maxsim_stride, W_nb,
                            )
                            # Cached on-device build (see _get_window_time_ranges_on).
                            loss_ws_nb, loss_we_nb = _get_window_time_ranges_on(
                                raw_ret.maxsim_windows,
                                raw_ret.maxsim_stride,
                                T_enc_nb,
                                device,
                                frame_sec=float(args.fixed_audio_seconds) / float(T_enc_nb),
                            )
                            loss_ms_nb = mfa_starts_nb
                            loss_me_nb = mfa_ends_nb

                    loss_outputs = compute_masked_contrastive_loss(
                        speech_embs=speech_embs,
                        text_embs=text_embs,
                        logit_scale=logit_scale,
                        local_group_ids=group_ids,
                        local_term_ids=term_ids,
                        local_positive_term_ids=positive_term_ids,
                        local_positive_term_mask=positive_term_mask,
                        local_valid_mask=valid,
                        neg_bank_embs=nb_embs,
                        neg_bank_term_ids=nb_tids,
                        per_sample_neg_embs=ps_embs,
                        per_sample_neg_term_ids=ps_tids,
                        margin=args.margin,
                        online_hard_neg_k=args.online_hard_neg_k,
                        easy_neg_weight=current_easy_neg_weight,
                        maxsim_agg=args.maxsim_agg,
                        maxsim_softmax_tau=args.maxsim_softmax_tau,
                        mfa_term_starts=loss_ms_nb,
                        mfa_term_ends=loss_me_nb,
                        win_starts=loss_ws_nb,
                        win_ends=loss_we_nb,
                        mfa_window_selection=getattr(
                            args, "mfa_window_selection", DEFAULT_MFA_WINDOW_SELECTION
                        ),
                        mfa_lse_temperature=getattr(
                            args, "mfa_lse_temperature", DEFAULT_MFA_LSE_TEMPERATURE
                        ),
                        mfa_positive_scope=getattr(
                            args, "mfa_positive_scope", DEFAULT_MFA_POSITIVE_SCOPE
                        ),
                        tcm_loss_weight=getattr(args, "tcm_loss_weight", 0.0),
                        tcm_pos_loss_weight=_effective_tcm_pos_weight,
                        tcm_neg_loss_weight=_effective_tcm_neg_weight,
                        tcm_pos_threshold=getattr(args, "tcm_pos_threshold", DEFAULT_TCM_POS_THRESHOLD),
                        tcm_neg_threshold=getattr(args, "tcm_neg_threshold", DEFAULT_TCM_NEG_THRESHOLD),
                        tcm_loss_form=getattr(args, "tcm_loss_form", DEFAULT_TCM_LOSS_FORM),
                        tcm_reduction=getattr(args, "tcm_reduction", DEFAULT_TCM_REDUCTION),
                        tcm_neg_scope=getattr(args, "tcm_neg_scope", DEFAULT_TCM_NEG_SCOPE),
                        tcm_neg_topk=getattr(args, "tcm_neg_topk", 0),
                        hcl_beta=getattr(args, "hcl_beta", DEFAULT_HCL_BETA),
                    )
                    total_loss = loss_outputs["total"]

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
                hnm_str = ""
                if args.online_hard_neg_k > 0:
                    hnm_str = f" easy_w={current_easy_neg_weight:.3f}"
                tau_str = ""
                if args.maxsim_agg == "softmax":
                    tau_str = f" sm_tau={args.maxsim_softmax_tau:.4f}"
                tcm_str = ""
                if _tcm_enabled:
                    tcm_str = (
                        f" infonce={loss_outputs['infonce'].item():.4f}"
                        f" tcm_pos={loss_outputs['tcm_pos'].item():.4f}"
                        f" tcm_neg={loss_outputs['tcm_neg'].item():.4f}"
                        f" pos_sim={loss_outputs['pos_sim_mean'].item():.3f}"
                        f" neg_sim={loss_outputs['neg_sim_mean'].item():.3f}"
                        f" tcm_w_pos={_effective_tcm_pos_weight:.3f}"
                        f" tcm_w_neg={_effective_tcm_neg_weight:.3f}"
                    )
                hcl_str = ""
                if args.hcl_beta > 0.0:
                    hcl_str = (
                        f" hcl_neg_sim_w={loss_outputs['hcl_neg_sim_weighted_mean'].item():.3f}"
                        f" hcl_logw_max={loss_outputs['hcl_log_weight_max'].item():.2f}"
                    )
                mask_str = (
                    f" pos_n={loss_outputs['pos_count_mean'].item():.2f}"
                    f" cochunk_neutral={loss_outputs['cochunk_neutral_count'].item():.0f}"
                    f" hn_pos_mask={loss_outputs['hn_false_positive_masked_count'].item():.0f}"
                )
                logger.info(
                    f"[TRAIN] step={global_step} loss={total_loss.item():.6f} "
                    f"logit_scale={ls_val:.4f} temperature={temp:.6f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}{hnm_str}{tau_str}{tcm_str}{hcl_str}{mask_str}{extra_str}"
                )
                if wandb_run is not None and global_step % args.wandb_log_interval == 0:
                    log_dict = {
                            "train/loss": total_loss.item(),
                            "train/loss_infonce": loss_outputs["infonce"].item(),
                            "train/logit_scale": ls_val,
                            "train/temperature": temp,
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "train/step": global_step,
                            "train/epoch": epoch,
                            "train/pos_sim_mean": loss_outputs["pos_sim_mean"].item(),
                            "train/neg_sim_mean": loss_outputs["neg_sim_mean"].item(),
                            "train/pos_count_mean": loss_outputs["pos_count_mean"].item(),
                            "train/cochunk_neutral_count": loss_outputs["cochunk_neutral_count"].item(),
                            "train/positive_term_mask_count": loss_outputs["positive_term_mask_count"].item(),
                            "train/hn_false_positive_masked_count": loss_outputs["hn_false_positive_masked_count"].item(),
                        }
                    if _tcm_enabled:
                        log_dict.update({
                            "train/loss_tcm_pos": loss_outputs["tcm_pos"].item(),
                            "train/loss_tcm_neg": loss_outputs["tcm_neg"].item(),
                            "train/tcm_pos_viol_rate": loss_outputs["tcm_pos_viol_rate"].item(),
                            "train/tcm_neg_viol_rate": loss_outputs["tcm_neg_viol_rate"].item(),
                            "train/tcm_loss_weight": max(
                                _effective_tcm_pos_weight,
                                _effective_tcm_neg_weight,
                            ),
                            "train/tcm_pos_loss_weight": _effective_tcm_pos_weight,
                            "train/tcm_neg_loss_weight": _effective_tcm_neg_weight,
                        })
                    if args.hcl_beta > 0.0:
                        log_dict.update({
                            "train/hcl_beta": args.hcl_beta,
                            "train/hcl_neg_sim_weighted_mean": loss_outputs["hcl_neg_sim_weighted_mean"].item(),
                            "train/hcl_log_weight_max": loss_outputs["hcl_log_weight_max"].item(),
                        })
                    if hard_neg_count > 0:
                        log_dict["train/hard_neg_count"] = hard_neg_count
                    if args.online_hard_neg_k > 0:
                        log_dict["train/easy_neg_weight"] = current_easy_neg_weight
                    # perf: wall-time per training step (excluding only the sync at
                    # the very top of next iteration). Measured from just before
                    # global_step += 1 so it includes fwd/bwd/opt + log overhead.
                    log_dict["train/step_time_ms"] = (time.time() - _step_t0) * 1000.0
                    wandb_run.log(log_dict,
                        step=global_step,
                    )

            if _profiler is not None:
                _profiler.step()

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
                    "best_metric_key": best_metric_key,
                    "best_metric_secondary_key": best_metric_secondary_key,
                    "best_metric_value": best_metric_value,
                    "best_metric_secondary_value": best_metric_secondary_value,
                },
                ep_path,
            )
            logger.info(f"[EPOCH_SAVE] {ep_path}")

    # Stop torch.profiler (if enabled) and dump ops summary.
    #
    # Bug fix vs first smoke (43856): key_averages() must be captured BEFORE
    # _profiler.stop() -- after stop the internal event buffers can be cleared
    # by the on_trace_ready handler, producing an empty table. We render both
    # sort orders into strings first, then stop, then write the file.
    if _profiler is not None:
        _summary_cuda = None
        _summary_cpu = None
        try:
            _ka = _profiler.key_averages()
            _summary_cuda = _ka.table(sort_by="self_cuda_time_total", row_limit=80)
            _summary_cpu = _ka.table(sort_by="self_cpu_time_total", row_limit=80)
        except Exception as _exc:
            logger.warning(f"[PROFILE] key_averages (pre-stop) failed: {_exc}")
        try:
            _profiler.stop()
        except Exception as _exc:
            logger.warning(f"[PROFILE] stop() raised: {_exc}")
        try:
            _summary_path = os.path.join(args.profile_out_dir, "key_averages.txt")
            with open(_summary_path, "w") as _sf:
                if _summary_cuda:
                    _sf.write("== sort: self_cuda_time_total ==\n")
                    _sf.write(_summary_cuda)
                if _summary_cpu:
                    _sf.write("\n\n== sort: self_cpu_time_total ==\n")
                    _sf.write(_summary_cpu)
                if not (_summary_cuda or _summary_cpu):
                    _sf.write("(key_averages was empty -- capture failed pre-stop)\n")
            logger.info(f"[PROFILE] summary -> {_summary_path}")
        except Exception as _exc:
            logger.warning(f"[PROFILE] key_averages dump failed: {_exc}")

    # Final save
    if is_main:
        torch.save(
            {
                "model_state_dict": raw_retriever.state_dict(),
                "text_model_state_dict": raw_text_encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "global_step": global_step,
                "epoch": last_epoch,
                "args": vars(args),
                "best_metric_key": best_metric_key,
                "best_metric_secondary_key": best_metric_secondary_key,
                "best_metric_value": best_metric_value,
                "best_metric_secondary_value": best_metric_secondary_value,
            },
            args.save_path,
        )
        logger.info(f"[FINAL_SAVE] {args.save_path}")

    if is_main and wandb_run is not None:
        finalize_wandb_run(
            wandb_run,
            success=True,
            verdict=args.run_verdict,
        )
        wandb_run.finish()

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
        "--max_steps", type=int, default=0,
        help="Hard cap on global optimizer steps. 0 = disabled (run epochs loop). "
             "Useful for fixed-budget ablations where wall-time variance from "
             "bank-refresh / eval overhead would skew comparisons.",
    )
    p.add_argument(
        "--max_train_seconds", type=int, default=0,
        help="Cap training wall-time (seconds). 0 disables. When > 0, training "
             "exits cleanly after this elapsed time; args.epochs becomes the "
             "hard upper bound (set to a large value if you want walltime to "
             "be the sole stop criterion). Post-train eval and final save "
             "still run.",
    )
    p.add_argument(
        "--profile_out_dir", type=str, default="",
        help="If non-empty, wrap rank-0 training in torch.profiler and dump "
             "Chrome traces + key_averages.txt here. Recommended together "
             "with --max_steps so you exit right after the active window. "
             "Leave empty to disable profiling (zero overhead).",
    )
    p.add_argument(
        "--profile_schedule", type=str, default="2,3,10,1",
        help="torch.profiler schedule as 'wait,warmup,active,repeat'. "
             "Default 2+3+10 = 15 steps recorded (wait 2, warmup 3, active 10). "
             "Only used when --profile_out_dir is set.",
    )
    p.add_argument(
        "--wiki_rank", type=int, default=0,
        help="Include wiki_synth entries with p31_rank < this value. "
             "0 = include all (no filtering). Gigaspeech (p31_rank=-1) always included.",
    )
    p.add_argument(
        "--noisy_ratio", type=float, default=1.0,
        help="Proportion of noisy wiki_synth samples to keep (0-1). "
             "1.0 = keep all noisy / drop all clean, "
             "0.0 = keep all clean / drop all noisy, "
             "0.5 = keep ~50%% of each. "
             "GigaSpeech (no audio_type) is always kept.",
    )
    p.add_argument(
        "--fixed_audio_samples",
        type=int,
        default=0,
        help=(
            "Pad/truncate every waveform to this many samples before feature "
            f"extraction. 0 = default {DEFAULT_FIXED_AUDIO_SAMPLES} "
            f"({DEFAULT_FIXED_AUDIO_SECONDS:.2f}s)."
        ),
    )
    p.add_argument(
        "--fixed_audio_seconds",
        type=float,
        default=0.0,
        help=(
            "Pad/truncate every waveform to this duration at 16kHz. "
            "Use for context-length ablations, e.g. 3.84."
        ),
    )
    p.add_argument(
        "--eval_fixed_audio_samples",
        type=int,
        default=0,
        help=(
            "Pad/truncate eval waveforms to this many samples. "
            "0 = use the train fixed audio length."
        ),
    )
    p.add_argument(
        "--eval_fixed_audio_seconds",
        type=float,
        default=0.0,
        help=(
            "Pad/truncate eval waveforms to this duration at 16kHz. "
            "0 = use the train fixed audio length."
        ),
    )

    p.add_argument(
        "--grad_cache_chunk_size", type=int, default=0,
        help="Sub-batch size for GradCache. 0 = disabled (standard forward-backward). "
             "When >0, decouples batch_size from GPU memory: embeddings are collected "
             "in no_grad sub-batches of this size, loss computed on full batch, then "
             "gradients propagated back through sub-batch re-forwards. "
             "~1.3x FLOPs but memory usage = chunk_size instead of per_rank_bs.",
    )

    # Model
    p.add_argument(
        "--audio_encoder_preset",
        type=str,
        default="qwen3-omni",
        choices=sorted(AUDIO_ENCODER_PRESETS),
        help="Audio encoder preset. Use custom with explicit --audio_encoder_type "
             "/ --audio_model_id / --audio_feature_extractor_id.",
    )
    p.add_argument(
        "--audio_encoder_type",
        type=str,
        default="qwen3_omni",
        choices=list(AUDIO_ENCODER_TYPES),
        help="Audio encoder implementation: qwen3_omni, whisper, or wavlm.",
    )
    p.add_argument("--audio_model_id", type=str, default=DEFAULT_QWEN_AUDIO_MODEL_ID)
    p.add_argument(
        "--audio_feature_extractor_id",
        type=str,
        default="",
        help="HF feature extractor id for the audio frontend. Empty = preset default.",
    )
    p.add_argument(
        "--audio_input_dtype",
        type=str,
        default="auto",
        choices=list(AUDIO_INPUT_DTYPES),
        help="Input tensor dtype after feature extraction. auto=bf16 for "
             "Qwen/Whisper log-mel features, fp32 for WavLM raw waveform.",
    )
    p.add_argument(
        "--audio_hidden_dim",
        type=int,
        default=0,
        help="Override audio encoder hidden size. 0 = infer from encoder config.",
    )
    p.add_argument(
        "--text_encoder_preset",
        type=str,
        default="bge-m3",
        choices=sorted(TEXT_ENCODER_PRESETS),
        help="Text encoder preset. multilingual-e5-large automatically uses "
             "the retrieval prefix unless --text_input_prefix overrides it.",
    )
    p.add_argument("--text_model_id", type=str, default=DEFAULT_TEXT_MODEL_ID)
    p.add_argument(
        "--text_input_prefix",
        type=str,
        default="",
        help="Prefix prepended to every term before tokenization, e.g. 'query: ' "
             "for multilingual-E5 retrieval mode.",
    )
    p.add_argument("--target_dim", type=int, default=DEFAULT_TARGET_DIM)
    p.add_argument("--use_lora", action="store_true", default=False)
    p.add_argument(
        "--pooling_type", type=str, default="attentive",
        choices=["attentive", "transformer"],
        help="Audio pooling strategy: 'attentive' = scalar attention, "
             "'transformer' = 1-layer cross-attention with learnable query.",
    )
    p.add_argument(
        "--use_maxsim", action="store_true", default=False,
        help="Enable Multi-Scale Max-Sim: output per-window embeddings "
             "instead of a single vector. Overrides pooling_type.",
    )
    p.add_argument(
        "--maxsim_windows", type=int, nargs="+",
        default=MAXSIM_DEFAULT_WINDOWS,
        help="Window sizes (in encoder frames) for multi-scale Max-Sim pooling. "
             "At 12.5 fps: 6=0.48s, 10=0.80s, 16=1.28s, 24=1.92s.",
    )
    p.add_argument(
        "--maxsim_stride", type=int, default=MAXSIM_DEFAULT_STRIDE,
        help="Stride (in frames) for all multi-scale sliding windows. "
             "Dense stride (1-2) ensures no term pronunciation is missed.",
    )
    p.add_argument(
        "--use_phoneme_append", action="store_true", default=False,
        help="Append ARPAbet phoneme sequence to term text for text encoder input. "
             "Format: 'term [SEP] PHONEMES: P1 P2 ...'",
    )
    p.add_argument(
        "--sparse_weight", type=float, default=0.0,
        help="Weight for sparse (lexical) embedding in hybrid text encoding. "
             "0.0 = pure dense (CLS), 1.0 = pure sparse-weighted pooling. "
             "Uses BGE-M3's pretrained sparse_linear for token importance.",
    )
    p.add_argument(
        "--text_pooling", type=str, default="cls",
        choices=sorted(TEXT_POOLING_MODES),
        help="Text encoder pooling strategy. 'cls' = CLS token only, "
             "'mean' = masked mean, 'max' = masked max, "
             "'cls_mean' = gated CLS+mean, 'cls_max' = gated CLS+max, "
             "'gated' = gated CLS+mean+max.",
    )
    p.add_argument(
        "--use_colbert", action="store_true", default=False,
        help="Use BGE-M3 ColBERT multi-vector output on text side. "
             "Outputs per-token embeddings [B,T,D] and uses full late interaction "
             "with audio MaxSim. Incompatible with sparse_weight > 0.",
    )
    p.add_argument(
        "--maxsim_agg", type=str, default="hard_max",
        choices=sorted(MAXSIM_AGG_MODES),
        help="MaxSim window aggregation: 'hard_max' (standard argmax) or "
             "'softmax' (weighted sum with temperature annealing).",
    )
    p.add_argument(
        "--maxsim_softmax_tau", type=float, default=MAXSIM_SOFTMAX_TAU_INIT,
        help="Initial temperature for softmax MaxSim aggregation. "
             f"Annealed toward {MAXSIM_SOFTMAX_TAU_MIN} during training.",
    )
    p.add_argument(
        "--mfa_supervised_maxsim", action="store_true", default=False,
        help="MFA-supervised MaxSim: only windows that fully COVER the term's "
             "time range [term_start, term_end] in the chunk participate in "
             "the max aggregation. Coverage = win_start <= term_start AND "
             "win_end >= term_end. Requires training JSONL enriched with "
             "mfa_term_start_in_chunk / mfa_term_end_in_chunk. Falls back to "
             "standard hard_max for samples without MFA data (start < 0).",
    )
    p.add_argument(
        "--mfa_window_selection", type=str,
        default=DEFAULT_MFA_WINDOW_SELECTION,
        choices=sorted(MFA_WINDOW_SELECTION_MODES),
        help="How to pick window(s) among those covering the term when "
             "--mfa_supervised_maxsim is set: 'hard_max' = argmax by similarity "
             "(baseline, original behavior); 'smallest' = argmin by window "
             "duration (tightest crop, single deterministic window); "
             "'logsumexp' = all covering windows aggregated via LSE with "
             "temperature --mfa_lse_temperature (softmax-weighted gradient "
             "across covering windows).",
    )
    p.add_argument(
        "--mfa_lse_temperature", type=float,
        default=DEFAULT_MFA_LSE_TEMPERATURE,
        help="Temperature for --mfa_window_selection logsumexp aggregation. "
             "Lower -> sharper (more hard_max-like); higher -> softer mean-like.",
    )
    p.add_argument(
        "--mfa_positive_scope", type=str,
        default=DEFAULT_MFA_POSITIVE_SCOPE,
        choices=sorted(MFA_POSITIVE_SCOPE_MODES),
        help="'auto' uses term-level positives when --mfa_supervised_maxsim is active "
             "and chunk-level positives otherwise. 'term' makes same-chunk different "
             "terms neutral so a term-specific MFA window is not trained to match "
             "other co-occurring terms.",
    )
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
    p.add_argument(
        "--scheduler_epochs",
        type=int,
        default=0,
        help="If >0, compute the cosine scheduler horizon from this epoch count "
             "instead of --epochs. Useful when stopping after 3 epochs but "
             "resuming for a fourth epoch without decaying LR to zero.",
    )
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--train_limit", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.03)
    p.add_argument("--learn_temp", action="store_true", default=False)
    p.add_argument(
        "--margin", type=float, default=0.0,
        help="CosFace margin subtracted from positive logits before softmax. "
             "Forces sim(pos) - sim(neg) >= margin for correct ranking.",
    )
    p.add_argument(
        "--online_hard_neg_k", type=int, default=0,
        help="Online Hard Negative Mining: keep only top-K hardest negatives per "
             "sample from the full global similarity matrix. 0 = disabled (use all). "
             "With GradCache 12K batch, 256-2048 is recommended.",
    )
    p.add_argument(
        "--easy_neg_weight", type=float, default=0.0,
        help="Soft O-HNM: weight for easy (non-top-K) negatives in softmax denominator. "
             "0=hard mask, (0,1)=soft weighting via logit+=log(w), >=1=no O-HNM effect. "
             "Only used when online_hard_neg_k > 0.",
    )
    p.add_argument(
        "--easy_neg_weight_final", type=float, default=0.05,
        help="Floor value for easy_neg_weight after decay (curriculum O-HNM).",
    )
    p.add_argument(
        "--hn_decay_steps", type=int, default=0,
        help="Steps to linearly decay easy_neg_weight from initial to final. "
             "0 = no decay (use constant easy_neg_weight). "
             "Decay starts from resume step (or step 0 if no resume).",
    )
    p.add_argument(
        "--reset_scheduler", action="store_true", default=False,
        help="When resuming, do NOT restore scheduler state. "
             "Create a fresh cosine schedule with --lr and step forward to resume step. "
             "Useful for curriculum learning with a different LR.",
    )
    p.add_argument(
        "--constant_lr", type=float, default=0.0,
        help="If > 0, bypass the cosine scheduler entirely and hold LR constant "
             "at this value (audio_lora/text_lora/text_full) with head_lr scaled "
             "by DEFAULT_HEAD_LR_SCALE. Intended for clean warm-continue without "
             "LR shock. Must be used together with --resume; incompatible with "
             "--reset_scheduler.",
    )
    p.add_argument(
        "--resume_cosine_decay_to_max_steps",
        action="store_true",
        default=False,
        help="When resuming from a checkpoint with optimizer LR, ignore the "
             "saved scheduler state and start a fresh no-warmup cosine decay "
             "from the checkpoint LR to zero at --max_steps. Intended for "
             "extending a completed scout without restarting warmup.",
    )

    # Negative bank
    p.add_argument(
        "--neg_bank_size",
        type=int,
        default=DEFAULT_NEG_BANK_SIZE,
        help="Number of random global negatives per step (0 = disabled). "
             "Per-sample hard negatives take precedence when enabled.",
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
        default=0,
        help="Deprecated shared-pool hard negatives. Must be 0; use "
             "--hard_neg_k_per_sample instead.",
    )
    p.add_argument(
        "--hard_neg_k_per_sample",
        type=int,
        default=0,
        help="Per-sample hard negatives: each row mines its own top-K (no dedup, no sharing "
             "across batch). Added as extra per-row columns via einsum. "
             "Masks every known GT term for the speech chunk, not only the anchor term.",
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
    p.add_argument(
        "--eval_only", action="store_true", default=False,
        help="Skip training loop; run one pass of dev + ACL eval on the "
             "loaded (or resumed) model, then exit.",
    )
    p.add_argument("--eval_steps_sample", type=int, default=DEFAULT_EVAL_STEPS_SAMPLE)
    p.add_argument("--eval_batch_size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    p.add_argument("--eval_topk", type=int, default=DEFAULT_EVAL_TOPK)
    p.add_argument("--eval_topk_extra", type=int, default=DEFAULT_EVAL_TOPK_EXTRA)
    p.add_argument(
        "--eval_sample_limit",
        type=int,
        default=0,
        help="Deterministically sample at most this many primary dev rows for inline eval. 0 = full dev.",
    )
    p.add_argument(
        "--acl_eval_sample_limit",
        type=int,
        default=0,
        help="Deterministically sample at most this many ACL6060 rows for inline eval. 0 = full ACL.",
    )
    p.add_argument(
        "--tagged_acl_eval_sample_limit",
        type=int,
        default=0,
        help="Deterministically sample at most this many tagged ACL6060 rows for inline eval. 0 = full tagged ACL.",
    )
    p.add_argument(
        "--medicine_eval_sample_limit",
        type=int,
        default=0,
        help="Deterministically sample at most this many medicine rows for inline eval. 0 = full medicine.",
    )
    p.add_argument(
        "--eval_sample_seed",
        type=int,
        default=17,
        help="Seed used for deterministic inline eval subsampling.",
    )
    p.add_argument(
        "--eval_score_device",
        type=str,
        default=DEFAULT_EVAL_SCORE_DEVICE,
        choices=["cuda", "cpu", "auto"],
        help="Device for eval similarity scoring. Default cuda uses query/text chunking and returns CPU logits.",
    )
    p.add_argument(
        "--eval_score_query_chunk",
        type=int,
        default=DEFAULT_EVAL_SCORE_QUERY_CHUNK,
        help="Query chunk size for eval similarity scoring.",
    )
    p.add_argument(
        "--eval_score_text_chunk",
        type=int,
        default=DEFAULT_EVAL_SCORE_TEXT_CHUNK,
        help="Text-bank chunk size for eval similarity scoring.",
    )
    p.add_argument(
        "--eval_top100_samples", type=int, default=0,
        help="Number of random queries to log top-100 retrieved terms with GT rank "
             "during glossary-scale eval. 0 = disabled; each non-zero sample emits "
             "a ~100-line qualitative dump per eval.",
    )
    p.add_argument(
        "--eval_glossary_match_min_norm_chars",
        type=int,
        default=2,
        help=(
            "Minimum normalized character count for a glossary term to participate "
            "in transcript text-match positives and optional glossary expansion. "
            "Default 2 drops terms like 'a+' -> 'a'. Set 1 to reproduce the old behavior."
        ),
    )
    p.add_argument("--save_steps", type=int, default=DEFAULT_SAVE_INTERVAL)
    p.add_argument("--keep_checkpoints", type=int, default=DEFAULT_KEEP_CHECKPOINTS)
    p.add_argument(
        "--save_latest_steps",
        type=int,
        default=0,
        help=(
            "If >0, overwrite <save_stem>_latest.pt every N train steps, "
            "independent of best metrics and eval cadence."
        ),
    )
    p.add_argument(
        "--save_latest_on_eval",
        action="store_true",
        default=False,
        help=(
            "After every eval pass, overwrite <save_stem>_latest.pt with a "
            "resumeable checkpoint even when best metrics do not improve."
        ),
    )
    p.add_argument("--force_dummy_audio", action="store_true", default=False)
    p.add_argument(
        "--augment_synth", action="store_true", default=False,
        help="Apply noise/speed/reverb augmentation to synthetic TTS audio (wiki_synth_* entries)",
    )
    p.add_argument(
        "--train_exclude_term_glossaries",
        type=str,
        nargs="*",
        default=[],
        help=(
            "Glossary JSON/JSONL files whose exact normalized term keys should be "
            "eligible for strict train/eval leakage filtering. Filtering is only "
            "applied when --strict_train_eval_term_filter is set. Matching is "
            "lower+strip on sample term_key/term."
        ),
    )
    p.add_argument(
        "--strict_train_eval_term_filter",
        action="store_true",
        default=False,
        help=(
            "Strictly remove train rows whose exact normalized term_key/term appears "
            "in --train_exclude_term_glossaries. Default is false because eval "
            "positives are derived from terms present in each speech chunk."
        ),
    )

    # Multi-domain / glossary-scale eval
    p.add_argument(
        "--acl_dev_jsonl",
        type=str,
        default=DEFAULT_ACL_DEV_JSONL,
        help="Path to ACL6060 dev JSONL for cross-domain eval",
    )
    p.add_argument(
        "--tagged_acl_dev_jsonl",
        type=str,
        default=DEFAULT_TAGGED_ACL_DEV_JSONL,
        help="Path to tagged ACL6060 dev JSONL for cross-domain eval",
    )
    p.add_argument(
        "--medicine_dev_jsonl",
        type=str,
        default=DEFAULT_MEDICINE_DEV_JSONL,
        help="Path to medicine dev/test JSONL for cross-domain eval",
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
        "--eval_metric_denominator",
        type=str,
        choices=("fixed_raw", "dynamic_retriever"),
        default="fixed_raw",
        help=(
            "How to define positives for glossary-scale recall/precision. "
            "fixed_raw keeps the raw/base metrics denominator fixed while "
            "only the retriever candidate bank changes; dynamic_retriever is "
            "the legacy behavior that rebuilds positives after bank expansion."
        ),
    )
    p.add_argument(
        "--eval_metrics_glossary",
        type=str,
        default="",
        help=(
            "Optional fixed metrics glossary for eval_dev. When empty, the "
            "raw/base eval bank is used as the fixed metrics denominator."
        ),
    )
    p.add_argument(
        "--acl_eval_wiki_glossary",
        type=str,
        default="",
        help=(
            "Optional ACL-specific wiki glossary JSON. When set, eval_dev "
            "uses --eval_wiki_glossary while eval_acl6060 uses this glossary."
        ),
    )
    p.add_argument(
        "--acl_eval_metrics_glossary",
        type=str,
        default="",
        help=(
            "Optional ACL-specific fixed metrics glossary. Defaults to "
            "--eval_metrics_glossary, then the raw/base ACL bank."
        ),
    )
    p.add_argument(
        "--acl_eval_glossary_sizes",
        type=int,
        nargs="*",
        default=[],
        help=(
            "Optional ACL-specific glossary sizes. Defaults to "
            "--eval_glossary_sizes when empty."
        ),
    )
    p.add_argument(
        "--tagged_acl_eval_wiki_glossary",
        type=str,
        default="",
        help=(
            "Optional tagged-ACL-specific wiki glossary JSON. When set, "
            "eval_tagged_acl uses this glossary."
        ),
    )
    p.add_argument(
        "--tagged_acl_eval_metrics_glossary",
        type=str,
        default="",
        help=(
            "Optional tagged-ACL-specific fixed metrics glossary. Defaults to "
            "--acl_eval_metrics_glossary, then --eval_metrics_glossary, then "
            "the raw/base tagged-ACL bank."
        ),
    )
    p.add_argument(
        "--tagged_acl_eval_glossary_sizes",
        type=int,
        nargs="*",
        default=[],
        help=(
            "Optional tagged-ACL-specific glossary sizes. Defaults to "
            "--acl_eval_glossary_sizes, then --eval_glossary_sizes."
        ),
    )
    p.add_argument(
        "--medicine_eval_wiki_glossary",
        type=str,
        default="",
        help=(
            "Optional medicine-specific wiki glossary JSON. When set, "
            "eval_medicine uses this glossary."
        ),
    )
    p.add_argument(
        "--medicine_eval_metrics_glossary",
        type=str,
        default="",
        help=(
            "Optional medicine-specific fixed metrics glossary. Defaults to "
            "--eval_metrics_glossary, then the raw/base medicine bank."
        ),
    )
    p.add_argument(
        "--medicine_eval_glossary_sizes",
        type=int,
        nargs="*",
        default=[],
        help=(
            "Optional medicine-specific glossary sizes. Defaults to "
            "--eval_glossary_sizes when empty."
        ),
    )
    p.add_argument(
        "--full_eval_wiki_glossary",
        type=str,
        default="",
        help=(
            "Optional larger wiki glossary used only for sparse full eval. "
            "It is loaded on the main rank only when full eval is triggered."
        ),
    )
    p.add_argument(
        "--full_eval_glossary_sizes",
        type=int,
        nargs="+",
        default=[],
        help="Glossary sizes for sparse full eval (e.g. 100000).",
    )
    p.add_argument(
        "--full_eval_every_n_evals",
        type=int,
        default=0,
        help=(
            "Run sparse full eval every N regular evals. 0 disables full eval."
        ),
    )
    p.add_argument(
        "--full_eval_name",
        type=str,
        default="dev_full",
        help="Eval prefix suffix for sparse full eval, producing eval_<name>/*.",
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
            "(e.g. eval_acl6060/recall@10); "
            "saved as <save_stem>_best_<metric_key>.pt"
        ),
    )
    p.add_argument(
        "--reset_best_on_resume",
        action="store_true",
        default=False,
        help=(
            "When resuming, ignore best_metric_value fields stored in the "
            "checkpoint and start primary/secondary best tracking from -inf. "
            "Use this when the resumed run tracks different best_metric keys."
        ),
    )
    p.add_argument(
        "--early_stop_best_patience_evals",
        type=int,
        default=0,
        help=(
            "Stop training after this many consecutive evals fail to improve "
            "--best_metric. 0 disables eval-patience early stopping. The "
            "counter is compared against the restored best_metric_value when "
            "resuming from a checkpoint."
        ),
    )
    p.add_argument(
        "--auto_full_eval_on_best",
        action="store_true",
        default=False,
        help="Submit a separate SLURM eval-only job whenever primary best improves.",
    )
    p.add_argument(
        "--auto_full_eval_launcher",
        type=str,
        default="",
        help="SLURM launcher to submit for auto full eval on primary best updates.",
    )
    p.add_argument(
        "--auto_full_eval_partition",
        type=str,
        default="",
        help="Optional sbatch --partition override for auto full eval jobs.",
    )
    p.add_argument(
        "--auto_full_eval_min_step_delta",
        type=int,
        default=0,
        help=(
            "Minimum step gap between auto full eval submissions. "
            "0 submits for every primary best update."
        ),
    )
    p.add_argument(
        "--auto_full_eval_extra_env",
        type=str,
        nargs="*",
        default=[],
        help="Extra KEY=VALUE environment overrides passed to auto eval sbatch.",
    )

    # WandB
    p.add_argument("--enable_wandb", action="store_true", default=False)
    p.add_argument("--wandb_project", type=str, default="qwen3_rag")
    p.add_argument("--wandb_exp_name", type=str, default="masked_neg_bank")
    p.add_argument(
        "--wandb_log_interval", type=int, default=DEFAULT_WANDB_LOG_INTERVAL
    )
    # Experiment-tracking schema (see .cursor/rules/experiment_tracking.mdc).
    # Required when --enable_wandb is on.
    p.add_argument(
        "--notes_file",
        type=str,
        default="",
        help="Path to markdown run notes. Must contain non-empty sections: "
             "Hypothesis, Background / Motivation, What changed vs baseline, "
             "Expected metrics, Verdict. Copy documents/code/_templates/"
             "run_notes_template.md.",
    )
    p.add_argument(
        "--experiment_family",
        type=str,
        default="",
        help="Experiment family for the 'family:<X>' tag used for baseline lookup "
             "(e.g. retriever_3variant, sst_density_ablation).",
    )
    p.add_argument(
        "--data_tag",
        type=str,
        default="",
        help="Short identifier for the training data artifact "
             "(e.g. 3variant_1m_mfa, adversarial_varlen_d5).",
    )
    p.add_argument(
        "--task_tag",
        type=str,
        default="train",
        choices=["train", "eval", "smoke"],
        help="Task kind for the 'task:<X>' tag.",
    )
    p.add_argument(
        "--extra_wandb_tags",
        type=str,
        nargs="*",
        default=[],
        help="Additional free-form tags to attach, e.g. 'rank:16' 'tp:2'.",
    )
    p.add_argument(
        "--baseline_run_ids",
        type=str,
        nargs="*",
        default=[],
        help="WandB run ids (short id or full path) of runs this experiment "
             "is compared against. Recorded in config.baseline_run_ids.",
    )
    p.add_argument(
        "--run_verdict",
        type=str,
        default="",
        help="One-sentence verdict to write into run.summary['verdict'] at the "
             "end of the run. Leave empty to mark 'pending - awaiting agent fill'.",
    )

    # TCM (Threshold-Consistent Margin) auxiliary loss
    p.add_argument(
        "--tcm_loss_weight", type=float, default=DEFAULT_TCM_LOSS_WEIGHT,
        help="Weight lambda for the TCM auxiliary loss (L_total = L_InfoNCE + "
             "lambda * (L_TCM_pos + L_TCM_neg)). 0 disables TCM.",
    )
    p.add_argument(
        "--tcm_pos_loss_weight", type=float, default=None,
        help="Optional override for the positive TCM branch weight. Defaults "
             "to --tcm_loss_weight when omitted.",
    )
    p.add_argument(
        "--tcm_neg_loss_weight", type=float, default=None,
        help="Optional override for the negative TCM branch weight. Defaults "
             "to --tcm_loss_weight when omitted.",
    )
    p.add_argument(
        "--tcm_warmup_steps", type=int, default=0,
        help="Linear warmup for TCM weight: ramp from 0 to tcm_loss_weight "
             "over this many steps. 0 = no warmup (full weight from step 0).",
    )
    p.add_argument(
        "--tcm_pos_threshold", type=float, default=DEFAULT_TCM_POS_THRESHOLD,
        help="Absolute cos-sim threshold T_beta; positives below this are penalized.",
    )
    p.add_argument(
        "--tcm_neg_threshold", type=float, default=DEFAULT_TCM_NEG_THRESHOLD,
        help="Absolute cos-sim threshold T_alpha; negatives above this are penalized.",
    )
    p.add_argument(
        "--tcm_loss_form", type=str, default=DEFAULT_TCM_LOSS_FORM,
        choices=list(TCM_LOSS_FORMS),
        help="TCM penalty form: 'squared_hinge' (relu(x)^2) or 'hinge' (relu(x)).",
    )
    p.add_argument(
        "--tcm_reduction", type=str, default=DEFAULT_TCM_REDUCTION,
        choices=list(TCM_REDUCTIONS),
        help="Denominator for TCM penalties: 'mean_viol' normalizes by number "
             "of violating pairs (default; robust to large negative banks), "
             "'mean_all' normalizes by total pair count (paper formulation).",
    )
    p.add_argument(
        "--tcm_neg_scope", type=str, default=DEFAULT_TCM_NEG_SCOPE,
        choices=list(TCM_NEG_SCOPES),
        help="Which negatives the TCM negative branch sees: 'all' preserves "
             "legacy full-matrix behavior; 'topk' keeps only the hardest "
             "per-row negatives after false-negative masking.",
    )
    p.add_argument(
        "--tcm_neg_topk", type=int, default=0,
        help="Per-row negative count used when --tcm_neg_scope=topk. "
             "0 keeps legacy all-negative behavior.",
    )
    p.add_argument(
        "--tcm_sweep_thresholds",
        type=float,
        nargs="*",
        default=[0.75],
        help="Absolute cos-sim thresholds to sweep during eval (default: "
             "0.75).  Each tau produces tcm_precision / recall / "
             "f1 / pass_rate, plus the filter-after-retrieval metrics "
             "topk{k}_chunk_any_positive_filtered_recall@tau and "
             "topk{k}_filtered_precision_*@tau "
             "(k = args.eval_topk), and (on no-term chunks) an avg-kept "
             "'noise' count.  Intended for post-hoc inference-threshold "
             "calibration; does not affect the TCM loss in any way.",
    )
    p.add_argument(
        "--tcm_sweep_fbeta",
        type=float,
        default=3.0,
        help=(
            "Deprecated compatibility option. Per-tau F-beta eval metrics are "
            "no longer emitted; report recall retention and precision columns "
            "directly instead."
        ),
    )
    p.add_argument(
        "--eval_minimal_metrics", action="store_true", default=False,
        help="Emit only the ablation-core metrics per bank: loss, top1, "
             "recall@primary, recall@extra, plus per-sweep-tau R_filt / "
             "P_mic / P_mac / avg_kept. Skips threshold-sweep (f2@tau / "
             "P@tau / score_gap / opt_threshold), matrix-level TCM "
             "(tbeta/talpha), and detection metrics (det/* including "
             "ROC-AUC / FPR@95TPR).  Reduces ACL6060 eval metric count "
             "from ~125 to ~45.",
    )
    p.add_argument(
        "--dump_sim_distributions", type=str, default="",
        help="If set, dump per-sample pos_sim / neg_top_sim arrays to an NPZ "
             "at this path (rank 0 only) during each run_sample_eval call.  "
             "Emits {eval_name}_base.npz + one {eval_name}_gs{size}.npz per "
             "glossary size. Each NPZ contains: pos_sim[N], neg_top_sim[N,K], "
             "neg_sim_mean[N], neg_sim_max[N], bank_size, K_neg. "
             "Used for analyzing score distributions under a no-TCM baseline.",
    )
    p.add_argument(
        "--dump_eval_misses_dir",
        type=str,
        default="",
        help="If set, dump JSONL/Markdown miss cases for selected eval domains "
             "and banks. A miss means no positive term appears in top eval_topk "
             "under the same positive-mask recall definition used for metrics.",
    )
    p.add_argument(
        "--dump_eval_misses_eval_names",
        type=str,
        nargs="*",
        default=[],
        help="Eval names to dump, e.g. medicine or eval_medicine. Empty means all.",
    )
    p.add_argument(
        "--dump_eval_misses_banks",
        type=str,
        nargs="*",
        default=[],
        help="Bank labels to dump, e.g. base gs1000 gs10000. Empty means all.",
    )
    p.add_argument(
        "--dump_eval_misses_topn",
        type=int,
        default=80,
        help="Number of sorted miss cases to include in the Markdown preview. "
             "The JSONL always contains all misses.",
    )

    # HCL (Robinson et al., ICLR 2021) hard-negative importance reweighting.
    p.add_argument(
        "--hcl_beta", type=float, default=DEFAULT_HCL_BETA,
        help="Concentration parameter beta for HCL hard-negative importance "
             "reweighting. 0 disables (uniform InfoNCE); typical values 0.5-2.0. "
             "Mutually exclusive with --online_hard_neg_k.",
    )

    # Term-id normalization (for HN near-variant suppression; see analyze_hn_variant_collision.py).
    p.add_argument(
        "--term_id_normalize", type=str, default=DEFAULT_TERM_ID_NORMALIZE,
        choices=list(TERM_ID_NORMALIZE_MODES),
        help="Normalize surface form BEFORE hashing into term_id, so that near-variants "
             "(e.g. 'propositions' vs 'proposition') collapse to the same id and are "
             "correctly caught by gt_match / fn_mask / fn_hn.  The raw term text still "
             "reaches the text encoder unchanged.  Modes: 'none' (legacy, bit-for-bit "
             "compatible with pre-fix ckpts); 'lower_strip' (.lower().strip() — nearly "
             "a no-op since term_key is already normalized this way); 'aggressive' "
             "(lower+strip + punctuation strip + naive per-token plural stripping).  "
             "Diagnostic at n=200 train anchors: aggressive folds 50.5%% of anchors "
             "onto at least one bank variant, and reduces average SM>=0.80 bank "
             "collisions inside a top-64 HN set from ~7.7 to sub-1.",
    )

    return p.parse_args()


def apply_encoder_presets(args: argparse.Namespace) -> None:
    text_preset = TEXT_ENCODER_PRESETS.get(args.text_encoder_preset, {})
    if text_preset:
        preset_model = text_preset.get("model_id", "")
        if preset_model and (
            args.text_model_id == DEFAULT_TEXT_MODEL_ID
            or args.text_encoder_preset != "bge-m3"
        ):
            args.text_model_id = preset_model
        preset_prefix = text_preset.get("input_prefix", "")
        if preset_prefix and args.text_input_prefix == "":
            args.text_input_prefix = preset_prefix

    audio_preset = AUDIO_ENCODER_PRESETS.get(args.audio_encoder_preset, {})
    if audio_preset:
        preset_type = audio_preset.get("type", "")
        if preset_type:
            args.audio_encoder_type = preset_type
        preset_model = audio_preset.get("model_id", "")
        if preset_model and (
            args.audio_model_id == DEFAULT_QWEN_AUDIO_MODEL_ID
            or args.audio_encoder_preset != "qwen3-omni"
        ):
            args.audio_model_id = preset_model
        preset_fe = audio_preset.get("feature_extractor_id", "")
        if preset_fe and not args.audio_feature_extractor_id:
            args.audio_feature_extractor_id = preset_fe
    if not args.audio_feature_extractor_id:
        args.audio_feature_extractor_id = args.audio_model_id


def main() -> None:
    args = parse_args()
    apply_encoder_presets(args)
    assert args.hard_neg_k <= 0, (
        "--hard_neg_k shared-pool mining has been removed; use "
        "--hard_neg_k_per_sample for per-row hard negatives."
    )
    fixed_audio_samples, eval_fixed_audio_samples = resolve_fixed_audio_samples(args)
    logger.info(
        f"[AUDIO_LENGTH] fixed_audio_samples={fixed_audio_samples} "
        f"fixed_audio_seconds={args.fixed_audio_seconds:.4f} "
        f"eval_fixed_audio_samples={eval_fixed_audio_samples} "
        f"eval_fixed_audio_seconds={args.eval_fixed_audio_seconds:.4f}"
    )
    # Must set BEFORE any stable_term_id() call anywhere in the process.
    set_term_id_normalize_mode(args.term_id_normalize)
    logger.info(f"[TERM_ID_NORMALIZE] mode={args.term_id_normalize}")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    t0 = time.time()
    train(rank=local_rank, world_size=world_size, args=args)
    if local_rank == 0:
        logger.info(f"[DONE] elapsed={time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
