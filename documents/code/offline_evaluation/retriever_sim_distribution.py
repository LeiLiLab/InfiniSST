#!/usr/bin/env python3
"""Retriever similarity-distribution diagnostics for Config C MaxSim.

For each dev chunk:
- S_pos     = cos-sim(chunk, GT term)          (has_term chunks only)
- S_neg_top1_in = top-1 cos-sim to non-GT bank  (has_term chunks only)
- S_neg_top1_pure = top-1 cos-sim to bank       (no-term chunks only)

Chunks are bucketed by domain (gigaspeech POD/YOU/AUD, wiki_synth, ACL6060).
Outputs:
- <out_dir>/sim_records.npz    raw per-chunk scores (S_pos, S_neg_top1, mask)
- <out_dir>/summary.tsv        per-domain stats (n, mean/std/p10/p50/p90 etc.)
- <out_dir>/hist_<domain>.png  S_pos vs S_neg_top1 histogram per domain

All strings / log messages are English.  Audio is processed one chunk at a
time (no batching) because each chunk goes through the retriever's
multi-scale window pool and produces a variable number of windows;
batching requires padding + masking we don't need for a diagnostic.
"""

from __future__ import annotations

# ======Configuration=====
REPO_ROOT = "/mnt/taurus/home/jiaxuanluo/InfiniSST"
TRAIN_DIR = f"{REPO_ROOT}/documents/code/train/term_train"

CONFIG_C_CKPT = (
    "/mnt/aries/data4/jiaxuanluo/train_outputs/"
    "q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_clean_gc_wr1000k_m0.1_"
    "maxsim_mfa_final_C_best_acl6060_gs10000.pt"
)

DEV_JSONL_GS = (
    "/mnt/gemini/data1/jiaxuanluo/term_dev_with_wiki_synth_normalized.jsonl"
)
DEV_JSONL_ACL = (
    "/mnt/gemini/data2/jiaxuanluo/"
    "acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl"
)
EVAL_WIKI_GLOSSARY = (
    f"{REPO_ROOT}/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs.json"
)

AUDIO_MODEL_ID = "Atotti/Qwen3-Omni-AudioTransformer"
TEXT_MODEL_ID = "BAAI/bge-m3"
WHISPER_FEAT_ID = "openai/whisper-large-v3"
EXPECTED_SAMPLE_RATE = 16000

LORA_RANK = 128
LORA_ALPHA = 256
TEXT_LORA_RANK = 128
TEXT_LORA_ALPHA = 256
TARGET_DIM = 1024
TEMPERATURE = 0.07
POOLING_TYPE = "transformer"
TEXT_POOLING = "cls"
SPARSE_WEIGHT = 0.0
USE_MAXSIM = True
MAXSIM_WINDOWS = [6, 10, 16, 24]
MAXSIM_STRIDE = 2
LORA_TARGET_MODULES = "q_proj k_proj v_proj out_proj fc1 fc2 proj1 proj2".split()
TEXT_LORA_TARGET_MODULES = "query key value dense".split()

TERM_ENCODE_BATCH = 512

HIST_BINS = 60
HIST_RANGE = (-0.05, 1.0)
# ======Configuration=====

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F


def _setup_logger() -> logging.Logger:
    lg = logging.getLogger("sim_diag")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        lg.addHandler(h)
    return lg


logger = _setup_logger()


def _import_train_modules():
    if TRAIN_DIR not in sys.path:
        sys.path.insert(0, TRAIN_DIR)
    from qwen3_glossary_neg_train import (
        BgeM3TextEncoder,
        Qwen3OmniRetriever,
        _encode_terms_batch,
        _maxsim_score,
    )
    return (
        Qwen3OmniRetriever,
        BgeM3TextEncoder,
        _encode_terms_batch,
        _maxsim_score,
    )


def _domain_of(utter_id: str, source: str) -> str:
    """Return coarse domain tag for per-domain bucketing."""
    if source == "acl":
        return "acl6060"
    uid = utter_id or ""
    if uid.startswith("wiki_synth_"):
        return "wiki_synth"
    if uid.startswith("POD"):
        return "gs_pod"
    if uid.startswith("YOU"):
        return "gs_you"
    if uid.startswith("AUD"):
        return "gs_aud"
    return "gs_other"


def _load_jsonl_rows(
    path: str, source: str, limit: Optional[int] = None
) -> List[Dict]:
    rows: List[Dict] = []
    assert os.path.isfile(path), f"JSONL not found: {path}"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            r["_source"] = source
            r["_domain"] = _domain_of(r.get("utter_id", ""), source)
            rows.append(r)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _build_retriever(device: torch.device):
    Qwen3OmniRetriever, BgeM3TextEncoder, _encode_terms_batch, _maxsim_score = (
        _import_train_modules()
    )
    retriever = Qwen3OmniRetriever(
        model_id=AUDIO_MODEL_ID,
        target_dim=TARGET_DIM,
        use_lora=True,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_target_modules=LORA_TARGET_MODULES,
        temperature=TEMPERATURE,
        learn_temp=False,
        pooling_type=POOLING_TYPE,
        use_maxsim=USE_MAXSIM,
        maxsim_windows=MAXSIM_WINDOWS,
        maxsim_stride=MAXSIM_STRIDE,
    ).to(device)

    text_encoder = BgeM3TextEncoder(
        model_id=TEXT_MODEL_ID,
        lora_rank=TEXT_LORA_RANK,
        lora_alpha=TEXT_LORA_ALPHA,
        target_modules=TEXT_LORA_TARGET_MODULES,
        full_finetune=False,
        sparse_weight=SPARSE_WEIGHT,
        text_pooling=TEXT_POOLING,
        use_colbert=False,
    ).to(device)

    from transformers import AutoTokenizer, WhisperFeatureExtractor

    text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
    feat_ext = WhisperFeatureExtractor.from_pretrained(WHISPER_FEAT_ID)
    return (
        retriever,
        text_encoder,
        text_tokenizer,
        feat_ext,
        _encode_terms_batch,
        _maxsim_score,
    )


def _strip_module(sd):
    return {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in sd.items()
    }


def _load_ckpt(
    retriever, text_encoder, ckpt_path: str, device: torch.device
) -> None:
    assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location=device)
    retriever.load_state_dict(
        _strip_module(ckpt.get("model_state_dict", {})), strict=False
    )
    assert "text_model_state_dict" in ckpt, (
        f"Checkpoint missing text_model_state_dict: {ckpt_path}"
    )
    text_encoder.load_state_dict(
        _strip_module(ckpt["text_model_state_dict"]), strict=False
    )
    retriever.eval()
    text_encoder.eval()
    logger.info("Loaded retriever + text encoder ckpt from %s", ckpt_path)


@torch.no_grad()
def _encode_audio_one(
    wav_path: str, retriever, feat_ext, device: torch.device
) -> Optional[torch.Tensor]:
    """Encode a single chunk wav into [1, W, D] multi-scale embedding."""
    if not os.path.isfile(wav_path):
        return None
    audio, sr = sf.read(wav_path)
    assert sr == EXPECTED_SAMPLE_RATE, (
        f"Unexpected sample rate {sr} for {wav_path}"
    )
    audio = np.asarray(audio, dtype=np.float32).flatten()
    if audio.size < EXPECTED_SAMPLE_RATE * 0.16:
        return None  # too short to be meaningful
    inp = feat_ext(
        [audio], sampling_rate=EXPECTED_SAMPLE_RATE,
        return_tensors="pt", padding=False,
    )
    mel = inp.input_features.squeeze(0)
    mel_len = mel.shape[-1]
    input_features = mel.to(device, dtype=torch.bfloat16)
    feature_lens = torch.tensor([mel_len], dtype=torch.long, device=device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        embs = retriever(input_features, feature_lens)
    return embs.float()


def _build_term_bank(
    rows: List[Dict], wiki_terms: List[str]
) -> Tuple[List[str], Dict[str, int]]:
    """Union of (deduped) dev GT terms + wiki glossary. Case-normalized."""
    seen: Dict[str, int] = {}
    terms: List[str] = []

    def _add(t_raw: str) -> None:
        t = (t_raw or "").strip()
        if not t:
            return
        key = t.lower()
        if key in seen:
            return
        seen[key] = len(terms)
        terms.append(t)

    for r in rows:
        _add(r.get("term", ""))
    for t in wiki_terms:
        _add(t)
    return terms, seen


def _load_wiki_terms(path: str) -> List[str]:
    if not path or not os.path.isfile(path):
        logger.info("Wiki glossary absent or not given, skipping.")
        return []
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    assert isinstance(entries, list)
    out: List[str] = []
    seen = set()
    for e in entries:
        t = e["term"].strip()
        k = t.lower()
        if not t or k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


def _compute_sim(
    speech_emb: torch.Tensor, text_bank: torch.Tensor, maxsim_fn
) -> torch.Tensor:
    """speech_emb: [1,W,D]. text_bank: [N,D] single-vec. Returns [N]."""
    if speech_emb.ndim == 3:
        sim = maxsim_fn(speech_emb, text_bank)
    else:
        sim = speech_emb @ text_bank.T
    return sim.squeeze(0)


def _percentiles(x: np.ndarray, ps=(10, 25, 50, 75, 90, 95, 99)) -> Dict[str, float]:
    if x.size == 0:
        return {f"p{p}": float("nan") for p in ps}
    return {f"p{p}": float(np.percentile(x, p)) for p in ps}


def _stat_block(x: np.ndarray) -> Dict[str, float]:
    if x.size == 0:
        return {
            "n": 0, "mean": float("nan"), "std": float("nan"),
            **{f"p{p}": float("nan") for p in (10, 25, 50, 75, 90, 95, 99)},
        }
    out = {
        "n": int(x.size), "mean": float(x.mean()),
        "std": float(x.std()),
    }
    out.update(_percentiles(x))
    return out


def _write_summary_tsv(
    out_path: str,
    records: Dict[str, Dict[str, np.ndarray]],
) -> None:
    cols = [
        "domain", "kind", "n", "mean", "std",
        "p10", "p25", "p50", "p75", "p90", "p95", "p99",
    ]
    lines = ["\t".join(cols)]
    for dom, kinds in records.items():
        for kind, arr in kinds.items():
            st = _stat_block(arr)
            row = [dom, kind] + [str(st.get(c, "")) for c in cols[2:]]
            lines.append("\t".join(row))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _plot_hist_per_domain(
    out_dir: str, records: Dict[str, Dict[str, np.ndarray]]
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        logger.warning("matplotlib unavailable (%s); skipping plots.", exc)
        return

    for dom, kinds in records.items():
        fig, ax = plt.subplots(figsize=(8, 4))
        for kind, arr in kinds.items():
            if arr.size == 0:
                continue
            ax.hist(
                arr, bins=HIST_BINS, range=HIST_RANGE,
                alpha=0.45, label=f"{kind} (n={arr.size})", density=True,
            )
        ax.set_title(f"Config C sim distribution -- {dom}")
        ax.set_xlabel("cos similarity")
        ax.set_ylabel("density")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = os.path.join(out_dir, f"hist_{dom}.png")
        fig.savefig(out, dpi=120)
        plt.close(fig)
    logger.info("Wrote per-domain histograms to %s", out_dir)


def _plot_hist_global(
    out_dir: str, records: Dict[str, Dict[str, np.ndarray]]
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    kinds = ["S_pos", "S_neg_top1_in", "S_neg_top1_pure"]
    fig, axes = plt.subplots(1, len(kinds), figsize=(15, 4), sharey=True)
    for ax, kind in zip(axes, kinds):
        for dom, dkinds in records.items():
            arr = dkinds.get(kind)
            if arr is None or arr.size == 0:
                continue
            ax.hist(
                arr, bins=HIST_BINS, range=HIST_RANGE,
                alpha=0.5, label=f"{dom} (n={arr.size})", density=True,
            )
        ax.set_title(kind)
        ax.set_xlabel("cos similarity")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("density")
    fig.suptitle("Config C MaxSim retriever - cross-domain sim distribution")
    fig.tight_layout()
    out = os.path.join(out_dir, "hist_all_domains.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("Wrote global histogram to %s", out)


def run(args: argparse.Namespace) -> None:
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "GPU is required for audio encoding."

    t0 = time.time()

    rows: List[Dict] = []
    rows.extend(_load_jsonl_rows(DEV_JSONL_GS, "gs", limit=args.limit_gs))
    if args.include_acl:
        rows.extend(_load_jsonl_rows(DEV_JSONL_ACL, "acl", limit=args.limit_acl))
    logger.info("Loaded %d chunk rows (gs+acl)", len(rows))

    wiki_terms = _load_wiki_terms(EVAL_WIKI_GLOSSARY) if args.use_wiki_glossary else []
    logger.info("Wiki glossary terms: %d", len(wiki_terms))

    terms, term_index = _build_term_bank(rows, wiki_terms)
    logger.info("Term bank size: %d (dedup, case-normalized)", len(terms))

    retr, txt_enc, txt_tok, feat_ext, encode_terms_batch, maxsim_fn = _build_retriever(device)
    _load_ckpt(retr, txt_enc, CONFIG_C_CKPT, device)

    logger.info("Encoding %d bank terms ...", len(terms))
    t_enc0 = time.time()
    bank_embs = encode_terms_batch(
        txt_enc, txt_tok, terms, device,
        batch_size=TERM_ENCODE_BATCH, use_phoneme_append=False,
    ).to(device)
    assert bank_embs.ndim == 2, (
        f"Expected 2D text bank [N,D], got shape {tuple(bank_embs.shape)}"
    )
    logger.info(
        "Encoded bank shape=%s in %.1fs", tuple(bank_embs.shape), time.time() - t_enc0
    )

    records: Dict[str, Dict[str, List[float]]] = {}

    def _push(dom: str, kind: str, val: float) -> None:
        records.setdefault(dom, {"S_pos": [], "S_neg_top1_in": [], "S_neg_top1_pure": []})
        records[dom][kind].append(val)

    n_used = n_missing_audio = n_missing_gt = 0
    t_loop = time.time()
    for i, r in enumerate(rows):
        dom = r["_domain"]
        wav = r.get("chunk_audio_path", "")
        emb = _encode_audio_one(wav, retr, feat_ext, device)
        if emb is None:
            n_missing_audio += 1
            continue
        sim = _compute_sim(emb, bank_embs, maxsim_fn).cpu().numpy()  # [N]

        gt_term = (r.get("term", "") or "").strip().lower()
        if gt_term:
            gt_idx = term_index.get(gt_term, -1)
            if gt_idx < 0:
                n_missing_gt += 1
                continue
            s_pos = float(sim[gt_idx])
            mask_val = sim[gt_idx]
            sim_masked = sim.copy()
            sim_masked[gt_idx] = -np.inf
            s_neg_top1 = float(sim_masked.max())
            _push(dom, "S_pos", s_pos)
            _push(dom, "S_neg_top1_in", s_neg_top1)
        else:
            s_neg_top1 = float(sim.max())
            _push(dom, "S_neg_top1_pure", s_neg_top1)

        n_used += 1
        if (i + 1) % 50 == 0 or i == 0:
            rate = (i + 1) / max(1e-3, time.time() - t_loop)
            logger.info(
                "[%d/%d] used=%d miss_audio=%d miss_gt=%d rate=%.2f chunks/s",
                i + 1, len(rows), n_used, n_missing_audio, n_missing_gt, rate,
            )

    logger.info(
        "Done chunk loop in %.1fs; used=%d miss_audio=%d miss_gt=%d",
        time.time() - t_loop, n_used, n_missing_audio, n_missing_gt,
    )

    np_records: Dict[str, Dict[str, np.ndarray]] = {
        dom: {k: np.asarray(v, dtype=np.float32) for k, v in kinds.items()}
        for dom, kinds in records.items()
    }

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    npz_path = os.path.join(out_dir, "sim_records.npz")
    flat: Dict[str, np.ndarray] = {}
    for dom, kinds in np_records.items():
        for k, v in kinds.items():
            flat[f"{dom}__{k}"] = v
    np.savez_compressed(npz_path, **flat)
    logger.info("Saved raw records -> %s", npz_path)

    summary_tsv = os.path.join(out_dir, "summary.tsv")
    _write_summary_tsv(summary_tsv, np_records)
    logger.info("Saved summary -> %s", summary_tsv)

    _plot_hist_per_domain(out_dir, np_records)
    _plot_hist_global(out_dir, np_records)

    logger.info("Total elapsed: %.1fs", time.time() - t0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True,
                   help="Output directory for npz / tsv / png.")
    p.add_argument("--limit_gs", type=int, default=None,
                   help="Limit number of gigaspeech+wiki_synth rows (smoke test).")
    p.add_argument("--limit_acl", type=int, default=None,
                   help="Limit number of ACL6060 rows.")
    p.add_argument("--include_acl", action="store_true", default=False,
                   help="Also process the ACL6060 dev jsonl.")
    p.add_argument("--use_wiki_glossary", action="store_true", default=False,
                   help="Extend term bank with the NLP/AI/CS wiki glossary.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
