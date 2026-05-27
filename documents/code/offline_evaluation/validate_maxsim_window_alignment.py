"""Validate MaxSim window alignment with MFA term timestamps on ACL6060 dev.

For each has_term sample:
1. Run the trained MaxSim retriever to get the argmax window index
2. Map that window index back to a time range within the 1.92s chunk
3. Compare with the MFA-derived term start/end time
4. Report IoU, overlap ratio, and distance statistics
"""
import os
import sys
import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, WhisperFeatureExtractor

sys.path.insert(0, str(Path(__file__).resolve().parents[2].parent))
from documents.code.train.term_train.qwen3_glossary_neg_train import (
    Qwen3OmniRetriever,
    BgeM3TextEncoder,
    MAXSIM_DEFAULT_WINDOWS,
    MAXSIM_DEFAULT_STRIDE,
    DEFAULT_TEXT_MAX_LENGTH,
)

# ======Configuration=====
MODEL_PATH = "/mnt/data/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs10752_t=0.03_3var_clean_gc_wr1000k_m0.1_maxsim_sp07_best_acl6060_gs10000.pt"
ACL_DEV_JSONL = "/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl"
MFA_TEXTGRID_DIR = "/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval/mfa_textgrids"
CHUNK_SEC = 1.92
STRIDE_SEC = 0.96
SAMPLE_RATE = 16000
ENCODER_FPS = 12.5
FRAME_SEC = 1.0 / ENCODER_FPS  # 0.08s per frame
MAXSIM_WINDOWS = [6, 10, 16, 24]
MAXSIM_STRIDE_FRAMES = 2
N_ENCODER_FRAMES = 24
# ======Configuration=====


@dataclass
class WordInterval:
    word: str
    start: float
    end: float


def parse_textgrid(tg_path: str) -> List[WordInterval]:
    with open(tg_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines()]
    idx = 0
    while idx < len(lines) and lines[idx] != "<exists>":
        idx += 1
    assert idx < len(lines), f"No <exists> in {tg_path}"
    idx += 1
    num_tiers = int(lines[idx])
    idx += 1
    words = []
    for _ in range(num_tiers):
        tier_class = lines[idx].strip('"'); idx += 1
        tier_name = lines[idx].strip('"'); idx += 1
        idx += 1  # xmin
        idx += 1  # xmax
        num_intervals = int(lines[idx]); idx += 1
        for _ in range(num_intervals):
            xmin = float(lines[idx]); idx += 1
            xmax = float(lines[idx]); idx += 1
            text = lines[idx].strip('"'); idx += 1
            if tier_name == "words" and text.strip():
                words.append(WordInterval(word=text.lower(), start=xmin, end=xmax))
    return words


def find_term_in_chunk(
    words: List[WordInterval], term: str, chunk_start: float, chunk_end: float
) -> Optional[Tuple[float, float]]:
    """Find term words within the chunk time range. Returns (start, end) relative to chunk_start."""
    term_words = term.lower().split()
    chunk_words = [
        w for w in words
        if w.start >= chunk_start - 0.05 and w.end <= chunk_end + 0.05
    ]
    for i in range(len(chunk_words) - len(term_words) + 1):
        candidate = [chunk_words[i + j].word for j in range(len(term_words))]
        if candidate == term_words:
            t_start = chunk_words[i].start - chunk_start
            t_end = chunk_words[i + len(term_words) - 1].end - chunk_start
            return (max(0, t_start), min(CHUNK_SEC, t_end))
    return None


def build_window_time_map(n_encoder_frames: int) -> List[Tuple[float, float, int]]:
    """Build a list of (start_sec, end_sec, kernel_size) for each MaxSim window.

    n_encoder_frames: actual number of encoder output frames for this audio.
    """
    windows = []
    for w in MAXSIM_WINDOWS:
        if w >= n_encoder_frames:
            windows.append((0.0, CHUNK_SEC, w))
        else:
            n_out = (n_encoder_frames - w) // MAXSIM_STRIDE_FRAMES + 1
            for k in range(n_out):
                frame_start = k * MAXSIM_STRIDE_FRAMES
                frame_end = frame_start + w
                windows.append((frame_start * FRAME_SEC, frame_end * FRAME_SEC, w))
    return windows


def compute_iou(a_start, a_end, b_start, b_end) -> float:
    inter_start = max(a_start, b_start)
    inter_end = min(a_end, b_end)
    inter = max(0, inter_end - inter_start)
    union = (a_end - a_start) + (b_end - b_start) - inter
    if union <= 0:
        return 0.0
    return inter / union


def compute_overlap_ratio(win_start, win_end, term_start, term_end) -> float:
    """What fraction of the term duration is covered by the window?"""
    inter_start = max(win_start, term_start)
    inter_end = min(win_end, term_end)
    inter = max(0, inter_end - inter_start)
    term_dur = term_end - term_start
    if term_dur <= 0:
        return 0.0
    return inter / term_dur


def load_model(device):
    ckpt = torch.load(MODEL_PATH, map_location=device)

    def _strip(sd):
        return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}

    retriever = Qwen3OmniRetriever(
        model_id="Atotti/Qwen3-Omni-AudioTransformer",
        lora_rank=128, lora_alpha=256,
        lora_target_modules="q_proj k_proj v_proj out_proj fc1 fc2 proj1 proj2".split(),
        target_dim=1024, pooling_type="transformer",
        use_lora=True, use_maxsim=True,
        maxsim_windows=MAXSIM_WINDOWS, maxsim_stride=MAXSIM_STRIDE_FRAMES,
        temperature=0.03, learn_temp=False,
    ).to(device)
    retriever.load_state_dict(_strip(ckpt.get("model_state_dict", {})), strict=False)

    text_encoder = BgeM3TextEncoder(
        model_id="BAAI/bge-m3",
        lora_rank=128, lora_alpha=256,
        target_modules="query key value dense".split(),
        full_finetune=False,
        sparse_weight=0.7,
        text_pooling="cls",
    ).to(device)
    if "text_model_state_dict" in ckpt:
        text_encoder.load_state_dict(_strip(ckpt["text_model_state_dict"]), strict=False)

    retriever.eval()
    text_encoder.eval()
    return retriever, text_encoder


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Model: {MODEL_PATH}")

    retriever, text_encoder = load_model(device)
    text_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")

    window_time_map = None  # built dynamically per actual encoder frame count

    # Load ACL dev samples with terms
    samples = []
    with open(ACL_DEV_JSONL) as f:
        for line in f:
            d = json.loads(line)
            if d.get("term", "").strip():
                samples.append(d)
    print(f"[INFO] ACL dev samples with terms: {len(samples)}")

    # Parse all TextGrids
    tg_cache: Dict[str, List[WordInterval]] = {}

    ious = []
    overlaps = []
    center_dists = []
    kernel_usage = {w: 0 for w in MAXSIM_WINDOWS}
    matched = 0
    unmatched_mfa = 0
    total = 0

    print("\n[RESULTS] Sample-level analysis:")
    print(f"{'idx':>4} {'term':<25} {'MFA range':>18} {'Win range':>18} {'IoU':>6} {'Ovlp':>6} {'WinSize':>7}")
    print("-" * 110)

    with torch.no_grad():
        for si, sample in enumerate(samples):
            utter_id = sample["utter_id"]
            chunk_idx = sample["chunk_idx"]
            term = sample["term"]
            audio_path = sample["chunk_audio_path"]

            chunk_start = chunk_idx * STRIDE_SEC
            chunk_end = chunk_start + CHUNK_SEC

            # Get MFA term time
            if utter_id not in tg_cache:
                tg_path = os.path.join(MFA_TEXTGRID_DIR, utter_id, f"{utter_id}.TextGrid")
                if os.path.exists(tg_path):
                    tg_cache[utter_id] = parse_textgrid(tg_path)
                else:
                    tg_cache[utter_id] = []

            mfa_words = tg_cache[utter_id]
            term_time = find_term_in_chunk(mfa_words, term, chunk_start, chunk_end)
            if term_time is None:
                unmatched_mfa += 1
                continue

            total += 1
            term_start, term_end = term_time

            # Encode audio
            wav, sr = sf.read(audio_path)
            assert sr == SAMPLE_RATE
            feats = feature_extractor(wav, sampling_rate=sr, return_tensors="pt")
            input_features = feats.input_features.to(device).to(torch.bfloat16)
            feat_lens = torch.tensor([input_features.shape[-1]], dtype=torch.long, device=device)

            # Get multi-scale window embeddings [1, W, D]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                speech_emb = retriever(input_features, feat_lens)

            # Encode text [1, D]
            tok = text_tokenizer(
                [term], padding=True, truncation=True,
                max_length=DEFAULT_TEXT_MAX_LENGTH, return_tensors="pt",
            ).to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                text_emb = text_encoder(tok.input_ids, tok.attention_mask)

            # Build window time map from actual encoder output frame count
            W = speech_emb.shape[1]
            if window_time_map is None or len(window_time_map) != W:
                for n_try in range(10, 500):
                    test_map = build_window_time_map(n_try)
                    if len(test_map) == W:
                        window_time_map = test_map
                        n_encoder_frames = n_try
                        break
                assert window_time_map is not None and len(window_time_map) == W, (
                    f"Could not find n_encoder_frames producing {W} windows"
                )
                # Build valid mask: only windows whose END <= CHUNK_SEC are valid
                # (windows extending into padding zone produce garbage embeddings)
                valid_window_mask = torch.tensor(
                    [we <= CHUNK_SEC + 0.01 for (ws, we, wk) in window_time_map],
                    dtype=torch.bool, device=device,
                )
                n_valid = valid_window_mask.sum().item()
                print(f"[INFO] Encoder output: {n_encoder_frames} frames -> {W} windows")
                print(f"[INFO] Valid windows (within {CHUNK_SEC}s): {n_valid} / {W}")
                for i, (ws, we, wk) in enumerate(window_time_map):
                    if valid_window_mask[i] and (i < 5 or i == n_valid - 1):
                        print(f"  window[{i}]: {ws:.3f}s - {we:.3f}s (dur={we-ws:.3f}s, kernel={wk})")
                print(f"  ... ({n_valid} valid windows)")

            # Compute per-window similarity, masking out padding-zone windows
            sim = torch.matmul(speech_emb, text_emb.T).squeeze(-1)  # [1, W]
            sim[:, ~valid_window_mask] = -float("inf")
            best_win_idx = sim.argmax(dim=1).item()

            win_start, win_end, win_kernel = window_time_map[best_win_idx]
            iou = compute_iou(win_start, win_end, term_start, term_end)
            ovlp = compute_overlap_ratio(win_start, win_end, term_start, term_end)
            center_dist = abs((win_start + win_end) / 2 - (term_start + term_end) / 2)

            ious.append(iou)
            overlaps.append(ovlp)
            center_dists.append(center_dist)

            win_size_label = f"{win_kernel}f/{win_kernel*FRAME_SEC:.2f}s"
            kernel_usage[win_kernel] = kernel_usage.get(win_kernel, 0) + 1

            if ovlp >= 1.0:
                matched += 1

            if si < 50 or (si < 200 and si % 5 == 0) or si % 20 == 0:
                print(
                    f"{si:4d} {term:<25} "
                    f"[{term_start:.3f}-{term_end:.3f}]s "
                    f"[{win_start:.3f}-{win_end:.3f}]s "
                    f"{iou:6.3f} {ovlp:6.3f} {win_size_label:>7}"
                )

    print("\n" + "=" * 80)
    print("[SUMMARY]")
    print(f"  Total with_term samples:     {len(samples)}")
    print(f"  MFA term found in chunk:     {total}")
    print(f"  MFA term NOT found:          {unmatched_mfa}")
    print(f"  ---")
    print(f"  Mean IoU:                    {np.mean(ious):.4f}")
    print(f"  Median IoU:                  {np.median(ious):.4f}")
    print(f"  Mean term overlap ratio:     {np.mean(overlaps):.4f}")
    print(f"  Median term overlap ratio:   {np.median(overlaps):.4f}")
    print(f"  100% term covered:           {matched}/{total} ({matched/total*100:.1f}%)")
    print(f"  Mean center distance:        {np.mean(center_dists):.4f}s")
    print(f"  Median center distance:      {np.median(center_dists):.4f}s")

    # Distribution of overlap
    bins = [0, 0.25, 0.5, 0.75, 0.9, 1.0, 1.01]
    labels = ["<0.25", "0.25-0.5", "0.5-0.75", "0.75-0.9", "0.9-1.0", "=1.0"]
    hist, _ = np.histogram(overlaps, bins=bins)
    print(f"\n  Term overlap distribution:")
    for label, count in zip(labels, hist):
        pct = count / total * 100
        bar = "#" * int(pct / 2)
        print(f"    {label:>10}: {count:4d} ({pct:5.1f}%) {bar}")

    print(f"\n  Window scale (kernel) usage:")
    for w in MAXSIM_WINDOWS:
        cnt = kernel_usage.get(w, 0)
        pct = cnt / total * 100 if total > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"    {w:2d}f ({w*FRAME_SEC:.2f}s): {cnt:4d} ({pct:5.1f}%) {bar}")


if __name__ == "__main__":
    main()
