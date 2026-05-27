"""
Generate qualitative top-k examples showing text/TTS recall and intersection.
"""
import sys
import json
import os
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, "/home/jiaxuanluo/InfiniSST")

# ======Configuration=====
DEV_JSONL = "/mnt/gemini/data/siqiouyang/term_dev_dataset_final.jsonl"
DEV_TTS_JSONL = "/mnt/gemini/data/siqiouyang/term_dev_dataset_final_with_tts.jsonl"
DUAL_TEXT_MODEL_PATH = (
    "/mnt/gemini/data/jiaxuanluo/"
    "q3rag_tts_lora-r32-tr16_bs4k_ttsw0.0_ttm=query key value_temperature=0.03_v2_epoch_5.pt"
)
DUAL_TTS_MODEL_PATH = (
    "/mnt/gemini/data/jiaxuanluo/"
    "q3rag_tts_lora-r32-tr16_bs4k_ttsw1.0_ttm=query key value_temperature=0.03_v2_step_2000.pt"
)
DUAL_INDEX_PKL = (
    "/mnt/gemini/data2/jiaxuanluo/offline_eval_dual_vs_single_model/"
    "dual_text_ttsw0.0_epoch5/index_v4_tr16_dual_text_ttsw0.0_epoch5.pkl"
)
WHISPER_FE = "openai/whisper-large-v3"
LORA_RANK = 32
LORA_ALPHA = 64
TARGET_DIM = 1024
TOP_K = 10
EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHUNK_SAMPLES = 30720
NUM_WITH_TERM_SAMPLES = 8
NUM_NO_TERM_SAMPLES = 4
RANDOM_SEED = 42
ENCODE_BATCH_SIZE = 256
# ======Configuration=====

import torch
import faiss
import soundfile as sf
from transformers import WhisperFeatureExtractor
from agents.streaming_qwen3_rag_retriever_v4 import Qwen3OmniRetriever


def load_dev_data():
    groups = {}
    for line in Path(DEV_JSONL).open():
        obj = json.loads(line.strip())
        term = str(obj.get("term", "")).strip().lower()
        uid = obj.get("utter_id", "")
        cidx = str(obj.get("chunk_idx", ""))
        apath = obj.get("chunk_audio_path", "")
        src = obj.get("chunk_src_text", "")
        cid = f"{uid}::{cidx}"
        if cid not in groups:
            groups[cid] = {
                "audio_path": apath,
                "src": src,
                "gt_terms": set(),
                "uid": uid,
                "cidx": cidx,
            }
        if term:
            groups[cid]["gt_terms"].add(term)
    return groups


def load_tts_paths():
    tts_paths = {}
    for line in Path(DEV_TTS_JSONL).open():
        obj = json.loads(line.strip())
        t = str(obj.get("term", "")).strip().lower()
        p = str(obj.get("tts_audio_path", "")).strip()
        if t and p:
            tts_paths.setdefault(t, [])
            if p not in tts_paths[t]:
                tts_paths[t].append(p)
    return tts_paths


def load_audio(path):
    audio, _ = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32).flatten()
    mx = float(np.max(np.abs(audio))) if audio.size else 0.0
    if mx > 0:
        audio = audio / mx
    if len(audio) < EXPECTED_CHUNK_SAMPLES:
        audio = np.pad(audio, (0, EXPECTED_CHUNK_SAMPLES - len(audio)))
    elif len(audio) > EXPECTED_CHUNK_SAMPLES:
        audio = audio[:EXPECTED_CHUNK_SAMPLES]
    return audio


def encode_batch(model, audios, fe, device):
    inputs = fe(
        list(audios),
        sampling_rate=EXPECTED_SAMPLE_RATE,
        return_tensors="pt",
        padding=False,
    )
    features = inputs.input_features
    bsz, ch, ml = features.shape
    inp = features.transpose(0, 1).reshape(ch, -1).to(device).to(torch.bfloat16)
    fl = torch.full((bsz,), ml, dtype=torch.long, device=device)
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            embs = model(inp, fl)
        embs = embs.detach().cpu().float().numpy()
    faiss.normalize_L2(embs)
    return embs


def load_model(ckpt_path, device):
    model = Qwen3OmniRetriever(
        model_id="Atotti/Qwen3-Omni-AudioTransformer",
        target_dim=TARGET_DIM,
        use_lora=True,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
    ).to(device).to(torch.bfloat16)
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def build_tts_bank(tts_paths, term_to_idx, tts_model, fe, device):
    proto_idxs = []
    proto_paths_list = []
    for t, paths in tts_paths.items():
        idx = term_to_idx.get(t)
        if idx is None:
            continue
        for p in paths:
            if os.path.exists(p):
                proto_idxs.append(idx)
                proto_paths_list.append(p)

    proto_embs_parts = []
    for s in range(0, len(proto_paths_list), ENCODE_BATCH_SIZE):
        batch_p = proto_paths_list[s : s + ENCODE_BATCH_SIZE]
        audios = [load_audio(p) for p in batch_p]
        proto_embs_parts.append(encode_batch(tts_model, audios, fe, device))
    proto_embs = np.concatenate(proto_embs_parts, axis=0)
    proto_idx_np = np.array(proto_idxs, dtype=np.int64)
    valid_terms = sorted(set(proto_idxs))
    term_pos = {t: i for i, t in enumerate(valid_terms)}
    return proto_embs, proto_idx_np, valid_terms, term_pos


def search_tts_bank(speech_emb, proto_embs, proto_idx_np, valid_terms, term_pos):
    scores = proto_embs @ speech_emb
    bank_size = len(valid_terms)
    ts = np.full(bank_size, -np.inf, dtype=np.float32)
    for pi in range(scores.shape[0]):
        ti = int(proto_idx_np[pi])
        pos = term_pos.get(ti)
        if pos is not None and scores[pi] > ts[pos]:
            ts[pos] = float(scores[pi])
    valid_mask = np.isfinite(ts)
    vp = np.where(valid_mask)[0]
    vs = ts[vp]
    k = min(TOP_K, len(vs))
    top = np.argpartition(-vs, k - 1)[:k]
    top = top[np.argsort(-vs[top])]
    return [(valid_terms[vp[j]], float(vs[j])) for j in top]


def print_with_term_sample(si, cid, data, text_topk, tts_topk, idx_to_term, gt_idx):
    text_set = {t[0] for t in text_topk}
    tts_set = {t[0] for t in tts_topk}
    inter = text_set & tts_set
    text_only = text_set - tts_set
    tts_only = tts_set - text_set

    print(f"\n--- Sample {si} ---")
    print(f'  chunk_id: {cid}')
    print(f'  src_text: "{data["src"]}"')
    print(f"  GT terms: {data['gt_terms']}")

    print(f"  Text Top-{TOP_K} (semantic, ttsw=0.0):")
    for idx_val, name, sc in text_topk:
        hit = " << GT" if idx_val in gt_idx else ""
        marker = " [INTER]" if idx_val in inter else ""
        print(f"    {name:<40s} score={sc:.4f}{hit}{marker}")

    print(f"  TTS Top-{TOP_K} (acoustic, ttsw=1.0):")
    for idx_val, sc in tts_topk:
        name = idx_to_term.get(idx_val, "?")
        hit = " << GT" if idx_val in gt_idx else ""
        marker = " [INTER]" if idx_val in inter else ""
        print(f"    {name:<40s} score={sc:.4f}{hit}{marker}")

    gt_in_inter = gt_idx & inter
    gt_in_text_only = gt_idx & text_only
    gt_in_tts_only = gt_idx & tts_only
    gt_missed = gt_idx - (text_set | tts_set)

    print(f"  Intersection ({len(inter)} terms):")
    for idx_val in sorted(inter):
        name = idx_to_term.get(idx_val, "?")
        hit = " << GT" if idx_val in gt_idx else " (noise)"
        print(f"    + {name}{hit}")

    print(f"  Text-only, filtered out ({len(text_only)} terms):")
    for idx_val in sorted(text_only):
        name = idx_to_term.get(idx_val, "?")
        hit = " << GT LOST!" if idx_val in gt_idx else " (noise removed)"
        print(f"    - {name}{hit}")

    print(f"  TTS-only, filtered out ({len(tts_only)} terms):")
    for idx_val in sorted(tts_only):
        name = idx_to_term.get(idx_val, "?")
        hit = " << GT LOST!" if idx_val in gt_idx else " (noise removed)"
        print(f"    - {name}{hit}")

    print(f"  Summary: GT={len(gt_idx)}, "
          f"GT in intersection={len(gt_in_inter)}, "
          f"GT lost in text-only={len(gt_in_text_only)}, "
          f"GT lost in TTS-only={len(gt_in_tts_only)}, "
          f"GT missed entirely={len(gt_missed)}")


def main():
    device = torch.device(os.environ.get("OFFLINE_EVAL_DEVICE", "cuda:0"))
    print(f"[INFO] device={device}")

    print("[INFO] Loading dev data...")
    groups = load_dev_data()
    with_term = [(cid, d) for cid, d in groups.items() if d["gt_terms"]]
    no_term = [(cid, d) for cid, d in groups.items() if not d["gt_terms"]]
    print(f"[INFO] with_term={len(with_term)}, no_term={len(no_term)}")

    print("[INFO] Loading TTS paths...")
    tts_paths = load_tts_paths()

    print("[INFO] Loading FAISS index...")
    with open(DUAL_INDEX_PKL, "rb") as f:
        idx_data = pickle.load(f)
    term_list = idx_data["term_list"]
    idx_to_term = {i: item["key"] for i, item in enumerate(term_list)}
    term_to_idx = {item["key"]: i for i, item in enumerate(term_list)}

    fe = WhisperFeatureExtractor.from_pretrained(WHISPER_FE)

    print("[INFO] Loading dual text model (ttsw=0.0)...")
    text_model = load_model(DUAL_TEXT_MODEL_PATH, device)
    text_faiss = faiss.deserialize_index(idx_data["faiss_index"])

    print("[INFO] Loading dual TTS model (ttsw=1.0)...")
    tts_model = load_model(DUAL_TTS_MODEL_PATH, device)

    print("[INFO] Building TTS proto bank...")
    proto_embs, proto_idx_np, valid_terms, term_pos = build_tts_bank(
        tts_paths, term_to_idx, tts_model, fe, device
    )
    print(f"[INFO] TTS bank: {len(valid_terms)} terms, {proto_embs.shape[0]} prototypes")

    np.random.seed(RANDOM_SEED)
    wt_indices = np.random.choice(len(with_term), size=NUM_WITH_TERM_SAMPLES, replace=False)
    nt_indices = np.random.choice(len(no_term), size=NUM_NO_TERM_SAMPLES, replace=False)

    print()
    print("=" * 100)
    print("QUALITATIVE EXAMPLES: WITH-TERM CHUNKS")
    print("=" * 100)

    for si, idx in enumerate(wt_indices, 1):
        cid, data = with_term[idx]
        audio = load_audio(data["audio_path"])
        text_emb = encode_batch(text_model, [audio], fe, device)
        tts_emb = encode_batch(tts_model, [audio], fe, device)

        D, I = text_faiss.search(text_emb, TOP_K)
        text_topk = [
            (int(I[0][j]), idx_to_term.get(int(I[0][j]), "?"), float(D[0][j]))
            for j in range(TOP_K)
            if int(I[0][j]) >= 0
        ]

        tts_topk = search_tts_bank(
            tts_emb[0], proto_embs, proto_idx_np, valid_terms, term_pos
        )

        gt_idx = {term_to_idx[t] for t in data["gt_terms"] if t in term_to_idx}
        print_with_term_sample(si, cid, data, text_topk, tts_topk, idx_to_term, gt_idx)

    print()
    print("=" * 100)
    print("QUALITATIVE EXAMPLES: NO-TERM CHUNKS (all predictions = noise)")
    print("=" * 100)

    for si, idx in enumerate(nt_indices, 1):
        cid, data = no_term[idx]
        audio = load_audio(data["audio_path"])
        text_emb = encode_batch(text_model, [audio], fe, device)
        tts_emb = encode_batch(tts_model, [audio], fe, device)

        D, I = text_faiss.search(text_emb, TOP_K)
        text_topk = [
            (int(I[0][j]), idx_to_term.get(int(I[0][j]), "?"), float(D[0][j]))
            for j in range(TOP_K)
            if int(I[0][j]) >= 0
        ]
        tts_topk = search_tts_bank(
            tts_emb[0], proto_embs, proto_idx_np, valid_terms, term_pos
        )

        text_set = {t[0] for t in text_topk}
        tts_set = {t[0] for t in tts_topk}
        inter = text_set & tts_set

        print(f"\n--- No-Term Sample {si} ---")
        print(f'  chunk_id: {cid}')
        print(f'  src_text: "{data["src"]}"')
        print(f"  Text Top-{TOP_K}: {[idx_to_term.get(t[0], '?') for t in text_topk]}")
        print(f"  TTS  Top-{TOP_K}: {[idx_to_term.get(t[0], '?') for t in tts_topk]}")
        print(
            f"  Intersection ({len(inter)} noise terms): "
            f"{[idx_to_term.get(i, '?') for i in sorted(inter)]}"
        )
        if not inter:
            print("  => Intersection removed ALL noise")

    print()
    print("[DONE]")


if __name__ == "__main__":
    main()
