"""
Full qualitative analysis: text vs TTS top-k overlap for ALL with-term chunks,
then show representative examples at different intersection sizes.
"""
import sys
import json
import os
import pickle
import numpy as np
from pathlib import Path
from collections import Counter

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
ENCODE_BATCH_SIZE = 64
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


def print_detailed_example(rank, cid, data, text_topk, tts_topk, idx_to_term, gt_idx):
    text_set = {t[0] for t in text_topk}
    tts_set = {t[0] for t in tts_topk}
    inter = text_set & tts_set
    text_only = text_set - tts_set
    tts_only = tts_set - text_set

    gt_in_inter = gt_idx & inter
    gt_in_text_only = gt_idx & text_only
    gt_in_tts_only = gt_idx & tts_only
    gt_missed = gt_idx - (text_set | tts_set)
    noise_in_inter = inter - gt_idx

    print(f"\n{'='*90}")
    print(f"  chunk_id : {cid}")
    print(f"  src_text : \"{data['src']}\"")
    gt_names = sorted(data["gt_terms"])
    print(f"  GT terms : {gt_names}  (count={len(gt_names)})")
    print(f"{'='*90}")

    # Side-by-side display
    print(f"\n  {'Rank':<5} {'Text Top-10 (semantic)':<45} {'TTS Top-10 (acoustic)':<45}")
    print(f"  {'----':<5} {'-'*43:<45} {'-'*43:<45}")
    for r in range(TOP_K):
        t_idx, t_name, t_sc = text_topk[r] if r < len(text_topk) else (-1, "", 0)
        tts_entry = tts_topk[r] if r < len(tts_topk) else ((-1, 0))
        tts_idx, tts_sc = tts_entry[0], tts_entry[1]
        tts_name = idx_to_term.get(tts_idx, "?")

        t_gt = "*" if t_idx in gt_idx else " "
        t_inter = "^" if t_idx in inter else " "
        tts_gt = "*" if tts_idx in gt_idx else " "
        tts_inter = "^" if tts_idx in inter else " "

        t_str = f"{t_gt}{t_inter} {t_name:<35s} {t_sc:.4f}"
        tts_str = f"{tts_gt}{tts_inter} {tts_name:<35s} {tts_sc:.4f}"
        print(f"  {r+1:<5} {t_str:<45} {tts_str:<45}")

    print(f"\n  Legend: * = GT term, ^ = in intersection")

    print(f"\n  INTERSECTION ({len(inter)} terms):")
    for idx_val in sorted(inter):
        name = idx_to_term.get(idx_val, "?")
        label = "GT" if idx_val in gt_idx else "NOISE"
        print(f"    {name:<40s} [{label}]")

    print(f"\n  FILTERED OUT by intersection ({len(text_only) + len(tts_only)} terms):")
    for idx_val in sorted(text_only):
        name = idx_to_term.get(idx_val, "?")
        label = "GT LOST" if idx_val in gt_idx else "noise removed"
        print(f"    {name:<40s} [text-only, {label}]")
    for idx_val in sorted(tts_only):
        name = idx_to_term.get(idx_val, "?")
        label = "GT LOST" if idx_val in gt_idx else "noise removed"
        print(f"    {name:<40s} [tts-only, {label}]")

    print(f"\n  VERDICT: intersection_size={len(inter)}, "
          f"GT_kept={len(gt_in_inter)}/{len(gt_idx)}, "
          f"noise_kept={len(noise_in_inter)}, "
          f"noise_removed={len(text_only) + len(tts_only) - len(gt_in_text_only) - len(gt_in_tts_only)}")


def main():
    device = torch.device(os.environ.get("OFFLINE_EVAL_DEVICE", "cuda:0"))

    print("[INFO] Loading data...")
    groups = load_dev_data()
    with_term = [(cid, d) for cid, d in groups.items() if d["gt_terms"]]
    print(f"[INFO] with_term={len(with_term)}")

    tts_paths = load_tts_paths()

    print("[INFO] Loading index & models...")
    with open(DUAL_INDEX_PKL, "rb") as f:
        idx_data = pickle.load(f)
    term_list = idx_data["term_list"]
    idx_to_term = {i: item["key"] for i, item in enumerate(term_list)}
    term_to_idx = {item["key"]: i for i, item in enumerate(term_list)}

    fe = WhisperFeatureExtractor.from_pretrained(WHISPER_FE)
    text_model = load_model(DUAL_TEXT_MODEL_PATH, device)
    text_faiss = faiss.deserialize_index(idx_data["faiss_index"])
    tts_model = load_model(DUAL_TTS_MODEL_PATH, device)

    print("[INFO] Building TTS bank...")
    proto_embs, proto_idx_np, valid_terms, term_pos = build_tts_bank(
        tts_paths, term_to_idx, tts_model, fe, device
    )
    print(f"[INFO] TTS bank: {len(valid_terms)} terms, {proto_embs.shape[0]} prototypes")

    # Run ALL with-term chunks
    print("[INFO] Running all with-term chunks...")
    all_results = []
    for i in range(0, len(with_term), ENCODE_BATCH_SIZE):
        batch = with_term[i : i + ENCODE_BATCH_SIZE]
        audios = [load_audio(d["audio_path"]) for _, d in batch]

        text_embs = encode_batch(text_model, audios, fe, device)
        tts_embs = encode_batch(tts_model, audios, fe, device)

        D_text, I_text = text_faiss.search(text_embs, TOP_K)

        for j, (cid, data) in enumerate(batch):
            text_topk = [
                (int(I_text[j][r]), idx_to_term.get(int(I_text[j][r]), "?"), float(D_text[j][r]))
                for r in range(TOP_K) if int(I_text[j][r]) >= 0
            ]
            tts_topk = search_tts_bank(
                tts_embs[j], proto_embs, proto_idx_np, valid_terms, term_pos
            )

            text_set = {t[0] for t in text_topk}
            tts_set = {t[0] for t in tts_topk}
            inter = text_set & tts_set
            gt_idx = {term_to_idx[t] for t in data["gt_terms"] if t in term_to_idx}
            gt_in_inter = gt_idx & inter
            noise_in_inter = inter - gt_idx

            all_results.append({
                "cid": cid,
                "data": data,
                "text_topk": text_topk,
                "tts_topk": tts_topk,
                "inter_size": len(inter),
                "gt_count": len(gt_idx),
                "gt_kept": len(gt_in_inter),
                "noise_kept": len(noise_in_inter),
                "gt_idx": gt_idx,
            })

        done = min(i + ENCODE_BATCH_SIZE, len(with_term))
        print(f"  [{done}/{len(with_term)}]")

    # Distribution
    inter_sizes = [r["inter_size"] for r in all_results]
    gt_kept_all = [r["gt_kept"] for r in all_results]
    noise_kept_all = [r["noise_kept"] for r in all_results]
    gt_counts = [r["gt_count"] for r in all_results]

    print("\n" + "=" * 90)
    print("DISTRIBUTION: Intersection size across ALL with-term chunks")
    print("=" * 90)
    size_counter = Counter(inter_sizes)
    for size in sorted(size_counter.keys()):
        bar = "#" * size_counter[size]
        print(f"  inter_size={size:<3d}: {size_counter[size]:>5d} chunks  {bar}")

    print(f"\n  Mean intersection size: {np.mean(inter_sizes):.2f}")
    print(f"  Median:                 {np.median(inter_sizes):.1f}")
    print(f"  Min/Max:                {min(inter_sizes)} / {max(inter_sizes)}")

    print(f"\n  Mean GT kept:    {np.mean(gt_kept_all):.2f} / {np.mean(gt_counts):.2f}")
    print(f"  Mean noise kept: {np.mean(noise_kept_all):.2f}")

    # Breakdown: for each intersection size, how many are GT vs noise
    print("\n" + "-" * 90)
    print("BREAKDOWN: What's in the intersection? (GT vs noise)")
    print("-" * 90)
    print(f"  {'Inter size':<12} {'Count':<8} {'Avg GT kept':<14} {'Avg noise':<12} {'Avg GT total':<14}")
    for size in sorted(size_counter.keys()):
        subset = [r for r in all_results if r["inter_size"] == size]
        avg_gt = np.mean([r["gt_kept"] for r in subset])
        avg_noise = np.mean([r["noise_kept"] for r in subset])
        avg_gt_total = np.mean([r["gt_count"] for r in subset])
        print(f"  {size:<12d} {len(subset):<8d} {avg_gt:<14.2f} {avg_noise:<12.2f} {avg_gt_total:<14.2f}")

    # Pick representative examples at each intersection size
    print("\n" + "=" * 90)
    print("REPRESENTATIVE EXAMPLES by intersection size")
    print("=" * 90)

    target_sizes = sorted(size_counter.keys())
    np.random.seed(42)
    shown = 0
    max_examples = 15
    for size in target_sizes:
        if shown >= max_examples:
            break
        subset = [r for r in all_results if r["inter_size"] == size]
        pick = subset[np.random.randint(len(subset))]
        print_detailed_example(
            shown + 1, pick["cid"], pick["data"],
            pick["text_topk"], pick["tts_topk"],
            idx_to_term, pick["gt_idx"],
        )
        shown += 1

    print("\n[DONE]")


if __name__ == "__main__":
    main()
