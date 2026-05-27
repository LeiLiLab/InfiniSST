#!/usr/bin/env python3
"""Debug: direct model loading TTS recall check (bypass retriever)."""
import sys, os, json, random
sys.path.insert(0, "/home/jiaxuanluo/InfiniSST")
import torch, numpy as np, faiss, soundfile as sf
from transformers import WhisperFeatureExtractor
from agents.streaming_qwen3_rag_retriever_v4 import Qwen3OmniRetriever

TTS_MODEL_PATH = "/mnt/gemini/data/jiaxuanluo/q3rag_tts_lora-r32-tr16_bs4k_ttsw1.0_ttm=query key value_temperature=0.03_v2_epoch_7.pt"
DEVICE = "cuda:0"
SR = 16000
TARGET_LEN = 30720
DEV_WITH_TTS = "/mnt/gemini/data/siqiouyang/term_dev_dataset_final_with_tts.jsonl"
TOP_K = 10
BATCH_SIZE = 256

def load_audio(path):
    audio, _ = sf.read(path)
    audio = np.asarray(audio, dtype=np.float32).flatten()
    mx = float(np.max(np.abs(audio))) if audio.size else 0.0
    if mx > 0: audio = audio / mx
    if len(audio) < TARGET_LEN: audio = np.pad(audio, (0, TARGET_LEN - len(audio)))
    elif len(audio) > TARGET_LEN: audio = audio[:TARGET_LEN]
    return audio

def encode_batch(model, fe, audio_list, device):
    inputs = fe(audio_list, sampling_rate=SR, return_tensors="pt", padding=False)
    features = inputs.input_features
    B, C, T = features.shape
    inp = features.transpose(0, 1).reshape(C, -1).to(device).to(torch.bfloat16)
    lens = torch.full((B,), T, dtype=torch.long, device=device)
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            embs = model(inp, lens)
        embs = embs.detach().cpu().float().numpy()
    faiss.normalize_L2(embs)
    return embs

def main():
    print("=== Loading TTS model directly (same as training) ===")
    model = Qwen3OmniRetriever(
        model_id="Atotti/Qwen3-Omni-AudioTransformer",
        target_dim=1024, use_lora=True, lora_rank=32, lora_alpha=64,
    ).to(DEVICE).to(torch.bfloat16)
    ckpt = torch.load(TTS_MODEL_PATH, map_location=DEVICE)
    sd = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")

    with open(DEV_WITH_TTS) as f:
        all_lines = f.readlines()

    first = json.loads(all_lines[0])
    speech_path = first["chunk_audio_path"]
    tts_path = first["tts_audio_path"]
    term = first["term"]
    print(f"Sanity: term={term!r}")
    embs = encode_batch(model, fe, [load_audio(speech_path), load_audio(tts_path)], DEVICE)
    cos_gt = float(np.dot(embs[0], embs[1]))
    print(f"  speech vs own TTS cosine = {cos_gt:.6f}")

    random.seed(42)
    for line in random.sample(all_lines[:200], min(10, len(all_lines))):
        obj = json.loads(line)
        t = obj["term"]
        te = encode_batch(model, fe, [load_audio(obj["tts_audio_path"])], DEVICE)
        s = float(np.dot(embs[0], te[0]))
        marker = " << SAME" if t.lower() == term.lower() else ""
        print(f"  vs tts({t!r:30s}): sim={s:.6f}{marker}")

    print(f"\n=== Full TTS recall@{TOP_K} eval (direct model, dev set) ===")
    term_to_protos = {}
    chunk_data = []
    for line in all_lines:
        obj = json.loads(line)
        t = obj["term"].strip().lower()
        if not t: continue
        term_to_protos.setdefault(t, []).append(obj["tts_audio_path"].strip())
        chunk_data.append({"term": t, "speech_path": obj["chunk_audio_path"].strip()})

    unique_terms = sorted(term_to_protos.keys())
    term_to_idx = {t: i for i, t in enumerate(unique_terms)}
    print(f"Unique terms: {len(unique_terms)}, chunks: {len(chunk_data)}")

    proto_term_indices, proto_paths = [], []
    for t in unique_terms:
        for p in term_to_protos[t]:
            proto_term_indices.append(term_to_idx[t])
            proto_paths.append(p)
    print(f"Total prototypes: {len(proto_paths)}")

    proto_embs_list = []
    for start in range(0, len(proto_paths), BATCH_SIZE):
        audios = [load_audio(p) for p in proto_paths[start:start+BATCH_SIZE]]
        proto_embs_list.append(encode_batch(model, fe, audios, DEVICE))
    proto_embs = np.concatenate(proto_embs_list, axis=0)
    proto_term_idx_np = np.array(proto_term_indices, dtype=np.int64)
    print(f"Proto bank: {proto_embs.shape}")

    hits, total = 0, 0
    for start in range(0, len(chunk_data), BATCH_SIZE):
        batch = chunk_data[start:start+BATCH_SIZE]
        speech_embs = encode_batch(model, fe, [load_audio(c["speech_path"]) for c in batch], DEVICE)
        for i, c in enumerate(batch):
            gt_idx = term_to_idx[c["term"]]
            scores = proto_embs @ speech_embs[i]
            term_scores = np.full(len(unique_terms), -np.inf, dtype=np.float32)
            for pi in range(scores.shape[0]):
                ti = proto_term_idx_np[pi]
                if scores[pi] > term_scores[ti]: term_scores[ti] = scores[pi]
            k = min(TOP_K, len(unique_terms))
            top_idx = np.argpartition(-term_scores, k-1)[:k]
            if gt_idx in top_idx: hits += 1
            total += 1
        if total % 500 < BATCH_SIZE:
            print(f"  {total}/{len(chunk_data)}, recall@{TOP_K}={hits/total:.4f}")

    print(f"\n=== RESULT: Direct TTS recall@{TOP_K} = {hits/total:.4f} ({hits}/{total}) ===")

if __name__ == "__main__":
    main()
