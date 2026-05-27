import os
import io
import json
import tarfile
from tqdm import tqdm
from multiprocessing import Pool

import numpy as np
import soundfile as sf
from transformers import WhisperFeatureExtractor

# ================= 🚀 配置区 =================
# 任务列表: (原始JSONL, 输出子目录)
TASKS = [
    ("/mnt/gemini/data1/jiaxuanluo/term_train_dataset.jsonl", "train"),
    ("/mnt/gemini/data1/jiaxuanluo/term_dev_dataset.jsonl", "dev")
]

# 目标输出根目录 (Mel/Feature shards)
OUTPUT_ROOT = "/mnt/data2/jiaxuanluo/gigaspeech_webdataset_fbank_v1/"

# 每个 Tar 包放多少个样本
SAMPLES_PER_SHARD = 5000

# 并发数：通常应 <= SLURM cpus-per-task
NUM_WORKERS = 16

# Whisper 输入波形固定长度（与训练脚本的 collate_fn 保持一致）
TARGET_AUDIO_LEN = 30720  # 1.92s @ 16kHz

# 每次送入 WhisperFeatureExtractor 的 batch（worker 内部小 batch，加速特征提取）
FE_BATCH_SIZE = 32
# =============================================

_FE = None

def _init_worker():
    global _FE
    # FeatureExtractor 很轻，但初始化也有开销，放到 worker initializer 里复用
    _FE = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")

def _normalize_and_fixlen(audio: np.ndarray) -> np.ndarray:
    """Mono + normalize + pad/trim to fixed length."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32, copy=False)
    m = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
    if m > 0:
        audio = audio / m
    if audio.shape[0] < TARGET_AUDIO_LEN:
        audio = np.pad(audio, (0, TARGET_AUDIO_LEN - audio.shape[0]), mode="constant")
    elif audio.shape[0] > TARGET_AUDIO_LEN:
        audio = audio[:TARGET_AUDIO_LEN]
    return audio

def pack_shard(args):
    """
    Worker 进程：读取一组 JSONL 行，加载音频，预计算 mel(feature)，写入一个 Tar。
    Tar 内部存:
      - {key}.fbank : numpy .npy bytes (float16, shape [80, T])
      - {key}.json  : metadata (includes line_idx)
    key 使用 JSONL 行号，保证唯一。
    """
    shard_idx, line_items, output_dir = args
    shard_name = os.path.join(output_dir, f"shard_{shard_idx:05d}.tar")
    
    if os.path.exists(shard_name) and os.path.getsize(shard_name) > 0:
        return None

    try:
        with tarfile.open(shard_name, "w") as tar:
            audios = []
            metas = []
            keys = []

            for line_idx, line in line_items:
                try:
                    sample = json.loads(line)
                except Exception:
                    continue

                audio_path = sample.get("chunk_audio_path")
                if not audio_path or not os.path.exists(audio_path):
                    continue

                try:
                    audio, sr = sf.read(audio_path)
                except Exception:
                    continue

                # 仅支持 16kHz；若不是 16kHz 直接跳过（避免引入重依赖）
                if sr != 16000:
                    continue

                audio = _normalize_and_fixlen(audio)

                # 唯一 key 使用 JSONL 行号
                unique_key = f"{line_idx:08d}"
                sample["line_idx"] = line_idx  # 训练脚本依赖该字段用于 mmap 索引
                sample["global_idx"] = line_idx

                audios.append(audio)
                metas.append(sample)
                keys.append(unique_key)

            # 没有可用样本也要产出一个空 tar？这里直接返回，避免生成空文件
            if not audios:
                return f"Shard {shard_idx} empty"

            global _FE
            if _FE is None:
                _init_worker()

            # 批量提取特征（worker 内部小 batch）
            for start in range(0, len(audios), FE_BATCH_SIZE):
                batch_audios = audios[start:start + FE_BATCH_SIZE]
                batch_metas = metas[start:start + FE_BATCH_SIZE]
                batch_keys = keys[start:start + FE_BATCH_SIZE]

                feats = _FE(batch_audios, sampling_rate=16000, return_tensors="np", padding=False).input_features
                # feats: [B, 80, T]
                feats = feats.astype(np.float16, copy=False)

                for i in range(feats.shape[0]):
                    key = batch_keys[i]
                    meta = batch_metas[i]
                    fbank = feats[i]  # [80, T]

                    # 1) 写入 fbank（用 .npy 格式序列化，但扩展名用 .fbank，便于 webdataset key）
                    fbank_buf = io.BytesIO()
                    np.save(fbank_buf, fbank, allow_pickle=False)
                    fbank_bytes = fbank_buf.getvalue()

                    fbank_info = tarfile.TarInfo(name=f"{key}.fbank")
                    fbank_info.size = len(fbank_bytes)
                    tar.addfile(fbank_info, io.BytesIO(fbank_bytes))

                    # 2) 写入元数据
                    json_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")
                    json_info = tarfile.TarInfo(name=f"{key}.json")
                    json_info.size = len(json_bytes)
                    tar.addfile(json_info, io.BytesIO(json_bytes))
                    
        return f"Shard {shard_idx} done"
    except Exception as e:
        return f"Error in shard {shard_idx}: {e}"

def main():
    for jsonl_path, sub_dir in TASKS:
        output_dir = os.path.join(OUTPUT_ROOT, sub_dir)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nReading JSONL and creating shard tasks: {jsonl_path} -> {output_dir}")

        # 流式读取 JSONL，按 shard 切分为 (shard_idx, [(line_idx, line_str), ...])
        shard_tasks = []
        current = []
        shard_idx = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                current.append((line_idx, line))
                if len(current) >= SAMPLES_PER_SHARD:
                    shard_tasks.append((shard_idx, current, output_dir))
                    shard_idx += 1
                    current = []

        if current:
            shard_tasks.append((shard_idx, current, output_dir))

        print(f"Dataset: {sub_dir} | Shards: {len(shard_tasks)} | Workers: {NUM_WORKERS}")

        with Pool(processes=NUM_WORKERS, initializer=_init_worker) as p:
            for _ in tqdm(p.imap_unordered(pack_shard, shard_tasks), total=len(shard_tasks), desc=f"Packing {sub_dir}", unit="shard"):
                pass

    print("\nDone! FBank shards written to: " + OUTPUT_ROOT)

if __name__ == "__main__":
    main()