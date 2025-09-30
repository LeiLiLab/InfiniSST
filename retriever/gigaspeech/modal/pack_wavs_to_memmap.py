#!/usr/bin/env python3
"""
音频文件打包脚本：把 POD/YOU/AUD 下所有小 wav 合成若干大 .dat 分片
用于 Modal 上传和 mmap 随机读取
"""

import os, sys, json, math, glob, time
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

# ====== 配置参数 ======
ROOTS = [
    "/mnt/gemini/data1/jiaxuanluo/term_chunks/POD",
    "/mnt/gemini/data1/jiaxuanluo/term_chunks/YOU", 
    "/mnt/gemini/data1/jiaxuanluo/term_chunks/AUD",
]
OUT_DIR = "/mnt/gemini/data1/jiaxuanluo/mmap_shards"                # 输出目录（打包结果）
TARGET_SR = 16_000                       # 统一采样率（已确认都是16k）
FORCE_RESAMPLE = False                   # 不需要重采样
MAX_SECONDS = 30.0                       # 每条音频最长，超出裁切
STORE_DTYPE = np.int16                   # int16 节省空间
SHARD_BYTES = 2 * (1024**3)              # 每个分片目标大小，~2GB/片
ALLOW_EXT = {".wav", ".flac", ".ogg"}    # 允许的音频后缀
PROGRESS_SAVE_INTERVAL = 1000            # 每处理多少个文件保存一次进度
# =================================

if FORCE_RESAMPLE:
    import librosa

def iter_all_wavs(roots):
    """遍历所有音频文件，生成 key 和路径"""
    for root in roots:
        root = Path(root)
        if not root.exists():
            print(f"[WARN] Root directory not found: {root}")
            continue
            
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in ALLOW_EXT:
                rel = p.relative_to(root)      # 相对各自根目录
                # key: 用 "顶级目录名/相对路径去后缀"，保证唯一且可回溯
                key = f"{root.name}/{rel.with_suffix('')}".replace("\\", "/")
                yield str(p), key, f"{root.name}/{rel}".replace("\\", "/")

def ensure_out():
    """确保输出目录存在"""
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

def open_new_shard(shard_id):
    """创建新的分片文件"""
    dat_path = Path(OUT_DIR, f"shard_{shard_id:05d}.dat")
    # 先创建空文件，后面用"追加写"直接 tofile
    with open(dat_path, "wb"):
        pass
    return dat_path

def finalize_index(shard_id, index_rows):
    """完成分片索引文件"""
    idx_path = Path(OUT_DIR, f"shard_{shard_id:05d}.index.npz")
    np.savez(
        idx_path,
        key=np.array([r["key"] for r in index_rows], dtype=object),
        relpath=np.array([r["relpath"] for r in index_rows], dtype=object),
        offset=np.array([r["offset"] for r in index_rows], dtype=np.int64),
        length=np.array([r["length"] for r in index_rows], dtype=np.int64),
        sr=np.full(len(index_rows), TARGET_SR, dtype=np.int32),
        dtype=np.array([STORE_DTYPE.__name__]*len(index_rows), dtype=object),
    )

def save_progress_checkpoint(processed_count, total_count, shard_id, current_time):
    """保存进度检查点"""
    progress_file = Path(OUT_DIR) / "progress.json"
    progress_data = {
        "processed_count": processed_count,
        "total_count": total_count,
        "current_shard": shard_id,
        "timestamp": current_time,
        "percentage": processed_count / total_count * 100 if total_count > 0 else 0
    }
    with open(progress_file, "w") as f:
        json.dump(progress_data, f, indent=2)
    print(f"[PROGRESS] Saved checkpoint: {processed_count}/{total_count} ({progress_data['percentage']:.1f}%)")

def load_and_standardize(path):
    """加载并标准化音频"""
    wav, sr = sf.read(path, always_2d=False)
    
    # 转 mono
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    
    # 检查采样率（应该都是16k）
    if sr != TARGET_SR:
        if not FORCE_RESAMPLE:
            raise ValueError(f"sr={sr} != {TARGET_SR} for {path}. 采样率不匹配")
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    
    # 裁切过长的音频
    max_len = int(TARGET_SR * MAX_SECONDS)
    if wav.shape[0] > max_len:
        wav = wav[:max_len]
    
    # 转换为 int16 节省空间
    if STORE_DTYPE == np.int16:
        wav = np.clip(wav, -1.0, 1.0)
        wav = (wav * 32767.0).astype(np.int16)
    else:
        wav = wav.astype(np.float32)
    
    return wav, sr


def _worker_load(args):
    abs_path, key, rel = args
    try:
        wav, sr = load_and_standardize(abs_path)
        return (key, rel, wav)
    except Exception as e:
        return ("__SKIP__", f"{abs_path} -> {e}", None)

def find_processed_keys():
    """查找已处理的文件key，用于续传"""
    processed_keys = set()
    shard_files = sorted(Path(OUT_DIR).glob("shard_*.index.npz"))
    
    for shard_file in shard_files:
        try:
            data = np.load(shard_file, allow_pickle=True)
            keys = data["key"]
            processed_keys.update(keys)
            print(f"[RESUME] Found {len(keys)} processed items in {shard_file.name}")
        except Exception as e:
            print(f"[WARN] Could not read {shard_file}: {e}")
    
    return processed_keys

def find_last_shard_info():
    """查找最后一个分片的信息"""
    shard_files = sorted(Path(OUT_DIR).glob("shard_*.dat"))
    if not shard_files:
        return 0, 0  # 从头开始
    
    # 找最大的分片ID
    last_shard_id = max(int(f.stem.split('_')[1]) for f in shard_files)
    
    # 检查对应的index文件是否存在
    index_file = Path(OUT_DIR) / f"shard_{last_shard_id:05d}.index.npz"
    if index_file.exists():
        # 该分片已完成，从下一个分片开始
        return last_shard_id + 1, 0
    else:
        # 该分片未完成，需要删除并重新开始
        dat_file = Path(OUT_DIR) / f"shard_{last_shard_id:05d}.dat"
        if dat_file.exists():
            print(f"[RESUME] Removing incomplete shard: {dat_file}")
            dat_file.unlink()
        return last_shard_id, 0

def main():
    """主函数（并行解码 + 主进程顺序写入，支持续传）"""
    ensure_out()

    # 收集所有音频文件
    print("[INFO] Scanning audio files...")
    all_files = list(iter_all_wavs(ROOTS))
    if not all_files:
        print("[ERROR] No audio files found!")
        return
    print(f"[INFO] Found {len(all_files)} audio files")
    
    # 检查已处理的文件
    print("[INFO] Checking for existing progress...")
    processed_keys = find_processed_keys()
    if processed_keys:
        print(f"[RESUME] Found {len(processed_keys)} already processed files")
        # 过滤掉已处理的文件
        files = [(path, key, rel) for path, key, rel in all_files if key not in processed_keys]
        print(f"[RESUME] {len(files)} files remaining to process")
    else:
        files = all_files
        print("[INFO] No previous progress found, starting from beginning")
    
    if not files:
        print("[INFO] All files have been processed!")
        return

    # 并行参数
    NUM_WORKERS   = min( max(1, (os.cpu_count() or 4) - 1), 32 )  # 保守一点，最多32
    MP_CHUNKSIZE  = 8

    # 确定起始分片ID
    shard_id, _ = find_last_shard_info()
    print(f"[RESUME] Starting from shard_{shard_id:05d}")
    
    # 开始打包
    dat_path      = open_new_shard(shard_id)
    written_bytes = 0
    index_rows    = []
    cur_offset    = 0

    print(f"[INFO] Starting audio packing with {NUM_WORKERS} workers...")
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # 用生成器避免一次性把所有结果放内存
    total = len(files)
    processed_count = 0
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        # 按块提交任务（更平衡）
        futures = []
        for i in range(0, total, MP_CHUNKSIZE):
            chunk = files[i:i+MP_CHUNKSIZE]
            for item in chunk:
                futures.append(ex.submit(_worker_load, item))

        from tqdm import tqdm
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Packing audio"):
            key, rel_or_msg, wav = fut.result()
            if key == "__SKIP__":
                print(f"[SKIP] {rel_or_msg}")
                continue

            rel = rel_or_msg
            n   = int(wav.shape[0])
            item_bytes = int(n * (2 if STORE_DTYPE == np.int16 else 4))

            # 若超过目标大小，换新分片
            if written_bytes > 0 and written_bytes + item_bytes > SHARD_BYTES:
                finalize_index(shard_id, index_rows)
                print(f"[DONE] shard_{shard_id:05d}: {len(index_rows)} items, {written_bytes/1024/1024:.1f} MB")
                shard_id      += 1
                dat_path       = open_new_shard(shard_id)
                written_bytes  = 0
                index_rows     = []
                cur_offset     = 0

            # 追加写入音频数据（主进程顺序写，避免写竞争）
            with open(dat_path, "ab") as f:
                wav.tofile(f)

            # 记录索引信息
            index_rows.append({
                "key": key,
                "relpath": rel,
                "offset": cur_offset,
                "length": n,
            })
            written_bytes += item_bytes
            cur_offset    += n
            processed_count += 1
            
            # 定期保存进度
            if processed_count % PROGRESS_SAVE_INTERVAL == 0:
                save_progress_checkpoint(processed_count, total, shard_id, time.time())

    # 收尾
    if index_rows:
        finalize_index(shard_id, index_rows)
        print(f"[DONE] shard_{shard_id:05d}: {len(index_rows)} items, {written_bytes/1024/1024:.1f} MB")

    # 最终进度报告
    end_time = time.time()
    elapsed = end_time - start_time
    save_progress_checkpoint(processed_count, total, shard_id, end_time)
    
    print(f"[INFO] All shards saved to: {OUT_DIR}")
    print(f"[INFO] Total shards: {shard_id + 1}")
    print(f"[INFO] Processed {processed_count} files in {elapsed:.1f} seconds")
    print(f"[INFO] Average speed: {processed_count/elapsed:.1f} files/sec")
    
    # 清理进度文件
    progress_file = Path(OUT_DIR) / "progress.json"
    if progress_file.exists():
        progress_file.unlink()
        print("[INFO] Cleaned up progress file")

if __name__ == "__main__":
    main()

