import json
import os
import tarfile
import io
import concurrent.futures
from tqdm import tqdm
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局变量，用于子进程共享内存，避免序列化开销
G_LINES = None

def init_worker(lines):
    global G_LINES
    G_LINES = lines

def process_single_shard(shard_info):
    """
    单个分片的打包任务
    """
    shard_idx, start_idx, end_idx, output_dir, prefix = shard_info
    shard_path = os.path.join(output_dir, f"{prefix}_{shard_idx:05d}.tar")
    
    # 修改点：跳过已经存在且非空的文件
    if os.path.exists(shard_path) and os.path.getsize(shard_path) > 0:
        return None # 返回 None 表示跳过

    count = 0
    try:
        with tarfile.open(shard_path, "w") as tar:
            for i in range(start_idx, end_idx):
                try:
                    sample = json.loads(G_LINES[i])
                    audio_path = sample["chunk_audio_path"]
                    
                    if not os.path.exists(audio_path):
                        continue
                    
                    basename = os.path.basename(audio_path)
                    name_no_ext = os.path.splitext(basename)[0]
                    
                    # 添加音频
                    tar.add(audio_path, arcname=f"{name_no_ext}.wav")
                    
                    # 添加 JSON 元数据
                    meta_bytes = json.dumps(sample, ensure_ascii=False).encode('utf-8')
                    meta_info = tarfile.TarInfo(name=f"{name_no_ext}.json")
                    meta_info.size = len(meta_bytes)
                    tar.addfile(meta_info, io.BytesIO(meta_bytes))
                    count += 1
                except Exception:
                    continue
        return f"Shard {shard_idx} finished: {count} samples."
    except Exception as e:
        if os.path.exists(shard_path):
            os.remove(shard_path) # 出错则删除不完整文件
        return f"Shard {shard_idx} failed: {e}"

def create_shards_parallel(input_jsonl, output_dir, prefix, samples_per_shard=20000, max_workers=16):
    """
    并行打包
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Reading {input_jsonl} into memory...")
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_samples = len(lines)
    num_shards = (total_samples + samples_per_shard - 1) // samples_per_shard
    
    logger.info(f"Total: {total_samples} -> {num_shards} shards. Workers: {max_workers}")

    # 准备任务参数（不再传递巨大的 lines，只传递索引）
    shard_tasks = []
    for shard_idx in range(num_shards):
        start_idx = shard_idx * samples_per_shard
        end_idx = min(start_idx + samples_per_shard, total_samples)
        shard_tasks.append((shard_idx, start_idx, end_idx, output_dir, prefix))

    # 使用 initializer 将 lines 注入到每个子进程的全局空间
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers, 
        initializer=init_worker, 
        initargs=(lines,)
    ) as executor:
        results = list(tqdm(executor.map(process_single_shard, shard_tasks), total=num_shards, desc=prefix))
    
    finished = [r for r in results if r is not None]
    logger.info(f"Completed {len(finished)} shards, skipped {len(results) - len(finished)} shards.")

def main():
    MAX_WORKERS = 16
    tasks = [
        {
            "jsonl": "/mnt/gemini/data1/jiaxuanluo/term_train_dataset.jsonl",
            "output": "/mnt/gemini/data2/jiaxuanluo/term_mmp_shards/train",
            "prefix": "train_shard"
        },
        {
            "jsonl": "/mnt/gemini/data1/jiaxuanluo/term_dev_dataset.jsonl",
            "output": "/mnt/gemini/data2/jiaxuanluo/term_mmp_shards/dev",
            "prefix": "dev_shard"
        }
    ]
    
    for task in tasks:
        create_shards_parallel(task["jsonl"], task["output"], task["prefix"], max_workers=MAX_WORKERS)

if __name__ == "__main__":
    main()
