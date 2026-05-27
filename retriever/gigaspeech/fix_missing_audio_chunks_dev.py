import os
import json
import ast
import logging
import soundfile as sf
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# 配置路径
INPUT_JSONL = "/mnt/gemini/data1/jiaxuanluo/term_dev_dataset_final.jsonl"
INPUT_TSV = "/mnt/gemini/data1/jiaxuanluo/dev_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv"
OUTPUT_DIR = "/mnt/gemini/data1/jiaxuanluo/term_dev_audio_chunks"

SAMPLE_RATE = 16000
UNIT_DURATION_SEC = 0.96
SAMPLES_PER_UNIT = int(SAMPLE_RATE * UNIT_DURATION_SEC) # 15360

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. 扫描 JSONL 找出缺失的音频
    missing_map = defaultdict(set) # utter_id -> set(chunk_indices)
    total_lines = 0
    missing_count = 0
    
    logger.info(f"Scanning {INPUT_JSONL} for missing audio...")
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            item = json.loads(line)
            path = item.get("chunk_audio_path")
            if not os.path.exists(path):
                uid = item["utter_id"]
                cidx = item["chunk_idx"]
                missing_map[uid].add(cidx)
                missing_count += 1

    if not missing_map:
        logger.info("No missing audio chunks found. Everything is fine!")
        return

    logger.info(f"Found {missing_count} missing chunks across {len(missing_map)} utterances (Total lines: {total_lines})")

    # 2. 遍历 TSV 补全音频
    logger.info(f"Reading TSV {INPUT_TSV} to regenerate chunks...")
    with open(INPUT_TSV, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        col_map = {name: i for i, name in enumerate(header)}
        
        for line in tqdm(f, desc="Processing TSV"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header): continue
            
            uid = parts[col_map["id"]]
            if uid not in missing_map:
                continue
            
            # 匹配到了需要补全的 utter_id
            audio_info = parts[col_map["audio"]]
            try:
                audio_path, start_frame, total_frames = audio_info.split(":")
                start_frame = int(start_frame)
                total_frames = int(total_frames)
            except:
                logger.error(f"Invalid audio info for {uid}: {audio_info}")
                continue
            
            # 读取完整音频段
            try:
                full_audio_data, sr = sf.read(audio_path, start=start_frame, frames=total_frames)
                if sr != SAMPLE_RATE:
                    # 如果采样率不对，这里理论上应该重采样，但保持与原脚本一致
                    pass
            except Exception as e:
                logger.error(f"Failed to read source audio {audio_path}: {e}")
                continue
            
            # 补全该 utter_id 下所有缺失的 chunk
            for j in missing_map[uid]:
                chunk_start_rel = j * SAMPLES_PER_UNIT
                chunk_end_rel = (j + 2) * SAMPLES_PER_UNIT
                
                if chunk_start_rel >= len(full_audio_data):
                    logger.warning(f"Chunk index {j} out of range for {uid}")
                    continue
                
                chunk_data = full_audio_data[chunk_start_rel : min(chunk_end_rel, len(full_audio_data))]
                if len(chunk_data) == 0: continue
                
                out_path = os.path.join(OUTPUT_DIR, f"{uid}_chunk_{j}.wav")
                sf.write(out_path, chunk_data, SAMPLE_RATE)
                
            # 从待处理名单中移除
            del missing_map[uid]
            if not missing_map:
                break

    logger.info("Regeneration complete.")

if __name__ == "__main__":
    main()