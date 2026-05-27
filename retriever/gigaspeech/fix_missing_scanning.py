import os
import json
from collections import defaultdict

INPUT_JSONL = "/mnt/gemini/data1/jiaxuanluo/term_train_dataset_final.jsonl"
OUTPUT_JSON = "/mnt/gemini/data1/jiaxuanluo/missing_chunks_info.json"
CHUNKS_DIR = "/mnt/gemini/data1/jiaxuanluo/term_train_audio_chunks"

missing_map = defaultdict(list)
missing_count = 0
line_count = 0

print(f"Listing files in {CHUNKS_DIR}...")
# 一次性列出所有文件名，利用 Set 进行 O(1) 查找
try:
    existing_files = set(os.listdir(CHUNKS_DIR))
except Exception as e:
    print(f"Error listing directory {CHUNKS_DIR}: {e}")
    existing_files = set()

print(f"Found {len(existing_files)} files in directory. Starting scan...")

with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        line_count += 1
        
        # 每 100,000 行打印一次总进度（因为现在跑得飞快）
        if line_count % 100000 == 0:
            print(f"Processed {line_count} lines... (Found {missing_count} missing so far)")
            
        try:
            item = json.loads(line)
            path = item.get("chunk_audio_path")
            if not path:
                continue
            
            filename = os.path.basename(path)
            
            # 使用 Set 查找，绕过昂贵的磁盘 stat 操作
            if filename not in existing_files:
                uid = item["utter_id"]
                cidx = item["chunk_idx"]
                missing_map[uid].append(cidx)
                missing_count += 1
                
                # 打印前 50 个缺失的文件，避免日志爆炸
                if missing_count <= 50:
                    print(f"[MISSING] {filename} (ID: {uid}, Chunk: {cidx})")
                elif missing_count == 51:
                    print("... and more missing chunks (suppressing further [MISSING] logs) ...")
                
        except Exception as e:
            if line_count < 10: # 只在开始时打印错误
                print(f"Error parsing line {line_count}: {e}")
            continue

if missing_count == 0:
    print(f"\nScanning complete. Total lines: {line_count}")
    print("Everything is complete! No missing chunks.")
else:
    print(f"\nScanning complete. Total lines: {line_count}")
    print(f"Found {missing_count} missing chunks. Saving info to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(dict(missing_map), f)
