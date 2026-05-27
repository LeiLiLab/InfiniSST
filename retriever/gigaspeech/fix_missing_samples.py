import json
import os
import shutil
from tqdm import tqdm

SRC_JSONL = "/mnt/gemini/data1/jiaxuanluo/term_train_dataset.jsonl"
DST_JSONL = "/mnt/data2/jiaxuanluo/local_train_dataset.jsonl"
SRC_BASE = "/mnt/gemini/data1/jiaxuanluo/"
DST_ROOT = "/mnt/data2/jiaxuanluo/local_wavs_from_shards/"

# 从这个索引开始补 (对应原始文件行号，0-indexed)
START_IDX = 8242821 

def fix_remaining():
    # 确保输出目录存在
    os.makedirs(DST_ROOT, exist_ok=True)
    
    print(f"Starting to supplement from index {START_IDX}...")
    
    # 用 append 模式打开
    with open(DST_JSONL, "a", encoding="utf-8") as out_f:
        with open(SRC_JSONL, "r", encoding="utf-8") as in_f:
            pbar = tqdm(total=8334585)
            for i, line in enumerate(in_f):
                pbar.update(1)
                if i < START_IDX:
                    continue
                
                curr_idx = i
                try:
                    item = json.loads(line.strip())
                except:
                    continue
                
                orig_path = item.get("chunk_audio_path", "")
                if not orig_path:
                    continue
                
                # 计算相对路径并构建本地路径
                rel_path = os.path.relpath(orig_path, SRC_BASE)
                local_path = os.path.join(DST_ROOT, rel_path)
                
                # 复制音频文件
                if not os.path.exists(local_path):
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    if os.path.exists(orig_path):
                        shutil.copy2(orig_path, local_path)
                
                # 更新 JSON 信息
                item["chunk_audio_path"] = local_path
                item["global_idx"] = curr_idx
                item["line_idx"] = curr_idx
                
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            pbar.close()
                
    print(f"Supplement completed.")

if __name__ == "__main__":
    fix_remaining()
