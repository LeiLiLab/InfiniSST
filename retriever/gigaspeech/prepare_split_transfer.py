import json
import os
from tqdm import tqdm

# ================= 配置区 =================
# 原始数据根目录 (用于计算相对路径)
SRC_BASE_DIR = "/mnt/gemini/data1/jiaxuanluo/"

# 任务列表: (原始JSONL, 目标本地JSONL)
TASKS = [
    {
        "orig_jsonl": "/mnt/gemini/data1/jiaxuanluo/term_train_dataset.jsonl",
        "new_jsonl": "local_train_dataset.jsonl"
    },
    {
        "orig_jsonl": "/mnt/gemini/data1/jiaxuanluo/term_dev_dataset.jsonl",
        "new_jsonl": "local_dev_dataset.jsonl"
    }
]

# 目标本地盘根目录 (现在全部放在 data2)
DEST_ROOT = "/mnt/data2/jiaxuanluo/"

# 输出的迁移清单文件 (供 rsync --files-from 使用)
LIST_FILE = "transfer_list_all.txt"
# ===========================================

def main():
    # 确保迁移清单是空的/新建的
    f_list = open(LIST_FILE, "w")

    for task in TASKS:
        orig_jsonl = task["orig_jsonl"]
        new_jsonl = task["new_jsonl"]
        
        print(f"Processing {orig_jsonl} -> {new_jsonl}...")
        
        with open(orig_jsonl, "r") as f_in, open(new_jsonl, "w") as f_out:
            for i, line in enumerate(tqdm(f_in, desc=os.path.basename(orig_jsonl))):
                try:
                    item = json.loads(line)
                    abs_audio_path = item["chunk_audio_path"]
                    
                    # 获取相对于 SRC_BASE_DIR 的路径
                    rel_path = os.path.relpath(abs_audio_path, SRC_BASE_DIR)
                    
                    # 写入总清单
                    f_list.write(rel_path + "\n")
                    
                    # 修改 JSONL 中的路径为本地磁盘的绝对路径
                    item["chunk_audio_path"] = os.path.join(DEST_ROOT, rel_path)
                    f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                    
                except Exception as e:
                    print(f"Error processing line {i} in {orig_jsonl}: {e}")

    f_list.close()

    print("\nDone! List and new JSONL files generated.")
    print(f"Next steps:")
    print(f"1. Run rsync to move all files: rsync -av --progress --files-from={LIST_FILE} {SRC_BASE_DIR} {DEST_ROOT}")
    print(f"2. Update your training/dev paths to use the new local_*.jsonl files.")

if __name__ == "__main__":
    main()
