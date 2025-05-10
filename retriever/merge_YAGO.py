import os, json

input_dir = "data/outputs"
output_path = "data/final_named_entities_info.json"

merged = {}
batch_size = 500
file_list = os.listdir(input_dir)

for i, f in enumerate(file_list):
    with open(os.path.join(input_dir, f), encoding="utf-8") as file:
        try:
            part = json.load(file)
            for term, info in part.items():
                if info["labels"] and info["comments"]:
                    merged[term] = info
        except Exception as e:
            print(f"[WARN] Failed to parse {f}: {e}")

    # 每 batch_size 个文件就写一次磁盘并清空缓存
    if (i + 1) % batch_size == 0 or (i + 1) == len(file_list):
        print(f"✅ Writing batch {i + 1} / {len(file_list)} with {len(merged)} entities...")
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        else:
            existing = {}

        existing.update(merged)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

        merged.clear()  # 清空内存