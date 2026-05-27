import json
import zhconv
from tqdm import tqdm

input_path = "/mnt/gemini/data1/jiaxuanluo/train_s_zh_baseline.jsonl"
#input_path = "/mnt/gemini/data1/jiaxuanluo/train_m_zh_baseline.jsonl"
output_path = "/mnt/gemini/data1/jiaxuanluo/train_s_zh_baseline_simplified.jsonl"

print(f"Converting {input_path} to Simplified Chinese...")

with open(input_path, 'r', encoding='utf-8') as f_in, \
     open(output_path, 'w', encoding='utf-8') as f_out:
    for line in tqdm(f_in):
        if not line.strip():
            f_out.write(line)
            continue
        # Convert the entire line to simplified
        simplified_line = zhconv.convert(line, 'zh-cn')
        f_out.write(simplified_line)

print(f"Done. Simplified file saved to {output_path}")


















