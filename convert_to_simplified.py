import json
import zhconv
from tqdm import tqdm

input_path = "/mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_freq_k20.jsonl"
output_path = "/mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_freq_k20_simplified.jsonl"

print(f"Converting {input_path} to Simplified Chinese...")

with open(input_path, 'r', encoding='utf-8') as f_in, \
     open(output_path, 'w', encoding='utf-8') as f_out:
    for line in tqdm(f_in):
        if not line.strip():
            continue
        # Convert the entire line to simplified
        simplified_line = zhconv.convert(line, 'zh-cn')
        f_out.write(simplified_line)

print(f"Done. Simplified file saved to {output_path}")


















