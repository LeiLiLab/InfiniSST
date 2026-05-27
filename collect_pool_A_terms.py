import json
import os
from tqdm import tqdm

input_files = [
    "/mnt/gemini/data1/jiaxuanluo/train_m_zh_v4_refined_pool_A_lagging.jsonl",
    "/mnt/gemini/data1/jiaxuanluo/train_m_zh_v4_refined_pool_A_GT.jsonl"
]
glossary_output = "/mnt/gemini/data1/jiaxuanluo/pool_A_glossary.json"

glossary = {}

print("Collecting terms from Pool A files...")
for file_path in input_files:
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        continue
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=os.path.basename(file_path)):
            if not line.strip(): continue
            instance = json.loads(line)
            gt_terms_by_chunk = instance.get('gt_terms_by_chunk', [])
            for chunk in gt_terms_by_chunk:
                for t in chunk:
                    term_en = t.get('term', '').strip()
                    term_zh = t.get('zh', '').strip()
                    if term_en and term_zh:
                        if term_en not in glossary:
                            glossary[term_en] = term_zh

print(f"Total unique terms collected: {len(glossary)}")
with open(glossary_output, 'w', encoding='utf-8') as f:
    json.dump(glossary, f, ensure_ascii=False, indent=2)

print(f"Glossary saved to: {glossary_output}")










