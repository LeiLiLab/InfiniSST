#!/usr/bin/env python3
import json
import os
from collections import defaultdict

def extract_terms():
    input_path = "/mnt/gemini/data1/jiaxuanluo/train_s_zh_v3_gt_terms_final_with_ner.jsonl"
    output_path = "/mnt/gemini/data1/jiaxuanluo/extracted_glossary_v2_with_examples.json"
    
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return

    # 使用 defaultdict(lambda: {"term": "", "candidates": []}) 来管理多义项
    glossary = {}

    print(f"Processing {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            
            messages = obj.get("messages", [])
            gt_terms_by_chunk = obj.get("gt_terms_by_chunk", [])
            
            # 遍历每个 chunk
            for i, chunk in enumerate(gt_terms_by_chunk):
                if not chunk:
                    continue
                
                # 获取该 chunk 对应的翻译例句 (Assistant 的回复)
                # 索引逻辑: 0 是 system, 1 是 user(audio 0), 2 是 assistant(trans 0)
                # 所以 chunk i 对应 messages[2*i + 2]
                example_idx = 2 * i + 2
                if example_idx >= len(messages):
                    continue
                    
                example_text = messages[example_idx].get("content", "").strip()
                if not example_text:
                    continue

                for entry in chunk:
                    term = entry.get("term", "").strip()
                    zh = entry.get("zh", "").strip()
                    
                    if not term:
                        continue
                    
                    key = term.lower()
                    
                    if key not in glossary:
                        glossary[key] = {
                            "term": term,
                            "candidates": []
                        }
                    
                    # 检查是否已经存在相同的 (zh, example) 对，避免完全重复
                    is_duplicate = False
                    for cand in glossary[key]["candidates"]:
                        if cand["target_translations"]["zh"] == zh and cand["example"] == example_text:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        glossary[key]["candidates"].append({
                            "target_translations": {
                                "zh": zh
                            },
                            "example": example_text,
                            # 预留其他字段
                            "classification_reason": "",
                            "short_description": "",
                            "full_form": "",
                            "is_acronym": False
                        })

    print(f"Extraction complete. Total unique terms: {len(glossary)}")
    
    # 写入 JSON
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(glossary, f_out, ensure_ascii=False, indent=2)
    
    print(f"Saved example-based glossary to {output_path}")

if __name__ == "__main__":
    extract_terms()
