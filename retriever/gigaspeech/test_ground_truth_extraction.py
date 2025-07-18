#!/usr/bin/env python3
"""
测试ground_truth_terms提取是否正确
"""

import json
import sys
import os
import re

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_samples_pre_handle import extract_ground_truth_terms, build_phrase_desc_index
from glossary_utils import load_clean_glossary_from_file

def test_ground_truth_extraction():
    # 加载glossary数据
    term_set_path = "data/terms/term_set.txt"
    alt2main_path = "data/terms/alt2main.json"
    glossary_path = "data/terms/glossary_filtered.json"
    
    if not all(os.path.exists(p) for p in [term_set_path, alt2main_path, glossary_path]):
        print("[ERROR] Glossary files not found")
        return
    
    term_set, alt2main, glossary = load_clean_glossary_from_file(term_set_path, alt2main_path, glossary_path)
    phrase2desc = build_phrase_desc_index(term_set, alt2main, glossary, "term")
    
    # 测试用例
    test_cases = [
        {
            "text": "it would've been impossible to record it under wild conditions",
            "named_entities": ["chris", "watson", "richard", "j", "hinton"],
            "expected_in_text": False  # 这些命名实体不在文本中
        },
        {
            "text": "douglas mcgray is a journalist and editor",
            "named_entities": ["douglas", "mcgray", "journalist"],
            "expected_in_text": True  # 这些应该能在文本中找到
        }
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n[TEST {i+1}] Testing text: '{case['text'][:50]}...'")
        print(f"[TEST {i+1}] Named entities: {case['named_entities']}")
        
        # 提取ground truth terms
        ground_truth_terms = extract_ground_truth_terms(
            case["text"], 
            phrase2desc, 
            case["named_entities"]
        )
        
        print(f"[TEST {i+1}] Extracted ground truth terms: {ground_truth_terms}")
        
        if ground_truth_terms:
            # 检查提取的术语是否确实在原文本中
            text_lower = case["text"].lower()
            tokens = re.findall(r"\b[\w']+\b", text_lower)
            text_phrase = ' '.join(tokens)
            
            for term in ground_truth_terms:
                if term in text_phrase:
                    print(f"[TEST {i+1}] ✅ Term '{term}' found in text")
                else:
                    print(f"[TEST {i+1}] ❌ Term '{term}' NOT found in text")
        else:
            print(f"[TEST {i+1}] No ground truth terms extracted")

if __name__ == "__main__":
    test_ground_truth_extraction() 