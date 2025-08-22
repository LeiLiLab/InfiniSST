#!/usr/bin/env python3

"""
测试ACL数据集加载功能
"""

import sys
import os
sys.path.append('/home/jiaxuanluo/InfiniSST/retriever/gigaspeech')

from SONAR_ACL_test import ACLDataset, load_acl_terminology, parse_acl_tagged_text

def test_acl_terminology():
    """测试ACL术语词汇表加载"""
    print("=== Testing ACL Terminology Loading ===")
    glossary_path = "data/acl-6060/2/intermediate_files/terminology_glossary.csv"
    
    if os.path.exists(glossary_path):
        terms = load_acl_terminology(glossary_path)
        print(f"Loaded {len(terms)} terms")
        print(f"First 10 terms: {terms[:10]}")
        print(f"Last 10 terms: {terms[-10:]}")
    else:
        print(f"Glossary file not found: {glossary_path}")

def test_acl_tagged_text():
    """测试ACL标注文本解析"""
    print("\n=== Testing ACL Tagged Text Parsing ===")
    tagged_path = "data/acl-6060/2/acl_6060/dev/text/tagged_terminology/ACL.6060.dev.tagged.en-xx.en.txt"
    
    if os.path.exists(tagged_path):
        terms = parse_acl_tagged_text(tagged_path)
        print(f"Extracted {len(terms)} unique terms from dev set")
        print(f"First 10 terms: {terms[:10]}")
        print(f"Last 10 terms: {terms[-10:]}")
    else:
        print(f"Tagged text file not found: {tagged_path}")

def test_acl_dataset():
    """测试ACL数据集加载"""
    print("\n=== Testing ACL Dataset Loading ===")
    acl_root = "data/acl-6060/2/acl_6060"
    
    if os.path.exists(acl_root):
        # 测试dev数据集
        print("Loading dev dataset...")
        dev_dataset = ACLDataset(acl_root, split="dev", segmentation="gold")
        print(f"Dev dataset size: {len(dev_dataset)}")
        
        if len(dev_dataset) > 0:
            sample = dev_dataset[0]
            print(f"First sample: {sample}")
        
        # 测试eval数据集
        print("\nLoading eval dataset...")
        eval_dataset = ACLDataset(acl_root, split="eval", segmentation="gold")
        print(f"Eval dataset size: {len(eval_dataset)}")
        
        if len(eval_dataset) > 0:
            sample = eval_dataset[0]
            print(f"First sample: {sample}")
    else:
        print(f"ACL root directory not found: {acl_root}")

def main():
    print("Testing ACL dataset functionality...\n")
    
    test_acl_terminology()
    test_acl_tagged_text()
    test_acl_dataset()
    
    print("\nACL dataset testing completed!")

if __name__ == "__main__":
    main() 