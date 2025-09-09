#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(__file__))

from Qwen2_Audio_term_level_train import TermLevelDataset
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_samples_path', type=str, default="data/samples/xl/term_level_chunks_0_500000.json")
    parser.add_argument('--enable_no_term', action='store_true', default=True)
    parser.add_argument('--disable_no_term', action='store_true')
    args = parser.parse_args()
    
    if args.disable_no_term:
        args.enable_no_term = False
    
    print(f"[DEBUG] Loading dataset from: {args.train_samples_path}")
    print(f"[DEBUG] enable_no_term: {args.enable_no_term}")
    
    try:
        # 加载训练数据集
        train_dataset = TermLevelDataset(
            args.train_samples_path, 
            split="train", 
            train_ratio=0.99, 
            enable_no_term=args.enable_no_term
        )
        
        print(f"[INFO] Training dataset loaded successfully!")
        print(f"[INFO] Dataset size: {len(train_dataset)}")
        
        if len(train_dataset) == 0:
            print(f"[ERROR] Dataset is empty! This will cause the DataLoader error.")
            return
        
        # 检查前几个样本
        print(f"[INFO] Checking first few samples...")
        for i in range(min(3, len(train_dataset))):
            try:
                sample = train_dataset[i]
                ground_truth_terms, audio_path, chunk_text, has_target = sample
                print(f"  Sample {i}:")
                print(f"    - Audio: {audio_path}")
                print(f"    - Text: {chunk_text[:50]}...")
                print(f"    - Terms: {ground_truth_terms}")
                print(f"    - Has target: {has_target}")
            except Exception as e:
                print(f"  Sample {i}: Error - {e}")
        
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
