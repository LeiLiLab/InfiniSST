#!/usr/bin/env python3
"""
调试脚本：直接运行MFA chunk处理的前几个样本，用于快速调试术语匹配问题
"""

import json
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from handle_MFA_n_chunk_samples import process_sample

def debug_mfa_chunks():
    # 输入文件路径
    input_json = "data/samples/xl/term_preprocessed_samples_0_500000.json"
    
    if not os.path.exists(input_json):
        print(f"[ERROR] Input file not found: {input_json}")
        return
    
    # 读取样本
    print(f"[INFO] Loading samples from {input_json}")
    with open(input_json, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    print(f"[INFO] Total samples: {len(samples)}")
    
    # 只处理前10个样本进行调试
    debug_samples = samples[:10]
    
    print(f"[INFO] Processing first {len(debug_samples)} samples for debugging...")
    
    processed_count = 0
    failed_count = 0
    
    for i, sample in enumerate(debug_samples):
        print(f"\n[DEBUG] ===== Processing sample {i+1}/{len(debug_samples)} =====")
        print(f"[DEBUG] Segment ID: {sample.get('segment_id', 'unknown')}")
        print(f"[DEBUG] Ground truth terms count: {len(sample.get('ground_truth_term', []))}")
        print(f"[DEBUG] Begin time: {sample.get('begin_time', 'N/A')}")
        print(f"[DEBUG] End time: {sample.get('end_time', 'N/A')}")
        print(f"[DEBUG] Audio path: {sample.get('audio', 'N/A')}")
        print(f"[DEBUG] Sample keys: {list(sample.keys())}")
        
        try:
            result = process_sample(sample, n=3, chunk_len=0.96)
            if result:
                processed_count += 1
                print(f"[SUCCESS] Sample processed successfully")
                print(f"[SUCCESS] Output fields: {list(result.keys())}")
                print(f"[SUCCESS] Chunk text length: {len(result.get('n_chunk_text', ''))}")
                print(f"[SUCCESS] Chunk text: '{result.get('n_chunk_text', '')[:100]}...'")
                print(f"[SUCCESS] Covered terms: {len(result.get('n_chunk_audio_ground_truth_terms', []))}")
                print(f"[SUCCESS] Chunk times: {result.get('chunk_start_time_abs', 0):.2f} - {result.get('chunk_end_time_abs', 0):.2f}")
            else:
                failed_count += 1
                print(f"[FAILED] Sample processing failed")
        except Exception as e:
            failed_count += 1
            print(f"[ERROR] Exception during processing: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n[SUMMARY] Debug results:")
    print(f"  Processed: {processed_count}/{len(debug_samples)}")
    print(f"  Failed: {failed_count}/{len(debug_samples)}")
    print(f"  Success rate: {processed_count/len(debug_samples)*100:.1f}%")

if __name__ == "__main__":
    debug_mfa_chunks() 