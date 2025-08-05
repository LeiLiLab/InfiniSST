#!/usr/bin/env python3
"""
测试term-level chunk处理的修复效果
只处理前10个样本来快速验证时间计算是否正确
"""

import json
import sys
import os
sys.path.append('.')

from handle_MFA_term_level_chunks import process_sample

def test_term_level_processing():
    # 加载前10个样本进行测试
    input_file = "data/samples/xl/term_preprocessed_samples_0_500000.json"
    textgrid_dir = "/mnt/data/siqiouyang/datasets/gigaspeech/textgrids"
    
    print(f"[INFO] Loading test samples from {input_file}")
    with open(input_file, 'r') as f:
        all_samples = json.load(f)
    
    # 只取前10个样本测试
    test_samples = all_samples[:10]
    print(f"[INFO] Testing with {len(test_samples)} samples")
    
    success_count = 0
    total_chunks = 0
    
    for i, sample in enumerate(test_samples):
        segment_id = sample['segment_id']
        print(f"\n[TEST {i+1}/10] Processing {segment_id}")
        print(f"  - begin_time: {sample.get('begin_time', 0)}")
        print(f"  - end_time: {sample.get('end_time', 0)}")
        print(f"  - duration: {sample.get('end_time', 0) - sample.get('begin_time', 0):.2f}s")
        print(f"  - ground_truth_terms: {sample.get('ground_truth_term', [])}")
        
        try:
            term_chunks = process_sample(sample, textgrid_dir)
            chunk_count = len(term_chunks)
            total_chunks += chunk_count
            
            if chunk_count > 0:
                success_count += 1
                print(f"  ✅ Generated {chunk_count} term chunks")
                for chunk in term_chunks:
                    print(f"    - {chunk['term_chunk_text']}: {chunk['term_start_time']:.2f}-{chunk['term_end_time']:.2f}s")
            else:
                print(f"  ⚠️  No term chunks generated")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n[SUMMARY]")
    print(f"- Test samples: {len(test_samples)}")
    print(f"- Successful samples: {success_count}")
    print(f"- Total term chunks: {total_chunks}")
    print(f"- Average chunks per sample: {total_chunks/len(test_samples):.2f}")
    print(f"- Success rate: {success_count/len(test_samples)*100:.1f}%")
    
    if total_chunks > 0:
        print(f"\n✅ Processing looks good! Expected ~{total_chunks * (88784/10):.0f} chunks from full dataset")
    else:
        print(f"\n❌ No chunks generated - there may still be issues")

if __name__ == "__main__":
    test_term_level_processing() 