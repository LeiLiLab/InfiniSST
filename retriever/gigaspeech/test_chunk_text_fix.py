#!/usr/bin/env python3
"""
测试修复后的chunk文本提取功能
"""

import json
from handle_MFA_n_chunk_samples import process_sample

# 创建一个测试样本
test_sample = {
    "segment_id": "POD0000000001_S0000011",
    "text": "doug mcgray is a journalist my name is douglas mcgray i'm a fellow at the new american foundation and i am the editor in chief of pop up magazine and a couple of years ago he wrote a story about check cashers and payday lenders which mcgray is quick to point out these are different services check cashing is when you have a check it's made out to you you go in with a check yeah",
    "audio": "/mnt/data/jiaxuanluo/audio/POD/POD0000000001/POD0000000001_S0000011.wav",
    "begin_time": 0.0,
    "end_time": 19.98,
    "audio_id": "POD0000000001",
    "ground_truth_term": ["Douglas McGray", "New American Foundation"],
    "has_target": True
}

print("Testing chunk text extraction...")
print(f"Original text: {test_sample['text']}")
print(f"Text length: {len(test_sample['text'])}")
print(f"Duration: {test_sample['end_time'] - test_sample['begin_time']:.2f}s")

# 测试处理
result = process_sample(test_sample, n=3, chunk_len=0.96)

if result:
    print("\n✅ Processing successful!")
    print(f"Chunk text: {result['n_chunk_text']}")
    print(f"Chunk duration: {result['chunk_end_time'] - result['chunk_start_time']:.2f}s")
    print(f"Chunk audio: {result['n_chunk_audio']}")
    print(f"Ground truth terms: {result['n_chunk_audio_ground_truth_terms']}")
else:
    print("\n❌ Processing failed!") 