#!/usr/bin/env python3
"""
测试实际样本的chunk文本提取
"""

import json
from handle_MFA_n_chunk_samples import process_sample

# 使用您提供的有问题的样本
test_sample = {
    "segment_id": "POD0000000686_S0000072",
    "text": "12 and it's called eunice the kennedy who changed the world",  # 简化版本，不包含音素
    "audio": "/mnt/data/jiaxuanluo/audio/POD/POD0000000686/POD0000000686_S0000072.wav",
    "begin_time": 623.76,
    "end_time": 626.64,
    "audio_id": "POD0000000686",
    "ground_truth_term": ["Eunice"],
    "has_target": True
}

print("Testing real sample chunk text extraction...")
print(f"Original text: {test_sample['text']}")
print(f"Duration: {test_sample['end_time'] - test_sample['begin_time']:.2f}s")

# 测试处理
result = process_sample(test_sample, n=3, chunk_len=0.96)

if result:
    print("\n✅ Processing successful!")
    print(f"Chunk text: '{result['n_chunk_text']}'")
    print(f"Chunk duration: {result['chunk_end_time'] - result['chunk_start_time']:.2f}s")
    print(f"Ground truth terms: {result['n_chunk_audio_ground_truth_terms']}")
    
    # 检查是否还包含音素
    chunk_text = result['n_chunk_text']
    phonetic_symbols = ['ɐ', 'ɡ', 'ɔ', 'ʎ', 'ɑ', 'ə', 'ɛ', 'ɹ', 'ɲ', 'uː', 'ʉː', 'tʃ', 'ʋ', 'ɜː', 'ɖ', 'ç', 'ɒ', 'ʈ', 'ɪ', 'cʷ', 'ɔj', 'vʲ', 'aː', 'aw', 'aj', 'spn']
    
    has_phonetic = any(symbol in chunk_text for symbol in phonetic_symbols)
    if has_phonetic:
        print("❌ Still contains phonetic symbols!")
    else:
        print("✅ No phonetic symbols found!")
else:
    print("\n❌ Processing failed!") 