#!/usr/bin/env python3
"""
测试当前代码的chunk文本提取功能
"""

from handle_MFA_n_chunk_samples import process_sample

# 使用您提供的有问题的样本
test_sample = {
    "segment_id": "POD0000000712_S0000010",
    "text": "he had with him in two thousand and sixteen",  # 简化的文本，不包含音素
    "audio": "/mnt/data/jiaxuanluo/audio/POD/POD0000000712/POD0000000712_S0000010.wav",
    "begin_time": 80.7,  # 假设的开始时间
    "end_time": 89.7,    # 根据您的数据
    "audio_id": "POD0000000712",
    "ground_truth_term": ["Sixteen"],
    "has_target": True
}

print("Testing current code chunk text extraction...")
print(f"Original text: '{test_sample['text']}'")
print(f"Duration: {test_sample['end_time'] - test_sample['begin_time']:.2f}s")

# 测试处理
result = process_sample(test_sample, n=3, chunk_len=0.96)

if result:
    print("\n✅ Processing successful!")
    print(f"Chunk text: '{result['n_chunk_text']}'")
    print(f"Chunk duration: {result['chunk_end_time'] - result['chunk_start_time']:.2f}s")
    print(f"Ground truth terms: {result['n_chunk_audio_ground_truth_terms']}")
    
    # 检查是否包含音素
    chunk_text = result['n_chunk_text']
    phonetic_symbols = ['ɐ', 'ɡ', 'ɔ', 'ʎ', 'ɑ', 'ə', 'ɛ', 'ɹ', 'ɲ', 'uː', 'ʉː', 'tʃ', 'ʋ', 'ɜː', 'ɖ', 'ç', 'ɒ', 'ʈ', 'ɪ', 'cʷ', 'ɔj', 'vʲ', 'aː', 'aw', 'aj', 'spn', 'd̪', 't̪', 'tʲ']
    
    found_phonetic = [symbol for symbol in phonetic_symbols if symbol in chunk_text]
    if found_phonetic:
        print(f"❌ Still contains phonetic symbols: {found_phonetic}")
        print(f"Full chunk text: {repr(chunk_text)}")
    else:
        print("✅ No phonetic symbols found!")
else:
    print("\n❌ Processing failed!") 