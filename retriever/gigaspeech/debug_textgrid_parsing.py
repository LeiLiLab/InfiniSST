#!/usr/bin/env python3
"""
调试TextGrid解析问题
"""

from handle_MFA_n_chunk_samples import parse_textgrid

textgrid_path = "/mnt/data/siqiouyang/datasets/gigaspeech/textgrids/POD0000000001_S0000011.TextGrid"

print(f"Testing TextGrid parsing for: {textgrid_path}")

# 测试解析
words = parse_textgrid(textgrid_path)

print(f"Parsed {len(words)} words")
if words:
    print("First 10 words:")
    for i, word in enumerate(words[:10]):
        print(f"  {i+1}: '{word['word']}' ({word['start']:.2f}-{word['end']:.2f})")
else:
    print("No words parsed. Let's check the file content...")
    
    # 手动检查文件内容
    with open(textgrid_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("File content preview:")
    lines = content.split('\n')
    for i, line in enumerate(lines[:30]):
        print(f"  {i+1:2d}: {repr(line)}")
    
    # 查找words层
    words_start = content.find('"words"')
    phones_start = content.find('"phones"')
    
    print(f"\nwords tier starts at position: {words_start}")
    print(f"phones tier starts at position: {phones_start}")
    
    if words_start != -1:
        if phones_start != -1:
            words_content = content[words_start:phones_start]
        else:
            words_content = content[words_start:]
        
        print(f"\nWords tier content (first 500 chars):")
        print(repr(words_content[:500])) 