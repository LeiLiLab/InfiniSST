#!/usr/bin/env python3
"""
测试集成到extract_ground_truth_terms中的过滤功能
"""

from train_samples_pre_handle import extract_ground_truth_terms, build_phrase_desc_index
from glossary_utils import load_clean_glossary_from_file

# 加载glossary数据
term_set, alt2main, glossary = load_clean_glossary_from_file(
    'data/terms/term_set.txt', 
    'data/terms/alt2main.json', 
    'data/terms/glossary_filtered.json'
)
phrase2desc = build_phrase_desc_index(term_set, alt2main, glossary, 'term')

# 模拟包含常见词的文本和命名实体
test_text = "today we met with chris watson and discussed the sixteen projects at eleven o'clock in the morning"

# 模拟命名实体（包含应该被过滤的词）
test_named_entities = [
    "today",           # 应该被过滤：时间词汇
    "chris watson",    # 应该保留：人名
    "sixteen",         # 应该被过滤：数字词汇  
    "eleven",          # 应该被过滤：数字词汇
    "morning"          # 应该被过滤：时间词汇
]

print("Testing integrated filtering in extract_ground_truth_terms...")
print(f"Text: {test_text}")
print(f"Named entities: {test_named_entities}")

# 提取ground truth terms
result = extract_ground_truth_terms(test_text, phrase2desc, test_named_entities)

print(f"\nResult: {result}")

if result:
    print(f"Number of terms: {len(result)}")
    
    # 检查是否还有应该被过滤的词
    filtered_out = []
    kept = []
    
    for entity in test_named_entities:
        if entity.lower() in ['today', 'sixteen', 'eleven', 'morning']:
            filtered_out.append(entity)
        else:
            kept.append(entity)
    
    print(f"Expected to be filtered out: {filtered_out}")
    print(f"Expected to be kept: {kept}")
    
    # 验证结果
    success = True
    for term in result:
        if any(filtered_word in term.lower() for filtered_word in ['today', 'sixteen', 'eleven', 'morning']):
            print(f"❌ Term '{term}' should have been filtered out!")
            success = False
    
    if success:
        print("✅ Filtering working correctly!")
    else:
        print("❌ Some terms were not properly filtered!")
else:
    print("No terms extracted (this might be expected if all were filtered)") 