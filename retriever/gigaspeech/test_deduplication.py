#!/usr/bin/env python3
"""
测试术语去重功能
"""

from handle_MFA_n_chunk_samples import get_covered_terms_in_chunks

# 模拟chunk数据
chunks = [(0.0, 0.96), (0.96, 1.92), (1.92, 2.88)]

# 模拟term_spans数据，包含重复的术语
term_spans = [
    {"term": "Japan", "start": 0.5, "end": 1.0},     # 与第1个chunk重叠
    {"term": "Japan", "start": 1.2, "end": 1.5},     # 与第2个chunk重叠（重复术语）
    {"term": "Tokyo", "start": 2.0, "end": 2.5},     # 与第3个chunk重叠
]

print("Testing term deduplication...")
print(f"Chunks: {chunks}")
print(f"Term spans: {term_spans}")

# 测试获取前3个chunk覆盖的术语
covered_terms = get_covered_terms_in_chunks(chunks, 0, 3, term_spans)

print(f"\nCovered terms: {covered_terms}")
print(f"Number of terms: {len(covered_terms)}")

# 检查是否还有重复
if len(covered_terms) != len(set(covered_terms)):
    print("❌ Still has duplicates!")
else:
    print("✅ No duplicates found!")

# 验证期望的术语都在结果中
expected_terms = {"Japan", "Tokyo"}
actual_terms = set(covered_terms)

if expected_terms == actual_terms:
    print("✅ All expected terms found!")
else:
    print(f"❌ Expected: {expected_terms}, Got: {actual_terms}") 