#!/usr/bin/env python3
"""
测试术语过滤功能
"""

from train_samples_pre_handle import should_filter_term

# 测试用例
test_cases = [
    # 应该被过滤的术语
    ("today", True, "时间词汇"),
    ("eleven", True, "数字词汇"),
    ("sixteen", True, "数字词汇"),
    ("the", True, "停用词"),
    ("and", True, "停用词"),
    ("123", True, "纯数字"),
    ("1,234", True, "带逗号数字"),
    ("a", True, "单字符"),
    ("monday", True, "星期"),
    ("january", True, "月份"),
    ("first", True, "序数词"),
    ("something", True, "常见代词"),
    
    # 不应该被过滤的术语
    ("Chris Watson", False, "人名"),
    ("New York", False, "地名"),
    ("Apple Inc", False, "公司名"),
    ("iPhone", False, "产品名"),
    ("Python", False, "技术术语"),
    ("University", False, "机构类型"),
    ("McDonald's", False, "品牌名"),
    ("Christmas", False, "节日名"),
    ("Einstein", False, "人名"),
    ("Microsoft", False, "公司名"),
]

print("Testing term filtering function...")
print("=" * 60)

passed = 0
failed = 0

for term, should_be_filtered, category in test_cases:
    result = should_filter_term(term)
    status = "✅ PASS" if result == should_be_filtered else "❌ FAIL"
    
    print(f"{status} | '{term}' | Expected: {should_be_filtered}, Got: {result} | {category}")
    
    if result == should_be_filtered:
        passed += 1
    else:
        failed += 1

print("=" * 60)
print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")

if failed == 0:
    print("🎉 All tests passed!")
else:
    print(f"⚠️  {failed} tests failed. Please review the filtering rules.") 