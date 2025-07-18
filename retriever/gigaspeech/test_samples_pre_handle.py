#!/usr/bin/env python3
"""
测试脚本：验证修改后的 train_samples_pre_handle.py
"""

import os
import tempfile
import json

# 创建测试TSV文件
test_tsv_content = """id	audio	n_frames	speaker	src_text	src_lang
POD0000000001_S0000001	/path/to/audio1.opus	100000	N/A	HELLO WORLD THIS IS A TEST	en
POD0000000002_S0000002	/path/to/audio2.opus	120000	N/A	ANOTHER TEST SAMPLE HERE	en
POD0000000003_S0000003	/path/to/audio3.opus	80000	N/A	THIRD SAMPLE FOR TESTING	en"""

# 创建测试命名实体JSON文件
test_ner_content = [
    ["hello", "world", "test"],
    ["another", "test", "sample"],
    ["third", "sample", "testing"]
]

def test_read_tsv():
    """测试TSV读取功能"""
    print("测试TSV读取功能...")
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
        f.write(test_tsv_content)
        tsv_path = f.name
    
    try:
        # 导入并测试函数
        from train_samples_pre_handle import read_tsv_samples
        
        samples = read_tsv_samples(tsv_path)
        
        print(f"读取到 {len(samples)} 个样本")
        print("样本示例:")
        for i, sample in enumerate(samples[:2]):
            print(f"  样本 {i+1}: {sample}")
        
        # 验证预期结果
        assert len(samples) == 3, f"期望3个样本，实际得到{len(samples)}"
        assert samples[0]["text"] == "hello world this is a test", f"文本小写转换失败: {samples[0]['text']}"
        assert "segment_id" in samples[0], "缺少segment_id字段"
        assert "audio" in samples[0], "缺少audio字段"
        
        print("✅ TSV读取测试通过")
        
    finally:
        # 清理临时文件
        os.unlink(tsv_path)

def test_full_processing():
    """测试完整处理流程"""
    print("\n测试完整处理流程...")
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
        f.write(test_tsv_content)
        tsv_path = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_ner_content, f)
        ner_path = f.name
    
    try:
        # 检查所需文件是否存在
        term_files = [
            "data/terms/term_set.txt",
            "data/terms/alt2main.json", 
            "data/terms/glossary_filtered.json"
        ]
        
        missing_files = [f for f in term_files if not os.path.exists(f)]
        if missing_files:
            print(f"⚠️  缺少必需文件，跳过完整测试: {missing_files}")
            return
        
        from train_samples_pre_handle import handle_split_samples
        
        samples = handle_split_samples(
            term_set_path="data/terms/term_set.txt",
            alt2main_path="data/terms/alt2main.json",
            glossary_path="data/terms/glossary_filtered.json",
            tsv_path=tsv_path,
            ner_json_path=ner_path,
            split_id=0,
            text_field="term"
        )
        
        print(f"处理得到 {len(samples)} 个样本")
        if samples:
            print("处理后样本示例:")
            for i, sample in enumerate(samples[:2]):
                print(f"  样本 {i+1}: {sample}")
        
        print("✅ 完整处理测试通过")
        
    except Exception as e:
        print(f"❌ 完整处理测试失败: {e}")
        
    finally:
        # 清理临时文件
        os.unlink(tsv_path)
        os.unlink(ner_path)

if __name__ == "__main__":
    print("开始测试修改后的样本预处理功能...\n")
    
    test_read_tsv()
    test_full_processing()
    
    print("\n测试完成！") 