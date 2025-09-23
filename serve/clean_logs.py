#!/usr/bin/env python3
"""
清理冗余的内存和beam-search日志
"""

import re
import os

def clean_file_logs(filepath):
    """清理单个文件中的冗余日志"""
    print(f"Cleaning logs in {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 需要注释掉的日志模式
    patterns_to_comment = [
        r'print\(f"🔍 \[MEMORY\].*?\)\n',
        r'print\(f"📈 \[MEMORY\].*?\)\n',
        r'print\(f"✅ \[MEMORY\].*?\)\n',
        r'print\(f"📎.*?\)\n',
        r'print\(f"🔍 \[ORCA.*?\)\n',
        r'print\(f"🔍 \[DECODE.*?\)\n',
        r'print\(f"🔍 \[SESSION-DEBUG\].*?\)\n',
        r'print\(f"🔍 \[BEAM-STATE\].*?\)\n',
        r'print\(f"🔍 \[BEAM-REQUEST\].*?\)\n',
        r'print\(f"🔍 \[BEAM-CACHE\].*?\)\n',
        r'print\(f"🔧 \[PREPARE-DATA\].*?\)\n',
        r'print\(f"🔍 \[DECODE-CACHE\].*?\)\n',
        r'print\(f"🔍 \[DECODE-RESULT\].*?\)\n',
        r'print\(f"🔍 \[DECODE-FINAL\].*?\)\n',
    ]
    
    original_content = content
    
    for pattern in patterns_to_comment:
        # 查找所有匹配的行
        matches = list(re.finditer(pattern, content, re.MULTILINE | re.DOTALL))
        
        # 从后往前替换，避免索引偏移
        for match in reversed(matches):
            start = match.start()
            end = match.end()
            matched_text = content[start:end]
            
            # 检查是否已经被注释
            if not matched_text.strip().startswith('#'):
                # 添加注释
                commented_text = '# ' + matched_text
                content = content[:start] + commented_text + content[end:]
    
    # 只有在内容有变化时才写入文件
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  - Updated {filepath}")
    else:
        print(f"  - No changes needed in {filepath}")

def main():
    """主函数"""
    files_to_clean = [
        'serve/inference_engine.py',
        'serve/scheduler.py',
        'model/flashinfer/engine.py',
        'model/flashinfer/wav2vec2.py'
    ]
    
    for filepath in files_to_clean:
        if os.path.exists(filepath):
            clean_file_logs(filepath)
        else:
            print(f"File not found: {filepath}")

if __name__ == "__main__":
    main() 