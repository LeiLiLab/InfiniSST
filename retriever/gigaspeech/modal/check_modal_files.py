#!/usr/bin/env python3
"""
检查Modal存储卷中的文件状态
"""

import os
import sys

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modal_qwen2_audio_training import check_existing_files, app

def main():
    """检查Modal中的文件状态"""
    
    # 定义要检查的文件
    files_to_check = [
        "xl_cleaned_term_level_chunks_merged.json",
        "glossary_merged.json", 
        "Qwen2_Audio_train.py",
        "train_ddp_simplified.py"
    ]
    
    print(f"[INFO] Checking {len(files_to_check)} files in Modal...")
    
    # 检查文件状态
    with app.run():
        existing_files = check_existing_files.remote(files_to_check)
        
        print("\n=== File Status Report ===")
        for file_path, exists in existing_files.items():
            status = "✅ EXISTS" if exists else "❌ NOT FOUND"
            print(f"{status}: {file_path}")
        
        existing_count = sum(1 for exists in existing_files.values() if exists)
        print(f"\nSummary: {existing_count}/{len(files_to_check)} files exist in Modal")
        
        if existing_count == len(files_to_check):
            print("🎉 All files are ready! You can skip upload and start training directly.")
        else:
            missing_files = [f for f, exists in existing_files.items() if not exists]
            print(f"📋 Missing files: {missing_files}")
            print("💡 Run the main training script to upload missing files.")

if __name__ == "__main__":
    main()
