#!/usr/bin/env python3
"""
简单的checkpoint评估脚本
用于快速评估已训练的checkpoint
"""

import subprocess
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 evaluate_checkpoint.py <checkpoint_path> [options]")
        print("Example:")
        print("  python3 evaluate_checkpoint.py data/clap_term_level_epoch1.pt")
        print("  python3 evaluate_checkpoint.py data/clap_term_level_epoch1.pt --enable_full_eval")
        print("  python3 evaluate_checkpoint.py data/clap_term_level_epoch1.pt --gpu_ids=0")
        return
    
    checkpoint_path = sys.argv[1]
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return
    
    # 构建命令
    cmd = [
        "python3", "SONAR_term_level_train_glossary.py",
        "--direct_evaluate",
        f"--checkpoint_path={checkpoint_path}",
        "--train_samples_path=data/xl_term_level_chunks_merged.json",
        "--test_samples_path=data/samples/xl/term_level_chunks_500000_1000000.json",
        "--glossary_path=data/terms/glossary_filtered.json",
        "--filter_no_term"
    ]
    
    # 添加额外参数
    if len(sys.argv) > 2:
        for arg in sys.argv[2:]:
            cmd.append(arg)
    
    print(f"Evaluating checkpoint: {checkpoint_path}")
    print(f"Command: {' '.join(cmd)}")
    print("="*60)
    
    # 执行命令
    try:
        result = subprocess.run(cmd, check=True)
        print("="*60)
        print("Evaluation completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error during evaluation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())


