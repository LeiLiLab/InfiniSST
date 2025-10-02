#!/usr/bin/env python3
"""
智能上传 GigaSpeech 到 Modal Volume
- 跳过压缩文件 (.tgz, .tgz.aes)
- 跳过 textgrids 目录
- 跳过已存在的文件
- 支持断点续传
"""

import subprocess
import os
import sys
from pathlib import Path
from tqdm import tqdm

# 配置
SOURCE_DIR = "/mnt/data/siqiouyang/datasets/gigaspeech"
VOLUME_NAME = "infinisst-data"
REMOTE_PATH = "gigaspeech"

# 需要跳过的模式
SKIP_PATTERNS = [
    "*.tgz.aes",      # 加密压缩文件跳过
    "**/textgrids/**", # textgrids 目录跳过
    "**/.DS_Store",
    "**/__pycache__/**",
    "**/*.opus",       # 跳过所有音频文件（在压缩包里）
]

# 需要包含的文件类型（上传压缩文件和元数据）
INCLUDE_PATTERNS = [
    "*.tsv",       # 元数据文件
    "*.tar.gz",    # 压缩音频文件
    "*.tgz",       # 压缩音频文件
    "*.json",      # 配置文件
]


def should_skip(file_path: Path, base_path: Path) -> bool:
    """判断文件是否应该跳过"""
    rel_path = file_path.relative_to(base_path)
    rel_path_str = str(rel_path)
    
    # 检查是否匹配跳过模式
    for pattern in SKIP_PATTERNS:
        if file_path.match(pattern):
            return True
        # 检查 textgrids 目录
        if "textgrids" in rel_path.parts:
            return True
    
    return False


def should_include(file_path: Path) -> bool:
    """判断文件是否应该包含"""
    # 如果没有指定包含模式，包含所有文件
    if not INCLUDE_PATTERNS:
        return True
    
    # 检查是否匹配包含模式
    for pattern in INCLUDE_PATTERNS:
        if file_path.match(pattern):
            return True
    
    return False


def get_uploaded_files(volume_name: str, remote_path: str) -> set:
    """获取已上传的文件列表"""
    print(f"[INFO] Checking existing files in volume '{volume_name}/{remote_path}'...")
    try:
        result = subprocess.run(
            ["modal", "volume", "ls", volume_name, remote_path],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            print(f"[WARN] Could not list volume contents (volume might be empty): {result.stderr}")
            return set()
        
        # 解析输出获取文件列表
        uploaded = set()
        for line in result.stdout.strip().split('\n'):
            if line and not line.startswith('Dir '):
                # 提取文件路径
                parts = line.split()
                if parts:
                    uploaded.add(parts[-1])
        
        print(f"[INFO] Found {len(uploaded)} existing files in volume")
        return uploaded
    
    except Exception as e:
        print(f"[WARN] Error checking uploaded files: {e}")
        return set()


def collect_files_to_upload(source_dir: Path, uploaded_files: set) -> list:
    """收集需要上传的文件"""
    print(f"[INFO] Scanning directory: {source_dir}")
    
    files_to_upload = []
    skipped_count = 0
    already_uploaded_count = 0
    
    for root, dirs, files in os.walk(source_dir):
        root_path = Path(root)
        
        # 跳过 textgrids 目录
        if 'textgrids' in dirs:
            dirs.remove('textgrids')
            print(f"[SKIP] Skipping textgrids directory")
        
        for filename in files:
            file_path = root_path / filename
            
            # 检查是否应该跳过
            if should_skip(file_path, source_dir):
                skipped_count += 1
                continue
            
            # 检查是否应该包含
            if not should_include(file_path):
                skipped_count += 1
                continue
            
            # 计算相对路径
            rel_path = file_path.relative_to(source_dir)
            
            # 检查是否已上传
            if str(rel_path) in uploaded_files:
                already_uploaded_count += 1
                continue
            
            files_to_upload.append((file_path, rel_path))
    
    print(f"\n[SUMMARY]")
    print(f"  - Files to upload: {len(files_to_upload)}")
    print(f"  - Already uploaded: {already_uploaded_count}")
    print(f"  - Skipped (compressed/textgrids): {skipped_count}")
    
    return files_to_upload


def upload_files(files: list, volume_name: str, remote_base: str, dry_run: bool = False):
    """上传文件到 Modal Volume"""
    if not files:
        print("[INFO] No files to upload!")
        return
    
    if dry_run:
        print("\n[DRY RUN] Would upload the following files:")
        for local_path, rel_path in files[:20]:  # 只显示前20个
            print(f"  {rel_path} ({local_path.stat().st_size / (1024**2):.2f} MB)")
        if len(files) > 20:
            print(f"  ... and {len(files) - 20} more files")
        return
    
    print(f"\n[INFO] Starting upload of {len(files)} files...")
    
    failed_uploads = []
    
    for i, (local_path, rel_path) in enumerate(tqdm(files, desc="Uploading"), 1):
        remote_path = f"{remote_base}/{rel_path}"
        
        try:
            # 使用 modal volume put 上传单个文件
            result = subprocess.run(
                ["modal", "volume", "put", volume_name, str(local_path), remote_path],
                capture_output=True,
                text=True,
                check=True
            )
            
            if i % 100 == 0:
                print(f"\n[PROGRESS] Uploaded {i}/{len(files)} files")
        
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Failed to upload {rel_path}: {e.stderr}")
            failed_uploads.append((local_path, rel_path))
            
            # 如果失败太多，询问是否继续
            if len(failed_uploads) > 10:
                response = input("\n[WARN] Too many failures. Continue? (y/n): ")
                if response.lower() != 'y':
                    break
    
    print(f"\n[COMPLETE]")
    print(f"  - Successfully uploaded: {len(files) - len(failed_uploads)}")
    print(f"  - Failed: {len(failed_uploads)}")
    
    if failed_uploads:
        print("\n[FAILED FILES]")
        for local_path, rel_path in failed_uploads:
            print(f"  {rel_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload GigaSpeech to Modal Volume")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded without actually uploading")
    parser.add_argument("--force", action="store_true", help="Re-upload all files, ignoring existing ones")
    args = parser.parse_args()
    
    source_path = Path(SOURCE_DIR)
    
    if not source_path.exists():
        print(f"[ERROR] Source directory not found: {SOURCE_DIR}")
        sys.exit(1)
    
    print(f"=" * 80)
    print(f"GigaSpeech Upload to Modal Volume")
    print(f"=" * 80)
    print(f"Source: {SOURCE_DIR}")
    print(f"Volume: {VOLUME_NAME}")
    print(f"Remote path: {REMOTE_PATH}")
    print(f"Dry run: {args.dry_run}")
    print(f"=" * 80)
    
    # 获取已上传的文件（除非 force 模式）
    uploaded_files = set() if args.force else get_uploaded_files(VOLUME_NAME, REMOTE_PATH)
    
    # 收集需要上传的文件
    files_to_upload = collect_files_to_upload(source_path, uploaded_files)
    
    if not files_to_upload:
        print("\n[INFO] All files are already uploaded!")
        return
    
    # 计算总大小
    total_size = sum(f[0].stat().st_size for f in files_to_upload)
    print(f"\n[INFO] Total size to upload: {total_size / (1024**3):.2f} GB")
    
    if not args.dry_run:
        response = input("\n[CONFIRM] Proceed with upload? (y/n): ")
        if response.lower() != 'y':
            print("[INFO] Upload cancelled")
            return
    
    # 上传文件
    upload_files(files_to_upload, VOLUME_NAME, REMOTE_PATH, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

