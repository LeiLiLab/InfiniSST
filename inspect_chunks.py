#!/usr/bin/env python3
"""
Inspection script for saved ACL chunks
"""

import json
import os
import sys
from pathlib import Path

def inspect_chunks(chunk_dir="data/acl_chunks"):
    """检查保存的chunk数据"""
    chunk_dir = Path(chunk_dir)
    
    if not chunk_dir.exists():
        print(f"❌ Chunk directory not found: {chunk_dir}")
        print("💡 Run ACL evaluation with --save_chunks to generate chunk data")
        return False
    
    print(f"🔍 Inspecting chunks in: {chunk_dir}")
    
    # 查找元数据文件
    metadata_dir = chunk_dir / "metadata"
    audio_dir = chunk_dir / "audio"
    
    if not metadata_dir.exists() or not audio_dir.exists():
        print(f"❌ Missing metadata or audio directories")
        return False
    
    # 查找所有元数据文件
    metadata_files = list(metadata_dir.glob("chunks_*.json"))
    sample_files = list(metadata_dir.glob("sample_chunks_*.json"))
    report_files = list(chunk_dir.glob("inspection_report.txt"))
    
    print(f"📂 Found files:")
    print(f"  - Metadata files: {len(metadata_files)}")
    print(f"  - Sample files: {len(sample_files)}")
    print(f"  - Audio files: {len(list(audio_dir.glob('*.wav')))}")
    print(f"  - Reports: {len(report_files)}")
    
    # 检查最新的sample文件
    if sample_files:
        latest_sample = max(sample_files, key=os.path.getctime)
        print(f"\n📊 Inspecting latest sample file: {latest_sample.name}")
        
        with open(latest_sample, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        
        print(f"🔬 Sample chunks ({len(samples)} total):")
        
        for i, chunk in enumerate(samples[:5]):  # 显示前5个
            print(f"\n  Chunk {i+1}: {chunk['chunk_id']}")
            print(f"    📁 Audio: {chunk['audio_file']}")
            print(f"    ⏱️  Duration: {chunk['chunk_duration']:.2f}s", end="")
            if chunk['is_short_chunk']:
                print(" (SHORT)", end="")
            print()
            print(f"    📝 Sentence: {chunk['original_sent_id']}")
            print(f"    🧩 Position: {chunk['chunk_index']}/{chunk['total_chunks']-1}")
            print(f"    📚 Original terms: {chunk['original_terms']}")
            print(f"    ✅ Filtered terms: {chunk['terms']}")
            
            if chunk['original_terms'] != chunk['terms']:
                removed = set(chunk['original_terms']) - set(chunk['terms'])
                print(f"    ❌ Removed: {list(removed)}")
            
            # 检查音频文件是否存在
            audio_path = audio_dir / chunk['audio_file']
            if audio_path.exists():
                file_size = audio_path.stat().st_size
                print(f"    🎵 Audio file: ✅ ({file_size} bytes)")
            else:
                print(f"    🎵 Audio file: ❌ Missing")
    
    # 显示报告
    if report_files:
        latest_report = max(report_files, key=os.path.getctime)
        print(f"\n📋 Inspection Report: {latest_report}")
        print("=" * 50)
        
        with open(latest_report, 'r', encoding='utf-8') as f:
            content = f.read()
            # 只显示前20行
            lines = content.split('\n')[:20]
            for line in lines:
                print(line)
        
        if len(content.split('\n')) > 20:
            print("... (see full report in file)")
    
    return True

def get_chunk_statistics(chunk_dir="data/acl_chunks"):
    """获取chunk统计信息"""
    chunk_dir = Path(chunk_dir)
    metadata_dir = chunk_dir / "metadata"
    
    metadata_files = list(metadata_dir.glob("chunks_*.json"))
    if not metadata_files:
        return None
    
    latest_metadata = max(metadata_files, key=os.path.getctime)
    
    with open(latest_metadata, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    stats = {
        'total_chunks': len(chunks),
        'filtering_methods': set(),
        'splits': set(),
        'segmentations': set(),
        'term_stats': {
            'total_original': 0,
            'total_filtered': 0,
            'unique_original': set(),
            'unique_filtered': set()
        },
        'short_chunks': 0,
        'duration_stats': []
    }
    
    for chunk in chunks:
        stats['filtering_methods'].add(chunk['term_filtering_method'])
        stats['splits'].add(chunk['chunk_id'].split('_')[1])  # Extract split from ID
        stats['segmentations'].add('gold')  # Assume gold for now
        
        stats['term_stats']['total_original'] += len(chunk['original_terms'])
        stats['term_stats']['total_filtered'] += len(chunk['terms'])
        stats['term_stats']['unique_original'].update(chunk['original_terms'])
        stats['term_stats']['unique_filtered'].update(chunk['terms'])
        
        if chunk['is_short_chunk']:
            stats['short_chunks'] += 1
        
        stats['duration_stats'].append(chunk['chunk_duration'])
    
    return stats

if __name__ == "__main__":
    chunk_dir = sys.argv[1] if len(sys.argv) > 1 else "data/acl_chunks"
    
    print("🚀 ACL Chunk Inspector\n")
    
    success = inspect_chunks(chunk_dir)
    
    if success:
        print(f"\n📈 Getting detailed statistics...")
        stats = get_chunk_statistics(chunk_dir)
        
        if stats:
            print(f"\n📊 Detailed Statistics:")
            print(f"  - Total chunks: {stats['total_chunks']}")
            print(f"  - Filtering methods: {list(stats['filtering_methods'])}")
            print(f"  - Term retention: {stats['term_stats']['total_filtered']}/{stats['term_stats']['total_original']} ({stats['term_stats']['total_filtered']/stats['term_stats']['total_original']*100:.1f}%)")
            print(f"  - Unique terms: {len(stats['term_stats']['unique_filtered'])}/{len(stats['term_stats']['unique_original'])}")
            print(f"  - Short chunks: {stats['short_chunks']} ({stats['short_chunks']/stats['total_chunks']*100:.1f}%)")
            
            if stats['duration_stats']:
                avg_duration = sum(stats['duration_stats']) / len(stats['duration_stats'])
                print(f"  - Average duration: {avg_duration:.2f}s")
        
        print(f"\n✅ Inspection completed!")
        print(f"💡 Tips:")
        print(f"  - Listen to audio files in {chunk_dir}/audio/")
        print(f"  - Check detailed report in {chunk_dir}/inspection_report.txt")
        print(f"  - Verify terms are actually present in the audio")
    else:
        print(f"\n❌ Inspection failed!")
        sys.exit(1)

