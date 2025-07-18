#!/usr/bin/env python3
"""
基于 MFA 对齐信息和已有语音片段数据，对每条 sample 提取覆盖 ground truth terms 最多的 n 个 chunk 音频片段
"""

import os
import json
import argparse
import re
import soundfile as sf
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm


def parse_textgrid(textgrid_path: str) -> List[Dict]:
    """
    解析TextGrid文件，提取words层的对齐信息
    返回格式: [{"word": "hello", "start": 1.0, "end": 1.5}, ...]
    """
    words = []
    
    if not os.path.exists(textgrid_path):
        print(f"[WARNING] TextGrid file not found: {textgrid_path}")
        return words
    
    try:
        with open(textgrid_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找words层的开始和结束位置
        words_tier_start = content.find('"words"')
        if words_tier_start == -1:
            print(f"[WARNING] No 'words' tier found in {textgrid_path}")
            return words
        
        # 查找下一个IntervalTier的开始位置（phones层）
        phones_tier_start = content.find('"phones"', words_tier_start)
        if phones_tier_start == -1:
            # 如果没有phones层，使用整个文件的剩余部分
            words_content = content[words_tier_start:]
        else:
            # 只处理words层的内容
            words_content = content[words_tier_start:phones_tier_start]
        
        lines = words_content.split('\n')
        
        # 找到intervals数量
        interval_count = 0
        start_parsing = False
        parse_start_idx = 0
        
        # 在words层中查找格式: "words" -> 0 -> 19.98 -> 82 (intervals数量)
        for i, line in enumerate(lines):
            line = line.strip()
            if line == '"words"' and i + 3 < len(lines):
                # 检查接下来的几行是否符合预期格式
                try:
                    # 跳过 "words" 后面的起始时间(0)和结束时间(19.98)
                    if (lines[i+1].strip().replace('.', '').isdigit() and 
                        lines[i+2].strip().replace('.', '').isdigit() and
                        lines[i+3].strip().isdigit()):
                        interval_count = int(lines[i+3].strip())
                        start_parsing = True
                        parse_start_idx = i + 4  # 从intervals开始的位置
                        break
                except (ValueError, IndexError):
                    continue
        
        if not start_parsing:
            print(f"[WARNING] Could not find interval count in words tier")
            return words
        
        # 解析每个interval（跳过前面的元信息）
        i = parse_start_idx
        parsed_intervals = 0
        
        while i < len(lines) and parsed_intervals < interval_count:
            line = lines[i].strip()
            
            # 跳过空行
            if not line:
                i += 1
                continue
                
            try:
                # 检查是否是时间数字
                if line.replace('.', '').replace('-', '').isdigit():
                    # 读取start time
                    start_time = float(line)
                    i += 1
                    
                    if i >= len(lines):
                        break
                        
                    # 读取end time  
                    end_time = float(lines[i].strip())
                    i += 1
                    
                    if i >= len(lines):
                        break
                    
                    # 读取word text
                    word_line = lines[i].strip()
                    word = word_line.strip('"')
                    i += 1
                    
                    # 只添加非空单词
                    if word and word != '':
                        words.append({
                            "word": word.lower(),
                            "start": start_time,
                            "end": end_time
                        })
                    
                    parsed_intervals += 1
                else:
                    i += 1
                    
            except (ValueError, IndexError) as e:
                i += 1
                continue
                
        print(f"[INFO] Parsed {len(words)} words from {textgrid_path} (expected {interval_count})")
        
    except Exception as e:
        print(f"[ERROR] Failed to parse TextGrid {textgrid_path}: {e}")
    
    return words


def get_textgrid_path(segment_id: str, textgrid_base_dir: str = "/mnt/data/siqiouyang/datasets/gigaspeech/textgrids") -> str:
    """根据segment_id获取对应的TextGrid文件路径"""
    textgrid_filename = f"{segment_id}.TextGrid"
    return os.path.join(textgrid_base_dir, textgrid_filename)


def find_term_time_spans(words: List[Dict], ground_truth_terms: List[str]) -> Tuple[List[Dict], List[str]]:
    """
    在words中找到ground_truth_terms对应的时间跨度
    返回格式: (term_spans, unmatched_terms)
    """
    term_spans = []
    unmatched_terms = []
    word_texts = [w["word"] for w in words]
    
    for term in ground_truth_terms:
        # 清理term，去掉描述部分，只保留主要词汇
        clean_term = term.lower()
        if ',' in clean_term:
            clean_term = clean_term.split(',')[0].strip()
        
        # 将term分解为单词
        term_words = clean_term.split()
        
        if not term_words:
            unmatched_terms.append(term)
            continue
        
        found = False
        
        # 精确序列匹配
        for i in range(len(word_texts) - len(term_words) + 1):
            match = True
            for j, term_word in enumerate(term_words):
                if term_word != word_texts[i + j]:
                    match = False
                    break
            
            if match:
                start_time = words[i]["start"]
                end_time = words[i + len(term_words) - 1]["end"]
                term_spans.append({
                    "term": term,
                    "start": start_time,
                    "end": end_time
                })
                found = True
                break
        
        if not found:
            unmatched_terms.append(term)
    
    print(f"[DEBUG] Found {len(term_spans)} term spans out of {len(ground_truth_terms)} terms")
    if unmatched_terms:
        print(f"[DEBUG] Unmatched terms ({len(unmatched_terms)}): {unmatched_terms[:5]}{'...' if len(unmatched_terms) > 5 else ''}")
    
    return term_spans, unmatched_terms


def calculate_chunks(begin_time: float, end_time: float, chunk_len: float = 0.96) -> List[Tuple[float, float]]:
    """
    计算chunk分割
    返回格式: [(start1, end1), (start2, end2), ...]
    """
    chunks = []
    current_time = begin_time
    
    while current_time < end_time:
        chunk_end = min(current_time + chunk_len, end_time)
        chunks.append((current_time, chunk_end))
        current_time = chunk_end
        
        # 如果剩余时间太短，直接结束
        if end_time - current_time < chunk_len * 0.1:
            break
    
    return chunks


def calculate_chunk_term_coverage(chunks: List[Tuple[float, float]], term_spans: List[Dict]) -> List[int]:
    """
    计算每个chunk覆盖的术语数量
    返回每个chunk覆盖的术语数量列表
    """
    coverage = []
    
    for chunk_start, chunk_end in chunks:
        covered_terms = 0
        for term_span in term_spans:
            # 检查是否有时间重叠
            if not (chunk_end <= term_span["start"] or chunk_start >= term_span["end"]):
                covered_terms += 1
        coverage.append(covered_terms)
    
    return coverage


def find_best_n_chunks(chunks: List[Tuple[float, float]], coverage: List[int], n: int) -> Tuple[int, int]:
    """
    找到覆盖术语最多的连续n个chunks
    返回: (起始chunk索引, 实际chunk数量)
    """
    if len(chunks) == 0:
        return 0, 0
    
    # 如果chunk数量少于n，使用所有available chunks
    actual_n = min(n, len(chunks))
    
    best_start_idx = 0
    best_coverage = sum(coverage[:actual_n])
    
    # 滑动窗口找最佳连续actual_n个chunks
    for i in range(1, len(chunks) - actual_n + 1):
        current_coverage = sum(coverage[i:i+actual_n])
        if current_coverage > best_coverage:
            best_coverage = current_coverage
            best_start_idx = i
    
    return best_start_idx, actual_n


def get_covered_terms_in_chunks(chunks: List[Tuple[float, float]], start_idx: int, actual_n: int, term_spans: List[Dict]) -> List[str]:
    """
    获取指定actual_n个chunk范围内实际覆盖的术语
    """
    if start_idx >= len(chunks) or actual_n == 0:
        return []
    
    # 确保不越界 
    end_idx = min(start_idx + actual_n, len(chunks))
    
    chunk_start = chunks[start_idx][0]
    chunk_end = chunks[end_idx - 1][1]
    
    covered_terms = set()  # 使用set来自动去重
    for term_span in term_spans:
        # 检查术语是否与chunk范围有重叠
        if not (chunk_end <= term_span["start"] or chunk_start >= term_span["end"]):
            covered_terms.add(term_span["term"])
    
    return list(covered_terms)  # 转换回list返回


def extract_chunk_text(words: List[Dict], chunk_start: float, chunk_end: float) -> str:
    """
    根据时间范围从words中提取对应的文本
    """
    chunk_words = []
    
    for word in words:
        word_start = word["start"]
        word_end = word["end"]
        
        # 检查单词是否与chunk时间范围有重叠
        if not (chunk_end <= word_start or chunk_start >= word_end):
            chunk_words.append(word["word"])
    
    return " ".join(chunk_words)


def extract_chunk_text_from_sample(original_text: str, chunk_start: float, chunk_end: float, total_duration: float) -> str:
    """
    从原始样本文本中按时间比例提取chunk对应的文本
    """
    if not original_text or total_duration <= 0:
        return ""
    
    # 计算时间比例
    start_ratio = chunk_start / total_duration
    end_ratio = chunk_end / total_duration
    
    # 确保比例在有效范围内
    start_ratio = max(0, min(1, start_ratio))
    end_ratio = max(start_ratio, min(1, end_ratio))
    
    # 按字符位置截取文本
    text_length = len(original_text)
    start_pos = int(start_ratio * text_length)
    end_pos = int(end_ratio * text_length)
    
    # 确保位置有效
    start_pos = max(0, min(text_length, start_pos))
    end_pos = max(start_pos, min(text_length, end_pos))
    
    chunk_text = original_text[start_pos:end_pos].strip()
    
    # 如果截取的文本太短或为空，尝试扩展一些
    if len(chunk_text) < 10 and end_pos < text_length:
        # 向后扩展一些字符
        extended_end = min(text_length, end_pos + 20)
        chunk_text = original_text[start_pos:extended_end].strip()
    
    return chunk_text


def extract_chunk_audio(original_audio_path: str, chunk_start: float, chunk_end: float, 
                       segment_id: str, output_dir: str = "/mnt/data/jiaxuanluo/audio_chunks") -> str:
    """
    从原始音频中提取chunk片段
    """
    try:
        # 创建输出目录结构
        audio_id = segment_id.split('_')[0] if '_' in segment_id else segment_id
        layer1 = audio_id[:3] if len(audio_id) >= 3 else audio_id
        chunk_dir = os.path.join(output_dir, layer1, audio_id)
        os.makedirs(chunk_dir, exist_ok=True)
        
        # 生成输出文件名
        chunk_filename = f"{segment_id}_chunk_{chunk_start:.2f}_{chunk_end:.2f}.wav"
        chunk_path = os.path.join(chunk_dir, chunk_filename)
        
        # 如果文件已存在，直接返回
        if os.path.exists(chunk_path):
            return chunk_path
        
        # 读取原始音频
        if not os.path.exists(original_audio_path):
            print(f"[ERROR] Original audio not found: {original_audio_path}")
            return None
            
        audio_data, sr = sf.read(original_audio_path)
        
        # 计算样本索引
        start_sample = int(chunk_start * sr)
        end_sample = int(chunk_end * sr)
        
        # 确保索引在有效范围内
        start_sample = max(0, start_sample)
        end_sample = min(len(audio_data), end_sample)
        
        if start_sample >= end_sample:
            print(f"[ERROR] Invalid chunk range for {segment_id}")
            return None
        
        # 提取音频片段
        chunk_audio = audio_data[start_sample:end_sample]
        
        # 保存chunk音频
        sf.write(chunk_path, chunk_audio, sr)
        
        return chunk_path
        
    except Exception as e:
        print(f"[ERROR] Failed to extract chunk audio for {segment_id}: {e}")
        return None


def process_sample(sample: Dict, n: int = 3, chunk_len: float = 0.96) -> Optional[Dict]:
    """
    处理单个样本，提取最佳n个chunk
    """
    segment_id = sample["segment_id"]
    begin_time = sample.get("begin_time", 0)
    end_time = sample.get("end_time", 0)
    audio_path = sample.get("audio", "")
    ground_truth_terms = sample.get("ground_truth_term", [])
    original_text = sample.get("text", "")
    
    if not ground_truth_terms:
        print(f"[SKIP] No ground truth terms for {segment_id}")
        return None
    
    # 获取TextGrid路径并解析
    textgrid_path = get_textgrid_path(segment_id)
    words = parse_textgrid(textgrid_path)
    
    if not words:
        print(f"[SKIP] No words found in TextGrid for {segment_id}")
        return None
    
    # 找到术语在音频中的时间跨度
    term_spans, unmatched_terms = find_term_time_spans(words, ground_truth_terms)
    
    # 调试信息：显示时间范围
    if words:
        textgrid_start = words[0]["start"]
        textgrid_end = words[-1]["end"]
        print(f"[DEBUG] {segment_id} - TextGrid time range: {textgrid_start:.2f} - {textgrid_end:.2f}")
        print(f"[DEBUG] {segment_id} - Sample time range: {begin_time:.2f} - {end_time:.2f}")
    
    # 如果有未匹配的术语，打印详细信息进行调试
    if unmatched_terms:
        print(f"[DEBUG] {segment_id} - Unmatched terms: {unmatched_terms}")
        print(f"[DEBUG] {segment_id} - Available words: {[w['word'] for w in words[:10]]}{'...' if len(words) > 10 else ''}")
        print(f"[DEBUG] {segment_id} - Ground truth terms: {ground_truth_terms}")
        
        # 比较原始文本和TextGrid文本（用于调试）
        if original_text:
            textgrid_text = " ".join([w["word"] for w in words]).lower()
            print(f"[DEBUG] {segment_id} - Original text: {original_text[:100]}...")
            print(f"[DEBUG] {segment_id} - TextGrid text: {textgrid_text[:100]}...")
    
    # 计算chunks（使用相对于音频片段的时间，从0开始）
    segment_duration = end_time - begin_time
    chunks = calculate_chunks(0, segment_duration, chunk_len)
    
    if len(chunks) == 0:
        print(f"[SKIP] No chunks generated for {segment_id}")
        return None
    
    # 计算每个chunk的术语覆盖
    coverage = calculate_chunk_term_coverage(chunks, term_spans)
    
    # 找到最佳的chunks（可能少于n个）
    best_start_idx, actual_n = find_best_n_chunks(chunks, coverage, n)
    
    # 获取实际覆盖的术语
    covered_terms = get_covered_terms_in_chunks(chunks, best_start_idx, actual_n, term_spans)
    
    # 计算chunk的时间范围（相对于音频片段，从0开始）
    chunk_start_time_rel = chunks[best_start_idx][0]
    chunk_end_time_rel = chunks[best_start_idx + actual_n - 1][1]
    
    # 转换为相对于原始长音频的绝对时间（用于记录）
    chunk_start_time_abs = chunk_start_time_rel + begin_time
    chunk_end_time_abs = chunk_end_time_rel + begin_time
    
    # 提取chunk文本（使用原始样本文本进行时间比例截取）
    chunk_text = extract_chunk_text_from_sample(original_text, chunk_start_time_rel, chunk_end_time_rel, segment_duration)
    
    # 提取chunk音频（使用相对时间）
    chunk_audio_path = extract_chunk_audio(audio_path, chunk_start_time_rel, chunk_end_time_rel, segment_id)
    
    if not chunk_audio_path:
        print(f"[SKIP] Failed to extract chunk audio for {segment_id}")
        return None
    
    # 即使没有覆盖的术语，也生成样本（因为原样本肯定有ground_truth_terms）
    if not covered_terms:
        # 如果没有在chunk中找到术语，使用原始的ground_truth_terms
        covered_terms = ground_truth_terms
        print(f"[INFO] Using original ground truth terms for {segment_id} (no terms found in chunks)")
    
    # 构建新样本
    new_sample = {
        "segment_id": segment_id,
        "n_chunk_audio": chunk_audio_path,
        "n_chunk_text": chunk_text,
        "n_chunk_audio_ground_truth_terms": covered_terms,
        "chunk_start_time": chunk_start_time_rel,  # 相对于音频片段的时间
        "chunk_end_time": chunk_end_time_rel,      # 相对于音频片段的时间
        "chunk_start_time_abs": chunk_start_time_abs,  # 相对于原始长音频的绝对时间
        "chunk_end_time_abs": chunk_end_time_abs,      # 相对于原始长音频的绝对时间
        "actual_chunk_count": actual_n
    }
    
    return new_sample


def main():
    parser = argparse.ArgumentParser(description="Extract n-chunk audio samples based on MFA alignment")
    parser.add_argument("--input_json", type=str, required=True, help="Input samples JSON file")
    parser.add_argument("--output_json", type=str, required=True, help="Output n-chunk samples JSON file")
    parser.add_argument("--n", type=int, default=3, help="Number of chunks to extract")
    parser.add_argument("--chunk_len", type=float, default=0.96, help="Chunk length in seconds")
    parser.add_argument("--textgrid_dir", type=str, 
                       default="/mnt/data/siqiouyang/datasets/gigaspeech/textgrids",
                       help="TextGrid files directory")
    
    args = parser.parse_args()
    
    print(f"[INFO] Loading samples from {args.input_json}")
    
    # 读取输入样本
    with open(args.input_json, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    print(f"[INFO] Processing {len(samples)} samples with n={args.n}, chunk_len={args.chunk_len}")
    
    # 处理每个样本
    processed_samples = []
    for sample in tqdm(samples, desc="Processing samples"):
        try:
            result = process_sample(sample, n=args.n, chunk_len=args.chunk_len)
            if result:
                processed_samples.append(result)
        except Exception as e:
            print(f"[ERROR] Failed to process {sample.get('segment_id', 'unknown')}: {e}")
            continue
    
    print(f"[INFO] Successfully processed {len(processed_samples)} out of {len(samples)} samples")
    
    # 保存结果
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(processed_samples, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
