#!/usr/bin/env python3
"""
基于 MFA 对齐信息，为每个 ground truth term 生成单独的 term-level chunk 音频片段
每个 chunk 只覆盖一个完整的 term，不被截断，并在前后各扩展1秒上下文
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
                    
                    # 只添加非空单词（排除静音标记等）
                    if word and word != '' and word not in ['<SIL>', 'SIL', '<s>', '</s>', 'sp', 'sil']:
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
    每个term_span包含: {"term": "original_term", "start": float, "end": float}
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
                    "term": term,  # 保留原始term
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


def extract_term_audio(original_audio_path: str, term_start: float, term_end: float, 
                      segment_id: str, term: str, output_dir: str = "/mnt/gemini/data1/jiaxuanluo/term_chunks",
                      context_seconds: float = 1.0) -> str:
    """
    从原始音频中提取单个term的音频片段，前后各扩展context_seconds秒
    """
    try:
        # 创建输出目录结构
        audio_id = segment_id.split('_')[0] if '_' in segment_id else segment_id
        layer1 = audio_id[:3] if len(audio_id) >= 3 else audio_id
        chunk_dir = os.path.join(output_dir, layer1, audio_id)
        os.makedirs(chunk_dir, exist_ok=True)
        
        # 生成输出文件名，包含term信息（清理特殊字符）
        safe_term = re.sub(r'[^\w\s-]', '', term).strip().replace(' ', '_')[:20]  # 限制长度
        chunk_filename = f"{segment_id}_term_{safe_term}_{term_start:.2f}_{term_end:.2f}_ctx{context_seconds:.1f}s.wav"
        chunk_path = os.path.join(chunk_dir, chunk_filename)
        
        # 如果文件已存在，直接返回
        if os.path.exists(chunk_path):
            return chunk_path
        
        # 读取原始音频
        if not os.path.exists(original_audio_path):
            print(f"[ERROR] Original audio not found: {original_audio_path}")
            return None
            
        audio_data, sr = sf.read(original_audio_path)
        
        # 计算扩展后的时间范围（前后各加context_seconds秒）
        extended_start = term_start - context_seconds
        extended_end = term_end + context_seconds
        
        # 计算样本索引
        start_sample = int(extended_start * sr)
        end_sample = int(extended_end * sr)
        
        # 确保索引在有效范围内
        start_sample = max(0, start_sample)
        end_sample = min(len(audio_data), end_sample)
        
        if start_sample >= end_sample:
            print(f"[ERROR] Invalid term range for {segment_id}: {term}")
            return None
        
        # 提取音频片段
        term_audio = audio_data[start_sample:end_sample]
        
        # 检查音频长度是否太短（扩展后的最小长度检查）
        if len(term_audio) < sr * 0.5:  # 少于0.5秒（考虑到扩展了上下文）
            print(f"[WARNING] Extended term audio too short for {segment_id}: {term} ({len(term_audio)/sr:.3f}s)")
            # 可以选择跳过或者扩展，这里选择继续保存
        
        # 保存term音频
        sf.write(chunk_path, term_audio, sr)
        
        return chunk_path
        
    except Exception as e:
        print(f"[ERROR] Failed to extract term audio for {segment_id} - {term}: {e}")
        return None


def process_sample(sample: Dict, textgrid_base_dir: str, context_seconds: float = 1.0, generate_no_term_ratio: float = 0.1) -> List[Dict]:
    """
    处理单个样本，为每个ground truth term生成单独的chunk，前后各扩展context_seconds秒
    同时根据generate_no_term_ratio比例生成一些no-term chunks用于训练拒答能力
    返回term-level chunk样本列表（包括term chunks和no-term chunks）
    """
    segment_id = sample["segment_id"]
    begin_time = sample.get("begin_time", 0)
    end_time = sample.get("end_time", 0)
    audio_path = sample.get("audio", "")
    ground_truth_terms = sample.get("ground_truth_term", [])
    original_text = sample.get("text", "")
    
    if not ground_truth_terms:
        print(f"[SKIP] No ground truth terms for {segment_id}")
        return []
    
    # 获取TextGrid路径并解析
    textgrid_path = get_textgrid_path(segment_id, textgrid_base_dir)
    words = parse_textgrid(textgrid_path)
    
    if not words:
        print(f"[SKIP] No words found in TextGrid for {segment_id}")
        return []
    
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
    
    # 为每个找到的term生成chunk
    term_chunks = []
    segment_duration = end_time - begin_time
    
    for term_span in term_spans:
        term = term_span["term"]
        term_start_abs = term_span["start"]  # MFA给出的是相对于整个音频的时间
        term_end_abs = term_span["end"]
        
        # MFA给出的时间应该已经是相对于当前音频片段的时间
        # 先尝试直接使用MFA时间，如果超出范围再考虑调整
        term_start_rel = term_start_abs
        term_end_rel = term_end_abs
        
        # 检查是否需要时间偏移调整
        if term_start_rel < 0 or term_end_rel > segment_duration:
            # 如果超出范围，可能MFA时间是绝对时间，需要减去begin_time
            term_start_rel_adjusted = term_start_abs - begin_time
            term_end_rel_adjusted = term_end_abs - begin_time
            
            # 检查调整后是否合理
            if (0 <= term_start_rel_adjusted <= segment_duration and 
                0 <= term_end_rel_adjusted <= segment_duration and
                term_end_rel_adjusted > term_start_rel_adjusted):
                print(f"[INFO] Using time offset adjustment for {segment_id} - {term}")
                term_start_rel = term_start_rel_adjusted
                term_end_rel = term_end_rel_adjusted
            else:
                print(f"[WARNING] Term '{term}' time range ({term_start_rel:.2f}-{term_end_rel:.2f}) exceeds segment duration ({segment_duration:.2f}) for {segment_id}")
                print(f"[WARNING] Adjusted range ({term_start_rel_adjusted:.2f}-{term_end_rel_adjusted:.2f}) also invalid")
                # 跳过这个术语
                continue
        
        # 提取term音频（前后各扩展指定秒数）
        term_audio_path = extract_term_audio(audio_path, term_start_rel, term_end_rel, segment_id, term, 
                                           context_seconds=context_seconds)
        
        if not term_audio_path:
            print(f"[SKIP] Failed to extract term audio for {segment_id} - {term}")
            continue
        
        # 计算扩展后的时间范围
        extended_start_rel = max(0, term_start_rel - context_seconds)
        extended_end_rel = min(segment_duration, term_end_rel + context_seconds)
        
        # 提取扩展后chunk的文本内容
        # 优先使用MFA对齐的words来提取精确的文本
        chunk_text = ""
        if words:
            # 使用MFA words提取扩展后时间范围内的文本
            chunk_text = extract_chunk_text(words, extended_start_rel, extended_end_rel)
        
        # 如果MFA文本提取失败或为空，回退到原始文本的时间比例提取
        if not chunk_text.strip() and original_text:
            chunk_text = extract_chunk_text_from_sample(
                original_text, extended_start_rel, extended_end_rel, segment_duration
            )
        
        # 如果仍然没有文本，使用原始term作为回退
        if not chunk_text.strip():
            chunk_text = term
            print(f"[WARNING] Using term as fallback text for {segment_id} - {term}")
        
        # 构建term-level chunk样本（包含上下文扩展信息）
        term_chunk = {
            "segment_id": segment_id,
            "term_chunk_audio": term_audio_path,
            "term_chunk_text": chunk_text,  # 使用扩展后chunk对应的文本
            "term_chunk_audio_ground_truth_terms": [term],  # 只包含这一个term
            "term_start_time": term_start_rel,      # 相对于音频片段的时间（原始term边界）
            "term_end_time": term_end_rel,          # 相对于音频片段的时间（原始term边界）
            "term_start_time_abs": term_start_abs,  # 相对于原始长音频的绝对时间（原始term边界）
            "term_end_time_abs": term_end_abs,      # 相对于原始长音频的绝对时间（原始term边界）
            "term_duration": term_end_rel - term_start_rel,  # 原始term时长
            # 扩展后的信息
            "extended_start_time": extended_start_rel,      # 扩展后的开始时间
            "extended_end_time": extended_end_rel,          # 扩展后的结束时间
            "extended_duration": extended_end_rel - extended_start_rel,  # 扩展后的总时长
            "context_seconds": context_seconds,             # 前后扩展的秒数
            "actual_extended_start": term_start_rel - context_seconds,  # 理论扩展开始时间（可能为负）
            "actual_extended_end": term_end_rel + context_seconds       # 理论扩展结束时间（可能超出边界）
        }
        
        term_chunks.append(term_chunk)
    
    print(f"[INFO] Generated {len(term_chunks)} term chunks for {segment_id} (out of {len(ground_truth_terms)} terms)")
    
    # 打印前几个chunk的文本提取信息用于调试
    for i, chunk in enumerate(term_chunks[:3]):  # 只打印前3个
        chunk_text_preview = chunk["term_chunk_text"][:50] + "..." if len(chunk["term_chunk_text"]) > 50 else chunk["term_chunk_text"]
        print(f"[DEBUG] Chunk {i+1} - Term: '{chunk['term_chunk_audio_ground_truth_terms'][0]}', Text: '{chunk_text_preview}'")
    
    # === 生成no-term chunks ===
    no_term_chunks = []
    if generate_no_term_ratio > 0 and len(term_chunks) > 0:
        # 计算需要生成的no-term chunk数量
        target_no_term_count = max(1, int(len(term_chunks) * generate_no_term_ratio))
        
        # 找到所有术语覆盖的时间区间（扩展后的）
        covered_intervals = []
        for chunk in term_chunks:
            start_time = chunk["extended_start_time"]
            end_time = chunk["extended_end_time"]
            covered_intervals.append((start_time, end_time))
        
        # 合并重叠的区间
        covered_intervals.sort()
        merged_intervals = []
        for start, end in covered_intervals:
            if merged_intervals and start <= merged_intervals[-1][1]:
                # 合并重叠区间
                merged_intervals[-1] = (merged_intervals[-1][0], max(merged_intervals[-1][1], end))
            else:
                merged_intervals.append((start, end))
        
        # 找到空白区间（未被术语覆盖的区域）
        gap_intervals = []
        prev_end = 0
        for start, end in merged_intervals:
            if start > prev_end:
                gap_intervals.append((prev_end, start))
            prev_end = end
        
        # 添加最后一个空白区间
        if prev_end < segment_duration:
            gap_intervals.append((prev_end, segment_duration))
        
        # 过滤掉太短的空白区间（至少需要2*context_seconds + 0.5秒）
        min_gap_duration = 2 * context_seconds + 0.5
        valid_gaps = [(start, end) for start, end in gap_intervals if (end - start) >= min_gap_duration]
        
        print(f"[DEBUG] {segment_id} - Found {len(valid_gaps)} valid gaps for no-term chunks (min duration: {min_gap_duration:.1f}s)")
        
        # 在有效空白区间中随机采样生成no-term chunks
        import random
        random.seed(42)  # 固定随机种子确保可复现
        
        generated_no_term = 0
        for gap_start, gap_end in valid_gaps:
            if generated_no_term >= target_no_term_count:
                break
            
            # 在这个空白区间中生成一个no-term chunk
            gap_duration = gap_end - gap_start
            chunk_duration = 2 * context_seconds + random.uniform(0.5, min(2.0, gap_duration - 2 * context_seconds))
            
            # 随机选择chunk在gap中的位置
            max_start = gap_end - chunk_duration
            chunk_start = random.uniform(gap_start, max_start)
            chunk_end = chunk_start + chunk_duration
            
            # 提取no-term chunk的音频
            no_term_audio_path = extract_term_audio(
                audio_path, chunk_start, chunk_end, segment_id, 
                f"no_term_{generated_no_term}", context_seconds=0  # no-term chunk不需要额外扩展
            )
            
            if no_term_audio_path:
                # 提取chunk的文本内容
                chunk_text = ""
                if words:
                    chunk_text = extract_chunk_text(words, chunk_start, chunk_end)
                
                if not chunk_text.strip() and original_text:
                    chunk_text = extract_chunk_text_from_sample(
                        original_text, chunk_start, chunk_end, segment_duration
                    )
                
                # 如果仍然没有文本，使用空字符串
                if not chunk_text.strip():
                    chunk_text = ""
                
                # 构建no-term chunk样本
                no_term_chunk = {
                    "segment_id": segment_id + f"_no_term_{generated_no_term}",
                    "term_chunk_audio": no_term_audio_path,
                    "term_chunk_text": chunk_text,
                    "term_chunk_audio_ground_truth_terms": [],  # 空术语列表
                    "term_start_time": chunk_start,
                    "term_end_time": chunk_end,
                    "term_start_time_abs": chunk_start,
                    "term_end_time_abs": chunk_end,
                    "term_duration": chunk_end - chunk_start,
                    "extended_start_time": chunk_start,
                    "extended_end_time": chunk_end,
                    "extended_duration": chunk_end - chunk_start,
                    "context_seconds": 0,  # no-term chunk不需要context扩展
                    "actual_extended_start": chunk_start,
                    "actual_extended_end": chunk_end,
                    "is_no_term_chunk": True  # 标记为no-term chunk
                }
                
                no_term_chunks.append(no_term_chunk)
                generated_no_term += 1
                
                print(f"[DEBUG] Generated no-term chunk {generated_no_term} at {chunk_start:.2f}-{chunk_end:.2f}s, text: '{chunk_text[:30]}...'")
    
    all_chunks = term_chunks + no_term_chunks
    if no_term_chunks:
        print(f"[INFO] Generated {len(no_term_chunks)} no-term chunks for {segment_id}")
    
    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="Extract term-level audio chunks based on MFA alignment with context extension")
    parser.add_argument("--input_json", type=str, required=True, help="Input samples JSON file")
    parser.add_argument("--output_json", type=str, required=True, help="Output term-level chunks JSON file")
    parser.add_argument("--textgrid_dir", type=str, 
                       default="/mnt/data/siqiouyang/datasets/gigaspeech/textgrids",
                       help="TextGrid files directory")
    parser.add_argument("--output_audio_dir", type=str,
                       default="/mnt/gemini/data1/jiaxuanluo/term_chunks",
                       help="Output directory for term audio chunks")
    parser.add_argument("--context_seconds", type=float, default=1.0,
                       help="Number of seconds to extend before and after each term (default: 1.0)")
    parser.add_argument("--generate_no_term_ratio", type=float, default=0.1,
                       help="Ratio of no-term chunks to generate relative to term chunks (default: 0.1)")
    
    args = parser.parse_args()
    
    print(f"[INFO] Loading samples from {args.input_json}")
    
    # 读取输入样本
    with open(args.input_json, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    print(f"[INFO] Processing {len(samples)} samples for term-level chunks with {args.context_seconds}s context extension")
    
    # 确保输出音频目录存在
    os.makedirs(args.output_audio_dir, exist_ok=True)
    
    # 处理每个样本
    all_term_chunks = []
    total_terms_processed = 0
    processed_samples = 0
    skipped_samples = 0
    
    for sample in tqdm(samples, desc="Processing samples"):
        try:
            term_chunks = process_sample(sample, args.textgrid_dir, args.context_seconds, args.generate_no_term_ratio)
            all_term_chunks.extend(term_chunks)
            total_terms_processed += len(term_chunks)
            processed_samples += 1
            
            # 每1000个样本打印进度
            if processed_samples % 1000 == 0:
                print(f"[PROGRESS] Processed {processed_samples}/{len(samples)} samples, generated {total_terms_processed} term chunks")
                
        except Exception as e:
            print(f"[ERROR] Failed to process {sample.get('segment_id', 'unknown')}: {e}")
            skipped_samples += 1
            continue
    
    # 统计term chunks和no-term chunks
    term_chunks_count = sum(1 for chunk in all_term_chunks if not chunk.get('is_no_term_chunk', False))
    no_term_chunks_count = sum(1 for chunk in all_term_chunks if chunk.get('is_no_term_chunk', False))
    
    print(f"[INFO] Processing completed!")
    print(f"[INFO] - Total input samples: {len(samples)}")
    print(f"[INFO] - Successfully processed samples: {processed_samples}")
    print(f"[INFO] - Skipped samples: {skipped_samples}")
    print(f"[INFO] - Generated term chunks: {term_chunks_count}")
    print(f"[INFO] - Generated no-term chunks: {no_term_chunks_count}")
    print(f"[INFO] - Total chunks: {total_terms_processed}")
    print(f"[INFO] - No-term ratio: {no_term_chunks_count/total_terms_processed:.1%}" if total_terms_processed > 0 else "[INFO] - No chunks generated")
    print(f"[INFO] - Context extension: ±{args.context_seconds}s per term")
    print(f"[INFO] - Average chunks per processed sample: {total_terms_processed/processed_samples:.2f}" if processed_samples > 0 else "[INFO] - No samples processed successfully")
    
    # 保存结果
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(all_term_chunks, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Results saved to {args.output_json}")
    print(f"✅ Term audio chunks (with ±{args.context_seconds}s context) saved to {args.output_audio_dir}")
    print(f"📊 Each chunk contains:")
    print(f"   - Original term boundaries and timing information")
    print(f"   - Extended audio with ±{args.context_seconds}s context")
    print(f"   - Metadata about actual vs. theoretical extension ranges")


if __name__ == "__main__":
    main() 