#!/usr/bin/env python3
"""
基于 MFA 对齐信息和已有语音片段数据，为每个 ground truth term 生成 term-level chunk 音频片段
"""

import os
import json
import argparse
import soundfile as sf
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
            words_content = content[words_tier_start:]
        else:
            words_content = content[words_tier_start:phones_tier_start]
        
        lines = words_content.split('\n')
        
        interval_count = 0
        start_parsing = False
        parse_start_idx = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line == '"words"' and i + 3 < len(lines):
                try:
                    if (lines[i+1].strip().replace('.', '').isdigit() and 
                        lines[i+2].strip().replace('.', '').isdigit() and
                        lines[i+3].strip().isdigit()):
                        interval_count = int(lines[i+3].strip())
                        start_parsing = True
                        parse_start_idx = i + 4
                        break
                except (ValueError, IndexError):
                    continue
        
        if not start_parsing:
            print(f"[WARNING] Could not find interval count in words tier")
            return words
        
        i = parse_start_idx
        parsed_intervals = 0
        
        while i < len(lines) and parsed_intervals < interval_count:
            line = lines[i].strip()
            if not line:
                i += 1
                continue
                
            try:
                if line.replace('.', '').replace('-', '').isdigit():
                    start_time = float(line)
                    i += 1
                    if i >= len(lines):
                        break
                    end_time = float(lines[i].strip())
                    i += 1
                    if i >= len(lines):
                        break
                    word_line = lines[i].strip()
                    word = word_line.strip('"')
                    i += 1
                    if word and word != '':
                        words.append({
                            "word": word.lower(),
                            "start": start_time,
                            "end": end_time
                        })
                    parsed_intervals += 1
                else:
                    i += 1
            except (ValueError, IndexError):
                i += 1
                continue
                
        print(f"[INFO] Parsed {len(words)} words from {textgrid_path} (expected {interval_count})")
        
    except Exception as e:
        print(f"[ERROR] Failed to parse TextGrid {textgrid_path}: {e}")
    
    return words


def get_textgrid_path(segment_id: str, textgrid_base_dir: str) -> str:
    """根据segment_id获取对应的TextGrid文件路径"""
    textgrid_filename = f"{segment_id}.TextGrid"
    return os.path.join(textgrid_base_dir, textgrid_filename)


def find_term_time_spans(words: List[Dict], ground_truth_terms: List[str]) -> List[Dict]:
    """
    在words中找到每个ground_truth_term的起止时间
    返回: [{"term": term, "start": start_time, "end": end_time}, ...]
    """
    term_spans = []
    word_texts = [w["word"] for w in words]
    
    for term in ground_truth_terms:
        clean_term = term.lower()
        if ',' in clean_term:
            clean_term = clean_term.split(',')[0].strip()
        term_words = clean_term.split()
        if not term_words:
            continue
        for i in range(len(word_texts) - len(term_words) + 1):
            if all(term_words[j] == word_texts[i + j] for j in range(len(term_words))):
                start_time = words[i]["start"]
                end_time = words[i + len(term_words) - 1]["end"]
                term_spans.append({
                    "term": term,
                    "start": start_time,
                    "end": end_time
                })
                break
    return term_spans


def extract_chunk_audio(original_audio_path: str, chunk_start: float, chunk_end: float, 
                       segment_id: str, output_dir: str) -> Optional[str]:
    """
    从原始音频中提取chunk片段
    """
    try:
        audio_id = segment_id.split('_')[0] if '_' in segment_id else segment_id
        layer1 = audio_id[:3] if len(audio_id) >= 3 else audio_id
        chunk_dir = os.path.join(output_dir, layer1, audio_id)
        os.makedirs(chunk_dir, exist_ok=True)
        
        chunk_filename = f"{segment_id}_term_{chunk_start:.2f}_{chunk_end:.2f}.wav"
        chunk_path = os.path.join(chunk_dir, chunk_filename)
        
        if os.path.exists(chunk_path):
            return chunk_path
        
        if not os.path.exists(original_audio_path):
            print(f"[ERROR] Original audio not found: {original_audio_path}")
            return None
            
        audio_data, sr = sf.read(original_audio_path)
        
        start_sample = int(chunk_start * sr)
        end_sample = int(chunk_end * sr)
        
        start_sample = max(0, start_sample)
        end_sample = min(len(audio_data), end_sample)
        
        if start_sample >= end_sample:
            print(f"[ERROR] Invalid chunk range for {segment_id}")
            return None
        
        chunk_audio = audio_data[start_sample:end_sample]
        sf.write(chunk_path, chunk_audio, sr)
        
        return chunk_path
        
    except Exception as e:
        print(f"[ERROR] Failed to extract chunk audio for {segment_id}: {e}")
        return None


def process_sample(sample: Dict, textgrid_dir: str, output_dir: str) -> List[Dict]:
    """
    对单个样本的每个term生成term-level chunk
    """
    results = []
    segment_id = sample["segment_id"]
    begin_time = sample.get("begin_time", 0)
    audio_path = sample.get("audio", "")
    ground_truth_terms = sample.get("ground_truth_term", [])
    
    if not ground_truth_terms:
        return results
    
    textgrid_path = get_textgrid_path(segment_id, textgrid_dir)
    words = parse_textgrid(textgrid_path)
    if not words:
        return results
    
    term_spans = find_term_time_spans(words, ground_truth_terms)
    for span in term_spans:
        term_start_abs = span["start"]
        term_end_abs = span["end"]
        
        # 相对时间（相对于该片段）
        term_start_rel = term_start_abs - begin_time
        term_end_rel = term_end_abs - begin_time
        
        chunk_audio_path = extract_chunk_audio(audio_path, term_start_rel, term_end_rel, segment_id, output_dir)
        if not chunk_audio_path:
            continue
        
        results.append({
            "segment_id": segment_id,
            "term_chunk_audio": chunk_audio_path,
            "term_chunk_text": span["term"],
            "term_chunk_audio_ground_truth_terms": [span["term"]],
            "term_start_time": term_start_rel,
            "term_end_time": term_end_rel,
            "term_start_time_abs": term_start_abs,
            "term_end_time_abs": term_end_abs
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Extract term-level audio chunks based on MFA alignment")
    parser.add_argument("--input_json", type=str, required=True, help="Input samples JSON file")
    parser.add_argument("--output_json", type=str, required=True, help="Output term-level chunks JSON file")
    parser.add_argument("--textgrid_dir", type=str, required=True, help="TextGrid files directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save term-level audio chunks")
    args = parser.parse_args()
    
    with open(args.input_json, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    all_results = []
    for sample in tqdm(samples, desc="Processing samples"):
        try:
            results = process_sample(sample, args.textgrid_dir, args.output_dir)
            all_results.extend(results)
        except Exception as e:
            print(f"[ERROR] Failed to process {sample.get('segment_id', 'unknown')}: {e}")
            continue
    
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] Saved {len(all_results)} term-level chunks to {args.output_json}")


if __name__ == "__main__":
    main()
