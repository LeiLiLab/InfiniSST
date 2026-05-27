#!/usr/bin/env python3
"""
Quick test script to validate the core logic without loading RAG model.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    from flashtext import KeywordProcessor
    HAS_FLASHTEXT = True
except ImportError:
    HAS_FLASHTEXT = False

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Copy helper functions (to avoid importing torch)
# ============================================================================

def extract_utter_id_from_audio_path(audio_path: str) -> Optional[str]:
    """Extract utter_id from audio path."""
    try:
        parts = Path(audio_path).parts
        if len(parts) >= 3:
            speaker_id = parts[-3]
            utt_num = parts[-2]
            return f"{speaker_id}_{utt_num}"
    except Exception as e:
        print(f"Warning: Failed to extract utter_id from {audio_path}: {e}")
    return None


def split_trajectory_by_chunks(
    trajectory: List[str],
    num_chunks: int
) -> List[List[str]]:
    """Split trajectory evenly into num_chunks parts."""
    if not trajectory or num_chunks <= 0:
        return [[] for _ in range(num_chunks)]
    
    chunk_size = (len(trajectory) + num_chunks - 1) // num_chunks
    
    chunks = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(trajectory))
        chunks.append(trajectory[start_idx:end_idx])
    
    return chunks


def generate_term_map_string(terms: List[Tuple[str, str]]) -> str:
    """Generate term_map string."""
    if not terms:
        return ""
    
    lines = ["term_map:"]
    for source, target in terms:
        lines.append(f"{source}={target}")
    
    return "\n".join(lines)


def load_glossary(glossary_path: str, target_lang: str = "zh"):
    """Load glossary and initialize FlashText keyword processor."""
    if not HAS_FLASHTEXT:
        print("FlashText not available, skipping")
        return {}, None
    
    with open(glossary_path, 'r', encoding='utf-8') as f:
        glossary = json.load(f)
    
    term_map = {}
    keyword_processor = KeywordProcessor(case_sensitive=False)
    
    for term_key, term_data in glossary.items():
        if 'target_translations' in term_data and target_lang in term_data['target_translations']:
            source_term = term_data.get('term', term_key)
            target_term = term_data['target_translations'][target_lang]
            
            term_map[source_term.lower()] = {
                'source': source_term,
                'target': target_term
            }
            
            keyword_processor.add_keyword(source_term.lower(), source_term.lower())
    
    return term_map, keyword_processor


def match_gt_terms(
    text: str,
    keyword_processor,
    term_map: Dict[str, Dict]
) -> List[Tuple[str, str]]:
    """Match ground truth terms in text using FlashText."""
    if not HAS_FLASHTEXT or keyword_processor is None:
        return []
    
    matched_keys = keyword_processor.extract_keywords(text.lower())
    
    gt_terms = []
    seen = set()
    
    for key in matched_keys:
        if key in term_map and key not in seen:
            term_info = term_map[key]
            gt_terms.append((term_info['source'], term_info['target']))
            seen.add(key)
    
    return gt_terms

def test_extract_utter_id():
    """Test utter_id extraction from audio path."""
    test_cases = [
        ("/mnt/gemini/data1/jiaxuanluo/audio_clips_siqi_v3/YOU0000010238/66/0.wav", "YOU0000010238_66"),
        ("/path/to/POD0000012057/5/1.wav", "POD0000012057_5"),
        ("invalid/path.wav", None),
    ]
    
    print("Testing extract_utter_id_from_audio_path...")
    for path, expected in test_cases:
        result = extract_utter_id_from_audio_path(path)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {path} → {result} (expected: {expected})")
    print()


def test_split_trajectory():
    """Test trajectory splitting into chunks."""
    print("Testing split_trajectory_by_chunks...")
    
    trajectory = ['But if', 'someone has a giant', "stack of them there's", 
                  'probably a fairly good', "chance they're not happy"]
    
    test_cases = [
        (2, 3),  # 2 chunks, expect ~2-3 items per chunk
        (3, 2),  # 3 chunks, expect ~2 items per chunk
    ]
    
    for num_chunks, expected_per_chunk in test_cases:
        result = split_trajectory_by_chunks(trajectory, num_chunks)
        print(f"  Split into {num_chunks} chunks:")
        for i, chunk in enumerate(result):
            print(f"    Chunk {i}: {chunk}")
        print()


def test_term_map_generation():
    """Test term_map string generation."""
    print("Testing generate_term_map_string...")
    
    terms = [
        ("social statement", "社会声明"),
        ("let", "让"),
        ("relationship", "关系"),
    ]
    
    result = generate_term_map_string(terms)
    print("Generated term_map:")
    print(result)
    print()


def test_glossary_matching():
    """Test glossary loading and term matching."""
    print("Testing glossary matching...")
    
    glossary_path = "/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_used.json"
    
    try:
        term_map, keyword_processor = load_glossary(glossary_path, target_lang="zh")
        print(f"  Loaded {len(term_map)} terms from glossary")
        
        # Test matching
        test_text = "Jessica Valenti is a feminist author. Bear Stearns was a company."
        matched = match_gt_terms(test_text, keyword_processor, term_map)
        
        print(f"  Test text: {test_text}")
        print(f"  Matched terms: {matched}")
        print()
    except Exception as e:
        print(f"  Could not test glossary matching: {e}")
        print()


def test_message_processing():
    """Test message processing logic."""
    print("Testing message processing flow...")
    
    # Sample input message
    input_msg = {
        "messages": [
            {"role": "system", "content": "You are a professional simultaneous interpreter."},
            {"role": "user", "content": "<audio>"},
            {"role": "assistant", "content": "translation chunk 1"},
            {"role": "user", "content": "<audio>"},
            {"role": "assistant", "content": "translation chunk 2"},
        ],
        "audios": [
            "/path/to/audio/chunk0.wav",
            "/path/to/audio/chunk1.wav",
        ]
    }
    
    # Simulate term augmentation
    term_map_str = generate_term_map_string([
        ("example term", "示例术语"),
        ("test word", "测试词"),
    ])
    
    new_messages = []
    for msg in input_msg["messages"]:
        if msg["role"] == "user" and msg["content"] == "<audio>":
            # Augment first audio message only (for demo)
            if sum(1 for m in new_messages if m["role"] == "user") == 0:
                new_content = f"<audio>\n\n{term_map_str}"
                new_messages.append({"role": "user", "content": new_content})
            else:
                new_messages.append(msg)
        else:
            new_messages.append(msg)
    
    print("  Input messages:")
    print(json.dumps(input_msg, indent=2, ensure_ascii=False))
    print("\n  Output messages (with term_map):")
    print(json.dumps({"messages": new_messages, "audios": input_msg["audios"]}, 
                     indent=2, ensure_ascii=False))
    print()


if __name__ == '__main__':
    print("=" * 80)
    print("Testing term_map dataset construction logic")
    print("=" * 80)
    print()
    
    test_extract_utter_id()
    test_split_trajectory()
    test_term_map_generation()
    test_glossary_matching()
    test_message_processing()
    
    print("=" * 80)
    print("All tests completed!")
    print("=" * 80)

