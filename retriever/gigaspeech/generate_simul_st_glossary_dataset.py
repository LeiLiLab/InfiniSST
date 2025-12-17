#!/usr/bin/env python3
"""
Generate GigaSpeech Simul-ST + Glossary training dataset (Parallel + Checkpoint Version).
"""

import argparse
import ast
import concurrent.futures
import glob
import json
import logging
import os
import random
import re
import time
from typing import Dict, List, Optional, Set, Tuple

import soundfile as sf

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

BASE_CHUNK_DURATION = 0.96  # seconds per zh piece (derived from 960 ms ASR pieces)

# Cache TextGrid lookups to avoid repeated globbing/disk hits.
_TEXTGRID_PATH_CACHE: Dict[str, Optional[str]] = {}
_TEXTGRID_PREFIX_CACHE: Dict[Tuple[str, str], List[str]] = {}

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """
    Tokenize English text into lowercase word tokens.
    """
    return [tok for tok in re.findall(r"[A-Za-z0-9']+", text.lower()) if tok]


SIL_WORDS = {'', 'sil', '<sil>', '<s>', '</s>', 'sp'}


def _format_textgrid_segment_id(utt_id: str) -> str:
    """
    Normalize IDs like AUD0000000003_1 -> AUD0000000003_S0000001 to match TextGrid files.
    """
    if not utt_id or '_' not in utt_id:
        return utt_id
    prefix, suffix = utt_id.rsplit('_', 1)
    if suffix.startswith('S'):
        return utt_id
    if suffix.isdigit():
        return f"{prefix}_S{int(suffix):07d}"
    return utt_id


def _list_prefix_textgrids(prefix: str, textgrid_root: str) -> List[str]:
    """
    List and cache all TextGrid files for a given audio prefix.
    """
    cache_key = (textgrid_root, prefix)
    if cache_key in _TEXTGRID_PREFIX_CACHE:
        return _TEXTGRID_PREFIX_CACHE[cache_key]
    pattern = os.path.join(textgrid_root, f"{prefix}_S*.TextGrid")
    files = sorted(glob.glob(pattern))
    _TEXTGRID_PREFIX_CACHE[cache_key] = files
    return files


def _resolve_textgrid_path(utt_id: str, textgrid_root: Optional[str]) -> Optional[str]:
    """
    Resolve the most likely TextGrid path for a given utterance id.
    """
    if not textgrid_root or not utt_id:
        return None
    cache_key = f"{textgrid_root}|{utt_id}"
    if cache_key in _TEXTGRID_PATH_CACHE:
        return _TEXTGRID_PATH_CACHE[cache_key]

    candidates = []
    direct_name = utt_id
    formatted_name = _format_textgrid_segment_id(utt_id)
    candidates.append(os.path.join(textgrid_root, f"{direct_name}.TextGrid"))
    if formatted_name != direct_name:
        candidates.append(os.path.join(textgrid_root, f"{formatted_name}.TextGrid"))

    for path in candidates:
        if os.path.exists(path):
            _TEXTGRID_PATH_CACHE[cache_key] = path
            return path

    if '_' not in utt_id:
        _TEXTGRID_PATH_CACHE[cache_key] = None
        return None

    prefix, suffix = utt_id.rsplit('_', 1)
    if not suffix.isdigit():
        _TEXTGRID_PATH_CACHE[cache_key] = None
        return None

    try:
        idx = int(suffix)
    except ValueError:
        _TEXTGRID_PATH_CACHE[cache_key] = None
        return None

    prefix_files = _list_prefix_textgrids(prefix, textgrid_root)
    resolved_path: Optional[str] = None
    if prefix_files:
        if 0 <= idx < len(prefix_files):
            resolved_path = prefix_files[idx]
        elif 0 <= idx + 1 < len(prefix_files):
            # Handle off-by-one numbering mismatches between TSV ids and TextGrid files.
            resolved_path = prefix_files[idx + 1]

    _TEXTGRID_PATH_CACHE[cache_key] = resolved_path
    return resolved_path


def _load_textgrid_words(utt_id: str, textgrid_root: Optional[str]) -> List[Dict]:
    """
    Read the 'words' tier from the TextGrid that corresponds to utt_id.
    Returns [{"word": "foo", "start": float, "end": float}, ...].
    """
    textgrid_path = _resolve_textgrid_path(utt_id, textgrid_root)
    if not textgrid_path:
        return []

    words: List[Dict] = []
    try:
        with open(textgrid_path, 'r', encoding='utf-8') as f:
            # keep blank lines so indexing stays aligned with the compact TextGrid format
            lines = [line.strip() for line in f.readlines()]
    except Exception:
        return words

    idx = 0
    while idx < len(lines):
        if lines[idx] == '"words"':
            if idx + 3 >= len(lines):
                break
            try:
                interval_count = int(lines[idx + 3])
            except ValueError:
                break
            ptr = idx + 4
            parsed = 0
            while ptr + 2 < len(lines) and parsed < interval_count:
                start_line = lines[ptr].strip()
                end_line = lines[ptr + 1].strip()
                text_line = lines[ptr + 2].strip().strip('"')
                ptr += 3
                parsed += 1
                try:
                    start_time = float(start_line)
                    end_time = float(end_line)
                except ValueError:
                    continue
                tokenized_text = text_line.lower()
                if tokenized_text in SIL_WORDS:
                    continue
                words.append({
                    'word': tokenized_text,
                    'start': start_time,
                    'end': end_time
                })
            break
        idx += 1
    return words


def _build_textgrid_token_sequence(words: List[Dict]) -> List[Dict]:
    """
    Expand TextGrid word entries into per-token entries to simplify matching.
    """
    token_sequence: List[Dict] = []
    for word_entry in words:
        tokens = _tokenize(word_entry['word'])
        if not tokens:
            continue
        for token in tokens:
            token_sequence.append({
                'token': token,
                'start': word_entry['start'],
                'end': word_entry['end']
            })
    return token_sequence


def _match_term_span(term: str, token_sequence: List[Dict]) -> Optional[Tuple[float, float]]:
    """
    Locate the time span of a glossary term within the TextGrid token sequence.
    """
    if not term or not token_sequence:
        return None
    term_tokens = _tokenize(term)
    if not term_tokens:
        return None

    seq_len = len(token_sequence)
    term_len = len(term_tokens)
    for i in range(seq_len - term_len + 1):
        matched = True
        for j in range(term_len):
            if token_sequence[i + j]['token'] != term_tokens[j]:
                matched = False
                break
        if matched:
            start = token_sequence[i]['start']
            end = token_sequence[i + term_len - 1]['end']
            return start, end
    return None


def _pick_chunk_for_span(span: Tuple[float, float], chunk_timings: List[Tuple[float, float]]) -> Optional[int]:
    """
    Select the chunk index whose time window best overlaps the span.
    """
    if not chunk_timings:
        return None
    span_start, span_end = span
    best_idx = None
    best_overlap = 0.0
    for idx, (start_time, duration) in enumerate(chunk_timings):
        chunk_start = start_time
        chunk_end = start_time + duration
        overlap = min(span_end, chunk_end) - max(span_start, chunk_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = idx
    if best_idx is not None and best_overlap > 0:
        return best_idx

    # fall back to nearest chunk midpoint if there was no direct overlap
    span_mid = (span_start + span_end) / 2.0
    best_distance = float('inf')
    best_idx = None
    for idx, (start_time, duration) in enumerate(chunk_timings):
        mid = start_time + duration / 2.0
        distance = abs(span_mid - mid)
        if distance < best_distance:
            best_distance = distance
            best_idx = idx
    return best_idx


def _evenly_split_references(all_references: List[Dict], num_chunks: int) -> List[List[Dict]]:
    """
    Reproduce the original even-distribution logic for glossary references.
    """
    if num_chunks <= 0:
        return []
    if not all_references:
        return [[] for _ in range(num_chunks)]

    refs_per_chunk = max(1, len(all_references) // num_chunks)
    chunk_refs: List[List[Dict]] = []
    for i in range(num_chunks):
        start_idx = i * refs_per_chunk
        if i < num_chunks - 1:
            end_idx = start_idx + refs_per_chunk
        else:
            end_idx = len(all_references)
        chunk_refs.append(all_references[start_idx:end_idx])
    return chunk_refs


def _assign_references_with_textgrid(
    utt_id: str,
    all_references: List[Dict],
    chunk_timings: List[Tuple[float, float]],
    textgrid_root: Optional[str]
) -> Optional[List[List[Dict]]]:
    """
    Use MFA alignments to map each glossary term to the chunk that covers its time span.
    Returns None if we cannot load/parse the TextGrid.
    """
    words = _load_textgrid_words(utt_id, textgrid_root)
    if not words:
        return None
    token_sequence = _build_textgrid_token_sequence(words)
    if not token_sequence:
        return None

    chunk_refs: List[List[Dict]] = [[] for _ in chunk_timings]
    unmatched: List[Dict] = []

    for ref in all_references:
        span = _match_term_span(ref['term'], token_sequence)
        if not span:
            unmatched.append(ref)
            continue
        chunk_idx = _pick_chunk_for_span(span, chunk_timings)
        if chunk_idx is None:
            unmatched.append(ref)
            continue
        chunk_refs[chunk_idx].append(ref)

    if unmatched:
        fallback = _evenly_split_references(unmatched, len(chunk_timings))
        for idx, refs in enumerate(fallback):
            if refs:
                chunk_refs[idx].extend(refs)

    return chunk_refs


def parse_audio_spec(audio_spec: str) -> Tuple[str, Optional[int], Optional[int]]:
    """
    Parse audio spec strings of the form "<path>:<start>:<frames>".
    """
    if not audio_spec:
        return '', None, None
    
    parts = audio_spec.split(':')
    path = parts[0]
    start_frame: Optional[int] = None
    num_frames: Optional[int] = None
    
    if len(parts) > 1:
        try:
            start_frame = int(parts[1])
        except ValueError:
            pass
    
    if len(parts) > 2:
        try:
            num_frames = int(parts[2])
        except ValueError:
            pass
    
    return path, start_frame, num_frames


def format_user_audio_content(references: Optional[List[Dict]]) -> str:
    """
    Build the canonical user content string that includes serialized references.
    """
    return f"<audio>, references: {json.dumps(references or [], ensure_ascii=False)}"


def extract_references_from_content(content: str) -> Optional[List[Dict]]:
    """
    Recover the reference list embedded in the user content string.
    """
    marker = ', references: '
    if not content:
        return None
    idx = content.find(marker)
    if idx == -1:
        return None
    refs_str = content[idx + len(marker):].strip()
    if not refs_str:
        return []
    try:
        refs = json.loads(refs_str)
    except json.JSONDecodeError:
        return None
    if isinstance(refs, list):
        return refs
    return None


def load_glossary(
    glossary_path: str,
    min_term_length: int,
    min_term_token_len: int
) -> Dict[str, List[Dict]]:
    """
    Load glossary and build an inverted index keyed by the first token.
    """
    logger.info(f"Loading glossary from {glossary_path}")
    with open(glossary_path, 'r', encoding='utf-8') as f:
        raw_glossary = json.load(f)
    
    index: Dict[str, List[Dict]] = {}
    total_terms = 0
    indexed_terms = 0
    
    for term_key, term_info in raw_glossary.items():
        total_terms += 1
        
        zh_translation = term_info.get('target_translations', {}).get('zh')
        if not zh_translation:
            continue
        
        normalized_term = term_key.lower().strip()
        if len(normalized_term) < min_term_length:
            continue
        
        term_tokens = _tokenize(normalized_term)
        if (
            not term_tokens
            or any(len(tok) < min_term_token_len for tok in term_tokens)
        ):
            continue
        
        entry = {
            'term': term_key,
            'translation': zh_translation,
            'tokens': term_tokens
        }
        first_token = term_tokens[0]
        index.setdefault(first_token, []).append(entry)
        indexed_terms += 1
    
    logger.info(
        "Glossary indexed: kept %d / %d terms (first-token buckets=%d)",
        indexed_terms,
        total_terms,
        len(index),
    )
    return index

def parse_tsv_line(line: str) -> Optional[Dict]:
    parts = line.strip().split('\t')
    if len(parts) < 9:
        return None
    
    audio_spec = parts[1]
    audio_path, audio_start_frame, audio_num_frames = parse_audio_spec(audio_spec)
    try:
        n_frames = int(parts[2])
    except (ValueError, TypeError):
        n_frames = None
    
    try:
        zh_pieces_str = parts[8]
        try:
            zh_pieces = ast.literal_eval(zh_pieces_str)
        except (ValueError, SyntaxError):
            zh_pieces_str_clean = zh_pieces_str.replace('\x00', '')
            zh_pieces = ast.literal_eval(zh_pieces_str_clean)
        
        if not isinstance(zh_pieces, list):
            return None
    except Exception:
        return None
    
    return {
        'utt_id': parts[0],
        'audio_spec': audio_spec,
        'audio_path': audio_path,
        'audio_start_frame': audio_start_frame,
        'audio_num_frames': audio_num_frames,
        'audio_len': parts[2],
        'n_frames': n_frames,
        'en_text': parts[4],
        'zh_pieces': zh_pieces
    }

def find_chunk_glossary_terms(
    chunk_tokens: List[str],
    glossary_index: Dict[str, List[Dict]]
) -> List[Dict]:
    if not chunk_tokens:
        return []

    references: List[Dict] = []
    seen_terms: Set[str] = set()
    token_count = len(chunk_tokens)

    for i, token in enumerate(chunk_tokens):
        candidates = glossary_index.get(token)
        if not candidates:
            continue

        for entry in candidates:
            term_tokens = entry['tokens']
            term_len = len(term_tokens)
            if term_len == 0 or i + term_len > token_count:
                continue
            term_key = entry['term']
            if term_key in seen_terms:
                continue
            if chunk_tokens[i:i + term_len] == term_tokens:
                references.append({
                    'term': term_key,
                    'translation': entry['translation']
                })
                seen_terms.add(term_key)
    return references

def analyze_utterance(
    utt_data: Dict,
    glossary_index: Dict[str, List[Dict]],
    min_chunks_per_merge: int,
    max_chunks_per_merge: int,
    textgrid_root: Optional[str]
) -> Optional[Dict]:
    """
    Analyze utterance to determine chunks and references.
    This function is lightweight and runs in the main process.
    """
    utt_id = utt_data['utt_id']
    en_text = utt_data['en_text']
    zh_pieces = utt_data['zh_pieces']
    audio_spec = utt_data['audio_spec']
    audio_path = utt_data.get('audio_path')
    audio_start_frame = utt_data.get('audio_start_frame')
    audio_num_frames = utt_data.get('audio_num_frames')
    
    chunks_per_merge = random.randint(min_chunks_per_merge, max_chunks_per_merge)
    
    if not zh_pieces:
        return None
    
    chunk_zh_strings = []
    chunk_timings = []
    
    for i in range(0, len(zh_pieces), chunks_per_merge):
        merged_pieces = zh_pieces[i:i + chunks_per_merge]
        merged_zh = ''.join(merged_pieces)
        chunk_zh_strings.append(merged_zh)
        
        start_time = i * BASE_CHUNK_DURATION
        duration = len(merged_pieces) * BASE_CHUNK_DURATION
        chunk_timings.append((start_time, duration))
    
    chunk_tokens = _tokenize(en_text)
    all_references = find_chunk_glossary_terms(chunk_tokens, glossary_index)
    
    has_references = len(all_references) > 0
    has_non_empty_chunk = any(chunk.strip() for chunk in chunk_zh_strings)
    if not has_non_empty_chunk:
        return None
    
    chunk_references = _evenly_split_references(all_references, len(chunk_zh_strings))
    if all_references:
        tg_refs = _assign_references_with_textgrid(
            utt_id,
            all_references,
            chunk_timings,
            textgrid_root
        )
        if tg_refs is not None:
            chunk_references = tg_refs
        
    return {
        'utt_id': utt_id,
        'audio_spec': audio_spec,
        'audio_path': audio_path,
        'audio_start_frame': audio_start_frame,
        'audio_num_frames': audio_num_frames,
        'chunk_zh_strings': chunk_zh_strings,
        'chunk_references': chunk_references,
        'chunk_timings': chunk_timings,
        'has_references': has_references,
        'num_chunks': len(chunk_zh_strings),
        'chunk_merge_factor': chunks_per_merge
    }

# -----------------------------------------------------------------------------
# Worker Function (Runs in Sub-Process)
# -----------------------------------------------------------------------------

def process_audio_task(task: Dict) -> Optional[Dict]:
    """
    Worker function to extract audio and build JSON.
    Args:
        task: contains meta, audio_root, audio_clips_root
    """
    try:
        meta = task['meta']
        audio_root = task['audio_root']
        audio_clips_root = task['audio_clips_root']
        
        utt_id = meta['utt_id']
        
        audio_spec = meta.get('audio_spec', '')
        meta_audio_path = meta.get('audio_path')
        meta_start_frame = meta.get('audio_start_frame')
        meta_num_frames = meta.get('audio_num_frames')
        
        parsed_path, parsed_start, parsed_frames = parse_audio_spec(audio_spec)
        audio_path = meta_audio_path or parsed_path
        audio_start_frame = meta_start_frame if meta_start_frame is not None else parsed_start
        audio_num_frames = meta_num_frames if meta_num_frames is not None else parsed_frames
        
        if not audio_path:
            return None
        
        if not os.path.isabs(audio_path):
            src_audio_path = os.path.join(audio_root, audio_path)
        else:
            src_audio_path = audio_path
        
        if not os.path.exists(src_audio_path):
            return None
        
        utt_start_frame = audio_start_frame if audio_start_frame is not None else 0
        utt_total_frames = audio_num_frames
        
        # Prepare output directory
        chunk_dir = os.path.join(audio_clips_root, utt_id)
        os.makedirs(chunk_dir, exist_ok=True)
        
        # Check if this directory is already populated (optional optimization)
        # But we need to ensure we have exactly the chunks we expect.
        
        # Prepare messages
        messages = [
            {
                'role': 'system',
                'content': 'You are a professional simultaneous interpreter. You will be given chunks of English audio and you need to translate the audio into Chinese text. Use the wordlings for term reference.'
            }
        ]
        audios = []
        
        chunk_zh_strings = meta['chunk_zh_strings']
        chunk_references = meta['chunk_references']
        chunk_timings = meta['chunk_timings']
        
        # Optimize: Open source file once
        with sf.SoundFile(src_audio_path) as f_in:
            sr = f_in.samplerate
            
            for chunk_idx, (zh_text, references, timing) in enumerate(zip(chunk_zh_strings, chunk_references, chunk_timings)):
                start_time, duration = timing
                chunk_filename = f"{chunk_idx}.wav"
                chunk_path = os.path.join(chunk_dir, chunk_filename)
                
                # Check if chunk already exists
                if not os.path.exists(chunk_path):
                    chunk_start_frame = int(round(start_time * sr))
                    frames_to_read = int(round(duration * sr))
                    chunk_start_frame = max(0, chunk_start_frame)
                    frames_to_read = max(1, frames_to_read)
                    
                    global_start_frame = utt_start_frame + chunk_start_frame
                    
                    if utt_total_frames is not None:
                        remaining = utt_total_frames - chunk_start_frame
                        if remaining <= 0:
                            return None
                        frames_to_read = min(frames_to_read, remaining)
                    
                    f_in.seek(global_start_frame)
                    data = f_in.read(frames_to_read)
                    
                    if data.size == 0:
                        return None
                    
                    # Mix down to mono if needed
                    if len(data.shape) > 1 and data.shape[1] > 1:
                        data = data.mean(axis=1)
                    
                    sf.write(chunk_path, data, sr)
                
                user_content = format_user_audio_content(references)
                messages.append({
                    'role': 'user',
                    'content': user_content
                })
                messages.append({
                    'role': 'assistant',
                    'content': zh_text
                })
                audios.append(chunk_path)
        
        return {
            'result': {
                'messages': messages,
                'audios': audios
            },
            'meta': meta
        }
        
    except Exception as e:
        # Log error silently or to a file if needed, to avoid spamming main process
        return None

# -----------------------------------------------------------------------------
# Main Loop
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Generate GigaSpeech Simul-ST Dataset (Parallel)')
    parser.add_argument('--tsv', nargs='+', default=[
            '/mnt/taurus/data/siqiouyang/datasets/gigaspeech/train_xl_case_ft-qwen2.5-32b-instruct_marked_mfa_punc_asr.tsv',
            # '/mnt/taurus/data/siqiouyang/datasets/gigaspeech/dev_case_ft-qwen2.5-32b-instruct_marked_mfa_punc.tsv'
        ], help='Input TSV files')
    parser.add_argument('--audio-root', default='/mnt/taurus/data/siqiouyang/datasets/gigaspeech', help='Source audio root')
    parser.add_argument('--glossary', default='/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_cleaned.json', help='Glossary JSON')
    parser.add_argument('--audio-clips-root', default='/mnt/gemini/data1/jiaxuanluo/audio_clips', help='Output audio clips dir')
    parser.add_argument('--output', default='/mnt/gemini/data1/jiaxuanluo/simul_st_glossary_train.jsonl', help='Output JSONL file')
    parser.add_argument('--checkpoint-file', default='/mnt/gemini/data1/jiaxuanluo/processed_utts_train.txt', help='Checkpoint file for processed IDs')
    parser.add_argument('--textgrid-root', default='/mnt/taurus/data/siqiouyang/datasets/gigaspeech/textgrids', help='Directory that contains MFA TextGrid files')
    parser.add_argument('--min-chunk-merge', type=int, default=1, help='Minimum number of 0.96s chunks to merge per instance.')
    parser.add_argument('--max-chunk-merge', type=int, default=12, help='Maximum number of 0.96s chunks to merge per instance.')
    parser.add_argument('--min-gloss-term-length', type=int, default=3)
    parser.add_argument('--min-gloss-term-token-len', type=int, default=2)
    parser.add_argument('--target-chunks', type=int, default=200000)
    parser.add_argument('--max-input-lines', type=int, default=20000, help='Max lines to read from TSVs')
    parser.add_argument('--num-workers', type=int, default=16, help='Number of parallel workers')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for processing')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    if args.min_chunk_merge < 1:
        parser.error('--min-chunk-merge must be at least 1.')
    if args.max_chunk_merge < args.min_chunk_merge:
        parser.error('--max-chunk-merge must be greater than or equal to --min-chunk-merge.')
    
    random.seed(args.seed)
    logger.info(
        "Random chunk merge span: %d-%d base chunks (%.2f-%.2f seconds).",
        args.min_chunk_merge,
        args.max_chunk_merge,
        args.min_chunk_merge * BASE_CHUNK_DURATION,
        args.max_chunk_merge * BASE_CHUNK_DURATION,
    )
    
    # Init directories
    os.makedirs(args.audio_clips_root, exist_ok=True)
    
    # Load Glossary index
    glossary_index = load_glossary(
        args.glossary,
        args.min_gloss_term_length,
        args.min_gloss_term_token_len
    )
    
    # Load Checkpoint
    processed_utts = set()
    if os.path.exists(args.checkpoint_file):
        with open(args.checkpoint_file, 'r') as f:
            for line in f:
                processed_utts.add(line.strip())
    logger.info(f"Loaded {len(processed_utts)} processed utterances from checkpoint.")
    
    # Statistics
    stats = {
        'total_chunks': 0,
        'chunks_ref': 0,
        'chunks_no_ref': 0,
        'examples_count': 0
    }
    
    # Recover stats from existing output file if appending
    if os.path.exists(args.output):
        logger.info("Scanning existing output to recover stats...")
        with open(args.output, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    stats['examples_count'] += 1
                    # Count chunks roughly
                    n_chunks = len(data.get('audios', []))
                    stats['total_chunks'] += n_chunks
                    
                    # Check references
                    has_ref = False
                    for msg in data.get('messages', []):
                        if msg.get('role') != 'user':
                            continue
                        if 'references' in msg:
                            if msg['references']:
                                has_ref = True
                                break
                            continue
                        refs = extract_references_from_content(msg.get('content', ''))
                        if refs:
                            has_ref = True
                            break
                    
                    if has_ref:
                        stats['chunks_ref'] += n_chunks
                    else:
                        stats['chunks_no_ref'] += n_chunks
                except:
                    pass
        logger.info(f"Recovered stats: {stats}")

    if stats['total_chunks'] >= args.target_chunks:
        logger.info(
            "Existing output already meets target chunks (%d >= %d); nothing to do.",
            stats['total_chunks'],
            args.target_chunks,
        )
        return

    # Process Pool
    # We use max_workers to control parallelism
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers)
    
    total_lines_read = 0
    batch_tasks: List[Dict] = []
    futures: List[concurrent.futures.Future] = []
    pending_utts: List[Dict] = []
    
    # Use append mode for both files
    out_f = open(args.output, 'a', encoding='utf-8')
    ckpt_f = open(args.checkpoint_file, 'a', encoding='utf-8')
    
    start_time = time.time()

    def handle_future_result(res: Optional[Dict]):
        if not res:
            return
        result = res['result']
        meta_res = res['meta']

        out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
        ckpt_f.write(meta_res['utt_id'] + '\n')
        processed_utts.add(meta_res['utt_id'])

        n_chunks = meta_res['num_chunks']
        is_ref = meta_res['has_references']
        stats['total_chunks'] += n_chunks
        stats['examples_count'] += 1
        if is_ref:
            stats['chunks_ref'] += n_chunks
        else:
            stats['chunks_no_ref'] += n_chunks

    def poll_futures():
        nonlocal futures
        if not futures:
            return

        done, not_done = concurrent.futures.wait(
            futures,
            timeout=0,
            return_when=concurrent.futures.FIRST_COMPLETED
        )

        while done:
            for future in done:
                handle_future_result(future.result())

            out_f.flush()
            ckpt_f.flush()

            futures = list(not_done)

            if stats['total_chunks'] >= args.target_chunks or not futures:
                break

            done, not_done = concurrent.futures.wait(
                futures,
                timeout=0,
                return_when=concurrent.futures.FIRST_COMPLETED
            )
    
    try:
        for tsv_path in args.tsv:
            if stats['total_chunks'] >= args.target_chunks:
                break
            if total_lines_read >= args.max_input_lines:
                break
                
            logger.info(f"Reading TSV: {tsv_path}")
            if not os.path.exists(tsv_path):
                logger.warning(f"File not found: {tsv_path}")
                continue

            with open(tsv_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line_num == 1:
                        continue  # Header
                    
                    total_lines_read += 1
                    if total_lines_read > args.max_input_lines:
                        logger.info(f"Reached max input lines {args.max_input_lines}")
                        break
                    
                    if total_lines_read % 1000 == 0:
                        logger.info(
                            "Scanned %d lines (candidates=%d, examples=%d, chunks=%d)",
                            total_lines_read,
                            len(pending_utts),
                            stats['examples_count'],
                            stats['total_chunks'],
                        )
                        
                    utt_data = parse_tsv_line(line)
                    if not utt_data:
                        continue
                    
                    utt_id = utt_data['utt_id']
                    if utt_id in processed_utts:
                        continue
                    
                    pending_utts.append(utt_data)

        if not pending_utts:
            logger.info("No new utterances collected; exiting.")
            return

        logger.info(f"Collected {len(pending_utts)} candidate utterances before shuffling.")
        random.shuffle(pending_utts)
        logger.info("Shuffled candidates to improve domain balance.")

        for utt_data in pending_utts:
            if stats['total_chunks'] >= args.target_chunks:
                break

            meta = analyze_utterance(
                utt_data,
                glossary_index,
                args.min_chunk_merge,
                args.max_chunk_merge,
                args.textgrid_root,
            )
            if not meta:
                continue

            task = {
                'meta': meta,
                'audio_root': args.audio_root,
                'audio_clips_root': args.audio_clips_root
            }
            batch_tasks.append(task)

            if len(batch_tasks) >= args.batch_size:
                for t in batch_tasks:
                    futures.append(executor.submit(process_audio_task, t))
                batch_tasks = []
                poll_futures()

                if stats['examples_count'] % 200 == 0:
                    elapsed = time.time() - start_time
                    rate = stats['examples_count'] / elapsed if elapsed > 0 else 0
                    logger.info(
                        "Processed %d examples (%.1f ex/s). Chunks=%d (Ref=%d, NoRef=%d)",
                        stats['examples_count'],
                        rate,
                        stats['total_chunks'],
                        stats['chunks_ref'],
                        stats['chunks_no_ref'],
                    )

                if stats['total_chunks'] >= args.target_chunks:
                    break

        if batch_tasks and stats['total_chunks'] < args.target_chunks:
            for t in batch_tasks:
                futures.append(executor.submit(process_audio_task, t))
            batch_tasks = []
            poll_futures()

        logger.info(f"Waiting for {len(futures)} remaining tasks...")
        for future in concurrent.futures.as_completed(futures):
            if stats['total_chunks'] >= args.target_chunks:
                break

            handle_future_result(future.result())

            if stats['examples_count'] % 100 == 0:
                out_f.flush()
                ckpt_f.flush()
                logger.info(
                    "Finishing... Examples=%d, Chunks=%d",
                    stats['examples_count'],
                    stats['total_chunks'],
                )

    except KeyboardInterrupt:
        logger.info("Interrupted! Saving progress...")
    finally:
        logger.info("Shutting down executor...")
        executor.shutdown(wait=True)
        out_f.close()
        ckpt_f.close()
        logger.info(f"Done. Final Stats: {stats}")

if __name__ == '__main__':
    main()
