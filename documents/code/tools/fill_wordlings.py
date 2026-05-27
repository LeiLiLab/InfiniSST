#!/usr/bin/env python3
"""
Populate "wordlings" dictionaries for Simul-ST JSONL datasets by reusing the
TextGrid-based glossary assignment logic from the generator pipeline.

Key optimizations:
1. Chunk timings are derived purely from trajectory size (len(zh_pieces)) and
   BASE_CHUNK_DURATION (0.96s), exactly like generate_simul_st_glossary_dataset.py.
   No audio file reading is ever needed.
2. Uses ProcessPoolExecutor with worker initializer for true multi-core parallelism.
   Heavy objects (glossary_index, meta_map) are initialized once per worker process.
3. TSV is pre-filtered to only load utterances that appear in the JSONL.
4. Supports checkpoint file to avoid reprocessing on restarts.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import concurrent.futures
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from retriever.gigaspeech.generate_simul_st_glossary_dataset import (
    _assign_references_with_textgrid,
    _tokenize,
    find_chunk_glossary_terms,
    load_glossary,
    parse_tsv_line,
    BASE_CHUNK_DURATION,
)


logger = logging.getLogger("fill_wordlings")

# Global variables for worker processes (initialized via worker_init)
_worker_glossary_index: Optional[Dict[str, List[Dict]]] = None
_worker_meta_map: Optional[Dict[str, Dict]] = None
_worker_textgrid_root: Optional[str] = None
_worker_keep_empty: bool = False


def worker_init(
    glossary_index: Dict[str, List[Dict]],
    meta_map: Dict[str, Dict],
    textgrid_root: str,
    keep_empty: bool,
) -> None:
    """
    Initializer function for worker processes.
    Sets up global variables that are shared across all tasks in a worker.
    """
    global _worker_glossary_index, _worker_meta_map, _worker_textgrid_root, _worker_keep_empty
    _worker_glossary_index = glossary_index
    _worker_meta_map = meta_map
    _worker_textgrid_root = textgrid_root
    _worker_keep_empty = keep_empty


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill wordlings for Simul-ST JSONL using glossary matches"
    )
    parser.add_argument("--json-input", required=True, help="Source JSONL file")
    parser.add_argument("--json-output", required=True, help="Target JSONL file")
    parser.add_argument(
        "--tsv",
        nargs="+",
        required=True,
        help="One or more GigaSpeech TSV files that contain utterance metadata",
    )
    parser.add_argument(
        "--glossary",
        default="/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_cleaned.json",
        help="Glossary JSON file with target translations",
    )
    parser.add_argument(
        "--textgrid-root",
        default="/mnt/data/siqiouyang/datasets/gigaspeech/textgrids",
        help="Directory that contains MFA TextGrid files",
    )
    parser.add_argument(
        "--checkpoint-file",
        default=None,
        help="Checkpoint file to track processed utterance IDs (enables resume)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=500,
        help="How many lines to process before emitting a progress log",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of worker processes for parallel processing",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of tasks to batch before submitting to executor",
    )
    parser.add_argument(
        "--min-gloss-term-length",
        type=int,
        default=3,
        help="Minimum character length for glossary terms",
    )
    parser.add_argument(
        "--min-gloss-term-token-len",
        type=int,
        default=2,
        help="Minimum token length per glossary term component",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep existing wordlings text if we cannot build references",
    )
    return parser.parse_args()


def infer_utt_id_from_audio_path(audio_path: str) -> Optional[str]:
    """
    Recover utterance id (PREFIX_SEGMENT) from a clip path such as
    .../POD0000010739/42/0.wav.
    """
    parts = Path(audio_path).parts
    if len(parts) < 3:
        return None
    prefix = parts[-3]
    segment = parts[-2]
    if not prefix or not segment:
        return None
    return f"{prefix}_{segment}"


def collect_utt_ids_from_jsonl(json_input: str) -> Set[str]:
    """
    First pass: scan JSONL to collect all unique utterance IDs.
    This allows us to filter TSV loading to only relevant rows.
    """
    utt_ids: Set[str] = set()
    logger.info("Scanning JSONL to collect utterance IDs: %s", json_input)
    with open(json_input, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                audios = record.get("audios") or []
                if audios:
                    utt_id = infer_utt_id_from_audio_path(audios[0])
                    if utt_id:
                        utt_ids.add(utt_id)
            except Exception:
                continue
    logger.info("Found %d unique utterance IDs in JSONL", len(utt_ids))
    return utt_ids


def load_tsv_metadata_filtered(
    tsv_paths: Sequence[str],
    required_utt_ids: Set[str],
) -> Dict[str, Dict]:
    """
    Load TSV metadata but only keep rows whose utt_id is in required_utt_ids.
    This dramatically reduces memory usage and loading time.
    """
    meta: Dict[str, Dict] = {}
    for tsv_path in tsv_paths:
        logger.info("Loading TSV metadata from %s (filtering to %d IDs)", tsv_path, len(required_utt_ids))
        with open(tsv_path, "r", encoding="utf-8") as f:
            header = f.readline()
            if not header:
                logger.warning("TSV %s is empty; skipping", tsv_path)
                continue
            for line in f:
                row = parse_tsv_line(line)
                if not row:
                    continue
                utt_id = row["utt_id"]
                if utt_id in required_utt_ids:
                    meta[utt_id] = row
    logger.info("Loaded %d utterances from %d TSV files (filtered)", len(meta), len(tsv_paths))
    return meta


def compute_chunk_timings_from_trajectory(
    trajectory_size: int,
    chunk_count: int,
) -> List[Tuple[float, float]]:
    """
    Compute chunk timings using the same logic as generate_simul_st_glossary_dataset.py:
    - trajectory_size = len(zh_pieces) = number of 0.96s base chunks
    - chunk_count = len(audios) in the JSONL record
    - chunks_per_merge = trajectory_size // chunk_count (how many 0.96s chunks per audio clip)
    
    Returns list of (start_time, duration) tuples.
    """
    if chunk_count <= 0 or trajectory_size <= 0:
        return []
    
    # Compute how many base chunks are merged into each audio clip
    chunks_per_merge = max(1, trajectory_size // chunk_count)
    
    timings: List[Tuple[float, float]] = []
    for i in range(chunk_count):
        # Each audio clip corresponds to chunks_per_merge base chunks starting at position i * chunks_per_merge
        base_idx = i * chunks_per_merge
        # For the last chunk, include any remaining base chunks
        if i == chunk_count - 1:
            num_base_chunks = trajectory_size - base_idx
        else:
            num_base_chunks = chunks_per_merge
        
        start_time = base_idx * BASE_CHUNK_DURATION
        duration = num_base_chunks * BASE_CHUNK_DURATION
        timings.append((start_time, duration))
    
    return timings


def build_chunk_references(
    utt_id: str,
    chunk_count: int,
    row: Dict,
    glossary_index: Dict[str, List[Dict]],
    textgrid_root: str,
) -> List[List[Dict]]:
    """
    Build per-chunk glossary references using trajectory-based timing (no audio reading).
    """
    en_text = row.get("en_text") or ""
    zh_pieces = row.get("zh_pieces") or []
    trajectory_size = len(zh_pieces)
    
    # Get all glossary terms that match the English text
    tokens = _tokenize(en_text)
    all_refs = find_chunk_glossary_terms(tokens, glossary_index) if tokens else []
    
    if not all_refs:
        return [[] for _ in range(chunk_count)]
    
    # Compute chunk timings from trajectory size
    chunk_timings = compute_chunk_timings_from_trajectory(trajectory_size, chunk_count)
    if not chunk_timings:
        logger.warning(
            "Unable to compute chunk timings for %s (chunk_count=%d, trajectory=%d); skipping wordlings.",
            utt_id,
            chunk_count,
            trajectory_size,
        )
        return [[] for _ in range(chunk_count)]
    
    # Use TextGrid to assign references to chunks
    tg_refs = _assign_references_with_textgrid(
        utt_id=utt_id,
        all_references=all_refs,
        chunk_timings=chunk_timings,
        textgrid_root=textgrid_root,
    )
    if tg_refs is None:
        logger.warning("Missing TextGrid alignment for %s; skipping wordlings.", utt_id)
        return [[] for _ in range(chunk_count)]
    return tg_refs


def rebuild_user_content(original: str, wordlings: Dict[str, str]) -> str:
    """Inject wordlings immediately after the <audio> marker."""
    replacement = json.dumps(wordlings, ensure_ascii=False)
    marker = "wordlings:"
    base_content = original
    if marker in base_content:
        base_content = base_content.split(marker, 1)[0].rstrip()

    insertion = f"\n\nwordlings: {replacement}"
    audio_tag = "<audio>"
    idx = base_content.find(audio_tag)
    if idx == -1:
        return f"{base_content.rstrip()}{insertion}"

    insert_pos = idx + len(audio_tag)
    before = base_content[:insert_pos]
    after = base_content[insert_pos:]
    return f"{before}{insertion}{after}"


def remove_wordlings_from_content(original: str) -> str:
    """Remove existing wordlings section, if any."""
    marker = "wordlings:"
    if marker not in original:
        return original
    prefix = original.split(marker, 1)[0]
    return prefix.rstrip()


def apply_wordlings_to_record(
    record: Dict,
    chunk_refs: List[List[Dict]],
    keep_empty: bool,
    en_text: str,
) -> bool:
    """Apply wordlings to each user message in the record."""
    user_indices = [
        idx for idx, msg in enumerate(record.get("messages", []))
        if msg.get("role") == "user"
    ]
    if len(user_indices) != len(chunk_refs):
        return False

    audios = record.get("audios") or []
    updated = False
    for chunk_idx, msg_idx in enumerate(user_indices):
        refs = chunk_refs[chunk_idx]
        if not refs:
            if not keep_empty:
                content = record["messages"][msg_idx].get("content", "")
                record["messages"][msg_idx]["content"] = remove_wordlings_from_content(content)
            continue
        mapping = OrderedDict()
        for ref in refs:
            term = ref.get("term")
            if not term:
                continue
            mapping[term] = ref.get("translation", "")
        if not mapping:
            continue
        record["messages"][msg_idx]["content"] = rebuild_user_content(
            record["messages"][msg_idx].get("content", ""),
            mapping,
        )
        updated = True
        if mapping:
            audio_path = audios[chunk_idx] if chunk_idx < len(audios) else "N/A"
            logger.info(
                "Wordlings applied | audio=%s | en_text=%s",
                audio_path,
                en_text,
            )
    return updated


def process_single_line(task: Tuple[int, str, str]) -> Dict:
    """
    Worker function that runs in a subprocess.
    Uses global variables initialized by worker_init.
    
    Args:
        task: (line_idx, line_text, utt_id)
    
    Returns:
        Dict with line_idx, utt_id, json_text, status
    """
    line_idx, line_text, utt_id = task
    
    record = json.loads(line_text)
    audios = record.get("audios") or []
    chunk_count = len(audios)
    
    status = "skipped"
    meta_row = _worker_meta_map.get(utt_id) if utt_id and _worker_meta_map else None
    
    if meta_row and chunk_count > 0 and _worker_glossary_index and _worker_textgrid_root:
        chunk_refs = build_chunk_references(
            utt_id, chunk_count, meta_row, _worker_glossary_index, _worker_textgrid_root
        )
        en_text = meta_row.get("en_text", "")
        if chunk_refs and apply_wordlings_to_record(
            record,
            chunk_refs,
            _worker_keep_empty,
            en_text,
        ):
            status = "updated"
    
    return {
        "line_idx": line_idx,
        "utt_id": utt_id,
        "json_text": json.dumps(record, ensure_ascii=False),
        "status": status,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )

    # Step 1: Scan JSONL to collect all utterance IDs
    required_utt_ids = collect_utt_ids_from_jsonl(args.json_input)
    
    # Step 2: Load glossary index
    glossary_index = load_glossary(
        args.glossary,
        args.min_gloss_term_length,
        args.min_gloss_term_token_len,
    )
    
    # Step 3: Load TSV metadata (filtered to only required IDs)
    meta_map = load_tsv_metadata_filtered(args.tsv, required_utt_ids)
    
    # Load checkpoint if exists
    processed_utts: Set[str] = set()
    if args.checkpoint_file and os.path.exists(args.checkpoint_file):
        with open(args.checkpoint_file, "r") as f:
            for line in f:
                processed_utts.add(line.strip())
        logger.info("Loaded %d processed utterances from checkpoint", len(processed_utts))
    
    # Statistics
    total = 0
    updated = 0
    skipped = 0
    
    # Open output and checkpoint files
    out_f = open(args.json_output, "w", encoding="utf-8")
    ckpt_f = open(args.checkpoint_file, "a") if args.checkpoint_file else None
    
    try:
        with open(args.json_input, "r", encoding="utf-8") as src:
            # Collect tasks in batches
            batch_tasks: List[Tuple[int, str, str]] = []
            pending_results: Dict[int, Dict] = {}
            next_to_write = 0
            
            def flush_results():
                nonlocal updated, skipped, next_to_write
                while next_to_write in pending_results:
                    result = pending_results.pop(next_to_write)
                    out_f.write(result["json_text"] + "\n")
                    if result["status"] == "updated":
                        updated += 1
                    else:
                        skipped += 1
                    if ckpt_f and result["utt_id"]:
                        ckpt_f.write(result["utt_id"] + "\n")
                    next_to_write += 1
            
            def process_batch(executor, tasks):
                futures = {
                    executor.submit(process_single_line, task): task[0]
                    for task in tasks
                }
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    pending_results[result["line_idx"]] = result
                flush_results()
            
            # Create executor with initializer
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.num_workers,
                initializer=worker_init,
                initargs=(glossary_index, meta_map, args.textgrid_root, args.keep_empty),
            ) as executor:
                for line_idx, line in enumerate(src):
                    total += 1
                    line_text = line.rstrip("\n")
                    
                    # Parse to get utt_id
                    try:
                        record = json.loads(line_text)
                        audios = record.get("audios") or []
                        utt_id = infer_utt_id_from_audio_path(audios[0]) if audios else None
                    except Exception:
                        utt_id = None
                    
                    # Skip if already processed
                    if utt_id and utt_id in processed_utts:
                        # Still need to write the original line
                        pending_results[line_idx] = {
                            "line_idx": line_idx,
                            "utt_id": utt_id,
                            "json_text": line_text,
                            "status": "skipped",
                        }
                        flush_results()
                        continue
                    
                    # Create lightweight task tuple (no heavy objects)
                    task = (line_idx, line_text, utt_id)
                    batch_tasks.append(task)
                    
                    # Process batch when full
                    if len(batch_tasks) >= args.batch_size:
                        process_batch(executor, batch_tasks)
                        batch_tasks = []
                        
                        if args.log_interval and (updated + skipped) % args.log_interval == 0:
                            logger.info(
                                "Processed %d lines (updated=%d skipped=%d)",
                                updated + skipped,
                                updated,
                                skipped,
                            )
                
                # Process remaining tasks
                if batch_tasks:
                    process_batch(executor, batch_tasks)
        
        logger.info(
            "Finished. Total=%d updated=%d skipped=%d (output=%s)",
            total,
            updated,
            skipped,
            args.json_output,
        )
    
    finally:
        out_f.close()
        if ckpt_f:
            ckpt_f.close()


if __name__ == "__main__":
    main()
