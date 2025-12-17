#!/usr/bin/env python3
"""
Rebuild audio clip paths for siqi_train.json by cutting fresh chunks from
GigaSpeech sources and writing them under a local root.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import soundfile as sf


logger = logging.getLogger("rebuild_siqi_audios")


@dataclass(frozen=True)
class AudioMeta:
    utt_id: str
    audio_path: str
    start_frame: int
    num_frames: int
    trajectory_len: int


def parse_audio_spec(audio_spec: str) -> Tuple[str, Optional[int], Optional[int]]:
    """
    Parse strings like "<path>:<start>:<frames>" into components.
    """
    if not audio_spec:
        return "", None, None

    parts = audio_spec.split(":")
    path = parts[0]

    def _parse_int(idx: int) -> Optional[int]:
        if len(parts) <= idx or not parts[idx]:
            return None
        try:
            return int(parts[idx])
        except ValueError:
            return None

    return path, _parse_int(1), _parse_int(2)


def safe_literal_eval(value: str) -> List[str]:
    if not value:
        return []
    cleaned = value.replace("\x00", "")
    try:
        parsed = ast.literal_eval(cleaned)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return []


def load_tsv_metadata(tsv_path: str, audio_root: Optional[str]) -> Dict[str, AudioMeta]:
    """
    Build a map {utt_id -> AudioMeta}.
    """
    meta: Dict[str, AudioMeta] = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline()
        if not header:
            raise RuntimeError("TSV file is empty")

        for line_num, line in enumerate(f, start=2):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                logger.warning("Line %d has %d columns (<9); skipped", line_num, len(parts))
                continue

            utt_id = parts[0]
            audio_spec = parts[1]
            n_frames_str = parts[2]
            trajectory_str = parts[8]

            audio_path, start_frame, num_frames_spec = parse_audio_spec(audio_spec)
            if not audio_path:
                logger.warning("Line %d has empty audio path; skipped", line_num)
                continue

            if not os.path.isabs(audio_path) and audio_root:
                audio_path = os.path.join(audio_root, audio_path)

            if not os.path.exists(audio_path):
                logger.warning("Audio file missing for %s: %s", utt_id, audio_path)
                continue

            try:
                n_frames = int(n_frames_str)
            except ValueError:
                n_frames = None

            if start_frame is None:
                start_frame = 0

            num_frames = num_frames_spec if num_frames_spec is not None else n_frames
            if num_frames is None:
                logger.warning("No frame count for %s; skipped", utt_id)
                continue

            traj_len = len(safe_literal_eval(trajectory_str))

            meta[utt_id] = AudioMeta(
                utt_id=utt_id,
                audio_path=audio_path,
                start_frame=start_frame,
                num_frames=num_frames,
                trajectory_len=traj_len,
            )

    if not meta:
        raise RuntimeError("No usable rows loaded from TSV")
    logger.info("Loaded %d utterances from %s", len(meta), tsv_path)
    return meta


def infer_utt_id_from_path(audio_path: str) -> Optional[str]:
    """
    Recover utt id (PREFIX_INDEX) from an existing clip path.
    """
    if not audio_path:
        return None
    parts = Path(audio_path).parts
    if len(parts) < 3:
        return None
    prefix = parts[-3]
    segment = parts[-2]
    return f"{prefix}_{segment}"


def chunk_boundaries(total_frames: int, chunk_count: int) -> List[Tuple[int, int]]:
    if chunk_count <= 0:
        raise ValueError("chunk_count must be positive")
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")

    bounds: List[int] = []
    for i in range(chunk_count + 1):
        if i == chunk_count:
            bounds.append(total_frames)
        else:
            bounds.append(math.floor(i * total_frames / chunk_count))

    segments: List[Tuple[int, int]] = []
    for i in range(chunk_count):
        start = bounds[i]
        end = bounds[i + 1]
        if end <= start:
            end = start + 1
        segments.append((start, end))
    return segments


def write_clips(meta: AudioMeta, chunk_count: int, clips_root: str, force: bool) -> List[str]:
    prefix, suffix = meta.utt_id.rsplit("_", 1)
    chunk_dir = os.path.join(clips_root, prefix, suffix)
    os.makedirs(chunk_dir, exist_ok=True)

    expected_paths = [os.path.join(chunk_dir, f"{idx}.wav") for idx in range(chunk_count)]
    if not force:
        existing = all(os.path.exists(p) and os.path.getsize(p) > 0 for p in expected_paths)
        if existing:
            return expected_paths

    with sf.SoundFile(meta.audio_path) as src:
        sr = src.samplerate
        available = len(src) - meta.start_frame
        total_frames = min(meta.num_frames, available)
        if total_frames <= 0:
            raise RuntimeError(f"No frames to read for {meta.utt_id}")

        segments = chunk_boundaries(total_frames, chunk_count)
        for idx, (seg_start, seg_end) in enumerate(segments):
            frames = max(1, seg_end - seg_start)
            absolute_start = meta.start_frame + seg_start
            src.seek(absolute_start)
            data = src.read(frames, dtype="float32", always_2d=False)
            if data.size == 0:
                raise RuntimeError(f"Empty chunk for {meta.utt_id} chunk={idx}")

            if data.ndim > 1:
                data = data.mean(axis=1)

            out_path = expected_paths[idx]
            sf.write(out_path, data, sr, subtype="PCM_16")
    return expected_paths


def process_json_line(
    line: str,
    meta_map: Dict[str, AudioMeta],
    clips_root: str,
    force: bool,
) -> str:
    record = json.loads(line)
    audios: List[str] = record.get("audios") or []
    if not audios:
        return json.dumps(record, ensure_ascii=False)

    utt_id = infer_utt_id_from_path(audios[0])
    if not utt_id:
        raise ValueError(f"Cannot infer utt_id from path: {audios[0]}")

    meta = meta_map.get(utt_id)
    if not meta:
        raise KeyError(f"No TSV metadata for utt_id={utt_id}")

    chunk_paths = write_clips(meta, len(audios), clips_root, force)
    record["audios"] = chunk_paths
    return json.dumps(record, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild siqi_train.json audio clip paths")
    parser.add_argument("--json-input", required=True, help="Path to original JSONL file")
    parser.add_argument("--json-output", required=True, help="Path to write updated JSONL")
    parser.add_argument("--tsv", required=True, help="Path to GigaSpeech TSV file")
    parser.add_argument(
        "--audio-root",
        default="/mnt/taurus/data/siqiouyang/datasets/gigaspeech",
        help="Base directory for relative audio paths",
    )
    parser.add_argument(
        "--clips-root",
        default="/mnt/gemini/data1/jiaxuanluo/audio_clips_siqi",
        help="Output directory for generated clips",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="Thread pool size")
    parser.add_argument(
        "--max-pending",
        type=int,
        default=32,
        help="Maximum number of in-flight futures to buffer before flushing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild clips even if target files already exist",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, ...)",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )

    os.makedirs(args.clips_root, exist_ok=True)
    meta_map = load_tsv_metadata(args.tsv, args.audio_root)

    total_lines = 0
    updated = 0

    def submit_task(executor, futures, line_idx, line_text):
        future = executor.submit(
            process_json_line,
            line_text,
            meta_map,
            args.clips_root,
            args.force,
        )
        futures[future] = line_idx

    with open(args.json_input, "r", encoding="utf-8") as in_f, open(
        args.json_output, "w", encoding="utf-8"
    ) as out_f, concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures: Dict[concurrent.futures.Future, int] = {}
        pending_results: Dict[int, str] = {}
        next_to_write = 0

        def flush_ready():
            nonlocal next_to_write, updated
            while next_to_write in pending_results:
                json_text = pending_results.pop(next_to_write)
                out_f.write(json_text + "\n")
                updated += 1
                next_to_write += 1

        for line_idx, line in enumerate(in_f):
            total_lines += 1
            submit_task(executor, futures, line_idx, line.rstrip("\n"))

            if len(futures) >= args.max_pending:
                done, _ = concurrent.futures.wait(
                    list(futures.keys()),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for fut in done:
                    idx = futures.pop(fut)
                    pending_results[idx] = fut.result()
                flush_ready()

        # Drain remaining futures
        if futures:
            done, _ = concurrent.futures.wait(list(futures.keys()))
            for fut in done:
                idx = futures.pop(fut)
                pending_results[idx] = fut.result()
            flush_ready()

    logger.info(
        "Finished. total=%d updated=%d (wrote %s)",
        total_lines,
        updated,
        args.json_output,
    )


if __name__ == "__main__":
    main()

