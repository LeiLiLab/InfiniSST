#!/usr/bin/env python3
"""Merge variable-context JSONL shards and their stats files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List


METADATA_KEYS = {
    "input",
    "output",
    "audio_output_dir",
    "wiki_audio_output_dir",
    "old_chunk_sec",
    "duration_secs",
    "duration_tags",
    "stride_sec",
    "include_mode",
    "duration_assignment",
    "num_shards",
    "shard_id",
    "dry_run",
    "write_empty_groups",
    "reuse_old_audio_for_1p92",
    "stats_json",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats-json", required=True)
    parser.add_argument("--audio-output-dir", default="")
    parser.add_argument("--wiki-audio-output-dir", default="")
    parser.add_argument("--duration-secs", default="2.88 3.84 4.80 5.76")
    parser.add_argument("--duration-assignment", default="balance_rows")
    args = parser.parse_args()

    shard_dir = Path(args.shard_dir)
    output = Path(args.output)
    stats_json = Path(args.stats_json)
    duration_secs = [float(x) for x in args.duration_secs.replace(",", " ").split()]
    duration_tags = [
        f"{x:.2f}".rstrip("0").rstrip(".").replace(".", "p")
        for x in duration_secs
    ]

    tmp_output = output.with_suffix(output.suffix + ".tmp")
    tmp_output.parent.mkdir(parents=True, exist_ok=True)

    merged: Dict[str, Any] = {}
    shard_stats: List[Dict[str, Any]] = []
    with open(tmp_output, "w", encoding="utf-8") as fout:
        for sid in range(args.num_shards):
            tag = f"{sid:02d}"
            part = shard_dir / f"part_{tag}.jsonl"
            stats_path = shard_dir / f"part_{tag}_stats.json"
            if not part.is_file():
                raise FileNotFoundError(part)
            if not stats_path.is_file():
                raise FileNotFoundError(stats_path)
            with open(part, "r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
            with open(stats_path, "r", encoding="utf-8") as fin:
                stats = json.load(fin)
            shard_stats.append({"shard_id": sid, "stats_path": str(stats_path)})
            for key, value in stats.items():
                if key in METADATA_KEYS or isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    merged[key] = merged.get(key, 0) + value

    os.replace(tmp_output, output)
    merged.update(
        {
            "input": "sharded:" + str(shard_dir),
            "output": str(output),
            "audio_output_dir": args.audio_output_dir,
            "wiki_audio_output_dir": args.wiki_audio_output_dir,
            "duration_secs": duration_secs,
            "duration_tags": duration_tags,
            "duration_assignment": args.duration_assignment,
            "num_shards": args.num_shards,
            "shard_stats": shard_stats,
        }
    )
    stats_json.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_json, "w", encoding="utf-8") as fout:
        json.dump(merged, fout, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"[MERGE] output={output}")
    print(f"[MERGE] stats={stats_json}")
    print(f"[MERGE] written_total_rows={merged.get('written_total_rows')}")
    for tag in duration_tags:
        print(f"[MERGE] duration_row_count_{tag}={merged.get('duration_row_count_' + tag)}")


if __name__ == "__main__":
    main()
