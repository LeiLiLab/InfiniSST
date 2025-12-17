#!/usr/bin/env python3
"""
Reshape or regroup term-level datasets.

Supported modes:
- Convert a single mega conversation into per-segment instances (default).
- Regroup existing segment instances into speech-level instances (grouping
  all segments that share the same recording id, e.g., POD0000000003).
- Remove legacy `instance_id` keys when only cleanup is needed.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import List


def derive_prefix(audio_path: str) -> str:
    base = os.path.basename(audio_path)
    marker = "_term_"
    if marker in base:
        return base.split(marker, 1)[0]
    return os.path.splitext(base)[0]


def derive_speech_id(prefix: str) -> str:
    if "_S" in prefix:
        return prefix.split("_S", 1)[0]
    return prefix


def reshape_dataset(dataset: dict) -> List[dict]:
    messages = dataset.get("messages")
    audios: List[str] = dataset.get("audios", [])
    if not isinstance(messages, list) or not isinstance(audios, list):
        raise ValueError("Dataset must contain 'messages' and 'audios' lists.")
    if not messages:
        return []
    system_msg = messages[0]
    if not isinstance(system_msg, dict) or system_msg.get("role") != "system":
        raise ValueError("First message must be a system prompt.")
    turns = messages[1:]
    if len(turns) != 2 * len(audios):
        raise ValueError(
            f"Mismatch between messages ({len(turns)}) and audios ({len(audios)}). "
            "Expected 2 messages per audio (user+assistant)."
        )

    grouped: "OrderedDict[str, Dict[str, object]]" = OrderedDict()
    for idx, audio in enumerate(audios):
        prefix = derive_prefix(audio)
        entry = grouped.get(prefix)
        if entry is None:
            entry = {"messages": [system_msg.copy()], "audios": []}
            grouped[prefix] = entry
        user_msg = turns[2 * idx]
        assistant_msg = turns[2 * idx + 1]
        entry["messages"].append(user_msg)
        entry["messages"].append(assistant_msg)
        entry["audios"].append(audio)

    return list(grouped.values())


def group_by_speech(instances: List[dict]) -> List[dict]:
    grouped: "OrderedDict[str, dict]" = OrderedDict()
    for entry in instances:
        messages = entry.get("messages", [])
        audios = entry.get("audios", [])
        if not messages or not isinstance(messages[0], dict):
            raise ValueError("Each instance must contain messages starting with a system prompt.")
        speech_key = None
        if audios:
            speech_key = derive_speech_id(derive_prefix(audios[0]))
        if not speech_key:
            speech_key = "__SPEECH_UNKNOWN__"
        agg = grouped.get(speech_key)
        if agg is None:
            agg = {
                "messages": [copy.deepcopy(messages[0])],
                "audios": [],
            }
            grouped[speech_key] = agg
        agg["messages"].extend(copy.deepcopy(messages[1:]))
        agg["audios"].extend(audios)
    return list(grouped.values())


def load_instances(input_path: Path) -> List[dict]:
    with input_path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            instances = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                instances.append(json.loads(line))
            return instances

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return reshape_dataset(data)
    raise ValueError("Unsupported dataset structure; expected list or mega conversation.")


def write_instances(instances: List[dict], target: Path, output_format: str) -> None:
    with target.open("w", encoding="utf-8") as f:
        if output_format == "jsonl":
            for entry in instances:
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")
        else:
            json.dump(instances, f, ensure_ascii=False)


def convert_file(
    input_path: Path,
    output_path: Path | None,
    drop_instance_id_only: bool,
    group_level: str,
    output_format: str,
) -> None:
    instances = load_instances(input_path)

    if drop_instance_id_only:
        for entry in instances:
            if isinstance(entry, dict):
                entry.pop("instance_id", None)

    if group_level == "speech":
        instances = group_by_speech(instances)

    target = output_path or input_path
    write_instances(instances, target, output_format)

    level_desc = "speech-level" if group_level == "speech" else "segment-level"
    segments = sum(len(item.get("audios", [])) for item in instances)
    print(
        f"Wrote {len(instances)} {level_desc} instances ({segments} segments) to {target}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reshape term datasets into per-prefix instances.")
    parser.add_argument("inputs", nargs="+", type=Path, help="JSON/JSONL files to convert.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory.")
    parser.add_argument(
        "--drop-instance-id-only",
        action="store_true",
        help="If input is already a list, only remove instance_id keys instead of skipping.",
    )
    parser.add_argument(
        "--group-level",
        choices=["segment", "speech"],
        default="segment",
        help="Granularity of the output instances.",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "jsonl"],
        default="json",
        help="Serialization format for the written instances.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for input_path in args.inputs:
        if not input_path.exists():
            raise FileNotFoundError(f"{input_path} not found.")
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = args.output_dir / input_path.name
        else:
            output_path = None
        convert_file(
            input_path,
            output_path,
            drop_instance_id_only=args.drop_instance_id_only,
            group_level=args.group_level,
            output_format=args.output_format,
        )


if __name__ == "__main__":
    main()

