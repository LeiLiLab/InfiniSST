#!/usr/bin/env python3

# python3 /home/jiaxuanluo/InfiniSST/documents/code/simuleval/instances_log_to_tsv.py \
#   --input-root /mnt/gemini/data2/jiaxuanluo/infinisst_omni_vllm_rag_rank32_iter_0000452_hf_zh_topk2_ablation \
#   --include-text 0

"""
Convert SimulEval JSONL instance logs (e.g., instances.log / instance*.log) into TSV tables.

This script is designed for large logs:
- It avoids dumping huge prediction/reference strings by default.
- It computes compact per-instance statistics from delay/elapsed arrays.
"""

from __future__ import annotations

# ======Configuration=====
DEFAULT_LOG_GLOB_PATTERNS = (
    "instances.log",
    "instance*.log",
)

DEFAULT_OUTPUT_INSTANCE_TSV_NAME = "instances_table.tsv"
DEFAULT_OUTPUT_RUN_TSV_NAME = "runs_table.tsv"

DEFAULT_INCLUDE_TEXT = False
DEFAULT_MAX_TEXT_CHARS = 200

DEFAULT_MAX_FILES = 0  # 0 means no limit
DEFAULT_MAX_INSTANCES_PER_FILE = 0  # 0 means no limit

TSV_DIALECT_DELIMITER = "\t"
TSV_LINE_TERMINATOR = "\n"

RUN_DIR_PATTERN = r"_cs(?P<cs>[0-9.]+)_hs(?P<hs>[0-9.]+)_lm(?P<lm>[0-9]+)_rk(?P<rk>[0-9]+)_vk(?P<vk>[0-9]+)"

PERCENTILES = (50, 90)
PERCENTILE_MIN = 0
PERCENTILE_MAX = 100
PERCENTILE_DENOMINATOR = 100.0
MIN_VALID_LIST_LENGTH = 1

FLOAT_DECIMALS = 6
# ======Configuration=====

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class RunMeta:
    run_dir: str
    cs: Optional[str]
    hs: Optional[str]
    lm: Optional[str]
    rk: Optional[str]
    vk: Optional[str]


def _iter_log_files(input_root: Path, glob_patterns: Sequence[str]) -> List[Path]:
    seen: Dict[str, Path] = {}
    for pat in glob_patterns:
        for p in input_root.rglob(pat):
            if not p.is_file():
                continue
            key = str(p.resolve())
            seen[key] = p
    files = list(seen.values())
    files.sort()
    return files


def _parse_run_meta_from_path(log_path: Path) -> RunMeta:
    run_dir = str(log_path.parent)
    m = re.search(RUN_DIR_PATTERN, run_dir)
    if not m:
        return RunMeta(run_dir=run_dir, cs=None, hs=None, lm=None, rk=None, vk=None)
    g = m.groupdict()
    return RunMeta(
        run_dir=run_dir,
        cs=g.get("cs"),
        hs=g.get("hs"),
        lm=g.get("lm"),
        rk=g.get("rk"),
        vk=g.get("vk"),
    )


def _safe_len_text(x: Any) -> Optional[int]:
    if isinstance(x, str):
        return len(x)
    return None


def _to_float_list(x: Any) -> Optional[List[float]]:
    if not isinstance(x, list):
        return None
    out: List[float] = []
    for v in x:
        try:
            out.append(float(v))
        except Exception:
            return None
    return out


def _mean(values: Sequence[float]) -> float:
    return float(sum(values)) / float(len(values))


def _percentile(values_sorted: Sequence[float], p: int) -> float:
    if not values_sorted:
        return float("nan")
    if p <= PERCENTILE_MIN:
        return float(values_sorted[0])
    if p >= PERCENTILE_MAX:
        return float(values_sorted[-1])
    # Linear interpolation between closest ranks.
    idx = (len(values_sorted) - 1) * (p / PERCENTILE_DENOMINATOR)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(values_sorted[lo])
    frac = idx - lo
    return float(values_sorted[lo] * (1.0 - frac) + values_sorted[hi] * frac)


@dataclass
class ListStats:
    count: int
    mean: float
    max: float
    p50: float
    p90: float


def _stats(values: Sequence[float]) -> Optional[ListStats]:
    if len(values) < MIN_VALID_LIST_LENGTH:
        return None
    vs = sorted(values)
    p50 = _percentile(vs, PERCENTILES[0])
    p90 = _percentile(vs, PERCENTILES[1])
    return ListStats(
        count=len(values),
        mean=_mean(values),
        max=float(vs[-1]),
        p50=p50,
        p90=p90,
    )


def _truncate_text(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars]


def _iter_json_lines(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _format_float(x: Optional[float]) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return ""
    return f"{x:.{FLOAT_DECIMALS}f}"


def _write_tsv(path: Path, header: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(header),
            delimiter=TSV_DIALECT_DELIMITER,
            lineterminator=TSV_LINE_TERMINATOR,
            extrasaction="ignore",
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _instance_rows_for_file(
    log_path: Path,
    include_text: bool,
    max_text_chars: int,
    max_instances_per_file: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    meta = _parse_run_meta_from_path(log_path)

    instances: List[Dict[str, Any]] = []
    run_agg: Dict[str, Any] = {
        "run_dir": meta.run_dir,
        "log_path": str(log_path),
        "cs": meta.cs or "",
        "hs": meta.hs or "",
        "lm": meta.lm or "",
        "rk": meta.rk or "",
        "vk": meta.vk or "",
        "num_instances": 0,
        "delay_mean_avg": "",
        "delay_p50_avg": "",
        "delay_p90_avg": "",
        "elapsed_mean_avg": "",
        "elapsed_p50_avg": "",
        "elapsed_p90_avg": "",
    }

    delay_mean_list: List[float] = []
    delay_p50_list: List[float] = []
    delay_p90_list: List[float] = []
    elapsed_mean_list: List[float] = []
    elapsed_p50_list: List[float] = []
    elapsed_p90_list: List[float] = []

    seen = 0
    for obj in _iter_json_lines(log_path):
        seen += 1
        if max_instances_per_file > 0 and seen > max_instances_per_file:
            break

        delays = _to_float_list(obj.get("delays"))
        elapsed = _to_float_list(obj.get("elapsed"))
        delay_stats = _stats(delays) if delays is not None else None
        elapsed_stats = _stats(elapsed) if elapsed is not None else None

        prediction = obj.get("prediction")
        reference = obj.get("reference")

        row: Dict[str, Any] = {
            "run_dir": meta.run_dir,
            "log_path": str(log_path),
            "cs": meta.cs or "",
            "hs": meta.hs or "",
            "lm": meta.lm or "",
            "rk": meta.rk or "",
            "vk": meta.vk or "",
            "index": obj.get("index", ""),
            "source_length": obj.get("source_length", ""),
            "prediction_length": obj.get("prediction_length", ""),
            "reference_length": _safe_len_text(reference) or "",
            "delay_count": delay_stats.count if delay_stats else "",
            "delay_mean": _format_float(delay_stats.mean) if delay_stats else "",
            "delay_p50": _format_float(delay_stats.p50) if delay_stats else "",
            "delay_p90": _format_float(delay_stats.p90) if delay_stats else "",
            "delay_max": _format_float(delay_stats.max) if delay_stats else "",
            "elapsed_count": elapsed_stats.count if elapsed_stats else "",
            "elapsed_mean": _format_float(elapsed_stats.mean) if elapsed_stats else "",
            "elapsed_p50": _format_float(elapsed_stats.p50) if elapsed_stats else "",
            "elapsed_p90": _format_float(elapsed_stats.p90) if elapsed_stats else "",
            "elapsed_max": _format_float(elapsed_stats.max) if elapsed_stats else "",
        }

        if include_text:
            if isinstance(prediction, str):
                row["prediction"] = _truncate_text(prediction, max_text_chars)
            else:
                row["prediction"] = ""
            if isinstance(reference, str):
                row["reference"] = _truncate_text(reference, max_text_chars)
            else:
                row["reference"] = ""

        if delay_stats:
            delay_mean_list.append(delay_stats.mean)
            delay_p50_list.append(delay_stats.p50)
            delay_p90_list.append(delay_stats.p90)
        if elapsed_stats:
            elapsed_mean_list.append(elapsed_stats.mean)
            elapsed_p50_list.append(elapsed_stats.p50)
            elapsed_p90_list.append(elapsed_stats.p90)

        instances.append(row)

    run_agg["num_instances"] = len(instances)
    if delay_mean_list:
        run_agg["delay_mean_avg"] = _format_float(_mean(delay_mean_list))
        run_agg["delay_p50_avg"] = _format_float(_mean(delay_p50_list))
        run_agg["delay_p90_avg"] = _format_float(_mean(delay_p90_list))
    if elapsed_mean_list:
        run_agg["elapsed_mean_avg"] = _format_float(_mean(elapsed_mean_list))
        run_agg["elapsed_p50_avg"] = _format_float(_mean(elapsed_p50_list))
        run_agg["elapsed_p90_avg"] = _format_float(_mean(elapsed_p90_list))

    return instances, run_agg


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert SimulEval instances.log JSONL to TSV.")
    parser.add_argument("--input-root", required=True, help="Root directory containing SimulEval outputs.")
    parser.add_argument(
        "--glob",
        default=",".join(DEFAULT_LOG_GLOB_PATTERNS),
        help="Comma-separated glob patterns to find logs (default: instances.log,instance*.log).",
    )
    parser.add_argument(
        "--output-instance-tsv",
        default="",
        help="Output TSV for per-instance rows. Default: <input-root>/instances_table.tsv",
    )
    parser.add_argument(
        "--output-run-tsv",
        default="",
        help="Output TSV for per-run aggregated rows. Default: <input-root>/runs_table.tsv",
    )
    parser.add_argument(
        "--include-text",
        type=int,
        default=1 if DEFAULT_INCLUDE_TEXT else 0,
        help="Include truncated prediction/reference text columns (0 or 1).",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=DEFAULT_MAX_TEXT_CHARS,
        help="Max characters to keep for prediction/reference if --include-text=1.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=DEFAULT_MAX_FILES,
        help="Max number of log files to process (0 means no limit).",
    )
    parser.add_argument(
        "--max-instances-per-file",
        type=int,
        default=DEFAULT_MAX_INSTANCES_PER_FILE,
        help="Max number of instances to read per file (0 means no limit).",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    if not input_root.exists():
        raise SystemExit(f"Input root does not exist: {input_root}")

    patterns = [x.strip() for x in str(args.glob).split(",") if x.strip()]
    if not patterns:
        raise SystemExit("No glob patterns provided.")

    output_instance = Path(args.output_instance_tsv) if args.output_instance_tsv else input_root / DEFAULT_OUTPUT_INSTANCE_TSV_NAME
    output_run = Path(args.output_run_tsv) if args.output_run_tsv else input_root / DEFAULT_OUTPUT_RUN_TSV_NAME

    include_text = bool(int(args.include_text))

    log_files = _iter_log_files(input_root, patterns)
    if args.max_files > 0:
        log_files = log_files[: args.max_files]

    if not log_files:
        raise SystemExit(f"No instance logs found under {input_root} with patterns: {patterns}")

    instance_rows: List[Dict[str, Any]] = []
    run_rows: List[Dict[str, Any]] = []

    for p in log_files:
        rows, run_row = _instance_rows_for_file(
            p,
            include_text=include_text,
            max_text_chars=int(args.max_text_chars),
            max_instances_per_file=int(args.max_instances_per_file),
        )
        instance_rows.extend(rows)
        run_rows.append(run_row)

    instance_header = [
        "run_dir",
        "log_path",
        "cs",
        "hs",
        "lm",
        "rk",
        "vk",
        "index",
        "source_length",
        "prediction_length",
        "reference_length",
        "delay_count",
        "delay_mean",
        "delay_p50",
        "delay_p90",
        "delay_max",
        "elapsed_count",
        "elapsed_mean",
        "elapsed_p50",
        "elapsed_p90",
        "elapsed_max",
    ]
    if include_text:
        instance_header += ["prediction", "reference"]

    run_header = [
        "run_dir",
        "log_path",
        "cs",
        "hs",
        "lm",
        "rk",
        "vk",
        "num_instances",
        "delay_mean_avg",
        "delay_p50_avg",
        "delay_p90_avg",
        "elapsed_mean_avg",
        "elapsed_p50_avg",
        "elapsed_p90_avg",
    ]

    _write_tsv(output_instance, instance_header, instance_rows)
    _write_tsv(output_run, run_header, run_rows)

    print(f"Wrote per-instance TSV: {output_instance}")
    print(f"Wrote per-run TSV: {output_run}")
    print(f"Processed log files: {len(log_files)}")
    print(f"Processed instances: {len(instance_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


