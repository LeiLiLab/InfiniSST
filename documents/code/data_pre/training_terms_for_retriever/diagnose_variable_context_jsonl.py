#!/usr/bin/env python3
"""Diagnose a variable-duration retriever JSONL.

Checks:
  - duration bucket row counts and fractions
  - domain distribution (GigaSpeech vs wiki_synth)
  - MFA span validity inside each chunk duration
  - sampled audio file duration/frame count
  - optional stats JSON row-count consistency
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import soundfile as sf


WIKI_SYNTH_PREFIX = "wiki_synth_"
SAMPLE_RATE = 16000


def duration_tag(sec: float) -> str:
    return f"{sec:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def parse_duration_secs(value: str) -> List[float]:
    durations = [float(v) for v in value.replace(",", " ").split() if v.strip()]
    if not durations:
        raise ValueError("--expected-duration-secs must not be empty")
    out = []
    for dur in durations:
        r = round(dur, 4)
        if r not in out:
            out.append(r)
    return out


def stable_u64(key: str) -> int:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def parse_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def nearest_duration_tag(value: float | None, durations: Iterable[float], eps: float = 0.015) -> str:
    if value is None:
        return "missing"
    best = min(durations, key=lambda d: abs(d - value))
    if abs(best - value) <= eps:
        return duration_tag(best)
    return f"unknown:{value:.4f}"


def should_keep_audio_sample(path: str, tag: str, limit: int, current_count: int) -> bool:
    if current_count < limit:
        return True
    # Deterministic sparse fallback if early rows are pathologically ordered.
    return stable_u64(f"{tag}\t{path}") % 100_000 == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--stats-json", default="")
    parser.add_argument(
        "--expected-duration-secs",
        default="0.96 1.92 2.88 3.84",
    )
    parser.add_argument("--target-frac", type=float, default=0.25)
    parser.add_argument(
        "--frac-tolerance",
        type=float,
        default=0.02,
        help="Allowed absolute deviation from target fraction before failing.",
    )
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--audio-check-per-duration", type=int, default=50)
    parser.add_argument("--max-audio-checks", type=int, default=500)
    parser.add_argument("--report-json", default="")
    parser.add_argument("--max-lines", type=int, default=0)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    durations = parse_duration_secs(args.expected_duration_secs)
    expected_tags = [duration_tag(d) for d in durations]

    counts = Counter()
    domain_by_duration: Dict[str, Counter] = defaultdict(Counter)
    context_build = Counter()
    mfa_missing = Counter()
    mfa_invalid = Counter()
    audio_missing_path = Counter()
    audio_samples: Dict[str, Dict[str, float]] = {tag: {} for tag in expected_tags}
    unique_terms_by_duration: Dict[str, set] = defaultdict(set)
    total_rows = 0
    json_errors = 0

    with open(args.input, "r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            if args.max_lines and line_no > args.max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                json_errors += 1
                continue

            total_rows += 1
            duration = parse_float(row.get("context_duration_sec"))
            if duration is None:
                duration = parse_float(row.get("chunk_duration_sec"))
            tag = nearest_duration_tag(duration, durations)
            counts[tag] += 1

            utter_id = str(row.get("utter_id") or "")
            domain = "wiki_synth" if utter_id.startswith(WIKI_SYNTH_PREFIX) else "gigaspeech"
            domain_by_duration[tag][domain] += 1
            context_build[str(row.get("context_build") or "missing")] += 1

            term = str(row.get("term_key") or row.get("term") or "").strip()
            if term:
                unique_terms_by_duration[tag].add(term.casefold())

            wav_path = str(row.get("chunk_audio_path") or "")
            if not wav_path:
                audio_missing_path[tag] += 1
            elif tag in audio_samples and len(audio_samples[tag]) < args.audio_check_per_duration:
                audio_samples[tag].setdefault(wav_path, duration or 0.0)
            elif tag in audio_samples and len(audio_samples[tag]) < args.max_audio_checks:
                if should_keep_audio_sample(wav_path, tag, args.audio_check_per_duration, len(audio_samples[tag])):
                    audio_samples[tag].setdefault(wav_path, duration or 0.0)

            mfa_start = parse_float(row.get("mfa_term_start_in_chunk"))
            mfa_end = parse_float(row.get("mfa_term_end_in_chunk"))
            if term:
                if mfa_start is None or mfa_end is None:
                    mfa_missing[tag] += 1
                elif duration is None or mfa_start < -1e-4 or mfa_end > duration + 1e-4 or mfa_end <= mfa_start:
                    mfa_invalid[tag] += 1

    audio_check_failures: List[Dict[str, Any]] = []
    audio_checked = 0
    for tag, paths in audio_samples.items():
        expected_sec = next(d for d in durations if duration_tag(d) == tag)
        expected_frames = int(round(expected_sec * args.sample_rate))
        for path, _ in paths.items():
            audio_checked += 1
            if not os.path.isfile(path):
                audio_check_failures.append(
                    {"duration": tag, "path": path, "error": "missing_file"}
                )
                continue
            try:
                info = sf.info(path)
            except Exception as exc:
                audio_check_failures.append(
                    {"duration": tag, "path": path, "error": str(exc)}
                )
                continue
            if info.samplerate != args.sample_rate or info.frames != expected_frames:
                audio_check_failures.append(
                    {
                        "duration": tag,
                        "path": path,
                        "samplerate": info.samplerate,
                        "frames": info.frames,
                        "expected_frames": expected_frames,
                    }
                )

    stats_payload = None
    stats_total = None
    if args.stats_json:
        with open(args.stats_json, "r", encoding="utf-8") as fin:
            stats_payload = json.load(fin)
        stats_total = stats_payload.get("written_total_rows")

    duration_summary = {}
    frac_failures = []
    for tag in expected_tags:
        count = counts[tag]
        frac = count / total_rows if total_rows else 0.0
        duration_summary[tag] = {
            "rows": count,
            "fraction": frac,
            "domains": dict(domain_by_duration[tag]),
            "unique_terms": len(unique_terms_by_duration[tag]),
            "mfa_missing": mfa_missing[tag],
            "mfa_invalid": mfa_invalid[tag],
            "audio_missing_path": audio_missing_path[tag],
            "audio_checked": len(audio_samples[tag]),
        }
        if abs(frac - args.target_frac) > args.frac_tolerance:
            frac_failures.append(
                {
                    "duration": tag,
                    "fraction": frac,
                    "target": args.target_frac,
                    "tolerance": args.frac_tolerance,
                }
            )

    report = {
        "input": args.input,
        "stats_json": args.stats_json,
        "total_rows": total_rows,
        "json_errors": json_errors,
        "expected_duration_tags": expected_tags,
        "duration_summary": duration_summary,
        "unknown_duration_counts": {
            k: v for k, v in counts.items() if k not in expected_tags
        },
        "context_build_counts": dict(context_build),
        "stats_written_total_rows": stats_total,
        "stats_total_matches": stats_total is None or int(stats_total) == total_rows,
        "audio_checked": audio_checked,
        "audio_check_failures": audio_check_failures[:50],
        "audio_check_failure_count": len(audio_check_failures),
        "fraction_failures": frac_failures,
    }

    if args.report_json:
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as fout:
            json.dump(report, fout, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"[DIAG] input={args.input}")
    print(f"[DIAG] total_rows={total_rows:,} json_errors={json_errors}")
    if stats_total is not None:
        print(f"[DIAG] stats_written_total_rows={stats_total} matches={report['stats_total_matches']}")
    for tag in expected_tags:
        item = duration_summary[tag]
        print(
            "[DIAG] "
            f"dur={tag} rows={item['rows']:,} frac={item['fraction']:.4f} "
            f"domains={item['domains']} unique_terms={item['unique_terms']:,} "
            f"mfa_missing={item['mfa_missing']:,} mfa_invalid={item['mfa_invalid']:,} "
            f"audio_checked={item['audio_checked']:,}"
        )
    if report["unknown_duration_counts"]:
        print(f"[DIAG][WARN] unknown_duration_counts={report['unknown_duration_counts']}")
    if audio_check_failures:
        print(f"[DIAG][WARN] audio_check_failure_count={len(audio_check_failures)}")
    if frac_failures:
        print(f"[DIAG][WARN] fraction_failures={frac_failures}")
    if args.report_json:
        print(f"[DIAG] report_json={args.report_json}")

    hard_fail = (
        json_errors
        or report["unknown_duration_counts"]
        or not report["stats_total_matches"]
        or audio_check_failures
        or any(mfa_invalid[tag] for tag in expected_tags)
        or frac_failures
    )
    if hard_fail and not args.no_fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
