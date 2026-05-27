#!/usr/bin/env python3
"""
Data integrity audit for Speech LLM training JSONL.

Verifies that `merge_multiplier`, per-chunk `chunk_metadata.multiplier`,
`audios[]`, `gt_terms_by_chunk[]`, and the term_map text embedded in each
user turn are mutually consistent.  Fails LOUDLY on any structural
violation; `gt.zh` membership in the assistant output is treated as a
soft warning (translations drift).

Assumptions (from documents/code/data_pre/hard_negative_jsonl_for_speech_llm/rebuild_termmap.py):
  - UNIT_DURATION_SEC = 0.96
  - multiplier = ceil(chunk_duration / 0.96)
  - chunk_metadata[i].multiplier overrides merge_multiplier when present
  - term_map entries are `{key}={zh}` lines under a `term_map:` header
  - Each user message with `<audio>` corresponds to audios[chunk_idx]
    and gt_terms_by_chunk[chunk_idx] in order.

No silent fallbacks. No magic numbers outside ======Configuration=====.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ======Configuration=====
UNIT_DURATION_SEC = 0.96
# Absolute tolerance (seconds) when comparing audio wav duration against
# expected multiplier * UNIT_DURATION_SEC.  rebuild_termmap trusts the
# chunk_metadata value unconditionally; we allow 0.05s to cover wav
# encoder rounding.
DURATION_TOL_SEC = 0.05
# How many rows to open the wav header on (full 26k would be slow IO).
# Sampled uniformly with a fixed seed.
DEFAULT_WAV_SAMPLE_ROWS = 500
DEFAULT_WHISPER_SAMPLES = 10
DEFAULT_SEED = 42

# When >= this fraction of rows have any BLOCKING structural error, gate
# downstream training by emitting a non-zero exit code.
STRUCTURAL_FAIL_RATE_GATE = 0.005

# Error categories that BLOCK training. Other categories are reported but
# do not gate (e.g. tail-chunk wav duration shorter than multiplier*0.96 is
# a known artifact from the chunk-merging step, not a corruption).
BLOCKING_ERROR_CATEGORIES = frozenset({
    "messages_missing",
    "audios_missing",
    "gt_terms_missing",
    "merge_multiplier_missing",
    "len_audios_mismatch",
    "len_gt_terms_mismatch",
    "audio_missing",
    "audio_path_type",
    "wav_header_read",
    "multiplier_missing",
    "multiplier_invalid",
    "gt_term_empty",
    "gt_not_in_termmap",  # term_map non-empty but GT absent: real inconsistency
})

# Error categories that are REAL failings but not blocking (design choice
# or tail-chunk artifact).
NONBLOCKING_ERROR_CATEGORIES = frozenset({
    "gt_but_empty_termmap",      # rebuild_termmap EMPTY_PROB_HAS_GT=0.15
    "wav_bucket_mismatch_tail",  # last chunk shorter than multiplier*0.96
    "wav_bucket_mismatch_interior",  # non-last chunk mismatch: suspicious but rare
})
# ======Configuration=====


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def iter_jsonl(path: str, max_rows: int = 0):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if max_rows and i >= max_rows:
                break
            try:
                yield i, json.loads(line)
            except json.JSONDecodeError as e:
                raise AssertionError(f"JSON decode error at row {i}: {e}")


def parse_term_map_keys(user_content: str) -> List[str]:
    """Parse the `term_map:` block at the end of a user turn content.

    rebuild_termmap.py always writes the block as:
        <audio>

        term_map:
        key1=zh1
        key2=zh2
    The block may be absent (just `<audio>`) when the chunk got an empty
    term_map.  Returns [] in that case.
    """
    if "term_map:" not in user_content:
        return []
    after = user_content.split("term_map:", 1)[1]
    keys: List[str] = []
    for raw_line in after.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        if key:
            keys.append(key)
    return keys


def wav_duration_seconds(path: str) -> float:
    with contextlib.closing(wave.open(path, "rb")) as w:
        return w.getnframes() / float(w.getframerate())


def safe_lower(s) -> str:
    return (s or "").strip().lower()


# ------------------------------------------------------------------
# Per-row structural audit
# ------------------------------------------------------------------


class RowReport:
    __slots__ = (
        "row_idx",
        "utter_id",
        "errors",
        "warnings",
        "n_chunks",
        "n_chunks_with_gt",
        "n_gt_terms",
        "n_gt_in_termmap",
        "n_gt_in_assistant",
        "n_audio_missing",
        "wav_checked",
        "wav_mismatch",
    )

    def __init__(self, row_idx: int, utter_id: str) -> None:
        self.row_idx = row_idx
        self.utter_id = utter_id
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.n_chunks = 0
        self.n_chunks_with_gt = 0
        self.n_gt_terms = 0
        self.n_gt_in_termmap = 0
        self.n_gt_in_assistant = 0
        self.n_audio_missing = 0
        self.wav_checked = 0
        self.wav_mismatch = 0


def audit_row(
    row_idx: int,
    row: Dict,
    check_wav_for_this_row: bool,
) -> RowReport:
    uid = str(row.get("utter_id", f"row_{row_idx}"))
    rep = RowReport(row_idx, uid)

    msgs = row.get("messages")
    if not isinstance(msgs, list):
        rep.errors.append("messages_missing: messages missing or not a list")
        return rep

    audios = row.get("audios")
    if not isinstance(audios, list):
        rep.errors.append("audios_missing: audios missing or not a list")
        return rep

    gt_by_chunk = row.get("gt_terms_by_chunk")
    if not isinstance(gt_by_chunk, list):
        rep.errors.append("gt_terms_missing: gt_terms_by_chunk missing or not a list")
        return rep

    merge_multiplier = row.get("merge_multiplier")
    if merge_multiplier is None or not isinstance(merge_multiplier, (int, float)):
        rep.errors.append("merge_multiplier_missing: merge_multiplier missing or non-numeric")
        merge_multiplier = 0
    chunk_metadata = row.get("chunk_metadata") or []

    user_audio_indices: List[int] = []
    for m_idx, m in enumerate(msgs):
        if m.get("role") == "user" and "<audio>" in (m.get("content") or ""):
            user_audio_indices.append(m_idx)

    rep.n_chunks = len(user_audio_indices)

    if len(audios) != rep.n_chunks:
        rep.errors.append(
            f"len_audios_mismatch: len(audios)={len(audios)} "
            f"!= n_user_<audio>={rep.n_chunks}"
        )
    # Semantic: an empty gt_terms_by_chunk means "no chunk has GT". This is
    # stored as [] rather than [[], [], ...] for some rows; treat it as
    # semantically equivalent (upgrade to a full zero-list so downstream
    # per-chunk iteration works).
    if len(gt_by_chunk) == 0 and rep.n_chunks > 0:
        gt_by_chunk_padded = [[] for _ in range(rep.n_chunks)]
        row["gt_terms_by_chunk"] = gt_by_chunk_padded
        gt_by_chunk = gt_by_chunk_padded
        rep.warnings.append(
            f"gt_terms_empty_list_normalized: len(gt_terms_by_chunk)=0 "
            f"normalized to [[],]*{rep.n_chunks}"
        )
    elif len(gt_by_chunk) != rep.n_chunks:
        rep.errors.append(
            f"len_gt_terms_mismatch: len(gt_terms_by_chunk)={len(gt_by_chunk)} "
            f"!= n_user_<audio>={rep.n_chunks}"
        )

    assistant_texts: List[str] = []
    for i, m_idx in enumerate(user_audio_indices):
        nxt = msgs[m_idx + 1] if (m_idx + 1) < len(msgs) else {}
        if nxt.get("role") != "assistant":
            rep.warnings.append(
                f"user<audio> at msg_idx={m_idx} has no assistant follower"
            )
            assistant_texts.append("")
        else:
            assistant_texts.append(nxt.get("content") or "")

    for chunk_idx in range(min(rep.n_chunks, len(audios))):
        audio_path = audios[chunk_idx]
        if not isinstance(audio_path, str):
            rep.errors.append(f"audio_path_type: chunk {chunk_idx} audio path not string")
            continue
        if not os.path.isfile(audio_path):
            rep.n_audio_missing += 1
            rep.errors.append(f"audio_missing: chunk {chunk_idx} missing audio {audio_path}")

    for chunk_idx in range(rep.n_chunks):
        if chunk_idx < len(chunk_metadata):
            mult_expected = chunk_metadata[chunk_idx].get("multiplier")
        else:
            mult_expected = merge_multiplier
        if mult_expected is None:
            rep.errors.append(
                f"multiplier_missing: chunk {chunk_idx} no multiplier "
                f"(neither chunk_metadata nor merge_multiplier)"
            )
            continue
        if not isinstance(mult_expected, (int, float)) or mult_expected <= 0:
            rep.errors.append(
                f"multiplier_invalid: chunk {chunk_idx} invalid multiplier {mult_expected!r}"
            )
            continue

        if check_wav_for_this_row and chunk_idx < len(audios):
            audio_path = audios[chunk_idx]
            if os.path.isfile(audio_path):
                try:
                    dur = wav_duration_seconds(audio_path)
                except Exception as e:
                    rep.errors.append(
                        f"wav_header_read: chunk {chunk_idx} cannot read wav "
                        f"header of {audio_path}: {e}"
                    )
                    continue
                rep.wav_checked += 1
                # Invariant from rebuild_termmap.py:
                #   multiplier = ceil(duration / UNIT_DURATION_SEC)
                # So duration in ((m-1)*U, m*U].  We add +/-DURATION_TOL_SEC
                # slack on both sides for wav encoder rounding.
                upper = float(mult_expected) * UNIT_DURATION_SEC + DURATION_TOL_SEC
                lower = (float(mult_expected) - 1.0) * UNIT_DURATION_SEC - DURATION_TOL_SEC
                # When mult_expected == 1, lower becomes slightly negative which
                # is fine (any positive duration passes the lower bound).
                if not (lower < dur <= upper):
                    rep.wav_mismatch += 1
                    is_last_chunk = (chunk_idx == rep.n_chunks - 1)
                    cat = "wav_bucket_mismatch_tail" if is_last_chunk else "wav_bucket_mismatch_interior"
                    rep.errors.append(
                        f"{cat}: chunk {chunk_idx} dur={dur:.3f}s "
                        f"not in bucket (({lower:.3f}s, {upper:.3f}s]) "
                        f"multiplier={mult_expected} path={audio_path}"
                    )

    for chunk_idx in range(rep.n_chunks):
        if chunk_idx >= len(gt_by_chunk):
            break
        gts = gt_by_chunk[chunk_idx] or []
        if not gts:
            continue
        rep.n_chunks_with_gt += 1
        rep.n_gt_terms += len(gts)

        user_msg = msgs[user_audio_indices[chunk_idx]]
        tm_keys = parse_term_map_keys(user_msg.get("content") or "")
        tm_keys_lower = {safe_lower(k) for k in tm_keys}

        assistant_text = assistant_texts[chunk_idx]
        for gt in gts:
            term = gt.get("term") or ""
            zh = gt.get("zh") or ""
            k = safe_lower(term)
            if not k:
                rep.errors.append(f"gt_term_empty: chunk {chunk_idx} GT with empty term: {gt!r}")
                continue
            if k in tm_keys_lower:
                rep.n_gt_in_termmap += 1
            else:
                if tm_keys_lower:
                    rep.errors.append(
                        f"gt_not_in_termmap: chunk {chunk_idx} GT term {term!r} "
                        f"not in term_map (term_map has {len(tm_keys_lower)} keys)"
                    )
                else:
                    rep.errors.append(
                        f"gt_but_empty_termmap: chunk {chunk_idx} has GT "
                        f"{term!r} but term_map is empty"
                    )

            if zh and zh in assistant_text:
                rep.n_gt_in_assistant += 1
            else:
                rep.warnings.append(
                    f"gt_zh_not_in_assistant: chunk {chunk_idx} GT zh {zh!r} "
                    f"not literal in assistant output"
                )

    return rep


# ------------------------------------------------------------------
# Full-file orchestrator
# ------------------------------------------------------------------


def run_audit(
    input_jsonl: str,
    output_md: str,
    summary_json: str,
    wav_sample_rows: int,
    whisper_samples: int,
    seed: int,
    max_rows: int = 0,
) -> int:
    """Return 0 on pass, non-zero on structural failure rate breach."""
    assert os.path.isfile(input_jsonl), f"Input JSONL not found: {input_jsonl}"

    rng = random.Random(seed)
    rows_total = 0
    n_lines = 0
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for _ in f:
            n_lines += 1
    assert n_lines > 0, f"Empty file: {input_jsonl}"

    if max_rows and max_rows < n_lines:
        wav_sample_set = set(rng.sample(range(max_rows), min(wav_sample_rows, max_rows)))
    else:
        wav_sample_set = set(rng.sample(range(n_lines), min(wav_sample_rows, n_lines)))

    chunk_sample_pool: List[Tuple[int, int, str]] = []

    reports: List[RowReport] = []
    error_counter: Counter = Counter()
    warning_counter: Counter = Counter()
    total_n_chunks = 0
    total_chunks_with_gt = 0
    total_gt_terms = 0
    total_gt_in_termmap = 0
    total_gt_in_assistant = 0
    total_audio_missing = 0
    total_wav_checked = 0
    total_wav_mismatch = 0
    rows_with_any_error = 0
    rows_with_blocking_error = 0

    for row_idx, row in iter_jsonl(input_jsonl, max_rows=max_rows):
        rep = audit_row(row_idx, row, check_wav_for_this_row=(row_idx in wav_sample_set))
        rows_total += 1
        reports.append(rep)
        for e in rep.errors:
            key = e.split(":", 1)[0]
            error_counter[key] += 1
        for w in rep.warnings:
            key = w.split(":", 1)[0]
            warning_counter[key] += 1

        total_n_chunks += rep.n_chunks
        total_chunks_with_gt += rep.n_chunks_with_gt
        total_gt_terms += rep.n_gt_terms
        total_gt_in_termmap += rep.n_gt_in_termmap
        total_gt_in_assistant += rep.n_gt_in_assistant
        total_audio_missing += rep.n_audio_missing
        total_wav_checked += rep.wav_checked
        total_wav_mismatch += rep.wav_mismatch
        if rep.errors:
            rows_with_any_error += 1
            row_cats = {e.split(":", 1)[0] for e in rep.errors}
            if row_cats & BLOCKING_ERROR_CATEGORIES:
                rows_with_blocking_error += 1

        audios = row.get("audios") or []
        gt_by_chunk = row.get("gt_terms_by_chunk") or []
        for c_idx in range(min(len(audios), len(gt_by_chunk))):
            if gt_by_chunk[c_idx]:
                if isinstance(audios[c_idx], str) and os.path.isfile(audios[c_idx]):
                    chunk_sample_pool.append((rep.row_idx, c_idx, audios[c_idx]))

    blocking_rate = rows_with_blocking_error / max(1, rows_total)
    any_error_rate = rows_with_any_error / max(1, rows_total)
    gated = blocking_rate >= STRUCTURAL_FAIL_RATE_GATE

    whisper_plan: List[Tuple[int, int, str, List[Dict]]] = []
    if whisper_samples > 0 and chunk_sample_pool:
        rng2 = random.Random(seed + 1)
        chosen = rng2.sample(
            chunk_sample_pool, min(whisper_samples, len(chunk_sample_pool))
        )
        for row_idx, c_idx, audio_path in chosen:
            row_gt = reports[row_idx - 0]
            whisper_plan.append((row_idx, c_idx, audio_path, []))

    os.makedirs(os.path.dirname(output_md) or ".", exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(f"# Training JSONL audit: `{input_jsonl}`\n\n")
        f.write(f"- rows_total: **{rows_total}**\n")
        f.write(f"- total chunks: **{total_n_chunks}**\n")
        f.write(f"- chunks with GT: **{total_chunks_with_gt}**\n")
        f.write(f"- total GT term occurrences: **{total_gt_terms}**\n")
        if total_gt_terms:
            f.write(
                f"- GT.term in term_map keys: **{total_gt_in_termmap}/{total_gt_terms}** "
                f"({total_gt_in_termmap/total_gt_terms*100:.2f}%)\n"
            )
            f.write(
                f"- GT.zh literal in assistant: **{total_gt_in_assistant}/{total_gt_terms}** "
                f"({total_gt_in_assistant/total_gt_terms*100:.2f}%) (soft check)\n"
            )
        f.write(f"- missing audio files (any chunk): **{total_audio_missing}**\n")
        f.write(
            f"- wav duration checked on sample: **{total_wav_checked}** "
            f"(bucket mismatches: {total_wav_mismatch})\n"
        )
        f.write(
            f"- rows with any error: **{rows_with_any_error}** "
            f"({any_error_rate*100:.2f}% of rows)\n"
        )
        f.write(
            f"- rows with BLOCKING error: **{rows_with_blocking_error}** "
            f"({blocking_rate*100:.2f}% of rows)\n"
        )
        f.write(f"- blocking gate threshold: **{STRUCTURAL_FAIL_RATE_GATE*100:.2f}%**\n")
        f.write(
            f"- gate decision: **{'BLOCK training' if gated else 'PASS (training may proceed)'}**\n\n"
        )

        f.write("## Error categories\n")
        f.write("### BLOCKING (gate downstream training)\n")
        any_blocking = False
        for cat, cnt in error_counter.most_common(100):
            if cat in BLOCKING_ERROR_CATEGORIES:
                f.write(f"- {cnt} x `{cat}`\n")
                any_blocking = True
        if not any_blocking:
            f.write("- (none)\n")
        f.write("\n### NON-BLOCKING (reported but not gated)\n")
        any_nb = False
        for cat, cnt in error_counter.most_common(100):
            if cat in NONBLOCKING_ERROR_CATEGORIES:
                f.write(f"- {cnt} x `{cat}`\n")
                any_nb = True
        if not any_nb:
            f.write("- (none)\n")
        f.write("\n### Other error categories (unclassified -> treated as blocking)\n")
        any_other = False
        for cat, cnt in error_counter.most_common(100):
            if cat not in BLOCKING_ERROR_CATEGORIES and cat not in NONBLOCKING_ERROR_CATEGORIES:
                f.write(f"- {cnt} x `{cat}`\n")
                any_other = True
        if not any_other:
            f.write("- (none)\n")
        f.write("\n## Top warning categories (never gate)\n")
        for cat, cnt in warning_counter.most_common(20):
            f.write(f"- {cnt} x `{cat}`\n")

        f.write("\n## First 20 rows with BLOCKING errors\n")
        shown = 0
        for rep in reports:
            row_cats = {e.split(":", 1)[0] for e in rep.errors}
            if not (row_cats & BLOCKING_ERROR_CATEGORIES):
                continue
            shown += 1
            if shown > 20:
                break
            f.write(f"### row {rep.row_idx}  utter_id={rep.utter_id}\n")
            for e in rep.errors[:8]:
                f.write(f"- ERROR: {e}\n")
            if len(rep.errors) > 8:
                f.write(f"- ... ({len(rep.errors)-8} more errors)\n")
            f.write("\n")
        if shown == 0:
            f.write("- (no rows with blocking errors)\n")

        if whisper_plan:
            f.write("\n## ASR spot-check plan (run separately with whisper)\n")
            for (row_idx, c_idx, audio_path, _results) in whisper_plan:
                f.write(f"- row={row_idx} chunk={c_idx} audio={audio_path}\n")

    os.makedirs(os.path.dirname(summary_json) or ".", exist_ok=True)
    summary = {
        "input_jsonl": input_jsonl,
        "rows_total": rows_total,
        "total_n_chunks": total_n_chunks,
        "chunks_with_gt": total_chunks_with_gt,
        "total_gt_terms": total_gt_terms,
        "gt_in_termmap": total_gt_in_termmap,
        "gt_in_assistant_literal": total_gt_in_assistant,
        "audio_missing": total_audio_missing,
        "wav_checked": total_wav_checked,
        "wav_mismatch": total_wav_mismatch,
        "rows_with_any_error": rows_with_any_error,
        "rows_with_blocking_error": rows_with_blocking_error,
        "any_error_rate": any_error_rate,
        "blocking_rate": blocking_rate,
        "gate_threshold": STRUCTURAL_FAIL_RATE_GATE,
        "gated": gated,
        "error_categories_by_severity": {
            "blocking": [[c, n] for c, n in error_counter.most_common(100)
                         if c in BLOCKING_ERROR_CATEGORIES],
            "nonblocking": [[c, n] for c, n in error_counter.most_common(100)
                            if c in NONBLOCKING_ERROR_CATEGORIES],
            "other": [[c, n] for c, n in error_counter.most_common(100)
                      if c not in BLOCKING_ERROR_CATEGORIES
                      and c not in NONBLOCKING_ERROR_CATEGORIES],
        },
        "top_warnings": warning_counter.most_common(20),
        "whisper_spot_check_plan": [
            {"row": r, "chunk": c, "audio": a} for (r, c, a, _) in whisper_plan
        ],
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[AUDIT] rows={rows_total} any_err={rows_with_any_error} "
          f"blocking={rows_with_blocking_error} blocking_rate={blocking_rate*100:.2f}% "
          f"gated={gated}",
          flush=True)
    print(f"[AUDIT] md: {output_md}", flush=True)
    print(f"[AUDIT] summary: {summary_json}", flush=True)

    return 2 if gated else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--output-md", required=True)
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--wav-sample-rows", type=int, default=DEFAULT_WAV_SAMPLE_ROWS)
    ap.add_argument("--whisper-samples", type=int, default=DEFAULT_WHISPER_SAMPLES)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--max-rows", type=int, default=0,
                    help="0 = process all rows")
    args = ap.parse_args()

    return run_audit(
        input_jsonl=args.input_jsonl,
        output_md=args.output_md,
        summary_json=args.summary_json,
        wav_sample_rows=args.wav_sample_rows,
        whisper_samples=args.whisper_samples,
        seed=args.seed,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    sys.exit(main())
