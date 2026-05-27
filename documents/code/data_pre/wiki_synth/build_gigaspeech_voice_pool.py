#!/usr/bin/env python3
"""Build a GigaSpeech-based voice-reference pool for CosyVoice zero-shot TTS.

Replaces the narrow VCTK (109 studio UK speakers) prompt set with a broader
set of GigaSpeech segments spanning audiobook/podcast/youtube domains. Used
as the zero-shot reference for CosyVoice during wiki-term synthesis, to
broaden the acoustic distribution the retriever's speech encoder sees
during training.

Output format mirrors the existing VCTK `speaker_index.json`:
  [
    {"speaker_id": "POD0000001234_S0000008",
     "wav_path": "/.../gigaspeech_speaker_prompts/POD..._S....wav",
     "text": "and these introductions are inevitably both monotonous ...",
     "gender": "unknown",
     "accent": "unknown"},
    ...
  ]

Reuses existing helpers:
  - parse_textgrid_words from enrich_jsonl_with_mfa_timestamps.py
  - SQLite schema from /mnt/gemini/data1/jiaxuanluo/gigaspeech_mfa_index/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_DATA_PRE = Path(__file__).resolve().parents[1]  # .../documents/code/data_pre
if str(_DATA_PRE) not in sys.path:
    sys.path.insert(0, str(_DATA_PRE))

from enrich_jsonl_with_mfa_timestamps import parse_textgrid_words  # noqa: E402

# Defaults
DEFAULT_SQLITE = "/mnt/gemini/data1/jiaxuanluo/gigaspeech_mfa_index/gigaspeech_mfa_index.sqlite"
DEFAULT_TG_DIR = "/mnt/taurus/data/siqiouyang/datasets/gigaspeech/textgrids"
# v2 (2026-04-23): moved to gemini/home because taurus/data is 97% full.
DEFAULT_OUTPUT_DIR = "/mnt/gemini/home/jiaxuanluo/gigaspeech_speaker_prompts"
SAMPLE_RATE = 16000
MIN_DUR_S = 5.0
MAX_DUR_S = 12.0
MIN_WORDS = 6

# v2 one-per-opus caps. GigaSpeech unique-opus census (2026-04-23):
#   audiobook: 1,092   podcast: 14,602   youtube: 18,823   total: ~34,517
# Default target split for n_voices=10000:
#   use all audiobook (narrator voices capped by supply), then fill podcast +
#   youtube equally to hit the global target.
OPUS_CAP_AUDIOBOOK = 1092  # hard cap from census; SQL may return fewer after filters
DEFAULT_DOMAIN_TARGETS = {
    "audiobook": OPUS_CAP_AUDIOBOOK,
    # podcast / youtube filled dynamically to hit args.n_voices
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def opus_domain(opus_path: str) -> str:
    """Return 'audiobook' | 'podcast' | 'youtube' based on opus path."""
    p = opus_path.lower()
    for d in ("audiobook", "podcast", "youtube"):
        if f"/{d}/" in p:
            return d
    return "unknown"


def fetch_candidate_segs(
    sqlite_path: str,
    domain_targets: Dict[str, int],
    rng: random.Random,
    one_per_opus: bool = True,
) -> List[Tuple[str, str, int, int, str]]:
    """Sample segments from each of audiobook/podcast/youtube.

    Filter by expected duration (opus SR=16kHz, so duration_samples / 16000)
    to keep things in the MIN_DUR_S..MAX_DUR_S window.

    If `one_per_opus` is True (default, v2), group candidates by `opus` path
    first and pick a single random seg per opus, so the returned pool has at
    most one reference per underlying long-form recording. This matches the
    intent of "10k distinct speakers" (each long opus is typically 1 speaker).

    Returns list of (seg_id, opus_path, start_samples, end_samples, domain),
    oversampled 3x per domain to absorb TextGrid/decode failures downstream.
    """
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()

    min_samples = int(MIN_DUR_S * SAMPLE_RATE)
    max_samples = int(MAX_DUR_S * SAMPLE_RATE)

    out: List[Tuple[str, str, int, int, str]] = []
    for domain in ("podcast", "audiobook", "youtube"):
        target = domain_targets.get(domain, 0)
        if target <= 0:
            continue
        needle = f"/{domain}/"
        logger.info(f"[SQL] scanning {domain} segments (target={target})")
        rows = cur.execute(
            "SELECT seg_id, opus, start, end FROM manifest_segments "
            "WHERE opus LIKE ? AND (end - start) BETWEEN ? AND ?",
            (f"%{needle}%", min_samples, max_samples),
        ).fetchall()
        logger.info(f"[SQL]   {domain}: {len(rows):,} raw candidates in duration window")

        if one_per_opus:
            by_opus: Dict[str, List[Tuple[str, str, int, int]]] = {}
            for row in rows:
                by_opus.setdefault(row[1], []).append(row)
            logger.info(
                f"[SQL]   {domain}: {len(by_opus):,} unique opus files after dedup"
            )
            rows = [rng.choice(segs) for segs in by_opus.values()]

        rng.shuffle(rows)
        # Oversample target*3 to absorb TG/decode failures without running out.
        keep = min(len(rows), target * 3)
        rows = rows[:keep]
        logger.info(f"[SQL]   {domain}: kept {len(rows):,} after oversample cap")
        for seg_id, opus, s, e in rows:
            out.append((seg_id, opus, int(s), int(e), domain))
    con.close()
    rng.shuffle(out)
    return out


def extract_voice_entry(
    seg_id: str,
    opus_path: str,
    start_samples: int,
    end_samples: int,
    domain: str,
    tg_dir: str,
    output_dir: str,
) -> Optional[Dict]:
    """Decode the opus segment to a wav and return the speaker_index entry.

    Returns None if TextGrid missing, transcript too short, or decode fails.
    """
    tg_path = os.path.join(tg_dir, f"{seg_id}.TextGrid")
    if not os.path.isfile(tg_path):
        return None

    try:
        intervals = parse_textgrid_words(tg_path)
    except Exception as e:
        logger.debug(f"[TG] parse failed {seg_id}: {e}")
        return None

    words = [w for _, _, w in intervals if w.strip()]
    if len(words) < MIN_WORDS:
        return None
    text = " ".join(words)

    wav_out = os.path.join(output_dir, f"{seg_id}.wav")
    if os.path.isfile(wav_out):
        # Idempotent - trust existing file.
        return {
            "speaker_id": seg_id,
            "wav_path": wav_out,
            "text": text,
            "gender": "unknown",
            "accent": "unknown",
            "domain": domain,
            "duration_s": (end_samples - start_samples) / SAMPLE_RATE,
        }

    import soundfile as sf
    try:
        with sf.SoundFile(opus_path) as sfo:
            assert sfo.samplerate == SAMPLE_RATE, f"{opus_path}: SR={sfo.samplerate}"
            sfo.seek(start_samples)
            audio = sfo.read(end_samples - start_samples, dtype="float32")
    except Exception as e:
        logger.debug(f"[DECODE] failed {seg_id}: {e}")
        return None

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    import numpy as np
    # Loudness-normalize to -1..1 (peak normalization; keeps voice clean).
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak * 0.95

    try:
        sf.write(wav_out, audio.astype("float32"), SAMPLE_RATE, subtype="PCM_16")
    except Exception as e:
        logger.debug(f"[WRITE] failed {seg_id}: {e}")
        return None

    return {
        "speaker_id": seg_id,
        "wav_path": wav_out,
        "text": text,
        "gender": "unknown",
        "accent": "unknown",
        "domain": domain,
        "duration_s": (end_samples - start_samples) / SAMPLE_RATE,
    }


def _compute_domain_targets(n_voices: int, skip_audiobook: bool) -> Dict[str, int]:
    """Split the pool size across domains.

    - audiobook capped at supply (~1092 unique opus). Narrator voices are
      clean but over-represented if we were to split evenly; use the full
      supply only (or skip entirely with --skip_audiobook).
    - podcast + youtube split the remainder evenly; they have far more unique
      opus than we need (14k / 18k).
    """
    if skip_audiobook:
        ab = 0
    else:
        ab = min(OPUS_CAP_AUDIOBOOK, n_voices // 3)
    remaining = n_voices - ab
    pod = remaining // 2
    yt = remaining - pod
    return {"audiobook": ab, "podcast": pod, "youtube": yt}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", default=DEFAULT_SQLITE)
    ap.add_argument("--textgrid_dir", default=DEFAULT_TG_DIR)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--n_voices", type=int, default=10000,
                    help="Target pool size; split per --domain-targets / caps.")
    ap.add_argument(
        "--one_per_opus", action=argparse.BooleanOptionalAction, default=True,
        help="v2 default: dedup candidate segs by opus file BEFORE sampling, so "
             "the pool maps to distinct long-form recordings (proxy for distinct "
             "speakers). Set --no-one_per_opus to reproduce the v1 (seg-biased) "
             "behavior.",
    )
    ap.add_argument(
        "--skip_audiobook", action="store_true", default=False,
        help="Skip audiobook domain entirely. Narrator voices are clean but "
             "stylistically concentrated; dropping them leaves 10k / 2 among "
             "podcast + youtube if you suspect audiobook leaks into ACL OOD.",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = random.Random(args.seed)

    domain_targets = _compute_domain_targets(args.n_voices, args.skip_audiobook)
    logger.info(
        f"[INIT] target={args.n_voices}, domain_targets={domain_targets}, "
        f"one_per_opus={args.one_per_opus}, output_dir={args.output_dir}"
    )

    t0 = time.time()
    segs = fetch_candidate_segs(
        args.sqlite, domain_targets, rng, one_per_opus=args.one_per_opus,
    )
    logger.info(f"[SQL] total candidates after oversample: {len(segs):,}")

    entries: List[Dict] = []
    domain_counts: Dict[str, int] = {"audiobook": 0, "podcast": 0, "youtube": 0}
    domain_target = {d: domain_targets.get(d, 0) for d in domain_counts}

    progress_every = 500
    for i, (seg_id, opus, s, e, domain) in enumerate(segs):
        if domain_counts.get(domain, 0) >= domain_target.get(domain, 0):
            continue
        entry = extract_voice_entry(seg_id, opus, s, e, domain, args.textgrid_dir, args.output_dir)
        if entry is None:
            continue
        entries.append(entry)
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

        if (i + 1) % progress_every == 0:
            logger.info(
                f"[WORK] scanned={i+1} kept={len(entries)} by_domain={domain_counts}"
            )

        if len(entries) >= args.n_voices:
            break

    logger.info(
        f"[DONE] built {len(entries)} voice prompts in {time.time()-t0:.1f}s, "
        f"by_domain={domain_counts}"
    )

    index_path = os.path.join(args.output_dir, "speaker_index.json")
    with open(index_path, "w") as f:
        json.dump(entries, f, indent=2)
    logger.info(f"[DONE] wrote {index_path}")


if __name__ == "__main__":
    main()
