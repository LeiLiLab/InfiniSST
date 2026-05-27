"""
Build an SQLite index to join:
- align TSV segments (id without _S) -> (opus_path, start, end)
- manifest segments (id with _S) -> (opus_path, start, end)

This enables fast range-overlap lookup from an align segment to the set of
manifest segments whose TextGrids exist, so we can locate term timing from MFA.
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple


# ======Configuration=====
DEFAULT_ALIGN_TSV = "/mnt/gemini/data1/jiaxuanluo/train_xl_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv"
DEFAULT_MANIFEST_TSVS = [
    "/mnt/taurus/data/siqiouyang/datasets/gigaspeech/manifests/train_xl.tsv",
    "/mnt/taurus/data/siqiouyang/datasets/gigaspeech/manifests/train_xl_case.tsv",
]
DEFAULT_SQLITE_PATH = "outputs/gigaspeech_mfa_index/gigaspeech_mfa_index.sqlite"

SQLITE_BATCH_SIZE = 50_000


@dataclass(frozen=True)
class Segment:
    seg_id: str
    opus: str
    start: int
    end: int


def parse_audio_field(audio_field: str) -> Tuple[str, int, int]:
    # Format: /path/to/file.opus:OFFSET:LENGTH
    # OFFSET/LENGTH are integers (unit consistent across files).
    parts = audio_field.rsplit(":", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid audio field: {audio_field}")
    opus = parts[0]
    start = int(parts[1])
    length = int(parts[2])
    return opus, start, start + length


def iter_tsv_rows(tsv_path: Path) -> Iterator[List[str]]:
    with tsv_path.open("r", encoding="utf-8", errors="replace") as f:
        header = next(f, None)
        if header is None:
            return
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            yield line.split("\t")


def build(db_path: Path, align_tsv: Path, manifest_tsvs: List[Path]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=OFF;")
    con.execute("PRAGMA synchronous=OFF;")
    con.execute("PRAGMA temp_store=MEMORY;")

    con.execute(
        """
        CREATE TABLE manifest_segments (
          seg_id TEXT PRIMARY KEY,
          opus   TEXT NOT NULL,
          start  INTEGER NOT NULL,
          end    INTEGER NOT NULL
        );
        """
    )
    con.execute(
        """
        CREATE TABLE align_segments (
          align_id TEXT PRIMARY KEY,
          opus     TEXT NOT NULL,
          start    INTEGER NOT NULL,
          end      INTEGER NOT NULL
        );
        """
    )
    con.commit()

    # Load manifests
    for mpath in manifest_tsvs:
        print(f"loading manifest {mpath}")
        buf: List[Tuple[str, str, int, int]] = []
        inserted = 0
        for cols in iter_tsv_rows(mpath):
            seg_id = cols[0]
            audio_field = cols[1]
            opus, start, end = parse_audio_field(audio_field)
            buf.append((seg_id, opus, start, end))
            if len(buf) >= SQLITE_BATCH_SIZE:
                con.executemany("INSERT OR IGNORE INTO manifest_segments(seg_id, opus, start, end) VALUES (?, ?, ?, ?)", buf)
                con.commit()
                inserted += len(buf)
                buf.clear()
        if buf:
            con.executemany("INSERT OR IGNORE INTO manifest_segments(seg_id, opus, start, end) VALUES (?, ?, ?, ?)", buf)
            con.commit()
            inserted += len(buf)
            buf.clear()
        print(f"manifest loaded rows={inserted}")

    # Index for overlap query
    con.execute("CREATE INDEX idx_manifest_opus_start_end ON manifest_segments(opus, start, end);")
    con.commit()

    # Load align tsv
    print(f"loading align {align_tsv}")
    buf2: List[Tuple[str, str, int, int]] = []
    inserted2 = 0
    for cols in iter_tsv_rows(align_tsv):
        align_id = cols[0]
        audio_field = cols[1]
        opus, start, end = parse_audio_field(audio_field)
        buf2.append((align_id, opus, start, end))
        if len(buf2) >= SQLITE_BATCH_SIZE:
            con.executemany("INSERT OR REPLACE INTO align_segments(align_id, opus, start, end) VALUES (?, ?, ?, ?)", buf2)
            con.commit()
            inserted2 += len(buf2)
            buf2.clear()
    if buf2:
        con.executemany("INSERT OR REPLACE INTO align_segments(align_id, opus, start, end) VALUES (?, ?, ?, ?)", buf2)
        con.commit()
        inserted2 += len(buf2)
        buf2.clear()
    print(f"align loaded rows={inserted2}")

    con.execute("CREATE INDEX idx_align_opus_start_end ON align_segments(opus, start, end);")
    con.commit()
    con.close()
    print(f"done sqlite={db_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SQLite index for MFA TextGrid lookup.")
    parser.add_argument("--align-tsv", default=DEFAULT_ALIGN_TSV)
    parser.add_argument("--manifest-tsv", action="append", default=None, help="Repeatable. If omitted, defaults are used.")
    parser.add_argument("--sqlite-path", default=DEFAULT_SQLITE_PATH)
    args = parser.parse_args()

    manifest_paths = [Path(p) for p in (args.manifest_tsv if args.manifest_tsv else DEFAULT_MANIFEST_TSVS)]
    build(Path(args.sqlite_path), Path(args.align_tsv), manifest_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

