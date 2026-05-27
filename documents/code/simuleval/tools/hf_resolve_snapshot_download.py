#!/usr/bin/env python3
"""Download a fixed Hugging Face snapshot file list with stdlib only.

This is intentionally small and dependency-free for clusters where
`huggingface_hub` is not available.  It downloads repo files through
`https://huggingface.co/<repo>/resolve/<revision>/<path>` and supports
best-effort resume via HTTP Range requests and `.part` files.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def read_manifest(path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, size_s = line.split("\t", 1)
        rows.append((name, int(size_s)))
    return rows


def request_url(url: str, token: str, start: int) -> urllib.request.Request:
    headers = {"User-Agent": "InfiniSST-PSC-HF-stdlib-downloader"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if start > 0:
        headers["Range"] = f"bytes={start}-"
    return urllib.request.Request(url, headers=headers)


def download_one(repo: str, revision: str, name: str, expected_size: int, out_dir: Path, token: str, chunk_size: int) -> None:
    out_path = out_dir / name
    part_path = out_dir / f"{name}.part"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size == expected_size:
        print(f"[SKIP] {name} size={expected_size}", flush=True)
        return
    if out_path.exists():
        raise RuntimeError(f"existing file has wrong size: {out_path} got={out_path.stat().st_size} expected={expected_size}")

    quoted = "/".join(urllib.parse.quote(p) for p in name.split("/"))
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{quoted}"
    start = part_path.stat().st_size if part_path.exists() else 0
    mode = "ab" if start else "wb"

    print(f"[GET] {name} resume={start} expected={expected_size}", flush=True)
    try:
        with urllib.request.urlopen(request_url(url, token, start), timeout=120) as resp:
            status = getattr(resp, "status", None)
            if start and status == 200:
                print(f"[WARN] server ignored Range for {name}; restarting", flush=True)
                start = 0
                mode = "wb"
            with part_path.open(mode + "") as fout:
                done = start
                last_report = time.time()
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    fout.write(chunk)
                    done += len(chunk)
                    now = time.time()
                    if now - last_report >= 30:
                        pct = (100.0 * done / expected_size) if expected_size else 0.0
                        print(f"[PROGRESS] {name} {done}/{expected_size} ({pct:.2f}%)", flush=True)
                        last_report = now
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} for {name}: {exc.reason}") from exc

    got = part_path.stat().st_size
    if got != expected_size:
        raise RuntimeError(f"incomplete download: {name} got={got} expected={expected_size}")
    part_path.rename(out_path)
    print(f"[DONE] {name}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--token-file", default=str(Path.home() / ".cache/huggingface/token"), type=Path)
    ap.add_argument("--chunk-size-mb", default=16, type=int)
    ap.add_argument("--retries", default=5, type=int)
    args = ap.parse_args()

    token = args.token_file.read_text(encoding="utf-8").strip() if args.token_file.exists() else ""
    if not token:
        print("[ERROR] missing Hugging Face token", file=sys.stderr)
        return 2

    rows = read_manifest(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = max(1, args.chunk_size_mb) * 1024 * 1024

    for name, size in rows:
        for attempt in range(1, args.retries + 1):
            try:
                download_one(args.repo, args.revision, name, size, args.out_dir, token, chunk_size)
                break
            except Exception as exc:  # noqa: BLE001 - log and retry any transport failure
                if attempt >= args.retries:
                    raise
                print(f"[RETRY] {name} attempt={attempt}/{args.retries} error={exc}", file=sys.stderr, flush=True)
                time.sleep(min(60, 5 * attempt))
    print(f"[ALL DONE] {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
