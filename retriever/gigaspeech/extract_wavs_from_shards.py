import argparse
import io
import json
import os
import tarfile
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

from tqdm import tqdm


@dataclass
class Stats:
    samples_written: int = 0
    wav_written: int = 0
    wav_skipped_exists: int = 0
    wav_conflict_renamed: int = 0
    missing_wav_for_json: int = 0
    bad_json: int = 0
    bad_wav: int = 0


def safe_relpath(abs_path: str, src_base: str) -> str:
    """
    Compute relpath under src_base. If abs_path is outside src_base, fall back to basename.
    """
    try:
        rel = os.path.relpath(abs_path, src_base)
        if rel.startswith(".."):
            return os.path.basename(abs_path)
        return rel
    except Exception:
        return os.path.basename(abs_path)


def ensure_dir(path: str, created: Set[str]) -> None:
    d = os.path.dirname(path)
    if d and d not in created:
        os.makedirs(d, exist_ok=True)
        created.add(d)


def write_file_if_needed(
    dst_path: str,
    content: bytes,
    created_dirs: Set[str],
    stats: Stats,
    conflict_suffix: str,
) -> str:
    """
    Write bytes to dst_path unless an existing file with same size exists.
    If dst_path exists but size differs, write to a conflict path with suffix and return that path.
    """
    ensure_dir(dst_path, created_dirs)

    if os.path.exists(dst_path):
        try:
            if os.path.getsize(dst_path) == len(content):
                stats.wav_skipped_exists += 1
                return dst_path
        except Exception:
            # If stat fails, fall through to rewrite via conflict path.
            pass

        root, ext = os.path.splitext(dst_path)
        conflict_path = f"{root}{conflict_suffix}{ext}"
        ensure_dir(conflict_path, created_dirs)
        with open(conflict_path, "wb") as f:
            f.write(content)
        stats.wav_conflict_renamed += 1
        return conflict_path

    with open(dst_path, "wb") as f:
        f.write(content)
    stats.wav_written += 1
    return dst_path


def process_shards(
    shards_dir: str,
    output_root: str,
    dest_subdir: str,
    output_jsonl: str,
    src_base: str,
    dry_run: bool = False,
    max_shards: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> Stats:
    """
    Extract wavs from WebDataset tar shards and rewrite jsonl with updated local paths.

    Expected tar members:
      - {key}.wav
      - {key}.json
    key is an 8-digit line index (string).
    """
    stats = Stats()
    created_dirs: Set[str] = set()

    shard_files = [
        os.path.join(shards_dir, f)
        for f in os.listdir(shards_dir)
        if f.startswith("shard_") and f.endswith(".tar")
    ]
    shard_files.sort()
    if max_shards is not None:
        shard_files = shard_files[: max(0, int(max_shards))]

    if not shard_files:
        raise FileNotFoundError(f"No shard_*.tar found under: {shards_dir}")

    pending_wav: Dict[str, bytes] = {}

    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    out_f = open(output_jsonl, "w", encoding="utf-8")

    try:
        for shard_path in tqdm(shard_files, desc=f"Scanning shards in {shards_dir}", unit="shard"):
            with tarfile.open(shard_path, "r") as tar:
                for member in tar:
                    if not member.isfile():
                        continue

                    name = member.name
                    key, ext = os.path.splitext(os.path.basename(name))
                    ext = ext.lstrip(".").lower()

                    fobj = tar.extractfile(member)
                    if fobj is None:
                        continue

                    if ext == "wav":
                        try:
                            pending_wav[key] = fobj.read()
                        except Exception:
                            stats.bad_wav += 1
                        continue

                    if ext == "json":
                        try:
                            meta = json.loads(fobj.read().decode("utf-8", errors="strict"))
                        except Exception:
                            stats.bad_json += 1
                            continue

                        # Determine destination wav path using original chunk_audio_path
                        original_path = meta.get("chunk_audio_path", "")
                        rel_path = safe_relpath(original_path, src_base)
                        dst_path = os.path.join(output_root, dest_subdir, rel_path)

                        wav_bytes = pending_wav.pop(key, None)
                        if wav_bytes is None:
                            # Some tar layouts may store json before wav; try a best-effort lookup.
                            stats.missing_wav_for_json += 1
                            continue

                        if not dry_run:
                            dst_path = write_file_if_needed(
                                dst_path=dst_path,
                                content=wav_bytes,
                                created_dirs=created_dirs,
                                stats=stats,
                                conflict_suffix=f"_dup{key}",
                            )

                        # Rewrite metadata to point to local path.
                        meta["chunk_audio_path"] = dst_path
                        if "line_idx" not in meta:
                            try:
                                meta["line_idx"] = int(key)
                            except Exception:
                                pass

                        out_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
                        stats.samples_written += 1
                        if max_samples is not None and stats.samples_written >= int(max_samples):
                            return stats
                        continue

            # Avoid unbounded growth if a shard is malformed.
            pending_wav.clear()

    finally:
        out_f.close()

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_shards_dir", type=str, required=True)
    parser.add_argument("--dev_shards_dir", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument(
        "--dest_subdir",
        type=str,
        default="local_wavs_from_shards",
        help="Subdirectory under output_root to store extracted wavs.",
    )
    parser.add_argument("--src_base", type=str, default="/mnt/gemini/data1/jiaxuanluo/")
    parser.add_argument("--out_train_jsonl", type=str, required=True)
    parser.add_argument("--out_dev_jsonl", type=str, required=True)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--max_shards", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    print("Extracting train shards...")
    train_stats = process_shards(
        shards_dir=args.train_shards_dir,
        output_root=args.output_root,
        dest_subdir=args.dest_subdir,
        output_jsonl=args.out_train_jsonl,
        src_base=args.src_base,
        dry_run=args.dry_run,
        max_shards=args.max_shards,
        max_samples=args.max_samples,
    )

    print("Extracting dev shards...")
    dev_stats = process_shards(
        shards_dir=args.dev_shards_dir,
        output_root=args.output_root,
        dest_subdir=args.dest_subdir,
        output_jsonl=args.out_dev_jsonl,
        src_base=args.src_base,
        dry_run=args.dry_run,
        max_shards=args.max_shards,
        max_samples=args.max_samples,
    )

    print("Done.")
    print(f"Train stats: {train_stats}")
    print(f"Dev stats:   {dev_stats}")


if __name__ == "__main__":
    main()


