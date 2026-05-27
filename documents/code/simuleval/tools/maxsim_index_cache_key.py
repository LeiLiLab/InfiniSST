#!/usr/bin/env python3
"""Resolve and validate stable MaxSim text-index cache paths.

The MaxSim index filename must be short enough for filesystem limits, but the
cache key must still encode the inputs that make an index reusable.  This tool
uses a short hash filename plus a sidecar manifest.  Existing indexes without a
matching manifest fail fast instead of being silently reused.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
KEY_VERSION = "maxsim-text-index-v1"


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _file_identity(path_s: str, mode: str) -> dict[str, Any]:
    path = Path(path_s).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"missing input path: {path_s}")
    st = path.stat()
    out: dict[str, Any] = {
        "path": str(path),
        "realpath": str(path.resolve()),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
        "mode": mode,
    }
    if mode == "sha256":
        out["sha256"] = _sha256_file(path)
        out["cache_identity"] = {
            "mode": mode,
            "sha256": out["sha256"],
            "size": int(st.st_size),
        }
    elif mode == "stat":
        out["cache_identity"] = {
            "mode": mode,
            "realpath": str(path.resolve()),
            "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
        }
    else:
        raise ValueError(f"unsupported hash mode: {mode}")
    return out


def _safe_tag(text: str, max_len: int = 48) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._-")
    return (text or "glossary")[:max_len]


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    inputs = {
        "retriever_checkpoint": _file_identity(args.model_path, args.checkpoint_hash_mode),
        "glossary": _file_identity(args.glossary_path, args.glossary_hash_mode),
        "builder_script": _file_identity(args.builder_script, args.builder_hash_mode),
    }
    cache_inputs = {
        name: ident["cache_identity"]
        for name, ident in inputs.items()
    }
    params = {
        "text_model_id": args.text_model_id,
        "text_lora_rank": int(args.text_lora_rank),
        "text_lora_alpha": int(args.text_lora_alpha),
        "index_format": "torch_pt",
        "index_kind": "maxsim_text_embeddings",
    }
    fingerprint_payload = {
        "schema_version": SCHEMA_VERSION,
        "key_version": KEY_VERSION,
        "inputs": cache_inputs,
        "params": params,
    }
    fingerprint = hashlib.sha256(_canonical_json(fingerprint_payload).encode("utf-8")).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "key_version": KEY_VERSION,
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "input_provenance": inputs,
    }


def resolve_paths(args: argparse.Namespace, manifest: dict[str, Any]) -> tuple[Path, Path, str]:
    cache_dir = Path(args.cache_dir).expanduser()
    tag = _safe_tag(args.glossary_tag or Path(args.glossary_path).stem)
    short = manifest["fingerprint"][:16]
    name = f"{args.prefix}_{tag}_{short}_tr{int(args.text_lora_rank)}_ta{int(args.text_lora_alpha)}.pt"
    index_path = cache_dir / name
    manifest_path = Path(str(index_path) + ".manifest.json")
    return index_path, manifest_path, short


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def verify_existing(index_path: Path, manifest_path: Path, expected: dict[str, Any]) -> None:
    if not index_path.exists():
        return
    if not manifest_path.exists():
        raise RuntimeError(
            f"index exists without manifest; refusing silent reuse: index={index_path} manifest={manifest_path}"
        )
    got = _load_json(manifest_path)
    if got.get("fingerprint") != expected.get("fingerprint"):
        raise RuntimeError(
            "index manifest fingerprint mismatch; refusing stale cache reuse: "
            f"index={index_path} manifest={manifest_path} "
            f"got={got.get('fingerprint')} expected={expected.get('fingerprint')}"
        )
    if got.get("status") != "ready":
        raise RuntimeError(f"index manifest status is not ready: {manifest_path} status={got.get('status')}")


def write_manifest_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def command_resolve(args: argparse.Namespace) -> int:
    manifest = build_manifest(args)
    index_path, manifest_path, short = resolve_paths(args, manifest)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    verify_existing(index_path, manifest_path, manifest)

    if args.output_format == "json":
        print(json.dumps({
            "index_path": str(index_path),
            "manifest_path": str(manifest_path),
            "cache_key": manifest["fingerprint"],
            "short_key": short,
            "exists": index_path.exists(),
        }, ensure_ascii=False, indent=2))
    else:
        print(f"INDEX_PATH={shlex.quote(str(index_path))}")
        print(f"INDEX_MANIFEST_PATH={shlex.quote(str(manifest_path))}")
        print(f"INDEX_CACHE_KEY={shlex.quote(manifest['fingerprint'])}")
        print(f"INDEX_CACHE_SHORT_KEY={shlex.quote(short)}")
    return 0


def command_finalize(args: argparse.Namespace) -> int:
    manifest = build_manifest(args)
    index_path = Path(args.index_path).expanduser()
    manifest_path = Path(args.manifest_path).expanduser()
    if not index_path.exists() or index_path.stat().st_size <= 0:
        raise RuntimeError(f"cannot finalize missing/empty index: {index_path}")
    if manifest_path.exists() and not args.force:
        got = _load_json(manifest_path)
        if got.get("fingerprint") != manifest.get("fingerprint"):
            raise RuntimeError(f"refusing to overwrite mismatched manifest: {manifest_path}")

    st = index_path.stat()
    data = dict(manifest)
    data.update({
        "status": "ready",
        "created_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "index": {
            "path": str(index_path),
            "realpath": str(index_path.resolve()),
            "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
        },
    })
    write_manifest_atomic(manifest_path, data)
    print(f"[INFO] wrote MaxSim index manifest: {manifest_path}", flush=True)
    return 0


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--glossary-path", required=True)
    parser.add_argument("--builder-script", required=True)
    parser.add_argument("--glossary-tag", default="")
    parser.add_argument("--text-model-id", default="BAAI/bge-m3")
    parser.add_argument("--text-lora-rank", type=int, required=True)
    parser.add_argument("--text-lora-alpha", type=int, default=256)
    parser.add_argument("--checkpoint-hash-mode", choices=["stat", "sha256"], default="stat")
    parser.add_argument("--glossary-hash-mode", choices=["stat", "sha256"], default="sha256")
    parser.add_argument("--builder-hash-mode", choices=["stat", "sha256"], default="sha256")
    parser.add_argument("--prefix", default="maxsim")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_resolve = sub.add_parser("resolve", help="Print a stable cache path and verify any existing index manifest.")
    add_common_args(p_resolve)
    p_resolve.add_argument("--cache-dir", required=True)
    p_resolve.add_argument("--output-format", choices=["shell", "json"], default="shell")

    p_finalize = sub.add_parser("finalize", help="Write the sidecar manifest after building an index.")
    add_common_args(p_finalize)
    p_finalize.add_argument("--index-path", required=True)
    p_finalize.add_argument("--manifest-path", required=True)
    p_finalize.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.command == "resolve":
            return command_resolve(args)
        if args.command == "finalize":
            return command_finalize(args)
        raise AssertionError(args.command)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
