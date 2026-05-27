#!/usr/bin/env python3

"""
Prepare per-paper inputs for running SimulEval using ALREADY EXTRACTED per-paper glossaries.

This script reads pre-generated per-paper glossary files (from extract_acl_terms_from_paper_v2.py)
and creates dev.source/dev.target subsets + a mapping JSON for SimulEval runs.

Input:
- Per-paper glossaries: extracted_glossaries_by_paper/extracted_glossary__<paper_id>.json
  (or read from manifest JSON)
- dev.source / dev.target.{lang_code}

Output:
- Symlinks/copies of per-paper glossaries (or just use original paths)
- Per-paper dev.source/dev.target subsets
- paper_inputs_map.json

All user-facing strings are in English.
"""

from __future__ import annotations

# ======Configuration=====
DEFAULT_GLOSSARIES_DIR = (
    "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/data/data_pre/extracted_glossaries_by_paper"
)
DEFAULT_MANIFEST_JSON = (
    "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/data/data_pre/extracted_glossary_by_paper_manifest.json"
)
DEFAULT_DATA_ROOT = "/mnt/taurus/data/siqiouyang/datasets/acl6060"
DEFAULT_LANG_CODE = "zh"

DEFAULT_OUTPUT_DIR = (
    "/mnt/gemini/data2/jiaxuanluo/"
    "infinisst_omni_vllm_rag_rank32_iter_0000452_hf_zh_k1_10_k2_sweep_glossary2/zh/"
    "__paper_inputs__"
)

DEFAULT_LISTS_DIRNAME = "lists"
DEFAULT_MAPPING_JSON_NAME = "paper_inputs_map.json"

# The upstream dev.source uses node-local prefixes (e.g. /mnt/data/...) which
# resolve differently on aries vs taurus.  We always rewrite to a fully
# qualified cross-node path so source lists are portable across partitions.
# Ordered: apply first match per line.  Fail loudly if no expected prefix
# matched; do NOT silently accept unknown paths.
PORTABLE_SOURCE_PATH_REWRITES = [
    ("/mnt/data/siqiouyang/", "/mnt/taurus/data/siqiouyang/"),
    ("/mnt/data1/siqiouyang/", "/mnt/taurus/data1/siqiouyang/"),
]
PORTABLE_PATH_ALLOWED_PREFIXES = (
    "/mnt/taurus/", "/mnt/aries/", "/mnt/gemini/",
)

EXIT_CONFIG_ERROR = 2
EXIT_DATA_ERROR = 3
# ======Configuration=====

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _err(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _portable_path(raw: str) -> str:
    """Rewrite a node-local wav path (e.g. /mnt/data/...) to the fully qualified
    cross-node form (/mnt/taurus/data/...), per user rule on portable paths.

    Fails loudly if the rewritten path does not start with one of the allowed
    partition-qualified prefixes - we do not want to silently propagate an
    unknown or ambiguous path into the per-paper source list.
    """
    s = raw.strip()
    if not s:
        return s
    rewritten = s
    for old_prefix, new_prefix in PORTABLE_SOURCE_PATH_REWRITES:
        if rewritten.startswith(old_prefix):
            rewritten = new_prefix + rewritten[len(old_prefix):]
            break
    if not rewritten.startswith(PORTABLE_PATH_ALLOWED_PREFIXES):
        raise ValueError(
            "source path is not cross-node portable after rewrite: "
            f"{raw!r} -> {rewritten!r}; expected one of {PORTABLE_PATH_ALLOWED_PREFIXES}"
        )
    return rewritten


def _paper_id_from_wav_path(wav_path: str) -> Optional[str]:
    s = str(wav_path).strip()
    if not s:
        return None
    base = os.path.basename(s)
    if base.lower().endswith(".wav"):
        base = base[: -len(".wav")]
    return base.strip() or None


def _load_per_paper_glossaries(glossaries_dir: Optional[Path], manifest_path: Optional[Path]) -> Dict[str, Path]:
    """
    Returns: {paper_id: glossary_json_path}
    """
    paper_gloss: Dict[str, Path] = {}

    if manifest_path and manifest_path.is_file():
        _info(f"Loading per-paper glossaries from manifest: {manifest_path}")
        manifest = json.loads(_read_text(manifest_path))
        papers = manifest.get("papers", {})
        for paper_id, info in papers.items():
            gp = info.get("glossary_path", "")
            if gp:
                p = Path(gp)
                if p.is_file():
                    paper_gloss[str(paper_id)] = p
        return paper_gloss

    if glossaries_dir and glossaries_dir.is_dir():
        _info(f"Scanning per-paper glossaries from dir: {glossaries_dir}")
        for f in glossaries_dir.glob("extracted_glossary__*.json"):
            stem = f.stem  # e.g. extracted_glossary__2022.acl-long.110
            paper_id = stem.replace("extracted_glossary__", "")
            if paper_id:
                paper_gloss[paper_id] = f
        return paper_gloss

    return {}


def _load_dev_lists(data_root: Path, lang_code: str) -> Tuple[List[str], List[str]]:
    src = data_root / "dev.source"
    tgt = data_root / f"dev.target.{lang_code}"
    if not tgt.is_file():
        tgt = data_root / "dev.target.zh"
    if not src.is_file() or not tgt.is_file():
        raise FileNotFoundError(f"Missing dev lists: {src} or {tgt}")
    src_lines = _read_text(src).splitlines()
    tgt_lines = _read_text(tgt).splitlines()
    if len(src_lines) != len(tgt_lines):
        raise ValueError(f"dev.source and dev.target are not aligned: {len(src_lines)} vs {len(tgt_lines)}")
    return src_lines, tgt_lines


def _subset_lists_for_paper(src_lines: List[str], tgt_lines: List[str], paper_id: str) -> Tuple[List[str], List[str]]:
    out_src: List[str] = []
    out_tgt: List[str] = []
    for s, t in zip(src_lines, tgt_lines):
        pid = _paper_id_from_wav_path(s)
        if pid == paper_id:
            out_src.append(s)
            out_tgt.append(t)
    return out_src, out_tgt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracted-glossaries-dir", default=None, help="Directory with per-paper glossaries")
    ap.add_argument("--extracted-glossary-manifest", default=None, help="Manifest JSON from extract_acl_terms_from_paper_v2.py")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--lang-code", default=DEFAULT_LANG_CODE)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--lists-dirname", default=DEFAULT_LISTS_DIRNAME)
    ap.add_argument("--mapping-json-name", default=DEFAULT_MAPPING_JSON_NAME)
    args = ap.parse_args()

    glossaries_dir = Path(args.extracted_glossaries_dir) if args.extracted_glossaries_dir else Path(DEFAULT_GLOSSARIES_DIR)
    manifest_path = Path(args.extracted_glossary_manifest) if args.extracted_glossary_manifest else Path(DEFAULT_MANIFEST_JSON)
    data_root = Path(args.data_root)
    out_root = Path(args.output_dir)
    lists_dir = out_root / str(args.lists_dirname)

    if not data_root.is_dir():
        _err(f"DATA_ROOT not found: {data_root}")
        return EXIT_CONFIG_ERROR

    # Load per-paper glossaries
    paper_gloss = _load_per_paper_glossaries(
        glossaries_dir if glossaries_dir.is_dir() else None,
        manifest_path if manifest_path.is_file() else None,
    )
    if not paper_gloss:
        _err("No per-paper glossaries found. Check --extracted-glossaries-dir or --extracted-glossary-manifest.")
        return EXIT_DATA_ERROR

    _info(f"Found {len(paper_gloss)} per-paper glossaries")

    # Load dev lists
    _info(f"Loading dev lists from: {data_root}")
    src_lines, tgt_lines = _load_dev_lists(data_root, str(args.lang_code))

    # Determine which papers are in dev.source
    dev_papers: List[str] = []
    for s in src_lines:
        pid = _paper_id_from_wav_path(s)
        if pid and pid in paper_gloss and pid not in dev_papers:
            dev_papers.append(pid)

    if not dev_papers:
        _err("No paper ids from dev.source matched the per-paper glossaries.")
        return EXIT_DATA_ERROR

    mapping: Dict[str, Dict[str, str]] = {}
    lists_dir.mkdir(parents=True, exist_ok=True)

    for paper_id in dev_papers:
        gloss_path = paper_gloss[paper_id]

        sub_src, sub_tgt = _subset_lists_for_paper(src_lines, tgt_lines, paper_id)
        if not sub_src:
            _info(f"Skip paper_id={paper_id}: no examples in dev lists.")
            continue

        src_path = lists_dir / f"dev.source__{paper_id}.txt"
        tgt_path = lists_dir / f"dev.target.{args.lang_code}__{paper_id}.txt"
        portable_sub_src = [_portable_path(p) for p in sub_src]
        _write_text(src_path, "\n".join(portable_sub_src) + "\n")
        _write_text(tgt_path, "\n".join(sub_tgt) + "\n")

        mapping[paper_id] = {
            "paper_id": paper_id,
            "glossary_path": str(gloss_path.resolve()),  # Use original path (absolute)
            "src_list": str(src_path),
            "tgt_list": str(tgt_path),
        }

    mapping_path = out_root / str(args.mapping_json_name)
    _write_json(mapping_path, {"meta": {"lang_code": str(args.lang_code)}, "papers": mapping})
    _info(f"Prepared papers: {len(mapping)}")
    _info(f"Mapping JSON: {mapping_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
