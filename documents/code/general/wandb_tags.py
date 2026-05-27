#!/usr/bin/env python3
"""Utilities for WandB tag hygiene.

WandB currently rejects tags longer than 64 characters.  Launchers in this
repository often compose structured tags from variant names and paths, so tag
length must be handled centrally instead of by ad hoc shell truncation.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, List, Sequence, Tuple

MAX_WANDB_TAG_LEN = 64
TAG_HASH_LEN = 8


def compress_wandb_tag(tag: object, max_len: int = MAX_WANDB_TAG_LEN) -> str:
    """Return a deterministic WandB-safe tag.

    The prefix before the first colon is preserved when possible, so structured
    filters such as ``variant:*`` and ``family:*`` remain usable.  Long values
    are shortened with a stable hash suffix to avoid accidental collisions.
    """

    text = str(tag).strip()
    if not text:
        raise ValueError("WandB tag must be non-empty after stripping.")
    if len(text) <= max_len:
        return text

    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:TAG_HASH_LEN]
    marker = f"__{digest}"
    if len(marker) >= max_len:
        raise ValueError(f"max_len={max_len} is too small for hash marker.")

    prefix = ""
    value = text
    if ":" in text:
        head, tail = text.split(":", 1)
        candidate_prefix = f"{head}:"
        # Preserve normal structured prefixes.  If the prefix itself is too
        # long, fall back to compressing the full tag.
        if 0 < len(candidate_prefix) < max_len - len(marker) - 4:
            prefix = candidate_prefix
            value = tail

    budget = max_len - len(prefix) - len(marker)
    if budget <= 0:
        return text[: max_len - len(marker)] + marker
    return prefix + value[:budget] + marker


def prepare_wandb_tags(
    tags: Iterable[object],
    max_len: int = MAX_WANDB_TAG_LEN,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Compress and de-duplicate tags while preserving order.

    Returns ``(safe_tags, changes)`` where ``changes`` contains ``(old, new)``
    pairs for tags that were shortened.
    """

    safe: List[str] = []
    seen = set()
    changes: List[Tuple[str, str]] = []
    for raw in tags:
        original = str(raw).strip()
        if not original:
            continue
        compressed = compress_wandb_tag(original, max_len=max_len)
        if compressed != original:
            changes.append((original, compressed))
        if compressed not in seen:
            safe.append(compressed)
            seen.add(compressed)
    return safe, changes

