#!/usr/bin/env python3
"""Patch vLLM 0.13 MoE topk_softmax for PSC V100 fallback.

This is a narrow compatibility patch for environments where vLLM imports the
``_moe_C`` namespace but the ``topk_softmax`` kernel is not registered.  It keeps
the native kernel path when available and falls back to a torch softmax+topk
implementation otherwise.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


MARKER = "# InfiniSST PSC V100 fallback for missing _moe_C.topk_softmax"

NEW_BLOCK = f'''def topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> None:
    {MARKER}
    if hasattr(torch.ops, "_moe_C") and hasattr(torch.ops._moe_C, "topk_softmax"):
        torch.ops._moe_C.topk_softmax(
            topk_weights, topk_ids, token_expert_indices, gating_output, renormalize
        )
        return

    scores = torch.softmax(gating_output, dim=-1)
    weights, indices = torch.topk(
        scores,
        k=topk_weights.shape[1],
        dim=-1,
        sorted=True,
    )
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    topk_weights.copy_(weights.to(dtype=topk_weights.dtype))
    topk_ids.copy_(indices.to(dtype=topk_ids.dtype))
    token_expert_indices.copy_(indices.to(dtype=token_expert_indices.dtype))
'''


def default_custom_ops_path() -> Path:
    candidates = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.extend(Path(conda_prefix).glob("lib/python*/site-packages/vllm/_custom_ops.py"))
    for path in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if path:
            candidates.append(Path(path) / "vllm" / "_custom_ops.py")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate vllm/_custom_ops.py; pass --custom-ops-path explicitly"
    )


def patch_file(path: Path, dry_run: bool = False) -> str:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return f"already_patched\t{path}"

    pattern = re.compile(
        r"def topk_softmax\(\n"
        r"    topk_weights: torch\.Tensor,\n"
        r"    topk_ids: torch\.Tensor,\n"
        r"    token_expert_indices: torch\.Tensor,\n"
        r"    gating_output: torch\.Tensor,\n"
        r"    renormalize: bool(?: = False)?,\n"
        r"\)(?: -> None)?:\n"
        r"    torch\.ops\._moe_C\.topk_softmax\(\n"
        r"        topk_weights, topk_ids, token_expert_indices, gating_output, renormalize\n"
        r"    \)\n",
        flags=re.MULTILINE,
    )
    if not pattern.search(text):
        raise RuntimeError(f"Expected topk_softmax block not found in {path}")

    patched = pattern.sub(NEW_BLOCK, text, count=1)
    backup = path.with_suffix(path.suffix + ".pre_infinisst_topk_fallback.bak")
    if not dry_run:
        if not backup.exists():
            backup.write_text(text, encoding="utf-8")
        path.write_text(patched, encoding="utf-8")
    return f"patched\t{path}\tbackup={backup}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom-ops-path", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    path = args.custom_ops_path or default_custom_ops_path()
    print(patch_file(path, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
