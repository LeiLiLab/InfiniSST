#!/usr/bin/env python3
"""
Sample term_map to reduce its density for ablation studies.

This script works on jsonl datasets with a structure like:
{
  "messages": [{"role": "...", "content": "..."}],
  "audios": [...],
  "gt_terms_by_chunk": [...]  # optional, zh-only
}

Chunk definition:
- A "chunk" corresponds to one user message whose content contains "<audio>".
- If "audios" exists and is a list, its length must exactly match the number of "<audio>" chunks in messages; otherwise the script will raise an error.

term_map encodings supported in message content:
1) Lines format:
   term_map:
   key=value
   key=value
2) Inline JSON dict:
   term_map: {"key": "value", ...}
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ======Configuration=====
# Ground-truth key schema (zh dataset)
GT_TERMS_BY_CHUNK_KEY = "gt_terms_by_chunk"
GT_TERM_KEY = "term"
GT_VALUE_KEY = "zh"
# ======Configuration=====

TERM_MAP_RE = re.compile(r"\bterm_map\s*:", flags=re.IGNORECASE)


def _extract_json_dict_span(s: str) -> Optional[Tuple[Dict[str, Any], int, int]]:
    """
    Try to parse the first JSON dict in s. Returns (dict, start, end) positions in s.
    """
    start = s.find("{")
    if start < 0:
        return None
    dec = json.JSONDecoder()
    try:
        obj, end = dec.raw_decode(s[start:])
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return obj, start, start + end


def parse_term_map_from_content(content: str) -> Dict[str, str]:
    m = TERM_MAP_RE.search(content)
    if not m:
        return {}
    after = content[m.end() :].strip()
    if not after:
        return {}

    # JSON dict format
    span = _extract_json_dict_span(after) if "{" in after else None
    if span is not None:
        d, _s, _e = span
        out: Dict[str, str] = {}
        for k, v in d.items():
            kk = str(k).strip()
            vv = "" if v is None else str(v).strip()
            if kk:
                out[kk] = vv
        return out

    # Lines format
    out: Dict[str, str] = {}
    for raw_line in after.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith(("translation:", "output:", "answer:", "assistant:", "user:")):
            break
        if "=" in line:
            k, v = line.split("=", 1)
            kk = k.strip()
            vv = v.strip()
            if kk:
                out[kk] = vv
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            kk = k.strip()
            vv = v.strip()
            if kk and vv and kk.lower() != "term_map":
                out[kk] = vv
    return out


def strip_and_rebuild_content(content: str, new_term_map: Optional[Dict[str, str]]) -> str:
    """
    Replace the term_map section with new_term_map.
    If new_term_map is None or empty, remove the term_map section entirely.
    Preserve any text before term_map, and preserve trailing remainder for JSON dict format.
    """
    m = TERM_MAP_RE.search(content)
    if not m:
        # No existing term_map section: only append when new_term_map is non-empty.
        if not new_term_map:
            return content
        lines = ["term_map:"]
        for k, v in new_term_map.items():
            lines.append(f"{k}={v}")
        suffix = "\n".join(lines)
        if content.rstrip():
            return (content.rstrip() + "\n\n" + suffix).strip()
        return suffix.strip()

    prefix = content[: m.start()].rstrip()
    after_full = content[m.end() :]
    after = after_full.lstrip()

    remainder = ""
    # If JSON dict format, try to preserve text after the dict
    span = _extract_json_dict_span(after) if "{" in after else None
    if span is not None:
        _d, _s, _e = span
        remainder = after[_e:].lstrip()
    else:
        # Lines format: treat rest as term_map payload (no remainder)
        remainder = ""

    if not new_term_map:
        out = prefix
        if remainder:
            out = (out + "\n" + remainder).strip()
        return out.strip()

    # Rebuild in lines format for consistency
    lines = ["term_map:"]
    for k, v in new_term_map.items():
        lines.append(f"{k}={v}")

    rebuilt = prefix
    if rebuilt:
        rebuilt += "\n\n" + "\n".join(lines)
    else:
        rebuilt = "\n".join(lines)

    if remainder:
        rebuilt = (rebuilt + "\n" + remainder).strip()
    return rebuilt


def iter_user_audio_message_indices(messages: List[Dict[str, Any]]) -> List[int]:
    idxs: List[int] = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str) and "<audio>" in c:
            idxs.append(i)
    return idxs


def sample_dict(d: Dict[str, str], rng: random.Random, entry_sample_ratio: float, max_entries: Optional[int]) -> Dict[str, str]:
    if not d:
        return {}
    keys = list(d.keys())
    # Ratio sampling
    if entry_sample_ratio >= 1.0:
        kept = keys
    else:
        kept = [k for k in keys if rng.random() < entry_sample_ratio]
        if not kept:
            # keep at least 1 if original non-empty and we didn't drop the chunk
            kept = [rng.choice(keys)]
    # Max cap
    if max_entries is not None and len(kept) > max_entries:
        rng.shuffle(kept)
        kept = kept[:max_entries]
    return {k: d[k] for k in kept}


def parse_gt_terms(gt_entry: Any) -> Dict[str, str]:
    """
    Parse one gt chunk entry into a dict {term -> value}.
    Expected formats:
    - [{"term": "...", "zh": "..."}, ...]
    - [{"term": "...", "target": "..."}, ...] (value key may vary; we fall back to empty string)
    """
    if not isinstance(gt_entry, list):
        return {}
    out: Dict[str, str] = {}
    for it in gt_entry:
        if not isinstance(it, dict):
            continue
        term = it.get(GT_TERM_KEY)
        if term is None:
            continue
        k = str(term).strip()
        if not k:
            continue
        v = it.get(GT_VALUE_KEY)
        vv = "" if v is None else str(v).strip()
        out[k] = vv
    return out


def sample_term_map_keep_gt(
    term_map_before: Dict[str, str],
    gt_terms: Dict[str, str],
    rng: random.Random,
    entry_sample_ratio: float,
    max_entries: Optional[int],
) -> Dict[str, str]:
    """
    Keep all GT terms, and sample only non-GT entries.
    This prevents turning positive chunks into all-negative chunks by accident.
    """
    if not term_map_before and not gt_terms:
        return {}

    out: Dict[str, str] = dict(gt_terms)
    non_gt = {k: v for k, v in term_map_before.items() if k not in gt_terms}
    sampled_non_gt = sample_dict(non_gt, rng, entry_sample_ratio, None)

    if max_entries is not None:
        budget = max_entries - len(out)
        if budget <= 0:
            # Keep GT terms only. If GT itself exceeds max_entries, keep all GT and warn.
            if budget < 0:
                print(
                    f"[WARN] GT terms exceed max_entries: gt_terms={len(out)} max_entries={max_entries}. "
                    "Keeping all GT terms and dropping non-GT entries."
                )
            return out
        if len(sampled_non_gt) > budget:
            keys = list(sampled_non_gt.keys())
            rng.shuffle(keys)
            keys = keys[:budget]
            sampled_non_gt = {k: sampled_non_gt[k] for k in keys}

    out.update(sampled_non_gt)
    return out


@dataclass
class RunningStats:
    instances: int = 0
    chunks_total: int = 0
    chunks_with_term_map_before: int = 0
    chunks_with_term_map_after: int = 0
    terms_total_before: int = 0
    terms_total_after: int = 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input jsonl path")
    ap.add_argument("--output", required=True, help="Output jsonl path")
    ap.add_argument("--keep-chunk-ratio", type=float,default=1.0, required=True, help="Keep non-empty term_map for this fraction of chunks")
    ap.add_argument("--entry-sample-ratio", type=float, default=0.5, help="Within kept chunks, keep this fraction of term_map entries")
    ap.add_argument("--max-entries", type=int, default=100, help="Within kept chunks, cap term_map size")
    ap.add_argument("--seed", type=int, default=1, help="Random seed")
    ap.add_argument("--max-lines", type=int, default=None, help="Optional cap for debugging")
    args = ap.parse_args()

    if not (0.0 <= args.keep_chunk_ratio <= 1.0):
        raise ValueError("--keep-chunk-ratio must be in [0,1]")
    if not (0.0 < args.entry_sample_ratio <= 1.0):
        raise ValueError("--entry-sample-ratio must be in (0,1]")
    if args.max_entries is not None and args.max_entries <= 0:
        raise ValueError("--max-entries must be > 0")

    rng = random.Random(args.seed)
    st = RunningStats()

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line_idx, line in enumerate(fin):
            if args.max_lines is not None and line_idx >= args.max_lines:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            st.instances += 1

            messages = obj.get("messages")
            if not isinstance(messages, list):
                messages = []
                obj["messages"] = messages

            audio_idxs = iter_user_audio_message_indices(messages)
            audios = obj.get("audios")
            if isinstance(audios, list):
                expected_chunks = len(audios)
                actual_chunks = len(audio_idxs)
                if expected_chunks != actual_chunks:
                    raise ValueError(
                        f"Chunk count mismatch at line {line_idx+1}: len(audios)={expected_chunks} "
                        f"but found {actual_chunks} user '<audio>' chunks in messages"
                    )
            else:
                expected_chunks = len(audio_idxs)
            st.chunks_total += expected_chunks

            gt = obj.get("gt_terms_by_chunk")
            has_gt = isinstance(gt, list)
            if has_gt and len(gt) < expected_chunks:
                gt.extend([[] for _ in range(expected_chunks - len(gt))])

            # Process each existing chunk message
            for chunk_i, msg_idx in enumerate(audio_idxs):
                m = messages[msg_idx]
                c = m.get("content")
                if not isinstance(c, str):
                    continue

                tm_before = parse_term_map_from_content(c)
                gt_terms: Dict[str, str] = parse_gt_terms(gt[chunk_i]) if has_gt and chunk_i < len(gt) else {}
                n_before = len(tm_before)
                if n_before > 0:
                    st.chunks_with_term_map_before += 1
                    st.terms_total_before += n_before

                # Decide keep/drop at chunk level
                any_terms_before = (n_before > 0) or (len(gt_terms) > 0)
                keep_chunk = any_terms_before and (rng.random() < args.keep_chunk_ratio)
                if not keep_chunk:
                    tm_after = {}
                else:
                    tm_after = sample_term_map_keep_gt(
                        term_map_before=tm_before,
                        gt_terms=gt_terms,
                        rng=rng,
                        entry_sample_ratio=args.entry_sample_ratio,
                        max_entries=args.max_entries,
                    )

                n_after = len(tm_after)
                if n_after > 0:
                    st.chunks_with_term_map_after += 1
                    st.terms_total_after += n_after

                # Rewrite message content
                m["content"] = strip_and_rebuild_content(c, tm_after)

            # No padding: if audios exists, we already asserted exact match above.

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # Print summary (English only)
    before_ratio = st.chunks_with_term_map_before / st.chunks_total if st.chunks_total else 0.0
    after_ratio = st.chunks_with_term_map_after / st.chunks_total if st.chunks_total else 0.0
    before_mean = st.terms_total_before / st.chunks_total if st.chunks_total else 0.0
    after_mean = st.terms_total_after / st.chunks_total if st.chunks_total else 0.0
    print("Done.")
    print(f"instances={st.instances}")
    print(f"chunks_total={st.chunks_total}")
    print(f"chunk_with_term_map_ratio_before={before_ratio:.6f} after={after_ratio:.6f}")
    print(f"mean_terms_per_chunk_before={before_mean:.6f} after={after_mean:.6f}")


if __name__ == "__main__":
    main()


