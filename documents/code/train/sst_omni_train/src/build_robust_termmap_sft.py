#!/usr/bin/env python3
"""Post-process retriever term_map SFT data into robustness curricula.

Input is the source-match + retriever-timeline JSONL produced by
``build_retriever_timeline_termmap_sft.py``.  This script does not rerun the
retriever.  It reshapes the per-chunk term_map distribution so Speech LLM SFT
sees the failure modes observed in simuleval:

* empty/sparse no-term chunks;
* clean GT chunks for term adoption;
* realistic retriever chunks;
* dense noisy chunks from large-glossary retrieval;
* optional adversarial translation/false-positive distractors.

The historical LLM-extracted ``gt_terms_by_chunk`` should not be used as the
main target.  Use source-glossary exact-match GT JSONL as input.

When ``--gt-target-match-policy`` is enabled, source-matched GT terms are only
kept as GT/backfill supervision if their target translation appears in the SFT
assistant target text.  Unmatched source terms are then treated like ordinary
retrieved/noisy terms instead of trusted GT terms.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

UNIT_DURATION_SEC = 0.96

SYSTEM_PROMPT_BY_LANG = {
    "zh": (
        "You are a professional simultaneous interpreter. You will be given "
        "chunks of English audio and you need to translate the audio into "
        "Chinese text. Use the 'term_map' as a reference for terminology if "
        "provided. Use only terms that are supported by the speech, and ignore "
        "irrelevant or unsupported term_map entries."
    ),
    "de": (
        "You are a professional simultaneous interpreter. You will be given "
        "chunks of English audio and you need to translate the audio into "
        "German text. Use the 'term_map' as a reference for terminology if "
        "provided. Use only terms that are supported by the speech, and ignore "
        "irrelevant or unsupported term_map entries."
    ),
    "ja": (
        "You are a professional simultaneous interpreter. You will be given "
        "chunks of English audio and you need to translate the audio into "
        "Japanese text. Use the 'term_map' as a reference for terminology if "
        "provided. Use only terms that are supported by the speech, and ignore "
        "irrelevant or unsupported term_map entries."
    ),
}


def _iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{lineno}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at {path}:{lineno}")
            yield lineno, obj


def _term_key(term: str) -> str:
    return " ".join(str(term or "").casefold().split())


def _extract_translation(entry: Mapping[str, Any], lang_code: str) -> str:
    value = entry.get("translation") or entry.get("target_translation") or entry.get(lang_code)
    if value is None and isinstance(entry.get("target_translations"), Mapping):
        value = entry["target_translations"].get(lang_code)
    return str(value or "").strip()


def _audio_user_indices(messages: Sequence[Mapping[str, Any]]) -> List[int]:
    return [
        idx for idx, msg in enumerate(messages)
        if msg.get("role") == "user" and "<audio>" in str(msg.get("content") or "")
    ]


def _assistant_target_text(
    messages: Sequence[Mapping[str, Any]],
    *,
    start: int = 0,
    end: Optional[int] = None,
) -> str:
    stop = len(messages) if end is None else min(len(messages), end)
    return " ".join(
        str(messages[idx].get("content") or "")
        for idx in range(max(0, start), stop)
        if messages[idx].get("role") == "assistant"
    )


def _target_match_supported_terms(
    gt_terms: Sequence[Mapping[str, str]],
    *,
    local_target_text: str,
    full_target_text: str,
    policy: str,
) -> List[Dict[str, str]]:
    if policy == "none":
        return [dict(x) for x in gt_terms]
    if policy == "local4":
        target_text = local_target_text
    elif policy in {"full_ref", "local_or_full"}:
        target_text = full_target_text
    else:
        raise ValueError(f"Unknown gt target match policy: {policy}")
    out: List[Dict[str, str]] = []
    seen = set()
    for item in gt_terms:
        translation = str(item.get("translation") or "").strip()
        key = str(item.get("key") or _term_key(str(item.get("term") or "")))
        if translation and translation in target_text and key and key not in seen:
            seen.add(key)
            out.append(dict(item))
    return out


def _terms_for_jsonl(items: Sequence[Mapping[str, str]], lang_code: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for item in items:
        term = str(item.get("term") or "").strip()
        translation = str(item.get("translation") or "").strip()
        if term and translation:
            out.append({"term": term, lang_code: translation})
    return out


def _parse_term_map(content: str) -> List[Dict[str, str]]:
    if "term_map:" not in content:
        return []
    tail = content.split("term_map:", 1)[1].strip()
    if not tail or tail.upper() == "NONE":
        return []
    out: List[Dict[str, str]] = []
    seen = set()
    for raw in tail.splitlines():
        line = raw.strip()
        if not line or line.upper() == "NONE":
            continue
        # Accept legacy ``src=tgt``, ``[TERM] src => tgt [/TERM]``, and
        # XML-style ``<term>src => tgt</term>``.
        line = re.sub(r"^\[TERM\]\s*", "", line)
        line = re.sub(r"\s*\[/TERM\]$", "", line)
        line = re.sub(r"^<term>\s*", "", line, flags=re.IGNORECASE)
        line = re.sub(r"\s*</term>$", "", line, flags=re.IGNORECASE)
        if "=>" in line:
            term, translation = line.split("=>", 1)
        elif "=" in line:
            term, translation = line.split("=", 1)
        else:
            continue
        term = term.strip()
        translation = translation.strip()
        key = _term_key(term)
        if not term or not translation or not key or key in seen:
            continue
        seen.add(key)
        out.append({"term": term, "translation": translation, "key": key})
    return out


def _normalize_gt_terms(raw_terms: Any, lang_code: str) -> List[Dict[str, str]]:
    if raw_terms is None:
        return []
    if not isinstance(raw_terms, list):
        raise ValueError(f"gt_terms_by_chunk entry must be a list, got {type(raw_terms).__name__}")
    out: List[Dict[str, str]] = []
    seen = set()
    for item in raw_terms:
        if not isinstance(item, Mapping):
            raise ValueError("gt_terms_by_chunk term entry must be an object")
        term = str(item.get("term") or item.get("source") or "").strip()
        translation = _extract_translation(item, lang_code)
        key = _term_key(term)
        if not term or not translation or not key or key in seen:
            continue
        seen.add(key)
        out.append({"term": term, "translation": translation, "key": key})
    return out


def _format_term_map(items: Sequence[Mapping[str, str]], *, style: str) -> str:
    if not items:
        return "<audio>\n\nterm_map:NONE"
    lines = ["<audio>", "", "term_map:"]
    seen = set()
    for item in items:
        term = str(item.get("term") or "").replace("\n", " ").strip()
        translation = str(item.get("translation") or "").replace("\n", " ").strip()
        key = _term_key(term)
        if not term or not translation or not key or key in seen:
            continue
        seen.add(key)
        if style == "tagged":
            lines.append(f"[TERM] {term} => {translation} [/TERM]")
        elif style == "xml_tagged":
            lines.append(f"<term>{term} => {translation}</term>")
        else:
            lines.append(f"{term}={translation}")
    if len(lines) == 3:
        return "<audio>\n\nterm_map:NONE"
    return "\n".join(lines)


def _chunk_multiplier(audio_path: str) -> int:
    # Fast path: the SFT audio filenames normally encode the duration poorly, so
    # use file size only as a last resort would be unsafe.  The caller can use
    # merge_multiplier for row-level logging; per-chunk lm is not required for
    # deterministic sampling.
    return 0


def _rng_for(seed: int, row_key: str, chunk_idx: int, variant: str) -> random.Random:
    h = hashlib.sha256(f"{seed}|{variant}|{row_key}|{chunk_idx}".encode("utf-8")).hexdigest()
    return random.Random(int(h[:16], 16))


def _take_unique(items: Sequence[Mapping[str, str]], n: int) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen = set()
    for item in items:
        term = str(item.get("term") or "").strip()
        translation = str(item.get("translation") or "").strip()
        key = _term_key(term)
        if not term or not translation or not key or key in seen:
            continue
        seen.add(key)
        out.append({"term": term, "translation": translation, "key": key})
        if len(out) >= n:
            break
    return out


def _shuffle_copy(rng: random.Random, items: Sequence[Mapping[str, str]]) -> List[Dict[str, str]]:
    out = [dict(x) for x in items]
    rng.shuffle(out)
    return out


def _translation_swap_adversary(
    rng: random.Random,
    gt_terms: Sequence[Mapping[str, str]],
    non_gt: Sequence[Mapping[str, str]],
) -> List[Dict[str, str]]:
    if not gt_terms or not non_gt:
        return []
    gt = dict(rng.choice(list(gt_terms)))
    wrong = dict(rng.choice(list(non_gt)))
    wrong_translation = str(wrong.get("translation") or "").strip()
    if not wrong_translation or wrong_translation == gt.get("translation"):
        return []
    return [{
        "term": str(gt["term"]),
        "translation": wrong_translation,
        "key": str(gt["key"]),
        "adversarial_type": "translation_swap",
    }]


def _choose_mode(rng: random.Random, has_gt: bool, chunk_idx: int, variant: str) -> str:
    if variant == "refmatch_r95":
        # Calibrated high-GT refmatch curriculum: target roughly 95% GT-term
        # inclusion, closer to the deployed retriever recall than the near-oracle
        # refmatch_higt curriculum.
        if not has_gt:
            r = rng.random()
            if chunk_idx < 3:
                return "empty" if r < 0.90 else "sparse_noise"
            if r < 0.78:
                return "empty"
            if r < 0.96:
                return "sparse_noise"
            return "dense_noise"
        r = rng.random()
        if r < 0.48:
            return "clean_gt_all"
        if r < 0.73:
            return "gt_all_sparse_noise"
        if r < 0.93:
            return "gt_drop_one_sparse_noise"
        return "realistic_gtboost"

    if variant == "refmatch_higt":
        # High-GT refmatch curriculum: after target-match filtering, the
        # remaining GT terms are reliable reference-compatible supervision.
        # Keep no-GT chunks mostly empty/sparse, but make GT chunks include
        # almost all available GT terms so the SFT signal approximates the
        # retriever's observed 90%+ recall regime.
        if not has_gt:
            r = rng.random()
            if chunk_idx < 3:
                return "empty" if r < 0.90 else "sparse_noise"
            if r < 0.78:
                return "empty"
            if r < 0.96:
                return "sparse_noise"
            return "dense_noise"
        r = rng.random()
        if r < 0.62:
            return "clean_gt_all"
        if r < 0.86:
            return "gt_all_sparse_noise"
        if r < 0.97:
            return "realistic_gtboost"
        return "term_preference"

    if variant == "precision":
        # Precision curriculum for zh exact-term regressions: keep realistic
        # retriever provenance, but make supported GT terms salient and keep
        # no-GT/noise exposure sparse.
        if not has_gt:
            r = rng.random()
            if chunk_idx < 3:
                return "empty" if r < 0.90 else "sparse_noise"
            if r < 0.78:
                return "empty"
            if r < 0.96:
                return "sparse_noise"
            return "dense_noise"
        r = rng.random()
        if r < 0.45:
            return "term_preference"
        if r < 0.70:
            return "clean_gt"
        if r < 0.88:
            return "realistic"
        if r < 0.96:
            return "sparse_real"
        return "empty"

    # Force early chunks to have enough empty/sparse supervision.  This targets
    # early no-term English-copy cascades in raw/small-glossary simuleval.
    if chunk_idx < 3 and rng.random() < 0.55:
        return "empty" if rng.random() < 0.75 else "sparse_noise"

    r = rng.random()
    if not has_gt:
        if r < 0.70:
            return "empty"
        if r < 0.88:
            return "sparse_noise"
        return "dense_noise"

    if variant == "adv" and r < 0.10:
        return "adversarial"
    if r < 0.20:
        return "clean_gt"
    if r < 0.40:
        return "sparse_real"
    if r < 0.70:
        return "realistic"
    if r < 0.88:
        return "partial_noisy"
    return "term_critical"


def _build_items_for_mode(
    *,
    rng: random.Random,
    mode: str,
    gt_terms: Sequence[Mapping[str, str]],
    retrieved: Sequence[Mapping[str, str]],
    variant: str,
    target_supported_gt_terms: Sequence[Mapping[str, str]] = (),
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    gt_by_key = {x["key"]: dict(x) for x in gt_terms}
    retrieved_gt: List[Dict[str, str]] = []
    non_gt: List[Dict[str, str]] = []
    for item in retrieved:
        key = str(item.get("key") or _term_key(str(item.get("term") or "")))
        if key in gt_by_key:
            cur = dict(gt_by_key[key])
            cur["retrieved_gt"] = "1"
            retrieved_gt.append(cur)
        else:
            non_gt.append(dict(item))

    meta = {
        "gt_backfilled": 0,
        "translation_swaps": 0,
        "false_positive_terms": 0,
    }

    if mode == "empty":
        return [], meta

    if mode == "sparse_noise":
        items = _take_unique(_shuffle_copy(rng, non_gt), rng.choice([1, 1, 2]))
        meta["false_positive_terms"] = len(items)
        return items, meta

    if mode == "dense_noise":
        cap = rng.randint(5, 10)
        items = _take_unique(_shuffle_copy(rng, non_gt), cap)
        meta["false_positive_terms"] = len(items)
        return items, meta

    if mode == "clean_gt":
        cap = rng.choice([1, 2, 2, 3])
        items = _take_unique(gt_terms, cap)
        meta["gt_backfilled"] = max(0, len(items) - len(retrieved_gt))
        return items, meta

    if mode == "clean_gt_all":
        items = _take_unique(gt_terms, 16)
        meta["gt_backfilled"] = max(0, len(items) - len(retrieved_gt))
        return items, meta

    if mode == "gt_all_sparse_noise":
        gt_items = _take_unique(gt_terms, 16)
        noise = _take_unique(_shuffle_copy(rng, non_gt), rng.choice([0, 0, 1, 1, 2]))
        items = _take_unique(gt_items + noise, 18)
        meta["gt_backfilled"] = max(0, len(gt_items) - len(retrieved_gt))
        meta["false_positive_terms"] = len(noise)
        return items, meta

    if mode == "gt_drop_one_sparse_noise":
        shuffled_gt = _shuffle_copy(rng, gt_terms)
        if len(shuffled_gt) > 1:
            gt_items = _take_unique(shuffled_gt[:-1], 16)
        else:
            gt_items = _take_unique(shuffled_gt, 16)
        noise = _take_unique(_shuffle_copy(rng, non_gt), rng.choice([0, 0, 1]))
        items = _take_unique(gt_items + noise, 18)
        meta["gt_backfilled"] = max(0, len(gt_items) - len(retrieved_gt))
        meta["false_positive_terms"] = len(noise)
        return items, meta

    if mode == "sparse_real":
        cap = rng.choice([1, 2, 3])
        ordered = list(retrieved_gt) + _shuffle_copy(rng, non_gt)
        items = _take_unique(ordered, cap)
        meta["false_positive_terms"] = sum(1 for x in items if x["key"] not in gt_by_key)
        return items, meta

    if mode == "realistic":
        cap = rng.choice([3, 4, 5, 6])
        # Keep original rank order to preserve real retriever priority.
        items = _take_unique(retrieved, cap)
        for item in items:
            if item["key"] in gt_by_key:
                item["translation"] = gt_by_key[item["key"]]["translation"]
        meta["false_positive_terms"] = sum(1 for x in items if x["key"] not in gt_by_key)
        return items, meta

    if mode == "realistic_gtboost":
        # Preserve the real retriever's ordering for visible negatives, then
        # backfill any missed reference-compatible GT terms.
        retrieved_prefix = _take_unique(retrieved, rng.choice([4, 5, 6, 8]))
        for item in retrieved_prefix:
            if item["key"] in gt_by_key:
                item["translation"] = gt_by_key[item["key"]]["translation"]
        seen = {x["key"] for x in retrieved_prefix}
        missing_gt = [dict(x) for x in gt_terms if x["key"] not in seen]
        noise = _take_unique(
            [x for x in retrieved_prefix if x["key"] not in gt_by_key],
            rng.choice([0, 1, 2]),
        )
        gt_items = _take_unique(
            [x for x in retrieved_prefix if x["key"] in gt_by_key] + missing_gt,
            16,
        )
        items = _take_unique(gt_items + noise, 18)
        meta["gt_backfilled"] = max(0, len(gt_items) - len(retrieved_gt))
        meta["false_positive_terms"] = sum(1 for x in items if x["key"] not in gt_by_key)
        return items, meta

    if mode == "partial_noisy":
        keep_gt = _take_unique(retrieved_gt, rng.choice([0, 1, 1, 2]))
        noise = _take_unique(_shuffle_copy(rng, non_gt), rng.randint(3, 8))
        items = _take_unique(keep_gt + noise, 10)
        meta["false_positive_terms"] = sum(1 for x in items if x["key"] not in gt_by_key)
        return items, meta

    if mode == "term_critical":
        gt_cap = rng.choice([1, 2, 3, 4])
        gt_items = _take_unique(gt_terms, gt_cap)
        noise = _take_unique(_shuffle_copy(rng, non_gt), rng.choice([0, 1, 2]))
        items = _take_unique(gt_items + noise, 6)
        meta["gt_backfilled"] = max(0, len(gt_items) - len(retrieved_gt))
        meta["false_positive_terms"] = len(noise)
        return items, meta

    if mode == "term_preference":
        # Prefer GT terms whose target translations appear in the local SFT
        # target window.  These examples provide direct supervision that the
        # term_map target spelling is the preferred rendering, while still
        # allowing a small amount of realistic retriever noise.
        preferred_source = target_supported_gt_terms or gt_terms
        gt_cap = rng.choice([4, 5, 6, 8])
        gt_items = _take_unique(preferred_source, gt_cap)
        if not gt_items:
            gt_items = _take_unique(gt_terms, gt_cap)
        noise = _take_unique(_shuffle_copy(rng, non_gt), rng.choice([0, 0, 1]))
        items = _take_unique(gt_items + noise, 9)
        meta["gt_backfilled"] = max(0, len(gt_items) - len(retrieved_gt))
        meta["false_positive_terms"] = len(noise)
        return items, meta

    if mode == "adversarial":
        adv = _translation_swap_adversary(rng, gt_terms, non_gt)
        if adv:
            noise = _take_unique(_shuffle_copy(rng, non_gt), rng.randint(2, 5))
            items = _take_unique(adv + noise, 6)
            meta["translation_swaps"] = 1
            meta["false_positive_terms"] = max(0, len(items) - 1)
            return items, meta
        noise = _take_unique(_shuffle_copy(rng, non_gt), rng.randint(4, 8))
        meta["false_positive_terms"] = len(noise)
        return noise, meta

    raise ValueError(f"Unknown mode: {mode}")


def _update_stats(
    stats: Dict[str, Any],
    *,
    mode: str,
    gt_terms: Sequence[Mapping[str, str]],
    items: Sequence[Mapping[str, str]],
    meta: Mapping[str, int],
) -> None:
    gt_keys = {x["key"] for x in gt_terms}
    item_keys = {x["key"] for x in items}
    gt_hits = len(gt_keys & item_keys)
    map_size = len(items)
    has_gt = bool(gt_terms)

    stats["chunks"] += 1
    stats["gt_terms_total"] += len(gt_terms)
    stats["gt_terms_in_term_map"] += gt_hits
    stats["term_map_entries_total"] += map_size
    stats["term_map_non_gt_entries"] += max(0, map_size - gt_hits)
    stats["gt_backfilled_entries"] += int(meta.get("gt_backfilled", 0))
    stats["translation_swap_entries"] += int(meta.get("translation_swaps", 0))
    stats["false_positive_entries"] += int(meta.get("false_positive_terms", 0))
    stats["mode_counts"][mode] += 1
    stats["term_map_size_hist"][str(map_size)] += 1
    if has_gt:
        stats["gt_chunks"] += 1
        if gt_hits:
            stats["gt_chunks_any_term_in_map"] += 1
        if gt_hits == len(gt_terms):
            stats["gt_chunks_all_terms_in_map"] += 1
    else:
        stats["no_gt_chunks"] += 1
        if map_size:
            stats["no_gt_nonempty_term_map_chunks"] += 1


def build(args: argparse.Namespace) -> Dict[str, Any]:
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.parent.mkdir(parents=True, exist_ok=True)
    if args.sample_json:
        args.sample_json.parent.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, Any] = {
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "variant": args.variant,
        "term_map_style": args.term_map_style,
        "gt_target_match_policy": args.gt_target_match_policy,
        "lang_code": args.lang_code,
        "seed": args.seed,
        "rows_seen": 0,
        "rows_written": 0,
        "chunks": 0,
        "gt_chunks": 0,
        "no_gt_chunks": 0,
        "raw_gt_chunks": 0,
        "raw_gt_terms_total": 0,
        "target_match_kept_gt_terms": 0,
        "target_match_dropped_gt_terms": 0,
        "target_match_kept_gt_chunks": 0,
        "gt_terms_total": 0,
        "gt_terms_in_term_map": 0,
        "gt_chunks_any_term_in_map": 0,
        "gt_chunks_all_terms_in_map": 0,
        "term_map_entries_total": 0,
        "term_map_non_gt_entries": 0,
        "gt_backfilled_entries": 0,
        "translation_swap_entries": 0,
        "false_positive_entries": 0,
        "no_gt_nonempty_term_map_chunks": 0,
        "mode_counts": Counter(),
        "term_map_size_hist": Counter(),
        "dropped_rows": 0,
        "dropped_reasons": Counter(),
    }
    samples: List[Dict[str, Any]] = []

    with args.output_jsonl.open("w", encoding="utf-8") as out:
        for lineno, obj_in in _iter_jsonl(args.input_jsonl):
            stats["rows_seen"] += 1
            try:
                obj = copy.deepcopy(obj_in)
                messages = obj.get("messages")
                audios = obj.get("audios")
                gt_by_chunk = obj.get("gt_terms_by_chunk")
                if not isinstance(messages, list) or not messages:
                    raise ValueError("missing non-empty messages")
                if not isinstance(audios, list) or not audios:
                    raise ValueError("missing non-empty audios")
                if not isinstance(gt_by_chunk, list):
                    raise ValueError("missing list gt_terms_by_chunk")
                audio_idxs = _audio_user_indices(messages)
                if len(audio_idxs) != len(audios):
                    raise ValueError(f"audio user messages={len(audio_idxs)} audios={len(audios)}")
                if len(gt_by_chunk) != len(audios):
                    raise ValueError(f"gt_terms_by_chunk={len(gt_by_chunk)} audios={len(audios)}")

                if messages[0].get("role") == "system":
                    messages[0]["content"] = SYSTEM_PROMPT_BY_LANG[args.lang_code]

                row_key = str(obj.get("utter_id") or lineno)
                full_target_text = _assistant_target_text(messages)
                filtered_gt_by_chunk: List[List[Dict[str, str]]] = []
                for chunk_idx, msg_idx in enumerate(audio_idxs):
                    content = str(messages[msg_idx].get("content") or "")
                    retrieved = _parse_term_map(content)
                    raw_gt_terms = _normalize_gt_terms(gt_by_chunk[chunk_idx], args.lang_code)
                    local_target_text = _assistant_target_text(
                        messages,
                        start=msg_idx + 1,
                        end=msg_idx + 5,
                    )
                    gt_terms = _target_match_supported_terms(
                        raw_gt_terms,
                        local_target_text=local_target_text,
                        full_target_text=full_target_text,
                        policy=args.gt_target_match_policy,
                    )
                    target_supported_gt_terms = [
                        x for x in gt_terms
                        if str(x.get("translation") or "").strip()
                        and str(x.get("translation") or "").strip() in local_target_text
                    ]
                    stats["raw_gt_terms_total"] += len(raw_gt_terms)
                    if raw_gt_terms:
                        stats["raw_gt_chunks"] += 1
                    stats["target_match_kept_gt_terms"] += len(gt_terms)
                    stats["target_match_dropped_gt_terms"] += len(raw_gt_terms) - len(gt_terms)
                    if gt_terms:
                        stats["target_match_kept_gt_chunks"] += 1
                    filtered_gt_by_chunk.append(_terms_for_jsonl(gt_terms, args.lang_code))
                    rng = _rng_for(args.seed, row_key, chunk_idx, args.variant)
                    mode = _choose_mode(rng, bool(gt_terms), chunk_idx, args.variant)
                    items, meta = _build_items_for_mode(
                        rng=rng,
                        mode=mode,
                        gt_terms=gt_terms,
                        retrieved=retrieved,
                        variant=args.variant,
                        target_supported_gt_terms=target_supported_gt_terms,
                    )
                    messages[msg_idx]["content"] = _format_term_map(items, style=args.term_map_style)
                    _update_stats(stats, mode=mode, gt_terms=gt_terms, items=items, meta=meta)
                    if len(samples) < args.sample_count and (items or gt_terms):
                        samples.append({
                            "line": lineno,
                            "utter_id": obj.get("utter_id"),
                            "chunk_idx": chunk_idx,
                            "mode": mode,
                            "gt_terms": gt_terms[:8],
                            "target_supported_gt_terms": target_supported_gt_terms[:8],
                            "term_map": items[:12],
                            "term_map_style": args.term_map_style,
                        })

                if args.gt_target_match_policy != "none":
                    obj["gt_terms_by_chunk"] = filtered_gt_by_chunk
                obj["robust_termmap_sft_policy"] = {
                    "version": "v5" if args.gt_target_match_policy != "none" else "v3",
                    "variant": args.variant,
                    "term_map_style": args.term_map_style,
                    "gt_target_match_policy": args.gt_target_match_policy,
                    "seed": args.seed,
                    "source_policy": obj.get("retriever_timeline_termmap_policy"),
                    "gt_source": "source-glossary exact-match gt_terms_by_chunk",
                    "purpose": (
                        "deployment-stress curriculum for empty/sparse term_map, "
                        "real retriever term_map, clean GT adoption, and noisy large-glossary robustness"
                    ),
                }
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                stats["rows_written"] += 1
            except Exception as exc:
                if args.drop_bad_rows:
                    stats["dropped_rows"] += 1
                    stats["dropped_reasons"][str(exc).splitlines()[0][:200]] += 1
                    continue
                raise RuntimeError(f"Failed processing {args.input_jsonl}:{lineno}: {exc}") from exc

    stats["mode_counts"] = dict(stats["mode_counts"])
    stats["term_map_size_hist"] = dict(stats["term_map_size_hist"])
    stats["dropped_reasons"] = dict(stats["dropped_reasons"])
    stats["gt_term_in_term_map_rate"] = (
        stats["gt_terms_in_term_map"] / stats["gt_terms_total"]
        if stats["gt_terms_total"] else 0.0
    )
    stats["target_match_kept_gt_term_rate"] = (
        stats["target_match_kept_gt_terms"] / stats["raw_gt_terms_total"]
        if stats["raw_gt_terms_total"] else 0.0
    )
    stats["target_match_kept_gt_chunk_rate"] = (
        stats["target_match_kept_gt_chunks"] / stats["raw_gt_chunks"]
        if stats["raw_gt_chunks"] else 0.0
    )
    stats["gt_chunk_any_term_in_map_rate"] = (
        stats["gt_chunks_any_term_in_map"] / stats["gt_chunks"]
        if stats["gt_chunks"] else 0.0
    )
    stats["gt_chunk_all_terms_in_map_rate"] = (
        stats["gt_chunks_all_terms_in_map"] / stats["gt_chunks"]
        if stats["gt_chunks"] else 0.0
    )
    stats["no_gt_nonempty_term_map_rate"] = (
        stats["no_gt_nonempty_term_map_chunks"] / stats["no_gt_chunks"]
        if stats["no_gt_chunks"] else 0.0
    )
    stats["avg_term_map_entries_per_chunk"] = (
        stats["term_map_entries_total"] / stats["chunks"] if stats["chunks"] else 0.0
    )
    stats["avg_non_gt_entries_per_chunk"] = (
        stats["term_map_non_gt_entries"] / stats["chunks"] if stats["chunks"] else 0.0
    )

    args.stats_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.sample_json:
        args.sample_json.write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--stats-json", type=Path, required=True)
    parser.add_argument("--sample-json", type=Path)
    parser.add_argument("--lang-code", choices=sorted(SYSTEM_PROMPT_BY_LANG), default="zh")
    parser.add_argument("--variant", choices=["real", "tagged", "adv", "precision", "refmatch_higt", "refmatch_r95"], required=True)
    parser.add_argument("--term-map-style", choices=["plain", "tagged", "xml_tagged"], default="plain")
    parser.add_argument(
        "--gt-target-match-policy",
        choices=["none", "local4", "full_ref", "local_or_full"],
        default="none",
        help=(
            "Filter source-matched GT terms before GT/backfill supervision. "
            "Use full_ref for V5 refmatch data; unmatched source terms remain ordinary retriever/noise terms."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--sample-count", type=int, default=80)
    parser.add_argument("--drop-bad-rows", action="store_true")
    args = parser.parse_args()
    if args.variant == "tagged" and args.term_map_style == "plain":
        raise ValueError("--variant tagged requires a tagged term-map style")
    if args.variant != "tagged" and args.term_map_style == "tagged":
        raise ValueError("--term-map-style tagged is only allowed for --variant tagged")
    if args.term_map_style == "xml_tagged" and args.variant not in {"tagged", "refmatch_r95"}:
        raise ValueError("--term-map-style xml_tagged is only allowed for tagged or refmatch_r95 variants")
    return args


def main() -> None:
    args = parse_args()
    stats = build(args)
    print(json.dumps({
        "output_jsonl": str(args.output_jsonl),
        "rows_written": stats["rows_written"],
        "chunks": stats["chunks"],
        "mode_counts": stats["mode_counts"],
        "gt_target_match_policy": stats["gt_target_match_policy"],
        "target_match_kept_gt_term_rate": stats["target_match_kept_gt_term_rate"],
        "target_match_kept_gt_chunk_rate": stats["target_match_kept_gt_chunk_rate"],
        "gt_term_in_term_map_rate": stats["gt_term_in_term_map_rate"],
        "no_gt_nonempty_term_map_rate": stats["no_gt_nonempty_term_map_rate"],
        "avg_term_map_entries_per_chunk": stats["avg_term_map_entries_per_chunk"],
        "translation_swap_entries": stats["translation_swap_entries"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
