#!/usr/bin/env python3
"""Build denoising term-map SFT data from retriever result JSONL.

This script keeps the cap16 retriever branch as the source of truth, but
rebuilds user-side term maps with smaller mixed budgets and score-aware
dropout for non-GT terms.  GT terms are always preserved; unsupported
retrieved terms are kept only as noise exposure, not as assistant-side tag
targets.
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


SYSTEM_PROMPTS = {
    "zh": (
        "You are a professional simultaneous interpreter. "
        "You will be given chunks of English audio and you need to translate the audio into Chinese text. "
        "Use the 'term_map' as a reference for terminology if provided."
    ),
    "de": (
        "You are a professional simultaneous interpreter. "
        "You will be given chunks of English audio and you need to translate the audio into German text. "
        "Use the 'term_map' as a reference for terminology if provided."
    ),
    "ja": (
        "You are a professional simultaneous interpreter. "
        "You will be given chunks of English audio and you need to translate the audio into Japanese text. "
        "Use the 'term_map' as a reference for terminology if provided."
    ),
}


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            yield line_no, obj


def term_key(value: Any) -> str:
    return str(value or "").strip().casefold()


def term_text(value: Any) -> str:
    return str(value or "").strip()


def extract_translation(entry: Mapping[str, Any], lang: str) -> str:
    value = (
        entry.get("zh")
        or entry.get(lang)
        or entry.get("translation")
        or entry.get("target_translation")
    )
    if value is None and isinstance(entry.get("target_translations"), Mapping):
        value = entry["target_translations"].get(lang)
    return term_text(value)


def audio_user_indices(messages: Sequence[Mapping[str, Any]]) -> List[int]:
    return [
        idx
        for idx, msg in enumerate(messages)
        if msg.get("role") == "user" and str(msg.get("content") or "").startswith("<audio>")
    ]


def future_assistant_text(messages: Sequence[Mapping[str, Any]], start_msg_idx: int) -> str:
    parts: List[str] = []
    for msg in messages[start_msg_idx + 1 :]:
        if msg.get("role") == "assistant":
            parts.append(str(msg.get("content") or ""))
    return "\n".join(parts)


def appears_in_future(translation: str, future_text: str, min_chars: int) -> bool:
    needle = term_text(translation)
    if len("".join(needle.split())) < min_chars:
        return False
    return needle in future_text


def format_term_map(items: Sequence[Mapping[str, Any]]) -> str:
    lines = ["term_map:"]
    seen = set()
    for item in items:
        key = term_key(item.get("term"))
        if not key or key in seen:
            continue
        seen.add(key)
        term = term_text(item.get("term"))
        trans = term_text(item.get("translation"))
        if term and trans:
            lines.append(f"{term}={trans}")
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def parse_csv_ints(value: str) -> List[int]:
    out = [int(x.strip()) for x in str(value).split(",") if x.strip()]
    if not out:
        raise ValueError("expected at least one integer budget")
    if any(x <= 0 for x in out):
        raise ValueError(f"budgets must be positive: {out}")
    return out


def parse_csv_floats(value: str) -> List[float]:
    out = [float(x.strip()) for x in str(value).split(",") if x.strip()]
    if not out:
        raise ValueError("expected at least one float weight")
    if any(x < 0 for x in out) or sum(out) <= 0:
        raise ValueError(f"weights must be non-negative and nonzero: {out}")
    return out


def choose_budget(rng: random.Random, budgets: Sequence[int], weights: Sequence[float]) -> int:
    return int(rng.choices(list(budgets), weights=list(weights), k=1)[0])


def keep_prob_for_candidate(
    *,
    score: Optional[float],
    supported_by_future: bool,
    args: argparse.Namespace,
) -> Tuple[float, str]:
    if supported_by_future:
        return args.supported_non_gt_keep_prob, "supported_non_gt"
    if score is None:
        return args.missing_score_keep_prob, "missing_score_unsupported"
    if score < args.low_score_cutoff:
        return args.low_score_keep_prob, "low_score_unsupported"
    if score < args.mid_score_cutoff:
        return args.mid_score_keep_prob, "mid_score_unsupported"
    return args.high_score_keep_prob, "high_score_unsupported"


def build_chunk_map(
    *,
    gt_terms: Sequence[Mapping[str, Any]],
    ret_terms: Sequence[Mapping[str, Any]],
    future_text: str,
    rng: random.Random,
    args: argparse.Namespace,
    stats: Counter,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    gt_by_key: Dict[str, Dict[str, Any]] = {}
    for gt in gt_terms:
        key = term_key(gt.get("term"))
        trans = extract_translation(gt, args.target_lang)
        term = term_text(gt.get("term"))
        if key and term and trans:
            gt_by_key[key] = {
                "term": term,
                "translation": trans,
                "role": "gt",
                "score": None,
            }

    has_gt = bool(gt_by_key)
    budget = choose_budget(rng, args.budget_choices, args.budget_weights)
    if has_gt:
        effective_budget = max(budget, len(gt_by_key))
    else:
        if rng.random() < args.no_gt_empty_prob:
            stats["chunks_no_gt_emptied_by_policy"] += 1
            return [], {
                "budget": min(budget, args.no_gt_max_terms),
                "has_gt": False,
                "kept": 0,
                "gt_count": 0,
                "dropped": 0,
            }
        effective_budget = min(budget, args.no_gt_max_terms)

    combined: List[Dict[str, Any]] = []
    seen = set()
    dropped = Counter()

    for ret in ret_terms:
        key = term_key(ret.get("term"))
        term = term_text(ret.get("term"))
        trans = extract_translation(ret, args.target_lang)
        if not key or not term or not trans or key in seen:
            continue
        seen.add(key)
        score_value = ret.get("score")
        score = float(score_value) if score_value is not None else None
        if key in gt_by_key:
            item = dict(gt_by_key[key])
            item["score"] = score
            item["source"] = "retriever_gt"
            combined.append(item)
            stats["gt_seen_in_retriever"] += 1
            continue

        supported = appears_in_future(trans, future_text, args.min_target_chars)
        prob, reason = keep_prob_for_candidate(
            score=score,
            supported_by_future=supported,
            args=args,
        )
        if rng.random() <= prob:
            combined.append(
                {
                    "term": term,
                    "translation": trans,
                    "role": reason,
                    "score": score,
                    "source": "retriever_non_gt",
                }
            )
            stats[f"kept_{reason}"] += 1
        else:
            dropped[reason] += 1
            stats[f"dropped_{reason}"] += 1

    for key, gt in gt_by_key.items():
        if key in seen:
            continue
        insert_at = rng.randint(0, len(combined))
        backfilled = dict(gt)
        backfilled["source"] = "gt_backfill_missing_from_retriever"
        combined.insert(insert_at, backfilled)
        seen.add(key)
        stats["gt_backfilled_missing_from_retriever"] += 1

    if len(combined) > effective_budget:
        kept: List[Dict[str, Any]] = list(combined)
        idx = len(kept) - 1
        while len(kept) > effective_budget and idx >= 0:
            if kept[idx].get("role") != "gt":
                removed = kept.pop(idx)
                dropped[str(removed.get("role") or "over_budget")] += 1
                stats[f"dropped_over_budget_{removed.get('role') or 'unknown'}"] += 1
            idx -= 1
        combined = kept

    gt_kept = sum(1 for item in combined if item.get("role") == "gt")
    if gt_kept != len(gt_by_key):
        raise RuntimeError(f"GT term loss: kept={gt_kept} expected={len(gt_by_key)}")

    rng.shuffle(combined) if args.shuffle_final_terms else None
    meta = {
        "budget": effective_budget,
        "raw_budget_choice": budget,
        "has_gt": has_gt,
        "gt_count": len(gt_by_key),
        "kept": len(combined),
        "dropped": sum(dropped.values()),
        "dropped_by_reason": dict(dropped),
    }
    return combined, meta


def process_file(args: argparse.Namespace) -> Dict[str, Any]:
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    stats_path = Path(args.stats_json)
    sample_path = Path(args.sample_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.parent.mkdir(parents=True, exist_ok=True)

    stats: Counter = Counter()
    size_hist: Counter = Counter()
    budget_hist: Counter = Counter()
    samples: List[Dict[str, Any]] = []

    rng = random.Random(args.seed)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=str(output_path.parent), suffix=".tmp"
    ) as tmp:
        tmp_path = Path(tmp.name)
        for line_no, obj in iter_jsonl(input_path):
            messages = obj.get("messages")
            audios = obj.get("audios")
            gt_by_chunk = obj.get("gt_terms_by_chunk")
            ret_by_chunk = obj.get("retriever_results_by_chunk")
            if not isinstance(messages, list) or not isinstance(audios, list):
                raise ValueError(f"missing messages/audios at {input_path}:{line_no}")
            if not isinstance(gt_by_chunk, list) or not isinstance(ret_by_chunk, list):
                raise ValueError(f"missing gt_terms_by_chunk/retriever_results_by_chunk at {input_path}:{line_no}")
            user_indices = audio_user_indices(messages)
            if len(user_indices) != len(audios):
                raise ValueError(f"user/audio count mismatch at {input_path}:{line_no}")
            if len(gt_by_chunk) != len(audios) or len(ret_by_chunk) != len(audios):
                raise ValueError(f"gt/retriever/audio count mismatch at {input_path}:{line_no}")

            if messages and messages[0].get("role") == "system":
                messages[0]["content"] = SYSTEM_PROMPTS[args.target_lang]

            chunk_metas: List[Dict[str, Any]] = []
            for chunk_idx, msg_idx in enumerate(user_indices):
                gt_terms = gt_by_chunk[chunk_idx]
                ret_terms = ret_by_chunk[chunk_idx]
                if not isinstance(gt_terms, list) or not isinstance(ret_terms, list):
                    raise ValueError(f"bad chunk lists at {input_path}:{line_no}:{chunk_idx}")
                future_text = future_assistant_text(messages, msg_idx)
                selected, meta = build_chunk_map(
                    gt_terms=gt_terms,
                    ret_terms=ret_terms,
                    future_text=future_text,
                    rng=rng,
                    args=args,
                    stats=stats,
                )
                term_map = format_term_map(selected)
                messages[msg_idx]["content"] = f"<audio>\n\n{term_map}" if term_map else "<audio>"

                stats["total_chunks"] += 1
                stats["chunks_with_gt"] += int(bool(gt_terms))
                stats["chunks_without_gt"] += int(not gt_terms)
                stats["chunks_with_term_map"] += int(bool(selected))
                stats["chunks_empty_term_map"] += int(not selected)
                stats["gt_total"] += len(gt_terms)
                stats["term_map_entries_total"] += len(selected)
                stats["non_gt_entries_total"] += sum(1 for item in selected if item.get("role") != "gt")
                size_hist[len(selected)] += 1
                budget_hist[meta["budget"]] += 1
                chunk_metas.append(meta)

                if len(samples) < args.sample_count and (gt_terms or selected):
                    samples.append(
                        {
                            "line_no": line_no,
                            "chunk_idx": chunk_idx,
                            "gt_terms": [
                                {
                                    "term": term_text(gt.get("term")),
                                    "translation": extract_translation(gt, args.target_lang),
                                }
                                for gt in gt_terms
                            ],
                            "selected": selected,
                            "meta": meta,
                        }
                    )

            obj.pop("retriever_results_by_chunk", None)
            obj.pop("chunk_metadata", None)
            obj["denoise_budget_policy"] = {
                "version": "cap16_denoise_budget_v1",
                "source": str(input_path),
                "target_lang": args.target_lang,
                "budget_choices": args.budget_choices,
                "budget_weights": args.budget_weights,
                "no_gt_max_terms": args.no_gt_max_terms,
                "no_gt_empty_prob": args.no_gt_empty_prob,
                "low_score_cutoff": args.low_score_cutoff,
                "mid_score_cutoff": args.mid_score_cutoff,
                "low_score_keep_prob": args.low_score_keep_prob,
                "mid_score_keep_prob": args.mid_score_keep_prob,
                "high_score_keep_prob": args.high_score_keep_prob,
                "supported_non_gt_keep_prob": args.supported_non_gt_keep_prob,
                "missing_score_keep_prob": args.missing_score_keep_prob,
                "min_target_chars": args.min_target_chars,
                "shuffle_final_terms": args.shuffle_final_terms,
            }
            obj["denoise_budget_chunk_metadata"] = chunk_metas
            tmp.write(json.dumps(obj, ensure_ascii=False) + "\n")
            stats["total_rows"] += 1

    tmp_path.replace(output_path)

    total_chunks = max(1, stats["total_chunks"])
    summary = {
        "input_jsonl": str(input_path),
        "output_jsonl": str(output_path),
        "seed": args.seed,
        "stats": dict(stats),
        "rates": {
            "chunks_with_gt": stats["chunks_with_gt"] / total_chunks,
            "chunks_with_term_map": stats["chunks_with_term_map"] / total_chunks,
            "chunks_empty_term_map": stats["chunks_empty_term_map"] / total_chunks,
            "avg_term_map_entries_per_chunk": stats["term_map_entries_total"] / total_chunks,
            "avg_non_gt_entries_per_chunk": stats["non_gt_entries_total"] / total_chunks,
        },
        "term_map_size_hist": {str(k): v for k, v in sorted(size_hist.items())},
        "budget_hist": {str(k): v for k, v in sorted(budget_hist.items())},
        "policy": {
            "version": "cap16_denoise_budget_v1",
            "budget_choices": args.budget_choices,
            "budget_weights": args.budget_weights,
            "no_gt_max_terms": args.no_gt_max_terms,
            "no_gt_empty_prob": args.no_gt_empty_prob,
            "score_dropout": {
                "low_score_cutoff": args.low_score_cutoff,
                "mid_score_cutoff": args.mid_score_cutoff,
                "low_score_keep_prob": args.low_score_keep_prob,
                "mid_score_keep_prob": args.mid_score_keep_prob,
                "high_score_keep_prob": args.high_score_keep_prob,
                "supported_non_gt_keep_prob": args.supported_non_gt_keep_prob,
                "missing_score_keep_prob": args.missing_score_keep_prob,
            },
        },
    }
    stats_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sample_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--stats-json", required=True)
    parser.add_argument("--sample-json", required=True)
    parser.add_argument("--target-lang", choices=sorted(SYSTEM_PROMPTS), required=True)
    parser.add_argument("--budget-choices", type=parse_csv_ints, default=parse_csv_ints("6,8,10"))
    parser.add_argument("--budget-weights", type=parse_csv_floats, default=parse_csv_floats("0.45,0.35,0.20"))
    parser.add_argument("--no-gt-max-terms", type=int, default=4)
    parser.add_argument("--no-gt-empty-prob", type=float, default=0.35)
    parser.add_argument("--low-score-cutoff", type=float, default=0.82)
    parser.add_argument("--mid-score-cutoff", type=float, default=0.85)
    parser.add_argument("--low-score-keep-prob", type=float, default=0.25)
    parser.add_argument("--mid-score-keep-prob", type=float, default=0.60)
    parser.add_argument("--high-score-keep-prob", type=float, default=0.90)
    parser.add_argument("--supported-non-gt-keep-prob", type=float, default=0.85)
    parser.add_argument("--missing-score-keep-prob", type=float, default=0.50)
    parser.add_argument("--min-target-chars", type=int, default=2)
    parser.add_argument("--shuffle-final-terms", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-count", type=int, default=200)
    args = parser.parse_args()

    if len(args.budget_choices) != len(args.budget_weights):
        raise SystemExit("--budget-choices and --budget-weights must have equal length")
    if args.no_gt_max_terms <= 0:
        raise SystemExit("--no-gt-max-terms must be positive")
    for name in (
        "no_gt_empty_prob",
        "low_score_keep_prob",
        "mid_score_keep_prob",
        "high_score_keep_prob",
        "supported_non_gt_keep_prob",
        "missing_score_keep_prob",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be in [0,1], got {value}")
    if args.low_score_cutoff > args.mid_score_cutoff:
        raise SystemExit("--low-score-cutoff must be <= --mid-score-cutoff")

    summary = process_file(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
