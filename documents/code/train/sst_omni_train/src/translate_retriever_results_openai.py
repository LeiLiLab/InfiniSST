#!/usr/bin/env python3
"""Translate source-only retriever results with OpenAI before term-map rebuild."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


def _iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            yield lineno, json.loads(line)


def _term_key(term: str) -> str:
    return " ".join(str(term or "").casefold().split())


def _cache_key(lang_code: str, term: str) -> str:
    return hashlib.sha256(json.dumps({"lang": lang_code, "term": term}, sort_keys=True).encode()).hexdigest()


def _clean(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^[\"'“”‘’`]+|[\"'“”‘’`]+$", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _valid(term: str, translation: str) -> Tuple[bool, str]:
    translation = _clean(translation)
    if not translation:
        return False, "empty"
    if _term_key(term) == _term_key(translation):
        return False, "source_equals_target"
    if "<term" in translation or "</term>" in translation or "=" in translation or "\n" in translation:
        return False, "bad_marker_or_delimiter"
    if len(translation) > max(24, len(term) * 4 + 12):
        return False, "too_long"
    return True, "ok"


def _load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Cache must be JSON object: {path}")
    return {str(k): dict(v) for k, v in obj.items() if isinstance(v, Mapping)}


def _write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def collect_terms(path: Path, max_rows: int) -> List[str]:
    seen = set()
    terms: List[str] = []
    for idx, obj in _iter_jsonl(path):
        if max_rows > 0 and idx > max_rows:
            break
        for chunk in obj.get("retriever_results_by_chunk") or []:
            if not isinstance(chunk, list):
                continue
            for item in chunk:
                if not isinstance(item, Mapping):
                    continue
                term = str(item.get("term") or "").strip()
                key = _term_key(term)
                if term and key and key not in seen:
                    seen.add(key)
                    terms.append(term)
    return terms


def _openai_translate_batch(
    *,
    api_key: str,
    model: str,
    lang_code: str,
    batch: List[str],
    timeout: int,
    max_retries: int,
) -> List[Dict[str, Any]]:
    lang_name = {"de": "German", "ja": "Japanese", "zh": "Chinese"}.get(lang_code, lang_code)
    payload = {
        "model": model,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    f"Translate English glossary terms into {lang_name}. Return strict JSON only. "
                    "Use concise terminology translations. Do not add explanations."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({
                    "schema": {"items": [{"id": "string", "translation": "string"}]},
                    "constraints": [
                        "translation must not be identical to the English source unless no translation is possible",
                        "no XML, no equals signs, no newline characters",
                    ],
                    "items": [{"id": str(i), "term": term} for i, term in enumerate(batch)],
                }, ensure_ascii=False),
            },
        ],
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
            content = json.loads(raw)["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            items = parsed.get("items")
            if not isinstance(items, list):
                raise ValueError("OpenAI JSON missing items")
            return [dict(x) for x in items if isinstance(x, Mapping)]
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(min(60.0, 2.0 * (2 ** attempt)))
    raise RuntimeError(f"OpenAI translation batch failed: {last_error}") from last_error


def fill_cache(args: argparse.Namespace, terms: List[str]) -> Dict[str, Dict[str, Any]]:
    cache = _load_cache(args.openai_cache_json)
    pending = [t for t in terms if _cache_key(args.lang_code, t) not in cache]
    if args.max_api_items > 0:
        pending = pending[: args.max_api_items]
    if not pending:
        return cache
    if args.cache_only:
        raise RuntimeError(f"Cache missing {len(pending)} translations and --cache-only was set")
    if args.dry_run:
        for term in pending:
            cache[_cache_key(args.lang_code, term)] = {
                "status": "invalid",
                "term": term,
                "translation": term + "_T",
                "reason": "dry_run",
                "model": "dry_run",
            }
        _write_json_atomic(args.openai_cache_json, cache)
        return cache
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required unless --dry-run or complete --cache-only")
    batches = [
        (start, pending[start : start + args.openai_batch_size])
        for start in range(0, len(pending), args.openai_batch_size)
    ]

    def _run_one(start_and_batch: Tuple[int, List[str]]) -> Tuple[int, List[str], List[Dict[str, Any]]]:
        start, chunk = start_and_batch
        results = _openai_translate_batch(
            api_key=api_key,
            model=args.openai_model,
            lang_code=args.lang_code,
            batch=chunk,
            timeout=args.openai_timeout,
            max_retries=args.openai_max_retries,
        )
        return start, chunk, results

    completed = 0
    if args.openai_workers <= 1:
        iterator = (_run_one(x) for x in batches)
    else:
        executor = ThreadPoolExecutor(max_workers=args.openai_workers)
        futures = [executor.submit(_run_one, x) for x in batches]
        iterator = (future.result() for future in as_completed(futures))

    try:
      for start, chunk, results in iterator:
        by_id = {str(x.get("id")): x for x in results}
        for i, term in enumerate(chunk):
            translation = _clean(str(by_id.get(str(i), {}).get("translation") or ""))
            ok, reason = _valid(term, translation)
            cache[_cache_key(args.lang_code, term)] = {
                "status": "ok" if ok else "invalid",
                "term": term,
                "translation": translation,
                "reason": reason,
                "model": args.openai_model,
            }
        completed += len(chunk)
        if args.openai_workers <= 1 or completed % max(args.openai_batch_size * args.openai_workers, 1) == 0 or completed >= len(pending):
            _write_json_atomic(args.openai_cache_json, cache)
        print(json.dumps({
            "openai_cache_json": str(args.openai_cache_json),
            "generated": min(completed, len(pending)),
            "pending_total": len(pending),
            "cache_total": len(cache),
            "openai_workers": args.openai_workers,
        }, ensure_ascii=False), flush=True)
        if args.openai_workers <= 1:
            time.sleep(args.openai_sleep_sec)
    finally:
        if args.openai_workers > 1:
            executor.shutdown(wait=False, cancel_futures=True)
    return cache


def apply(args: argparse.Namespace, cache: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    stats = Counter()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as fout:
        for idx, obj in _iter_jsonl(args.input_jsonl):
            if args.max_rows > 0 and idx > args.max_rows:
                break
            out_chunks = []
            for chunk in obj.get("retriever_results_by_chunk") or []:
                out_chunk = []
                if isinstance(chunk, list):
                    for item in chunk:
                        if not isinstance(item, Mapping):
                            continue
                        term = str(item.get("term") or "").strip()
                        cached = cache.get(_cache_key(args.lang_code, term))
                        if not cached or cached.get("status") != "ok":
                            stats[f"dropped:{(cached or {}).get('reason', 'missing_cache')}"] += 1
                            continue
                        new_item = dict(item)
                        translation = str(cached.get("translation") or "").strip()
                        new_item["zh"] = translation
                        new_item["translation"] = translation
                        new_item[args.lang_code] = translation
                        out_chunk.append(new_item)
                        stats["kept_retriever_terms"] += 1
                out_chunks.append(out_chunk)
            obj["retriever_results_by_chunk"] = out_chunks
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            stats["rows_written"] += 1
    return dict(stats)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-jsonl", type=Path, required=True)
    p.add_argument("--output-jsonl", type=Path, required=True)
    p.add_argument("--stats-json", type=Path, required=True)
    p.add_argument("--openai-cache-json", type=Path, required=True)
    p.add_argument("--lang-code", choices=["de", "ja", "zh"], required=True)
    p.add_argument("--openai-model", default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
    p.add_argument("--openai-batch-size", type=int, default=32)
    p.add_argument("--openai-timeout", type=int, default=120)
    p.add_argument("--openai-max-retries", type=int, default=4)
    p.add_argument("--openai-sleep-sec", type=float, default=0.2)
    p.add_argument("--openai-workers", type=int, default=1)
    p.add_argument("--max-api-items", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--cache-only", action="store_true")
    args = p.parse_args()
    if not args.input_jsonl.exists():
        raise FileNotFoundError(args.input_jsonl)
    return args


def main() -> None:
    args = parse_args()
    terms = collect_terms(args.input_jsonl, args.max_rows)
    cache = fill_cache(args, terms)
    stats = apply(args, cache)
    stats.update({
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "unique_retriever_terms": len(terms),
        "cache_entries": len(cache),
        "lang_code": args.lang_code,
    })
    args.stats_json.parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
