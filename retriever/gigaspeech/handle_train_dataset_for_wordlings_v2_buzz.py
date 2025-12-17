#!/usr/bin/env python3
"""
Add LLM-generated shape-similar buzz terms to existing term_map in a JSONL file.

- Input default: siqi_train_wordlings_filled_gt_v2.jsonl
- For each user message containing "term_map: {...}", ask local Qwen3 (vLLM) to
  propose additional English terms that are surface/shape variants (not synonyms)
  of the existing terms.
- Reuse the same Chinese translation from any existing term for the added buzz terms.
- Cap term_map size to <=5. If original size already >=5, truncate to max_size first.
- Output default: siqi_train_wordlings_filled_gt_v2_buzz.jsonl
"""

import argparse
import json
import logging
import os
import random
import re
import shutil
import tempfile
from functools import lru_cache
from typing import Dict, List, Tuple

logger = logging.getLogger("buzz_fill")

# Prefer V0 engine by default to avoid potential V1 init/compile stalls.
# Users can override by exporting VLLM_USE_V1=1 or using --vllm-use-v1.
os.environ.setdefault("VLLM_USE_V1", "0")


DEFAULT_INPUT = "/mnt/gemini/data1/jiaxuanluo/siqi_train_wordlings_filled_gt_v2.jsonl"
DEFAULT_OUTPUT = "/mnt/gemini/data1/jiaxuanluo/siqi_train_wordlings_filled_gt_v2_buzz.jsonl"
DEFAULT_MODEL_PATH = "/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Add shape-similar buzz terms to term_map (cap size<=5).")
    ap.add_argument("--input", default=DEFAULT_INPUT, help="Input jsonl with term_map in user messages.")
    ap.add_argument("--output", default=DEFAULT_OUTPUT, help="Output jsonl path.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--max-size", type=int, default=5, help="Maximum term_map size after augmentation.")
    ap.add_argument(
        "--min-buzz",
        type=int,
        default=1,
        help="Ensure at least this many buzz terms by truncating originals if needed.",
    )
    # local vLLM
    ap.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Local Qwen3 model path for vLLM.")
    ap.add_argument("--tensor-parallel-size", type=int, default=0, help="vLLM tensor parallel size (0=auto).")
    ap.add_argument("--dtype", default="bfloat16", help="Model dtype for vLLM.")
    ap.add_argument("--temperature", type=float, default=0.2, help="LLM temperature.")
    ap.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens.")
    ap.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Max model length for vLLM (reduce startup/memory).",
    )
    ap.add_argument(
        "--enforce-eager",
        type=int,
        default=0,
        help="Set enforce_eager for vLLM (0/1). Use 0 to match v2 script behavior.",
    )
    ap.add_argument(
        "--vllm-use-v1",
        type=int,
        default=0,
        help="Set VLLM_USE_V1 (0=use V0 engine to avoid compile hang, 1=use V1 engine).",
    )
    ap.add_argument("--batch-size", type=int, default=8, help="Batch size for prompts.")
    return ap.parse_args()


def is_valid_en(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False
    if re.search(r"[\u4e00-\u9fff]", s):
        return False
    if not re.search(r"[a-zA-Z]", s):
        return False
    lower = s.lower()
    if lower in {"n/a", "na", "none", "null"}:
        return False
    return True


def parse_term_map(content: str) -> Tuple[Dict[str, str], str]:
    marker = "term_map:"
    if marker not in content:
        return {}, content
    prefix, rest = content.split(marker, 1)
    rest = rest.strip()
    tm = {}
    try:
        tm = json.loads(rest)
        if not isinstance(tm, dict):
            tm = {}
    except Exception:
        m = re.search(r"\{.*\}", rest)
        if m:
            try:
                tm = json.loads(m.group(0))
                if not isinstance(tm, dict):
                    tm = {}
            except Exception:
                tm = {}
    return tm, prefix


def _abs(path: str) -> str:
    return os.path.abspath(path)


def resolve_tp(tp_flag: int) -> int:
    if tp_flag and tp_flag > 0:
        return tp_flag
    env_tp = os.environ.get("VLLM_TENSOR_PARALLEL_SIZE")
    if env_tp and env_tp.isdigit() and int(env_tp) > 0:
        return int(env_tp)
    try:
        import torch  # type: ignore

        return max(1, torch.cuda.device_count())
    except Exception:
        return 1


def build_prompt(en_terms: List[str], need: int) -> str:
    return (
        "You generate English buzz terms that are shape/orthography-similar (not synonyms) to given terms.\n"
        "Rules:\n"
        "- Output real English words (or common variants), lower-case, single token, no spaces/punctuation.\n"
        "- They should look confusable by shape/spelling (e.g., map->mop, cat->cap, run->ran, table->cable).\n"
        "- Do NOT output the original terms; avoid semantic synonyms; avoid transliterations or non-English.\n"
        f"- Return at most {need} items.\n"
        f"Given terms: {en_terms}\n"
        'Return JSON: {"buzz_terms": ["...","..."]}\n'
        "Only output JSON."
    )


@lru_cache(maxsize=4)
def _get_llm(model_path: str, tp: int, dtype: str, max_model_len: int, enforce_eager: bool):
    try:
        from vllm import LLM  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"vLLM not available: {exc}")
    logger.info(
        "Initializing vLLM LLM: model=%s tp=%s dtype=%s max_model_len=%s enforce_eager=%s VLLM_USE_V1=%s",
        model_path,
        tp,
        dtype,
        max_model_len,
        enforce_eager,
        os.environ.get("VLLM_USE_V1"),
    )
    return LLM(
        model=model_path,
        tensor_parallel_size=tp,
        dtype=dtype,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
        disable_log_stats=True,
    )


def llm_generate_buzz_batch(
    batches: List[List[str]],
    needs: List[int],
    args: argparse.Namespace,
) -> List[List[str]]:
    """
    batches: list of en_terms lists
    return list of buzz lists aligned with batches
    """
    if not batches:
        return []
    if len(batches) != len(needs):
        raise ValueError("batches and needs must have same length")
    try:
        from vllm import SamplingParams  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"vLLM SamplingParams missing: {exc}")

    tp = resolve_tp(args.tensor_parallel_size)
    llm = _get_llm(args.model_path, tp, args.dtype, args.max_model_len, bool(args.enforce_eager))
    prompts = [build_prompt(en_terms, need) for en_terms, need in zip(batches, needs)]
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens, n=1)
    logger.info("Calling vLLM generate: batch=%s", len(prompts))
    outputs = llm.generate(prompts, sampling)
    logger.info("vLLM generate returned: batch=%s", len(outputs))
    results: List[List[str]] = []
    for out, need in zip(outputs, needs):
        content = (out.outputs[0].text or "").strip()
        terms = []
        try:
            data = json.loads(content)
        except Exception:
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if m:
                data = json.loads(m.group(0))
            else:
                data = {}
        buzz = data.get("buzz_terms") if isinstance(data, dict) else []
        if isinstance(buzz, list):
            for t in buzz:
                if isinstance(t, str):
                    t = t.strip().lower()
                    if is_valid_en(t):
                        terms.append(t)
                if len(terms) >= need:
                    break
        logger.info("LLM buzz count=%s (need=%s)", len(terms), need)
        results.append(terms[:need])
    return results


def process_line(
    obj: Dict,
    max_size: int,
    min_buzz: int,
    rng: random.Random,
    llm_cache: Dict[str, List[str]],
    args: argparse.Namespace,
) -> Dict:
    msgs = obj.get("messages", [])
    new_msgs = []
    pending_prompts: List[Tuple[int, List[str]]] = []
    pending_needs: List[int] = []
    # First pass: collect prompts to batch
    for idx, msg in enumerate(msgs):
        content = msg.get("content", "")
        tm, prefix = parse_term_map(content)
        if not tm:
            continue
        clean_tm_items = [(k.strip(), str(v).strip()) for k, v in tm.items() if is_valid_en(k) and v]
        # truncate originals to leave room for buzz if needed
        keep_orig = max(0, max_size - max(0, min_buzz))
        clean_tm_items = clean_tm_items[:keep_orig]
        clean_keys = [k for k, _ in clean_tm_items]
        need = max_size - len(clean_keys)
        if need <= 0:
            continue
        pending_prompts.append((idx, clean_keys))
        pending_needs.append(need)

    # Batch LLM
    prompt_batches: List[List[str]] = []
    map_idx_to_buzz: Dict[int, List[str]] = {}
    batch_size = max(1, args.batch_size)
    for chunk_start in range(0, len(pending_prompts), batch_size):
        chunk = pending_prompts[chunk_start : chunk_start + batch_size]
        chunk_needs = pending_needs[chunk_start : chunk_start + batch_size]
        prompt_batches = [keys for _, keys in chunk]
        # cache key to avoid repeated calls for identical prompts
        cache_hits = []
        to_call_batches: List[List[str]] = []
        to_call_needs: List[int] = []
        to_call_msg_idx: List[int] = []
        for (msg_idx, keys), need in zip(chunk, chunk_needs):
            cache_key = f"{need}|" + "|".join(sorted(keys))
            if cache_key in llm_cache:
                cache_hits.append((msg_idx, llm_cache[cache_key][:need]))
            else:
                to_call_batches.append(keys)
                to_call_needs.append(need)
                to_call_msg_idx.append(msg_idx)
        for msg_idx, buzz in cache_hits:
            map_idx_to_buzz[msg_idx] = buzz
        if to_call_batches:
            buzz_lists = llm_generate_buzz_batch(to_call_batches, to_call_needs, args)
            for msg_idx, keys, need, buzz in zip(to_call_msg_idx, to_call_batches, to_call_needs, buzz_lists):
                cache_key = f"{need}|" + "|".join(sorted(keys))
                llm_cache[cache_key] = buzz
                map_idx_to_buzz[msg_idx] = buzz

    # Second pass: apply
    for idx, msg in enumerate(msgs):
        content = msg.get("content", "")
        tm, prefix = parse_term_map(content)
        if not tm:
            new_msgs.append(msg)
            continue
        clean_tm_items = [(k.strip(), str(v).strip()) for k, v in tm.items() if is_valid_en(k) and v]
        keep_orig = max(0, max_size - max(0, min_buzz))
        clean_tm_items = clean_tm_items[:keep_orig]
        clean_tm = dict(clean_tm_items)
        need = max_size - len(clean_tm)
        buzz_en = map_idx_to_buzz.get(idx, []) if need > 0 else []
        if buzz_en and clean_tm:
            keys = list(clean_tm.keys())
            for b in buzz_en:
                if len(clean_tm) >= max_size:
                    break
                zh = clean_tm[rng.choice(keys)]
                if b not in clean_tm:
                    clean_tm[b] = zh
        new_content = f"{prefix}term_map: {json.dumps(clean_tm, ensure_ascii=False)}"
        new_msgs.append({**msg, "content": new_content})
    obj["messages"] = new_msgs
    return obj


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    # Must be set before vLLM is imported/initialized.
    os.environ["VLLM_USE_V1"] = str(args.vllm_use_v1)
    args.input = _abs(args.input)
    args.output = _abs(args.output)
    rng = random.Random(args.seed)
    logger.info(
        "Args: input=%s output=%s max_size=%s min_buzz=%s model_path=%s tp=%s dtype=%s batch_size=%s max_model_len=%s enforce_eager=%s VLLM_USE_V1=%s",
        args.input,
        args.output,
        args.max_size,
        args.min_buzz,
        args.model_path,
        resolve_tp(args.tensor_parallel_size),
        args.dtype,
        args.batch_size,
        args.max_model_len,
        args.enforce_eager,
        os.environ.get("VLLM_USE_V1"),
    )
    if os.path.abspath(args.output) == os.path.abspath(args.input):
        bak = args.input + ".bak"
        if os.path.exists(bak):
            os.remove(bak)
        shutil.copy2(args.input, bak)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="term_buzz_", suffix=".jsonl")
    os.close(tmp_fd)
    count = 0
    llm_cache: Dict[str, List[str]] = {}
    with open(args.input, "r", encoding="utf-8") as fin, open(tmp_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj = process_line(obj, args.max_size, args.min_buzz, rng, llm_cache, args)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1
            if count % 500 == 0:
                logger.info("Processed %d records...", count)
    shutil.move(tmp_path, args.output)
    print(f"processed {count} records -> {args.output}")


if __name__ == "__main__":
    main()




