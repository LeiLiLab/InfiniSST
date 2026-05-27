"""
Identify "trivial" GT terms that a helper LLM can translate correctly zero-shot.

A trivial term is one whose canonical Chinese translation matches the helper LLM's
zero-shot translation (normalized). These are the terms where the speech LLM can
get the right answer WITHOUT needing the term_map, so training on them creates the
"copy faith" problem (model learns term_map is optional confirmation).

Inputs:
  - training JSONL (e.g., train_cleaned_with_retriever_results_varlen.jsonl)
    -> we extract unique (en_term, canonical_zh) pairs from gt_terms_by_chunk

Outputs:
  - trivial_terms.json: {"en": "canonical_zh"} for pairs the helper LLM nailed

All user-facing strings are in English.
"""

# ======Configuration=====
import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict

DEFAULT_HELPER_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
DEFAULT_INPUT_JSONL = "/mnt/gemini/data1/jiaxuanluo/train_cleaned_with_retriever_results_varlen.jsonl"
DEFAULT_OUTPUT_JSON = "/mnt/gemini/data1/jiaxuanluo/adversarial/trivial_terms.json"
DEFAULT_RAW_OUTPUTS_JSONL = "/mnt/gemini/data1/jiaxuanluo/adversarial/helper_raw_outputs.jsonl"

PROMPT_TEMPLATE = (
    "Translate the following English word or phrase into Chinese. "
    "Output ONLY the Chinese translation with no explanation, no pinyin, no English. "
    "If there are multiple senses, give the most common one.\n"
    "English: {en}\n"
    "Chinese:"
)

SAMPLING_PARAMS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 32,
    "stop": ["\n", "。", "English:"],
}

MIN_PAIR_FREQ = 1  # include all unique pairs by default (override via --min-freq)
# ======Configuration=====


PUNCT_RE = re.compile(r"[\s\u3000,.!?;:'\"()（）【】《》“”‘’、。！？；：—\-_/\\]+")


def _normalize_zh(text: str) -> str:
    """Normalize a Chinese string: NFKC, strip punctuation/whitespace, lowercase Latin."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = PUNCT_RE.sub("", s)
    return s.lower().strip()


def _is_trivial_match(pred_zh: str, canonical_zh: str) -> bool:
    """Decide if helper output matches the canonical Chinese translation.

    Rule: after normalization, either string is contained in the other, OR they are equal.
    Containment handles cases like pred="启发式方法" vs canonical="启发式".
    """
    p = _normalize_zh(pred_zh)
    c = _normalize_zh(canonical_zh)
    if not p or not c:
        return False
    if p == c:
        return True
    if p in c or c in p:
        return True
    return False


def extract_unique_pairs(input_jsonl: Path, min_freq: int) -> List[Tuple[str, str, int]]:
    """Extract unique (en, canonical_zh) pairs with their frequency."""
    assert input_jsonl.is_file(), f"Input JSONL not found: {input_jsonl}"
    counter: Counter = Counter()
    with input_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            for chunk in d.get("gt_terms_by_chunk", []):
                for term in chunk:
                    en = (term.get("term") or "").strip()
                    zh = (term.get("zh") or "").strip()
                    if en and zh:
                        counter[(en, zh)] += 1
    pairs = [
        (en, zh, count)
        for (en, zh), count in counter.items()
        if count >= min_freq
    ]
    pairs.sort(key=lambda x: (-x[2], x[0], x[1]))
    return pairs


def build_prompts(pairs: List[Tuple[str, str, int]], tokenizer) -> List[str]:
    """Build chat-template prompts for each English term."""
    prompts: List[str] = []
    for en, _zh, _cnt in pairs:
        msg = [{"role": "user", "content": PROMPT_TEMPLATE.format(en=en)}]
        prompt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    return prompts


def run_inference(
    prompts: List[str],
    model_name: str,
    tp_size: int,
    gpu_mem_util: float,
    max_model_len: int,
) -> List[str]:
    """Run batched vLLM inference. Returns list of raw output strings (one per prompt)."""
    from vllm import LLM, SamplingParams

    sp = SamplingParams(
        temperature=SAMPLING_PARAMS["temperature"],
        top_p=SAMPLING_PARAMS["top_p"],
        max_tokens=SAMPLING_PARAMS["max_tokens"],
        stop=SAMPLING_PARAMS["stop"],
    )

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        enforce_eager=True,
        trust_remote_code=True,
        dtype="auto",
    )

    outputs = llm.generate(prompts, sp)
    results: List[str] = []
    for out in outputs:
        text = out.outputs[0].text if out.outputs else ""
        results.append(text)
    return results


def main():
    parser = argparse.ArgumentParser(description="Identify trivial GT terms via helper LLM zero-shot translation.")
    parser.add_argument("--input-jsonl", type=str, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--raw-outputs-jsonl", type=str, default=DEFAULT_RAW_OUTPUTS_JSONL)
    parser.add_argument("--model-name", type=str, default=DEFAULT_HELPER_MODEL)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--gpu-mem-util", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--min-freq", type=int, default=MIN_PAIR_FREQ)
    parser.add_argument("--smoke-n", type=int, default=0,
                        help="If > 0, only process first N unique pairs (for smoke testing)")
    args = parser.parse_args()

    input_jsonl = Path(args.input_jsonl)
    output_json = Path(args.output_json)
    raw_outputs_jsonl = Path(args.raw_outputs_jsonl)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    raw_outputs_jsonl.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Extracting unique (en, zh) pairs from {input_jsonl} ...", flush=True)
    pairs = extract_unique_pairs(input_jsonl, args.min_freq)
    print(f"[INFO] Found {len(pairs)} unique pairs (min_freq={args.min_freq}).", flush=True)

    if args.smoke_n > 0:
        pairs = pairs[: args.smoke_n]
        print(f"[INFO] Smoke test: processing first {len(pairs)} pairs.", flush=True)

    assert pairs, "No pairs to process"

    print(f"[INFO] Loading tokenizer + model: {args.model_name}", flush=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    prompts = build_prompts(pairs, tokenizer)
    print(f"[INFO] Built {len(prompts)} prompts. Running vLLM ...", flush=True)

    raw_outputs = run_inference(
        prompts=prompts,
        model_name=args.model_name,
        tp_size=args.tp_size,
        gpu_mem_util=args.gpu_mem_util,
        max_model_len=args.max_model_len,
    )
    assert len(raw_outputs) == len(pairs), \
        f"output count {len(raw_outputs)} != pair count {len(pairs)}"

    trivial: Dict[str, str] = {}
    stats = {"total": 0, "trivial": 0, "mismatch": 0, "empty_output": 0, "duplicates_skipped": 0}
    with raw_outputs_jsonl.open("w", encoding="utf-8") as rf:
        for (en, zh, cnt), raw in zip(pairs, raw_outputs):
            raw_s = (raw or "").strip()
            rec = {
                "en": en,
                "canonical_zh": zh,
                "freq": cnt,
                "helper_output": raw_s,
            }
            stats["total"] += 1
            if not raw_s:
                stats["empty_output"] += 1
                rec["verdict"] = "empty"
            elif _is_trivial_match(raw_s, zh):
                if en in trivial:
                    stats["duplicates_skipped"] += 1
                    rec["verdict"] = "trivial_dup"
                else:
                    trivial[en] = zh
                    stats["trivial"] += 1
                    rec["verdict"] = "trivial"
            else:
                stats["mismatch"] += 1
                rec["verdict"] = "mismatch"
            rf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    output_json.write_text(
        json.dumps(trivial, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[INFO] Wrote trivial_terms.json: {output_json} (n={len(trivial)})", flush=True)
    print(f"[INFO] Wrote raw outputs: {raw_outputs_jsonl}", flush=True)
    print(f"[INFO] Stats: {stats}", flush=True)

    if stats["total"] > 0:
        rate = stats["trivial"] / stats["total"]
        print(f"[INFO] Trivial rate: {rate:.3f} ({stats['trivial']}/{stats['total']})", flush=True)


if __name__ == "__main__":
    main()
