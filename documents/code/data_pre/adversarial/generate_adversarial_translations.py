"""
Generate adversarial (non-canonical but plausible) Chinese alternatives for trivial terms.

For each trivial term (en, canonical_zh), ask a helper LLM for 3 alternative translations.
Pick the first alternative that:
  - is non-empty
  - differs from canonical_zh (after normalization)
  - consists predominantly of Chinese characters (no English, minimal punctuation)
  - has length in [0.5x, 2x] of canonical (sanity check on plausibility)

Adversarial alternatives are used in training to force the speech LLM to copy from term_map
instead of defaulting to its zero-shot prior.

Inputs:
  - trivial_terms.json from identify_trivial_terms.py

Outputs:
  - adversarial_translations.json: {"en": "adversarial_zh"}
  - adversarial_raw_outputs.jsonl: diagnostic log of what the helper LLM produced

All user-facing strings are in English.
"""

# ======Configuration=====
import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List

DEFAULT_HELPER_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_TRIVIAL_JSON = "/mnt/gemini/data1/jiaxuanluo/adversarial/trivial_terms.json"
DEFAULT_OUTPUT_JSON = "/mnt/gemini/data1/jiaxuanluo/adversarial/adversarial_translations.json"
DEFAULT_RAW_OUTPUTS_JSONL = "/mnt/gemini/data1/jiaxuanluo/adversarial/adversarial_raw_outputs.jsonl"

PROMPT_TEMPLATE = (
    "For the English word/phrase '{en}' (canonical Chinese translation: '{canonical_zh}'), "
    "give 3 alternative Chinese translations that are plausible but NOT the canonical one. "
    "The alternatives should be real Chinese words/phrases, not romanization or gibberish. "
    "Output exactly 3 alternatives, one per line, Chinese only, no numbering, no explanation.\n"
    "Alternatives:"
)

SAMPLING_PARAMS = {
    "temperature": 0.7,  # some diversity for alternatives
    "top_p": 0.9,
    "max_tokens": 64,
    "stop": ["English:", "\n\n"],
}

CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
PUNCT_RE = re.compile(r"[\s\u3000,.!?;:'\"()（）【】《》“”‘’、。！？；：—\-_/\\]+")
# ======Configuration=====


def _normalize_zh(text: str) -> str:
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = PUNCT_RE.sub("", s)
    return s.lower().strip()


def _is_mostly_chinese(text: str, min_ratio: float = 0.6) -> bool:
    if not text:
        return False
    chinese_chars = len(CHINESE_RE.findall(text))
    total_non_space = len([c for c in text if not c.isspace()])
    if total_non_space == 0:
        return False
    return chinese_chars / total_non_space >= min_ratio


def _parse_alternatives(raw: str) -> List[str]:
    """Extract candidate alternatives from helper LLM raw text.

    Strips numbering, bullets, and whitespace. Splits on newlines.
    """
    alts: List[str] = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line:
            continue
        # Remove common prefixes: "1.", "1)", "- ", "* ", etc.
        line = re.sub(r"^\s*[\-\*\d]+[\.\)、]?\s*", "", line)
        line = line.strip("“”\"'：: ")
        if line:
            alts.append(line)
    return alts


def _pick_adversarial(canonical_zh: str, alternatives: List[str]) -> str:
    """Choose the first alternative satisfying quality constraints.

    Returns "" if no acceptable candidate found.
    """
    canonical_norm = _normalize_zh(canonical_zh)
    canonical_len = max(1, len(canonical_norm))
    for alt in alternatives:
        alt_norm = _normalize_zh(alt)
        if not alt_norm:
            continue
        if alt_norm == canonical_norm:
            continue
        if alt_norm in canonical_norm or canonical_norm in alt_norm:
            # Containment is too close to canonical; skip
            continue
        if not _is_mostly_chinese(alt):
            continue
        alt_len = len(alt_norm)
        ratio = alt_len / canonical_len
        if ratio < 0.5 or ratio > 2.0:
            continue
        return alt
    return ""


def build_prompts(items: List[Dict], tokenizer) -> List[str]:
    prompts: List[str] = []
    for it in items:
        msg = [{
            "role": "user",
            "content": PROMPT_TEMPLATE.format(en=it["en"], canonical_zh=it["canonical_zh"]),
        }]
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
    from vllm import LLM, SamplingParams

    sp = SamplingParams(
        temperature=SAMPLING_PARAMS["temperature"],
        top_p=SAMPLING_PARAMS["top_p"],
        max_tokens=SAMPLING_PARAMS["max_tokens"],
        stop=SAMPLING_PARAMS["stop"],
        seed=42,
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
    return [(o.outputs[0].text if o.outputs else "") for o in outputs]


def main():
    parser = argparse.ArgumentParser(description="Generate adversarial alternatives for trivial GT terms.")
    parser.add_argument("--trivial-json", type=str, default=DEFAULT_TRIVIAL_JSON)
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--raw-outputs-jsonl", type=str, default=DEFAULT_RAW_OUTPUTS_JSONL)
    parser.add_argument("--model-name", type=str, default=DEFAULT_HELPER_MODEL)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--gpu-mem-util", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--smoke-n", type=int, default=0,
                        help="If > 0, only process first N trivial terms (for smoke testing)")
    args = parser.parse_args()

    trivial_json = Path(args.trivial_json)
    output_json = Path(args.output_json)
    raw_outputs_jsonl = Path(args.raw_outputs_jsonl)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    raw_outputs_jsonl.parent.mkdir(parents=True, exist_ok=True)

    assert trivial_json.is_file(), f"Trivial terms JSON not found: {trivial_json}"
    trivial_map: Dict[str, str] = json.loads(trivial_json.read_text(encoding="utf-8"))
    assert trivial_map, f"Empty trivial map: {trivial_json}"

    items = [{"en": en, "canonical_zh": zh} for en, zh in trivial_map.items()]
    items.sort(key=lambda x: x["en"])

    if args.smoke_n > 0:
        items = items[: args.smoke_n]
        print(f"[INFO] Smoke test: processing first {len(items)} trivial terms.", flush=True)

    print(f"[INFO] Loading tokenizer + model: {args.model_name}", flush=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    prompts = build_prompts(items, tokenizer)
    print(f"[INFO] Built {len(prompts)} prompts. Running vLLM ...", flush=True)

    raw_outputs = run_inference(
        prompts=prompts,
        model_name=args.model_name,
        tp_size=args.tp_size,
        gpu_mem_util=args.gpu_mem_util,
        max_model_len=args.max_model_len,
    )
    assert len(raw_outputs) == len(items), \
        f"output count {len(raw_outputs)} != item count {len(items)}"

    adversarial: Dict[str, str] = {}
    stats = {"total": 0, "accepted": 0, "no_acceptable": 0}
    with raw_outputs_jsonl.open("w", encoding="utf-8") as rf:
        for it, raw in zip(items, raw_outputs):
            raw_s = (raw or "").strip()
            alts = _parse_alternatives(raw_s)
            picked = _pick_adversarial(it["canonical_zh"], alts)
            rec = {
                "en": it["en"],
                "canonical_zh": it["canonical_zh"],
                "raw": raw_s,
                "parsed_alts": alts,
                "picked": picked,
            }
            stats["total"] += 1
            if picked:
                adversarial[it["en"]] = picked
                stats["accepted"] += 1
            else:
                stats["no_acceptable"] += 1
            rf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    output_json.write_text(
        json.dumps(adversarial, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[INFO] Wrote adversarial_translations.json: {output_json} (n={len(adversarial)})", flush=True)
    print(f"[INFO] Wrote raw outputs: {raw_outputs_jsonl}", flush=True)
    print(f"[INFO] Stats: {stats}", flush=True)


if __name__ == "__main__":
    main()
