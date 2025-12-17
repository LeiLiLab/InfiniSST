#!/usr/bin/env python3
"""
Fill wordlings for a subset of the siqi train set, using local Qwen3 via vLLM.

Steps:
- Randomly sample 50% (configurable) of the input jsonl lines, with optional limit.
- For each sampled line, resolve the first audio path to an utterance id
  like AUD0000000915_2159, fetch the English ASR text from the TSV (col 4)
  plus Chinese segmented tokens (last column), and combine with the assistant-side
  Chinese translation.
- Ask Qwen3 to propose 10 core term pairs and optionally 10 extra buzz terms (English->Chinese),
  using both English ASR and Chinese tokens. Buzz terms are surface/shape variants
  (e.g., happiness/happy), not semantic synonyms, and can be disabled via flag.
- Compute multiple_number = ceil(len(zh_tokens) / clip_count); Chinese tokens are
  chunked by this size to form clips. Distribute terms + buzz to the corresponding
  clip, capped at 5 per clip, preserving order (terms first, then buzz).
- Inject per-clip wordlings into the matched user messages only.
"""

import argparse
import ast
import json
import logging
import math
import os
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DEFAULT_INPUT = "/mnt/gemini/data1/jiaxuanluo/siqi_train_wordlings.jsonl"
DEFAULT_OUTPUT = "/mnt/gemini/data1/jiaxuanluo/siqi_train_wordlings_filled_v2_test.jsonl"
DEFAULT_TSV = (
    "/mnt/taurus/data/siqiouyang/datasets/gigaspeech/"
    "train_xl_case_ft-qwen2.5-32b-instruct_marked_mfa_punc_asr.tsv"
)
DEFAULT_MODEL_PATH = "/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct"

logger = logging.getLogger("fill_wordlings")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill wordlings using local Qwen3 (vLLM) based on English ASR and Chinese translations."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Source jsonl path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Destination jsonl path.")
    parser.add_argument(
        "--tsv",
        default=DEFAULT_TSV,
        help="TSV with ASR text (English in column 4).",
    )
    parser.add_argument("--sample-ratio", type=float, default=0.5, help="Fraction to process.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional hard limit on number of sampled records to process (e.g., 100).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Local Qwen3 model path for vLLM.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=0,
        help="vLLM tensor parallel size (0 = auto by GPU count or env VLLM_TENSOR_PARALLEL_SIZE).",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="Model dtype for vLLM (e.g., bfloat16, float16).",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2, help="Sampling temperature for the LLM."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max new tokens for generation.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retries for LLM failures before giving up on a sample.",
    )
    parser.add_argument(
        "--no-buzz",
        action="store_true",
        help="Disable buzz terms generation/injection.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for vLLM generate calls.",
    )
    return parser.parse_args()


def _abs(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return os.path.abspath(path)


def normalize_paths(args: argparse.Namespace) -> argparse.Namespace:
    args.input = _abs(args.input)
    args.output = _abs(args.output)
    args.tsv = _abs(args.tsv)
    args.model_path = _abs(args.model_path)
    return args


def resolve_tensor_parallel_size(args: argparse.Namespace) -> int:
    if args.tensor_parallel_size and args.tensor_parallel_size > 0:
        return args.tensor_parallel_size
    env_tp = os.environ.get("VLLM_TENSOR_PARALLEL_SIZE")
    if env_tp and env_tp.isdigit() and int(env_tp) > 0:
        return int(env_tp)
    try:
        import torch  # type: ignore

        gpu_count = torch.cuda.device_count()
        return max(1, gpu_count)
    except Exception:
        return 1


def load_jsonl(path: str) -> List[Dict]:
    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def derive_utt_id(audio_path: str) -> Optional[str]:
    """
    Convert .../AUD0000000915/2159/0.wav -> AUD0000000915_2159.
    """
    path = Path(audio_path)
    try:
        segment = path.parent.name
        root = path.parent.parent.name
        if not root or not segment:
            return None
        return f"{root}_{segment}"
    except Exception:
        return None


def collect_target_ids(records: Iterable[Dict]) -> List[str]:
    ids: List[str] = []
    for rec in records:
        audios = rec.get("audios") or []
        if not audios:
            continue
        utt = derive_utt_id(audios[0])
        if utt:
            ids.append(utt)
    return ids


def load_asr_map(tsv_path: str, target_ids: Iterable[str]) -> Dict[str, Dict[str, object]]:
    targets = set(target_ids)
    asr_map: Dict[str, Dict[str, object]] = {}
    if not targets:
        return asr_map
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(asr_map) == len(targets):
                break
            parts = line.rstrip("\n").split("\t")
            if not parts:
                continue
            utt_id = parts[0]
            if utt_id not in targets:
                continue
            english_asr = parts[3] if len(parts) > 3 else ""
            zh_tokens: List[str] = []
            if parts:
                try:
                    zh_tokens = ast.literal_eval(parts[-1])
                    if not isinstance(zh_tokens, list):
                        zh_tokens = []
                except Exception:
                    zh_tokens = []
            asr_map[utt_id] = {"en": english_asr, "zh_tokens": zh_tokens}
    return asr_map


def join_assistant_text(messages: List[Dict]) -> str:
    return " ".join(m.get("content", "") for m in messages if m.get("role") == "assistant")


def build_prompt(english_asr: str, zh_translation: str, zh_tokens: List[str], enable_buzz: bool) -> str:
    buzz_text = (
        "2) Add 10 extra buzz terms (English) that are SHAPE/MORPH variants of the core terms "
        "(e.g., happiness/happy, running/run, plural/singular), NOT semantic synonyms. "
        "Provide Chinese translations for them as well.\n"
        '  "buzz_terms": [ {"en": "...", "zh": "..."}, ... x10 ... ]\n'
    )
    no_buzz_text = (
        "2) Do not add buzz terms. Return an empty list for buzz_terms.\n"
        '  "buzz_terms": []\n'
    )
    return (
        "You generate bilingual glossary entries for ASR-driven simultaneous translation.\n"
        "Given:\n"
        f"- English ASR sentence: {english_asr}\n"
        f"- Chinese translation: {zh_translation}\n"
        f"- Chinese tokens (segmented): {zh_tokens}\n"
        "Tasks:\n"
        "1) Extract exactly 10 key terms from the English sentence, keep them in lower-case,\n"
        "   and pair each with its best Chinese translation.\n"
        f"{buzz_text if enable_buzz else no_buzz_text}"
        "Return pure JSON with this structure:\n"
        "{\n"
        '  "terms": [ {"en": "...", "zh": "..."}, ... x10 ... ],\n'
        '  "buzz_terms": ...\n'
        "}\n"
        "Only output JSON. No comments, no extra text."
    )


@lru_cache(maxsize=1)
def _get_llm(model_path: str, tensor_parallel_size: int, dtype: str):
    try:
        from vllm import LLM  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency issue
        raise RuntimeError(f"vLLM not available: {exc}")
    return LLM(model=model_path, tensor_parallel_size=tensor_parallel_size, dtype=dtype)


def call_qwen3_local(
    english_asr: str,
    zh_translation: str,
    zh_tokens: List[str],
    args: argparse.Namespace,
) -> Dict:
    try:
        from vllm import SamplingParams  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency issue
        raise RuntimeError(f"vLLM SamplingParams missing: {exc}")

    tp = resolve_tensor_parallel_size(args)
    llm = _get_llm(args.model_path, tp, args.dtype)
    prompt = build_prompt(english_asr, zh_translation, zh_tokens, enable_buzz=not args.no_buzz)
    logger.info("LLM prompt (truncated): %s", _truncate_log(prompt))
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        n=1,
    )
    outputs = llm.generate([prompt], sampling)
    if not outputs or not outputs[0].outputs:
        raise RuntimeError("Empty response from Qwen3 local.")
    content = (outputs[0].outputs[0].text or "").strip()
    logger.info("LLM raw output (truncated): %s", _truncate_log(content))
    if not content:
        raise RuntimeError("Empty response from Qwen3 local.")
    return _parse_model_json(content)


def _parse_model_json(content: str) -> Dict:
    def try_json(text: str) -> Optional[Dict]:
        try:
            return json.loads(text)
        except Exception:
            return None

    def try_literal(text: str) -> Optional[Dict]:
        fixed = (
            text.replace("true", "True")
            .replace("false", "False")
            .replace("null", "None")
        )
        try:
            return ast.literal_eval(fixed)
        except Exception:
            return None

    # 1) direct json
    parsed = try_json(content)
    if parsed is not None:
        return parsed
    # 2) extract JSON-looking substring
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        sub = match.group(0)
        parsed = try_json(sub)
        if parsed is not None:
            return parsed
        parsed = try_literal(sub)
        if parsed is not None:
            return parsed

    # 3) fallback regex extraction: collect en/zh pairs
    pairs = re.findall(r'"en"\s*:\s*"([^"]+)"\s*,\s*"zh"\s*:\s*"([^"]+)"', content)
    if not pairs:
        # try reversed order zh then en
        pairs = re.findall(r'"zh"\s*:\s*"([^"]+)"\s*,\s*"en"\s*:\s*"([^"]+)"', content)
        if pairs:
            pairs = [(en, zh) for zh, en in pairs]
    if pairs:
        filtered = [(en, zh) for en, zh in pairs if _is_valid_en(en) and _is_valid_zh(zh)]
        terms = [{"en": en, "zh": zh} for en, zh in filtered[:10]]
        buzz = [{"en": en, "zh": zh} for en, zh in filtered[10:20]]
        return {"terms": terms, "buzz_terms": buzz}

    raise RuntimeError(f"Failed to parse model JSON: {content[:200]}")


def _truncate_log(text: str, limit: int = 2000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def _is_valid_en(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False
    lower = s.lower()
    if lower in {"n/a", "na", "none", "null"}:
        return False
    if re.search(r"[\u4e00-\u9fff]", s):
        return False
    if not re.search(r"[a-zA-Z]", s):
        return False
    return True


def _is_valid_zh(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False
    if not re.search(r"[\u4e00-\u9fff]", s):
        return False
    return True


def _chunk_list(items: List[int], size: int) -> List[List[int]]:
    if size <= 0:
        size = 1
    return [items[i : i + size] for i in range(0, len(items), size)]


def process_batch(
    indices: List[int],
    records: List[Dict],
    asr_map: Dict[str, Dict[str, object]],
    args: argparse.Namespace,
) -> Dict[int, Dict]:
    """
    Batch process selected record indices via a single vLLM generate per chunk.
    """
    result: Dict[int, Dict] = {}
    tp = resolve_tensor_parallel_size(args)
    llm = _get_llm(args.model_path, tp, args.dtype)
    try:
        from vllm import SamplingParams  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"vLLM SamplingParams missing: {exc}")

    for chunk in _chunk_list(indices, args.batch_size):
        prompt_data = []
        for idx in chunk:
            rec = records[idx]
            audios = rec.get("audios") or []
            if not audios:
                result[idx] = rec
                continue
            utt_id = derive_utt_id(audios[0])
            if not utt_id:
                result[idx] = rec
                continue
            asr_entry = asr_map.get(utt_id)
            if not asr_entry:
                result[idx] = rec
                continue
            english_asr = str(asr_entry.get("en") or "")
            zh_translation = join_assistant_text(rec.get("messages", []))
            prompt = build_prompt(
                english_asr,
                zh_translation,
                asr_entry.get("zh_tokens", []),
                enable_buzz=not args.no_buzz,
            )
            prompt_data.append(
                {
                    "idx": idx,
                    "rec": rec,
                    "audios": audios,
                    "utt_id": utt_id,
                    "asr_entry": asr_entry,
                    "english_asr": english_asr,
                    "prompt": prompt,
                }
            )

        if not prompt_data:
            continue

        sampling = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
            n=1,
        )
        prompts = [p["prompt"] for p in prompt_data]
        outputs = llm.generate(prompts, sampling)

        for pdata, out in zip(prompt_data, outputs):
            idx = pdata["idx"]
            rec = pdata["rec"]
            try:
                content = (out.outputs[0].text or "").strip()
                logger.info("LLM raw output (truncated, idx=%s): %s", idx, _truncate_log(content))
                parsed = _parse_model_json(content)
                terms, buzz = build_wordling_lists(parsed)
                if args.no_buzz:
                    buzz = []
                if len(terms) < 2:
                    raise RuntimeError(f"Too few terms returned: {terms}")

                num_clips = count_clip_slots(rec.get("messages", []), pdata["audios"])
                multiple_number, token_groups = compute_clip_groups(pdata["asr_entry"].get("zh_tokens", []), num_clips)
                per_clip = distribute_wordlings_by_clip(terms, buzz, num_clips=num_clips, max_per_clip=5)

                new_rec = {**rec, "messages": inject_wordlings_by_clip(rec.get("messages", []), per_clip)}
                result[idx] = new_rec
            except Exception as exc:
                logger.warning("LLM batch failed for idx=%s utt_id=%s: %s", idx, pdata["utt_id"], exc)
                result[idx] = rec

    return result


def build_wordling_lists(payload: Dict) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    terms: List[Tuple[str, str]] = []
    buzz: List[Tuple[str, str]] = []
    for item in payload.get("terms", []):
        en = item.get("en") or item.get("english") or item.get("term")
        zh = item.get("zh") or item.get("chinese")
        if en and zh and _is_valid_en(en) and _is_valid_zh(zh):
            terms.append((str(en).strip(), str(zh).strip()))
    for item in payload.get("buzz_terms", []):
        en = item.get("en") or item.get("english") or item.get("term")
        zh = item.get("zh") or item.get("chinese")
        if en and zh and _is_valid_en(en) and _is_valid_zh(zh):
            buzz.append((str(en).strip(), str(zh).strip()))
    return terms, buzz


def distribute_wordlings_by_clip(
    terms: List[Tuple[str, str]],
    buzz: List[Tuple[str, str]],
    num_clips: int,
    max_per_clip: int = 5,
) -> List[Dict[str, str]]:
    clip_wordlings: List[Dict[str, str]] = []
    if num_clips <= 0:
        return clip_wordlings
    term_chunk = max(1, math.ceil(len(terms) / num_clips)) if terms else 0
    buzz_chunk = max(1, math.ceil(len(buzz) / num_clips)) if buzz else 0
    for i in range(num_clips):
        per_clip: List[Tuple[str, str]] = []
        if term_chunk:
            per_clip.extend(terms[i * term_chunk : (i + 1) * term_chunk])
        if buzz_chunk:
            per_clip.extend(buzz[i * buzz_chunk : (i + 1) * buzz_chunk])
        per_clip = per_clip[:max_per_clip]
        clip_wordlings.append({en: zh for en, zh in per_clip})
    return clip_wordlings


def count_clip_slots(messages: List[Dict], audios: List[str]) -> int:
    msg_slots = sum(
        1 for m in messages if m.get("role") == "user" and "wordlings:" in m.get("content", "")
    )
    return msg_slots if msg_slots > 0 else len(audios)


def compute_clip_groups(zh_tokens: List[str], clip_count: int) -> Tuple[int, List[List[str]]]:
    if clip_count <= 0:
        return 0, []
    multiple_number = max(1, math.ceil(len(zh_tokens) / clip_count)) if zh_tokens else 1
    groups: List[List[str]] = []
    for i in range(clip_count):
        start = i * multiple_number
        end = (i + 1) * multiple_number
        groups.append(zh_tokens[start:end])
    return multiple_number, groups


def inject_wordlings_by_clip(messages: List[Dict], per_clip_wordlings: List[Dict[str, str]]) -> List[Dict]:
    new_messages: List[Dict] = []
    clip_idx = 0
    for msg in messages:
        if msg.get("role") == "user" and "wordlings:" in msg.get("content", ""):
            wl = per_clip_wordlings[clip_idx] if clip_idx < len(per_clip_wordlings) else {}
            wordlings_str = json.dumps(wl, ensure_ascii=False)
            content = msg["content"].replace("wordlings: {}", f"wordlings: {wordlings_str}")
            new_messages.append({**msg, "content": content})
            clip_idx += 1
        else:
            new_messages.append(msg)
    return new_messages


def process_record(
    rec: Dict,
    asr_map: Dict[str, Dict[str, object]],
    args: argparse.Namespace,
) -> Dict:
    audios = rec.get("audios") or []
    if not audios:
        return rec
    utt_id = derive_utt_id(audios[0])
    if not utt_id:
        return rec
    asr_entry = asr_map.get(utt_id)
    if not asr_entry:
        logger.warning("Missing ASR for %s", utt_id)
        return rec
    english_asr = str(asr_entry.get("en") or "")
    zh_translation = join_assistant_text(rec.get("messages", []))
    retries = 0
    last_err: Optional[Exception] = None
    while retries <= args.max_retries:
        try:
            payload = call_qwen3_local(
                english_asr=english_asr,
                zh_translation=zh_translation,
                zh_tokens=asr_entry.get("zh_tokens", []),
                args=args,
            )
            terms, buzz = build_wordling_lists(payload)
            if len(terms) < 2:
                raise RuntimeError(f"Too few terms returned: {terms}")

            num_clips = count_clip_slots(rec.get("messages", []), audios)
            multiple_number, token_groups = compute_clip_groups(asr_entry.get("zh_tokens", []), num_clips)
            per_clip = distribute_wordlings_by_clip(terms, buzz, num_clips=num_clips, max_per_clip=5)

            rec = {**rec, "messages": inject_wordlings_by_clip(rec.get("messages", []), per_clip)}
            rec["wordlings_meta"] = {
                "utt_id": utt_id,
                "english_asr": english_asr,
                "zh_tokens": asr_entry.get("zh_tokens", []),
                "token_groups": token_groups,
                "source": "qwen3_local_vllm",
                "term_count": sum(len(x) for x in per_clip),
                "clip_count": num_clips,
                "multiple_number": multiple_number,
            }
            return rec
        except Exception as exc:  # pragma: no cover - LLM/runtime errors
            retries += 1
            last_err = exc
            logger.warning("LLM failed for %s (attempt %s/%s): %s", utt_id, retries, args.max_retries, exc)
    if last_err:
        logger.error("Give up on %s due to repeated failures: %s", utt_id, last_err)
    return rec


def main() -> None:
    args = normalize_paths(parse_args())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    random.seed(args.seed)

    logger.info("Loading input: %s", args.input)
    records = load_jsonl(args.input)
    total = len(records)
    logger.info("Loaded %d records", total)

    sample_size = int(total * args.sample_ratio)
    sample_indices = set(random.sample(range(total), sample_size))
    if args.limit is not None:
        sample_indices = set(list(sample_indices)[: args.limit])
        logger.info(
            "Sampling %d records (ratio=%.2f, limit=%s)", len(sample_indices), args.sample_ratio, args.limit
        )
    else:
        logger.info("Sampling %d records (ratio=%.2f)", sample_size, args.sample_ratio)

    target_ids = collect_target_ids(records[i] for i in sample_indices)
    logger.info("Collected %d target utt ids", len(target_ids))
    asr_map = load_asr_map(args.tsv, target_ids)
    logger.info("Resolved ASR for %d/%d ids", len(asr_map), len(target_ids))

    os.makedirs(Path(args.output).parent, exist_ok=True)
    processed_map = process_batch(sorted(sample_indices), records, asr_map, args)

    with open(args.output, "w", encoding="utf-8") as out_f:
        if args.limit is not None:
            for idx in sorted(sample_indices):
                rec = processed_map.get(idx, records[idx])
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            for idx, rec in enumerate(records):
                if idx in sample_indices:
                    rec = processed_map.get(idx, rec)
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info("Done. Wrote %s", args.output)


if __name__ == "__main__":
    main()
