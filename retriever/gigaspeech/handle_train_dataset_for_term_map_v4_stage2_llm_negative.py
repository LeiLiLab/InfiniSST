#!/bin/env python3
"""
Stage 2 (v4 LLM-driven): generate "sound similar" negative terms via LLM and build final term_map.

Improvements:
1. Strict "Valid English" constraint in prompt.
2. Noun/Phrase validation via spaCy.
3. Multi-pass retry logic to ensure distractor count.
4. Streaming processing to prevent data loss on crashes.
"""

import os
import sys
import json
import re
import random
import argparse
import logging
import ast
import multiprocessing as mp
import zhconv
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict

from tqdm import tqdm

# vLLM setup
os.environ.setdefault("VLLM_USE_V1", "0")

logger = logging.getLogger(__name__)

DEFAULT_INPUT_GT_JSONL = "/mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_v2.jsonl"
DEFAULT_OUTPUT_BASE = "/mnt/gemini/data1/jiaxuanluo/train_s_zh_with_candidates_v4_llm_negative"

DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
DEFAULT_GPU_MEMORY_UTIL = 0.9
DEFAULT_BATCH_SIZE = 32

def _lang_name_from_code(lang_code: str) -> str:
    lc = (lang_code or "").strip().lower()
    if lc in ("zh", "zh-cn", "zh-hans", "zh-hant"):
        return "Chinese"
    if lc in ("ja", "jp"):
        return "Japanese"
    if lc in ("de",):
        return "German"
    # Fallback for other languages; keep English prompt.
    return "the target language"


NEGATIVE_GEN_PROMPT = """You are a terminology distraction generator for simultaneous interpretation training.

Task:
For each given English Ground Truth (GT) term and its {target_lang_name} translation, generate {num_distractors} **VALID English nouns or noun phrases** as distractors.

Rules:
1. **Valid English Only**: DO NOT invent non-existent words (e.g., "AUSTRALIAZ", "COVIDIA" are FORBIDDEN). Distractors MUST be real English words or established phrases found in a dictionary.
2. **Similarity Priority**: 
   - **Priority 1**: Sound-similar or spelling-similar valid words (e.g., "Election" -> "Selection", "Taiwan" -> "Thailand").
   - **Priority 2**: If no sound-similar valid words exist, choose terms from the same semantic category (e.g., "Australia" -> "Austria", "Germany", or "Oceania").
3. **Distinct Meaning**: The distractors must have a different meaning from the GT term.
4. **Plausible Translation**: Provide a correct and professional {target_lang_name} translation for each distractor.
5. **Format**: Return STRICT JSON only.

Input:
{gt_json}

Return JSON with this schema:
{{
  "negatives": [
    {{
      "gt_term": "<original_gt_term>",
      "distractors": [
        {{"term": "<valid_distractor_1>", "translation": "<translation_1>"}},
        ...
      ]
    }}
  ]
}}
"""

def _load_vocab():
    """Load English vocabulary from system dictionary or NLTK as fallback."""
    # Option A: System dictionary (fastest, no dependencies)
    dict_paths = ["/usr/share/dict/words", "/usr/share/dict/american-english", "/usr/share/dict/british-english"]
    for path in dict_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return set(word.strip().lower() for word in f)
            except Exception as e:
                logger.warning(f"Error reading dictionary at {path}: {e}")

    # Option B: NLTK fallback
    try:
        import nltk
        try:
            from nltk.corpus import words
            return set(w.lower() for w in words.words())
        except LookupError:
            logger.info("Downloading NLTK words corpus...")
            nltk.download('words')
            from nltk.corpus import words
            return set(w.lower() for w in words.words())
    except ImportError:
        pass

    logger.warning("No dictionary found! Validation will be skipped.")
    return None

def safe_parse_json(text: str) -> Dict:
    txt = (text or "").strip()
    if not txt: return {}
    candidates = []
    for m in re.finditer(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", txt):
        candidates.append(m.group(1))
    match_all = re.search(r"(\{[\s\S]*\})", txt)
    if match_all: candidates.append(match_all.group(1))
    for cand in candidates:
        cand = cand.strip()
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict): return obj
        except Exception: pass
        try:
            obj = ast.literal_eval(cand)
            if isinstance(obj, dict): return obj
        except Exception: pass
    return {}

class VLLMNegativeGenerator:
    def __init__(self, model: str, gpu_util: float, tensor_parallel_size: int = 1):
        from vllm import LLM, SamplingParams
        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=4096,
            gpu_memory_utilization=gpu_util,
            trust_remote_code=True,
            enforce_eager=True,
            disable_custom_all_reduce=True,
        )
        self.SamplingParams = SamplingParams

    def generate_batch(
        self,
        items: List[List[Dict[str, str]]],
        num_distractors: int = 9,
        target_lang_name: str = "Chinese",
    ) -> List[Dict[str, List[Dict[str, str]]]]:
        prompts = []
        for gt_list in items:
            gt_json = json.dumps(gt_list, ensure_ascii=False)
            user_content = NEGATIVE_GEN_PROMPT.format(
                num_distractors=num_distractors,
                gt_json=gt_json,
                target_lang_name=target_lang_name,
            )
            full_prompt = f"<|im_start|>system\nYou are a terminology distraction generator.<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
            prompts.append(full_prompt)
        
        sp = self.SamplingParams(
            temperature=0.7, # Increased slightly for variety in retries
            max_tokens=2048,
            repetition_penalty=1.1,
            stop=["<|im_end|>", "<|endoftext|>", "```\n"]
        )
        outputs = self.llm.generate(prompts, sp)
        
        results = []
        for out in outputs:
            raw_text = out.outputs[0].text if out.outputs else ""
            obj = safe_parse_json(raw_text)
            mapping = {}
            for entry in obj.get("negatives", []):
                gt_term = entry.get("gt_term")
                distractors = entry.get("distractors", [])
                if gt_term and distractors:
                    mapping[gt_term] = distractors
            results.append(mapping)
        return results

def validate_distractors(vocab_set: Optional[Set[str]], distractors: List[Dict[str, str]], gt_term: str) -> List[Dict[str, str]]:
    """
    Lightweight validation using dictionary lookup.
    """
    valid = []
    seen = {gt_term.lower()}
    
    for d in distractors:
        if not isinstance(d, dict):
            continue
        term = d.get("term", "").strip()
        # Backward-compatible: accept {"translation": ...} (preferred) or {"<lang_code>": ...} or {"zh": ...}.
        tr = (d.get("translation") or d.get("zh") or "").strip()
        
        if not term or not tr:
            continue
        if term.lower() in seen:
            continue
        if len(term) < 2:
            continue
            
        is_valid = True
        if vocab_set:
            sub_words = re.split(r"[\s\-]+", term)
            for w in sub_words:
                clean_w = "".join(filter(str.isalpha, w))
                if not clean_w: continue
                
                lower_w = clean_w.lower()
                if clean_w.islower():
                    if lower_w not in vocab_set:
                        is_valid = False
                        break
                else:
                    if clean_w[0].isupper() and (len(clean_w) == 1 or clean_w[1:].islower()):
                        continue
                    if clean_w.isupper():
                        if lower_w in vocab_set or len(clean_w) <= 4:
                            continue
                        else:
                            is_valid = False
                            break
                    if lower_w not in vocab_set:
                        is_valid = False
                        break

        if is_valid:
            valid.append({"term": term, "translation": tr})
            seen.add(term.lower())
    return valid

def generate_term_map_string(terms: List[Tuple[str, str]]) -> str:
    if not terms: return ""
    lines = ["term_map:"]
    for s, t in terms:
        lines.append(f"{s}={t}")
    return "\n".join(lines)


def _pick_gt_translation(gt: dict, target_lang_code: str) -> str:
    """
    Pick target translation string from a GT term dict in a language-agnostic way.
    Backward compatible with older zh-only format using "zh".
    """
    if not isinstance(gt, dict):
        return ""
    lang = (target_lang_code or "").strip()
    for k in (lang, "translation", "zh"):
        v = gt.get(k)
        if v:
            return str(v)
    return ""


def _normalize_target_text(text: str, target_lang_code: str) -> str:
    """
    Normalize target translation string for a given language.
    - For zh: convert to simplified Chinese (zh-cn) for consistency.
    - For other languages: keep original.
    """
    s = (text or "").strip()
    lc = (target_lang_code or "").strip().lower()
    if lc in ("zh", "zh-cn", "zh-hans"):
        return zhconv.convert(s, "zh-cn")
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-gt-jsonl", default=DEFAULT_INPUT_GT_JSONL)
    parser.add_argument("--output-base", default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gpu-memory-util", type=float, default=DEFAULT_GPU_MEMORY_UTIL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--num-distractors", type=int, default=9, help="Target distractors per GT term")
    parser.add_argument("--max-messages", type=int, default=None)
    parser.add_argument("--all-negative-ratio", type=float, default=0.1)
    parser.add_argument("--multiple-range", type=int, nargs=2, default=[0, 9])
    parser.add_argument("--target-lang-code", type=str, default="zh", help="Target language code (e.g., zh, ja, de).")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--total-gpus", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    vocab_set = _load_vocab()
    
    generator = VLLMNegativeGenerator(args.model, args.gpu_memory_util, tensor_parallel_size=args.tensor_parallel_size)
    target_lang_name = _lang_name_from_code(args.target_lang_code)
    
    def process_batch_single_pass(items_to_proc, target_count):
        if not items_to_proc:
            return []
        
        raw_results = generator.generate_batch(items_to_proc, num_distractors=target_count, target_lang_name=target_lang_name)
        
        batch_mappings = []
        for i, mapping in enumerate(raw_results):
            gt_list = items_to_proc[i]
            res_map = {}
            for gt_item in gt_list:
                term = gt_item["term"]
                distractors = mapping.get(term, [])
                valid_distractors = validate_distractors(vocab_set, distractors, term)
                res_map[term] = valid_distractors
            batch_mappings.append(res_map)
            
        return batch_mappings

    def augment_records(records, all_neg_maps):
        results = []
        for rec, neg_map in zip(records, all_neg_maps):
            messages = rec.get("messages", [])
            audios = rec.get("audios", [])
            gt_chunks = rec.get("gt_terms_by_chunk", []) or []
            
            new_messages = []
            audio_turn_idx = 0
            for m in messages:
                if m.get("role") == "user" and m.get("content") == "<audio>":
                    if audio_turn_idx >= len(audios):
                        new_messages.append(m)
                        continue
                    
                    gt_list = gt_chunks[audio_turn_idx] if audio_turn_idx < len(gt_chunks) else []
                    gt_unique = []
                    seen_gt = set()
                    for it in gt_list:
                        s = it.get("term")
                        t = _normalize_target_text(_pick_gt_translation(it, args.target_lang_code), args.target_lang_code)
                        if s and t and s.lower() not in seen_gt:
                            gt_unique.append((s, t))
                            seen_gt.add(s.lower())
                    
                    if not gt_unique:
                        new_messages.append(m)
                        audio_turn_idx += 1
                        continue
                    
                    hard_neg_pool = []
                    for s, _ in gt_unique:
                        distractors = neg_map.get(s, [])
                        for d in distractors:
                            dt = (d.get("term") or "").strip()
                            dtr = _normalize_target_text((d.get("translation") or d.get("zh") or "").strip(), args.target_lang_code)
                            if dt and dtr and dt.lower() not in seen_gt:
                                hard_neg_pool.append((dt, dtr))
                    
                    base_count = len(gt_unique)
                    multiple = random.randint(args.multiple_range[0], args.multiple_range[1])
                    
                    if not hard_neg_pool:
                        final_terms = list(gt_unique)
                    elif random.random() < args.all_negative_ratio:
                        target_len = max(1, base_count + (base_count * multiple))
                        random.shuffle(hard_neg_pool)
                        final_terms = hard_neg_pool[:target_len]
                    else:
                        num_neg = base_count * multiple
                        random.shuffle(hard_neg_pool)
                        final_terms = list(gt_unique) + hard_neg_pool[:num_neg]
                    
                    random.shuffle(final_terms)
                    seen = set()
                    deduped = []
                    for s, t in final_terms:
                        if s.lower() not in seen:
                            deduped.append((s, t))
                            seen.add(s.lower())
                    
                    term_map_str = generate_term_map_string(deduped)
                    new_messages.append({"role": "user", "content": f"<audio>\n\n{term_map_str}"})
                    audio_turn_idx += 1
                else:
                    new_messages.append(m)
            results.append({"messages": new_messages, "audios": audios})
        return results

    output_file = f"{args.output_base}_gpu{args.gpu_id}.jsonl" if args.total_gpus > 1 else f"{args.output_base}.jsonl"
    
    # We will store records and their distractor maps in memory to allow a second pass
    all_records = []
    all_neg_maps = []
    
    with open(args.input_gt_jsonl, "r", encoding="utf-8") as f_in:
        logger.info(f"Pass 1: Initial generation...")
        
        current_batch_records = []
        current_batch_gt_sets = []
        
        # Count total lines for better progress monitoring if possible
        total_lines = None
        if args.gpu_id == 0:
            try:
                import subprocess
                res = subprocess.run(["wc", "-l", args.input_gt_jsonl], capture_output=True, text=True)
                total_lines = int(res.stdout.split()[0])
                if args.total_gpus > 1:
                    total_lines = total_lines // args.total_gpus
            except:
                pass

        pbar = tqdm(total=args.max_messages or total_lines, desc=f"GPU {args.gpu_id} Pass 1")
        
        for idx, line in enumerate(f_in):
            if args.total_gpus > 1 and idx % args.total_gpus != args.gpu_id:
                continue
            if args.max_messages and len(all_records) >= args.max_messages:
                break
            
            rec = json.loads(line)
            chunk_terms = rec.get("gt_terms_by_chunk", [])
            sample_gt = []
            seen = set()
            for chunk in chunk_terms:
                for item in chunk:
                    t = item.get("term")
                    if t and t not in seen:
                        tr = _normalize_target_text(_pick_gt_translation(item, args.target_lang_code), args.target_lang_code)
                        sample_gt.append({"term": t, "translation": tr})
                        seen.add(t)
            
            current_batch_records.append(rec)
            current_batch_gt_sets.append(sample_gt)
            
            if len(current_batch_records) >= args.batch_size:
                valid_indices = [i for i, b in enumerate(current_batch_gt_sets) if b]
                valid_batch_gt = [current_batch_gt_sets[i] for i in valid_indices]
                
                batch_res_maps = [{} for _ in current_batch_gt_sets]
                if valid_batch_gt:
                    pass1_results = process_batch_single_pass(valid_batch_gt, target_count=args.num_distractors)
                    for v_idx, r_map in zip(valid_indices, pass1_results):
                        batch_res_maps[v_idx] = r_map
                
                all_records.extend(current_batch_records)
                all_neg_maps.extend(batch_res_maps)
                
                pbar.update(len(current_batch_records))
                current_batch_records = []
                current_batch_gt_sets = []
        
        if current_batch_records:
            valid_indices = [i for i, b in enumerate(current_batch_gt_sets) if b]
            valid_batch_gt = [current_batch_gt_sets[i] for i in valid_indices]
            batch_res_maps = [{} for _ in current_batch_gt_sets]
            if valid_batch_gt:
                pass1_results = process_batch_single_pass(valid_batch_gt, target_count=args.num_distractors)
                for v_idx, r_map in zip(valid_indices, pass1_results):
                    batch_res_maps[v_idx] = r_map
            all_records.extend(current_batch_records)
            all_neg_maps.extend(batch_res_maps)
            pbar.update(len(current_batch_records))
            
        pbar.close()

    # Pass 2: Retry failed ones in a large batch
    to_retry_indices = []
    to_retry_gt = []
    
    for i, (rec, neg_map) in enumerate(zip(all_records, all_neg_maps)):
        chunk_terms = rec.get("gt_terms_by_chunk", [])
        missing_gt = []
        for chunk in chunk_terms:
            for item in chunk:
                term = item.get("term")
                if not term: continue
                if len(neg_map.get(term, [])) < args.num_distractors:
                            tr = _normalize_target_text(_pick_gt_translation(item, args.target_lang_code), args.target_lang_code)
                            missing_gt.append({"term": term, "translation": tr})
        
        if missing_gt:
            to_retry_indices.append(i)
            to_retry_gt.append(missing_gt)
    
    if to_retry_gt:
        logger.info(f"Pass 2: Retrying {len(to_retry_gt)} samples with insufficient distractors...")
        pbar2 = tqdm(total=len(to_retry_gt), desc=f"GPU {args.gpu_id} Pass 2")
        
        # Process Pass 2 in batches
        for i in range(0, len(to_retry_gt), args.batch_size):
            batch_gt = to_retry_gt[i : i + args.batch_size]
            batch_indices = to_retry_indices[i : i + args.batch_size]
            
            pass2_results = process_batch_single_pass(batch_gt, target_count=args.num_distractors)
            
            for idx_in_batch, new_map in enumerate(pass2_results):
                orig_idx = batch_indices[idx_in_batch]
                # Merge new distractors into existing neg_map
                for term, new_distractors in new_map.items():
                    existing = all_neg_maps[orig_idx].get(term, [])
                    seen_terms = {d["term"].lower() for d in existing}
                    for nd in new_distractors:
                        if nd["term"].lower() not in seen_terms:
                            existing.append(nd)
                            seen_terms.add(nd["term"].lower())
                    all_neg_maps[orig_idx][term] = existing
            
            pbar2.update(len(batch_gt))
        pbar2.close()
    else:
        logger.info("No samples need Pass 2.")

    # Write final results
    logger.info(f"Writing final results to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f_out:
        final_augmented = augment_records(all_records, all_neg_maps)
        for rec in final_augmented:
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info("Done. Output written to %s", output_file)

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
