#!/usr/bin/env python3
"""
Generate sound-like and shape-like hard negatives for terms using LLM (vLLM).
Filters out non-English terms using a dictionary.
"""

import os
import json
import re
import argparse
import logging
import ast
from typing import Dict, List, Set, Optional
from tqdm import tqdm

# vLLM setup
os.environ.setdefault("VLLM_USE_V1", "0")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

LLM_HN_GEN_PROMPT = """You are a terminology distraction generator for simultaneous interpretation training.

Task:
For each given English term, generate {num_negatives} hard negatives following this priority:

1. **Top Priority (Best)**: Words that are **BOTH** sound-alike (phonetically similar) AND shape-alike (visually/spelling similar) to the original term. (e.g., "compliment" -> "complement", "principal" -> "principle").
2. **Secondary Priority**: If Top Priority words are unavailable, provide words that are **EITHER** sound-alike OR shape-alike.
   - **Sound-alike**: Words that sound similar but may look different.
   - **Shape-alike**: Words that look similar or have very similar spelling but may sound different.

Rules:
1. **Valid English Only**: DO NOT invent non-existent words. Distractors MUST be real English words or established phrases.
2. **Different Meaning**: The distractors must have a different meaning from the original term.
3. **Format**: Return STRICT JSON only.

Input:
{terms_json}

Return JSON with this schema:
{{
  "results": [
    {{
      "term": "<original_term>",
      "hard_negatives": ["<neg1>", "<neg2>", ...]
    }},
    ...
  ]
}}
"""

def _load_vocab():
    """Load English vocabulary from system dictionary."""
    dict_paths = ["/usr/share/dict/words", "/usr/share/dict/american-english", "/usr/share/dict/british-english"]
    for path in dict_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return set(word.strip().lower() for word in f)
            except Exception as e:
                logger.warning(f"Error reading dictionary at {path}: {e}")

    try:
        import nltk
        try:
            from nltk.corpus import words
            return set(w.lower() for w in words.words())
        except LookupError:
            nltk.download('words')
            from nltk.corpus import words
            return set(w.lower() for w in words.words())
    except ImportError:
        pass

    logger.warning("No dictionary found! English validation will be loose.")
    return None

def is_valid_english(term: str, vocab_set: Optional[Set[str]]) -> bool:
    """Check if a term or phrase consists of valid English words."""
    if not term:
        return False
    if not vocab_set:
        return True
    
    # Check if it contains letters (to filter out pure symbols/numbers if any)
    if not any(c.isalpha() for c in term):
        return False

    # Split into words and check each
    sub_words = re.split(r"[\s\-]+", term)
    for w in sub_words:
        # Remove punctuation for lookup
        clean_w = "".join(filter(str.isalpha, w)).lower()
        if not clean_w:
            continue
        if clean_w not in vocab_set:
            # Check for common plurals or endings if not in vocab
            if clean_w.endswith('s') and clean_w[:-1] in vocab_set: continue
            if clean_w.endswith('es') and clean_w[:-2] in vocab_set: continue
            if clean_w.endswith('ed') and clean_w[:-2] in vocab_set: continue
            if clean_w.endswith('ing') and clean_w[:-3] in vocab_set: continue
            return False
    return True

def safe_parse_json(text: str) -> Dict:
    txt = (text or "").strip()
    if not txt: return {}
    candidates = []
    # Try finding JSON in code blocks
    for m in re.finditer(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", txt):
        candidates.append(m.group(1))
    # Try finding any { ... } block
    match_all = re.search(r"(\{[\s\S]*\})", txt)
    if match_all: candidates.append(match_all.group(1))
    
    for cand in candidates:
        cand = cand.strip()
        try:
            return json.loads(cand)
        except Exception: pass
        try:
            return ast.literal_eval(cand)
        except Exception: pass
    return {}

class VLLMHNGenerator:
    def __init__(self, model: str, gpu_util: float, tensor_parallel_size: int = 1):
        from vllm import LLM, SamplingParams
        
        # 针对 8 卡并行的稳健配置
        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=4096,
            gpu_memory_utilization=gpu_util,
            trust_remote_code=True,
            # 强制使用 Eager 模式，跳过耗时且易崩的 CUDA Graph 捕获
            enforce_eager=True,
            # 明确指定分布式后端
            distributed_executor_backend="mp",
        )
        self.SamplingParams = SamplingParams

    def generate_batch(self, list_of_batches: List[List[str]], num_negatives: int = 5) -> List[Dict[str, List[str]]]:
        prompts = []
        for batch_terms in list_of_batches:
            terms_json = json.dumps(batch_terms, ensure_ascii=False)
            user_content = LLM_HN_GEN_PROMPT.format(num_negatives=num_negatives, terms_json=terms_json)
            full_prompt = f"<|im_start|>system\nYou are a terminology distraction generator.<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
            prompts.append(full_prompt)
        
        sp = self.SamplingParams(
            temperature=0.3, # Low temperature for consistency
            max_tokens=2048,
            stop=["<|im_end|>", "<|endoftext|>"]
        )
        
        logger.info(f"Sending {len(prompts)} prompts to vLLM engine...")
        outputs = self.llm.generate(prompts, sp)
        
        batch_results = []
        for out in outputs:
            raw_text = out.outputs[0].text if out.outputs else ""
            obj = safe_parse_json(raw_text)
            mapping = {}
            for entry in obj.get("results", []):
                term_raw = entry.get("term")
                hns = entry.get("hard_negatives", [])
                # Normalize term key to match training-side lookup (.strip().lower()).
                term_key = (term_raw or "").strip().lower()
                if term_key and isinstance(hns, list) and len(hns) > 0:
                    # Normalize candidates early to reduce mismatch + simplify downstream filtering.
                    cleaned = []
                    for hn in hns:
                        if not isinstance(hn, str):
                            continue
                        s = hn.strip()
                        if s:
                            cleaned.append(s)
                    if cleaned:
                        mapping[term_key] = cleaned
            batch_results.append(mapping)
        return batch_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--gpu_util", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=10, help="Number of terms per prompt")
    parser.add_argument("--vllm_batch_size", type=int, default=500, help="Number of prompts per vLLM call")
    parser.add_argument("--num_negatives", type=int, default=10, help="Number of distractors to generate per term")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--shard_id", type=int, default=0, help="ID of the current shard")
    parser.add_argument("--total_shards", type=int, default=1, help="Total number of shards")
    args = parser.parse_args()

    vocab_set = _load_vocab()
    # 强制将 tp 设为 1，因为每个进程只管一张卡
    generator = VLLMHNGenerator(args.model, args.gpu_util, tensor_parallel_size=1)

    # 1. Load unique terms
    logger.info(f"Loading unique terms from {args.train_jsonl}...")
    unique_terms = set()
    with open(args.train_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                term = item["term"].strip().lower()
                if term:
                    unique_terms.add(term)
            except: continue
    
    all_term_list = sorted(list(unique_terms))
    
    # 根据 shard 分片
    if args.total_shards > 1:
        shard_size = (len(all_term_list) + args.total_shards - 1) // args.total_shards
        start_idx = args.shard_id * shard_size
        end_idx = min(start_idx + shard_size, len(all_term_list))
        term_list = all_term_list[start_idx:end_idx]
        logger.info(f"Shard {args.shard_id}/{args.total_shards}: Processing {len(term_list)} terms (indices {start_idx}-{end_idx})")
    else:
        term_list = all_term_list
        logger.info(f"Processing all {len(term_list)} unique terms.")

    # 2. Generate HNs in batches
    hard_negatives: Dict[str, List[str]] = {}
    stats_total_terms = 0
    stats_has_raw = 0
    stats_has_valid = 0
    
    # 根据分片调整输出路径
    final_output_path = args.output_path
    if args.total_shards > 1:
        final_output_path = args.output_path.replace(".json", f"_shard_{args.shard_id}.json")
    
    # Each vLLM call will handle (args.batch_size * args.vllm_batch_size) terms
    terms_per_vllm_call = args.batch_size * args.vllm_batch_size
    pbar = tqdm(total=len(term_list), desc="Generating LLM HNs")
    
    for i in range(0, len(term_list), terms_per_vllm_call):
        vllm_chunk_terms = term_list[i : i + terms_per_vllm_call]
        
        # Split chunk into multiple prompts
        list_of_batches = []
        for j in range(0, len(vllm_chunk_terms), args.batch_size):
            list_of_batches.append(vllm_chunk_terms[j : j + args.batch_size])
            
        if not list_of_batches:
            continue
            
        results_list = generator.generate_batch(list_of_batches, num_negatives=args.num_negatives)
        
        # Aggregate results
        for idx, results in enumerate(results_list):
            batch = list_of_batches[idx]
            for term in batch:
                stats_total_terms += 1
                raw_hns = results.get(term, [])
                if raw_hns:
                    stats_has_raw += 1
                valid_hns = []
                seen = {term}
                for hn in raw_hns:
                    if not isinstance(hn, str):
                        continue
                    hn_clean = hn.strip().lower()
                    if hn_clean and hn_clean not in seen and is_valid_english(hn_clean, vocab_set):
                        # Store normalized lowercase strings to match training pipeline expectations.
                        valid_hns.append(hn_clean)
                        seen.add(hn_clean)
                if valid_hns:
                    stats_has_valid += 1
                    hard_negatives[term] = valid_hns
            
        pbar.update(len(vllm_chunk_terms))
        
        # Periodic saving
        if (i // terms_per_vllm_call) % 10 == 0:
            with open(final_output_path + ".tmp", "w", encoding="utf-8") as f:
                json.dump(hard_negatives, f, ensure_ascii=False, indent=2)
            if stats_total_terms > 0:
                logger.info(
                    f"Stats: total_terms={stats_total_terms} has_raw={stats_has_raw} ({stats_has_raw/max(1,stats_total_terms):.2%}) "
                    f"has_valid={stats_has_valid} ({stats_has_valid/max(1,stats_total_terms):.2%}) saved_keys={len(hard_negatives)}"
                )

    pbar.close()

    # 3. Final Save
    logger.info(f"Saving final results to {final_output_path}...")
    with open(final_output_path, "w", encoding="utf-8") as f:
        json.dump(hard_negatives, f, ensure_ascii=False, indent=2)
    
    if os.path.exists(final_output_path + ".tmp"):
        os.remove(final_output_path + ".tmp")
        
    logger.info("Done!")

if __name__ == "__main__":
    main()

