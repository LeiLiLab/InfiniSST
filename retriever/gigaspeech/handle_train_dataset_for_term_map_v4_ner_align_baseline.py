#!/usr/bin/env python3
"""
Stage 1.4 (NER Alignment from Baseline)

1) 直接从 baseline JSONL 开始处理。
2) 从 TSV 的 src_text 识别实体 (PERSON/LOC/GPE/NORP/FAC/ORG)，带有 ALL-CAPS 归一化。
3) 用 LLM 对实体进行对齐。
4) 结合轨迹做分块定位，写回包含 gt_terms_by_chunk 的 JSONL。
"""

import os
import sys
import json
import re
import argparse
import logging
import random
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter

from tqdm import tqdm

# vLLM setup
os.environ.setdefault("VLLM_USE_V1", "0")

logger = logging.getLogger(__name__)

# ----------------------------
# Configuration
# ----------------------------
DEFAULT_INPUT_GT = "/mnt/gemini/data1/jiaxuanluo/train_s_zh_baseline.jsonl"
DEFAULT_INPUT_TSV = "/mnt/gemini/data1/jiaxuanluo/train_xl_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv"
DEFAULT_OUTPUT_GT = "/mnt/gemini/data1/jiaxuanluo/train_s_zh_v3_gt_terms_ner.jsonl"

DEFAULT_ALIGN_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
DEFAULT_ALIGN_GPU_MEMORY_UTIL = 0.9
# ----------------------------
# Prompt
# ----------------------------
ENTITY_ALIGN_PROMPT = """You are a translation alignment assistant.

Task:
Find the corresponding {tgt_name} translation or equivalent term in the [{tgt_tag}] text for the given [ENG] Terms and Nouns.

Rules:
1. **Semantic Match**: Look for the meaning.
2. **Acronyms**: Look for the expanded or translated acronym.
3. **Transliteration**: Look for phonetic matches when applicable.
4. **Copying is Valid**: If the term is kept in English in [{tgt_tag}], output the English word as the translation value.
5. **Format**: Return STRICT JSON.

Examples:

Input:
[ENG]: The government announced new funding for public education.
[TGT]: (example target sentence)
Terms: ["government", "funding", "public education"]
Response:
{{"alignments":[{{"term":"government","translation":"..."}},{{"term":"funding","translation":"..."}},{{"term":"public education","translation":"..."}}]}}

Input:
[ENG]: {src_text}
[TGT]: {tgt_text}

Terms to Align:
{terms_json}

Return JSON object:
{{
  "alignments": [
    {{"term": "<eng_term>", "translation": "<tgt_span_or_null>"}}
  ]
}}
"""

RETRY_ENTITY_ALIGN_PROMPT = """You are a translation alignment assistant.
The following terms/nouns were NOT found in the previous pass. Please try HARDER to find them in the [ZH] text.
They might be translated differently, referred to by a synonym, or the translation might be slightly different.

Rules:
1. **Be Flexible**: Even if the translation isn't perfect, if it clearly refers to the [ENG] concept, align it.
2. **Format**: Return STRICT JSON.

Input:
[ENG]: {src_text}
[TGT]: {tgt_text}

Terms to Retry:
{terms_json}

Return JSON object:
{{
  "alignments": [
    {{"term": "<eng_term>", "translation": "<tgt_span_or_null>"}}
  ]
}}
"""


# ----------------------------
# Helpers
# ----------------------------
def get_next_version_path(path_str: str) -> str:
    path = Path(path_str)
    stem = path.stem
    ext = path.suffix
    # Look for vN in stem
    match = re.search(r'_v(\d+)$', stem)
    if match:
        v = int(match.group(1))
        new_stem = stem[:match.start()] + f'_v{v+1}'
    else:
        new_stem = stem + '_v2'
    return str(path.with_name(new_stem + ext))


def load_tsv_index(tsv_path: str) -> Dict[str, Dict]:
    logger.info("Loading TSV index...")
    index: Dict[str, Dict] = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        col_id = header.index("id") if "id" in header else 0
        col_src = header.index("src_text") if "src_text" in header else None
        col_tgt = header.index("tgt_text") if "tgt_text" in header else None
        col_traj = header.index("src_trajectory") if "src_trajectory" in header else None
        for line in tqdm(f, desc="TSV Index"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= col_id or col_src is None or col_tgt is None:
                continue
            uid = parts[col_id]
            src_text = parts[col_src]
            tgt_text = parts[col_tgt]
            traj = []
            if col_traj is not None and len(parts) > col_traj:
                try:
                    traj = eval(parts[col_traj])
                except Exception:
                    traj = []
            index[uid] = {"src_text": src_text, "tgt_text": tgt_text, "src_trajectory": traj}
    logger.info("TSV rows loaded: %d", len(index))
    return index


def extract_utter_id_from_audio_path(audio_path: str) -> Optional[str]:
    try:
        parts = Path(audio_path).parts
        if len(parts) >= 3:
            return f"{parts[-3]}_{parts[-2]}"
    except Exception:
        return None
    return None


def _load_spacy(model_name="en_core_web_trf"):
    import spacy
    try:
        # 使用 require_gpu() 强制检查，如果失败会抛出异常
        spacy.require_gpu()
        logger.info("spaCy SUCCESS: GPU activated.")
    except Exception as e:
        logger.warning(f"spaCy WARNING: Could not activate GPU ({e}). Falling back to CPU.")
        
    try:
        logger.info(f"Loading spaCy model: {model_name}...")
        return spacy.load(model_name)
    except Exception as e:
        logger.warning(f"Could not load {model_name}: {e}. Falling back to en_core_web_sm")
        return spacy.load("en_core_web_sm")


def normalize_text_casing(text: str) -> str:
    """Handle GigaSpeech-style ALL CAPS text."""
    if not text:
        return ""
    upper_count = sum(1 for c in text if c.isupper())
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count > 0 and (upper_count / alpha_count) > 0.7:
        return text.title()
    return text


def split_trajectory_by_chunks(trajectory: List[str], num_chunks: int, merge_multiplier: Optional[int] = None) -> List[List[str]]:
    if num_chunks <= 0:
        return []
    if not trajectory:
        return [[] for _ in range(num_chunks)]
    
    if merge_multiplier is not None:
        chunks = []
        for i in range(num_chunks):
            start = i * merge_multiplier
            end = min((i + 1) * merge_multiplier, len(trajectory))
            chunks.append(trajectory[start:end])
        return chunks
        
    chunk_size = (len(trajectory) + num_chunks - 1) // num_chunks
    return [trajectory[i * chunk_size : min((i + 1) * chunk_size, len(trajectory))] for i in range(num_chunks)]


def locate_term_chunks_robust(src_chunks: List[str], term: str) -> List[int]:
    """Locate term in all possible chunks; supports cross-chunk by checking i and i+1 combined. Fallback to chunk 0."""
    tl = term.strip().lower()
    indices = []
    # 1. Check each chunk individually
    for i, ch in enumerate(src_chunks):
        if tl in (ch or "").lower():
            indices.append(i)
            
    # 2. Check cross-chunk boundaries
    for i in range(len(src_chunks) - 1):
        combined = re.sub(r"\s+", " ", ((src_chunks[i] or "") + " " + (src_chunks[i + 1] or "")))
        if tl in combined.lower():
            # If it's already in chunk i or i+1, we might not need to add it again,
            # but if it spans across both and wasn't found in either fully, we add i+1.
            if i not in indices and (i+1) not in indices:
                indices.append(i + 1)
                
    if not indices:
        return [0]
    return sorted(list(set(indices)))


import ast

def safe_parse_json(text: str) -> Dict:
    txt = (text or "").strip()
    if not txt:
        return {}
    candidates = []
    for m in re.finditer(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", txt):
        candidates.append(m.group(1))
    match_all = re.search(r"(\{[\s\S]*\})", txt)
    if match_all:
        candidates.append(match_all.group(1))
    for cand in candidates:
        cand = cand.strip()
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict) and "alignments" in obj:
                return obj
        except Exception:
            pass
        try:
            obj = ast.literal_eval(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    blocks = re.split(r"```(?:json)?", txt)
    for b in blocks:
        b = b.strip()
        if not b: continue
        match = re.search(r"(\{[\s\S]*\})", b)
        if match:
            try:
                obj = json.loads(match.group(1))
                if isinstance(obj, dict): return obj
            except Exception:
                pass
    return {}


# ----------------------------
# vLLM Aligner
# ----------------------------
class VLLMAligner:
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

    def align_batch(
        self,
        items: List[Tuple[str, str, List[str]]],
        tgt_name: str,
        tgt_tag: str,
        max_tokens: int = 1024,
        is_retry: bool = False,
    ) -> List[Dict[str, str]]:
        prompts = []
        prompt_tmpl = RETRY_ENTITY_ALIGN_PROMPT if is_retry else ENTITY_ALIGN_PROMPT
        for src, tgt, names in items:
            user_content = prompt_tmpl.format(
                src_text=src,
                tgt_text=tgt,
                terms_json=json.dumps(names, ensure_ascii=False),
                tgt_name=tgt_name,
                tgt_tag=tgt_tag,
            )
            full_prompt = f"<|im_start|>system\nYou are a translation alignment assistant.<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
            prompts.append(full_prompt)
        sp = self.SamplingParams(
            temperature=0.0, 
            max_tokens=max_tokens,
            repetition_penalty=1.1,
            stop=["<|im_end|>", "<|endoftext|>", "```\n", "```\r"]
        )
        outputs = self.llm.generate(prompts, sp)
        results: List[Dict[str, str]] = []
        for out in outputs:
            try:
                txt = out.outputs[0].text if out.outputs else ""
                obj = safe_parse_json(txt)
                mapping: Dict[str, str] = {}
                for a in obj.get("alignments", []):
                    if not isinstance(a, dict):
                        continue
                    term = a.get("term")
                    translation = a.get("translation")
                    if not term or translation is None:
                        continue
                    term = str(term).strip()
                    translation = str(translation).strip()
                    if not term or not translation:
                        continue
                    if len([w for w in re.split(r"\s+", term) if w]) > 3:
                        continue
                    mapping[term] = translation
                results.append(mapping)
            except Exception:
                results.append({})
        return results


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-gt", default=DEFAULT_INPUT_GT)
    parser.add_argument("--input-tsv", default=DEFAULT_INPUT_TSV)
    parser.add_argument("--output-gt", default=DEFAULT_OUTPUT_GT)
    parser.add_argument("--align-model", default=DEFAULT_ALIGN_MODEL)
    parser.add_argument("--gpu-memory-util", type=float, default=DEFAULT_ALIGN_GPU_MEMORY_UTIL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    # Sharding (multi-process / multi-GPU): consistent with other pipeline stages.
    # Note: CUDA device selection is controlled by CUDA_VISIBLE_DEVICES in the runner script.
    parser.add_argument("--gpu-id", type=int, default=0, help="Shard index for dataset partitioning.")
    parser.add_argument("--total-gpus", type=int, default=1, help="Total shards for dataset partitioning.")
    parser.add_argument("--max-messages", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=1, help="Number of retry passes for rejected terms.")
    parser.add_argument("--sampling-rate", type=float, default=0.3, help="[Deprecated by freq-sampling] Sample X% of extracted nouns.")
    parser.add_argument("--max-terms-per-utter", type=int, default=20, help="Max terms to keep per utterance (K).")
    parser.add_argument("--spacy-model", default="en_core_web_trf", help="SpaCy model to use for noun extraction.")
    parser.add_argument("--no-freq-sampling", dest="enable_freq_sampling", action="store_false", help="Disable frequency-based sampling.")
    parser.set_defaults(enable_freq_sampling=True)
    parser.add_argument("--ner-candidates-path", help="Path to pre-extracted NER candidates JSONL.")
    parser.add_argument("--target-lang-code", default="zh", help="Target language code (zh/ja/de).")
    args = parser.parse_args()

    tgt_code = (args.target_lang_code or "zh").strip().lower()
    tgt_name_map = {"zh": "Chinese", "ja": "Japanese", "de": "German"}
    tgt_name = tgt_name_map.get(tgt_code, tgt_code)
    tgt_tag = tgt_code.upper()
    # Only normalize Chinese outputs; avoid importing zhconv for other languages.
    zh_converter = None
    if tgt_code == "zh":
        try:
            import zhconv  # type: ignore
            zh_converter = zhconv
        except Exception:
            zh_converter = None

    # 加载预提取的候选词
    ner_map = {}
    if args.ner_candidates_path and os.path.exists(args.ner_candidates_path):
        logger.info(f"Loading NER candidates from {args.ner_candidates_path}...")
        with open(args.ner_candidates_path, "r", encoding="utf-8") as f_ner:
            for line in f_ner:
                item = json.loads(line)
                ner_map[item["utter_id"]] = item["ner_candidates"]
    
    index = load_tsv_index(args.input_tsv)
    # 移除 nlp 加载
    
    # Common English stopwords to ignore
    STOPWORDS = {"the", "a", "an", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their", "it", "they", "them", "who", "whom", "whose"}

    def get_filtered_candidates(doc):
        candidates = []
        # 1. 提取命名实体 (NER) - 解决人名、机构名被拆分的问题
        # 重点关注 PERSON, ORG, GPE, LOC, FAC, PRODUCT, EVENT
        for ent in doc.ents:
            if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "FAC", "PRODUCT", "EVENT"}:
                candidates.append(ent.text.strip())

        # 2. Noun chunks (e.g. "public education")
        for chunk in doc.noun_chunks:
            text = chunk.text.strip()
            # Strip leading articles
            toks = text.split()
            if len(toks) > 1 and toks[0].lower() in {"the", "a", "an"}:
                text = " ".join(toks[1:])
            candidates.append(text)
        
        # 3. Single NOUN or PROPN not caught in chunks/entities
        for token in doc:
            if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop:
                candidates.append(token.text.strip())
        
        # Filter candidates: remove short ones, stopwords, and limit word count
        raw_unique = set(candidates)
        filtered_basic = []
        for cand in raw_unique:
            c = cand.strip()
            if not c or len(c) < 3: continue
            if c.lower() in STOPWORDS: continue
            if len(c.split()) > 4: continue # 允许最长 4 个单词 (如 "The New York Times")
            filtered_basic.append(c)
            
        # 移除子集字符串 (例如有了 "John Mearsheimer"，就移除 "John" 和 "Mearsheimer")
        # 按照长度从大到小排序，优先保留长的
        filtered_basic.sort(key=len, reverse=True)
        final_filtered = []
        for i, cand in enumerate(filtered_basic):
            is_subset = False
            for j, other in enumerate(filtered_basic):
                if i == j: continue
                # 检查 cand 是否是 other 的一部分 (不区分大小写)
                # 必须是独立的单词包含，防止 "Iran" 被 "Iranian" 过滤
                c_low = cand.lower()
                o_low = other.lower()
                if c_low in o_low:
                    # 使用正则检查是否为完整单词包含
                    if re.search(r'\b' + re.escape(c_low) + r'\b', o_low):
                        is_subset = True
                        break
            if not is_subset:
                final_filtered.append(cand)

        return final_filtered

    def keep_prob(f):
        if f <= 2:  return 1.0
        if f <= 5:  return 0.8
        if f <= 20: return 0.3
        if f <= 50: return 0.1
        return 0.02

    # ----------------------------
    # Pass 0: Global Frequency Counting
    # ----------------------------
    term_freq = Counter()
    if args.enable_freq_sampling:
        logger.info("Pass 0: Counting global term frequencies...")
        
        def src_text_generator():
            count = 0
            with open(args.input_gt, "r", encoding="utf-8") as f_pass0:
                for line in f_pass0:
                    if args.max_messages is not None and count >= args.max_messages:
                        break
                    obj = json.loads(line)
                    uid = obj.get("utter_id")
                    if not uid:
                        audios = obj.get("audios", [])
                        if audios:
                            uid = extract_utter_id_from_audio_path(audios[0])
                    if uid and uid in index:
                        row = index[uid]
                        raw_src = row.get("src_text", "") or ""
                        yield normalize_text_casing(raw_src)
                    else:
                        yield ""
                    count += 1

        # 替换 nlp.pipe 调用，直接从 ner_map 获取
        def candidate_generator():
            count = 0
            with open(args.input_gt, "r", encoding="utf-8") as f_pass0:
                for line in f_pass0:
                    if args.max_messages is not None and count >= args.max_messages: break
                    obj = json.loads(line)
                    uid = obj.get("utter_id") or extract_utter_id_from_audio_path(obj.get("audios", [""])[0])
                    yield ner_map.get(uid, [])
                    count += 1

        for candidates in tqdm(candidate_generator(), desc="Pass 0 Freq"):
            for c in candidates:
                term_freq[c.lower()] += 1
        
        logger.info(f"Pass 0 done. Unique terms: {len(term_freq)}")

    aligner = None
    window_objs = []
    items_to_align = []
    
    processed_count = 0
    total_ner_entities = 0
    total_aligned_entities = 0
    total_final_rejected_entities = 0
    rejected_counter = defaultdict(int) # term -> count
    rejected_examples = []

    def flush_window(f_out, win, to_align):
        nonlocal aligner, total_ner_entities, total_aligned_entities, total_final_rejected_entities
        
        # 1. Prepare results map
        results_map = defaultdict(dict) # win_idx -> {term -> zh}
        
        # Keep track of what we planned to align
        planned_idx_map = {win_idx: (m, r) for win_idx, m, r in to_align}

        if to_align:
            if aligner is None:
                logger.info("Initializing VLLMAligner...")
                aligner = VLLMAligner(args.align_model, args.gpu_memory_util, tensor_parallel_size=args.tensor_parallel_size)
            
            # Pass 1: Initial alignment
            batch_input = []
            for _, missing, row in to_align:
                raw_src = row.get("src_text", "") or ""
                norm_src = normalize_text_casing(raw_src)
                batch_input.append((norm_src, row["tgt_text"], missing))
            
            logger.info(f"Calling LLM for {len(to_align)} items...")
            aligned_results = aligner.align_batch(batch_input, tgt_name=tgt_name, tgt_tag=tgt_tag)
            
            retry_items = [] # list of (to_align_idx, missing_terms, row)
            for i, mapping in enumerate(aligned_results):
                win_idx, missing, row = to_align[i]
                results_map[win_idx].update(mapping)
                
                still_missing = [m for m in missing if m not in mapping]
                if still_missing:
                    retry_items.append((i, still_missing, row))

            # Optional Retries
            for attempt in range(args.max_retries):
                if not retry_items:
                    break
                logger.info(f"Retry Attempt {attempt+1}/{args.max_retries} for {len(retry_items)} items...")
                
                retry_batch_input = []
                for _, still_missing, row in retry_items:
                    raw_src = row.get("src_text", "") or ""
                    norm_src = normalize_text_casing(raw_src)
                    retry_batch_input.append((norm_src, row["tgt_text"], still_missing))
                
                retry_results = aligner.align_batch(retry_batch_input, tgt_name=tgt_name, tgt_tag=tgt_tag, is_retry=True)
                
                new_retry_items = []
                for j, mapping in enumerate(retry_results):
                    original_to_align_idx, still_missing, row = retry_items[j]
                    win_idx, _, _ = to_align[original_to_align_idx]
                    results_map[win_idx].update(mapping)
                    
                    even_more_missing = [m for m in still_missing if m not in mapping]
                    if even_more_missing:
                        new_retry_items.append((original_to_align_idx, even_more_missing, row))
                retry_items = new_retry_items

        # 2. Update all window objects and statistics
        for win_idx, obj in enumerate(win):
            # Calculate NER entities for this object (even if not in to_align)
            # We already calculated 'missing' in the main loop, but we need it here for stats.
            # If win_idx was in to_align, we use that.
            
            num_chunks = len(obj.get("audios", []))
            # Support incremental: keep existing if any
            gt_by_chunk = obj.get("gt_terms_by_chunk", []) or [[] for _ in range(max(1, num_chunks))]
            if len(gt_by_chunk) < max(1, num_chunks):
                gt_by_chunk += [[] for _ in range(max(1, num_chunks) - len(gt_by_chunk))]
            
            existing_count = sum(len(c) for c in gt_by_chunk)
            
            if win_idx in planned_idx_map:
                missing, row = planned_idx_map[win_idx]
                mapping = results_map[win_idx]
                
                total_ner_entities += (len(missing) + existing_count)
                total_aligned_entities += (len(mapping) + existing_count)
                
                final_missing = [m for m in missing if m not in mapping]
                total_final_rejected_entities += len(final_missing)
                for m in final_missing:
                    rejected_counter[m] += 1
                    rejected_examples.append(f"{m} (in: {row['tgt_text']})")

                traj = row.get("src_trajectory", []) or []
                mm = obj.get("merge_multiplier")
                src_chunks = [" ".join(x) for x in split_trajectory_by_chunks(traj, num_chunks, merge_multiplier=mm)] if num_chunks else [row.get("src_text", "")]
                
                for term, zh in mapping.items():
                    cis = locate_term_chunks_robust(src_chunks, term)
                    for ci in cis:
                        # Deduplicate
                        existing_terms_in_chunk = {t.get("term", "").lower() for t in gt_by_chunk[ci]}
                        if term.lower() not in existing_terms_in_chunk:
                            out_item = {"term": term}
                            # Normalize Chinese output if requested/available.
                            if zh_converter is not None:
                                try:
                                    out_item[tgt_code] = zh_converter.convert(str(zh).strip(), "zh-cn")
                                except Exception:
                                    out_item[tgt_code] = str(zh).strip()
                            else:
                                out_item[tgt_code] = str(zh).strip()
                            gt_by_chunk[ci].append(out_item)
                
                obj["gt_terms_by_chunk"] = gt_by_chunk
            else:
                # No new entities to align for this one
                total_ner_entities += existing_count
                total_aligned_entities += existing_count

            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    logger.info("Starting streaming processing...")
    output_file = f"{args.output_gt.replace('.jsonl', '')}_gpu{args.gpu_id}.jsonl" if args.total_gpus > 1 else args.output_gt
    
    with open(args.input_gt, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        
        pbar = tqdm(desc=f"GPU {args.gpu_id} Processing")
        for idx, line in enumerate(f_in):
            if args.total_gpus > 1 and idx % args.total_gpus != args.gpu_id:
                continue
            if args.max_messages is not None and processed_count >= args.max_messages:
                break
            
            obj = json.loads(line)
            uid = obj.get("utter_id") or extract_utter_id_from_audio_path(obj.get("audios", [""])[0])
            if not obj.get("utter_id"): obj["utter_id"] = uid

            missing = []
            row = None
            if uid and uid in index:
                row = index[uid]
                candidates = ner_map.get(uid, [])
                
                # Apply sampling: Frequency-based or simple random
                ner_entities = []
                if args.enable_freq_sampling:
                    for cand in candidates:
                        f = term_freq[cand.lower()]
                        if random.random() < keep_prob(f):
                            ner_entities.append(cand)
                else:
                    for cand in candidates:
                        if random.random() <= args.sampling_rate:
                            ner_entities.append(cand)
                
                # Budget Control: Max K terms per utterance
                if len(ner_entities) > args.max_terms_per_utter:
                    # Prioritize low-frequency terms (usually more valuable entities)
                    ner_entities.sort(key=lambda x: term_freq[x.lower()])
                    ner_entities = ner_entities[:args.max_terms_per_utter]
                
                # Check existing terms to avoid duplicates and only align what's new/missing
                existing_terms = set()
                for chunk in obj.get("gt_terms_by_chunk", []) or []:
                    for t_item in chunk:
                        existing_terms.add(t_item.get("term", "").lower())

                seen = set()
                for ent in ner_entities:
                    el = ent.lower()
                    if el in seen or el in existing_terms:
                        continue
                    if len([w for w in re.split(r"\s+", ent) if w]) > 3:
                        continue
                    missing.append(ent)
                    seen.add(el)

            window_objs.append(obj)
            if missing:
                items_to_align.append((len(window_objs) - 1, missing, row))
            
            # Flush if we have enough items to align OR if the window is getting too large
            if len(items_to_align) >= args.batch_size or (len(window_objs) >= args.batch_size * 10 and window_objs):
                flush_window(f_out, window_objs, items_to_align)
                window_objs = []
                items_to_align = []
            
            processed_count += 1
            pbar.update(1)

        if window_objs:
            flush_window(f_out, window_objs, items_to_align)
        pbar.close()

    if rejected_examples:
        logger.info(f"=== DIAGNOSIS: Total {total_final_rejected_entities} Rejected Terms ===")
        logger.info(f"NER Total (incl. existing): {total_ner_entities}, Aligned: {total_aligned_entities}, Final Rejected: {total_final_rejected_entities}")
        
        # Frequency analysis
        top_rejected = sorted(rejected_counter.items(), key=lambda x: x[1], reverse=True)[:20]
        logger.info("--- Top 20 Most Frequent Rejected Terms ---")
        for term, count in top_rejected:
            logger.info(f"  {term}: {count} times")
            
        logger.info("--- Examples of Rejections ---")
        for msg in rejected_examples[:20]:
            logger.info(f"Rejected Example: {msg}")
    logger.info("Done. Processed %d messages.", processed_count)


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    random.seed(42) # For reproducibility
    main()

