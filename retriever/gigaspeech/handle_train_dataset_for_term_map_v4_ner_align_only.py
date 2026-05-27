#!/usr/bin/env python3
"""
Stage 1.4 (Pure NER Alignment)
1) 直接从 baseline JSONL 开始处理。
2) 仅识别命名实体 (PERSON/LOC/GPE/NORP/FAC/ORG)。
3) 支持多 GPU 分片并行和重试机制。
"""

import os
import sys
import json
import re
import random
import argparse
import logging
import multiprocessing as mp
import zhconv
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

from tqdm import tqdm

# vLLM setup
os.environ.setdefault("VLLM_USE_V1", "0")

logger = logging.getLogger(__name__)

# ----------------------------
# Configuration & Prompts
# ----------------------------
DEFAULT_INPUT_GT = "/mnt/gemini/data1/jiaxuanluo/train_s_zh_baseline.jsonl"
DEFAULT_INPUT_TSV = "/mnt/gemini/data1/jiaxuanluo/train_xl_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv"
DEFAULT_OUTPUT_GT = "/mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_only_aligned.jsonl"

DEFAULT_ALIGN_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
DEFAULT_ALIGN_GPU_MEMORY_UTIL = 0.9

ENTITY_ALIGN_PROMPT = """You are a translation alignment assistant.

Task:
Find the corresponding Chinese translation, transliteration, or equivalent term in the [ZH] text for the given [ENG] Named Entities.

Rules:
1. **Semantic Match**: Look for the meaning. (e.g., "Southerner" -> "南方人", "Covid" -> "新冠").
2. **Acronyms**: Look for the expanded or translated acronym (e.g., "CDC" -> "疾控中心").
3. **Transliteration**: Look for phonetic matches (e.g., "Carlo" -> "卡洛").
4. **Copying is Valid**: If the name is kept in English in [ZH] (e.g., "Dice" -> "Dice"), output the English word as the zh value.
5. **Format**: Return STRICT JSON.

Input:
[ENG]: {src_text}
[ZH]: {tgt_text}

Entities to Align:
{terms_json}

Return JSON object:
{{
  "alignments": [
    {{"term": "<eng_term>", "zh": "<zh_span_or_null>"}}
  ]
}}
"""

RETRY_ENTITY_ALIGN_PROMPT = """You are a translation alignment assistant.
The following entities were NOT found in the previous pass. Please try HARDER to find them in the [ZH] text.
Be flexible with phonetic matches or shorter names.

Input:
[ENG]: {src_text}
[ZH]: {tgt_text}

Entities to Retry:
{terms_json}

Return JSON object:
{{
  "alignments": [
    {{"term": "<eng_term>", "zh": "<zh_span_or_null>"}}
  ]
}}
"""

# ----------------------------
# Helpers
# ----------------------------
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
    return index

def extract_utter_id_from_audio_path(audio_path: str) -> Optional[str]:
    try:
        parts = Path(audio_path).parts
        if len(parts) >= 3:
            return f"{parts[-3]}_{parts[-2]}"
    except Exception: return None
    return None

def _load_spacy():
    import spacy
    try: return spacy.load("en_core_web_trf")
    except Exception: return spacy.load("en_core_web_lg")

def normalize_text_casing(text: str) -> str:
    if not text: return ""
    upper_count = sum(1 for c in text if c.isupper())
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count > 0 and (upper_count / alpha_count) > 0.7:
        return text.title()
    return text

def split_trajectory_by_chunks(trajectory: List[str], num_chunks: int, merge_multiplier: Optional[int] = None) -> List[List[str]]:
    if num_chunks <= 0: return []
    if merge_multiplier is not None:
        chunks = []
        for i in range(num_chunks):
            start = i * merge_multiplier
            end = min((i + 1) * merge_multiplier, len(trajectory))
            chunks.append(trajectory[start:end])
        return chunks
    chunk_size = (len(trajectory) + num_chunks - 1) // (num_chunks or 1)
    return [trajectory[i * chunk_size : min((i + 1) * chunk_size, len(trajectory))] for i in range(num_chunks)]

def locate_term_chunk_robust(src_chunks: List[str], term: str) -> int:
    tl = term.strip().lower()
    for i, ch in enumerate(src_chunks):
        if tl in (ch or "").lower(): return i
    for i in range(len(src_chunks) - 1):
        combined = ((src_chunks[i] or "") + " " + (src_chunks[i + 1] or ""))
        if tl in combined.lower(): return i + 1
    return 0

import ast
def safe_parse_json(text: str) -> Dict:
    txt = (text or "").strip()
    if not txt: return {}
    match = re.search(r"(\{[\s\S]*\})", txt)
    if match:
        cand = match.group(1)
        try: return json.loads(cand)
        except:
            try: return ast.literal_eval(cand)
            except: pass
    return {}

class VLLMAligner:
    def __init__(self, model: str, gpu_util: float, tensor_parallel_size: int = 1):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=model, tensor_parallel_size=tensor_parallel_size, max_model_len=4096, gpu_memory_utilization=gpu_util, trust_remote_code=True, enforce_eager=True)
        self.SamplingParams = SamplingParams

    def align_batch(self, items: List[Tuple[str, str, List[str]]], max_tokens: int = 1024, is_retry: bool = False) -> List[Dict[str, str]]:
        prompts = []
        prompt_tmpl = RETRY_ENTITY_ALIGN_PROMPT if is_retry else ENTITY_ALIGN_PROMPT
        for src, tgt, names in items:
            user_content = prompt_tmpl.format(src_text=src, tgt_text=tgt, terms_json=json.dumps(names, ensure_ascii=False))
            prompts.append(f"<|im_start|>system\nYou are a translation alignment assistant.<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n")
        sp = self.SamplingParams(temperature=0.0, max_tokens=max_tokens, stop=["<|im_end|>", "```\n"])
        outputs = self.llm.generate(prompts, sp)
        results = []
        for out in outputs:
            obj = safe_parse_json(out.outputs[0].text if out.outputs else "")
            mapping = {}
            for a in obj.get("alignments", []):
                term, zh = a.get("term"), a.get("zh")
                if term and zh:
                    mapping[str(term).strip()] = zhconv.convert(str(zh).strip(), 'zh-cn')
            results.append(mapping)
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
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--total-gpus", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    index = load_tsv_index(args.input_tsv)
    nlp = _load_spacy()
    TARGET_LABELS = {"PERSON", "GPE", "LOC", "NORP", "FAC", "ORG"}

    aligner = None
    window_objs = []
    items_to_align = []
    processed_count = 0
    total_ner = 0
    total_aligned = 0

    def flush_window(f_out, win, to_align):
        nonlocal aligner, total_ner, total_aligned
        results_map = defaultdict(dict)
        planned_idx_map = {win_idx: (m, r) for win_idx, m, r in to_align}

        if to_align:
            if aligner is None: aligner = VLLMAligner(args.align_model, args.gpu_memory_util, tensor_parallel_size=args.tensor_parallel_size)
            batch_input = [(normalize_text_casing(r["src_text"]), r["tgt_text"], m) for _, m, r in to_align]
            logger.info(f"Calling LLM for {len(to_align)} items...")
            aligned_results = aligner.align_batch(batch_input)
            
            retry_items = []
            for i, mapping in enumerate(aligned_results):
                win_idx, missing, row = to_align[i]
                results_map[win_idx].update(mapping)
                still_missing = [m for m in missing if m not in mapping]
                if still_missing: retry_items.append((i, still_missing, row))

            for attempt in range(args.max_retries):
                if not retry_items: break
                logger.info(f"Retry Attempt {attempt+1} for {len(retry_items)} items...")
                retry_results = aligner.align_batch([(normalize_text_casing(r["src_text"]), r["tgt_text"], sm) for _, sm, r in retry_items], is_retry=True)
                new_retry = []
                for j, mapping in enumerate(retry_results):
                    orig_to_align_idx, sm, row = retry_items[j]
                    results_map[to_align[orig_to_align_idx][0]].update(mapping)
                    even_more = [m for m in sm if m not in mapping]
                    if even_more: new_retry.append((orig_to_align_idx, even_more, row))
                retry_items = new_retry

        for win_idx, obj in enumerate(win):
            num_chunks = len(obj.get("audios", []))
            gt_by_chunk = obj.get("gt_terms_by_chunk", []) or [[] for _ in range(max(1, num_chunks))]
            if len(gt_by_chunk) < num_chunks: gt_by_chunk += [[] for _ in range(num_chunks - len(gt_by_chunk))]
            
            if win_idx in planned_idx_map:
                missing, row = planned_idx_map[win_idx]
                mapping = results_map[win_idx]
                total_ner += len(missing)
                total_aligned += len(mapping)
                mm = obj.get("merge_multiplier")
                src_chunks = [" ".join(x) for x in split_trajectory_by_chunks(row.get("src_trajectory", []), num_chunks, merge_multiplier=mm)] if num_chunks else [row.get("src_text", "")]
                for term, zh in mapping.items():
                    ci = locate_term_chunk_robust(src_chunks, term)
                    exist = {t.get("term", "").lower() for t in gt_by_chunk[ci]}
                    if term.lower() not in exist: gt_by_chunk[ci].append({"term": term, "zh": zh})
                obj["gt_terms_by_chunk"] = gt_by_chunk
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    logger.info("Starting processing...")
    output_file = f"{args.output_gt.replace('.jsonl', '')}_gpu{args.gpu_id}.jsonl" if args.total_gpus > 1 else args.output_gt
    with open(args.input_gt, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        pbar = tqdm(desc=f"GPU {args.gpu_id}")
        for idx, line in enumerate(f_in):
            if args.total_gpus > 1 and idx % args.total_gpus != args.gpu_id: continue
            obj = json.loads(line)
            uid = obj.get("utter_id") or (extract_utter_id_from_audio_path(obj.get("audios", [""])[0]) if obj.get("audios") else None)
            missing = []
            row = None
            if uid and uid in index:
                row = index[uid]
                doc = nlp(normalize_text_casing(row.get("src_text", "")))
                ner_ents = {ent.text.strip() for ent in doc.ents if ent.label_ in TARGET_LABELS and ent.text.strip()}
                exist = {t.get("term", "").lower() for chunk in (obj.get("gt_terms_by_chunk", []) or []) for t in chunk}
                for ent in ner_ents:
                    if ent.lower() not in exist: missing.append(ent)
            
            window_objs.append(obj)
            if missing: items_to_align.append((len(window_objs) - 1, missing, row))
            if len(items_to_align) >= args.batch_size or len(window_objs) >= 500:
                flush_window(f_out, window_objs, items_to_align)
                window_objs, items_to_align = [], []
            processed_count += 1
            pbar.update(1)
        if window_objs: flush_window(f_out, window_objs, items_to_align)
    logger.info(f"Done. NER Total: {total_ner}, Aligned: {total_aligned}")

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()

