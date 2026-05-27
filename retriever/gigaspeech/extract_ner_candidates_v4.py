#!/usr/bin/env python3
import os
import json
import argparse
import logging
import re
from pathlib import Path
from tqdm import tqdm
import spacy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

STOPWORDS = {"the", "a", "an", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their", "it", "they", "them", "who", "whom", "whose"}

def normalize_text_casing(text: str) -> str:
    if not text: return ""
    upper_count = sum(1 for c in text if c.isupper())
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count > 0 and (upper_count / alpha_count) > 0.7:
        return text.title()
    return text

def get_filtered_candidates(doc):
    candidates = []
    # 1. NER
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "FAC", "PRODUCT", "EVENT"}:
            candidates.append(ent.text.strip())
    # 2. Noun chunks
    for chunk in doc.noun_chunks:
        text = chunk.text.strip()
        toks = text.split()
        if len(toks) > 1 and toks[0].lower() in {"the", "a", "an"}:
            text = " ".join(toks[1:])
        candidates.append(text)
    # 3. Individual Nouns
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop:
            candidates.append(token.text.strip())
    
    raw_unique = set(candidates)
    filtered_basic = []
    for cand in raw_unique:
        c = cand.strip()
        if not c or len(c) < 3: continue
        if c.lower() in STOPWORDS: continue
        if len(c.split()) > 4: continue
        filtered_basic.append(c)
        
    filtered_basic.sort(key=len, reverse=True)
    final_filtered = []
    for i, cand in enumerate(filtered_basic):
        is_subset = False
        for j, other in enumerate(filtered_basic):
            if i == j: continue
            c_low = cand.lower()
            o_low = other.lower()
            if c_low in o_low:
                if re.search(r'\b' + re.escape(c_low) + r'\b', o_low):
                    is_subset = True
                    break
        if not is_subset:
            final_filtered.append(cand)
    return final_filtered

def load_tsv_index(tsv_path: str):
    index = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        col_id = header.index("id")
        col_src = header.index("src_text")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > col_src:
                index[parts[col_id]] = parts[col_src]
    return index

def extract_utter_id_from_audio_path(audio_path: str):
    parts = Path(audio_path).parts
    if len(parts) >= 3: return f"{parts[-3]}_{parts[-2]}"
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-gt", required=True)
    parser.add_argument("--input-tsv", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--spacy-model", default="en_core_web_trf")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--total-gpus", type=int, default=1)
    args = parser.parse_args()

    # 显式启动 GPU
    try:
        # 在 Slurm 环境下，CUDA_VISIBLE_DEVICES 已经指定了卡，这里不需要传 index
        # 即使传 0，也是指 CUDA_VISIBLE_DEVICES 里的第一个可见卡
        spacy.require_gpu()
        logger.info(f"GPU {args.gpu_id} (logical) initialized for spaCy.")
    except Exception as e:
        logger.error(f"spaCy GPU initialization failed: {e}")
        logger.info("Falling back to CPU (this will be VERY slow for trf models).")

    nlp = spacy.load(args.spacy_model)
    index = load_tsv_index(args.input_tsv)

    def src_generator():
        with open(args.input_gt, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx % args.total_gpus != args.gpu_id: continue
                try:
                    obj = json.loads(line)
                    uid = obj.get("utter_id") or extract_utter_id_from_audio_path(obj.get("audios", [""])[0])
                    src_text = index.get(uid, "")
                    if src_text:
                        yield uid, normalize_text_casing(src_text)
                except Exception as e:
                    logger.warning(f"Error parsing line {idx}: {e}")

    # 处理并写入
    output_shard = f"{args.output_jsonl.replace('.jsonl', '')}_gpu{args.gpu_id}.jsonl"
    
    # 转换为列表以避免重复读取生成器
    items = list(src_generator())
    if not items:
        logger.info(f"No items to process for GPU {args.gpu_id}")
        return

    uids, texts = zip(*items)
    
    with open(output_shard, "w", encoding="utf-8") as f_out:
        # 使用 nlp.pipe 批量处理提升吞吐
        for uid, doc in tqdm(zip(uids, nlp.pipe(texts, batch_size=128)), total=len(uids), desc=f"GPU {args.gpu_id}"):
            candidates = get_filtered_candidates(doc)
            f_out.write(json.dumps({"utter_id": uid, "ner_candidates": candidates}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()

