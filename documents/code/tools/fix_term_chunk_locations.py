#!/usr/bin/env python3
import json
import os
import sys
import re
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple

def load_tsv_index(tsv_path: str) -> Dict[str, Dict]:
    index = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        col_id = header.index("id") if "id" in header else 0
        col_traj = header.index("src_trajectory") if "src_trajectory" in header else None
        col_src = header.index("src_text") if "src_text" in header else None
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= col_id: continue
            uid = parts[col_id]
            traj = []
            if col_traj is not None and len(parts) > col_traj:
                try:
                    traj = eval(parts[col_traj])
                except:
                    traj = []
            src_text = parts[col_src] if col_src is not None and len(parts) > col_src else ""
            index[uid] = {"src_trajectory": traj, "src_text": src_text}
    return index

def split_trajectory_by_chunks(trajectory: List[str], num_chunks: int, merge_multiplier: Optional[int] = None) -> List[List[str]]:
    if num_chunks <= 0: return []
    if not trajectory: return [[] for _ in range(num_chunks)]
    if merge_multiplier is not None:
        chunks = []
        for i in range(num_chunks):
            start = i * merge_multiplier
            end = min((i + 1) * merge_multiplier, len(trajectory))
            chunks.append(trajectory[start:end])
        return chunks
    chunk_size = (len(trajectory) + num_chunks - 1) // num_chunks
    return [trajectory[i * chunk_size : min((i + 1) * chunk_size, len(trajectory))] for i in range(num_chunks)]

def locate_term_chunk_robust(src_chunks: List[str], term: str) -> int:
    tl = term.strip().lower()
    for i, ch in enumerate(src_chunks):
        if tl in (ch or "").lower():
            return i
    for i in range(len(src_chunks) - 1):
        combined = re.sub(r"\s+", " ", ((src_chunks[i] or "") + " " + (src_chunks[i + 1] or "")))
        if tl in combined.lower():
            return i + 1
    return 0

def generate_term_map_string(terms: List[Tuple[str, str]]) -> str:
    if not terms: return ""
    lines = ["term_map:"]
    for s, t in terms:
        lines.append(f"{s}={t}")
    return "\n".join(lines)

def parse_term_map(text: str) -> List[Tuple[str, str]]:
    if "term_map:" not in text: return []
    lines = text.split("\n")
    terms = []
    start = False
    for l in lines:
        if l.strip() == "term_map:":
            start = True
            continue
        if start:
            if "=" in l:
                parts = l.split("=", 1)
                terms.append((parts[0].strip(), parts[1].strip()))
            else:
                # If we encounter a line without '=', the term_map probably ended
                # But sometimes there are empty lines. Let's just continue if empty.
                if l.strip():
                    break
    return terms

def extract_utter_id_from_audio_path(audio_path: str) -> Optional[str]:
    try:
        parts = audio_path.split("/")
        if len(parts) >= 3:
            # e.g., .../AUD0000000918/371/0.wav -> AUD0000000918_371
            return f"{parts[-3]}_{parts[-2]}"
    except Exception:
        pass
    return None

def fix_jsonl(input_path: str, output_path: str, tsv_index: Dict[str, Dict]):
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        for line in tqdm(f_in, desc=f"Fixing {os.path.basename(input_path)}"):
            obj = json.loads(line)
            uid = obj.get("utter_id")
            if not uid:
                audios = obj.get("audios", [])
                if audios:
                    uid = extract_utter_id_from_audio_path(audios[0])
            
            if not uid or uid not in tsv_index:
                f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue
            
            row = tsv_index[uid]
            traj = row.get("src_trajectory", [])
            num_chunks = len(obj.get("audios", []))
            mm = obj.get("merge_multiplier")
            
            # Re-calculate correct src_chunks
            src_chunks = [" ".join(x) for x in split_trajectory_by_chunks(traj, num_chunks, merge_multiplier=mm)] if num_chunks else [row.get("src_text", "")]
            
            # 1. Correct gt_terms_by_chunk
            old_gt_by_chunk = obj.get("gt_terms_by_chunk", [])
            all_gt_items = []
            for chunk in old_gt_by_chunk:
                for item in chunk:
                    all_gt_items.append(item)
            
            new_gt_by_chunk = [[] for _ in range(max(1, num_chunks))]
            seen_in_chunk = [set() for _ in range(max(1, num_chunks))]
            
            # Map of term_lc -> new_chunk_idx
            term_to_new_chunk = {}
            
            for item in all_gt_items:
                term = item.get("term", "")
                if not term: continue
                ci = locate_term_chunk_robust(src_chunks, term)
                if term.lower() not in seen_in_chunk[ci]:
                    new_gt_by_chunk[ci].append(item)
                    seen_in_chunk[ci].add(term.lower())
                term_to_new_chunk[term.lower()] = ci
            
            obj["gt_terms_by_chunk"] = new_gt_by_chunk

            # 2. Correct term_map in messages (if present)
            messages = obj.get("messages", [])
            if any("term_map:" in m.get("content", "") for m in messages):
                # This is a Stage 2 file. We need to move terms between messages.
                audio_turn_idx = 0
                all_term_map_data = [] # List of (chunk_idx, list of (term, zh))
                
                # First pass: extract all term_maps
                for m in messages:
                    if m.get("role") == "user" and "<audio>" in m.get("content", ""):
                        terms = parse_term_map(m.get("content", ""))
                        all_term_map_data.append(terms)
                        audio_turn_idx += 1
                
                if len(all_term_map_data) == num_chunks:
                    # We have term_map for each audio chunk. Let's redistribute.
                    new_term_map_data = [[] for _ in range(num_chunks)]
                    gt_terms_lc = {item.get("term", "").lower() for item in all_gt_items}
                    
                    for chunk_idx, terms in enumerate(all_term_map_data):
                        for term, zh in terms:
                            t_lc = term.lower()
                            if t_lc in gt_terms_lc:
                                # This is a GT term. Move it to its new correct chunk.
                                new_ci = term_to_new_chunk.get(t_lc, chunk_idx) # fallback to current if not found
                                new_term_map_data[new_ci].append((term, zh))
                            else:
                                # This is a negative/distractor. Keep it in the same chunk.
                                # Alternatively, we could move distractors too, but they aren't strictly tied to a GT location.
                                # Keeping them where they are is safer for "preserving" the sampled pool.
                                new_term_map_data[chunk_idx].append((term, zh))
                    
                    # Second pass: inject back into messages
                    audio_turn_idx = 0
                    for m in messages:
                        if m.get("role") == "user" and "<audio>" in m.get("content", ""):
                            if audio_turn_idx < num_chunks:
                                new_terms = new_term_map_data[audio_turn_idx]
                                if new_terms:
                                    term_map_str = generate_term_map_string(new_terms)
                                    m["content"] = f"<audio>\n\n{term_map_str}"
                                else:
                                    m["content"] = "<audio>"
                            audio_turn_idx += 1
            
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

import difflib

def find_best_gt_match(distractor: str, gt_terms: List[str]) -> Optional[str]:
    if not gt_terms: return None
    matches = difflib.get_close_matches(distractor, gt_terms, n=1, cutoff=0.0)
    return matches[0] if matches else None

def fix_stage2_jsonl(stage2_path: str, stage1_fixed_path: str, output_path: str):
    # Load fixed stage1 data to get the correct GT locations
    fixed_data = {}
    with open(stage1_fixed_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            uid = obj.get("utter_id")
            if not uid:
                audios = obj.get("audios", [])
                if audios: uid = extract_utter_id_from_audio_path(audios[0])
            if uid:
                # Store new_gt_chunks and a map of term -> new_chunk_idx
                new_gt_chunks = obj.get("gt_terms_by_chunk", [])
                term_to_new_chunk = {}
                for ci, chunk in enumerate(new_gt_chunks):
                    for item in chunk:
                        term_to_new_chunk[item["term"].lower()] = ci
                fixed_data[uid] = {
                    "gt_chunks": new_gt_chunks,
                    "term_map": term_to_new_chunk
                }

    with open(stage2_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        for line in tqdm(f_in, desc=f"Fixing Stage 2 {os.path.basename(stage2_path)}"):
            obj = json.loads(line)
            audios = obj.get("audios", [])
            if not audios:
                f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue
            
            uid = extract_utter_id_from_audio_path(audios[0])
            if not uid or uid not in fixed_data:
                f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue
            
            f_meta = fixed_data[uid]
            new_gt_chunks = f_meta["gt_chunks"]
            term_to_new_chunk = f_meta["term_map"]
            num_chunks = len(audios)
            
            messages = obj.get("messages", [])
            
            # 1. First pass: extract original term_maps and identify which GT terms were in which OLD chunk
            old_chunk_data = [] # List of (gt_terms_in_this_chunk, distractors_in_this_chunk)
            audio_turn_idx = 0
            all_gt_terms_lc = set(term_to_new_chunk.keys())
            
            for m in messages:
                if m.get("role") == "user" and "<audio>" in m.get("content", ""):
                    if audio_turn_idx < num_chunks:
                        entries = parse_term_map(m.get("content", ""))
                        chunk_gt = []
                        chunk_neg = []
                        for t, zh in entries:
                            if t.lower() in all_gt_terms_lc:
                                chunk_gt.append((t, zh))
                            else:
                                chunk_neg.append((t, zh))
                        old_chunk_data.append((chunk_gt, chunk_neg))
                    audio_turn_idx += 1
            
            if len(old_chunk_data) != num_chunks:
                f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

            # 2. Redistribute
            new_term_map_content = [[] for _ in range(num_chunks)]
            
            # For each old chunk, we look at its negatives and assign them to the new chunk 
            # where their corresponding GT terms moved.
            for old_ci, (old_gts, old_negs) in enumerate(old_chunk_data):
                # GT terms move to their absolute correct chunks
                for t, zh in old_gts:
                    new_ci = term_to_new_chunk.get(t.lower(), old_ci)
                    new_term_map_content[new_ci].append((t, zh))
                
                # Negatives follow the GT terms they were originally with
                if old_negs:
                    if old_gts:
                        # Find which GT term each negative is most similar to
                        gt_names = [gt[0] for gt in old_gts]
                        for nt, nzh in old_negs:
                            best_gt = find_best_gt_match(nt, gt_names)
                            if best_gt:
                                new_ci = term_to_new_chunk.get(best_gt.lower(), old_ci)
                                new_term_map_content[new_ci].append((nt, nzh))
                            else:
                                # Fallback: if no GT match, keep in original chunk (unlikely to happen with similarity)
                                new_term_map_content[old_ci].append((nt, nzh))
                    else:
                        # If the old chunk had no GT terms but had negatives (e.g. All-Negative case)
                        # We just keep them in the same chunk index.
                        for nt, nzh in old_negs:
                            new_term_map_content[old_ci].append((nt, nzh))
            
            # 3. Inject back into messages
            audio_turn_idx = 0
            for m in messages:
                if m.get("role") == "user" and "<audio>" in m.get("content", ""):
                    if audio_turn_idx < num_chunks:
                        chunk_terms = new_term_map_content[audio_turn_idx]
                        if chunk_terms:
                            # Dedup and shuffle
                            seen = set()
                            final_chunk_terms = []
                            for t, zh in chunk_terms:
                                if t.lower() not in seen:
                                    final_chunk_terms.append((t, zh))
                                    seen.add(t.lower())
                            random.shuffle(final_chunk_terms)
                            m["content"] = f"<audio>\n\n{generate_term_map_string(final_chunk_terms)}"
                        else:
                            m["content"] = "<audio>"
                    audio_turn_idx += 1
            
            # Update gt_terms_by_chunk as well
            obj["gt_terms_by_chunk"] = new_gt_chunks
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    import random
    random.seed(42)
    tsv_path = "/mnt/gemini/data1/jiaxuanluo/train_xl_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv"
    tsv_index = load_tsv_index(tsv_path)
    
    # Stage 1 files
    stage1_files = [
        "/mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_50percent.jsonl",
        "/mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_freq_k20.jsonl"
    ]
    
    for f in stage1_files:
        if os.path.exists(f):
            out = f.replace(".jsonl", "_fixed.jsonl")
            fix_jsonl(f, out, tsv_index)
            print(f"Fixed Stage 1 {f} -> {out}")
        else:
            print(f"File not found: {f}")

    # Stage 2 file (the final training data)
    final_file = "/mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_freq_k20_final.jsonl"
    stage1_fixed = "/mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_freq_k20_fixed.jsonl"
    
    if os.path.exists(final_file) and os.path.exists(stage1_fixed):
        out = final_file.replace(".jsonl", "_fixed.jsonl")
        fix_stage2_jsonl(final_file, stage1_fixed, out)
        print(f"Fixed Stage 2 {final_file} -> {out}")

