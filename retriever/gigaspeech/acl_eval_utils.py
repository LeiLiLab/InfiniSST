import os
import re
import glob
import json
import logging
import math
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import faiss
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple, Set

try:
    from flashtext import KeywordProcessor
except ImportError:
    KeywordProcessor = None

logger = logging.getLogger(__name__)

def l2_distance_to_score(similarity: float) -> float:
    """
    Convert Inner Product (Cosine Similarity) to a score based on 1 / (1 + L2_dist).
    For L2 normalized vectors: L2_dist^2 = 2 - 2 * Cosine_Similarity.
    """
    l2_dist = math.sqrt(max(0, 2 - 2 * float(similarity)))
    return 1.0 / (1.0 + l2_dist)

def _canonicalize_plural_english(term: str) -> str:
    """
    A lightweight English plural -> singular canonicalizer for evaluation.
    This is intentionally conservative (heuristic-based), and is only meant to merge
    common pairs like "text/texts", "language/languages".
    """
    t = (term or "").strip().lower()
    if len(t) <= 3:
        return t
    # Do not singularize words like "ss"
    if t.endswith("ss"):
        return t
    # companies -> company
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    # classes -> class, boxes -> box, watches -> watch, etc.
    if t.endswith("es") and len(t) > 4:
        if t.endswith(("ches", "shes")):
            return t[:-2]
        if t[-3] in ("s", "x", "z"):
            return t[:-2]
    # default plural s -> singular
    if t.endswith("s"):
        return t[:-1]
    return t

def _dedup_overlapping_occurrences(occ: List[Tuple[str, int, int]]) -> List[Tuple[str, int, int]]:
    """
    Deduplicate overlapping occurrences by keeping the longer span when overlaps exist.
    Also removes exact-duplicate spans.
    """
    if not occ:
        return []
    # Sort by start asc, length desc
    occ_sorted = sorted(occ, key=lambda x: (x[1], -(x[2] - x[1]), x[0]))
    kept: List[Tuple[str, int, int]] = []
    last_end = -1
    seen_spans = set()
    for term, s, e in occ_sorted:
        span = (s, e)
        if span in seen_spans:
            continue
        # If overlaps with previous kept span, skip (previous is longer due to sort)
        if kept and s < kept[-1][2]:
            continue
        kept.append((term, s, e))
        seen_spans.add(span)
        last_end = e
    # Restore chronological order
    kept.sort(key=lambda x: x[1])
    return kept

def build_dynamic_index(text_encoder, tokenizer, glossary_entries, device, batch_size=512):
    """Build a FAISS index on the fly using pre-loaded glossary entries."""
    all_keys = [e["key"] for e in glossary_entries]
    all_embeddings = []
    text_encoder.eval()
    with torch.no_grad():
        for i in range(0, len(all_keys), batch_size):
            batch_keys = all_keys[i : i + batch_size]
            inputs = tokenizer(batch_keys, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                embeddings = text_encoder(inputs.input_ids, inputs.attention_mask)
            all_embeddings.append(embeddings.cpu().float().numpy())
            
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    faiss.normalize_L2(all_embeddings)
    dim = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(all_embeddings)
    return index

def build_keyword_processor(glossary_terms: List[str]):
    if KeywordProcessor is None:
        return None
    kp = KeywordProcessor(case_sensitive=False)
    for t in glossary_terms:
        t = t.strip().lower()
        if len(t) >= 3:
            kp.add_keyword(t)
    return kp

def extract_gt_terms_from_text(text: str, glossary_terms: List[str], kp=None) -> Set[str]:
    txt = (text or "").lower()
    if kp is not None:
        found = kp.extract_keywords(txt)
        return set(k.lower() for k in found)
    out = set()
    for t in glossary_terms:
        if len(t) >= 3 and t in txt:
            out.add(t)
    return out

def extract_gt_term_occurrences_from_text(
    text: str,
    glossary_terms: List[str],
    kp=None,
    term_canonical_map: Optional[Dict[str, str]] = None,
    merge_plural_terms: bool = False,
) -> List[Tuple[str, int, int]]:
    """
    Extract GT term occurrences (with spans) from transcript.

    Returns a list of (term_lower, start_char, end_char) sorted by appearance.
    This keeps duplicates (same term can appear multiple times).
    """
    txt = (text or "").lower()
    if not txt:
        return []

    # Fast path: FlashText provides span info efficiently.
    if kp is not None:
        try:
            found = kp.extract_keywords(txt, span_info=True)
            # found: List[Tuple[keyword, start, end]]
            occ = [(k.lower(), int(s), int(e)) for (k, s, e) in found]
            occ = _dedup_overlapping_occurrences(occ)
            if merge_plural_terms:
                if term_canonical_map is None:
                    occ = [(_canonicalize_plural_english(k), s, e) for (k, s, e) in occ]
                else:
                    occ = [(term_canonical_map.get(k, k), s, e) for (k, s, e) in occ]
            return occ
        except TypeError:
            # Older flashtext may not support span_info; fall back below.
            pass

    # Fallback (slower): shortlist by substring, then regex finditer with word boundaries.
    occ: List[Tuple[str, int, int]] = []
    for t in glossary_terms:
        term = (t or "").strip().lower()
        if not term or len(term) < 3:
            continue
        if term not in txt:
            continue
        pattern = re.compile(r"\b" + re.escape(term) + r"\b")
        for m in pattern.finditer(txt):
            occ.append((term, int(m.start()), int(m.end())))
    occ = _dedup_overlapping_occurrences(occ)
    if merge_plural_terms:
        if term_canonical_map is None:
            occ = [(_canonicalize_plural_english(k), s, e) for (k, s, e) in occ]
        else:
            occ = [(term_canonical_map.get(k, k), s, e) for (k, s, e) in occ]
    return occ

def sequential_match(pred_list: List[str], transcript: str, gt_terms: Set[str]):
    text = (transcript or "").lower()
    hits = 0
    fps = 0
    matched_gt = []
    matched_set = set()

    for term in pred_list:
        term_lower = (term or "").lower()
        if not term_lower:
            continue

        # Independent matching (no ordering constraint). Also prevent double-counting:
        # `gt_terms` is a set (unique terms), so each GT term should contribute at most 1 hit.
        if term_lower in gt_terms and term_lower not in matched_set:
            pattern = re.compile(r"\b" + re.escape(term_lower) + r"\b")
            match = pattern.search(text)
            if match:
                hits += 1
                matched_set.add(term_lower)
                matched_gt.append(term_lower)
                continue

        # If not a new GT hit, count as FP (includes duplicates and non-GT predictions).
        fps += 1
    return hits, fps, matched_gt

def max_subsequence_matches_over_windows(
    pred_windows: List[List[str]],
    gt_seq: List[str],
) -> Tuple[int, List[Tuple[int, str]]]:
    """
    Find the maximum number of GT terms that can be matched as an ordered subsequence
    over an ordered list of prediction windows.

    - Each window is a set/list of candidate terms (top-k). A GT term matches a window
      if the term is contained in the window candidates.
    - Order is preserved across both GT sequence and window sequence.
    - Both windows and GT terms can be skipped (noise / missing term).

    Returns:
      - best_match_count
      - alignment: list of (window_index, gt_term) pairs in matched order
    """
    W = len(pred_windows)
    G = len(gt_seq)
    if W == 0 or G == 0:
        return 0, []

    pred_sets = [set(w) for w in pred_windows]

    # Unlimited matches per window (still ordered across windows and GT sequence).
    #
    # dp[i][j] = best matches using first i windows and first j gt terms.
    # For window i-1, we can match *any number* of gt terms in a suffix [t, j),
    # as long as each matched term is contained in pred_sets[i-1]. Earlier windows
    # only see the prefix [0, t).
    #
    # dp[i][j] = max_{t in [0..j]} dp[i-1][t] + count(gt_seq[t:j] ∩ pred_sets[i-1])
    #
    # This can be computed in O(W*G) using a running max of (dp[i-1][t] - prefix_hits[t]).
    dp = [[0] * (G + 1) for _ in range(W + 1)]
    parent_t = [[0] * (G + 1) for _ in range(W + 1)]  # best split t for dp[i][j]

    for i in range(1, W + 1):
        ps = pred_sets[i - 1]
        # prefix_hits[j] = number of gt terms among first j that are in ps
        prefix_hits = [0] * (G + 1)
        for j in range(1, G + 1):
            prefix_hits[j] = prefix_hits[j - 1] + (1 if gt_seq[j - 1] in ps else 0)

        # best_val = max_t (dp[i-1][t] - prefix_hits[t]) so far
        best_val = dp[i - 1][0] - prefix_hits[0]
        best_t = 0

        for j in range(0, G + 1):
            if j > 0:
                cand_val = dp[i - 1][j] - prefix_hits[j]
                if cand_val > best_val:
                    best_val = cand_val
                    best_t = j

            dp[i][j] = prefix_hits[j] + best_val
            parent_t[i][j] = best_t

    # Backtrace: recover splits and emit per-window matches.
    i, j = W, G
    alignment_rev: List[Tuple[int, str]] = []
    while i > 0 and j >= 0:
        t = parent_t[i][j]
        # Terms in [t, j) that are in pred_sets[i-1] are matched to window (i-1)
        ps = pred_sets[i - 1]
        for k in range(j - 1, t - 1, -1):
            if gt_seq[k] in ps:
                alignment_rev.append((i - 1, gt_seq[k]))
        i -= 1
        j = t

    alignment_rev.reverse()
    return dp[W][G], alignment_rev

def get_current_topk_from_agg(
    term_scores: Dict[str, float],
    term_votes: Optional[Dict[str, int]],
    top_k: int,
    voting_min_votes: int,
    strategy: str,
):
    if strategy == "voting" and term_votes is not None:
        cand = [(t, term_scores.get(t, -1e9)) for t, v in term_votes.items() if v >= voting_min_votes]
        cand.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in cand[:top_k]]
    cand = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
    return [t for t, _ in cand[:top_k]]

def run_acl_simulation(
    retriever, 
    faiss_index, 
    glossary_terms, 
    wav_files, 
    text_lines, 
    args, 
    device, 
    feature_extractor,
    limit=100
):
    """Core simulation loop for ACL evaluation."""
    # Optional term normalization for evaluation-only canonicalization.
    merge_plural_terms = bool(getattr(args, "merge_plural_terms", False))
    term_canonical_map = getattr(args, "term_canonical_map", None)
    kp = build_keyword_processor(glossary_terms)
    SR = 16000
    chunk_samples = int(args.rag_chunk_size * SR)
    hop_samples = int(args.rag_hop_size * SR)
    vllm_interval_samples = int(args.vllm_interval * SR)

    total_gt = 0
    total_hits = 0
    total_fps = 0
    used_samples = 0
    skipped_no_gt = 0

    pos_scores = []
    neg_scores = []
    gaps_top1_top5 = []
    gt_minus_mean_top5 = []

    # Debug printing controls (default: no extra printing)
    debug_print_limit = int(getattr(args, "debug_print_limit", 0) or 0)
    debug_miss_limit = int(getattr(args, "debug_miss_limit", 0) or 0)
    debug_printed = 0
    debug_missed = 0

    for wav_path, txt in tqdm(list(zip(wav_files, text_lines))[:limit], desc="ACL Eval", leave=False):
        # Extract GT occurrences with positions (keep duplicates) and keep transcript order.
        gt_occ = extract_gt_term_occurrences_from_text(
            txt,
            glossary_terms,
            kp=kp,
            term_canonical_map=term_canonical_map,
            merge_plural_terms=merge_plural_terms,
        )
        if not gt_occ:
            skipped_no_gt += 1
            continue
        used_samples += 1
        gt_seq = [t for (t, _s, _e) in gt_occ]
        total_gt += len(gt_seq)
        
        audio, sr = sf.read(wav_path)
        if sr != SR: continue
        if audio.ndim > 1: audio = audio.mean(axis=1)

        term_scores: Dict[str, float] = {}
        term_votes: Optional[Dict[str, int]] = {} if args.rag_strategy == "voting" else None
        pred_windows: List[List[str]] = []
        last_vllm_trigger = 0

        # Build trigger windows
        trigger_windows = []
        last_pos = 0
        for end in range(0, len(audio) + hop_samples, hop_samples):
            end = min(len(audio), end)
            if end == 0: continue
            
            is_last = (end >= len(audio))
            if (end - last_vllm_trigger) >= vllm_interval_samples or is_last:
                start = max(0, end - chunk_samples)
                w = audio[start:end]
                if len(w) < chunk_samples:
                    w = np.pad(w, (0, chunk_samples - len(w)))
                trigger_windows.append(w)
                last_vllm_trigger = end
            
            if is_last: break

        # Batch encode
        for j in range(0, len(trigger_windows), getattr(args, 'acl_audio_batch_size', 32)):
            batch_win = trigger_windows[j : j + getattr(args, 'acl_audio_batch_size', 32)]
            inputs = feature_extractor(batch_win, sampling_rate=SR, return_tensors="pt", padding=False)
            input_features = inputs.input_features.to(device).to(torch.bfloat16)
            f_lens = torch.full((input_features.shape[0],), input_features.shape[-1], device=device)

            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    audio_embs = retriever(input_features, f_lens)
                audio_embs_np = audio_embs.cpu().float().numpy()
                faiss.normalize_L2(audio_embs_np)
                D, I = faiss_index.search(audio_embs_np, args.rag_voting_k)

            for b in range(D.shape[0]):
                for rank in range(I.shape[1]):
                    idx = int(I[b, rank])
                    if idx < 0: continue
                    term = glossary_terms[idx]
                    if merge_plural_terms:
                        if term_canonical_map is None:
                            term = _canonicalize_plural_english(term)
                        else:
                            term = term_canonical_map.get(term, term)
                    # Convert raw cosine similarity to 1/(1+L2) score to match online inference.
                    score = l2_distance_to_score(float(D[b, rank]))
                    if score < args.score_threshold: continue
                    if term not in term_scores or score > term_scores[term]:
                        term_scores[term] = score
                    if term_votes is not None:
                        term_votes[term] = term_votes.get(term, 0) + 1

                curr_topk = get_current_topk_from_agg(
                    term_scores=term_scores,
                    term_votes=term_votes,
                    top_k=args.top_k,
                    voting_min_votes=args.rag_voting_min_votes,
                    strategy=args.rag_strategy,
                )
                # One prediction group per trigger window (even if empty after thresholding)
                pred_windows.append(curr_topk or [])

        # ---------------- Sequence-aware matching (skip allowed) ----------------
        # Treat each trigger window as a token (a set of top-k candidates). Find the best
        # ordered subsequence match between GT term sequence and window sequence.
        n_windows = len(pred_windows)
        if n_windows <= 0:
            continue

        best_hits, alignment = max_subsequence_matches_over_windows(pred_windows, gt_seq)
        total_hits += best_hits

        # Define FP as all predicted candidates that were not used for GT matching.
        # This keeps precision comparable to the previous "flattened pred list" approach.
        total_pred_terms = sum(len(w) for w in pred_windows)
        fps = max(0, total_pred_terms - best_hits)
        total_fps += fps

        if debug_printed < debug_print_limit:
            # Print first N segments for manual inspection.
            # Use JSON-like list of dicts: [{...}, {...}, ...]
            pred_dbg = [{"w": i, "pred": w} for i, w in enumerate(pred_windows)]
            align_dbg = [{"w": wi, "gt": t} for (wi, t) in alignment]
            seg_recall = (best_hits / len(gt_seq)) if gt_seq else 0.0
            print("\n" + "-" * 60)
            print("[DBG] Segment preview")
            print(f"[DBG] wav: {wav_path}")
            print(f"[DBG] text: {txt.strip()}")
            print(f"[DBG] gt_seq(len={len(gt_seq)}): {gt_seq}")
            print(f"[DBG] pred_windows(len={len(pred_windows)}): {pred_dbg}")
            print(f"[DBG] alignment(hits={best_hits}, recall={seg_recall:.3f}): {align_dbg}")
            print("-" * 60)
            debug_printed += 1

        # Print missed samples (where not all GT occurrences are matched)
        if debug_missed < debug_miss_limit and best_hits < len(gt_seq):
            from collections import Counter
            gt_cnt = Counter(gt_seq)
            matched_terms = [t for (_wi, t) in alignment]
            matched_cnt = Counter(matched_terms)
            missing_cnt = gt_cnt - matched_cnt
            missing_terms = []
            for t, c in missing_cnt.items():
                missing_terms.extend([t] * int(c))

            pred_dbg = [{"w": i, "pred": w} for i, w in enumerate(pred_windows)]
            align_dbg = [{"w": wi, "gt": t} for (wi, t) in alignment]
            seg_recall = (best_hits / len(gt_seq)) if gt_seq else 0.0

            print("\n" + "!" * 60)
            print("[MISS] Segment has missing GT terms")
            print(f"[MISS] wav: {wav_path}")
            print(f"[MISS] text: {txt.strip()}")
            print(f"[MISS] gt_seq(len={len(gt_seq)}): {gt_seq}")
            print(f"[MISS] missing_terms(len={len(missing_terms)}): {missing_terms}")
            print(f"[MISS] pred_windows(len={len(pred_windows)}): {pred_dbg}")
            print(f"[MISS] alignment(hits={best_hits}, recall={seg_recall:.3f}): {align_dbg}")
            print("!" * 60)
            debug_missed += 1

        if term_scores:
            sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
            top5 = sorted_terms[:5]
            top5_scores = [s for _, s in top5]
            if len(top5_scores) >= 5:
                gaps_top1_top5.append(top5_scores[0] - top5_scores[4])
            # For score stats, treat unique GT terms (set) as before.
            gt_terms_unique = set(t for t, _, _ in gt_occ)
            gt_scores = [term_scores[t] for t in gt_terms_unique if t in term_scores]
            if gt_scores:
                pos_scores.append(float(np.mean(gt_scores)))
                neg_cands = [s for t, s in top5 if t not in gt_terms_unique]
                neg_scores.append(float(neg_cands[0] if neg_cands else 0.0))
                gt_minus_mean_top5.append(float(np.mean(gt_scores) - float(np.mean(top5_scores))))

    recall = total_hits / total_gt if total_gt > 0 else 0.0
    precision = total_hits / (total_hits + total_fps) if (total_hits + total_fps) > 0 else 0.0
    
    return {
        "recall": recall,
        "precision": precision,
        "pos_score_mean": float(np.mean(pos_scores)) if pos_scores else 0.0,
        "neg_score_mean": float(np.mean(neg_scores)) if neg_scores else 0.0,
        "margin": float(np.mean(pos_scores) - np.mean(neg_scores)) if pos_scores and neg_scores else 0.0,
        "gap_top1_top5_mean": float(np.mean(gaps_top1_top5)) if gaps_top1_top5 else 0.0,
        "gt_minus_mean_top5_mean": float(np.mean(gt_minus_mean_top5)) if gt_minus_mean_top5 else 0.0,
        "used_samples": used_samples,
        "skipped_no_gt": skipped_no_gt,
        "total_gt": total_gt,
        "total_hits": total_hits,
        "total_fps": total_fps
    }

