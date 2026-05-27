#!/usr/bin/env python3
"""
Merge positive term dataset and negative (no-term) dataset with a specific ratio.
Final dataset is shuffled.
"""

import os
import sys
import json
import random
import argparse
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos-jsonl", required=True, help="Positive term dataset")
    parser.add_argument("--neg-jsonl", required=True, help="Negative (no-term) dataset")
    parser.add_argument("--output-jsonl", required=True, help="Final merged dataset")
    parser.add_argument("--ratio", type=float, default=0.1, help="Target ratio of negative samples in the final dataset")
    parser.add_argument("--all-neg", action="store_true", help="Use all negative samples regardless of ratio")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # 1. Load all positive samples
    logger.info(f"Loading positive samples from {args.pos_jsonl}...")
    pos_samples = []
    with open(args.pos_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                pos_samples.append(line.strip())
            except:
                continue
    n_pos = len(pos_samples)
    logger.info(f"Loaded {n_pos} positive samples.")

    # 2. Load all negative samples (we don't load all if the file is huge, but here we probably need to count first or sample on the fly)
    # For simplicity, let's load them all since memory should be enough for metadata.
    logger.info(f"Loading negative samples from {args.neg_jsonl}...")
    neg_samples = []
    with open(args.neg_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                neg_samples.append(line.strip())
            except:
                continue
    n_neg_total = len(neg_samples)
    logger.info(f"Loaded {n_neg_total} negative samples.")

    # 3. Calculate target negative count
    if args.all_neg:
        n_neg_final = n_neg_total
        logger.info(f"Using ALL negative samples as requested. Count: {n_neg_final}")
    else:
        # n_neg / (n_pos + n_neg) = ratio  => n_neg = ratio * n_pos / (1 - ratio)
        if args.ratio >= 1.0 or args.ratio < 0:
            logger.error("Ratio must be between 0 and 1 (exclusive of 1).")
            return

        n_neg_target = int((n_pos * args.ratio) / (1 - args.ratio))
        n_neg_final = min(n_neg_target, n_neg_total)
        
        logger.info(f"Target negative samples: {n_neg_target}, Available: {n_neg_total}, Final: {n_neg_final}")

    # 4. Sampling negatives
    if n_neg_final < n_neg_total:
        selected_neg = random.sample(neg_samples, n_neg_final)
    else:
        selected_neg = neg_samples

    # 5. Merge and Shuffle
    logger.info("Merging and shuffling...")
    all_samples = pos_samples + selected_neg
    random.shuffle(all_samples)

    # 6. Write to output
    logger.info(f"Writing {len(all_samples)} samples to {args.output_jsonl}...")
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for item in tqdm(all_samples, desc="Writing"):
            f.write(item + "\n")

    logger.info("Done.")

if __name__ == "__main__":
    main()

