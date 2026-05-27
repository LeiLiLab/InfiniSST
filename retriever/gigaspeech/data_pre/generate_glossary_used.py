#!/usr/bin/env python3
"""
Generate glossary_used.json by taking the intersection of:
1. All ground truth terms from balanced_train_set.json and balanced_test_set.json
2. Terms in glossary_cleaned.json
"""

import json
import os


def extract_terms_from_dataset(dataset_path):
    """Extract all ground truth terms from a dataset file."""
    print(f"[INFO] Loading dataset from: {dataset_path}")
    with open(dataset_path, "r") as f:
        samples = json.load(f)
    
    terms = set()
    for sample in samples:
        ground_truth_terms = sample.get('term_chunk_audio_ground_truth_terms', [])
        if not isinstance(ground_truth_terms, list):
            continue
        for term in ground_truth_terms:
            if isinstance(term, str) and len(term.strip()) >= 3:
                terms.add(term.lower())
    
    print(f"[INFO] Extracted {len(terms)} unique terms from {len(samples)} samples")
    return terms


def main():
    # Paths
    base_path = "/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data"
    train_path = os.path.join(base_path, "balanced_train_set.json")
    test_path = os.path.join(base_path, "balanced_test_set.json")
    glossary_cleaned_path = "/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_cleaned.json"
    output_path = "/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_used.json"
    
    # Step 1: Extract terms from train and test datasets
    print("=" * 80)
    print("STEP 1: Extracting terms from balanced datasets")
    print("=" * 80)
    
    train_terms = extract_terms_from_dataset(train_path)
    test_terms = extract_terms_from_dataset(test_path)
    
    # Combine train and test terms
    used_terms = train_terms | test_terms
    print(f"[INFO] Combined unique terms (train + test): {len(used_terms)}")
    print(f"[INFO]   - Train only: {len(train_terms - test_terms)}")
    print(f"[INFO]   - Test only: {len(test_terms - train_terms)}")
    print(f"[INFO]   - Overlap: {len(train_terms & test_terms)}")
    
    # Step 2: Load glossary_cleaned.json and find intersection
    print("\n" + "=" * 80)
    print("STEP 2: Loading glossary_cleaned.json")
    print("=" * 80)
    
    print(f"[INFO] Loading glossary from: {glossary_cleaned_path}")
    with open(glossary_cleaned_path, "r") as f:
        glossary_cleaned = json.load(f)
    
    glossary_keys = set(glossary_cleaned.keys())
    print(f"[INFO] Total terms in glossary_cleaned: {len(glossary_keys)}")
    
    # Step 3: Find intersection
    print("\n" + "=" * 80)
    print("STEP 3: Finding intersection")
    print("=" * 80)
    
    # Match used_terms with glossary keys (both are lowercase)
    intersection_keys = used_terms & glossary_keys
    print(f"[INFO] Intersection (used terms in glossary): {len(intersection_keys)}")
    print(f"[INFO] Used terms NOT in glossary: {len(used_terms - glossary_keys)}")
    
    # Show some examples of terms not in glossary
    not_in_glossary = list(used_terms - glossary_keys)[:20]
    if not_in_glossary:
        print(f"[DEBUG] Sample terms not in glossary: {not_in_glossary}")
    
    # Step 4: Create glossary_used with entries from glossary_cleaned
    print("\n" + "=" * 80)
    print("STEP 4: Creating glossary_used.json")
    print("=" * 80)
    
    glossary_used = {}
    for key in intersection_keys:
        glossary_used[key] = glossary_cleaned[key]
    
    print(f"[INFO] glossary_used entries: {len(glossary_used)}")
    
    # Step 5: Save to output file
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[INFO] Saving to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(glossary_used, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] ✅ Successfully created glossary_used.json with {len(glossary_used)} entries")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Train terms: {len(train_terms)}")
    print(f"Test terms: {len(test_terms)}")
    print(f"Combined used terms: {len(used_terms)}")
    print(f"Glossary cleaned terms: {len(glossary_keys)}")
    print(f"Final glossary_used terms: {len(glossary_used)}")


if __name__ == "__main__":
    main()
