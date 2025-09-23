#!/usr/bin/env python3
"""
Data cleaning script for term_preprocessed_samples files.
This script removes ground_truth_terms that are either:
1. Not present in glossary_merged.json
2. Marked as confused=true in glossary_merged.json

Only keeps terms that exist in glossary and have confused=false.
"""

import json
import os
import argparse
from typing import Dict, List, Set
from pathlib import Path


def load_glossary(glossary_path: str) -> Set[str]:
    """
    Load glossary_merged.json and return a set of valid terms (confused=false).
    
    Args:
        glossary_path: Path to glossary_merged.json file
        
    Returns:
        Set of valid term names that have confused=false
    """
    print(f"Loading glossary from {glossary_path}...")
    valid_terms = set()
    
    with open(glossary_path, 'r', encoding='utf-8') as f:
        glossary = json.load(f)
    
    for term_key, term_data in glossary.items():
        # Only keep terms that exist and have confused=false
        if not term_data.get('confused', True):  # Default to True if confused field is missing
            valid_terms.add(term_data.get('term', term_key))
    
    print(f"Loaded {len(valid_terms)} valid terms from glossary")
    return valid_terms


def clean_sample_terms(sample: Dict, valid_terms: Set[str]) -> Dict:
    """
    Clean ground_truth_term field in a single sample.
    
    Args:
        sample: Single sample dictionary
        valid_terms: Set of valid terms from glossary
        
    Returns:
        Cleaned sample dictionary
    """
    if 'ground_truth_term' in sample and isinstance(sample['ground_truth_term'], list):
        original_terms = sample['ground_truth_term']
        cleaned_terms = [term for term in original_terms if term in valid_terms]
        
        sample['ground_truth_term'] = cleaned_terms
        sample['has_target'] = len(cleaned_terms) > 0
        
        # Log if terms were removed
        removed_terms = set(original_terms) - set(cleaned_terms)
        if removed_terms:
            print(f"Sample {sample.get('segment_id', 'unknown')}: Removed terms {removed_terms}")
    
    return sample


def process_file(input_path: str, output_path: str, valid_terms: Set[str]) -> None:
    """
    Process a single term_preprocessed_samples file.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output cleaned JSON file
        valid_terms: Set of valid terms from glossary
    """
    print(f"Processing {input_path}...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    print(f"Loaded {len(samples)} samples")
    
    # Clean each sample
    cleaned_samples = []
    original_term_count = 0
    cleaned_term_count = 0
    samples_with_targets = 0
    
    for sample in samples:
        if 'ground_truth_term' in sample:
            original_term_count += len(sample['ground_truth_term'])
        
        cleaned_sample = clean_sample_terms(sample, valid_terms)
        cleaned_samples.append(cleaned_sample)
        
        if cleaned_sample.get('has_target', False):
            samples_with_targets += 1
            cleaned_term_count += len(cleaned_sample['ground_truth_term'])
    
    # Save cleaned data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_samples, f, indent=2, ensure_ascii=False)
    
    print(f"Saved cleaned data to {output_path}")
    print(f"Statistics:")
    print(f"  - Total samples: {len(cleaned_samples)}")
    print(f"  - Samples with targets: {samples_with_targets}")
    print(f"  - Original terms: {original_term_count}")
    print(f"  - Cleaned terms: {cleaned_term_count}")
    print(f"  - Removed terms: {original_term_count - cleaned_term_count}")
    print()


def main():
    parser = argparse.ArgumentParser(description='Clean ground_truth_terms using glossary')
    parser.add_argument('--glossary', required=True, help='Path to glossary_merged.json')
    parser.add_argument('--input-dir', required=True, help='Directory containing term_preprocessed_samples files')
    parser.add_argument('--output-dir', required=True, help='Output directory for cleaned files')
    parser.add_argument('--files', nargs='*', help='Specific files to process (optional)')
    
    args = parser.parse_args()
    
    # Load glossary
    valid_terms = load_glossary(args.glossary)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define all target files
    all_files = [
        'term_preprocessed_samples_0_500000.json',
        'term_preprocessed_samples_500000_1000000.json',
        'term_preprocessed_samples_1000000_1500000.json',
        'term_preprocessed_samples_1500000_2000000.json',
        'term_preprocessed_samples_2000000_2500000.json',
        'term_preprocessed_samples_2500000_3000000.json',
        'term_preprocessed_samples_3000000_3500000.json',
        'term_preprocessed_samples_3500000_4000000.json',
        'term_preprocessed_samples_4000000_4500000.json',
        'term_preprocessed_samples_4500000_5000000.json',
        'term_preprocessed_samples_5000000_5500000.json',
        'term_preprocessed_samples_5500000_6000000.json',
        'term_preprocessed_samples_6000000_6500000.json',
        'term_preprocessed_samples_6500000_7000000.json',
        'term_preprocessed_samples_7000000_7500000.json',
        'term_preprocessed_samples_7500000_8000000.json',
        'term_preprocessed_samples_8000000_end.json'
    ]
    
    # Use specified files or all files
    files_to_process = args.files if args.files else all_files
    
    # Process each file
    for filename in files_to_process:
        input_path = os.path.join(args.input_dir, filename)
        output_path = os.path.join(args.output_dir, filename)
        
        if os.path.exists(input_path):
            try:
                process_file(input_path, output_path, valid_terms)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        else:
            print(f"Warning: File {input_path} not found, skipping...")
    
    print("Data cleaning completed!")


if __name__ == '__main__':
    main()

