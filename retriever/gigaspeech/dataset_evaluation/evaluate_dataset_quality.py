#!/usr/bin/env python3
"""
Evaluate dataset quality by analyzing ground truth terms in xl_cleaned samples
and calculating their coverage in the glossary.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Set, Dict, Any
import argparse

def load_glossary_terms(glossary_path: str) -> Set[str]:
    """Load all valid terms from glossary file, excluding confused terms."""
    print(f"Loading glossary from {glossary_path}...")
    
    if not os.path.exists(glossary_path):
        raise FileNotFoundError(f"Glossary file not found: {glossary_path}")
    
    with open(glossary_path, 'r', encoding='utf-8') as f:
        glossary_data = json.load(f)
    
    # Check if this is the merged glossary format with term info
    if isinstance(next(iter(glossary_data.values())), dict):
        # This is glossary_merged.json format - filter out confused terms
        valid_terms = set()
        confused_count = 0
        
        for term, info in glossary_data.items():
            if info.get('confused', False):
                confused_count += 1
            else:
                valid_terms.add(term.lower())  # Convert to lowercase for consistency
        
        print(f"Total terms in glossary: {len(glossary_data):,}")
        print(f"Terms with confused=true (excluded): {confused_count:,}")
        print(f"Valid terms loaded: {len(valid_terms):,}")
        return valid_terms
    else:
        # This is glossary_term2idx.json format - use all terms
        terms = {term.lower() for term in glossary_data.keys()}
        print(f"Loaded {len(terms):,} terms from glossary (term2idx format)")
        return terms

def extract_ground_truth_terms(samples_dir: str) -> tuple[Set[str], int]:
    """Extract all ground truth terms from sample files.
    
    Returns:
        tuple: (unique_terms_set, total_term_count_including_duplicates)
    """
    print(f"Extracting ground truth terms from {samples_dir}...")
    
    samples_path = Path(samples_dir)
    if not samples_path.exists():
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")
    
    all_terms = set()
    total_term_count = 0  # Count including duplicates
    sample_files = list(samples_path.glob("term_preprocessed_samples_*.json"))
    
    if not sample_files:
        raise FileNotFoundError(f"No sample files found in {samples_dir}")
    
    print(f"Found {len(sample_files)} sample files to process")
    
    for i, file_path in enumerate(sorted(sample_files), 1):
        print(f"Processing file {i}/{len(sample_files)}: {file_path.name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                samples = json.load(f)
            
            file_terms = set()
            file_term_count = 0
            for sample in samples:
                if 'ground_truth_term' in sample and sample['ground_truth_term']:
                    for term in sample['ground_truth_term']:
                        if term and isinstance(term, str):
                            all_terms.add(term)
                            file_terms.add(term)
                            total_term_count += 1
                            file_term_count += 1
            
            print(f"  Found {len(file_terms):,} unique terms, {file_term_count:,} total terms in this file")
            
        except Exception as e:
            print(f"  Error processing {file_path.name}: {e}")
            continue
    
    print(f"Total unique ground truth terms found: {len(all_terms):,}")
    print(f"Total ground truth terms (including duplicates): {total_term_count:,}")
    return all_terms, total_term_count

def calculate_coverage(ground_truth_terms: Set[str], glossary_terms: Set[str], total_term_count: int = None) -> Dict[str, Any]:
    """Calculate coverage statistics."""
    print("Calculating coverage statistics...")
    
    # Convert ground truth terms to lowercase for comparison
    gt_terms_lower = {term.lower() for term in ground_truth_terms}
    
    # Terms that are in both ground truth and glossary (case-insensitive)
    covered_terms_lower = gt_terms_lower.intersection(glossary_terms)
    
    # Find original casing for covered terms
    covered_terms_original = set()
    for original_term in ground_truth_terms:
        if original_term.lower() in covered_terms_lower:
            covered_terms_original.add(original_term)
    
    # Terms in ground truth but not in glossary (case-insensitive)
    missing_terms_lower = gt_terms_lower - glossary_terms
    missing_terms_original = set()
    for original_term in ground_truth_terms:
        if original_term.lower() in missing_terms_lower:
            missing_terms_original.add(original_term)
    
    # Coverage percentage
    coverage_percentage = (len(covered_terms_lower) / len(ground_truth_terms) * 100) if ground_truth_terms else 0
    
    # Percentage of glossary used
    glossary_usage_percentage = (len(covered_terms_lower) / len(glossary_terms) * 100) if glossary_terms else 0
    
    stats = {
        'total_ground_truth_terms': len(ground_truth_terms),
        'total_ground_truth_terms_with_duplicates': total_term_count if total_term_count else len(ground_truth_terms),
        'total_glossary_terms': len(glossary_terms),
        'covered_terms': len(covered_terms_lower),
        'missing_terms': len(missing_terms_lower),
        'coverage_percentage': coverage_percentage,
        'glossary_usage_percentage': glossary_usage_percentage,
        'covered_terms_list': sorted(list(covered_terms_original)),
        'missing_terms_list': sorted(list(missing_terms_original))
    }
    
    return stats

def print_statistics(stats: Dict[str, Any]):
    """Print coverage statistics in a readable format."""
    print("\n" + "="*60)
    print("DATASET QUALITY EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nGlossary Statistics:")
    print(f"  Total terms in glossary: {stats['total_glossary_terms']:,}")
    
    print(f"\nGround Truth Terms Statistics:")
    print(f"  Total unique ground truth terms: {stats['total_ground_truth_terms']:,}")
    print(f"  Total ground truth terms (including duplicates): {stats['total_ground_truth_terms_with_duplicates']:,}")
    print(f"  Terms found in glossary: {stats['covered_terms']:,}")
    print(f"  Terms missing from glossary: {stats['missing_terms']:,}")
    
    # Calculate duplication ratio
    if stats['total_ground_truth_terms'] > 0:
        duplication_ratio = stats['total_ground_truth_terms_with_duplicates'] / stats['total_ground_truth_terms']
        print(f"  Average term repetition: {duplication_ratio:.2f}x")
    
    print(f"\nCoverage Analysis:")
    print(f"  Coverage rate: {stats['coverage_percentage']:.2f}%")
    print(f"  Glossary utilization: {stats['glossary_usage_percentage']:.2f}%")
    
    if stats['missing_terms'] > 0:
        print(f"\nSample missing terms (first 10):")
        for term in stats['missing_terms_list'][:10]:
            print(f"  - '{term}'")
        if len(stats['missing_terms_list']) > 10:
            print(f"  ... and {len(stats['missing_terms_list']) - 10} more")

def save_detailed_results(stats: Dict[str, Any], output_path: str):
    """Save detailed results to JSON file."""
    print(f"\nSaving detailed results to {output_path}")
    
    # Remove the full lists from stats for summary
    summary_stats = {k: v for k, v in stats.items() 
                    if k not in ['covered_terms_list', 'missing_terms_list']}
    
    detailed_results = {
        'summary': summary_stats,
        'covered_terms': stats['covered_terms_list'],
        'missing_terms': stats['missing_terms_list']
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description='Evaluate dataset quality by analyzing ground truth terms coverage')
    parser.add_argument('--samples_dir', 
                       default='/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/samples/xl_cleaned',
                       help='Directory containing sample files')
    parser.add_argument('--glossary_path',
                       default='/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_merged.json',
                       help='Path to glossary file')
    parser.add_argument('--output',
                       default='dataset_quality_results.json',
                       help='Output file for detailed results')
    
    args = parser.parse_args()
    
    try:
        # Load glossary terms
        glossary_terms = load_glossary_terms(args.glossary_path)
        
        # Extract ground truth terms from samples
        ground_truth_terms, total_term_count = extract_ground_truth_terms(args.samples_dir)
        
        # Calculate coverage statistics
        stats = calculate_coverage(ground_truth_terms, glossary_terms, total_term_count)
        
        # Print results
        print_statistics(stats)
        
        # Save detailed results
        save_detailed_results(stats, args.output)
        
        print(f"\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
