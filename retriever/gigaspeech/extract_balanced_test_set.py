#!/usr/bin/env python3
"""
Balanced Test Set Extraction Script

从训练集JSON中提取1000个unique terms的样本，形成新的测试集，
确保20%的terms完全不在剩余的训练集中，实现真正的unseen terms测试。

使用方法:
python extract_balanced_test_set.py \
    --input_path data/xl_cleaned_term_level_chunks_merged.json \
    --output_train_path data/balanced_train_set.json \
    --output_test_path data/balanced_test_set.json \
    --test_size 1000 \
    --unseen_ratio 0.20

输出:
- balanced_train_set.json: 新的训练集（移除了测试集样本）
- balanced_test_set.json: 新的测试集（1000个样本，20% unseen terms）
"""

import json
import argparse
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Set
import os

def load_samples(path: str) -> List[Dict]:
    """加载样本数据"""
    print(f"Loading samples from: {path}")
    with open(path, "r") as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} samples")
    return samples

def filter_valid_samples(samples: List[Dict]) -> List[Dict]:
    """过滤有效样本，只保留有术语的样本"""
    valid_samples = []
    
    for sample in samples:
        terms = sample.get('term_chunk_audio_ground_truth_terms', [])
        if not isinstance(terms, list):
            continue
            
        # 过滤术语
        filtered_terms = []
        for t in terms:
            if isinstance(t, str) and len(t) >= 3:
                # 过滤数字过多的术语
                if sum(c.isdigit() for c in t) <= len(t) // 2:
                    filtered_terms.append(t.lower())
        
        # 过滤黑名单前后缀
        black_words = ['yeah', 'this ']
        black_suffixes = ['years']
        filtered_terms = [
            t for t in filtered_terms 
            if not any(t.startswith(prefix.lower()) for prefix in black_words)
            and not any(t.endswith(suffix.lower()) for suffix in black_suffixes)
        ]
        
        # 只保留有有效术语的样本
        if filtered_terms and sample.get('term_chunk_text', '').strip() and sample.get('term_chunk_audio', ''):
            sample_copy = dict(sample)
            sample_copy['term_chunk_audio_ground_truth_terms'] = filtered_terms
            valid_samples.append(sample_copy)
    
    print(f"Filtered to {len(valid_samples)} valid samples with terms")
    return valid_samples

def build_term_to_samples_map(samples: List[Dict]) -> Dict[str, List[Dict]]:
    """构建术语到样本的映射"""
    term_to_samples = defaultdict(list)
    
    for sample in samples:
        terms = sample.get('term_chunk_audio_ground_truth_terms', [])
        for term in terms:
            if isinstance(term, str):
                term_to_samples[term.lower()].append(sample)
    
    print(f"Found {len(term_to_samples)} unique terms")
    return term_to_samples

def select_balanced_test_samples(
    term_to_samples: Dict[str, List[Dict]], 
    test_size: int = 1000, 
    unseen_ratio: float = 0.20,
    seed: int = 42
) -> Tuple[List[Dict], Set[str], Set[str]]:
    """
    选择平衡的测试样本
    
    Returns:
        test_samples: 测试样本列表
        seen_terms: 在训练集中也会出现的术语
        unseen_terms: 只在测试集中出现的术语
    """
    random.seed(seed)
    
    # 按术语频次排序，优先选择有足够样本的术语
    terms_by_frequency = sorted(
        term_to_samples.items(), 
        key=lambda x: len(x[1]), 
        reverse=True
    )
    
    print(f"Term frequency distribution (top 10):")
    for i, (term, samples) in enumerate(terms_by_frequency[:10]):
        print(f"  {i+1}. '{term}': {len(samples)} samples")
    
    # 计算需要的unseen和seen术语数量
    target_unseen_count = int(test_size * unseen_ratio)
    target_seen_count = test_size - target_unseen_count
    
    print(f"\nTarget composition:")
    print(f"  Total test samples: {test_size}")
    print(f"  Unseen terms: {target_unseen_count} ({unseen_ratio:.1%})")
    print(f"  Seen terms: {target_seen_count} ({1-unseen_ratio:.1%})")
    
    # 选择术语，确保每个术语至少有1个样本
    available_terms = [(term, samples) for term, samples in terms_by_frequency if len(samples) >= 1]
    
    if len(available_terms) < test_size:
        print(f"WARNING: Only {len(available_terms)} terms have samples, but need {test_size}")
        test_size = len(available_terms)
        target_unseen_count = int(test_size * unseen_ratio)
        target_seen_count = test_size - target_unseen_count
        print(f"Adjusted test_size to {test_size}")
    
    # 随机选择术语
    selected_terms = random.sample(available_terms, test_size)
    
    # 随机分配seen/unseen
    random.shuffle(selected_terms)
    unseen_terms_data = selected_terms[:target_unseen_count]
    seen_terms_data = selected_terms[target_unseen_count:]
    
    print(f"\nActual selection:")
    print(f"  Unseen terms: {len(unseen_terms_data)}")
    print(f"  Seen terms: {len(seen_terms_data)}")
    
    # 为每个选中的术语选择一个样本
    test_samples = []
    unseen_terms = set()
    seen_terms = set()
    
    # 处理unseen术语
    for term, samples in unseen_terms_data:
        # 随机选择一个样本
        selected_sample = random.choice(samples)
        test_samples.append(selected_sample)
        unseen_terms.add(term)
    
    # 处理seen术语
    for term, samples in seen_terms_data:
        # 随机选择一个样本
        selected_sample = random.choice(samples)
        test_samples.append(selected_sample)
        seen_terms.add(term)
    
    print(f"\nFinal test set:")
    print(f"  Total samples: {len(test_samples)}")
    print(f"  Unseen terms: {len(unseen_terms)}")
    print(f"  Seen terms: {len(seen_terms)}")
    
    return test_samples, seen_terms, unseen_terms

def remove_test_samples_from_train(
    all_samples: List[Dict], 
    test_samples: List[Dict],
    unseen_terms: Set[str]
) -> List[Dict]:
    """
    从训练集中移除测试样本，并移除包含unseen terms的训练样本
    """
    test_sample_ids = set()
    for sample in test_samples:
        # 使用音频路径和文本作为唯一标识
        sample_id = (sample.get('term_chunk_audio', ''), sample.get('term_chunk_text', ''))
        test_sample_ids.add(sample_id)
    
    train_samples = []
    removed_for_test = 0
    removed_for_unseen = 0
    
    for sample in all_samples:
        sample_id = (sample.get('term_chunk_audio', ''), sample.get('term_chunk_text', ''))
        
        # 跳过测试样本
        if sample_id in test_sample_ids:
            removed_for_test += 1
            continue
        
        # 检查是否包含unseen术语
        sample_terms = sample.get('term_chunk_audio_ground_truth_terms', [])
        sample_terms_lower = {t.lower() for t in sample_terms if isinstance(t, str)}
        
        # 如果包含任何unseen术语，则从训练集中移除
        if sample_terms_lower & unseen_terms:
            removed_for_unseen += 1
            continue
        
        train_samples.append(sample)
    
    print(f"\nTraining set filtering:")
    print(f"  Original samples: {len(all_samples)}")
    print(f"  Removed as test samples: {removed_for_test}")
    print(f"  Removed for containing unseen terms: {removed_for_unseen}")
    print(f"  Remaining training samples: {len(train_samples)}")
    
    return train_samples

def verify_separation(train_samples: List[Dict], test_samples: List[Dict]) -> None:
    """验证训练集和测试集的术语分离"""
    # 收集训练集术语
    train_terms = set()
    for sample in train_samples:
        terms = sample.get('term_chunk_audio_ground_truth_terms', [])
        train_terms.update(t.lower() for t in terms if isinstance(t, str))
    
    # 收集测试集术语
    test_terms = set()
    for sample in test_samples:
        terms = sample.get('term_chunk_audio_ground_truth_terms', [])
        test_terms.update(t.lower() for t in terms if isinstance(t, str))
    
    # 计算重叠
    overlap = train_terms & test_terms
    unseen_terms = test_terms - train_terms
    seen_terms = test_terms & train_terms
    
    print(f"\nVerification:")
    print(f"  Train terms: {len(train_terms)}")
    print(f"  Test terms: {len(test_terms)}")
    print(f"  Seen terms (in both): {len(seen_terms)}")
    print(f"  Unseen terms (test only): {len(unseen_terms)}")
    print(f"  Unseen ratio: {len(unseen_terms)/len(test_terms):.2%}")
    
    if len(overlap) != len(seen_terms):
        print(f"  ERROR: Overlap calculation mismatch!")
    
    # 显示一些unseen术语示例
    if unseen_terms:
        print(f"\nUnseen terms examples:")
        for i, term in enumerate(sorted(unseen_terms)[:10]):
            print(f"  {i+1}. '{term}'")
        if len(unseen_terms) > 10:
            print(f"  ... and {len(unseen_terms) - 10} more")

def save_dataset(samples: List[Dict], path: str, description: str) -> None:
    """保存数据集"""
    print(f"\nSaving {description} to: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(samples)} samples")

def main():
    parser = argparse.ArgumentParser(description="Extract balanced test set with controlled unseen terms ratio")
    parser.add_argument('--input_path', type=str, required=True,
                       help="Path to input training samples JSON")
    parser.add_argument('--output_train_path', type=str, required=True,
                       help="Path to output training set JSON")
    parser.add_argument('--output_test_path', type=str, required=True,
                       help="Path to output test set JSON")
    parser.add_argument('--test_size', type=int, default=1000,
                       help="Number of test samples (default: 1000)")
    parser.add_argument('--unseen_ratio', type=float, default=0.20,
                       help="Ratio of unseen terms in test set (default: 0.20)")
    parser.add_argument('--seed', type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    print("=== Balanced Test Set Extraction ===")
    print(f"Input: {args.input_path}")
    print(f"Output train: {args.output_train_path}")
    print(f"Output test: {args.output_test_path}")
    print(f"Test size: {args.test_size}")
    print(f"Unseen ratio: {args.unseen_ratio:.1%}")
    print(f"Random seed: {args.seed}")
    print()
    
    # 1. 加载和过滤样本
    all_samples = load_samples(args.input_path)
    valid_samples = filter_valid_samples(all_samples)
    
    # 2. 构建术语到样本的映射
    term_to_samples = build_term_to_samples_map(valid_samples)
    
    # 3. 选择平衡的测试样本
    test_samples, seen_terms, unseen_terms = select_balanced_test_samples(
        term_to_samples, 
        test_size=args.test_size, 
        unseen_ratio=args.unseen_ratio,
        seed=args.seed
    )
    
    # 4. 从训练集中移除测试样本和unseen术语相关样本
    train_samples = remove_test_samples_from_train(all_samples, test_samples, unseen_terms)
    
    # 5. 验证分离效果
    verify_separation(train_samples, test_samples)
    
    # 6. 保存数据集
    save_dataset(train_samples, args.output_train_path, "training set")
    save_dataset(test_samples, args.output_test_path, "test set")
    
    print("\n=== Extraction Complete ===")
    print(f"Training samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")
    print(f"Unseen terms in test: {len(unseen_terms)} ({len(unseen_terms)/len(test_samples):.1%})")
    
    # 保存术语映射信息（用于调试）
    terms_info = {
        'seen_terms': sorted(seen_terms),
        'unseen_terms': sorted(unseen_terms),
        'stats': {
            'total_test_samples': len(test_samples),
            'seen_terms_count': len(seen_terms),
            'unseen_terms_count': len(unseen_terms),
            'unseen_ratio': len(unseen_terms) / len(test_samples) if test_samples else 0
        }
    }
    
    terms_info_path = args.output_test_path.replace('.json', '_terms_info.json')
    with open(terms_info_path, 'w') as f:
        json.dump(terms_info, f, indent=2, ensure_ascii=False)
    print(f"Terms info saved to: {terms_info_path}")

if __name__ == "__main__":
    main()

