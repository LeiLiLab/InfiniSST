#!/bin/bash

# 扩充MFA samples的ground_truth_terms，添加chunk文本中的日常词汇
# 这将增加训练信号密度，改善模型在大数据集上的表现

#SBATCH --job-name=expand_ground_truth
#SBATCH --partition=taurus
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --output=logs/expand_ground_truth_%j.out
#SBATCH --error=logs/expand_ground_truth_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

# 确保日志目录存在
mkdir -p logs

# 输入和输出文件
INPUT_FILE=${1:-"data/xl_mfa_2chunks_samples_merged.json"}
OUTPUT_FILE=${2:-"data/xl_mfa_2chunks_samples_expanded.json"}

echo "[INFO] Starting ground truth expansion..."
echo "[INFO] Input file: $INPUT_FILE"
echo "[INFO] Output file: $OUTPUT_FILE"

# 运行Python脚本进行扩充
python3 -c "
import json
import re
import nltk
from tqdm import tqdm
from collections import Counter
import argparse

# 确保下载停用词
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def should_include_word(word):
    '''判断是否应该包含某个词作为目标术语'''
    word = word.lower().strip()
    
    # 基本过滤条件
    if len(word) < 2:
        return False
    
    # 过滤停用词
    if word in stop_words:
        return False
    
    # 过滤纯数字
    if re.match(r'^[\d,]+$', word):
        return False
    
    # 过滤常见数字词
    number_words = {
        'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
        'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty',
        'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety', 'hundred', 'thousand', 'million', 'billion'
    }
    if word in number_words:
        return False
    
    # 过滤时间词
    time_words = {
        'today', 'yesterday', 'tomorrow', 'morning', 'afternoon', 'evening', 'night',
        'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
        'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december',
        'week', 'month', 'year', 'day', 'hour', 'minute', 'second', 'time'
    }
    if word in time_words:
        return False
    
    # 过滤常见代词
    pronouns = {
        'i', 'me', 'my', 'mine', 'myself',
        'you', 'your', 'yours', 'yourself',
        'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
        'it', 'its', 'itself', 'we', 'us', 'our', 'ours', 'ourselves',
        'they', 'them', 'their', 'theirs', 'themselves',
        'this', 'that', 'these', 'those', 'what', 'which', 'who', 'whom', 'whose',
        'something', 'someone', 'somewhere', 'anything', 'anyone', 'anywhere',
        'everything', 'everyone', 'everywhere', 'nothing', 'nobody', 'nowhere'
    }
    if word in pronouns:
        return False
    
    # 只保留包含字母的词（过滤纯符号）
    if not re.search(r'[a-zA-Z]', word):
        return False
    
    # 过滤数字占比过高的词
    if sum(c.isdigit() for c in word) > len(word) // 2:
        return False
    
    return True

def extract_meaningful_words(text):
    '''从文本中提取有意义的词汇'''
    # 基本清理和分词
    text = re.sub(r'[<>]', ' ', text)  # 移除尖括号
    words = re.findall(r\"\\b[a-zA-Z][a-zA-Z0-9']*\\b\", text.lower())
    
    # 过滤并去重
    meaningful_words = []
    seen = set()
    for word in words:
        word = word.strip(\"'\")  # 移除首尾单引号
        if should_include_word(word) and word not in seen:
            meaningful_words.append(word)
            seen.add(word)
    
    return meaningful_words

def expand_ground_truth_terms(samples, expansion_strategy='moderate'):
    '''扩充ground truth terms'''
    expanded_samples = []
    
    # 统计信息
    original_term_counts = []
    expanded_term_counts = []
    
    for sample in tqdm(samples, desc='Expanding ground truth terms'):
        # 复制原始样本
        expanded_sample = sample.copy()
        
        # 获取原始术语
        original_terms = set(t.lower() for t in sample.get('n_chunk_audio_ground_truth_terms', []))
        original_term_counts.append(len(original_terms))
        
        # 从chunk文本中提取词汇
        chunk_text = sample.get('n_chunk_text', '')
        meaningful_words = extract_meaningful_words(chunk_text)
        
        # 根据策略扩充
        if expansion_strategy == 'conservative':
            # 保守策略：只添加名词和动词（需要POS标注）
            additional_terms = meaningful_words[:3]  # 最多添加3个
        elif expansion_strategy == 'moderate':
            # 适中策略：添加所有有意义的词，但限制数量
            additional_terms = meaningful_words[:6]  # 最多添加6个
        elif expansion_strategy == 'aggressive':
            # 激进策略：添加所有有意义的词
            additional_terms = meaningful_words
        else:
            additional_terms = meaningful_words[:4]  # 默认策略
        
        # 合并原始术语和新增术语
        all_terms = list(original_terms) + [t for t in additional_terms if t not in original_terms]
        expanded_sample['n_chunk_audio_ground_truth_terms'] = all_terms
        
        # 添加扩充标记
        expanded_sample['ground_truth_expanded'] = True
        expanded_sample['original_term_count'] = len(original_terms)
        expanded_sample['expanded_term_count'] = len(all_terms)
        
        expanded_term_counts.append(len(all_terms))
        expanded_samples.append(expanded_sample)
    
    # 打印统计信息
    print(f'[INFO] 原始平均术语数: {np.mean(original_term_counts):.2f}')
    print(f'[INFO] 扩充后平均术语数: {np.mean(expanded_term_counts):.2f}')
    print(f'[INFO] 术语数增长: {np.mean(expanded_term_counts) / np.mean(original_term_counts):.2f}x')
    print(f'[INFO] 扩充后术语数分布: {np.percentile(expanded_term_counts, [25, 50, 75, 90, 95])}')
    
    return expanded_samples

# 主处理流程
def main():
    import sys
    input_file = '$INPUT_FILE'
    output_file = '$OUTPUT_FILE'
    
    print(f'[INFO] Loading samples from {input_file}...')
    with open(input_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    print(f'[INFO] Loaded {len(samples)} samples')
    
    # 扩充ground truth terms
    expanded_samples = expand_ground_truth_terms(samples, expansion_strategy='moderate')
    
    # 保存结果
    print(f'[INFO] Saving expanded samples to {output_file}...')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(expanded_samples, f, ensure_ascii=False, indent=2)
    
    print(f'[INFO] Successfully saved {len(expanded_samples)} expanded samples')
    
    # 验证几个样本
    print('\n[INFO] Sample comparison:')
    for i in range(min(3, len(samples))):
        original = samples[i]
        expanded = expanded_samples[i]
        print(f'Sample {i+1}:')
        print(f'  Chunk text: {original.get(\"n_chunk_text\", \"\")[:100]}...')
        print(f'  Original terms: {original.get(\"n_chunk_audio_ground_truth_terms\", [])}')
        print(f'  Expanded terms: {expanded.get(\"n_chunk_audio_ground_truth_terms\", [])}')
        print()

if __name__ == '__main__':
    import numpy as np
    main()
"

echo "[INFO] Ground truth expansion completed!"
echo "[INFO] Output saved to: $OUTPUT_FILE"

# 提供使用示例
echo ""
echo "=== Usage Examples ==="
echo "# 使用扩充后的数据训练："
echo "python3 SONAR_train.py --train_samples_path=$OUTPUT_FILE --batch_size=256 --save_path=data/clap_sonar_expanded.pt"
echo ""
echo "# 比较扩充前后的效果："
echo "python3 SONAR_train.py --train_samples_path=$INPUT_FILE --save_path=data/clap_sonar_original.pt"
echo "python3 SONAR_train.py --train_samples_path=$OUTPUT_FILE --save_path=data/clap_sonar_expanded.pt" 