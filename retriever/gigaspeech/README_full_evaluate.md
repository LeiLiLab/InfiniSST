# SONAR Full Evaluation Script

这个脚本用于在完整的glossary上评估已训练的SONAR模型，无需重新训练。

## 主要特性

- **专注评估**: 只加载测试数据并评估，不涉及训练逻辑
- **Term-level数据处理**: 使用与`SONAR_term_level_train.py`相同的数据处理逻辑
- **音频验证**: 自动验证音频文件有效性，跳过损坏的文件
- **完整glossary**: 在完整词汇表上构建索引进行评估
- **详细分析**: 提供seen/unseen术语分析和未命中术语详情

## 使用方法

### 直接运行Python脚本

```bash
python3 SONAR_full_evaluate.py \
    --model_path data/clap_sonar_full_n2_best.pt \
    --test_samples_path data/xl_term_level_chunks_merged.json \
    --glossary_path data/terms/glossary_filtered.json \
    --max_eval 1000
```

### 使用SLURM脚本

```bash
# 基本用法
./SONAR_full_evaluate.sh data/clap_sonar_full_n2_best.pt

# 指定测试数据路径
./SONAR_full_evaluate.sh data/clap_sonar_full_n2_best.pt data/custom_test_samples.json
```

## 参数说明

### 必需参数
- `--model_path`: 训练好的模型文件路径 (.pt文件)

### 可选参数
- `--test_samples_path`: 测试样本文件路径 (默认: `data/xl_term_level_chunks_merged.json`)
- `--glossary_path`: 完整词汇表文件路径 (默认: `data/terms/glossary_filtered.json`)
- `--train_ratio`: 训练/测试分割比例 (默认: 0.99，用于seen/unseen分析)
- `--max_eval`: 最大评估样本数 (默认: 1000)
- `--batch_size`: 文本编码batch size (默认: 512，会自动优化)
- `--audio_batch_size`: 音频编码batch size (默认: 1000，会自动优化)

## 数据格式要求

测试样本文件应包含以下字段：
```json
{
    "term_chunk_audio": "path/to/audio/file.wav",
    "term_chunk_text": "corresponding text content",
    "term_chunk_audio_ground_truth_terms": ["term1", "term2", ...]
}
```

## 输出结果

脚本会输出：
1. **Sample-level召回率**: 每个样本的平均召回率
2. **Term-level召回率**: 所有术语的微平均召回率
3. **Seen/Unseen分析**: 训练集见过vs未见过术语的性能对比
4. **未命中术语详情**: 哪些术语检索失败及其上下文
5. **评估结果JSON**: 保存到`{model_path}_full_eval_results.json`

## 性能优化

- **自动batch size调优**: 根据GPU显存自动调整batch size
- **分段处理**: 大词汇表自动分段编码，避免内存溢出
- **多GPU支持**: 自动检测并使用多GPU加速
- **音频验证**: 预先过滤无效音频文件，避免编码错误

## 示例输出

```
[EVAL] Sample-level Average Recall@5: 72.30%
[EVAL] Term-level Micro-Average Recall@5: 69.85% (1456/2085 terms)
[EVAL] Sample-level - Seen Recall@5: 78.20% (892/1000 samples)
[EVAL] Term-level - Seen Recall@5: 74.30% (1234/1661 terms)
[EVAL] Unseen Term Percentage: 20.3%
```

## 故障排除

1. **模型加载失败**: 检查模型路径和设备兼容性
2. **音频文件错误**: 脚本会自动跳过损坏文件并报告
3. **显存不足**: 脚本会自动降低batch size
4. **术语表格式**: 支持多种JSON格式，自动识别term字段 