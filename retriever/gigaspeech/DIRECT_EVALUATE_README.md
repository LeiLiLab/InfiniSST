# 直接评估功能说明

已为 `SONAR_term_level_train_glossary.py` 添加了直接评估功能，可以跳过训练直接对checkpoint进行评估。

## 功能特点

- 🚀 跳过训练过程，直接加载checkpoint进行评估
- 📊 支持多种top-k召回率评估 (5, 10, 20)
- 🔍 详细的未命中术语分析
- 📈 支持完整词汇表评估
- 🎯 支持seen/unseen术语分析
- 💻 支持GPU选择

## 使用方法

### 1. 基本用法

```bash
python3 SONAR_term_level_train_glossary.py \
    --direct_evaluate \
    --checkpoint_path=data/clap_term_level_epoch1.pt
```

### 2. 完整参数示例

```bash
python3 SONAR_term_level_train_glossary.py \
    --direct_evaluate \
    --checkpoint_path=data/clap_term_level_epoch1.pt \
    --train_samples_path=data/xl_term_level_chunks_merged.json \
    --test_samples_path=data/samples/xl/term_level_chunks_500000_1000000.json \
    --glossary_path=data/terms/glossary_filtered.json \
    --enable_full_eval \
    --filter_no_term \
    --gpu_ids="0"
```

### 3. 使用便捷脚本

```bash
# 简单评估
python3 evaluate_checkpoint.py data/clap_term_level_epoch1.pt

# 带完整词汇表评估
python3 evaluate_checkpoint.py data/clap_term_level_epoch1.pt --enable_full_eval

# 指定GPU
python3 evaluate_checkpoint.py data/clap_term_level_epoch1.pt --gpu_ids="0"
```

## 主要参数

| 参数 | 必需 | 说明 |
|------|------|------|
| `--direct_evaluate` | 是 | 启用直接评估模式 |
| `--checkpoint_path` | 是 | checkpoint文件路径 |
| `--train_samples_path` | 否 | 训练样本路径（用于构建术语集） |
| `--test_samples_path` | 否 | 测试样本路径 |
| `--enable_full_eval` | 否 | 启用完整词汇表评估 |
| `--gpu_ids` | 否 | 指定GPU（如"0,1"或"2"） |

## 输出内容

### 基本评估输出
- Sample-level 召回率 (Recall@5, @10, @20)
- Term-level 微平均召回率
- Seen/Unseen 术语分析
- 详细的未命中术语列表

### 完整词汇表评估输出（如果启用）
- 使用完整glossary的召回率评估
- 更全面的性能分析

## 示例输出

```
=== Evaluation Results for Top-10 ===
[EVAL] Term Samples - Sample-level Average Recall@10: 45.67% (1000 samples)
[EVAL] Term Samples - Term-level Micro-Average Recall@10: 43.21% (1234/2856 terms)
[EVAL] Sample-level - Seen Recall@10: 52.34% (800/1000 samples), Unseen Recall@10: 31.25% (200/1000 samples)
[EVAL] Term-level - Seen Recall@10: 48.90% (1000/2045 terms), Unseen Recall@10: 28.87% (234/811 terms)
```

## 文件说明

- `SONAR_term_level_train_glossary.py`: 主脚本，已添加直接评估功能
- `evaluate_checkpoint.py`: 便捷评估脚本
- `direct_evaluate_example.sh`: 使用示例脚本

## 注意事项

1. checkpoint文件必须存在且可读
2. 需要提供训练数据路径以构建术语词汇表
3. 直接评估模式会跳过所有训练相关的设置
4. 建议使用GPU进行评估以提高速度


