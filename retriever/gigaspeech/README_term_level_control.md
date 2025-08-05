# SONAR Term-Level Control Group Evaluation

## 🎯 概述

Term-Level Control Group是一个**无需训练**的纯净baseline评估系统，专门用于测试预训练SONAR编码器在精准对齐term-level音频chunks上的检索性能。

## 🔬 核心理念

**问题**: 我们真的需要训练一个专门的term-level模型吗？

**答案**: 不一定！如果term chunks已经通过MFA精准对齐，每个音频片段都完美对应一个术语，那么预训练编码器可能就足够了。

## 📊 与训练方法的对比

| 方面 | Term-Level Training | **Term-Level Control** |
|------|-------------------|----------------------|
| **时间成本** | 数小时训练 | ~30分钟评估 |
| **计算资源** | 需要GPU训练 | 仅需GPU推理 |
| **数据纯度** | 训练过程可能引入噪音 | ✅ 纯净baseline |
| **可解释性** | 训练效果混合因素 | ✅ 直接测试对齐质量 |
| **上界性能** | 未知 | ✅ 提供理论上界 |

## 🛠️ 系统组件

### 1. 核心脚本
```bash
SONAR_term_level_control.py          # 主评估脚本
SONAR_term_level_control_pipeline.sh # 完整流水线
```

### 2. 依赖脚本
```bash
handle_MFA_term_level_chunks.py      # Term-level chunk生成
handle_MFA_term_level_chunks.sh      # 并行处理脚本
```

## 🚀 使用方法

### 快速验证（推荐）
```bash
# 使用单个分片快速测试
cd /mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech
bash SONAR_term_level_control_pipeline.sh true
```

### 完整评估
```bash
# 使用完整数据集
bash SONAR_term_level_control_pipeline.sh
```

### 手动运行
```bash
# 如果数据已准备好，可直接运行评估
python3 SONAR_term_level_control.py \
    --samples_path /path/to/term_level_chunks.json \
    --glossary_path /path/to/glossary_filtered.json \
    --max_eval 2000 \
    --audio_batch_size 32
```

## 📁 文件路径（绝对路径）

```
/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/
├── data/
│   ├── terms/glossary_filtered.json              # 完整词汇表
│   ├── samples/xl/term_preprocessed_samples_*.json # 输入数据
│   ├── xl_term_level_chunks_merged.json          # 合并的term chunks
│   └── term_level_control_results.json           # 输出结果
├── logs/                                          # 日志文件
└── /mnt/gemini/data1/jiaxuanluo/term_chunks/      # 音频文件
```

## 📈 预期结果

### 输出格式
```
[RESULT] Average Recall@1: 45.23%
[RESULT] Average Recall@5: 72.14%
[RESULT] Average Recall@10: 84.67%
[RESULT] Average Recall@20: 91.28%

[RESULT] Seen Recall@10: 89.45% (1789/2000 samples)
[RESULT] Unseen Recall@10: 65.32% (211/2000 samples)
```

### 结果文件
```json
{
  "experiment_type": "term_level_control_group",
  "description": "Direct evaluation using pre-trained SONAR encoders",
  "total_samples": 250000,
  "evaluated_samples": 2000,
  "glossary_terms": 15000,
  "train_terms_coverage_in_glossary": 0.85,
  "results": {
    "recall@1": 0.4523,
    "recall@5": 0.7214,
    "recall@10": 0.8467,
    "recall@20": 0.9128
  }
}
```

## 🔬 科学价值

### 1. **Pure Baseline**
- 无训练干扰，直接测试MFA对齐质量
- 为term-level任务提供理论性能上界

### 2. **快速验证**
- 30分钟内获得结果，快速验证想法
- 无需等待长时间训练

### 3. **对照实验**
- 与训练方法形成对照
- 量化训练的真实收益

### 4. **系统调试**
- 快速定位问题：是对齐问题还是模型问题？
- 验证数据质量

## 🎯 使用场景

### ✅ 适用情况
- **数据质量验证**: 测试MFA对齐是否足够精准
- **快速baseline**: 为新方法提供对比基准  
- **上界估计**: 了解理论最佳性能
- **系统调试**: 快速定位性能瓶颈

### ❌ 不适用情况
- 音频包含大量噪音或多个词汇混合
- 需要处理unseen terms的泛化能力
- MFA对齐质量较差的数据

## 📝 监控和日志

```bash
# 查看任务状态
squeue -u $USER

# 查看主日志
tail -f /mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/logs/sonar_term_level_control_*.log

# 查看评估日志  
tail -f /mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/logs/term_level_control_eval_*.out
```

## 🎉 总结

Term-Level Control Group提供了一个**快速、纯净、可靠**的评估方法：

1. **30分钟 vs 数小时**: 极大提升实验效率
2. **无训练干扰**: 直接测试数据对齐质量  
3. **理论上界**: 为后续方法提供性能目标
4. **完美对照**: 量化训练方法的真实价值

如果控制组的性能已经很好，说明MFA对齐质量高，可能不需要额外训练。如果性能不佳，则说明需要更好的对齐或训练方法。 