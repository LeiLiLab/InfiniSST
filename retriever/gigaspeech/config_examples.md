# Qwen2-Audio Term-Level Training Configuration Examples

本文档提供了各种训练场景的配置示例。

## 🚀 快速开始配置

### 基础验证配置
```bash
# 最简单的配置，用于快速验证系统是否正常工作
bash Qwen2_Audio_term_level_pipeline.sh term true
```
**特点**: 单分片模式，快速验证，使用默认参数

### 标准训练配置
```bash
# 适合大多数场景的标准配置
bash Qwen2_Audio_term_level_pipeline.sh term false 0.3 0.7 false
```
**特点**: 完整数据集，平衡的损失权重，包含no-term样本

## 🎯 专项训练配置

### 仅术语检索训练
```bash
# 专注于术语检索性能，忽略拒答能力
bash Qwen2_Audio_term_level_pipeline.sh \
    term \                              # 使用term字段
    false \                             # 完整数据集
    0.1 \                               # 低音频-文本损失权重
    0.9 \                               # 高音频-术语损失权重
    false \                             # 不启用完整评估
    "data/samples/xl/term_level_chunks_500000_1000000.json" \
    "data/qwen2_audio_term_level_best.pt" \
    "0,1" \                             # 使用GPU 0和1
    "Qwen/Qwen2-Audio-7B-Instruct" \
    false \                             # 禁用no-term样本
    false                               # 禁用hard negative mining
```
**适用场景**: 
- 只关心术语检索准确率
- 训练资源有限
- 快速迭代验证

### 拒答能力专项训练
```bash
# 专注于训练模型的拒答能力
bash Qwen2_Audio_term_level_pipeline.sh \
    term \
    false \
    0.3 \
    0.7 \
    false \
    "data/samples/xl/term_level_chunks_500000_1000000.json" \
    "data/qwen2_audio_term_level_best.pt" \
    "0" \
    "Qwen/Qwen2-Audio-7B-Instruct" \
    true \                              # 启用no-term样本
    false                               # 不使用hard negative mining
```
**适用场景**:
- 需要模型能够拒绝回答不包含术语的查询
- 部署在开放域环境
- 对假阳性敏感的应用

### 高性能对比学习训练
```bash
# 使用hard negative mining增强对比学习效果
bash Qwen2_Audio_term_level_pipeline.sh \
    term \
    false \
    0.2 \                               # 更低的音频-文本权重
    0.8 \                               # 更高的音频-术语权重
    true \                              # 启用完整评估
    "data/samples/xl/term_level_chunks_500000_1000000.json" \
    "data/qwen2_audio_term_level_best.pt" \
    "0,1,2,3" \                         # 使用多GPU
    "Qwen/Qwen2-Audio-7B-Instruct" \
    true \                              # 启用no-term样本
    true                                # 启用hard negative mining
```
**适用场景**:
- 追求最佳性能
- 有充足的计算资源
- 有FAISS索引文件支持

## 🔬 实验配置

### 损失权重对比实验
```bash
# 配置A: 平衡权重
bash Qwen2_Audio_term_level_pipeline.sh term false 0.5 0.5 false \
    test_data.json model_balanced.pt "0" model true false

# 配置B: 偏向术语检索
bash Qwen2_Audio_term_level_pipeline.sh term false 0.1 0.9 false \
    test_data.json model_term_focused.pt "1" model true false

# 配置C: 偏向文本对齐
bash Qwen2_Audio_term_level_pipeline.sh term false 0.8 0.2 false \
    test_data.json model_text_focused.pt "2" model true false
```

### 数据配置对比实验
```bash
# 实验1: 仅术语样本
bash Qwen2_Audio_term_level_pipeline.sh term false 0.3 0.7 false \
    test_data.json model_term_only.pt "0" model false false

# 实验2: 包含no-term样本
bash Qwen2_Audio_term_level_pipeline.sh term false 0.3 0.7 false \
    test_data.json model_with_noterm.pt "1" model true false

# 实验3: 包含hard negative mining
bash Qwen2_Audio_term_level_pipeline.sh term false 0.3 0.7 false \
    test_data.json model_with_hardneg.pt "2" model true true
```

## 🔧 资源受限配置

### 低内存配置
```bash
# 适用于GPU内存较小的情况（<16GB）
bash Qwen2_Audio_term_level_pipeline.sh \
    term \
    true \                              # 使用单分片减少数据量
    0.3 \
    0.7 \
    false \
    "data/samples/xl/term_level_chunks_0_500000.json" \
    "data/qwen2_audio_term_level_best.pt" \
    "0" \                               # 单GPU
    "Qwen/Qwen2-Audio-7B-Instruct" \
    false \                             # 禁用no-term减少内存
    false                               # 禁用hard negative mining
```

### 单GPU配置
```bash
# 适用于只有一个GPU的情况
bash Qwen2_Audio_term_level_pipeline.sh \
    term false 0.3 0.7 false \
    "data/samples/xl/term_level_chunks_500000_1000000.json" \
    "data/qwen2_audio_term_level_best.pt" \
    "0" \                               # 指定单个GPU
    "Qwen/Qwen2-Audio-7B-Instruct" \
    true false
```

## 🎛️ 自定义配置

### 使用自定义模型
```bash
# 使用本地或自定义的Qwen2-Audio模型
bash Qwen2_Audio_term_level_pipeline.sh \
    term false 0.3 0.7 false \
    "data/samples/xl/term_level_chunks_500000_1000000.json" \
    "data/qwen2_audio_term_level_best.pt" \
    "0,1" \
    "/path/to/your/custom/qwen2-audio-model" \
    true false
```

### 使用自定义测试数据
```bash
# 使用自定义的测试数据集
bash Qwen2_Audio_term_level_pipeline.sh \
    term false 0.3 0.7 true \
    "/path/to/your/custom_test_samples.json" \
    "data/qwen2_audio_term_level_best.pt" \
    "0" \
    "Qwen/Qwen2-Audio-7B-Instruct" \
    true false
```

## 📊 性能基准配置

### 基准测试配置
```bash
# 用于性能基准测试的标准配置
bash Qwen2_Audio_term_level_pipeline.sh \
    term \
    false \
    0.3 \
    0.7 \
    true \                              # 启用完整评估
    "data/samples/xl/term_level_chunks_500000_1000000.json" \
    "data/qwen2_audio_term_level_best.pt" \
    "0,1" \
    "Qwen/Qwen2-Audio-7B-Instruct" \
    true \
    false
```

## 💡 配置选择指南

### 根据目标选择配置

| 训练目标 | 推荐配置 | 说明 |
|----------|----------|------|
| 快速验证系统 | `term true` | 最小配置，快速验证 |
| 开发调试 | `term false 0.3 0.7 false ... true false` | 标准配置，平衡性能 |
| 生产部署 | `term false 0.2 0.8 true ... true true` | 高性能配置 |
| 术语检索专用 | `term false 0.1 0.9 false ... false false` | 专注术语检索 |
| 拒答能力训练 | `term false 0.3 0.7 false ... true false` | 包含no-term样本 |

### 根据资源选择配置

| 资源情况 | GPU内存 | 推荐配置 |
|----------|---------|----------|
| 资源充足 | >24GB | 完整配置 + hard negative mining |
| 中等资源 | 16-24GB | 标准配置，禁用hard negative mining |
| 资源受限 | <16GB | 单分片 + 禁用no-term |

### 根据数据选择配置

| 数据情况 | 推荐配置 |
|----------|----------|
| 有完整FAISS索引 | 启用hard negative mining |
| 只有基础数据 | 使用标准配置 |
| 数据量较小 | 使用单分片模式 |
| 需要拒答能力 | 启用no-term样本 |

## 🔍 调试配置

### 调试模式配置
```bash
# 用于调试的详细日志配置
export TRANSFORMERS_VERBOSITY=debug
bash Qwen2_Audio_term_level_pipeline.sh term true 0.3 0.7 false \
    "data/samples/xl/term_level_chunks_0_500000.json" \
    "data/qwen2_audio_term_level_best.pt" \
    "0" \
    "Qwen/Qwen2-Audio-7B-Instruct" \
    true false
```

### 测试集成配置
```bash
# 运行集成测试
python test_qwen2_audio.py

# 使用快速启动脚本的测试选项
./quick_start_qwen2_audio.sh
# 选择选项 6 (Run integration test only)
```
