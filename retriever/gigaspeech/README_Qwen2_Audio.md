# Qwen2-Audio Term-Level Training System

本系统将原有的SONAR模型替换为Qwen2-Audio-Instruction模型，用于术语级别的音频-文本检索训练。

## 🚀 主要特性

- **先进模型**: 使用Qwen2-Audio-Instruction替代SONAR，提供更强的多模态理解能力
- **术语级对齐**: 基于MFA精确对齐的术语级音频块进行训练
- **灵活配置**: 支持多种训练参数和GPU配置
- **拒答能力**: 内置无术语样本的拒答机制
- **自动化流水线**: 从数据预处理到模型训练的完整自动化

## 📋 系统要求

### 硬件要求
- **GPU**: 建议使用至少24GB显存的GPU（如RTX 4090, A100等）
- **内存**: 建议64GB以上系统内存
- **存储**: 建议500GB以上可用存储空间

### 软件依赖
```bash
# 核心依赖
pip install transformers>=4.30.0
pip install librosa>=0.10.0
pip install soundfile>=0.12.0
pip install datasets>=2.10.0

# 现有依赖
pip install torch torchvision torchaudio
pip install numpy faiss-gpu tqdm
```

## 🛠️ 安装和设置

### 1. 环境准备
```bash
# 激活conda环境
conda activate infinisst

# 安装新依赖
pip install transformers librosa datasets soundfile
```

### 2. 数据准备
确保以下数据文件存在：
```
data/
├── xl_term_level_chunks_merged.json          # 主训练数据
├── samples/xl/
│   ├── term_level_chunks_*.json              # 分片数据
│   └── term_level_chunks_500000_1000000.json # 测试数据
└── terms/
    ├── glossary_filtered.json                # 词汇表
    └── alt2main.json                         # 术语映射
```

### 3. 集成测试
运行测试脚本验证系统设置：
```bash
python test_qwen2_audio.py
```

## 🎯 使用方法

### 快速开始
```bash
# 单分片快速验证（推荐首次使用）
bash Qwen2_Audio_term_level_pipeline.sh term true

# 完整数据集训练
bash Qwen2_Audio_term_level_pipeline.sh term false
```

### 详细参数说明
```bash
bash Qwen2_Audio_term_level_pipeline.sh [参数1] [参数2] ... [参数11]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| $1 text_field | term | 文本字段类型 |
| $2 single_slice | false | 是否使用单分片模式 |
| $3 audio_text_loss_ratio | 0.3 | 音频-文本损失权重 |
| $4 audio_term_loss_ratio | 0.7 | 音频-术语损失权重 |
| $5 enable_full_eval | false | 是否启用完整评估 |
| $6 test_samples_path | data/samples/xl/term_level_chunks_500000_1000000.json | 测试数据路径 |
| $7 best_model_path | data/qwen2_audio_term_level_best.pt | 最佳模型路径 |
| $8 gpu_ids | "" | GPU编号（空表示使用所有GPU） |
| $9 model_name | Qwen/Qwen2-Audio-7B-Instruct | 模型名称 |
| $10 enable_no_term | true | 是否启用no-term样本处理 |
| $11 enable_hard_neg | false | 是否启用hard negative mining |

### 常用配置示例

#### 1. 开发调试模式
```bash
# 使用单分片，快速验证
bash Qwen2_Audio_term_level_pipeline.sh term true 0.3 0.7 false
```

#### 2. 生产训练模式
```bash
# 完整数据集，启用评估
bash Qwen2_Audio_term_level_pipeline.sh term false 0.3 0.7 true
```

#### 3. 多GPU训练
```bash
# 使用GPU 0和1
bash Qwen2_Audio_term_level_pipeline.sh term false 0.3 0.7 true \
    "data/samples/xl/term_level_chunks_500000_1000000.json" \
    "data/qwen2_audio_term_level_best.pt" \
    "0,1"
```

#### 4. 仅术语训练模式
```bash
# 禁用no-term样本，专注于术语检索
bash Qwen2_Audio_term_level_pipeline.sh term false 0.3 0.7 false \
    "data/samples/xl/term_level_chunks_500000_1000000.json" \
    "data/qwen2_audio_term_level_best.pt" \
    "0,1" \
    "Qwen/Qwen2-Audio-7B-Instruct" \
    false \
    false
```

#### 5. Hard Negative Mining训练
```bash
# 启用hard negative mining增强对比学习
bash Qwen2_Audio_term_level_pipeline.sh term false 0.3 0.7 false \
    "data/samples/xl/term_level_chunks_500000_1000000.json" \
    "data/qwen2_audio_term_level_best.pt" \
    "0" \
    "Qwen/Qwen2-Audio-7B-Instruct" \
    true \
    true
```

#### 6. 自定义模型
```bash
# 使用本地或自定义的Qwen2-Audio模型
bash Qwen2_Audio_term_level_pipeline.sh term false 0.3 0.7 true \
    "data/samples/xl/term_level_chunks_500000_1000000.json" \
    "data/qwen2_audio_term_level_best.pt" \
    "0" \
    "/path/to/your/qwen2-audio-model" \
    true \
    false
```

## 📊 模型架构对比

| 特性 | SONAR | Qwen2-Audio |
|------|-------|-------------|
| 音频编码器 | SONAR Speech Encoder | Qwen2-Audio Tower |
| 文本编码器 | SONAR Text Encoder | Qwen2 Language Model |
| 隐藏维度 | 1024 | 4096 |
| 投影维度 | 512 | 512 |
| 多语言支持 | ✅ | ✅ |
| 指令跟随 | ❌ | ✅ |
| 上下文理解 | 基础 | 强化 |

## 🔧 训练参数调优

### 内存优化
```bash
# 减小批次大小
--batch_size=32

# 冻结更多层
--unfreeze_layers=0

# 禁用no-term样本（减少数据量）
--disable_no_term

# 使用梯度检查点
--gradient_checkpointing
```

### 性能优化
```bash
# 调整损失权重
--audio_text_loss_ratio=0.1
--audio_term_loss_ratio=0.9

# 启用hard negative mining
--enable_hard_neg --hard_neg_source=glossary

# 启用混合精度训练
--fp16
```

## 🎛️ 配置选项详解

### No-term样本处理
- **启用** (`enable_no_term=true`): 包含无术语的音频样本，训练模型的拒答能力
- **禁用** (`enable_no_term=false`): 仅使用有术语的样本，专注于术语检索性能

```bash
# 启用no-term处理（默认）
bash Qwen2_Audio_term_level_pipeline.sh term false 0.3 0.7 false \
    test_samples.json model.pt "" "Qwen/Qwen2-Audio-7B-Instruct" true

# 禁用no-term处理
bash Qwen2_Audio_term_level_pipeline.sh term false 0.3 0.7 false \
    test_samples.json model.pt "" "Qwen/Qwen2-Audio-7B-Instruct" false
```

### Hard Negative Mining
- **启用** (`enable_hard_neg=true`): 使用困难负样本增强对比学习
- **禁用** (`enable_hard_neg=false`): 使用标准对比学习

```bash
# 启用hard negative mining
bash Qwen2_Audio_term_level_pipeline.sh term false 0.3 0.7 false \
    test_samples.json model.pt "" "Qwen/Qwen2-Audio-7B-Instruct" true true

# 禁用hard negative mining（默认）
bash Qwen2_Audio_term_level_pipeline.sh term false 0.3 0.7 false \
    test_samples.json model.pt "" "Qwen/Qwen2-Audio-7B-Instruct" true false
```

### 配置组合建议

| 训练目标 | No-term | Hard-neg | 说明 |
|----------|---------|----------|------|
| 快速验证 | false | false | 最快训练，仅术语检索 |
| 标准训练 | true | false | 平衡性能和速度 |
| 高性能训练 | true | true | 最佳性能，训练时间较长 |
| 拒答专项 | true | false | 专注于拒答能力训练 |

## 📈 监控和评估

### 训练监控
```bash
# 查看作业状态
squeue -u $USER

# 监控日志
tail -f logs/qwen2_audio_term_level_pipeline_*.log
```

### 评估指标
- **Recall@K**: 术语检索召回率
- **Rejection Rate**: 无术语样本拒答率
- **Loss Components**: 各损失组件的值

## 🐛 故障排除

### 常见问题

#### 1. CUDA内存不足
```
RuntimeError: CUDA out of memory
```
**解决方案**:
- 减小batch_size（推荐32或更小）
- 使用更少的GPU
- 增加unfreeze_layers=0

#### 2. 模型加载失败
```
OSError: Can't load tokenizer for 'Qwen/Qwen2-Audio-7B-Instruct'
```
**解决方案**:
- 检查网络连接
- 使用本地模型路径
- 更新transformers版本

#### 3. 音频文件无效
```
[WARN] Invalid audio: Failed to read
```
**解决方案**:
- 检查音频文件路径
- 验证音频格式（建议16kHz WAV）
- 运行音频验证脚本

### 调试模式
```bash
# 启用详细日志
export TRANSFORMERS_VERBOSITY=debug

# 运行测试脚本
python test_qwen2_audio.py
```

## 📁 输出文件

训练完成后，系统会生成以下文件：
```
data/
├── qwen2_audio_term_level_single.pt        # 单分片模型
├── qwen2_audio_term_level_full.pt          # 完整模型
├── qwen2_audio_term_level_best.pt          # 最佳模型
└── qwen2_audio_term_level_epoch*.pt        # 检查点

logs/
├── qwen2_audio_term_level_pipeline_*.log   # 流水线日志
├── qwen2_train_term_level_*.out            # 训练输出
└── qwen2_train_term_level_*.err            # 训练错误
```

## 🔄 从SONAR迁移

如果您之前使用SONAR系统，可以按以下步骤迁移：

### 1. 保留现有数据
```bash
# 现有的term-level数据可以直接使用
# 无需重新生成
```

### 2. 更新训练脚本
```bash
# 旧命令
bash SONAR_term_level_pipeline_glossary.sh term false

# 新命令  
bash Qwen2_Audio_term_level_pipeline.sh term false
```

### 3. 模型兼容性
- 模型权重不兼容，需要重新训练
- 数据格式完全兼容
- 评估指标保持一致

## 📚 参考资料

- [Qwen2-Audio官方文档](https://qwenlm.github.io/blog/qwen2-audio/)
- [Transformers库文档](https://huggingface.co/docs/transformers/)
- [原SONAR系统文档](README_term_level_control.md)

## 🤝 贡献和支持

如有问题或建议，请：
1. 查看故障排除部分
2. 运行测试脚本诊断
3. 检查日志文件
4. 提交详细的错误报告

---

**注意**: Qwen2-Audio模型较大，首次运行时会自动下载模型文件（约13GB），请确保网络连接稳定。
