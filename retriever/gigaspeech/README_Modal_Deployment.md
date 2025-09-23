# Qwen2-Audio Term-Level DDP训练 - Modal部署指南

本文档介绍如何使用Modal云平台部署和运行Qwen2-Audio的分布式训练。

## 文件结构

```
retriever/gigaspeech/
├── Qwen2_Audio_term_level_train_ddp.py      # DDP训练脚本（本地版）
├── qwen2_audio_train_ddp_fixed.sh           # 本地DDP训练启动脚本
├── modal_complete_training.py               # Modal完整部署脚本
├── deploy_modal_training.py                 # Modal简化部署脚本
└── README_Modal_Deployment.md               # 本文档
```

## 主要改动

### 1. DDP转换
- 将原有的`Qwen2_Audio_term_level_train.py`改造为DDP版本
- 支持多GPU分布式训练
- 保持与SONAR版本相同的DDP架构

### 2. Modal部署
- 创建完整的Modal部署脚本
- 自动处理数据上传和模型训练
- 支持8个A100 GPU的大规模训练

## 使用方法

### 方式1：本地DDP训练

```bash
# 给脚本添加执行权限
chmod +x qwen2_audio_train_ddp_fixed.sh

# 运行DDP训练
./qwen2_audio_train_ddp_fixed.sh
```

### 方式2：Modal云端训练

#### 前置条件

1. 安装Modal SDK：
```bash
pip install modal
```

2. 配置Modal认证：
```bash
modal token new
```

3. 创建必要的Secrets：
```bash
# HuggingFace Token（用于下载模型）
modal secret create huggingface-token HUGGING_FACE_HUB_TOKEN=xxxx

# 可选：W&B Token（用于实验跟踪）
modal secret create wandb-token WANDB_API_KEY=your_wandb_key
```

#### 运行训练

```bash
# 使用完整版Modal脚本
modal run modal_complete_training.py

# 或使用简化版
modal run deploy_modal_training.py
```

## 配置参数

### 训练参数
- **epochs**: 训练轮数（默认40，但通常会提前停止）
- **batch_size**: 总批次大小（默认128，会自动分配到8个GPU）
- **lr**: 学习率（默认1e-4）
- **patience**: 早停耐心值（默认3，即3个epoch无改善就停止）
- **model_name**: 模型名称（默认"Qwen/Qwen2-Audio-7B-Instruct"）

### LoRA参数
- **lora_r**: LoRA秩（默认16）
- **lora_alpha**: LoRA缩放参数（默认32）
- **lora_dropout**: LoRA dropout率（默认0.1）

### 损失权重
- **audio_text_loss_ratio**: 音频-文本对比损失权重（默认0.3）
- **audio_term_loss_ratio**: 音频-术语对比损失权重（默认0.7）

### Hard Negative Mining
- **enable_hard_neg**: 启用困难负样本挖掘（默认True）
- **hard_neg_k**: 每个样本的困难负样本数量（默认10）
- **hard_neg_weight**: 困难负样本损失权重（默认0.2）

## 数据要求

训练需要以下数据文件：

1. **训练数据**: `data/xl_term_level_chunks_merged.json`
   - 包含term-level音频chunk和对应的ground truth terms

2. **测试数据**: `data/samples/xl/term_level_chunks_500000_1000000.json`
   - 独立的测试数据集

3. **词汇表**: `data/terms/glossary_filtered.json`（可选）
   - 用于困难负样本挖掘

## Modal资源配置

### GPU配置
- **GPU类型**: A100 40GB
- **GPU数量**: 8个
- **总显存**: 320GB

### 计算资源
- **CPU核心**: 64个
- **内存**: 256GB
- **超时时间**: 24小时

### 存储
- **持久化存储**: Modal Volume
- **缓存目录**: `/data/hf_cache`（用于HuggingFace模型缓存）

## 监控和日志

### 本地训练
- 日志文件：`qwen2_audio_train_ddp_fixed_YYYYMMDD_HHMMSS.log`
- GPU监控：脚本自动显示GPU使用情况

### Modal训练
- 实时日志：Modal Dashboard
- 模型保存：自动保存到Modal Volume
- 进度追踪：支持W&B集成

### 评估和早停机制
- **自动评估**: 每个epoch结束后自动评估Recall@10
- **早停**: 当Recall@10连续`patience`个epoch无改善时自动停止
- **学习率调度**: 使用ReduceLROnPlateau，当性能停滞时自动降低学习率
- **模型保存**: 
  - 每5个epoch保存检查点
  - 自动保存性能最佳的模型为`model_best.pt`
  - 训练结束保存最终模型为`model_final.pt`
- **数值稳定性**: 自动检测和跳过NaN/Inf损失的batch

## 故障排除

### 常见问题

1. **CUDA环境问题**
   - 确保CUDA_HOME正确设置
   - 检查GPU驱动版本兼容性

2. **内存不足**
   - 减少batch_size
   - 启用gradient checkpointing

3. **数据文件缺失**
   - 检查数据文件路径
   - 确保JSON格式正确

4. **Modal认证失败**
   - 重新运行`modal token new`
   - 检查Secret配置

### 调试建议

1. **本地测试**：先在本地小规模测试，确保代码无误
2. **数据验证**：检查数据文件完整性和格式
3. **资源监控**：关注GPU/内存使用情况
4. **日志分析**：仔细查看错误日志定位问题

## 性能优化

### 训练加速
- 使用混合精度训练（AMP）
- 启用Flash Attention
- 优化数据加载（多进程+预取）

### 内存优化
- LoRA微调减少显存占用
- Gradient checkpointing
- 合理设置batch size

## 成本估算

### Modal费用（估算）
- A100 GPU: ~$1.10/小时/GPU
- 8个GPU训练20个epoch: ~$176
- 存储费用：根据数据量计算

### 建议
- 使用spot实例降低成本
- 合理设置超时时间
- 及时清理不需要的数据

## 后续开发

### 计划功能
1. 自动超参数调优
2. 分布式评估
3. 模型版本管理
4. A/B测试支持

### 扩展性
- 支持更多GPU类型
- 多云平台部署
- 自动扩缩容

## 联系方式

如有问题，请联系开发团队或提交Issue。
