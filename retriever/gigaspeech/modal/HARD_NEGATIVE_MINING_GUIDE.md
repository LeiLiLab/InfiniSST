# Hard Negative Mining 使用指南

## 概述

这个系统支持多GPU并行挖掘hard negatives，并提供三种挖矿模式：覆盖、跳过和增量更新。

## 关键特性

### 1. 多GPU并行挖矿

- **脚本**: `mine_hard_negatives_multi_gpu.py`
- **加速效果**: 使用N个GPU可以将挖矿时间缩短约N倍
- **显存需求**: 每个GPU约需20-30GB显存

### 2. 三种挖矿模式

在 `run_hard_negative_training.sh` 中配置 `HN_MODE` 变量：

```bash
HN_MODE="update"      # 推荐：增量更新
# HN_MODE="overwrite" # 覆盖模式（会先备份）
# HN_MODE="skip"      # 跳过已存在的文件
```

#### 模式说明

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `update` | 在已有HN基础上增量更新，合并新旧结果 | 迭代训练，累积更多样化的HN |
| `overwrite` | 完全覆盖旧文件（自动备份） | 重新开始，使用新模型重新挖掘 |
| `skip` | 跳过已存在的文件 | 只想继续训练，不想重新挖矿 |

### 3. GPU配置

```bash
# 挖矿配置
MINE_NUM_GPUS=6       # 挖矿时使用的GPU数量
MINE_BATCH_SIZE=128   # 每个GPU的batch size

# 训练配置  
NUM_GPUS=6            # 训练时使用的GPU数量
```

## 使用流程

### 标准流程（使用SLURM）

1. **修改配置参数**：
   ```bash
   vim run_hard_negative_training.sh
   # 修改 HN_MODE、MINE_NUM_GPUS 等参数
   ```

2. **提交作业**：
   ```bash
   sbatch run_hard_negative_training.sh
   ```

3. **查看进度**：
   ```bash
   tail -f hn_train_<job_id>.out
   ```

### 增量更新工作流

典型的迭代训练流程：

1. **第一轮**：首次挖掘和训练
   ```bash
   HN_MODE="update"  # 或 "overwrite"
   # 运行脚本...
   # 得到 model_v1
   ```

2. **第二轮**：使用新模型更新HN
   ```bash
   HN_MODE="update"
   BEST_MODEL="model_v1"
   # 运行脚本...
   # hard negatives 会合并旧的和新的
   # 得到 model_v2
   ```

3. **第N轮**：持续迭代
   ```bash
   HN_MODE="update"
   BEST_MODEL="model_v(N-1)"
   # 每次都在之前的基础上增加新的HN
   ```

## 更新模式的优势

### 为什么使用更新模式？

1. **累积多样性**：
   - 不同训练阶段的模型会找到不同的hard negatives
   - 合并后的HN集合更加多样化
   - 有助于模型学习更robust的特征

2. **避免遗忘**：
   - 保留之前阶段发现的困难样本
   - 防止模型在新训练轮次中遗忘之前的困难区分

3. **自动去重**：
   - 系统会自动去重，避免重复的HN
   - 只保留唯一的术语

4. **安全备份**：
   - 每次更新前自动备份原文件
   - 文件名格式: `hard_negs_train.jsonl.backup_YYYYMMDD_HHMMSS`

## 直接使用Python脚本

如果不想使用完整的训练流程，可以单独运行挖矿脚本：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 首次生成
python mine_hard_negatives_multi_gpu.py \
    --samples_path /path/to/samples.json \
    --mmap_dir /path/to/mmap \
    --faiss_index_pkl /path/to/index.pkl \
    --model_path /path/to/model.pt \
    --out_path /path/to/output.jsonl \
    --num_gpus 6 \
    --batch_size 128

# 增量更新
python mine_hard_negatives_multi_gpu.py \
    --samples_path /path/to/samples.json \
    --mmap_dir /path/to/mmap \
    --faiss_index_pkl /path/to/index.pkl \
    --model_path /path/to/new_model.pt \
    --out_path /path/to/output.jsonl \
    --num_gpus 6 \
    --batch_size 128 \
    --update_existing
```

## 性能调优

### GPU数量选择

- **挖矿**: 可以使用所有可用GPU（如6个）
- **训练**: 根据模型大小和batch size调整

### Batch Size调整

如果遇到OOM错误：
```bash
MINE_BATCH_SIZE=64    # 从128降低到64
# 或者
MINE_BATCH_SIZE=32    # 进一步降低
```

### 显存优化

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 输出文件

### Hard Negatives文件格式

JSONL格式，每行一个样本：
```json
{
  "audio_key": "sample_001",
  "hard_negs": ["term1", "term2", "term3", ...],
  "topk": 200,
  "num_gt": 3
}
```

### 备份文件

自动生成的备份文件：
- `hard_negs_train.jsonl.backup_20250126_143022`
- `hard_negs_test.jsonl.backup_20250126_143022`

## 故障排查

### 问题：GPU OOM

**解决方案**：
1. 减小 `MINE_BATCH_SIZE`
2. 减少 `MINE_NUM_GPUS`
3. 确保设置了环境变量 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

### 问题：SLURM作业卡住

**原因**：可能在等待交互式输入

**解决方案**：
- 确保设置了 `HN_MODE` 变量
- 脚本会自动检测SLURM环境并避免交互

### 问题：文件未更新

**检查**：
1. 确认 `HN_MODE="update"`
2. 查看日志文件中的 "MERGING WITH EXISTING RESULTS" 部分
3. 检查备份文件是否生成

## 最佳实践

1. **第一次训练**：使用 `HN_MODE="update"` 或 `"overwrite"`
2. **迭代训练**：始终使用 `HN_MODE="update"`
3. **重新开始**：使用 `HN_MODE="overwrite"`，检查备份文件
4. **快速测试**：使用 `HN_MODE="skip"`，直接用现有HN训练
5. **定期清理**：删除旧的备份文件以节省空间

## 监控和日志

查看实时日志：
```bash
# 查看输出
tail -f hn_train_<job_id>.out

# 查看错误
tail -f hn_train_<job_id>.err

# 查看GPU使用情况
watch -n 1 nvidia-smi
```

关键日志信息：
- `MERGING WITH EXISTING RESULTS`: 开始合并旧HN
- `Merged X entries, added Y new entries`: 合并统计
- `Average HN per sample`: 每个样本的HN数量







