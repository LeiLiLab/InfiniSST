# 多GPU并行处理指南

## 🚀 概述

使用SLURM job array在4张GPU上并行处理数据集构造任务，可以将处理时间缩短到原来的1/4。

## 📊 工作原理

### 数据分片策略

使用**轮询分片（Round-robin Sharding）**：

```python
# GPU 0 处理: line 0, 4, 8, 12, ...
# GPU 1 处理: line 1, 5, 9, 13, ...
# GPU 2 处理: line 2, 6, 10, 14, ...
# GPU 3 处理: line 3, 7, 11, 15, ...

if line_idx % total_gpus == gpu_id:
    process_this_line()
```

### SLURM Job Array

```bash
#SBATCH --array=0-3  # 创建4个任务，ID为0,1,2,3
```

每个任务：
- 使用不同的GPU（通过SLURM_ARRAY_TASK_ID）
- 处理数据的不同部分
- 输出到独立的文件

## 🎯 使用方法

### 1. 提交多GPU任务

```bash
# 完整处理（4张GPU并行）
sbatch retriever/gigaspeech/run_term_map_construction.sh

# 测试模式（每个GPU处理10条）
sbatch retriever/gigaspeech/run_term_map_construction.sh --dry-run
```

### 2. 监控任务状态

```bash
# 查看任务队列
squeue -u $USER

# 查看特定job的所有array任务
squeue -j <JOB_ID>

# 查看日志（实时）
tail -f /mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/logs/<JOB_ID>_*_term_map_construction.out
```

### 3. 检查输出文件

每个GPU会生成独立的输出文件：

```bash
ls -lh /mnt/gemini/data1/jiaxuanluo/train_s_zh_with_candidates_gpu*.jsonl
```

应该看到：
```
train_s_zh_with_candidates_gpu0.jsonl
train_s_zh_with_candidates_gpu1.jsonl
train_s_zh_with_candidates_gpu2.jsonl
train_s_zh_with_candidates_gpu3.jsonl
```

### 4. 合并输出

#### 方法1：自动合并（推荐）

GPU 3完成后会自动合并所有输出（脚本内置）。

#### 方法2：手动合并

```bash
./retriever/gigaspeech/merge_multi_gpu_outputs.sh
```

## 📈 性能对比

| 配置 | 处理时间（12K条） | 加速比 |
|------|-----------------|--------|
| 单GPU | ~2小时 | 1x |
| 4 GPU | ~30分钟 | 4x |

**注意**：实际加速比可能略小于4x，因为：
- TSV/Glossary加载时间（每个GPU都要加载）
- 模型加载时间（每个GPU都要加载）
- I/O开销

## 🔧 配置说明

### SLURM参数

```bash
#SBATCH --array=0-3        # 4个GPU任务
#SBATCH --gres=gpu:1       # 每个任务1张GPU
#SBATCH --cpus-per-task=4  # 每个任务4个CPU
#SBATCH --mem=128GB        # 每个任务128GB内存
```

### Python参数

```python
--gpu-id 0           # GPU ID（由SLURM自动设置）
--total-gpus 4       # 总GPU数
--dry-run            # 测试模式
--max-messages 100   # 限制消息数
```

## 🐛 故障排除

### 问题1：某个GPU任务失败

**症状**：只有3个输出文件

**解决**：
1. 检查失败GPU的日志
2. 单独重新提交失败的GPU

```bash
# 只重新运行GPU 2
sbatch --array=2 retriever/gigaspeech/run_term_map_construction.sh
```

### 问题2：输出文件顺序混乱

**不影响**：合并时会按GPU顺序（0,1,2,3）拼接，数据顺序与单GPU处理略有不同，但不影响训练。

### 问题3：GPU OOM

**解决**：
1. 减小`RAG_BATCH_SIZE`
2. 增加内存申请

```bash
#SBATCH --mem=256GB  # 增加到256GB
```

### 问题4：某个GPU一直卡住

**症状**：其他GPU都完成了，只有一个还在运行

**原因**：可能这个GPU分到的数据恰好包含很多长音频

**解决**：
- 等待完成（轮询分片通常能保证负载均衡）
- 或者检查是否真的卡住了（看日志）

## 📁 文件结构

```
retriever/gigaspeech/
├── handle_train_dataset_for_term_map_v2_buzz.py  # 主脚本（支持多GPU）
├── run_term_map_construction.sh                  # SLURM脚本（job array）
├── merge_multi_gpu_outputs.sh                    # 合并脚本
├── MULTI_GPU_USAGE.md                            # 本文档
└── modal/logs/
    ├── <JOB_ID>_0_term_map_construction.out      # GPU 0日志
    ├── <JOB_ID>_1_term_map_construction.out      # GPU 1日志
    ├── <JOB_ID>_2_term_map_construction.out      # GPU 2日志
    └── <JOB_ID>_3_term_map_construction.out      # GPU 3日志
```

## 💡 最佳实践

### 1. 先测试再全量

```bash
# 第一步：dry-run测试（每个GPU 10条）
sbatch retriever/gigaspeech/run_term_map_construction.sh --dry-run

# 第二步：小规模测试（每个GPU 100条）
sbatch retriever/gigaspeech/run_term_map_construction.sh --max 100

# 第三步：全量处理
sbatch retriever/gigaspeech/run_term_map_construction.sh
```

### 2. 监控资源使用

```bash
# 查看GPU利用率
watch -n 1 'squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"'

# 查看内存使用
# 在日志中会有报告
```

### 3. 保留个体输出文件

在确认合并结果正确之前，不要删除`*_gpu*.jsonl`文件，以防需要重新合并。

## 🎓 高级用法

### 自定义GPU数量

修改SLURM脚本和Python调用：

```bash
# 使用8个GPU
#SBATCH --array=0-7

ARGS="--gpu-id $GPU_ID --total-gpus 8"
```

### 处理特定范围

```bash
# 只处理GPU 0-1
sbatch --array=0-1 retriever/gigaspeech/run_term_map_construction.sh
```

### 单机测试（非SLURM）

```bash
# 手动指定GPU
python handle_train_dataset_for_term_map_v2_buzz.py \
    --gpu-id 0 \
    --total-gpus 4 \
    --dry-run
```

## ⚠️ 注意事项

1. **模型加载**：每个GPU都会加载完整的RAG模型（~7GB显存）
2. **TSV加载**：每个GPU都会加载完整的TSV索引（~1-2GB内存）
3. **输出顺序**：合并后的数据顺序与单GPU略有不同（但不影响训练）
4. **原子性**：每个GPU的写入是独立的，不会冲突
5. **失败恢复**：可以单独重新运行失败的GPU任务

## 📊 示例输出

```bash
$ sbatch run_term_map_construction.sh
Submitted batch job 123456

$ squeue -j 123456
JOBID    PARTITION  NAME                USER     ST  TIME  NODES
123456_0 taurus     term_map_const...   user     R   5:30  1
123456_1 taurus     term_map_const...   user     R   5:32  1
123456_2 taurus     term_map_const...   user     R   5:28  1
123456_3 taurus     term_map_const...   user     R   5:31  1

# 等待所有任务完成...

$ ls -lh train_s_zh_with_candidates*.jsonl
-rw-r--r-- 1 user group 150M Dec 26 10:15 train_s_zh_with_candidates_gpu0.jsonl
-rw-r--r-- 1 user group 151M Dec 26 10:16 train_s_zh_with_candidates_gpu1.jsonl
-rw-r--r-- 1 user group 149M Dec 26 10:14 train_s_zh_with_candidates_gpu2.jsonl
-rw-r--r-- 1 user group 152M Dec 26 10:17 train_s_zh_with_candidates_gpu3.jsonl
-rw-r--r-- 1 user group 602M Dec 26 10:17 train_s_zh_with_candidates.jsonl  # 合并后
```

## ✅ 总结

多GPU并行处理可以显著加速大规模数据集构造：

- ✅ **简单易用**：只需提交一次sbatch命令
- ✅ **自动分片**：轮询策略确保负载均衡
- ✅ **自动合并**：GPU 3完成后自动合并
- ✅ **容错性好**：单个GPU失败可独立重试
- ✅ **4倍加速**：理论可达4x，实际约3-3.5x

---

**更新时间**: 2025-12-26  
**版本**: v1.0


















