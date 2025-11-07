# GPU重置说明

## 问题

遇到了 "CUDA error: an illegal memory access was encountered" 错误后，GPU可能处于不稳定状态，需要重置。

## 解决方法

### 方法1：重新提交作业（推荐）

由于你使用的是SLURM，最简单的方法是重新提交作业，SLURM会分配干净的GPU：

```bash
cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal

# 取消当前作业（如果还在运行）
scancel -n build_index

# 重新提交
sbatch run_build_index.sh
```

### 方法2：手动检查和清理

如果需要手动清理：

```bash
# 检查是否有残留进程
nvidia-smi

# 如果有你的Python进程还在运行，找到PID并kill
ps aux | grep build_index_multi_gpu
kill -9 <PID>

# 然后重新提交作业
sbatch run_build_index.sh
```

## 修复内容

已将 `BATCH_SIZE` 从 **64 降低到 8**，这应该能解决 illegal memory access 错误。

### 如果仍然出错

如果 batch_size=8 还是出错，可以进一步降低：

```bash
# 编辑 run_build_index.sh
vim run_build_index.sh

# 修改这一行：
BATCH_SIZE=4    # 或者 2
```

## 错误原因分析

`illegal memory access` 错误与 OOM 不同：
- **OOM**: 显存不足，可以通过降低batch_size解决
- **Illegal memory access**: 内存访问越界，通常是因为：
  1. batch太大导致中间激活值超出分配的内存范围
  2. LoRA层在处理大batch时的数值问题
  3. 多GPU并发时的竞争条件

## 预期结果

使用 batch_size=8 后：
- 每个GPU处理约 764,552 / 8 = 95,569 个batch
- 每个batch处理8个terms
- 总处理时间约 2-3小时（取决于GPU速度）













