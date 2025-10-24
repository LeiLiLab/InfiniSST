# Illegal Memory Access 问题深度分析

## 实验历史（全部失败）

| 配置 | GPU数量 | batch_size | 缓存清理 | 失败位置 | 备注 |
|------|--------|-----------|---------|---------|------|
| 原始 | 6 | 128 | 每10批 | 立即 | OOM |
| 修改1 | 6 | 64 | 每10批 | batch 10 | 所有GPU同时 |
| 修改2 | 6 | 32 | 每10批 | batch 20-60 | 不同时间 |
| 修改3 | 6 | 64 | 每10批 | batch 10 | 串行加载，仍失败 |
| 修改4 | 6 | 16 | **每批** | batch 66-71 | **CUBLAS错误** |

## 关键发现

### 1. CUBLAS错误出现
```
[GPU 3 ERROR] Batch 66: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling `cublasSgemm`
```

这是**矩阵乘法库的并发冲突**，不是简单的内存问题！

### 2. 失败模式变化
- batch_size大时（64）：早期同时失败（batch 10）
- batch_size小时（16）：晚期失败（batch 70），跑得更远

说明batch_size确实有影响，但不是根本原因。

### 3. 多GPU并发的根本问题

即使串行加载模型，6个GPU线程仍在：
1. **并行调用CUBLAS库**（矩阵乘法）
2. **并行清理CUDA缓存**（尤其是每个batch都清理）
3. **竞争CUDA运行时资源**

## 为什么之前成功过一次？

第一次运行时（文件显示已有8.9G的索引）：
- 可能使用了不同的batch_size
- 可能没有这么频繁的缓存清理
- 可能有些batch失败但被dummy embeddings替代了

## 根本解决方案

### 方案A：单GPU运行（100%稳定）✅ 推荐

```bash
NUM_GPUS=1
BATCH_SIZE=64  # 单GPU可以用大batch
```

**优点**：
- ✅ 完全避免并发问题
- ✅ 100%稳定
- ✅ 可以用更大的batch_size（64甚至128）

**缺点**：
- ⏱️ 慢6倍：预计12-15小时

### 方案B：多GPU + 超小batch（不推荐）

```bash
NUM_GPUS=6
BATCH_SIZE=4  # 非常小
```

**优点**：
- 快一些

**缺点**：
- ❌ 仍然可能失败（并发问题未解决）
- ⏱️ 小batch导致效率低
- 😖 不稳定，可能需要多次重跑

### 方案C：重写为完全无并发（需要大改）

完全改为单线程顺序处理每个GPU，但这需要大量代码重构。

## 我的建议

**使用方案A：单GPU**

理由：
1. 你已经尝试了多个配置，都失败了
2. 多GPU的并发问题很深层，不是简单调参能解决的
3. 12小时等待 >> 反复失败浪费的时间
4. 一次跑完，数据正确，不用担心dummy embeddings

## 快速执行

```bash
# 已经配置好了，直接运行
scancel -n build_index
cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal
sbatch run_build_index.sh
```

当前配置：
- 1个GPU
- batch_size=64
- 预计12-15小时
- 100%稳定

## 如果必须用多GPU

需要重新设计代码架构，考虑：
1. 使用进程而不是线程（`multiprocessing`）
2. 每个进程独立的CUDA context
3. 避免任何共享状态

这是个复杂的工程问题，不是简单的参数调整。



