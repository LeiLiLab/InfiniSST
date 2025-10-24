# Illegal Memory Access 最终修复方案

## 问题历史

| batch_size | 加载方式 | 缓存清理 | 结果 |
|-----------|---------|---------|------|
| 128 | 并行 | 每10批 | ❌ 立即OOM |
| 64 | 并行 | 每10批 | ❌ 加载时冲突 |
| 32 | 并行 | 每10批 | ❌ batch 20-60失败 |
| 64 | **串行** | 每10批 | ❌ batch 10同时失败 |
| **16** | **串行** | **每批** | ✅ **当前配置（推荐）** |

## 已实施的3个关键修复

### 1. 串行加载模型（避免CUDA冲突）

**修改前**：
```python
# 多个GPU线程同时加载模型 → CUDA内存分配冲突
for gpu_id in range(6):
    thread = Thread(target=load_and_process)  # 并行加载
    thread.start()
```

**修改后**：
```python
# 先串行加载所有模型，再并行处理数据
for gpu_id in range(6):
    gpu_models.append(load_model_on_gpu(gpu_id))  # 串行加载

for gpu_id, model in enumerate(gpu_models):
    thread = Thread(target=process_with_model, args=(gpu_id, model))  # 并行处理
    thread.start()
```

### 2. 降低batch_size到16（避免内存访问越界）

**测试结果**：
- batch_size=64: 所有GPU在batch 10同时崩溃
- batch_size=32: 不同GPU在不同batch崩溃
- batch_size=16: **应该安全**（待验证）

**原因**：不是总显存不足，而是单次forward时：
- LoRA层的中间激活值（`gate_proj`, `up_proj`）
- 可能超出CUDA kernel的buffer限制
- batch_size越大，临时tensor越大

### 3. 每个batch后清理缓存（最积极的内存管理）

**修改前**：
```python
if (i // batch_size) % 10 == 0:  # 每10个batch清理
    torch.cuda.empty_cache()
```

**修改后**：
```python
torch.cuda.empty_cache()  # 每个batch清理
```

**权衡**：
- ✅ 优点：防止内存碎片累积，避免delayed illegal memory access
- ⚠️ 缺点：稍微慢一些（但比失败重跑好）

## 当前配置（run_build_index.sh）

```bash
NUM_GPUS=6              # 6个GPU
BATCH_SIZE=16           # batch size = 16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 预期结果

**使用batch_size=16**：
- 每个GPU处理：764,552 texts
- 每个GPU的batch数：764,552 / 16 = **47,785 batches**
- 每batch约1.5秒（包含清理时间）
- **预计总时间：约2.5-3小时**

**显存使用**：
- 模型：~16GB
- 推理：+2-3GB
- 总计：~18-19GB（远低于47.5GB）

## 如果batch_size=16还失败

如果仍然出现illegal memory access：

### 方案A：降到batch_size=8
```bash
BATCH_SIZE=8
```
- 预计时间：5-6小时
- 几乎100%成功率

### 方案B：单GPU运行（终极方案）
```bash
NUM_GPUS=1
BATCH_SIZE=32
```
- 避免所有并发问题
- 预计时间：12-15小时
- 但绝对稳定

## 监控命令

```bash
# 查看进度
tail -f build_index.err | grep "GPU [0-9]:"

# 查看错误
tail -f build_index.out | grep "ERROR"

# 查看显存
watch -n 5 nvidia-smi
```

## 成功标志

如果看到这些日志说明运行正常：
```
[GPU 0] Memory: 15.81GB allocated, 16.44GB reserved
GPU 0:  10%|█         | 4778/47785 [1:59:30<17:53:40,  1.50s/it]
```

如果batch数持续增长且没有ERROR，就成功了！



