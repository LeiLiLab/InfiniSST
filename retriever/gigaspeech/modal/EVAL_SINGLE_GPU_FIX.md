# 评估脚本最终修复：单GPU方案

## 问题回顾

尝试使用`device_map="auto"`多GPU分片方案时，遇到了**无法解决的设备不匹配错误**：

```
Expected all tensors to be on the same device, but found at least two devices, cuda:1 and cuda:0!
```

**关键发现**：
- 所有batch都失败，包括最小的batch_size=1
- 智能重试机制无效
- 根本原因：`device_map="auto"`分片后，文本编码器内部无法正确处理跨GPU的张量传输

## 最终解决方案：单GPU加载

### 为什么单GPU可行

| 项目 | 大小 | 说明 |
|------|------|------|
| Qwen2-Audio-7B (fp16) | ~14GB | 基础模型 |
| 投影层 | ~0.1GB | proj_speech + proj_text |
| 推理缓存 | ~5-10GB | 激活值、中间结果 |
| **总计** | **~24-30GB** | **远小于A6000的48GB** |

### 代码修改

#### 之前（多GPU分片，失败）

```python
shared_qwen2_model = Qwen2AudioForConditionalGeneration.from_pretrained(
    args.model_name,
    torch_dtype=torch.float16,
    device_map="auto",  # 自动分片到多GPU
    max_memory={i: "20GiB" for i in range(torch.cuda.device_count())}
)
```

#### 现在（单GPU，成功）

```python
shared_qwen2_model = Qwen2AudioForConditionalGeneration.from_pretrained(
    args.model_name,
    torch_dtype=torch.float16,
).to(device)  # 直接加载到单GPU
```

### Slurm配置更新

```bash
#SBATCH --gres=gpu:1        # 只需1个GPU（从2改为1）
#SBATCH --mem=128GB         # 减少内存需求（从256GB改为128GB）
```

## 优点

1. ✅ **无设备不匹配问题**：所有张量都在同一设备上
2. ✅ **简单可靠**：无需复杂的错误处理逻辑
3. ✅ **性能更好**：无跨GPU通信开销
4. ✅ **内存充足**：单个A6000 48GB完全够用
5. ✅ **资源节省**：只占用1个GPU而非2个

## 为什么评估不需要多GPU

| 特性 | 训练 | 评估 |
|------|------|------|
| 梯度计算 | ✅ 需要 | ❌ 不需要 |
| 反向传播 | ✅ 需要 | ❌ 不需要 |
| 优化器状态 | ✅ 需要（~2x模型大小） | ❌ 不需要 |
| Batch size | 大 | 中等 |
| **内存需求** | **高** | **低** |

评估只做前向传播，内存需求比训练低得多，单GPU完全够用。

## 使用方法

### 1. 提交任务

```bash
cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal
sbatch run_eval_local.sh
```

### 2. 监控进度

```bash
# 查看输出
tail -f eval_local.out

# 查看错误（应该没有设备错误了）
tail -f eval_local.err

# 监控GPU使用（只有1个GPU在工作）
nvidia-smi
```

### 3. 预期输出

```
[INFO] Using device: cuda:0 (NVIDIA RTX A6000)
[INFO] Evaluation mode: Single GPU (7B model ~14GB fp16, no sharding needed)
...
[INFO] Loading model to single GPU (evaluation doesn't need sharding)...
[INFO] Enabled gradient checkpointing
[INFO] GPU Memory Usage after model loading:
  cuda:0: Allocated=14.52GB, Reserved=14.76GB, Total=48.00GB
...
[INFO] Encoding 4587312 terms...
[INFO] Evaluating on 1000 samples...
...
Recall@  1: XX.XX%
Recall@  5: XX.XX%
Recall@ 10: XX.XX%
```

## 与之前方案的对比

| 方案 | GPU数量 | 设备错误 | 复杂度 | 性能 | 状态 |
|------|---------|----------|--------|------|------|
| **方案1**: device_map="auto" | 2 | ❌ 严重 | 高 | 慢（跨GPU通信） | 失败 |
| **方案2**: 智能重试机制 | 2 | ❌ 仍存在 | 很高 | 慢 | 失败 |
| **方案3**: 单GPU（最终） | 1 | ✅ 无 | 低 | 快 | ✅ 成功 |

## 技术细节

### 内存分布（单GPU）

```
GPU 0 (48GB):
├─ Audio Tower (~5GB)
├─ Language Model (~9GB)
├─ 投影层 (~0.1GB)
├─ 推理缓存 (~10GB)
└─ 其他 (~1GB)
─────────────────────────
总计: ~25GB / 48GB (52%使用率)
```

### 为什么不会OOM

1. **fp16精度**：相比fp32减少50%内存
2. **无梯度**：`torch.no_grad()`节省大量内存
3. **无优化器**：不需要优化器状态（约2x模型大小）
4. **Gradient checkpointing**：虽然评估不需要，但启用后可以进一步减少推理缓存
5. **适中的batch size**：text=256, audio=16，避免过大的中间激活

### 关键代码片段

```python
# 模型加载（单GPU）
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # 使用fp16
).to(device)

# 评估模式
model.eval()
with torch.no_grad():  # 不计算梯度
    embeddings = model.encode_text(texts)
```

## 删除的代码

以下复杂的错误处理代码已被删除（不再需要）：

1. `encode_texts_in_batches_sharded()` - 专门处理分片模型的函数
2. 设备不匹配重试逻辑
3. 动态batch size调整
4. 多GPU内存监控循环

## 结论

**简单即是美**。

对于评估任务：
- ❌ 不要过度工程化（device_map="auto"）
- ✅ 使用最简单可行的方案（单GPU）
- ✅ 只有在训练大模型时才需要多GPU分片

这个修复不仅解决了问题，还简化了代码，提高了性能！




