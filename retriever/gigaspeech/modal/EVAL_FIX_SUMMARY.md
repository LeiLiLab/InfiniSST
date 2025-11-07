# 评估脚本OOM问题修复总结

## 问题根源

原始评估脚本尝试在单个GPU上加载完整的Qwen2-Audio-7B模型（约13GB参数），导致OOM错误。

## 解决方案概述

### 1. 模型加载策略改变

**之前**:
```python
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map=None  # 手动放到单个GPU
).to("cuda:0")
```

**现在**:
```python
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",  # 自动分片到多GPU
    max_memory={i: "20GiB" for i in range(torch.cuda.device_count())}
)
```

### 2. 创建简化的wrapper模型

**问题**: 原来的`ContrastiveQwen2AudioModel`会在`__init__`中重新加载模型

**解决**: 创建`SimpleContrastiveModel`类，只包含：
- 投影层（proj_speech, proj_text）
- encode_audio() 和 encode_text() 方法
- 使用已经加载好的分片模型

### 3. 权重加载策略

**关键**: 只加载投影层的权重，不加载基础模型权重

```python
# 只提取投影层权重
proj_state_dict = {}
for k, v in state_dict.items():
    if 'proj_speech' in k or 'proj_text' in k:
        proj_state_dict[k] = v

model.load_state_dict(proj_state_dict, strict=False)
```

### 4. 内存优化措施

1. **减小batch size**:
   - Text编码: 1024 → 256
   - Audio编码: 64 → 16

2. **定期清理GPU缓存**:
   ```python
   if (i // batch_size) % 10 == 0:
       torch.cuda.empty_cache()
   ```

3. **内存监控**: 加载后显示每个GPU的内存使用

## 使用方法

### 提交Slurm任务

```bash
cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal
sbatch run_eval_local.sh
```

### 查看日志

```bash
# 实时查看输出
tail -f eval_local.out

# 实时查看错误
tail -f eval_local.err

# 监控GPU使用
watch -n 1 nvidia-smi
```

### 快速测试

运行快速测试验证配置：

```bash
# 测试模型加载
python test_model_loading.py

# 测试完整流程
python test_eval_quick.py
```

## 预期行为

### GPU内存分布（2x A6000 48GB）

| GPU | 组件 | 预期使用 |
|-----|------|----------|
| GPU 0 | Audio tower (前半) + Language model (前半) + 投影层 | ~25-30GB |
| GPU 1 | Audio tower (后半) + Language model (后半) | ~20-25GB |

### 评估流程

1. **模型加载** (~2分钟)
   - 下载/缓存模型
   - 自动分片到多GPU
   - 加载投影层权重

2. **索引构建** (~1-2分钟)
   - 编码所有glossary terms
   - 构建FAISS索引

3. **样本评估** (~5-10分钟，取决于max_eval）
   - 加载样本
   - 编码音频
   - 计算recall
   - 记录失败案例

## 文件说明

| 文件 | 作用 |
|------|------|
| `eval_local.py` | 主评估脚本（已修复OOM） |
| `run_eval_local.sh` | Slurm提交脚本 |
| `test_model_loading.py` | 测试模型加载和内存使用 |
| `test_eval_quick.py` | 测试完整评估流程 |
| `EVAL_CHANGES.md` | 详细修改说明 |
| `EVAL_FIX_SUMMARY.md` | 本文档（快速参考） |

## 故障排除

### 问题1: 仍然OOM

**解决方案**:
1. 增加GPU数量: `#SBATCH --gres=gpu:3`
2. 减小max_memory: `"20GiB"` → `"15GiB"`
3. 减小评估样本数: `MAX_EVAL=500`

### 问题2: 找不到投影层权重

**现象**: `No projection layer weights found in checkpoint`

**可能原因**:
- Checkpoint文件损坏
- 使用了错误的checkpoint（不是contrastive模型）

**解决**: 检查checkpoint内容
```python
checkpoint = torch.load(checkpoint_path)
print(checkpoint.keys())  # 应该包含 proj_speech.weight 等
```

### 问题3: 模型层分配不均

**检查**:
```bash
nvidia-smi  # 应该看到多个GPU都在使用
```

**调整**: 修改max_memory设置让分配更均匀

## 关键代码片段

### SimpleContrastiveModel定义

```python
class SimpleContrastiveModel(nn.Module):
    def __init__(self, speech_encoder, text_encoder, speech_hidden, text_hidden, proj_dim, device):
        super().__init__()
        self.speech_encoder = speech_encoder
        self.text_encoder = text_encoder
        self.proj_speech = nn.Linear(speech_hidden, proj_dim).to(device)
        self.proj_text = nn.Linear(text_hidden, proj_dim).to(device)
    
    def encode_audio(self, audio_inputs):
        with torch.no_grad():
            emb = self.speech_encoder.predict(audio_inputs)
        emb = emb.float().to(self.proj_speech.weight.device)
        if emb.dim() == 3:
            emb = emb.mean(dim=1)
        return F.normalize(self.proj_speech(emb), dim=-1)
    
    def encode_text(self, texts):
        with torch.no_grad():
            emb = self.text_encoder.predict(texts)
        emb = emb.float().to(self.proj_text.weight.device)
        return F.normalize(self.proj_text(emb), dim=-1)
```

## 与训练脚本的差异

| 特性 | 训练脚本 | 评估脚本 |
|------|----------|----------|
| 并行方式 | DDP (数据并行) | 单进程多GPU (模型并行) |
| 模型加载 | 每个GPU完整模型 | 模型分片到多GPU |
| LoRA | 应用LoRA | 不使用LoRA |
| 梯度 | 需要计算 | 不需要（no_grad） |
| 内存需求 | 高（每GPU完整模型） | 低（分片） |

## 成功标志

运行成功时应该看到：

```
[INFO] Primary device: cuda:0 (NVIDIA RTX A6000)
[INFO] Note: Model will be automatically sharded across 2 GPUs
[INFO] GPU Memory Usage after model loading:
  GPU 0: Allocated=25.43GB, Reserved=26.12GB
  GPU 1: Allocated=22.18GB, Reserved=22.89GB
[INFO] ✅ Loaded 4 projection layer weights
[INFO] Loaded weights: ['proj_speech.weight', 'proj_speech.bias', 'proj_text.weight', 'proj_text.bias']
...
Recall@  1: 45.30% (982 samples)
Recall@  5: 68.50% (982 samples)
Recall@ 10: 78.20% (982 samples)
```

## 总结

核心改动：
1. ✅ 使用`device_map="auto"`自动分片模型
2. ✅ 创建简化的wrapper类避免重新加载模型  
3. ✅ 只加载投影层权重
4. ✅ 减小batch size和定期清理内存
5. ✅ 添加内存监控和诊断工具

这些改动让7B模型可以在2个48GB GPU上顺利评估，避免OOM错误。














