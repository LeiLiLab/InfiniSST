# 评估脚本修改说明

## 问题

原始脚本在单GPU上加载完整的Qwen2-Audio-7B模型，导致OOM错误：
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 192.00 MiB. 
GPU 0 has a total capacity of 47.53 GiB of which 2.25 MiB is free.
```

## 解决方案

### 1. 使用`device_map="auto"`自动分片

**修改位置**: `eval_local.py` 第270-335行

**原理**: 
- 不再将整个模型加载到单个GPU
- 使用HuggingFace的`device_map="auto"`自动将模型层分配到多个GPU
- 每个GPU最多使用20GB显存

**关键代码**:
```python
shared_qwen2_model = Qwen2AudioForConditionalGeneration.from_pretrained(
    args.model_name,
    torch_dtype=torch.float16,
    device_map="auto",  # 自动分片
    max_memory={i: "20GiB" for i in range(torch.cuda.device_count())}
)
```

### 2. 减小batch size

**修改位置**: 
- `eval_local.py` 第135行: text编码batch size从1024降到256
- `eval_local.py` 第172行: audio编码batch size从64降到16

**原理**: 减少单次前向传播的内存占用

### 3. 定期清理GPU缓存

**修改位置**: `eval_local.py` 第109-130行

**新增功能**:
```python
# 每10个batch清理一次缓存
if (i // batch_size) % 10 == 0:
    torch.cuda.empty_cache()
```

### 4. 添加内存监控

**修改位置**: `eval_local.py` 第341-346行

**功能**: 在模型加载后显示每个GPU的内存使用情况

### 5. Slurm配置优化

**修改位置**: `run_eval_local.sh`

**建议配置**:
```bash
#SBATCH --gres=gpu:2    # 使用2个GPU（可根据需要调整为3或4）
#SBATCH --mem=256GB     # 足够的系统内存
```

## 关键概念说明

### device参数的作用

脚本中的`DEVICE="cuda:0"`参数**不是**指定整个模型的设备，而是：

1. **投影层位置**: `proj_speech`和`proj_text`会放在这个设备上
2. **FAISS索引位置**: 检索索引会放在这个设备上
3. **主设备标识**: 作为某些默认操作的设备

**模型本身**会通过`device_map="auto"`自动分片到**所有可用GPU**上，例如：
- GPU 0: audio_tower的前几层 + language_model的前几层 + 投影层
- GPU 1: audio_tower的后几层 + language_model的后几层

所以即使`DEVICE="cuda:0"`，模型实际上会使用所有GPU。

## 使用方法

### 方式1: 使用Slurm（推荐）

```bash
cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal
sbatch run_eval_local.sh
```

查看结果：
```bash
tail -f eval_local.out  # 查看输出
tail -f eval_local.err  # 查看错误
```

### 方式2: 直接运行（需要激活环境）

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst
cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal

python eval_local.py \
    --model_path /mnt/gemini/data2/jiaxuanluo/models/qwen2_audio_term_level_modal_v2_best.pt \
    --test_samples_path /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/balanced_test_set.json \
    --glossary_path /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_cleaned.json \
    --mmap_shard_dir /mnt/gemini/data1/jiaxuanluo/mmap_shards \
    --max_eval 1000 \
    --device cuda:0
```

## 测试工具

### 测试模型加载

运行测试脚本验证模型能否正确加载：

```bash
python test_model_loading.py
```

这会显示：
- 每个GPU的总内存
- 模型加载后的内存使用
- 模型层在GPU之间的分配情况

## 预期内存使用

在2个A6000 GPU（每个48GB）上：

| 组件 | GPU 0 | GPU 1 | 说明 |
|------|-------|-------|------|
| Audio Tower | ~15GB | ~12GB | 音频编码器的主要部分 |
| Language Model | ~8GB | ~8GB | 语言模型部分 |
| 投影层 | ~0.5GB | - | 仅在GPU 0 |
| 推理缓存 | ~5GB | ~3GB | 激活值和中间结果 |
| **总计** | **~28GB** | **~23GB** | 安全范围内 |

## 故障排除

### 仍然OOM

1. **增加GPU数量**:
   ```bash
   #SBATCH --gres=gpu:3  # 在run_eval_local.sh中修改
   ```

2. **减小max_memory**:
   ```python
   # eval_local.py第293行
   max_memory={i: "15GiB" for i in range(torch.cuda.device_count())}
   ```

3. **减小评估样本数**:
   ```bash
   MAX_EVAL=500  # 在run_eval_local.sh中修改
   ```

4. **进一步减小batch size**:
   ```python
   # eval_local.py
   batch_size=128  # 第135行（text）
   batch_size=8    # 第172行（audio）
   ```

### 检查GPU使用情况

在另一个终端运行：
```bash
watch -n 1 nvidia-smi
```

应该看到模型均匀分布在多个GPU上。

## 与原版本的差异

| 特性 | 原版本 | 新版本 |
|------|--------|--------|
| GPU使用 | 单GPU | 多GPU自动分片 |
| 内存管理 | 无优化 | 定期清理 + 监控 |
| Batch size | 大 (text:1024, audio:64) | 小 (text:256, audio:16) |
| OOM风险 | 高 | 低 |
| 支持模型 | 7B以下 | 7B及以上 |

## 参考

- [HuggingFace Device Map文档](https://huggingface.co/docs/accelerate/usage_guides/big_modeling)
- [PyTorch内存管理](https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

