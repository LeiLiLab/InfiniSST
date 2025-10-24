# 本地模型评估指南

## 概述

这个工具用于在本地评估训练好的Qwen2-Audio模型，不需要Modal或DDP，支持单GPU运行。

## 文件说明

- `eval_local.py`: 本地评估主脚本（单进程版本）
- `run_eval_local.sh`: 便捷运行脚本
- `LOCAL_EVAL_README.md`: 本文档

## 快速开始

### 1. 修改配置

编辑 `run_eval_local.sh`，修改以下参数：

```bash
# 模型路径（必须修改）
MODEL_PATH="/mnt/gemini/data2/jiaxuanluo/models/qwen2_audio_term_level_modal_v2_best.pt"

# 数据路径
TEST_SAMPLES="/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/balanced_test_set.json"
GLOSSARY_PATH="/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_cleaned.json"

# mmap音频分片目录（可选，如果你的音频数据在mmap中）
MMAP_SHARD_DIR="/mnt/gemini/data1/jiaxuanluo/mmap_shards"

# 评估配置
MAX_EVAL=1000           # 最大评估样本数
NUM_FAILED_CASES=10     # 打印的失败案例数量
DEVICE="cuda:0"         # 使用的GPU设备
```

### 2. 运行评估

```bash
cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal
./run_eval_local.sh
```

或者直接使用Python脚本：

```bash
python eval_local.py \
    --model_path /path/to/your/model.pt \
    --test_samples_path /path/to/test_samples.json \
    --glossary_path /path/to/glossary.json \
    --mmap_shard_dir /path/to/mmap_shards \
    --max_eval 1000 \
    --num_failed_cases 10 \
    --device cuda:0
```

## 输出说明

评估脚本会输出以下信息：

### 1. 总体评估结果

```
================================================================================
EVALUATION RESULTS
================================================================================
Recall@  1: 45.30% (1000 samples)
Recall@  5: 68.50% (1000 samples)
Recall@ 10: 78.20% (1000 samples)
Recall@ 20: 85.10% (1000 samples)
Recall@ 50: 92.30% (1000 samples)
```

### 2. 失败案例分析

对于Recall@10 = 0的样本（完全召回失败），脚本会打印前N个案例：

```
================================================================================
FAILED CASES (Recall@10 = 0, showing first 10)
================================================================================

--- Failed Case #1 ---
Text chunk: This is the transcription text of the audio...
Ground truth terms: ['artificial intelligence', 'machine learning', 'neural network']
Retrieved top-10: ['deep learning', 'computer vision', 'natural language processing', ...]
Audio path: /path/to/audio.wav
Top-10 distances: ['0.3245', '0.3567', '0.3890', ...]

--- Failed Case #2 ---
...
```

**失败案例包含的信息：**
- **Text chunk**: 音频对应的转录文本
- **Ground truth terms**: 该音频应该召回的正确术语
- **Retrieved top-10**: 模型实际检索到的前10个术语
- **Audio path**: 音频文件路径（用于复查）
- **Top-10 distances**: 前10个检索结果的距离（越小越相似）

### 3. 失败率统计

```
[INFO] Total failed cases (Recall@10=0): 218
[INFO] Failure rate: 21.8%
```

## 参数说明

### 必需参数

- `--model_path`: 训练好的模型checkpoint路径
- `--test_samples_path`: 测试样本JSON文件路径
- `--glossary_path`: 词汇表JSON文件路径

### 可选参数

- `--mmap_shard_dir`: mmap音频分片目录（如果使用mmap格式的音频数据）
- `--model_name`: 基础模型名称（默认: `Qwen/Qwen2-Audio-7B-Instruct`）
- `--lora_r`: LoRA rank（默认: 16）
- `--lora_alpha`: LoRA alpha（默认: 32）
- `--lora_dropout`: LoRA dropout（默认: 0.1）
- `--max_eval`: 最大评估样本数（默认: 1000）
- `--device`: 使用的GPU设备（默认: `cuda:0`）
- `--num_failed_cases`: 打印的失败案例数量（默认: 10）

## 注意事项

1. **内存需求**: 
   - **单GPU方案**: Qwen2-Audio-7B模型（fp16约14GB）直接加载到单个GPU
   - 推荐配置: 1个GPU，至少24GB显存（48GB A6000足够）
   - 评估无需梯度，内存需求远小于训练
   - 如果仍然OOM，可以：
     - 减小`--max_eval`参数
     - 减小batch size（在eval_local.py中修改text=256, audio=16）

2. **数据格式**:
   - 支持两种音频加载方式：
     - mmap格式（推荐，更快）：提供`--mmap_shard_dir`
     - 文件路径格式：不提供`--mmap_shard_dir`，从JSON中的路径直接加载

3. **模型兼容性**:
   - 脚本会自动处理DDP包装的模型（去除`module.`前缀）
   - 支持从checkpoint字典或直接state_dict加载

4. **失败案例分析建议**:
   - 查看失败案例的文本和术语，分析是否存在：
     - 术语不在词汇表中（OOV问题）
     - 音频质量问题
     - 模型对特定领域/口音的泛化能力不足
     - 检索到的术语与ground truth语义相近但表述不同

## 对比Modal评估

相比Modal上的DDP评估，本地评估：

**优势：**
- ✅ 不消耗Modal credits
- ✅ 更详细的失败案例分析
- ✅ 更灵活的调试和分析
- ✅ 单进程更容易调试

**限制：**
- ⚠️ 仅支持单GPU（如需多GPU可以手动修改）
- ⚠️ 评估速度可能略慢于多GPU DDP

## 故障排除

### 问题1: CUDA内存不足

**症状**: `RuntimeError: CUDA out of memory` 或 `torch.OutOfMemoryError`

**解决方案**:

1. **检查GPU显存**:
   ```bash
   nvidia-smi
   ```
   确保有至少24GB可用显存

2. **减小评估样本数**:
   ```bash
   MAX_EVAL=500  # 在run_eval_local.sh中修改
   ```

3. **减小batch size**:
   - 编辑`eval_local.py`
   - 第190行: 修改`batch_size=256`为更小的值（如128）
   - 第227行: 修改`batch_size=16`为更小的值（如8）

4. **使用更小的GPU**:
   如果只有较小的GPU（如V100 32GB），考虑减小模型精度或使用量化

### 问题2: 找不到模型文件

**症状**: `FileNotFoundError` 或 `Failed to load model weights`

**解决方案**:
- 检查`MODEL_PATH`是否正确
- 确认模型文件存在：`ls -lh /path/to/model.pt`

### 问题3: mmap音频加载失败

**症状**: `Failed to load audio for key xxx`

**解决方案**:
- 检查`MMAP_SHARD_DIR`路径是否正确
- 确认mmap分片文件完整：`ls -lh /path/to/mmap_shards/`

### 问题4: 导入错误

**症状**: `ModuleNotFoundError: No module named 'xxx'`

**解决方案**:
```bash
# 确保在正确的conda环境中
conda activate infinisst

# 检查依赖
pip list | grep -E "torch|transformers|faiss"
```

## 示例输出

完整的评估输出示例：

```
=== 本地模型评估 ===
模型路径: /mnt/gemini/data2/jiaxuanluo/models/qwen2_audio_term_level_modal_v2_best.pt
...

[INFO] Using device: cuda:0 (NVIDIA RTX A6000)

================================================================================
LOADING DATASET
================================================================================
[INFO] Loading samples from: /path/to/test_samples.json
[INFO] Initializing mmap audio database from: /path/to/mmap_shards
[INFO] Loaded 5000 valid samples

================================================================================
LOADING GLOSSARY
================================================================================
[INFO] Loading glossary from: /path/to/glossary.json
[INFO] Loaded 15234 unique terms

================================================================================
LOADING MODEL
================================================================================
[INFO] Loading base model: Qwen/Qwen2-Audio-7B-Instruct
...
[INFO] ✅ Model weights loaded successfully

================================================================================
SETTING UP RETRIEVER
================================================================================
[INFO] Retriever index size: 15234 terms

================================================================================
RUNNING EVALUATION
================================================================================
[INFO] Loading evaluation samples...
Loading samples: 100%|████████████| 1000/1000 [00:05<00:00, 198.23it/s]
[INFO] Evaluating on 982 samples...
Evaluating: 100%|████████████████| 982/982 [01:23<00:00, 11.78it/s]

================================================================================
EVALUATION RESULTS
================================================================================
Recall@  1: 45.30% (982 samples)
Recall@  5: 68.50% (982 samples)
Recall@ 10: 78.20% (982 samples)
Recall@ 20: 85.10% (982 samples)
Recall@ 50: 92.30% (982 samples)

================================================================================
FAILED CASES (Recall@10 = 0, showing first 10)
================================================================================
...

[INFO] Total failed cases (Recall@10=0): 214
[INFO] Failure rate: 21.8%

================================================================================
EVALUATION COMPLETED
================================================================================
```

## 联系方式

如有问题，请联系开发者或查看相关文档。

