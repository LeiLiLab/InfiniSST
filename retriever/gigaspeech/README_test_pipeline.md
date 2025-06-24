# 测试数据集处理流水线

本目录包含用于处理测试数据集的完整流水线，可以生成用于模型训练和评估的MFA chunk样本。

## 文件说明

### 核心脚本
- `test_samples_generate_pipeline.sh` - 主流水线脚本（SLURM版本）
- `test_pipeline_direct.sh` - 直接执行版本（适用于小数据集）
- `test_pipeline_batch.sh` - 批处理版本（支持多个n值）

### 处理脚本
- `extract_ner_cache_test.py` - 测试数据集命名实体提取
- `train_samples_pre_handle_test.py` - 测试样本预处理
- `handle_MFA_n_chunk_samples_test.py` - MFA chunk处理

### 流水线脚本
- `test_samples_generate_pipeline.sh` - 主要的测试流水线（SLURM版本）
- `test_pipeline_batch.sh` - 批量运行多个n值的流水线
- `test_pipeline_direct.sh` - 直接执行版本（不依赖SLURM）

### 其他文件
- `README_test_pipeline.md` - 本说明文件

## 使用方法

### 1. SLURM流水线（推荐）

```bash
# 基本用法：生成3-chunk样本，使用term字段
bash test_samples_generate_pipeline.sh

# 指定chunk数量
bash test_samples_generate_pipeline.sh 5

# 指定chunk数量和文本字段
bash test_samples_generate_pipeline.sh 3 term
bash test_samples_generate_pipeline.sh 5 short_description
```

### 2. 直接执行版本

```bash
# 在远程服务器上直接运行（适用于小数据集或调试）
bash test_pipeline_direct.sh 3 term
```

### 3. 批处理版本

```bash
# 同时生成多个不同n值的样本
bash test_pipeline_batch.sh
```

## 参数说明

### n (chunk数量)
- 控制每个样本提取的chunk数量
- 默认值：3
- 不同的n值会生成不同的输出文件和目录

### text_field
- `term`: 使用术语标题作为文本输入
- `short_description`: 使用完整描述作为文本输入
- 默认值：term

## 输出文件结构

```
data/
├── named_entities_test.json                    # NER提取结果
├── samples/
│   └── test/
│       ├── term_preprocessed_samples_test.json        # 预处理样本(term)
│       ├── preprocessed_samples_test.json             # 预处理样本(description)
│       ├── test_mfa_1chunks_samples.json              # n=1的chunk样本
│       ├── test_mfa_3chunks_samples.json              # n=3的chunk样本
│       └── test_mfa_5chunks_samples.json              # n=5的chunk样本
└── clap_test_n*.pt                             # 训练的模型文件
```

## 音频输出结构

```
/mnt/data/jiaxuanluo/audio_chunks/
└── test_n*/                                   # 按n值分组
    └── [前3字符]/                             # 第一层目录
        └── [audio_id]/                        # 第二层目录
            └── [segment_id]_chunk_*.wav       # chunk音频文件
```

## 流水线步骤

1. **NER提取** (`extract_ner_cache_test.py`)
   - 从测试TSV文件提取命名实体
   - 输出：`data/named_entities_test.json`

2. **样本预处理** (`train_samples_pre_handle_test.py`)
   - 处理音频文件，提取ground truth terms
   - 输出：`data/samples/test/[term_]preprocessed_samples_test.json`

3. **MFA Chunk处理** (`handle_MFA_n_chunk_samples_test.py`)
   - 基于MFA对齐提取最佳n个chunk
   - 输出：`data/samples/test/test_mfa_Nchunks_samples.json`

4. **模型训练**（可选）
   - 使用生成的chunk样本训练SONAR模型
   - 输出：`data/clap_test_nN.pt`

## 输出格式

生成的MFA chunk样本格式：
```json
{
  "segment_id": "POD0000000663_S0000021",
  "n_chunk_audio": "/mnt/data/jiaxuanluo/audio_chunks/POD/POD0000000663/POD0000000663_S0000021_chunk_0.00_2.43.wav",
  "n_chunk_text": "i'm krista tippett, and this is on being",
  "n_chunk_audio_ground_truth_terms": ["Krista Siegfrids"],
  "chunk_start_time": 0,
  "chunk_end_time": 2.430000000000007,
  "chunk_start_time_abs": 148.37,
  "chunk_end_time_abs": 150.8,
  "actual_chunk_count": 3
}
```

## 监控和调试

### 检查任务状态
```bash
squeue -u $USER
```

### 查看日志
```bash
# 查看流水线日志
tail -f logs/test_pipeline_n3_YYYYMMDD_HHMMSS.log

# 查看具体任务日志
tail -f logs/test_ner_cache_JOBID.out
tail -f logs/test_preprocess_JOBID.out
tail -f logs/test_mfa_chunks_nN_JOBID.out
```

### 取消任务
```bash
# 取消特定任务
scancel JOBID

# 取消所有自己的任务
scancel -u $USER
```

## 故障排除

### 常见问题

1. **任务状态显示 "DependencyNeverSatisfied"**
   - 前置任务失败，检查前置任务的错误日志
   - 解决方案：修复问题后重新提交

2. **"source: not found" 错误**
   - SLURM环境中shell兼容性问题
   - 已修复：使用 `. ~/miniconda3/etc/profile.d/conda.sh` 替代 `source`

3. **"File not found" 错误**
   - 数据集路径问题，确保在正确的服务器上运行
   - 测试数据集路径：`/mnt/data/siqiouyang/datasets/gigaspeech/manifests/test.tsv`

4. **内存不足**
   - 调整SLURM内存参数
   - 对于大数据集，考虑增加 `--mem` 参数

### 环境要求

- **spaCyEnv**: 用于NER提取，需要安装spacy和en_core_web_trf模型
- **infinisst**: 用于样本处理和训练，需要安装项目依赖

### 数据依赖

- 测试TSV文件：`/mnt/data/siqiouyang/datasets/gigaspeech/manifests/test.tsv`
- TextGrid文件：`/mnt/data/siqiouyang/datasets/gigaspeech/textgrids/`
- 术语词典：`data/terms/` 目录下的文件

## 性能优化

- **并行处理**: 使用SLURM数组任务处理大数据集
- **内存管理**: 根据数据大小调整内存分配
- **GPU使用**: NER提取使用GPU加速，chunk处理使用CPU

## 扩展使用

可以通过修改脚本参数来适应不同的需求：
- 调整chunk长度（默认0.96秒）
- 修改chunk数量n
- 更改文本字段（term vs short_description）
- 调整训练参数（epochs, batch_size等）

## 常见问题

### 1. 测试数据集路径错误
确保测试数据集存在于：`/mnt/data/siqiouyang/datasets/gigaspeech/manifests/test.tsv`

### 2. 内存不足
如果处理大量数据时内存不足，可以：
- 减小batch_size
- 增加SLURM内存配置
- 使用直接执行版本分批处理

### 3. TextGrid文件缺失
确保TextGrid文件存在于：`/mnt/data/siqiouyang/datasets/gigaspeech/textgrids/`

### 4. 环境问题
确保已正确配置conda环境：
- `spaCyEnv`: 用于NER提取
- `infinisst`: 用于样本处理和训练

## 性能优化

### 并行处理
- SLURM版本支持任务依赖和并行执行
- 批量处理脚本可以同时处理多个n值

### 存储优化
- 音频文件按层级目录存储，避免单目录文件过多
- 支持文件存在性检查，避免重复处理

### 内存优化
- 使用流式处理大文件
- 及时释放不需要的数据结构

## 扩展功能

### 自定义chunk长度
修改脚本中的 `--chunk_len` 参数（默认0.96秒）

### 自定义输出目录
修改脚本中的输出路径配置

### 添加新的评估指标
在训练脚本中添加更多评估函数 