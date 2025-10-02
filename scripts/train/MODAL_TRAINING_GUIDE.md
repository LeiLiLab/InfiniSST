# InfiniSST Modal 训练指南

本指南介绍如何使用 Modal 云平台训练 InfiniSST 模型。

## 概述

`modal_stage1_gigaspeech_rag.py` 脚本将原本的 SLURM 训练脚本迁移到 Modal 云平台，支持：

- ✅ 多GPU分布式训练 (默认4x A100)
- ✅ 通过 Volume 挂载所有数据和模型
- ✅ 自动数据本地化以提高 I/O 性能
- ✅ DeepSpeed 优化
- ✅ 检查点保存和恢复
- ✅ 实时日志输出

## 前置要求

### 1. 安装 Modal

```bash
pip install modal
```

### 2. 配置 Modal

```bash
# 登录 Modal
modal token new

# 设置 HuggingFace token (用于下载模型)
modal secret create huggingface-token HF_TOKEN=your_hf_token_here
```

### 3. 创建 Volume 并上传数据

Modal 使用 Volume 来持久化存储数据。您需要创建三个 Volume：

```bash
# 创建 Volume
modal volume create infinisst-data
modal volume create infinisst-models
modal volume create infinisst-outputs
```

#### 上传数据到 Volume

```bash
# 上传 GigaSpeech 数据集
modal volume put infinisst-data \
    /mnt/data/siqiouyang/datasets/gigaspeech/ \
    gigaspeech/

# 上传模型文件
modal volume put infinisst-models \
    /mnt/aries/data6/jiaxuanluo/Qwen2.5-7B-Instruct/ \
    Qwen2.5-7B-Instruct/

modal volume put infinisst-models \
    /mnt/aries/data6/xixu/demo/wav2_vec_vox_960h_pl.pt \
    wav2_vec_vox_960h_pl.pt
```

**注意**: 大文件上传可能需要较长时间。建议使用 `modal volume` CLI 工具进行批量上传。

#### 查看 Volume 内容

```bash
# 列出 Volume 内容
modal volume ls infinisst-data
modal volume ls infinisst-models
modal volume ls infinisst-outputs
```

### 4. 上传代码库

由于 Modal 函数需要访问 InfiniSST 代码库，您需要将代码添加到镜像中。有两种方式：

#### 方式 A: 在脚本中添加代码（推荐用于开发）

修改 `modal_stage1_gigaspeech_rag.py` 中的镜像定义：

```python
# 在 image 定义后添加
image = image.add_local_dir(
    "/home/jiaxuanluo/InfiniSST",
    remote_path="/root/InfiniSST"
)
```

#### 方式 B: 使用 Git 克隆（推荐用于生产）

修改镜像定义，添加 git 克隆命令：

```python
image = (
    # ... 现有的镜像配置
    .run_commands([
        "cd /root && git clone https://github.com/your-repo/InfiniSST.git"
    ])
)
```

## 使用方法

### 1. 检查 Volume 结构

首次运行前，建议检查 Volume 是否正确配置：

```bash
cd /home/jiaxuanluo/InfiniSST/scripts/train
modal run modal_stage1_gigaspeech_rag.py --check-volume
```

### 2. 启动训练

```bash
# 基础训练（使用默认参数）
modal run modal_stage1_gigaspeech_rag.py

# 启用数据本地化（推荐，提高I/O性能）
modal run modal_stage1_gigaspeech_rag.py

# 禁用数据本地化（直接从Volume读取）
modal run modal_stage1_gigaspeech_rag.py --no-local-copy

# 从检查点恢复训练
modal run modal_stage1_gigaspeech_rag.py --resume
```

### 3. 部署为持久化服务（可选）

```bash
# 部署到 Modal
modal deploy modal_stage1_gigaspeech_rag.py

# 之后可以通过 Modal 仪表盘或 API 触发训练
```

## 参数配置

脚本支持丰富的参数配置，主要参数包括：

### 模型配置

- `llama_path`: LLM 模型路径（默认: `/models/Qwen2.5-7B-Instruct`）
- `w2v2_path`: Wav2Vec2 模型路径（默认: `/models/wav2_vec_vox_960h_pl.pt`）
- `model_type`: 模型类型（默认: `w2v2_qwen25`）
- `use_flash_attn`: 是否使用 Flash Attention（默认: `True`）

### 数据配置

- `data_path`: 数据集路径（默认: `/data/gigaspeech`）
- `data_split_train`: 训练集分割
- `data_split_eval`: 验证集分割
- `source_lang`: 源语言（默认: `English`）
- `target_lang`: 目标语言（默认: `Chinese`）

### 训练超参数

- `train_bsz`: 训练批次大小（默认: `1800`）
- `learning_rate`: 学习率（默认: `2e-4`）
- `warmup_steps`: 预热步数（默认: `1000`）
- `max_epochs`: 最大训练轮数（默认: `1`）
- `grad_acc_steps`: 梯度累积步数（默认: `4`）

### 硬件配置

- `n_device`: GPU 数量（默认: `4`）
- `gpu`: GPU 类型（在函数装饰器中设置，默认: `A100:4`）

要修改这些参数，可以在 `main()` 函数中的 `train_infinisst.remote()` 调用中添加参数：

```python
result = train_infinisst.remote(
    run_name="my_custom_run",
    n_device=8,  # 使用8个GPU
    learning_rate=1e-4,  # 自定义学习率
    max_epochs=2,  # 训练2个epoch
)
```

## 监控训练

### 1. Modal 仪表盘

访问 [Modal Dashboard](https://modal.com/apps) 查看：
- 实时日志输出
- GPU 使用率
- 训练进度
- 成本统计

### 2. W&B 集成（可选）

要启用 Weights & Biases 监控，修改脚本中的环境变量：

```python
os.environ["WANDB_MODE"] = "online"  # 从 "disabled" 改为 "online"
```

然后设置 W&B API key：

```bash
modal secret create wandb-token WANDB_API_KEY=your_wandb_key
```

## 获取训练结果

### 查看输出文件

```bash
# 列出输出目录
modal volume ls infinisst-outputs

# 下载训练好的模型
modal volume get infinisst-outputs \
    en-zh/runs/stage1_M=12_ls-cv-vp_norm0_qwen_rope_modal/ \
    ./output/
```

### 访问检查点

训练过程中会定期保存检查点到 `/output/en-zh/runs/{run_name}/`：

- `last.ckpt/`: 最新检查点（DeepSpeed 格式）
- `last.ckpt/pytorch_model.bin`: 转换后的标准 PyTorch 格式

## 成本优化建议

1. **选择合适的 GPU 类型**
   - A100: 性价比高，适合大多数训练任务
   - H100: 性能最强，但成本较高
   - A10G: 成本最低，适合小规模实验

2. **启用数据本地化**
   - 设置 `use_local_copy=True`（默认）将数据复制到本地 NVMe
   - 可显著提升 I/O 性能，减少训练时间

3. **调整批次大小和梯度累积**
   - 增大 `train_bsz` 和 `grad_acc_steps` 可以更好地利用 GPU
   - 根据 GPU 内存调整参数

4. **使用 spot instances（可选）**
   - Modal 支持 spot instances，成本更低但可能被中断
   - 启用检查点恢复（`--resume`）以应对中断

## 故障排除

### 1. 找不到训练脚本

**错误**: `FileNotFoundError: Training script not found: /root/InfiniSST/train/main.py`

**解决方案**: 确保代码库已添加到镜像中（参见"上传代码库"部分）

### 2. CUDA 内存不足

**错误**: `RuntimeError: CUDA out of memory`

**解决方案**:
- 减小 `train_bsz` 或 `bsz_sent`
- 增大 `grad_acc_steps`
- 使用更大的 GPU（如 H100）

### 3. 数据文件找不到

**错误**: `FileNotFoundError: [Errno 2] No such file or directory: '/data/gigaspeech/...'`

**解决方案**:
- 运行 `--check-volume` 检查 Volume 结构
- 确保数据已正确上传到 Volume

### 4. DeepSpeed 初始化失败

**错误**: `DeepSpeed initialization failed`

**解决方案**:
- 检查 GPU 数量是否匹配 `n_device`
- 确保 NCCL 环境变量正确设置
- 尝试降低 `deepspeed_stage`（从 3 降到 2 或 1）

## 高级用法

### 自定义训练脚本

如果需要修改训练逻辑，可以：

1. 修改本地的 `train/main.py`
2. 重新构建镜像（Modal 会自动检测代码变化）
3. 重新运行训练

### 多任务并行训练

Modal 支持同时运行多个训练任务：

```python
# 启动多个训练任务
results = []
for run_name in ["exp1", "exp2", "exp3"]:
    result = train_infinisst.spawn(
        run_name=run_name,
        learning_rate=1e-4 * (1 + 0.5 * i),
    )
    results.append(result)

# 等待所有任务完成
for result in results:
    print(result.get())
```

### 导出训练指标

可以修改脚本以导出更多训练指标：

```python
# 在训练完成后读取 tensorboard 日志
import tensorboard
# ... 解析和导出逻辑
```

## 参考资料

- [Modal 官方文档](https://modal.com/docs)
- [DeepSpeed 文档](https://www.deepspeed.ai/)
- [PyTorch Lightning 文档](https://lightning.ai/docs/pytorch/stable/)
- [InfiniSST 项目文档](../../../README.md)

## 技术支持

如有问题，请：
1. 查看 Modal 仪表盘中的详细日志
2. 检查本指南的故障排除部分
3. 联系项目维护者

---

**最后更新**: 2025-09-30



