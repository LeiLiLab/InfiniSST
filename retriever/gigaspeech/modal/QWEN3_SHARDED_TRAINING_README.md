# Qwen3-Omni 30B 分片训练方案（4x A6000）

本方案解决了在4张A6000 (48GB)上训练Qwen3-Omni-30B的显存问题。

## 🎯 核心策略

### 与DDP方案的区别

| 维度 | DDP方案（旧） | 分片方案（新） |
|------|--------------|---------------|
| **进程数** | 4个进程（每GPU一个） | 1个进程 |
| **模型复制** | 每卡复制完整模型 | 模型分片到多卡 |
| **显存占用** | 30B × 4 = 120B参数量 | 30B ÷ 4 ≈ 7.5B/卡 |
| **并行方式** | 数据并行（DDP） | 张量并行（device_map） |
| **适用场景** | 小模型/大显存 | 大模型/小显存 |

### 三大优化

1. **模型分片（Tensor Parallelism）**
   - 使用 `device_map="auto"` 自动将30B模型层切分到4张卡
   - HuggingFace自动管理跨卡通信
   - 显存占用：~8-12GB/卡（30B÷4）

2. **4bit量化 + 梯度检查点**
   - 4bit NF4量化：模型权重压缩4倍
   - 梯度检查点：trade计算换显存
   - 训练时计算用bf16，存储用4bit

3. **梯度累积**
   - 物理batch=8（小，避免OOM）
   - 累积16步 → 有效batch=128
   - 保持训练稳定性

## 📦 依赖安装

```bash
conda activate infinisst

# 关键依赖
pip install bitsandbytes  # 量化支持 (>=0.46.1，当前0.48.1)
pip install accelerate    # 分片支持
pip install peft          # LoRA支持

# 已安装的其他依赖（确认版本）
pip list | grep -E "torch|transformers|peft|bitsandbytes|accelerate"
```

## 🚀 快速开始

### 方式1: 使用提供的脚本（推荐）

```bash
cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech
./modal/run_qwen3_sharded.sh
```

### 方式2: 手动运行

```bash
# 1. 环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export HF_HOME="${HOME}/.cache/huggingface"

# 2. 创建offload目录（用于CPU溢出）
mkdir -p /mnt/data2/jiaxuanluo/offload_qwen3

# 3. 运行训练
python modal/Qwen3_AuT_train_sharded.py \
    --train_samples_path data/balanced_train_set.json \
    --test_samples_path data/balanced_test_set.json \
    --mmap_shard_dir /mnt/gemini/data1/jiaxuanluo/mmap_shards \
    --save_path models/qwen3_aut_sharded.pt \
    --offload_folder /mnt/data2/jiaxuanluo/offload_qwen3 \
    --epochs 20 \
    --batch_size 8 \
    --gradient_accumulation_steps 16 \
    --lr 1e-4 \
    --enable_speech_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --audio_text_loss_ratio 0.3 \
    --audio_term_loss_ratio 0.7
```

## ⚙️ 核心参数说明

### 显存优化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch_size` | 8 | 物理batch size（小以避免OOM） |
| `--gradient_accumulation_steps` | 16 | 梯度累积步数（有效batch = 8×16 = 128） |
| `--offload_folder` | 必需 | CPU/磁盘溢出目录（需SSD，>300GB） |

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--aut_model_name` | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | 30B语音模型 |
| `--text_model_name` | `Qwen/Qwen2-Audio-7B-Instruct` | 7B文本模型 |
| `--enable_speech_lora` | False | 是否在AuT上启用LoRA微调 |
| `--lora_r` | 8 | LoRA秩（建议8-16） |
| `--lora_alpha` | 16 | LoRA alpha（建议16-32） |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 10 | 训练轮数 |
| `--lr` | 1e-4 | 学习率 |
| `--audio_text_loss_ratio` | 0.3 | audio-text损失权重 |
| `--audio_term_loss_ratio` | 0.7 | audio-term损失权重 |

## 🔧 环境变量配置

训练脚本会自动设置以下环境变量（在`Qwen3_AuT_train_sharded.py`中）：

```python
os.environ["AUT_DEVICE_MAP"] = "auto"           # 启用自动分片
os.environ["AUT_LOAD_IN_4BIT"] = "1"            # 4bit量化
os.environ["AUT_MAX_MEMORY"] = "46GiB"          # 每卡显存上限
os.environ["AUT_NO_FLASH_ATTENTION"] = "1"      # 禁用FA2节省显存
os.environ["AUT_DTYPE"] = "bfloat16"            # 计算精度
os.environ["AUT_OFFLOAD_FOLDER"] = "..."        # CPU溢出目录
os.environ["AUT_MAX_CPU_MEMORY"] = "200GiB"     # CPU内存上限
```

这些配置在`Qwen3_AuT_speech_encoder_local.py`中被读取并应用。

## 📊 显存占用估算

### 4x A6000 (48GB/卡) 配置

| 组件 | 显存占用 | 说明 |
|------|---------|------|
| **AuT模型 (30B, 4bit)** | ~10GB/卡 | 30B ÷ 4卡 ÷ 4bit ≈ 10GB |
| **Text模型 (7B)** | ~4GB | 仅在GPU0，frozen |
| **Projection heads** | <1GB | 512维投影层 |
| **LoRA adapters** | ~2GB/卡 | q/k/v LoRA |
| **梯度 + 优化器状态** | ~5GB/卡 | AdamW states |
| **激活值（batch=8）** | ~3GB/卡 | 前向激活+梯度 |
| **其他开销** | ~2GB/卡 | CUDA context等 |
| **总计** | ~25-30GB/卡 | 留余量 |

**关键点**：
- ✅ GPU0占用稍高（~32GB）：AuT部分层 + Text模型 + 投影层
- ✅ GPU1-3占用均衡（~25GB）：AuT切分的层
- ✅ 总体在48GB限制内，有15-20GB余量

## 🚨 故障排查

### Q1: 仍然OOM怎么办？

**方案A: 降低batch size**
```bash
--batch_size 6 \              # 8 → 6
--gradient_accumulation_steps 21  # 保持有效batch≈128
```

**方案B: 使用8bit量化**
```bash
# 在训练脚本中修改：
os.environ["AUT_LOAD_IN_4BIT"] = "0"
os.environ["AUT_LOAD_IN_8BIT"] = "1"
```

**方案C: 减少LoRA秩**
```bash
--lora_r 8 \      # 16 → 8
--lora_alpha 16   # 32 → 16
```

**方案D: 只用2-3张卡**
```bash
export CUDA_VISIBLE_DEVICES=0,1  # 只用2卡
# 模型会自动分片到2卡，每卡~15GB模型
```

### Q2: 训练很慢？

**原因分析**：
- 4bit量化会降低计算速度（2-3倍）
- 禁用Flash Attention也会变慢
- 梯度累积增加总步数

**优化建议**：
1. 确保`offload_folder`在高速SSD上
2. 增加`num_workers=16`加速数据加载
3. 如果显存充足，尝试启用FA2：
   ```python
   os.environ["AUT_NO_FLASH_ATTENTION"] = "0"
   ```

### Q3: 如何验证模型正在分片？

训练开始时查看日志：

```
[INFO] Model loaded successfully
[STRUCT] Model class: Qwen3OmniMoeForConditionalGeneration
...
```

使用`nvidia-smi`监控：
```bash
watch -n 1 nvidia-smi
```

应该看到：
- GPU0: ~32GB占用
- GPU1-3: ~25GB占用（均衡）

### Q4: bitsandbytes报错？

```bash
# 确认CUDA版本
nvcc --version

# 重新安装匹配的bitsandbytes
pip uninstall bitsandbytes -y
pip install bitsandbytes==0.43.3

# 如果还不行，尝试从源码安装
pip install git+https://github.com/TimDettmers/bitsandbytes.git
```

## 📈 训练监控

### 日志输出示例

```
[INFO] Starting sharded training on 4 GPUs
[INFO] Available devices: ['NVIDIA RTX A6000', ...]
[INFO] Loading speech encoder (will be sharded)...
✅ Model loaded successfully
[INFO] Using 4-bit quantization (type=nf4, compute=torch.bfloat16)
[INFO] Text encoder frozen (no gradients)
[INFO] Trainable parameters: 245 tensors
[INFO] Total trainable params: 67,108,864  # ~67M (projection + LoRA)

Epoch 1/20: 100%|████████| 1250/1250 [45:23<00:00, loss=0.4521]
[INFO] Epoch 1 avg loss: 0.4521
[INFO] Evaluating...
[INFO] Recall@5: 45.2%, Recall@10: 58.7%
[INFO] New best model saved (Recall@10: 58.70%)
```

### 关键指标

- **Trainable params**: ~60-100M（projection + LoRA）
- **每epoch时间**: ~30-60分钟（取决于数据量）
- **Recall@10**: 目标 >60%

## 📂 文件结构

```
retriever/gigaspeech/modal/
├── Qwen3_AuT_train_sharded.py          # 分片训练主脚本
├── Qwen3_AuT_speech_encoder_local.py   # 语音编码器（支持量化+分片）
├── run_qwen3_sharded.sh                # 启动脚本
├── QWEN3_SHARDED_TRAINING_README.md    # 本文档
├── Qwen3_AuT_term_level_train_ddp_local.py  # DDP版本（参考）
└── train_ddp_simplified.py             # 数据集工具
```

## 🔄 与其他方案对比

| 方案 | GPU需求 | 显存/卡 | 速度 | 适用场景 |
|------|---------|---------|------|----------|
| **DDP (原方案)** | 8x H200 | >60GB | 快 | 充足资源 |
| **分片+4bit (本方案)** | 4x A6000 | 48GB | 中 | 有限资源 |
| **8bit量化** | 4x A6000 | 48GB | 中慢 | 更稳定 |
| **单卡offload** | 1x A6000 | 48GB | 很慢 | 极限场景 |

## 🎓 原理说明

### 为什么不用DDP？

**DDP问题**：
```
30B模型 × 4卡 × fp16 = 240GB显存
即使4bit: 30B × 4 × 0.5byte = 60GB
4x A6000 (192GB total) → OOM
```

**分片方案**：
```
30B模型 ÷ 4卡 × 4bit = 7.5GB/卡
加上激活+梯度 ≈ 25GB/卡
4x A6000 (192GB total) → ✅ 充足
```

### Text Encoder为什么frozen？

1. **显存节省**：不保存梯度和优化器状态
2. **计算节省**：始终`torch.no_grad()`
3. **效果保证**：text编码器已预训练好，无需微调

### 梯度累积的作用？

```python
# 等价于batch=128的训练，但显存只需batch=8
for i in range(16):  # 累积16步
    loss = forward(batch_size=8) / 16  # scale loss
    loss.backward()  # 累积梯度
optimizer.step()  # 一次更新
```

## 📝 后续优化方向

1. **混合精度优化**：尝试FP8量化（需A100/H100）
2. **更激进的LoRA**：只训练projection，冻结AuT
3. **知识蒸馏**：用30B教7B模型
4. **动态batch**：根据音频长度调整batch size

## 🆘 获取帮助

- **查看完整日志**：`tail -f logs/qwen3_sharded_*.log`
- **监控GPU**：`watch -n 1 nvidia-smi`
- **检查进程**：`ps aux | grep Qwen3_AuT_train_sharded`

---

**重要提醒**：
- ✅ 首次运行会下载30B模型（~60GB），确保网络和磁盘空间充足
- ✅ `offload_folder`必须在高速SSD上，且有>300GB空间
- ✅ 训练前备份现有模型，避免覆盖

