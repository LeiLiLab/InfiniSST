# Hard Negative Mining 使用指南

## 📋 概述

本方案通过离线挖掘 Hard Negatives (HN) 来提升模型的 **Recall@5/10** 性能。

### 问题诊断
- ✅ **Recall@100 高**：说明模型全局语义对齐良好
- ❌ **Recall@10 低**：说明局部近邻存在混淆

### 解决方案
通过 Hard Negative Mining，将训练梯度集中在"近邻干扰项"，优化决策边界间距，把真词往更靠前挤。

---

## 🚀 快速开始

### 一键运行完整流程

```bash
cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal
chmod +x run_hard_negative_training.sh
./run_hard_negative_training.sh
```

该脚本会自动执行：
1. 检查前置条件（模型、索引）
2. 挖掘训练集 Hard Negatives
3. 使用 HN 训练新模型
4. 评估新模型性能

---

## 📚 详细步骤

### 步骤 0: 准备工作

**前置条件：**
1. 已训练的基础模型（例如：`qwen2_audio_term_level_modal_v2_best.pt`）
2. 预构建的 FAISS 索引（如果没有，脚本会自动构建）
3. mmap 音频数据库

**构建 FAISS 索引（如果需要）：**

```bash
python build_index_multi_gpu.py \
    --model_path /mnt/gemini/data2/jiaxuanluo/models/qwen2_audio_term_level_modal_v2_best.pt \
    --glossary_path /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_cleaned.json \
    --output_path /mnt/gemini/data2/jiaxuanluo/models/qwen2_audio_term_index.pkl \
    --model_name Qwen/Qwen2-Audio-7B-Instruct \
    --lora_r 16 \
    --lora_alpha 32 \
    --num_gpus 6 \
    --batch_size 4
```

### 步骤 1: 挖掘 Hard Negatives

**挖掘训练集：**

```bash
python mine_hard_negatives.py \
    --samples_path /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/balanced_train_set.json \
    --mmap_dir /mnt/gemini/data1/jiaxuanluo/mmap_shards \
    --faiss_index_pkl /mnt/gemini/data2/jiaxuanluo/models/qwen2_audio_term_index.pkl \
    --model_path /mnt/gemini/data2/jiaxuanluo/models/qwen2_audio_term_level_modal_v2_best.pt \
    --model_name Qwen/Qwen2-Audio-7B-Instruct \
    --lora_r 16 \
    --lora_alpha 32 \
    --out_path /mnt/gemini/data2/jiaxuanluo/models/hard_negs_train.jsonl \
    --topk 200 \
    --batch_size 128
```

**输出格式（JSONL）：**

```json
{"audio_key": "audio_123", "hard_negs": ["term1", "term2", ...], "topk": 200, "num_gt": 3}
```

**挖掘测试集（可选，用于分析）：**

```bash
python mine_hard_negatives.py \
    --samples_path /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/balanced_test_set.json \
    --mmap_dir /mnt/gemini/data1/jiaxuanluo/mmap_shards \
    --faiss_index_pkl /mnt/gemini/data2/jiaxuanluo/models/qwen2_audio_term_index.pkl \
    --model_path /mnt/gemini/data2/jiaxuanluo/models/qwen2_audio_term_level_modal_v2_best.pt \
    --out_path /mnt/gemini/data2/jiaxuanluo/models/hard_negs_test.jsonl \
    --topk 200
```

### 步骤 2: 使用 HN 训练

**本地训练（多GPU）：**

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m torch.distributed.run \
    --nproc_per_node=4 \
    train_ddp_simplified.py \
    --train_samples_path /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/balanced_train_set.json \
    --test_samples_path /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/balanced_test_set.json \
    --glossary_path /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_cleaned.json \
    --mmap_shard_dir /mnt/gemini/data1/jiaxuanluo/mmap_shards \
    --save_path /mnt/gemini/data2/jiaxuanluo/models/qwen2_audio_hn_v1.pt \
    --best_model_path /mnt/gemini/data2/jiaxuanluo/models/qwen2_audio_term_level_modal_v2_best.pt \
    --epochs 15 \
    --batch_size 256 \
    --gradient_accumulation_steps 8 \
    --lr 5e-5 \
    --patience 3 \
    --audio_text_loss_ratio 0.2 \
    --audio_term_loss_ratio 0.3 \
    --hard_neg_jsonl /mnt/gemini/data2/jiaxuanluo/models/hard_negs_train.jsonl \
    --max_hn_per_sample 15 \
    --rand_neg_per_sample 5 \
    --hard_neg_loss_ratio 0.5
```

**Modal 云端训练：**

修改 `modal_qwen2_audio_training.py`，在 `train_ddp_modal` 函数调用时添加 HN 参数：

```python
training_args = {
    # ... 其他参数 ...
    "hard_neg_jsonl": "/data/hard_negs_train.jsonl",
    "max_hn_per_sample": 15,
    "rand_neg_per_sample": 5,
    "hard_neg_loss_ratio": 0.5,
}
```

### 步骤 3: 评估新模型

```bash
python eval_local.py \
    --model_path /mnt/gemini/data2/jiaxuanluo/models/qwen2_audio_hn_v1_best.pt \
    --test_samples_path /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/balanced_test_set.json \
    --glossary_path /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_cleaned.json \
    --mmap_shard_dir /mnt/gemini/data1/jiaxuanluo/mmap_shards \
    --max_eval 1000 \
    --device cuda:0
```

---

## 🔧 超参数调优指南

### 挖矿阶段参数

| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|----------|
| `--topk` | 200 | 检索的候选数 | 200-500，越大越难但计算量大 |
| `--batch_size` | 128 | 批处理大小 | 根据GPU显存调整 |

### 训练阶段参数

| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|----------|
| `--max_hn_per_sample` | 15 | 每样本使用的HN数量 | **关键参数**，10-20 |
| `--rand_neg_per_sample` | 5 | 每样本使用的随机负例 | 3-10，防止过拟合 |
| `--hard_neg_loss_ratio` | 0.5 | HN损失权重 | **关键参数**，0.3-0.7 |
| `--audio_text_loss_ratio` | 0.2 | 音频-文本损失权重 | 降低为0.1-0.2 |
| `--audio_term_loss_ratio` | 0.3 | 音频-术语损失权重 | 0.2-0.5 |
| `--lr` | 5e-5 | 学习率 | 降低以微调，1e-5到5e-5 |

### 推荐配置组合

**保守配置（稳定优先）：**
- `max_hn_per_sample=10`
- `rand_neg_per_sample=5`
- `hard_neg_loss_ratio=0.3`
- `lr=1e-5`

**激进配置（性能优先）：**
- `max_hn_per_sample=20`
- `rand_neg_per_sample=3`
- `hard_neg_loss_ratio=0.7`
- `lr=5e-5`

**平衡配置（推荐起点）：**
- `max_hn_per_sample=15`
- `rand_neg_per_sample=5`
- `hard_neg_loss_ratio=0.5`
- `lr=5e-5`

---

## 📊 效果评估

### 关键指标

重点关注以下指标的提升：

1. **Recall@5** - 主要优化目标
2. **Recall@10** - 主要优化目标  
3. **Recall@100** - 应保持不降

### 对比分析

```bash
# 基础模型评估
Recall@5:  XX.X%
Recall@10: XX.X%
Recall@100: XX.X%

# HN训练后评估
Recall@5:  YY.Y% (+Z.Z%)  ← 期望提升 5-15%
Recall@10: YY.Y% (+Z.Z%)  ← 期望提升 5-10%
Recall@100: YY.Y% (±Z.Z%) ← 期望保持稳定
```

### Seen vs Unseen 术语

训练脚本会自动报告：
- **Seen Recall**: 训练集中见过的术语
- **Unseen Recall**: 训练集中未见过的术语

HN 训练应该同时提升两者的性能。

---

## 🔄 在线刷新策略（进阶）

### 为什么需要在线刷新？

随着模型训练，Hard Negatives 会变得"不够hard"，需要定期更新。

### 实现方案

**简单版（每 2 epoch 全量刷新）：**

在训练脚本中添加：

```python
if (epoch + 1) % 2 == 0 and rank == 0:
    # 重新挖矿
    print("[INFO] Refreshing hard negatives...")
    # 调用挖矿脚本...
```

**高效版（只刷新子集）：**

每个 epoch 只对当前使用的样本子集（如随机 50k 条）重新挖矿。

---

## 🐛 故障排查

### 问题 1: CUDA OOM

**症状：** `RuntimeError: CUDA out of memory`

**解决方案：**
1. 降低 `--batch_size`（挖矿）或 `--max_hn_per_sample`（训练）
2. 降低 `--rand_neg_per_sample`
3. 启用 `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

### 问题 2: Hard Negative Loss 为 0

**症状：** 日志显示 `hard_neg_loss=0.0`

**原因：**
1. HN 文件路径错误或文件不存在
2. audio_key 匹配失败（mmap 模式 vs 文件路径模式）

**解决方案：**
1. 检查 `--hard_neg_jsonl` 路径
2. 确保挖矿时使用的 audio_key 格式与训练时一致

### 问题 3: Recall 反而下降

**症状：** 训练后 Recall@10 反而降低

**原因：**
1. `hard_neg_loss_ratio` 过大（如 > 0.8），压制了其他损失
2. `max_hn_per_sample` 过大，导致过拟合局部边界
3. 学习率过大，破坏了原有知识

**解决方案：**
1. 降低 `hard_neg_loss_ratio` 到 0.3-0.5
2. 降低 `max_hn_per_sample` 到 10-15
3. 降低学习率到 1e-5 或 2e-5
4. 增加 `rand_neg_per_sample` 以增加多样性

### 问题 4: audio_key 不匹配

**症状：** 所有样本的 HN 都为空

**原因：**
- mmap 模式：audio_key 是提取的短key（如 `"XL/audio_000123"`）
- 文件路径模式：audio_key 是完整路径

**解决方案：**
确保挖矿和训练使用相同的数据加载模式（都用 mmap 或都用文件路径）。

---

## 💡 最佳实践

### 1. 迭代策略

```
第一轮：离线挖矿 + 训练（本文档方案）
   ↓
评估效果，调整超参数
   ↓
第二轮：在线刷新 + 继续训练
   ↓
达到满意效果
```

### 2. 难度课程表

```python
# 第 1-2 个 epoch：简单 HN
max_hn_per_sample = 5

# 第 3-5 个 epoch：中等 HN  
max_hn_per_sample = 10

# 第 6+ 个 epoch：困难 HN
max_hn_per_sample = 15
```

### 3. 混合采样比例

推荐 HN:随机 = 3:1
```python
max_hn_per_sample = 15
rand_neg_per_sample = 5  # 15:5 = 3:1
```

完全用 HN 会过拟合边界，混入随机负例更稳健。

### 4. 温度调节

如果 logits 过"软"，R@5/10 难以提升，可以：
- 降低温度：`temperature=0.05` 或 `0.04`（默认 0.07）
- 或引入可学习的 `logit_scale`

### 5. 监控指标

训练时重点监控：
- `hard_neg_loss` 的数值（应该在合理范围，如 1-3）
- Recall@5/10 是否持续提升
- Recall@100 是否保持稳定

---

## 📖 原理解释

### 为什么 HN 能提升 Recall@5/10？

**问题诊断：**
- Recall@100 高 → 模型能把真词放进前 100 → 全局对齐OK
- Recall@10 低 → 真词在第 11-100 位 → 局部混淆

**HN 的作用：**
1. **集中梯度**：只在"近邻干扰项"上优化，而非全库随机负例
2. **拉开间距**：增大正例与 HN 之间的 margin
3. **推动正例前移**：等价于把真词从第 50 位推到第 5 位

**数学直觉：**

传统 InfoNCE：
```
L = -log( exp(s_pos) / (exp(s_pos) + Σ exp(s_neg_all)) )
```
梯度分散在所有负例上。

HN 对比损失：
```
L = -log( exp(s_pos) / (exp(s_pos) + Σ exp(s_hard_neg)) )
```
梯度集中在困难负例上，优化效率更高。

---

## 🔗 相关文件

- `mine_hard_negatives.py` - HN 挖矿脚本
- `train_ddp_simplified.py` - 支持 HN 的训练脚本
- `run_hard_negative_training.sh` - 完整流程自动化脚本
- `build_index_multi_gpu.py` - FAISS 索引构建
- `eval_local.py` - 模型评估脚本

---

## 🎯 快速参考

### 最小化命令集

```bash
# 1. 挖矿
python mine_hard_negatives.py \
    --samples_path train.json \
    --mmap_dir mmap_shards/ \
    --faiss_index_pkl index.pkl \
    --model_path best.pt \
    --out_path hn_train.jsonl \
    --topk 200

# 2. 训练
python -m torch.distributed.run --nproc_per_node=4 train_ddp_simplified.py \
    --train_samples_path train.json \
    --mmap_shard_dir mmap_shards/ \
    --best_model_path best.pt \
    --hard_neg_jsonl hn_train.jsonl \
    --max_hn_per_sample 15 \
    --rand_neg_per_sample 5 \
    --hard_neg_loss_ratio 0.5

# 3. 评估
python eval_local.py --model_path new_best.pt --max_eval 1000
```

---

## 📚 参考资料

- [DPR (Dense Passage Retrieval)](https://arxiv.org/abs/2004.04906) - Hard Negative Mining 在检索中的应用
- [ANCE (Approximate nearest neighbor Negative Contrastive Estimation)](https://arxiv.org/abs/2007.00808) - 在线 HN 更新
- [InfoNCE Loss](https://arxiv.org/abs/1807.03748) - 对比学习理论基础

---

## ❓ 常见问题

**Q: 第一轮训练需要多久？**  
A: 取决于数据量和 GPU。50k 样本在 4x A100 上约 2-3 小时。

**Q: 需要重新挖矿多少次？**  
A: 建议至少 2 轮。第一轮离线挖矿+训练，第二轮根据新模型重新挖矿。

**Q: HN 数量越多越好吗？**  
A: 不是。过多 HN 会导致过拟合。15-20 个通常是最佳平衡点。

**Q: 可以跳过随机负例吗？**  
A: 不推荐。随机负例提供多样性，防止模型只学习局部边界。

**Q: 如何判断是否过拟合？**  
A: 如果训练集 Recall 很高但测试集下降，或者 Unseen Recall 明显低于 Seen Recall。

---

**祝训练顺利！如有问题请查看故障排查章节或联系开发者。** 🚀

