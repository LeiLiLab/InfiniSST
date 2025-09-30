# LoRA 应用策略说明

## 核心观点：只需要在 audio_tower 加 LoRA

### 为什么？

#### 1. 我们的任务不需要 language_model

**当前流程（对比学习）：**
```
音频输入 
  ↓
processor (mel spectrogram)
  ↓
audio_tower/audio_encoder ← 这里是关键！提取音频特征
  ↓
mean pooling
  ↓
projection layer (trainable)
  ↓
normalized embedding
  ↓
对比损失 (与文本embedding对比)
```

**关键点：**
- ✅ 整个过程只使用了 `audio_tower` 提取特征
- ❌ 完全没有使用 `language_model`（不需要生成文本）
- ✅ 投影层已经是可训练的，足够学习映射关系

#### 2. Language Model 的作用

Language Model 在 Qwen2-Audio 中的作用是：
- 生成文本回答
- 理解音频后进行推理
- 多模态融合

但在我们的任务中：
- ❌ 不需要生成文本
- ❌ 不需要推理
- ✅ 只需要提取特征向量

#### 3. 性能和效率优势

**只在 audio_tower 加 LoRA：**
- ✅ 更少的参数（~几百万 vs 几千万）
- ✅ 更快的训练速度
- ✅ 更小的显存占用
- ✅ 更有针对性（直接优化音频理解）
- ✅ 避免过拟合（参数更少）

**在 language_model 加 LoRA：**
- ❌ 更多参数但不会被使用
- ❌ 浪费计算资源
- ❌ 增加过拟合风险

## 代码实现

### 自动策略选择

```python
if self.speech_encoder.encoding_strategy == 'audio_tower':
    # 只在音频编码器加LoRA
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
else:
    # 在language model加LoRA（fallback）
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                     "gate_proj", "up_proj", "down_proj"]
```

### LoRA 会应用到哪里？

#### 使用 audio_tower 编码时：
```
audio_tower/audio_encoder
  ├── layers[0]
  │   ├── self_attn
  │   │   ├── q_proj ← LoRA
  │   │   ├── k_proj ← LoRA
  │   │   ├── v_proj ← LoRA
  │   │   └── o_proj ← LoRA
  │   └── mlp (不加LoRA，保持原始能力)
  ├── layers[1]
  │   ├── self_attn ← LoRA
  │   └── mlp
  └── ...

language_model (完全不使用，不加LoRA)
```

#### 使用 full_forward 编码时（fallback）：
```
language_model
  ├── layers[0]
  │   ├── self_attn ← LoRA
  │   └── mlp ← LoRA
  └── ...

audio_tower (可能没有或不使用)
```

## 实验验证

### 预期结果

**只加 audio_tower LoRA：**
- ✅ LoRA 参数有梯度
- ✅ 训练正常进行
- ✅ Recall 指标提升
- ✅ 训练速度快

**加 language_model LoRA（使用 audio_tower 编码）：**
- ❌ LoRA 参数没有梯度（因为不经过这些层）
- ❌ 浪费内存和计算
- ⚠️  训练仍能进行但效率低

### 如何验证策略正确

运行诊断工具，查看步骤 7：
```
【步骤 7/7】检查使用的编码策略
✅ Using audio_tower (recommended)
   Audio tower name: audio_encoder
   LoRA params in audio_encoder: 256  ← 应该 > 0
   
如果是 0，说明策略有问题
```

## 最佳实践

### 推荐配置

```python
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,              # rank，控制LoRA参数量
    lora_alpha=32,     # scaling，通常是 r*2
    lora_dropout=0.1,  # dropout防止过拟合
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 只attention层
    bias="none",
)
```

### 参数量对比

**Audio Tower LoRA only:**
- Layers: ~12-24层
- 每层4个LoRA矩阵（q,k,v,o）
- 每个矩阵: hidden_dim × r + r × hidden_dim
- 总参数: ~几百万

**Language Model LoRA:**
- Layers: 32层
- 每层7个LoRA矩阵（q,k,v,o + gate,up,down）
- 总参数: ~几千万

**节省：** 10倍以上的参数量！

## 特殊情况

### 什么时候需要在 language_model 加 LoRA？

1. **没有 audio_tower**（模型架构不支持）
   - 只能使用 full_forward
   - 必须在 language_model 加 LoRA

2. **需要多模态理解**（不只是特征提取）
   - 需要理解音频内容
   - 需要推理和生成
   - 需要在 language_model 加 LoRA

3. **端到端微调**
   - 不使用对比学习
   - 直接优化下游任务
   - 可能需要两者都加

### 我们的场景

- ✅ 有 audio_tower
- ✅ 只需要特征提取
- ✅ 使用对比学习

**结论：只在 audio_tower 加 LoRA 是最优选择！**

## 常见问题

### Q: 投影层不够吗，为什么还要LoRA？

A: 投影层是线性变换，无法改变特征的语义表示。LoRA让audio_tower学习更适合对比学习的特征表示。

### Q: 会不会破坏预训练模型的能力？

A: LoRA的设计就是为了保留原始能力同时适应新任务。而且我们只调整attention层，保持MLP层不变。

### Q: r=16够吗？

A: 对于特征提取任务，r=16通常足够。如果需要更强的表达能力，可以尝试r=32或r=64。

### Q: 为什么不调整MLP层？

A: 
- Attention层负责特征交互，对下游任务影响更大
- MLP层负责非线性变换，保持原始能力更重要
- 只调Attention可以减少参数量和过拟合风险

## 总结

**核心原则：根据实际使用的编码路径，只在相关模块加LoRA**

- 使用 audio_tower → 只在 audio 模块加LoRA ✅
- 使用 full_forward → 在 language_model 加LoRA
- 两者都用 → 两者都加（但我们不需要）

**我们的最优策略：audio_tower + LoRA on attention only**
