# LoRA 故障诊断指南

## 问题描述
当你发现 LoRA 参数没有梯度（训练后参数没有更新）时，按照以下步骤排查。

## 自动诊断工具

我已经在代码中添加了一个 `diagnose_lora_step_by_step()` 方法，它会自动运行 7 个诊断步骤。

### 使用方法

在训练脚本中，DDP 包装后会自动调用：
```python
# 在 train_ddp() 函数中，DDP 包装后
raw_model = model.module
raw_model.diagnose_lora_step_by_step()
```

### 诊断流程

#### 【步骤 1/7】检查 LoRA 适配器是否正确应用

**检查内容：**
- `peft_config` 是否存在
- active_adapters 是否设置

**可能的问题：**
- ❌ `No peft_config found` → LoRA 未应用
  - **原因**：`get_peft_model()` 调用失败或未执行
  - **解决**：检查 PEFT 库是否正确安装，查看初始化日志

#### 【步骤 2/7】检查 LoRA 参数是否被创建

**检查内容：**
- 统计包含 'lora' 的参数数量
- 显示前 3 个 LoRA 参数的信息

**可能的问题：**
- ❌ `No LoRA parameters found` → LoRA 参数未创建
  - **原因**：LoRA 适配器配置有误，target_modules 不匹配
  - **解决**：检查 `LoraConfig` 中的 `target_modules` 是否匹配模型结构

#### 【步骤 3/7】检查 LoRA 参数的 requires_grad 标志

**检查内容：**
- 统计 `requires_grad=True` 的 LoRA 参数
- 统计被冻结的 LoRA 参数

**可能的问题：**
- ❌ `All LoRA parameters are frozen` → 参数被意外冻结
  - **原因**：在应用 LoRA 后调用了全局冻结
  - **解决**：自动调用 `force_enable_lora_gradients()`

#### 【步骤 4/7】检查模型是否处于训练模式

**检查内容：**
- `ContrastiveQwen2AudioModel.training`
- `speech_encoder.model.training`
- `text_encoder.model.training`

**可能的问题：**
- ❌ `Model is in EVAL mode` → 模型处于评估模式
  - **原因**：忘记调用 `model.train()` 或某处调用了 `model.eval()`
  - **解决**：在训练前调用 `model.train()`

#### 【步骤 5/7】测试前向传播是否经过 LoRA 层

**检查内容：**
- 创建测试音频输入
- 运行前向传播
- 检查输出是否 `requires_grad=True`

**可能的问题：**
- ❌ `Output does not require gradients` → 梯度在前向传播中被断开
  - **原因**：使用了 `torch.no_grad()` 或 `.detach()`
  - **解决**：检查编码器代码，确保使用 `torch.set_grad_enabled(self.training)`

#### 【步骤 6/7】测试反向传播是否更新 LoRA 参数

**检查内容：**
- 创建 dummy loss
- 运行反向传播
- 检查 LoRA 参数是否有梯度

**可能的问题：**
- ❌ `No LoRA parameters received gradients` → **这是最关键的问题！**
  - **可能原因：**
    1. **前向传播未经过 LoRA 层**
       - 使用了错误的编码路径
       - 编码器直接访问底层模块而绕过了 LoRA wrapper
    2. **梯度在某处被阻断**
       - 使用了 `.detach()` 或 `no_grad()`
       - 投影层的输入被 detach
    3. **使用了错误的编码策略**
       - 使用 `audio_tower` 编码但 LoRA 只在 `language_model` 上

**解决方案（按优先级）：**

##### 问题 A：LoRA 只在 language_model，但使用了 audio_tower 编码

**症状：**
```
Encoding strategy: audio_tower
LoRA params in audio_tower: 0
```

**原因：**
- LoRA 的 `target_modules` 配置为 language_model 的模块（q_proj, k_proj 等）
- 但实际编码使用了 `audio_tower`，完全绕过了 language_model
- 结果：前向传播不经过任何 LoRA 层

**解决方案 1（推荐）：修改 LoRA 配置，应用到 audio_tower**
```python
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    # 添加 audio_tower 的模块
    target_modules=[
        # language_model 模块
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj",
        # audio_tower 模块（如果需要）
        "audio_tower.*.self_attn.q_proj",
        "audio_tower.*.self_attn.k_proj",
        # ... 等等
    ],
    bias="none",
)
```

**解决方案 2：改用 full_forward 编码策略**
- 修改 `_analyze_model_structure()` 强制使用 `full_forward`
- 这样会经过 language_model，LoRA 才能生效

##### 问题 B：编码器中使用了 no_grad 或 detach

**症状：**
```
✅ Forward pass successful
   Output requires_grad: False  ← 这里是问题！
```

**检查位置：**
1. `Qwen2AudioSpeechEncoder.predict()` 方法
2. `_batch_extract_embeddings()` 方法
3. `_extract_from_audio_tower()` 方法

**确保使用：**
```python
with torch.set_grad_enabled(self.model.training):
    # 编码逻辑
```

**不要使用：**
```python
with torch.no_grad():  # ❌ 错误！
    # 编码逻辑
```

##### 问题 C：投影层输入被 detach

**检查 `encode_audio()` 方法：**
```python
def encode_audio(self, audio_inputs):
    if self.training:
        speech_embeddings = self.speech_encoder.predict(audio_inputs)
    else:
        with torch.no_grad():
            speech_embeddings = self.speech_encoder.predict(audio_inputs)
    
    # 确保不要在这里 detach
    # speech_embeddings = speech_embeddings.detach()  # ❌ 错误！
    
    return F.normalize(self.proj_speech(speech_embeddings), dim=-1)
```

#### 【步骤 7/7】检查使用的编码策略

**检查内容：**
- 编码策略（audio_tower vs full_forward）
- audio_tower 中的 LoRA 参数数量

**分析：**
- ✅ `Using audio_tower` + `LoRA params in audio_tower > 0` → 正确配置
- ⚠️  `Using audio_tower` + `LoRA params in audio_tower = 0` → **问题根源！**
  - LoRA 只在 language_model 但不会被使用
- ⚠️  `Using full_forward` → 使用 fallback 方法

## 手动诊断步骤

如果自动诊断工具不够用，可以手动逐步检查：

### 1. 检查 LoRA 配置
```python
print(model.speech_encoder.model.peft_config)
print(model.speech_encoder.model.active_adapters)
```

### 2. 统计 LoRA 参数
```python
lora_params = [
    (name, param) 
    for name, param in model.speech_encoder.model.named_parameters() 
    if 'lora' in name.lower()
]
print(f"Found {len(lora_params)} LoRA parameters")
```

### 3. 检查参数分布
```python
# 检查 LoRA 在哪些模块
audio_tower_lora = sum(1 for name, _ in lora_params if 'audio_tower' in name)
language_model_lora = sum(1 for name, _ in lora_params if 'language_model' in name)

print(f"LoRA in audio_tower: {audio_tower_lora}")
print(f"LoRA in language_model: {language_model_lora}")
```

### 4. 测试前向传播
```python
model.train()
test_audio = [np.random.randn(16000).astype(np.float32)]

with torch.set_grad_enabled(True):
    output = model.encode_audio(test_audio)
    
print(f"Output requires_grad: {output.requires_grad}")
```

### 5. 测试反向传播
```python
loss = output.sum()
loss.backward()

# 检查梯度
for name, param in lora_params[:5]:
    has_grad = param.grad is not None and param.grad.abs().sum() > 0
    print(f"{name}: {'✅ HAS GRAD' if has_grad else '❌ NO GRAD'}")
```

## 常见问题总结

| 症状 | 原因 | 解决方案 |
|------|------|----------|
| No peft_config | LoRA 未应用 | 检查 `get_peft_model()` 调用 |
| No LoRA parameters | target_modules 不匹配 | 修正 LoRA 配置 |
| LoRA frozen | requires_grad=False | 调用 `force_enable_lora_gradients()` |
| Model in eval mode | 未调用 train() | 调用 `model.train()` |
| Output no grad | 使用了 no_grad | 改用 `set_grad_enabled(training)` |
| LoRA no grad | 未经过 LoRA 层 | **最常见！见步骤 6 的详细解决方案** |
| Audio tower + LM LoRA | 编码路径不匹配 | 修改 LoRA 配置或改用 full_forward |

## 最可能的问题（基于你的日志）

根据你的日志输出：
```
LoRA params without gradients: 640
```

**诊断：**
1. LoRA 参数存在（640 个）
2. LoRA 参数 requires_grad=True
3. 但是没有梯度

**最可能的原因：**
使用了 `audio_tower` 编码策略，但 LoRA 只应用在 `language_model` 上，导致前向传播完全绕过了 LoRA 层。

**验证方法：**
运行诊断工具，查看步骤 7 的输出。

**解决方案：**
修改 LoRA 配置，将 LoRA 也应用到 audio_tower 的相应层。
