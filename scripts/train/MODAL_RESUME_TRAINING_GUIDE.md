# Modal 断点续训指南

## 问题背景
第一个 epoch 需要超过 24 小时才能完成，导致 Modal 超时且没有保存任何 checkpoint。

## 解决方案

### 1. 自动保护机制（已实施）

#### a) 时间限制保护
- **位置**: `train/main.py`
- **配置**: `max_time="00:23:30:00"` (格式: DD:HH:MM:SS)
- **效果**: 训练运行 23.5 小时后自动停止并保存 checkpoint，留 30 分钟缓冲避免 Modal 超时

#### b) 定期保存 checkpoint
- **位置**: `scripts/train/modal_stage1_gigaspeech_rag.py`
- **配置**: `save_step: int = 2000`
- **效果**: 每 2000 步保存一次 checkpoint
- **建议**: 如果训练仍然太慢无法达到第一个 save_step，可以减小这个值（如 500 或更小）

#### c) Epoch 结束保存
- **位置**: `train/main.py` 
- **配置**: `save_on_train_epoch_end=True`
- **效果**: 每个 epoch 结束时也会保存 checkpoint

### 2. 使用断点续训

#### 首次训练
```bash
modal run scripts/train/modal_stage1_gigaspeech_rag.py
```

默认情况下 `resume_training=True`，训练会自动寻找并恢复最新的 checkpoint。

#### 强制从头开始（清空之前的输出）
```bash
modal run scripts/train/modal_stage1_gigaspeech_rag.py --no-resume-training
```

注意：需要在脚本中添加 `--no-resume-training` 支持，或直接修改参数。

### 3. Checkpoint 查找逻辑

训练会按以下顺序查找 checkpoint：

1. **优先**: `{save_path}/last.ckpt/checkpoint`
2. **备用**: 最新的 `epoch=*` 或 `step=*` 目录中的 checkpoint
3. **无 checkpoint**: 从头开始训练

### 4. 多次运行策略

#### 场景：一个 epoch 需要 72 小时（3 × 24h）

**第一次运行**（0-23.5h）:
```bash
modal run scripts/train/modal_stage1_gigaspeech_rag.py
```
- 自动在 23.5 小时停止
- 保存 checkpoint 到 `/output/en-zh/runs/{run_name}/last.ckpt/`

**第二次运行**（23.5-47h）:
```bash
modal run scripts/train/modal_stage1_gigaspeech_rag.py
```
- 自动检测到 `last.ckpt/checkpoint`
- 从上次停止的地方继续训练
- 再次在 23.5 小时停止并保存

**第三次运行**（47-72h）:
```bash
modal run scripts/train/modal_stage1_gigaspeech_rag.py
```
- 继续训练直到 epoch 完成

### 5. 监控训练进度

检查 Modal 日志，查找以下关键信息：

```
[INFO] Resuming training from checkpoint: /output/.../last.ckpt/checkpoint
[INFO] Epoch X/Y, Step Z
```

### 6. Volume 持久化

所有 checkpoint 自动保存到 Modal Volume：
- **Volume 名称**: `infinisst-outputs`
- **挂载路径**: `/output`
- **Checkpoint 路径**: `/output/en-zh/runs/{run_name}/last.ckpt/`

Volume 数据在 Modal 函数运行之间持久化，所以可以安全地多次运行。

### 7. 调优建议

如果训练仍然很慢，可以考虑：

1. **进一步减小 save_step**:
   ```python
   save_step: int = 200  # 或更小
   ```

2. **增加 Modal timeout**（会增加成本）:
   ```python
   timeout=86400*2  # 48 小时
   max_time="47:30:00"  # 相应调整
   ```

3. **减少 batch size** 加快迭代:
   ```python
   train_bsz: int = 3600  # 从 7200 减半
   grad_acc_steps: int = 2  # 补偿 batch size 减少
   ```

4. **只训练数据子集**（测试用）:
   在 dataset 加载时添加限制，验证整个流程正常工作。

### 8. 故障排查

#### 问题：找不到 checkpoint
检查 Volume 内容：
```python
@app.function(volumes={"/output": output_volume})
def check_checkpoints():
    import os
    for root, dirs, files in os.walk("/output"):
        print(f"{root}: {dirs[:5]}, {files[:5]}")
```

#### 问题：checkpoint 损坏
Lightning 会保存多个 checkpoint（`last.ckpt`, `epoch=X`, `step=Y`），如果一个损坏，可以尝试使用其他的。

#### 问题：训练没有进度
检查：
- DataLoader 是否正常加载数据
- GPU 利用率是否正常
- 是否有 NCCL 超时错误

### 9. 估算训练时间

根据前几步的速度估算：
```
每步耗时 = (当前运行时间) / (当前步数)
总步数估算 = (数据集大小) / (有效 batch size)
总时间估算 = 每步耗时 × 总步数
需要运行次数 = ceil(总时间估算 / 23.5小时)
```

### 10. 完成训练后

训练完成后，checkpoint 在：
```
/output/en-zh/runs/{run_name}/last.ckpt/pytorch_model.bin
```

可以使用以下命令导出：
```python
# 在 Modal 函数或本地使用
ckpt = torch.load('last.ckpt/checkpoint', map_location='cpu')
torch.save(ckpt['state_dict'], 'pytorch_model.bin')
```

---

**最后更新**: 2025-10-03

