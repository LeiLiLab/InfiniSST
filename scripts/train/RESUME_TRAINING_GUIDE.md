# 断点续传训练指南

## 📋 功能说明

训练脚本 `stage1_gigaspeech_zh_norm0_qwen_rope_rag.sh` 现在支持**自动断点续传**功能。

### 特性
- ✅ **自动检测** checkpoint 并从断点继续
- ✅ **显示训练进度**（Epoch, Global Step）
- ✅ **保存详细信息**到 `checkpoint_info.txt`
- ✅ **支持手动清理**从头开始训练

---

## 🚀 使用方法

### 1. 正常训练（自动断点续传）

```bash
sbatch scripts/train/stage1_gigaspeech_zh_norm0_qwen_rope_rag.sh
```

**效果**：
- 如果存在 checkpoint → 自动从断点继续
- 如果不存在 checkpoint → 从头开始训练

### 2. 从头开始训练（清理旧 checkpoint）

```bash
CLEAN_START=1 sbatch scripts/train/stage1_gigaspeech_zh_norm0_qwen_rope_rag.sh
```

**效果**：
- 删除旧的 checkpoint 目录
- 从 Epoch 0, Step 0 重新开始

### 3. 查看 Checkpoint 信息

#### 方法 A：查看保存的信息文件

```bash
cat /mnt/aries/data6/jiaxuanluo/demo/en-zh/runs/stage1_M=12_ls-cv-vp_norm0_qwen_rope_vv2_gigaspeech_v2/checkpoint_info.txt
```

**输出示例**：
```
============================================================
InfiniSST Training Checkpoint Info
============================================================
保存时间: 2025-10-06 08:30:15
Epoch: 0
Global Step: 5000
Checkpoint Path: /path/to/last.ckpt/checkpoint
============================================================
```

#### 方法 B：使用检查脚本

```bash
# 使用默认路径
./scripts/train/check_checkpoint.sh

# 或指定路径
./scripts/train/check_checkpoint.sh /path/to/checkpoint/dir
```

**输出示例**：
```
==========================================
检查 Checkpoint: /path/to/checkpoint/dir
==========================================

=== Checkpoint 信息 ===
...
=== Lightning Checkpoint (2.x格式) ===
路径: /path/to/last.ckpt/checkpoint
大小: 3.5G

详细信息:
  Epoch: 0
  Global Step: 5000
  Checkpoint Keys: ['epoch', 'global_step', 'state_dict', ...]
  Optimizer States: 1 optimizer(s)
  LR Schedulers: 1 scheduler(s)

=== 所有 Checkpoint 文件 ===
  - last.ckpt/checkpoint (3.5G)
  - epoch=0-step=1000/checkpoint (3.5G)
  - epoch=0-step=2000/checkpoint (3.5G)

=== 目录统计 ===
总大小: 10.5G
文件数: 15
==========================================
```

---

## 📁 Checkpoint 文件结构

```
save_dir/
├── last.ckpt/
│   ├── checkpoint              # Lightning 2.x checkpoint (主要文件)
│   ├── pytorch_model.bin       # 提取的模型权重
│   └── .latest                 # Lightning 元数据
├── epoch=0-step=1000/
│   └── checkpoint              # Step 1000 的checkpoint
├── epoch=0-step=2000/
│   └── checkpoint              # Step 2000 的checkpoint
├── checkpoint_info.txt         # 训练信息摘要
└── ...
```

---

## 🔍 训练流程说明

### 启动时
1. **检查 checkpoint**
   ```
   [INFO] ==========================================
   [INFO] 发现 checkpoint，将从断点继续训练
   [INFO] Checkpoint: /path/to/last.ckpt/checkpoint
   [INFO] Epoch: 0, Global Step: 5000
   [INFO] ==========================================
   ```

2. **Lightning 自动加载**
   - Lightning 检测到 `last.ckpt` 后自动恢复：
     - 模型权重
     - Optimizer 状态
     - LR Scheduler 状态
     - 训练步数和 Epoch

3. **继续训练**
   - 从 `Global Step 5001` 开始
   - 保持原有的学习率和优化器状态

### 训练完成后
1. **提取模型权重**
   ```
   [INFO] ✓ 已提取模型权重
   ```

2. **保存训练信息**
   ```
   [INFO] ✓ 已保存checkpoint信息到 checkpoint_info.txt
   ```

3. **导出模型**
   ```
   [INFO] ✓ 已导出模型到 /path/to/pytorch_model.bin
   ```

---

## ⚙️ 配置说明

### Checkpoint 保存频率

在训练脚本中配置：

```bash
--save_step 1000    # 每 1000 步保存一次
--save_dir ${save_path}
```

### 验证频率

```bash
--eval_step 676680  # 禁用中间验证（只在epoch结束时验证）
```

修改为 `--eval_step 1000` 可启用每 1000 步验证。

---

## 🐛 故障排查

### 问题 1：训练没有从checkpoint恢复

**检查**：
```bash
ls -lh /path/to/save_dir/last.ckpt/
```

**可能原因**：
- Checkpoint 文件损坏
- 路径不正确
- Lightning 版本不兼容

**解决**：
```bash
# 检查checkpoint内容
python3 -c "
import torch
ckpt = torch.load('/path/to/last.ckpt/checkpoint', map_location='cpu')
print(f'Epoch: {ckpt.get(\"epoch\")}')
print(f'Step: {ckpt.get(\"global_step\")}')
"
```

### 问题 2：想从头开始但checkpoint还在

**解决**：
```bash
# 方法1：使用 CLEAN_START
CLEAN_START=1 sbatch scripts/train/xxx.sh

# 方法2：手动删除
rm -rf /path/to/save_dir
```

### 问题 3：Checkpoint 占用空间过大

**说明**：
- 每个 checkpoint ≈ 3-4GB（包含模型、optimizer、scheduler）
- 保存间隔越小，占用空间越多

**优化**：
```bash
# 增大保存间隔
--save_step 2000  # 从 1000 改为 2000

# 或定期清理旧checkpoint（保留最新的）
find ${save_path} -name "epoch=*-step=*" -type d | sort -V | head -n -3 | xargs rm -rf
```

---

## 📊 监控训练进度

### 查看实时日志

```bash
# 查看标准输出
tail -f train_infinisst_JOBID.out

# 查看错误日志
tail -f train_infinisst_JOBID.err
```

### 查看 WandB

访问: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/infinisst

---

## 💡 最佳实践

### 1. 定期检查 Checkpoint

训练过程中，定期运行：
```bash
./scripts/train/check_checkpoint.sh
```

### 2. 备份关键 Checkpoint

```bash
# 备份当前最好的checkpoint
cp -r ${save_path}/last.ckpt ${save_path}/backup_step_5000
```

### 3. 监控训练状态

每次重启训练前，查看 `checkpoint_info.txt` 确认进度。

### 4. 处理 NCCL 超时

如果遇到 NCCL 超时（exit code 137）：
- ✅ 检查 checkpoint 是否已保存
- ✅ 直接重新提交任务（会自动从最新checkpoint恢复）
- ✅ 如果持续失败，考虑减少 GPU 数量或调整 batch size

---

## 🎯 常见场景

### 场景 1：Modal/Slurm 超时中断

```bash
# 问题：训练运行23小时后被终止
# 解决：重新提交即可，会自动从最新checkpoint继续

sbatch scripts/train/xxx.sh
```

### 场景 2：手动停止训练后继续

```bash
# 1. 停止任务
scancel JOB_ID

# 2. 检查checkpoint
./scripts/train/check_checkpoint.sh

# 3. 重新提交（自动从断点继续）
sbatch scripts/train/xxx.sh
```

### 场景 3：发现配置错误需要重新训练

```bash
# 1. 修改脚本配置
vim scripts/train/xxx.sh

# 2. 清理旧checkpoint并重新开始
CLEAN_START=1 sbatch scripts/train/xxx.sh
```

---

## 📝 注意事项

### ⚠️ 重要
1. **不要手动删除 `last.ckpt`** - 这是 Lightning 用于resume的主要checkpoint
2. **`checkpoint_info.txt` 仅供查看** - 不影响训练，删除也可以
3. **Resume 会保持原有的训练配置** - 如果需要改配置（如learning rate），建议从头训练

### ✅ 安全操作
- 多次提交同一个任务 → 自动续传，安全
- 查看 checkpoint 信息 → 只读，安全
- 使用 `CLEAN_START=1` → 会删除旧数据，请确认后操作

### ❌ 危险操作
- `rm -rf ${save_path}` 在训练过程中 → 会丢失进度
- 手动修改 checkpoint 文件 → 可能导致无法加载
- 同时运行多个训练到同一目录 → checkpoint 冲突

---

## 📞 支持

如有问题，查看：
1. 训练日志: `train_infinisst_*.err` 和 `train_infinisst_*.out`
2. Checkpoint 信息: `checkpoint_info.txt`
3. 使用检查脚本: `./scripts/train/check_checkpoint.sh`

**更新时间**: 2025-10-06

