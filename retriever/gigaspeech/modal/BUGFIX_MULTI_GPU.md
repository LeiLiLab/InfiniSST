# Bug修复说明 - 多GPU并发加载模型错误

## 问题

在运行多GPU挖掘时遇到两个错误：

1. **模型文件找不到**：
   ```
   OSError: Qwen/Qwen2-Audio-7B-Instruct does not appear to have a file named model-00001-of-00005.safetensors
   ```

2. **Meta tensor错误**：
   ```
   Cannot copy out of meta tensor; no data!
   ```

## 根本原因

多个GPU线程同时并发加载模型时，由于Hugging Face的缓存机制和模型加载逻辑，导致：
- 文件访问冲突
- 模型权重未正确加载到GPU
- Meta tensor状态不一致

## 解决方案

### 1. 添加模型加载锁

使用 `threading.Lock()` 确保模型串行加载：

```python
model_load_lock = threading.Lock()

def mine_on_gpu(..., model_load_lock):
    with model_load_lock:
        # 模型加载代码
        speech_encoder = Qwen2AudioSpeechEncoder(...)
        text_encoder = Qwen2AudioTextEncoder(...)
        model = ContrastiveQwen2AudioModel(...)
```

### 2. 添加线程启动延迟

在启动线程时添加小延迟，让模型加载更有序：

```python
for gpu_id in range(num_gpus):
    t = threading.Thread(...)
    t.start()
    time.sleep(0.5)  # 500ms延迟
```

### 3. 改进日志输出

添加更详细的日志，便于追踪模型加载过程：
- `[GPU X] Waiting to initialize model...`
- `[GPU X] Initializing model (locked)...`
- `[GPU X] Model initialization complete`

## 修改的文件

- `mine_hard_negatives_multi_gpu.py`
  - 添加 `model_load_lock` 参数
  - 使用锁保护模型加载代码块
  - 添加启动延迟
  - 改进日志输出

## 效果

- ✅ 避免模型文件访问冲突
- ✅ 确保模型正确加载到各个GPU
- ✅ 串行加载不会显著影响总体性能（加载时间远小于推理时间）
- ✅ 推理阶段仍然是并行的

## 性能影响

- **模型加载**: 串行，总共增加约 6 × 30秒 = 3分钟
- **数据处理**: 并行，6个GPU同时处理，节省约5倍时间
- **总体**: 对于大规模数据集，加载开销可以忽略不计

## 使用方法

无需改变使用方式，修复已集成到脚本中：

```bash
sbatch run_hard_negative_training.sh
```

脚本会自动：
1. 串行加载模型到各个GPU
2. 并行处理数据
3. 合并结果

## 验证

运行后应该看到如下日志：

```
[INFO] Loading models sequentially to avoid conflicts...
[GPU 0] Waiting to initialize model...
[GPU 0] Initializing model (locked)...
[GPU 0] ✅ Model weights loaded
[GPU 0] Model initialization complete
[GPU 1] Waiting to initialize model...
[GPU 1] Initializing model (locked)...
...
```

## 如果仍然遇到问题

1. **减少GPU数量**:
   ```bash
   MINE_NUM_GPUS=2  # 从6降到2
   ```

2. **减少batch size**:
   ```bash
   MINE_BATCH_SIZE=64  # 从128降到64
   ```

3. **设置环境变量**:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   export HF_HOME=/path/to/large/cache  # 使用大容量缓存目录
   ```

4. **检查显存**:
   ```bash
   nvidia-smi
   # 确保每个GPU有至少30GB可用显存
   ```










