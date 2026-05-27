# GPU分配问题和解决方案

## 问题描述

在使用SLURM的job array运行多GPU并行任务时，可能遇到两个问题：

1. **CUDA out of memory**：多个任务被分配到同一个GPU上
2. **GPU已被占用**：某些GPU已经被其他进程占用（不在SLURM管理下），但SLURM仍然分配任务到这些GPU

### 错误信息示例

```
CUDA out of memory. Tried to allocate 20.00 MiB. 
GPU 0 has a total capacity of 47.53 GiB of which 5.88 MiB is free. 
Process 3084067 has 14.63 GiB memory in use. 
Process 3681802 has 15.47 GiB memory in use.
```

这表明GPU 0上有多个进程（3084067和3681802）都在使用内存。

### nvidia-smi显示GPU被占用

```
|   2  NVIDIA RTX A6000  | 77C | 298W | 31799MiB / 49140MiB | 100% |
```

GPU 2已经被占用，但SLURM可能不知道（进程不是通过SLURM启动的）。

## 根本原因

### 1. SLURM分配 vs PyTorch使用

- **SLURM分配**：通过`#SBATCH --gres=gpu:1`为每个任务分配1个GPU
- **PyTorch默认行为**：如果不设置`CUDA_VISIBLE_DEVICES`，PyTorch可以看到所有GPU
- **冲突**：即使SLURM分配了GPU 2，代码中的`cuda:0`仍然指向物理GPU 0

### 2. 环境变量设置时机错误

在Python代码中设置`CUDA_VISIBLE_DEVICES`太晚了：

```python
# ❌ 错误：PyTorch已经初始化，设置无效
import torch  # PyTorch初始化时已经检测到所有GPU
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)  # 太晚了！
```

## 解决方案

### ✅ 方案1：在Shell脚本中提前设置CUDA_VISIBLE_DEVICES

在运行Python之前，在shell脚本中设置`CUDA_VISIBLE_DEVICES`：

```bash
#!/bin/bash
#SBATCH --array=0-3
#SBATCH --gres=gpu:1

# Get SLURM array task ID
GPU_ID=$SLURM_ARRAY_TASK_ID

# 方法1: 使用SLURM自动分配的GPU
if [ -n "$SLURM_JOB_GPUS" ]; then
    export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS
    echo "Using SLURM allocated GPU: $SLURM_JOB_GPUS"
fi

# 方法2: 手动映射（如果SLURM_JOB_GPUS不可用）
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    echo "Manually setting CUDA_VISIBLE_DEVICES=$GPU_ID"
fi

# 现在运行Python（此时CUDA_VISIBLE_DEVICES已经设置好）
python your_script.py --gpu-id $GPU_ID
```

### ✅ 方案2：跳过已占用的GPU（推荐）

如果某些GPU已经被其他进程占用（不在SLURM管理下），使用GPU映射跳过它们：

```bash
#!/bin/bash
#SBATCH --array=0-3
#SBATCH --gres=gpu:1

# GPU映射：跳过GPU 2（已被占用）
# 任务ID → 物理GPU ID
# Task 0 → GPU 0, Task 1 → GPU 1, Task 2 → GPU 3, Task 3 → GPU 4
GPU_MAP=(0 1 3 4)

# Get array task ID (0, 1, 2, 3)
ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID

# Map to physical GPU
PHYSICAL_GPU_ID=${GPU_MAP[$ARRAY_TASK_ID]}

# Set CUDA_VISIBLE_DEVICES to the mapped GPU
export CUDA_VISIBLE_DEVICES=$PHYSICAL_GPU_ID

echo "Task $ARRAY_TASK_ID → Physical GPU $PHYSICAL_GPU_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Run Python (use ARRAY_TASK_ID for data sharding logic)
python your_script.py --gpu-id $ARRAY_TASK_ID --total-gpus 4
```

### GPU映射说明

当GPU 2被占用时：

| Array Task ID | Physical GPU ID | 用途 |
|---------------|-----------------|------|
| 0 | 0 | 处理数据分片0/4 |
| 1 | 1 | 处理数据分片1/4 |
| 2 | 3 | 处理数据分片2/4（跳过GPU 2）|
| 3 | 4 | 处理数据分片3/4 |

**关键点**：
- **物理GPU ID**：实际使用的GPU（用于`CUDA_VISIBLE_DEVICES`）
- **Array Task ID**：用于数据分片逻辑（保持0-3，确保数据均匀分配）

### 关键点

1. **在Python运行前设置**：必须在任何Python/PyTorch代码运行前设置`CUDA_VISIBLE_DEVICES`
2. **使用SLURM变量**：优先使用`$SLURM_JOB_GPUS`，它包含SLURM实际分配的GPU ID
3. **映射到cuda:0**：设置后，被选中的GPU在PyTorch中会被映射为`cuda:0`

## CUDA_VISIBLE_DEVICES工作原理

```bash
# 假设系统有4个GPU: 0, 1, 2, 3

# Task 0: CUDA_VISIBLE_DEVICES=0
# PyTorch看到: cuda:0 → 物理GPU 0

# Task 1: CUDA_VISIBLE_DEVICES=1  
# PyTorch看到: cuda:0 → 物理GPU 1

# Task 2: CUDA_VISIBLE_DEVICES=2
# PyTorch看到: cuda:0 → 物理GPU 2

# Task 3: CUDA_VISIBLE_DEVICES=3
# PyTorch看到: cuda:0 → 物理GPU 3
```

这样每个任务都使用`cuda:0`，但实际上它们映射到不同的物理GPU。

## 代码修改

### Shell脚本 (`run_term_map_construction.sh`)

**方案1：基础版（不跳过GPU）**

```bash
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    GPU_ID=0
    TOTAL_GPUS=1
else
    GPU_ID=$SLURM_ARRAY_TASK_ID
    TOTAL_GPUS=4
    
    # 在Python运行前设置CUDA_VISIBLE_DEVICES
    if [ -n "$SLURM_JOB_GPUS" ]; then
        export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS
    else
        export CUDA_VISIBLE_DEVICES=$GPU_ID
    fi
    
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi
```

**方案2：跳过GPU 2（当前使用）**

```bash
# GPU映射：跳过GPU 2（已被占用）
GPU_MAP=(0 1 3 4)

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    PHYSICAL_GPU_ID=0
    GPU_ID=0
    TOTAL_GPUS=1
else
    ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID
    TOTAL_GPUS=4
    
    # 映射到物理GPU
    PHYSICAL_GPU_ID=${GPU_MAP[$ARRAY_TASK_ID]}
    GPU_ID=$ARRAY_TASK_ID  # 用于数据分片
    
    # 设置CUDA_VISIBLE_DEVICES
    export CUDA_VISIBLE_DEVICES=$PHYSICAL_GPU_ID
    
    echo "Task $ARRAY_TASK_ID → Physical GPU $PHYSICAL_GPU_ID"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi
```

### Python脚本修改

```python
# ✅ 不要在Python中设置CUDA_VISIBLE_DEVICES
# 只需记录它已经被设置了
if total_gpus > 1:
    logger.info(f"Multi-GPU mode: GPU {gpu_id}/{total_gpus}")
    logger.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    
# 使用cuda:0（会自动映射到正确的物理GPU）
RAG_DEVICE = "cuda:0"
```

## 验证方法

### 1. 检查GPU使用情况

```bash
# 在运行任务时，使用nvidia-smi监控
watch -n 1 nvidia-smi
```

**使用GPU映射时，应该看到：**
- GPU 0: 1个进程（Task 0）
- GPU 1: 1个进程（Task 1）
- GPU 2: 原有进程（被跳过）
- GPU 3: 1个进程（Task 2）
- GPU 4: 1个进程（Task 3）

### 2. 检查日志

查看每个任务的日志文件：

```bash
tail -f logs/*_0_*.err  # Task 0 (使用GPU 0)
tail -f logs/*_1_*.err  # Task 1 (使用GPU 1)
tail -f logs/*_2_*.err  # Task 2 (使用GPU 3, 跳过GPU 2)
tail -f logs/*_3_*.err  # Task 3 (使用GPU 4)
```

应该看到：
- 每个任务都显示`CUDA_VISIBLE_DEVICES`设置正确
- Task 2显示"Physical GPU 3"（不是2）
- 没有"CUDA out of memory"错误
- 每个任务处理不同的数据分片

## SLURM相关环境变量

| 变量 | 说明 | 示例 |
|------|------|------|
| `SLURM_ARRAY_TASK_ID` | 当前任务在array中的索引 | `0`, `1`, `2`, `3` |
| `SLURM_JOB_GPUS` | SLURM分配的GPU ID（可能不存在） | `0`, `1`, `2`, `3` |
| `SLURM_STEP_GPUS` | Step分配的GPU ID | `0`, `1`, `2`, `3` |
| `CUDA_VISIBLE_DEVICES` | 进程可见的GPU（手动设置） | `0`, `1`, `2`, `3` |

## 常见错误

### ❌ 错误1：在Python中设置

```python
import torch  # 已经太晚了
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```

### ❌ 错误2：不设置CUDA_VISIBLE_DEVICES

```bash
# 没有设置，所有任务都能看到所有GPU
python script.py --gpu-id 0
python script.py --gpu-id 1
```

### ❌ 错误3：硬编码device而不使用环境变量

```python
# 硬编码，不受CUDA_VISIBLE_DEVICES影响
device = torch.device("cuda:2")
```

应该使用：

```python
# ✅ 总是使用cuda:0，让CUDA_VISIBLE_DEVICES处理映射
device = torch.device("cuda:0")
```

## 参考资源

- [SLURM GPU Scheduling](https://slurm.schedmd.com/gres.html)
- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [CUDA Environment Variables](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars)

