# Latency Evaluation Framework 使用指南

## 🎯 概述

新的评估框架现已支持对不同`latency_multiplier`的全面测试，可以分析不同延迟策略对streamLAAL的影响。

## 🔥 新增功能

### 1. Latency配置
- `latency_range`: 支持的latency multiplier范围，如`[1, 2, 3, 4]`
- `latency_distribution`: 自定义latency分布权重，如`[0.2, 0.3, 0.3, 0.2]`

### 2. 自动分配
- 每个用户会自动分配一个latency_multiplier
- 支持均匀分布或自定义权重分布

### 3. 结果分析
- 按latency分组的详细统计
- latency与streamLAAL的相关性分析
- 多维度性能对比

## 🚀 快速开始

### 基础测试
```bash
# 运行包含所有latency的快速测试
python run_evaluation.py --test quick

# 运行专门的latency对比测试
python run_evaluation.py --test latency_comparison
```

### 自定义配置
```python
from evaluation_framework import EvaluationFramework, TestConfig

# 均匀分布latency测试
config = TestConfig(
    num_users=16,
    latency_range=[1, 2, 3, 4],        # 测试所有latency
    latency_distribution=None          # 均匀分布
)

# 加权分布latency测试
config = TestConfig(
    num_users=16,
    latency_range=[1, 2, 3, 4],
    latency_distribution=[0.1, 0.4, 0.4, 0.1]  # 偏向2x和3x
)

# 单一latency测试
config = TestConfig(
    num_users=8,
    latency_range=[3],                 # 只测试3x latency
)
```

## 📊 测试场景

### 1. 均匀分布测试
```python
python latency_evaluation_example.py  # 运行所有示例
```

测试所有latency值的均匀分布，了解整体性能分布。

### 2. 加权分布测试
偏向某些latency值，模拟真实用户的使用偏好。

### 3. 单一latency测试
专门测试特定latency值的性能特征。

### 4. 极端对比测试
对比最低(1x)和最高(4x)latency的性能差异。

### 5. 混合语言+latency测试
结合语言和latency两个维度进行综合分析。

## 📈 结果分析

### 基础指标
- 每个latency的用户数量
- 平均streamLAAL
- 标准差、最小值、最大值

### 高级分析
- Latency与streamLAAL的相关性
- 语言+latency的交叉分析
- 系统性能瓶颈识别

### 输出格式
```
📊 Latency distribution:
   - Latency 1x: 4 users, avg streamLAAL = 2.156s ± 0.234s
   - Latency 2x: 3 users, avg streamLAAL = 2.089s ± 0.156s
   - Latency 3x: 4 users, avg streamLAAL = 2.234s ± 0.198s
   - Latency 4x: 3 users, avg streamLAAL = 2.345s ± 0.267s
```

## 🔧 配置选项详解

### TestConfig参数
```python
latency_range: List[int] = [1, 2, 3, 4]
# 支持的latency multiplier值

latency_distribution: Optional[List[float]] = None
# 自定义分布权重，长度必须与latency_range相同
# 如果为None，使用均匀分布
```

### 分布示例
```python
# 均匀分布（默认）
latency_distribution=None

# 偏向低latency
latency_distribution=[0.5, 0.3, 0.15, 0.05]

# 偏向中等latency
latency_distribution=[0.1, 0.4, 0.4, 0.1]

# 偏向高latency
latency_distribution=[0.05, 0.15, 0.3, 0.5]
```

## 🎯 最佳实践

### 1. 测试策略
- 先运行快速均匀分布测试了解基线
- 然后进行单一latency的深度测试
- 最后进行混合场景的压力测试

### 2. 样本大小
- 单一latency测试：每个latency至少6-8个用户
- 混合测试：总用户数16-32个
- 对比测试：确保每组有足够样本

### 3. 结果解读
- 关注平均值和标准差
- 分析latency与performance的关系
- 考虑系统负载对结果的影响

## 📁 输出文件

### 详细结果文件
```json
{
  "summary": {
    "latency_results": {
      "1": {
        "count": 4,
        "avg_stream_laal": 2.156,
        "std_stream_laal": 0.234,
        "min_stream_laal": 1.892,
        "max_stream_laal": 2.445
      }
    }
  }
}
```

### 用户详细数据
```json
{
  "users": [
    {
      "user_id": "eval_user_001",
      "latency_multiplier": 2,
      "stream_laal": 2.089,
      "language_pair": "English -> Chinese"
    }
  ]
}
```

## 🐛 常见问题

### Q: 为什么有些latency的用户数量不均匀？
A: 使用自定义分布时会出现这种情况。检查`latency_distribution`配置。

### Q: 如何确保每个latency都有足够的样本？
A: 增加`num_users`或调整`latency_distribution`使分布更均匀。

### Q: Latency高但streamLAAL反而低是什么原因？
A: 可能是系统负载导致的。高latency可能减少了系统争抢，反而提高了效率。

## 🚀 进阶用法

### 动态latency测试
结合动态调度测试不同latency的影响：
```python
config = TestConfig(
    use_dynamic_schedule=True,
    latency_range=[1, 2, 3, 4],
    batch_timeout=0.05  # 更激进的调度
)
```

### 大规模latency压力测试
```python
config = TestConfig(
    num_users=64,
    test_duration=600,  # 10分钟
    latency_range=[1, 2, 3, 4],
    arrival_rate=3.0    # 高并发
)
```

---

更多信息请参考：
- `evaluation_framework.py` - 核心框架代码
- `run_evaluation.py` - 预定义测试场景
- `latency_evaluation_example.py` - 详细使用示例 