# InfiniSST Evaluation Framework

这是一个comprehensive evaluation framework，用于测试InfiniSST低延迟同声传译系统的性能，支持多用户并发测试、streamLAAL延迟计算和动态调度评估。

## 功能特性

### Task 1: 延迟评估框架 (streamLAAL)
- ✅ 字符级延迟记录和streamLAAL计算
- ✅ Simuleval兼容的instance.log格式输出
- ✅ 自动延迟数据收集和统计

### Task 2: 并发用户测试框架
- ✅ 模拟N个并发用户 (支持16/32/64用户)
- ✅ 泊松分布的用户到达时间模拟
- ✅ 50/50中英/意大利语分配
- ✅ 支持5个测试视频的轮换使用

### Task 3: 动态调度支持
- ✅ 基于等待时间和批处理大小的动态调度
- ✅ 可配置的调度参数
- ✅ 详细的调度触发事件日志记录
- ✅ A/B测试支持（静态vs动态调度对比）

## 文件结构

```
serve/
├── evaluation_framework.py    # 核心评估框架
├── run_evaluation.py         # 示例评估脚本
├── scheduler.py               # 调度器（已修改支持延迟记录和动态调度）
├── api.py                     # API服务器（已添加评估端点）
└── EVALUATION_README.md       # 本文档
```

## 快速开始

### 1. 启动服务器

```bash
# 启动InfiniSST API服务器
cd serve
python api.py --host 0.0.0.0 --port 8001
```

### 2. 运行快速测试

```bash
# 运行8用户30秒快速测试
python run_evaluation.py --test quick
```

### 3. 查看结果

测试结果将保存在 `evaluation_results/` 目录下：
- 详细JSON报告
- Simuleval格式的延迟日志
- 汇总统计报告

## 使用指南

### 基本测试

```bash
# 快速测试 (8用户, 30秒)
python run_evaluation.py --test quick

# 中等规模测试 (16用户, 2分钟)
python run_evaluation.py --test moderate

# 大规模测试 (32用户, 5分钟)
python run_evaluation.py --test large

# 动态调度对比测试
python run_evaluation.py --test dynamic_comparison
```

### 高级配置

```python
from evaluation_framework import EvaluationFramework, TestConfig

# 自定义测试配置
config = TestConfig(
    num_users=32,                    # 并发用户数
    language_split=0.5,              # 50%中文, 50%意大利语
    arrival_rate=2.0,                # 泊松到达率 (用户/秒)
    test_duration=300,               # 测试持续时间 (秒)
    server_url="http://localhost:8000",
    output_dir="my_evaluation_results",
    use_dynamic_schedule=True,       # 启用动态调度
    max_batch_size=32,
    batch_timeout=0.1,
    dynamic_wait_threshold=0.05,     # 50ms等待阈值
    dynamic_batch_min_size=2
)

framework = EvaluationFramework(config)
results = await framework.run_evaluation()
```

## API端点

### 延迟评估相关

```bash
# 启用会话的评估模式
POST /enable_evaluation_mode?session_id=<session_id>

# 获取会话延迟统计
GET /session_delays/<session_id>

# 导出会话延迟数据
POST /export_session_delays?session_id=<session_id>&filepath=<path>
```

### 动态调度相关

```bash
# 获取动态调度统计
GET /dynamic_schedule_stats

# 配置动态调度参数
POST /configure_dynamic_schedule
{
    "enabled": true,
    "wait_threshold_ms": 50.0,
    "min_batch_size": 2,
    "max_batch_size": 32
}
```

## 输出格式

### 1. 延迟统计
```json
{
    "stream_laal": 0.245,
    "total_characters": 1250,
    "segments": 85,
    "avg_delay_per_char": 0.196,
    "min_delay": 0.123,
    "max_delay": 0.567
}
```

### 2. Simuleval格式日志
```json
{"segment_id": 0, "src": "今天天气不错", "tgt": "The weather is nice today.", "tokens": ["T", "h", "e", ...], "delays": [0.5, 0.6, 0.8, ...]}
{"segment_id": 1, "src": "我喜欢这个地方", "tgt": "I like this place.", "tokens": ["I", " ", "l", ...], "delays": [0.4, 0.5, 0.7, ...]}
```

### 3. 汇总报告
```
================================================================================
InfiniSST Evaluation Framework - Summary Report
================================================================================

Test Configuration:
  - Number of users: 32
  - Language split: 50% Chinese, 50% Italian
  - Arrival rate: 2.0 users/second
  - Test duration: 300s
  - Dynamic scheduling: true

Results Summary:
  - Completed users: 30
  - Failed users: 2
  - Success rate: 93.8%

StreamLAAL Metrics:
  - Average: 0.245s
  - Median: 0.231s
  - Std Dev: 0.067s
  - Min: 0.123s
  - Max: 0.456s

Chinese Translation Results:
  - Count: 15
  - Average streamLAAL: 0.238s
  - Std Dev: 0.064s

Italian Translation Results:
  - Count: 15
  - Average streamLAAL: 0.252s
  - Std Dev: 0.071s
================================================================================
```

## 动态调度配置

### 参数说明

- `use_dynamic_schedule`: 是否启用动态调度
- `dynamic_wait_threshold`: 触发调度的最大等待时间 (秒)
- `dynamic_batch_min_size`: 最小批处理大小
- `max_batch_size`: 最大批处理大小

### 触发条件

动态调度在以下情况下触发：
1. **批处理大小达到上限**: 队列中请求数 >= max_batch_size
2. **等待时间超过阈值**: 最老请求等待时间 > dynamic_wait_threshold
3. **达到最小批处理大小**: 队列中请求数 >= dynamic_batch_min_size

## 性能指标

### StreamLAAL (Stream Latency At All Levels)
- 计算每个字符从输入到输出的延迟
- 取所有字符延迟的平均值
- 是同声传译系统的关键延迟指标

### 系统吞吐量
- 每秒处理的用户数
- 并发用户成功率
- 错误率和失败原因分析

## 故障排查

### 常见问题

1. **连接失败**
   ```bash
   # 检查服务器状态
   curl http://localhost:8000/health
   ```

2. **测试视频缺失**
   ```bash
   # 将测试视频放在以下位置之一：
   # - serve/static/test_video/
   # - ~/Downloads/
   # - /tmp/
   ```

3. **内存不足**
   ```bash
   # 减少并发用户数或测试持续时间
   # 监控GPU内存使用情况
   ```

### 调试模式

```bash
# 启用详细日志
export DEBUG_MODE=true
python run_evaluation.py --test quick
```

## 扩展功能

### 添加新的测试视频

1. 将视频文件放在 `serve/static/test_video/` 目录
2. 在 `TestConfig.test_videos` 中添加文件名
3. 确保视频格式为MP4，音频为16kHz

### 自定义延迟计算

可以在 `DelayTracker` 类中修改延迟计算逻辑：

```python
def calculate_custom_metric(self) -> float:
    # 实现自定义延迟指标
    pass
```

### 添加新的语言对

在 `EvaluationFramework` 中添加新的语言对配置：

```python
# 在_generate_user_simulations中添加
if random.random() < 0.33:
    language_pair = "English -> German"
elif random.random() < 0.66:
    language_pair = "English -> Chinese"
else:
    language_pair = "English -> Italian"
```

## 许可证

本evaluation framework作为InfiniSST项目的一部分，遵循项目的开源许可证。 