# InfiniSST Evaluation模式完整说明

## 概述
Evaluation模式已经完整实现，专门用于记录每个字符的文本和延迟，并生成符合simuleval标准的instance log文件。

## 完整流程

### 1. 评估模式启用
- **位置**: `serve/evaluation_framework.py` 第222行
- **触发**: 在音频处理开始时通过API调用启用
- **API端点**: `POST /enable_evaluation_mode?session_id={session_id}`

### 2. 延迟数据收集

#### 输入记录
- **位置**: `serve/scheduler.py` 第579行
- **触发**: 每次音频segment提交时
- **记录内容**: 音频segment的时间戳（使用占位符文本，因为没有ground truth）
```python
input_text = f"[Audio segment {len(session.source)} samples]"
session.record_input(input_text, input_timestamp)
```

#### 输出记录
- **位置**: `serve/scheduler.py` 第1034行
- **触发**: 每次生成翻译文本时
- **记录内容**: 实际的翻译文本和每个字符的延迟
```python
session.record_output(generated_text, output_timestamp, is_final=True)
```

### 3. 字符级延迟计算
- **类**: `DelayTracker` in `serve/scheduler.py`
- **功能**: 
  - 记录每个字符的输入时间戳
  - 计算每个输出字符相对于对应输入字符的延迟
  - 生成segment级别的统计信息

### 4. 数据导出

#### 方式1: 服务器端导出（推荐）
- **API端点**: `POST /export_session_delays?session_id={session_id}&filepath={path}`
- **格式**: Simuleval兼容的instance.log格式
- **内容**: 包含实际的翻译文本和精确的延迟数据

#### 方式2: 评估框架导出（备用）
- **位置**: `serve/evaluation_framework.py` 第804行
- **触发**: 当服务器端导出失败时
- **格式**: JSON格式，包含估计的延迟数据

### 5. 数据格式

#### Simuleval Instance Log格式
```json
{
  "segment_id": 0,
  "src": "[Audio segment 0 samples]",
  "tgt": "实际翻译文本",
  "tokens": ["实", "际", "翻", "译", "文", "本"],
  "delays": [0.123, 0.156, 0.189, 0.223, 0.256, 0.289],
  "input_start_time": 1234567890.123,
  "output_time": 1234567890.412,
  "average_delay": 0.206
}
```

#### 统计信息格式
```json
{
  "stream_laal": 0.206,
  "min_delay": 0.123,
  "max_delay": 0.289,
  "median_delay": 0.195,
  "std_delay": 0.045,
  "total_characters": 6,
  "segments": 1,
  "session_id": "eval_user_000_be3a4f12"
}
```

## 关键特性

### ✅ 已实现的功能
1. **字符级延迟记录**: 每个字符都有精确的延迟时间
2. **实际翻译文本**: 输出记录包含真实的翻译结果
3. **Simuleval兼容**: 输出格式符合标准评估工具要求
4. **streamLAAL计算**: 自动计算平均延迟指标
5. **多用户并发**: 支持同时评估多个用户session
6. **API集成**: 完整的REST API支持

### 🔄 输入数据处理
- 由于没有ground truth源文本，输入记录使用音频segment占位符
- 输入时间戳精确记录每个音频chunk的接收时间
- 输出文本使用实际的翻译结果

### 📊 评估指标
- **streamLAAL**: 所有字符延迟的平均值
- **字符级统计**: 最小、最大、中位数、标准差
- **segment级统计**: 每个翻译段的平均延迟

## 使用示例

### 1. 启用评估模式
```python
await self._enable_evaluation_mode()
```

### 2. 获取延迟统计
```python
# 获取基本统计
stats = session.get_delay_statistics()

# 获取详细字符数据
detailed_stats = session.get_delay_statistics(include_character_details=True)
```

### 3. 导出instance log
```python
# 服务器端导出
export_path = session.export_delays("/path/to/instance.log")

# 或通过API
POST /export_session_delays?session_id=test_session&filepath=/path/to/instance.log
```

## 测试运行

使用evaluation framework进行测试：
```bash
cd serve
python run_evaluation.py
```

测试结果将包含：
- 每个用户的instance log文件
- 详细的JSON统计报告
- 人类可读的摘要报告

## 注意事项

1. **评估模式必须在音频处理开始前启用**
2. **Session必须保持活跃状态**（通过ping机制）
3. **输入文本是占位符**，主要关注延迟计算
4. **输出文本是实际翻译结果**，可用于质量分析
5. **支持中英文和意大利语**翻译对 