# InfiniSST 整合系统使用指南

## 概述

本项目已成功将 `origin/flashinfer` 分支与 `electron` 分支合并，实现了基于 ORCA 思路的多请求并发推理服务系统。整合的系统包含以下核心组件：

### 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Electron 前端界面                          │
├─────────────────────────────────────────────────────────────┤
│                    API 服务层                                │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              HTTP REST API                              │ │
│  │    /translate  /health  /stats  /session              │ │
│  └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    调度器 (Scheduler)                        │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  FCFS Queue Management (PREFILL vs DECODE)             │ │
│  │  Multi-GPU Support & Session Management                │ │
│  └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                  推理引擎 (Inference Engine)                  │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  InfiniSST Faster + FlashInfer Backend                 │ │
│  │  Batch Processing & Token-by-Token Streaming           │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 📁 新增文件结构

```
InfiniSST/
├── serve/
│   ├── scheduler.py                 # ✨ 新增：ORCA风格调度器
│   ├── api_with_scheduler.py        # ✨ 新增：整合版API服务
│   ├── inference_engine.py         # ✨ 新增：推理引擎封装
│   └── start_integrated_server.py  # ✨ 新增：整合启动脚本
├── test_integrated_system.py       # ✨ 新增：系统测试脚本
└── README_Integration.md            # ✨ 新增：本文档
```

## 🚀 快速开始

### 1. 环境准备

确保已安装以下依赖：

```bash
# Python 依赖
pip install flask flask-cors torch numpy requests

# Electron 依赖 (如需要前端界面)
npm install
```

### 2. 启动整合服务器

#### 方式一：使用模拟推理（推荐用于开发和测试）

```bash
# 启动服务器，使用模拟推理引擎
python serve/start_integrated_server.py --host 0.0.0.0 --port 8000

# 或者使用多GPU配置
python serve/start_integrated_server.py \
    --gpus "0,1" \
    --languages "English -> Chinese,English -> German"
```

#### 方式二：使用实际模型推理

```bash
# 启动服务器，加载实际模型
python serve/start_integrated_server.py --load-models

# 注意：需要配置正确的模型路径和参数
```

### 3. 验证系统运行

```bash
# 运行完整测试套件
python test_integrated_system.py

# 或运行特定测试
python test_integrated_system.py --test health
python test_integrated_system.py --test concurrent
```

### 4. 启动前端界面

```bash
# 重新打包Electron应用（包含新的拖拽功能）
npm run dist

# 或直接运行开发版本
npm run electron-dev
```

## 🔧 核心功能

### 1. ORCA风格调度策略

- **FCFS队列**：维护每个GPU的独立prefill和decode队列
- **优先级调度**：PREFILL请求优先于DECODE请求
- **批处理解耦**：同一批次只包含相同阶段的请求
- **多GPU支持**：不同语言对可分配到不同GPU

### 2. 多请求并发处理

- **最大批处理大小**：32个请求
- **超时机制**：批处理等待超时（默认0.1秒）
- **会话管理**：维护用户会话状态和KV缓存
- **错误处理**：请求失败时的降级和重试机制

### 3. Token-by-Token流式推理

- **增量处理**：支持音频流的增量输入
- **状态保持**：维护speech cache和LLM cache
- **实时响应**：token级别的流式输出

## 📡 API 接口

### 健康检查
```http
GET /health
```

### 加载模型
```http
POST /load_model
Content-Type: application/json

{
    "gpu_id": 0,
    "language_pair": "English -> Chinese"
}
```

### 翻译请求
```http
POST /translate
Content-Type: application/json

{
    "user_id": "user123",
    "language_pair": "English -> Chinese",
    "audio_data": [0.1, 0.2, 0.3, ...],
    "is_final": false,
    "max_new_tokens": 20
}
```

### 会话管理
```http
GET /session/{user_id}/{language_id}
POST /session/{user_id}/{language_id}/reset
```

### 系统统计
```http
GET /stats
```

## 🔨 开发和调试

### 查看日志

```bash
# 实时查看日志
tail -f infinisst_integrated.log

# 查看特定组件日志
grep "scheduler" infinisst_integrated.log
grep "inference" infinisst_integrated.log
```

### 性能监控

```bash
# 查看系统统计信息
curl http://localhost:8000/stats | jq

# 监控队列状态
watch -n 1 'curl -s http://localhost:8000/stats | jq .scheduler_stats.queue_sizes'
```

### 调试模式

```bash
# 启动调试模式
python serve/start_integrated_server.py --debug

# 使用具体的GPU和语言配置
python serve/start_integrated_server.py \
    --debug \
    --gpus "0" \
    --languages "English -> Chinese" \
    --max-batch-size 16 \
    --batch-timeout 0.05
```

## 🧪 测试和验证

### 1. 基础功能测试

```bash
# 健康检查
python test_integrated_system.py --test health

# 单个翻译请求
python test_integrated_system.py --test single

# 并发请求测试
python test_integrated_system.py --test concurrent
```

### 2. 压力测试

```bash
# 并发翻译测试（自定义请求数量）
python -c "
from test_integrated_system import InfiniSSTSystemTester
tester = InfiniSSTSystemTester()
tester.test_concurrent_translations(num_requests=50)
"
```

### 3. 流式处理测试

```bash
# 测试流式翻译
python test_integrated_system.py --test stream
```

## 🛠️ 接下来需要完成的工作

基于当前的整合状态，以下是您需要重点关注的事项：

### 1. 模型服务集成 🔴 高优先级

```python
# 在 serve/inference_engine.py 中的 _create_model_args 函数
# 需要替换为实际的模型参数
def _create_model_args(self, gpu_id: int) -> Any:
    # TODO: 替换为实际的 InfiniSST Faster 参数
    class RealModelArgs:
        def __init__(self):
            self.model_name = "your_model_path"  # 实际模型路径
            self.w2v2_path = "your_w2v2_path"   # W2V2路径
            self.state_dict_path = "your_state_dict_path"  # 状态字典路径
            # 添加其他必要参数...
    
    return RealModelArgs()
```

### 2. FlashInfer KV缓存格式修复 🔴 高优先级

当前问题："page attention table 后格式报错（KV 缓存 format 不对）"

**建议解决方案：**
1. 检查 `agents/infinisst_faster.py` 中的 `init_paged_kv_cache` 调用
2. 验证 `blocksize` 和 `max_cache_size` 参数配置
3. 确认 `SpeechCache` 和 `LLMCache` 的数据格式

### 3. 前端集成 🟡 中优先级

```javascript
// 在 serve/static/index.html 中
// 修改翻译请求的目标URL
async function performTranslation() {
    const response = await fetch('/translate', {  // 新的API端点
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            user_id: generateUserId(),
            language_pair: getSelectedLanguagePair(),
            audio_data: audioData,
            is_final: isStreamFinished,
            max_new_tokens: 20
        })
    });
}
```

### 4. 配置和部署 🟡 中优先级

#### 创建配置文件

```yaml
# config/production.yaml
gpu_language_map:
  0: "English -> Chinese"
  1: "English -> German"

model_configs:
  gpu_0:
    model_name: "/path/to/model"
    w2v2_path: "/path/to/w2v2"
    state_dict_path: "/path/to/state_dict"
    
scheduler:
  max_batch_size: 32
  batch_timeout: 0.1
  session_timeout: 300

server:
  host: "0.0.0.0"
  port: 8000
```

#### 启动脚本

```bash
#!/bin/bash
# start.sh

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=$PWD:$PYTHONPATH

python serve/start_integrated_server.py \
    --config config/production.yaml \
    --load-models
```

## 💡 使用建议

### 1. 开发阶段
- 使用模拟推理模式进行快速迭代
- 重点测试API接口和调度逻辑
- 验证前端和后端的数据格式匹配

### 2. 模型集成阶段
- 先在单GPU上测试模型加载和推理
- 逐步启用batch处理和并发
- 监控内存使用和性能指标

### 3. 生产部署阶段
- 配置实际的GPU和模型路径
- 启用完整的错误处理和日志
- 实施监控和告警机制

## 🐛 常见问题

### Q1: KV缓存格式错误
**A**: 检查 `model/flashinfer/engine.py` 中的page table初始化参数，确保与模型期望的格式匹配。

### Q2: 调度器无响应
**A**: 检查GPU设备可用性，验证 `CUDA_VISIBLE_DEVICES` 环境变量。

### Q3: 前端连接失败
**A**: 确认API服务器已启动，检查端口和防火墙设置。

## 📞 技术支持

如果遇到问题，请：

1. 查看 `infinisst_integrated.log` 日志文件
2. 运行诊断测试：`python test_integrated_system.py --test health`
3. 检查系统状态：`curl http://localhost:8000/stats`

---

🎉 **恭喜！** 您的InfiniSST系统现在已经具备了完整的多请求并发推理能力。接下来只需要配置实际的模型参数，就可以进行完整的端到端测试了。 