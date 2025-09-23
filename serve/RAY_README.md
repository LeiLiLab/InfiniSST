# Ray-based InfiniSST Serving System

## 概述

这是基于Ray重构的InfiniSST实时翻译系统，支持动态批处理调度和分布式计算。相比原始系统，Ray版本提供了更好的可扩展性、资源管理和批处理优化。

## 主要特性

### 🚀 Ray分布式架构
- **ModelActor**: 在单个GPU上运行模型推理
- **SessionActor**: 管理用户会话状态
- **SchedulerActor**: 全局调度器，负载均衡和批处理优化
- **Ray集群**: 自动资源管理和故障恢复

### 🔄 动态批处理调度
- 智能批处理：根据队列状态和延迟要求自动调整批次大小
- 可配置的批处理参数（大小、超时时间）
- 负载均衡策略：least_loaded, round_robin, gpu_memory
- 实时性能监控和优化

### 📈 性能优化
- 异步处理管道
- GPU内存优化
- 自动资源回收
- 并发连接管理

### 🛠️ 易于配置和管理
- JSON配置文件
- 环境变量支持
- 命令行参数
- 实时配置更新

## 文件结构

```
serve/
├── ray_serving_system.py    # Ray核心系统实现
├── ray_api.py               # FastAPI服务器
├── ray_config.py            # 配置管理
├── ray_api.sh               # 启动脚本
├── ray_evaluation.py        # 评估框架
├── ray_config.json          # 默认配置（自动生成）
└── RAY_README.md           # 本文档
```

## 快速开始

### 1. 环境准备

```bash
# 安装Ray（如果尚未安装）
pip install ray[default]

# 确保其他依赖已安装
pip install fastapi uvicorn websockets aiohttp

# 检查GPU可用性
nvidia-smi
```

### 2. 创建配置文件

```bash
# 生成默认配置
python ray_config.py --create-default --config ray_config.json

# 查看配置
python serve/ray_config.py --show --config serve/ray_config.json

# 验证配置
python serve/ray_config.py --validate --config serve/ray_config.json
```

### 3. 启动系统

#### 方式1：使用SLURM脚本（推荐）
```bash
cd serve
sbatch ray_api.sh
```

#### 方式2：直接运行
```bash
# 设置GPU环境
export CUDA_VISIBLE_DEVICES=0,1

# 启动Ray集群
ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265

# 启动API服务器
python serve/ray_api.py --host 0.0.0.0 --port 8000 --max-batch-size 32 --batch-timeout-ms 100.0
```

### 4. 验证系统

```bash
# 健康检查
curl http://localhost:8000/health

# Ray统计信息
curl http://localhost:8000/ray/stats

# Ray Dashboard
open http://localhost:8265
```

## 配置说明

### Ray集群配置
```json
{
  "ray_cluster": {
    "ray_address": null,          // null表示本地模式
    "num_cpus": null,             // 自动检测
    "num_gpus": null,             // 自动检测
    "dashboard_host": "0.0.0.0",
    "dashboard_port": 8265
  }
}
```

### 服务配置
```json
{
  "serving": {
    "host": "0.0.0.0",
    "port": 8000,
    "cuda_visible_devices": "0,1",
    "gpu_language_map": {
      "0": "English -> Chinese",
      "1": "English -> Italian"
    },
    "max_batch_size": 32,
    "batch_timeout_ms": 100.0,
    "enable_dynamic_batching": true,
    "load_balance_strategy": "least_loaded"
  }
}
```

### 语言配置
```json
{
  "languages": {
    "English -> Chinese": {
      "source_lang": "English",
      "target_lang": "Chinese",
      "src_code": "en",
      "tgt_code": "zh",
      "model_config": {
        "model_name": "/path/to/model",
        "state_dict_path": "/path/to/weights",
        "latency_multiplier": 2
      }
    }
  }
}
```

## API接口

### 会话管理
```bash
# 创建会话
curl -X POST "http://localhost:8000/init" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "InfiniSST",
    "language_pair": "English -> Chinese",
    "client_id": "user_001"
  }'

# 查询会话状态
curl "http://localhost:8000/queue_status/{session_id}"

# 删除会话
curl -X POST "http://localhost:8000/delete_session" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "session_id"}'
```

### Ray系统管理
```bash
# 获取Ray统计信息
curl "http://localhost:8000/ray/stats"

# 配置Ray系统
curl -X POST "http://localhost:8000/ray/configure" \
  -H "Content-Type: application/json" \
  -d '{
    "max_batch_size": 16,
    "batch_timeout_ms": 150.0,
    "enable_dynamic_batching": true
  }'
```

### WebSocket连接
```javascript
// 连接WebSocket
const ws = new WebSocket(`ws://localhost:8000/wss/${sessionId}`);

// 发送音频数据
ws.send(audioArrayBuffer);

// 接收翻译结果
ws.onmessage = (event) => {
  console.log('Translation:', event.data);
};

// 结束音频流
ws.send('EOF');
```

## 性能评估

### 运行基本功能测试
```bash
python serve/ray_evaluation.py --basic-test-only
```

### 运行批处理性能测试
```bash
python serve/ray_evaluation.py --batch-test-only --num-users 16
```

### 运行完整评估
```bash
python serve/ray_evaluation.py --num-users 32 --test-duration 300
```

### 评估结果
评估完成后会在`ray_evaluation_results/`目录下生成：
- `summary.json`: 测试总结
- `user_test_charts.png`: 用户测试可视化
- `batch_test_heatmaps.png`: 批处理性能热力图
- `README.md`: 详细报告

## 监控和调试

### Ray Dashboard
访问 `http://localhost:8265` 查看：
- 集群状态
- Actor状态
- 任务执行情况
- 资源使用情况

### 日志查看
```bash
# Ray集群日志
ray logs

# API服务器日志
tail -f logs/ray_infinisst_api_*.log

# 特定Actor日志
ray logs --actor-id <actor_id>
```

### 性能调优

#### 批处理参数调优
```bash
# 测试不同批处理配置
python serve/ray_evaluation.py --batch-test-only

# 根据结果调整配置
curl -X POST "http://localhost:8000/ray/configure" \
  -H "Content-Type: application/json" \
  -d '{"max_batch_size": 16, "batch_timeout_ms": 80.0}'
```

#### 负载均衡策略
- `least_loaded`: 选择负载最低的GPU（推荐）
- `round_robin`: 轮询分配
- `gpu_memory`: 基于GPU内存使用情况

## 故障排除

### 常见问题

#### 1. Ray集群启动失败
```bash
# 检查端口占用
lsof -i:8265
lsof -i:6379

# 清理Ray进程
ray stop --force
pkill -f ray

# 重新启动
ray start --head
```

#### 2. GPU内存不足
```bash
# 检查GPU使用情况
nvidia-smi

# 调整批处理大小
export MAX_BATCH_SIZE=16

# 重启系统
```

#### 3. 模型加载失败
```bash
# 检查模型路径
python serve/ray_config.py --validate

# 检查文件权限
ls -la /path/to/model/files

# 更新配置文件中的路径
```

#### 4. WebSocket连接失败
```bash
# 检查防火墙设置
sudo ufw status

# 检查服务器日志
tail -f logs/ray_infinisst_api_*.log

# 测试连接
curl -i -N -H "Connection: Upgrade" -H "Upgrade: websocket" \
  http://localhost:8000/wss/test_session
```

## 与原系统的对比

| 特性 | 原系统 | Ray系统 |
|------|--------|---------|
| 并发处理 | multiprocessing | Ray Actors |
| 批处理调度 | 手动实现 | 动态智能调度 |
| 负载均衡 | 简单轮询 | 多种策略可选 |
| 资源管理 | 手动管理 | 自动管理 |
| 可扩展性 | 单机限制 | 分布式扩展 |
| 监控能力 | 基础日志 | Ray Dashboard |
| 配置管理 | 硬编码 | 灵活配置文件 |
| 故障恢复 | 手动重启 | 自动恢复 |

## 开发和扩展

### 添加新的语言对
1. 修改配置文件：
```json
{
  "languages": {
    "English -> French": {
      "source_lang": "English",
      "target_lang": "French",
      "src_code": "en",
      "tgt_code": "fr",
      "model_config": {
        "model_path": "/path/to/en-fr/model"
      }
    }
  }
}
```

2. 更新GPU映射：
```json
{
  "serving": {
    "gpu_language_map": {
      "0": "English -> Chinese",
      "1": "English -> Italian", 
      "2": "English -> French"
    }
  }
}
```

### 自定义ModelActor
```python
@ray.remote(num_gpus=1)
class CustomModelActor(ModelActor):
    def custom_preprocessing(self, data):
        # 自定义预处理逻辑
        pass
    
    def custom_postprocessing(self, result):
        # 自定义后处理逻辑
        pass
```

### 添加新的调度策略
```python
class CustomSchedulerActor(SchedulerActor):
    def _select_gpu_for_language(self, language_id: str) -> Optional[int]:
        # 实现自定义GPU选择逻辑
        pass
```

## 贡献指南

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 创建Pull Request

## 许可证

请参考项目根目录的LICENSE文件。

## 联系方式

如有问题或建议，请创建Issue或联系维护者。 