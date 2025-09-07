# Ray API 升级说明

## 概述

Ray版本的InfiniSST API现已补全了与原版API的兼容性，确保前端可以无缝连接到后端服务。

## 🚀 新增功能

### 1. 完整的API端点兼容
Ray版本现在包含所有原版API的重要端点：

#### 基础功能
- ✅ `/init` - 会话初始化
- ✅ `/queue_status/{session_id}` - 队列状态查询
- ✅ `/wss/{session_id}` - WebSocket实时翻译
- ✅ `/health` - 健康检查
- ✅ `/ping` - 会话保活

#### 新增兼容端点
- ✅ `/download_youtube` - YouTube视频下载
- ✅ `/load_models` - 模型加载
- ✅ `/load_model` - 单个模型加载
- ✅ `/test_video/{filename}` - 测试视频服务
- ✅ `/debug/session_stats` - 调试会话统计
- ✅ `/debug/session_history` - 调试会话历史
- ✅ `/enable_evaluation_mode` - 启用评估模式
- ✅ `/session_delays/{session_id}` - 会话延迟信息

#### Ray专用端点
- ✅ `/ray/stats` - Ray系统统计信息
- ✅ `/ray/configure` - Ray系统配置

### 2. 改进的静态文件服务
- 🔍 **智能路径检测**: 自动在多个可能位置查找静态文件
- 🔧 **自动回退**: 如果找不到静态文件，提供基础HTML页面
- 📁 **目录自动创建**: 启动时自动创建必要的目录结构

### 3. 增强的错误处理
- 📊 **详细错误信息**: 提供更详细的错误描述和调试信息
- 🔄 **优雅降级**: 在部分功能不可用时提供基本功能
- 🛡️ **异常保护**: 防止单个端点错误影响整个服务

## 🔧 技术改进

### Ray分布式优势
1. **自动资源管理**: Ray自动管理GPU和CPU资源
2. **动态批处理**: 智能批处理调度优化性能
3. **故障恢复**: 自动处理节点故障和重启
4. **负载均衡**: 多种负载均衡策略可选

### 性能优化
1. **异步处理**: 全异步架构提高并发性能
2. **内存优化**: 更好的GPU内存管理
3. **批处理优化**: 动态调整批次大小和超时时间

## 🚦 使用方法

### 1. 启动Ray API服务器

```bash
# 使用SLURM启动（推荐）
cd serve
sbatch ray_api.sh

# 或直接启动
python serve/ray_api.py --host 0.0.0.0 --port 8000
```

### 2. 验证服务状态

```bash
# 测试所有API功能
python serve/test_ray_api.py

# 检查健康状态
curl http://localhost:8000/health

# 查看Ray统计信息
curl http://localhost:8000/ray/stats
```

### 3. 前端连接

前端代码无需修改，Ray版本完全兼容原版API：

```javascript
// 创建会话
const response = await fetch('/init', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        agent_type: 'InfiniSST',
        language_pair: 'English -> Chinese',
        client_id: 'user_001'
    })
});

// WebSocket连接
const ws = new WebSocket(`ws://localhost:8000/wss/${sessionId}`);
```

## 📊 监控和调试

### 1. Ray Dashboard
访问 `http://localhost:8265` 查看：
- 集群状态和资源使用
- Actor状态和任务执行
- 性能指标和日志

### 2. API调试端点
```bash
# 会话统计
curl http://localhost:8000/debug/session_stats

# 会话历史
curl http://localhost:8000/debug/session_history

# Ray系统状态
curl http://localhost:8000/ray/stats
```

### 3. 日志查看
```bash
# Ray集群日志
ray logs

# API服务器日志
tail -f logs/ray_infinisst_api_*.log
```

## 🔄 迁移指南

### 从原版API迁移到Ray版本

1. **配置更新**:
   ```bash
   # 生成Ray配置文件
   python serve/ray_config.py --create-default --config serve/ray_config.json
   ```

2. **启动脚本替换**:
   ```bash
   # 原版启动
   # python serve/api.py
   
   # Ray版本启动
   sbatch serve/ray_api.sh
   # 或
   python serve/ray_api.py
   ```

3. **前端代码**: 无需修改，API完全兼容

### 兼容性检查清单

- ✅ 所有API端点可用
- ✅ WebSocket连接正常
- ✅ 静态文件服务
- ✅ YouTube下载功能
- ✅ 模型加载机制
- ✅ 调试和监控端点

## 🐛 故障排除

### 常见问题

1. **静态文件找不到**
   ```bash
   # 检查静态文件目录
   ls -la serve/static/
   # 或从原版复制
   cp -r static serve/static
   ```

2. **Ray集群启动失败**
   ```bash
   # 清理现有Ray进程
   ray stop --force
   # 重新启动
   ray start --head
   ```

3. **端口占用**
   ```bash
   # 检查端口使用
   lsof -i:8000
   lsof -i:8265
   # 清理占用进程
   pkill -f ray_api
   ```

## 📈 性能对比

| 特性 | 原版API | Ray版本 |
|------|---------|---------|
| 并发处理 | 多进程 | Ray Actors |
| 资源管理 | 手动 | 自动 |
| 批处理调度 | 简单 | 智能动态 |
| 负载均衡 | 基础 | 多策略 |
| 故障恢复 | 手动 | 自动 |
| 监控能力 | 有限 | Ray Dashboard |
| 可扩展性 | 单机 | 分布式 |

## 🎯 未来计划

1. **延迟追踪增强**: 完善Ray版本的延迟统计功能
2. **配置热更新**: 支持运行时配置更新
3. **多模型支持**: 增强多语言对模型管理
4. **性能优化**: 进一步优化批处理和内存使用

## 📞 技术支持

如果遇到问题：
1. 查看Ray Dashboard: `http://localhost:8265`
2. 检查API日志: `/logs/ray_infinisst_api_*.log`
3. 运行测试脚本: `python serve/test_ray_api.py`
4. 查看Ray日志: `ray logs` 