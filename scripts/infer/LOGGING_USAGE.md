# SimulEval 日志持久化使用说明

## 改进内容

新脚本 `run_simuleval_with_logging.sh` 相比原脚本的改进：

1. **完整日志持久化**：所有终端输出（stdout + stderr）都会保存到日志文件
2. **实时查看**：使用 `tee` 命令，日志既写入文件也显示在终端
3. **自动组织**：日志文件按时间戳命名，保存在 `${OUTPUT_PATH}/logs/` 目录下
4. **退出码保留**：正确保留 SimulEval 的退出码，便于脚本集成

## 关键改动

```bash
# 1. 定义日志文件路径
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${OUTPUT_PATH}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/simuleval_${TIMESTAMP}.log"

# 2. 将所有命令包装在 main() 函数中
main() {
    # ... 所有原有命令 ...
}

# 3. 使用 tee 同时输出到终端和文件
main 2>&1 | tee "${LOG_FILE}"
```

## 使用方法

### 方式 1：直接使用新脚本（推荐）

在 Docker 容器启动时使用新脚本：

```bash
docker run ... \
  bash /workspace/InfiniSST/scripts/infer/run_simuleval_with_logging.sh
```

### 方式 2：在现有脚本中添加日志重定向

如果你想修改现有脚本，只需在最外层添加：

```bash
#!/bin/bash
set -e
set -x

# 定义日志文件
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${OUTPUT_PATH}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/simuleval_${TIMESTAMP}.log"

# 定义 main 函数包含所有原有内容
main() {
    # ... 你的所有原有命令 ...
    echo "===== Installing Python dependencies ====="
    pip install ...
    
    echo "===== Running SimulEval ====="
    python -u "$(which simuleval)" ...
}

# 使用 tee 重定向
main 2>&1 | tee "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]}
exit ${EXIT_CODE}
```

### 方式 3：最简单的日志重定向（不用修改脚本）

在调用脚本时直接重定向：

```bash
# Docker 容器内执行
bash your_script.sh 2>&1 | tee "${OUTPUT_PATH}/logs/simuleval_$(date +%Y%m%d_%H%M%S).log"
```

## 日志文件位置

日志文件保存在：
```
${OUTPUT_PATH}/logs/simuleval_YYYYMMDD_HHMMSS.log
```

例如：
```
/mnt/gemini/data2/jiaxuanluo/infinisst_omni_vllm_rag_docker_acl6060/logs/simuleval_20241217_143052.log
```

## 实时查看日志

在另一个终端中实时查看日志：

```bash
# 宿主机上（如果挂载了日志目录）
tail -f /mnt/gemini/data2/jiaxuanluo/infinisst_omni_vllm_rag_docker_acl6060/logs/simuleval_*.log

# 或者进入 Docker 容器查看
docker exec -it <container_id> tail -f /workspace/OUTPUT_PATH/logs/simuleval_*.log
```

## 日志内容说明

日志文件包含：
- ✅ 所有环境变量配置
- ✅ pip 安装输出
- ✅ SimulEval 运行日志
- ✅ 所有 print() 输出
- ✅ RAG 检索结果
- ✅ LLM 输入输出调试信息
- ✅ 错误信息和异常堆栈

## 补充：多种日志文件说明

### 1. instances.log (SimulEval 原始输出)
- **位置**：`${OUTPUT_PATH}/instances.log`
- **格式**：Unicode 转义编码（`\u4f60\u597d` 这种）
- **用途**：SimulEval 测评使用，机器可读
- **示例**：
  ```json
  {"\u4f60\u597d": "\u6211\u53eb\u7c73\u54c8\u5c14"}
  ```

### 2. back.log (人类可读版本) ✨ 新增
- **位置**：`${OUTPUT_PATH}/back.log`
- **格式**：正常的中文显示（自动转换）
- **用途**：人类阅读和调试
- **示例**：
  ```json
  {
    "你好": "我叫米哈尔"
  }
  ```
- **生成方式**：脚本自动生成，也可手动转换

### 3. runtime_omni_vllm_rag_*.jsonl
- **位置**：`/home/jiaxuanluo/InfiniSST/converted_logs/runtime_omni_vllm_rag_*.jsonl`
- **内容**：结构化的 JSON 记录（RAG 检索、LLM 输入输出等）
- **用途**：便于后续分析和处理

### 4. simuleval_*.log (终端完整日志)
- **位置**：`${OUTPUT_PATH}/logs/simuleval_*.log`
- **内容**：完整的终端输出（包括所有 print、error 等）
- **用途**：调试和问题排查

## 手动转换 instances.log

如果需要单独转换已有的 `instances.log` 文件：

```bash
# 方式 1：使用 Python 脚本（推荐）
python /workspace/InfiniSST/scripts/tools/convert_instances_log.py \
    /path/to/instances.log

# 方式 2：指定输出文件名
python /workspace/InfiniSST/scripts/tools/convert_instances_log.py \
    /path/to/instances.log \
    /path/to/readable.log

# 方式 3：使用一行 Python 命令
python -c "
import json
with open('instances.log', 'r') as f, open('back.log', 'w') as out:
    for line in f:
        if line.strip():
            obj = json.loads(line)
            out.write(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
"
```

## 故障排查

如果日志没有正确保存：

1. 检查 `${OUTPUT_PATH}` 目录权限
2. 确保 `tee` 命令可用（通常默认安装）
3. 查看是否有磁盘空间

```bash
# 检查权限
ls -la "${OUTPUT_PATH}"

# 检查磁盘空间
df -h "${OUTPUT_PATH}"
```






















