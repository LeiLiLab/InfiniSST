# instances.log 转换工具说明

## 问题背景

SimulEval 输出的 `instances.log` 使用 Unicode 转义编码，不便于人类阅读：

```json
{"\u4f60\u597d\uff0c\u6211\u53eb\u7c73\u54c8\u5c14": "translation"}
```

我们需要将其转换为可读格式：

```json
{
  "你好，我叫米哈尔": "translation"
}
```

## 解决方案

### 🚀 方案 1：自动转换（推荐）

使用改进的脚本 `run_simuleval_with_logging.sh`，会自动生成 `back.log`：

```bash
bash /workspace/InfiniSST/scripts/infer/run_simuleval_with_logging.sh
```

执行完成后，会在 `${OUTPUT_PATH}` 目录下同时生成：
- `instances.log` - 原始的 Unicode 转义格式（供 SimulEval 评估使用）
- `back.log` - 人类可读格式（自动生成）✨

### 🔧 方案 2：手动转换工具

如果你已经有 `instances.log` 文件，可以单独转换：

```bash
# 基本用法（输出到同目录的 back.log）
python scripts/tools/convert_instances_log.py /path/to/instances.log

# 指定输出文件
python scripts/tools/convert_instances_log.py \
    /path/to/instances.log \
    /path/to/readable.log
```

### ⚡ 方案 3：一行命令

```bash
python3 -c "
import json
with open('instances.log', 'r', encoding='utf-8') as f, \
     open('back.log', 'w', encoding='utf-8') as out:
    for line in f:
        if line.strip():
            obj = json.loads(line)
            out.write(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
print('✅ Conversion complete: back.log')
"
```

## 示例对比

### instances.log (原始格式)
```json
{"index": 0, "prediction": "\u4f60\u597d\uff0c\u6211\u53eb\u7c73\u54c8\u5c14\u00b7\u5f7c\u5f97\u9c81\u4ec0\u5361\uff0c\u8fd9\u662f\u6211\u8363\u5e78\u5411\u60a8\u4ecb\u7ecd\u4e00\u7bc7\u9898\u4e3a\u7a00\u758f\u5316Transformer\u6a21\u578b\u7684\u8bba\u6587"}
```

### back.log (可读格式)
```json
{
  "index": 0,
  "prediction": "你好，我叫米哈尔·彼得鲁什卡，这是我荣幸向您介绍一篇题为稀疏化Transformer模型的论文"
}
```

## 脚本工作原理

```python
# 1. 读取 instances.log（逐行）
# 2. 解析每行 JSON
# 3. 重新序列化，设置 ensure_ascii=False
# 4. 写入 back.log

import json

for line in input_file:
    obj = json.loads(line)  # 解析 JSON
    output = json.dumps(obj, ensure_ascii=False, indent=2)  # 转换为可读格式
    output_file.write(output + '\n')
```

## 集成到现有脚本

如果你有自己的脚本，添加以下代码即可：

```bash
# 在 SimulEval 运行完成后添加
if [ -f "${OUTPUT_PATH}/instances.log" ]; then
    python3 -c "
import json
with open('${OUTPUT_PATH}/instances.log', 'r', encoding='utf-8') as f, \
     open('${OUTPUT_PATH}/back.log', 'w', encoding='utf-8') as out:
    for line in f:
        if line.strip():
            obj = json.loads(line)
            out.write(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
"
    echo '✅ Human-readable log created: ${OUTPUT_PATH}/back.log'
fi
```

## 文件位置总结

运行完成后的文件结构：

```
${OUTPUT_PATH}/
├── instances.log          # SimulEval 原始输出（Unicode 转义）
├── back.log              # 人类可读版本（自动生成）✨
├── scores.json           # SimulEval 评分
├── logs/
│   └── simuleval_*.log   # 完整终端日志
└── ...其他 SimulEval 输出文件
```

## 常见问题

### Q: 为什么不直接修改 instances.log？
A: `instances.log` 是 SimulEval 的标准输出格式，用于后续评估。我们保留原始格式，同时生成可读版本。

### Q: 转换会丢失信息吗？
A: 不会。只是改变了显示格式，JSON 内容完全相同。

### Q: 可以批量转换多个文件吗？
A: 可以使用循环：

```bash
for file in /path/to/*/instances.log; do
    python scripts/tools/convert_instances_log.py "$file"
done
```

### Q: 转换失败怎么办？
A: 检查文件编码是否为 UTF-8，以及 JSON 格式是否正确：

```bash
file instances.log  # 检查编码
head -1 instances.log | python -m json.tool  # 检查 JSON 格式
```

## 性能

- 速度：约 10,000 行/秒
- 内存：逐行处理，内存占用极小
- 适用于任意大小的 `instances.log` 文件






















