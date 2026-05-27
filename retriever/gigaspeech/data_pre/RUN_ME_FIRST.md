# 🚀 运行指南

## 步骤 1: 设置 Gemini API Key

在运行脚本之前，你需要设置 API key：

```bash
export GEMINI_API_KEY='your-actual-gemini-api-key'
```

**获取 API Key:**
1. 访问: https://aistudio.google.com/app/apikey
2. 点击 "Create API Key"
3. 复制 API key (格式类似: `AIzaSy...`)

## 步骤 2: 测试 API 连接（可选但推荐）

```bash
cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre
python test_gemini_api.py
```

如果成功，你会看到：
```
✓ Successfully parsed X terms!
```

## 步骤 3: 运行完整提取

```bash
./run_extraction.sh
```

或者直接运行 Python 脚本：
```bash
python extract_acl_terms_from_paper.py
```

## 步骤 4: 查看结果

```bash
cat extracted_glossary.json | python -m json.tool | head -100
```

## 完整示例（复制粘贴运行）

```bash
# 1. 设置 API key（替换为你的实际 key）
export GEMINI_API_KEY='AIzaSy...'

# 2. 进入目录
cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre

# 3. 测试连接
python test_gemini_api.py

# 4. 如果测试成功，运行完整提取
./run_extraction.sh

# 5. 查看结果
cat extracted_glossary.json | python -m json.tool
```

## 故障排除

### 问题：google-genai 包安装失败

尝试使用旧包：
```bash
pip install google-generativeai
```

脚本会自动检测并使用可用的包。

### 问题：JSON 解析错误

这通常是因为 LLM 返回了额外的文本。脚本已经更新为：
- 自动移除 markdown 代码块
- 使用正则表达式提取 JSON 数组
- 打印完整响应以便调试

如果仍然失败，查看输出中的 "Full response was:" 部分。

### 问题：API 配额限制

Gemini 免费版限制：
- Flash: 15 requests/minute
- Pro: 2 requests/minute

如果遇到限制，在代码中添加延迟（已在脚本中处理）。

## 需要帮助？

查看详细文档：
- `README.md` - 完整使用说明
- `QUICKSTART_GEMINI.md` - Gemini 快速开始
- `USAGE_EXAMPLE.md` - 使用示例

## 当前文件

你的 `papers/` 目录包含 5 个 PDF 文件：
- 2022.acl-long.110.pdf
- 2022.acl-long.117.pdf  
- 2022.acl-long.268.pdf
- 2022.acl-long.367.pdf
- 2022.acl-long.590.pdf

预计处理时间：2-5 分钟
预计术语数量：80-150 个























