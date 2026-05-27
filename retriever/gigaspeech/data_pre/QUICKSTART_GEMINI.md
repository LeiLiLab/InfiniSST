# 快速开始 - 使用 Google Gemini

## 一条命令运行

```bash
export GEMINI_API_KEY='your-gemini-api-key' && cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre && ./run_extraction.sh
```

## 详细步骤

### 1. 获取 Gemini API Key

访问: https://aistudio.google.com/app/apikey

点击 "Create API Key" 获取你的 API key

### 2. 设置环境变量

```bash
export GEMINI_API_KEY='AIza...'  # 替换为你的 API key
```

### 3. 运行提取脚本

```bash
cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre
./run_extraction.sh
```

### 4. 查看结果

```bash
ls -lh extracted_glossary.json
cat extracted_glossary.json | python -m json.tool | head -50
```

## 使用不同的 Gemini 模型

### 使用 Gemini 1.5 Flash (默认，快速且便宜)
```bash
export GEMINI_API_KEY='your-key'
export MODEL_NAME='gemini-1.5-flash'
./run_extraction.sh
```

### 使用 Gemini 1.5 Pro (更高质量)
```bash
export GEMINI_API_KEY='your-key'
export MODEL_NAME='gemini-1.5-pro'
./run_extraction.sh
```

### 使用 Gemini 2.0 Flash Experimental (最新实验版)
```bash
export GEMINI_API_KEY='your-key'
export MODEL_NAME='gemini-2.0-flash-exp'
./run_extraction.sh
```

## 预期输出

```
===================================
ACL Paper Glossary Extraction
===================================

Installing dependencies...

Configuration:
  LLM Provider: gemini
  Model: gemini-1.5-flash

Running glossary extraction...
Using Gemini model: gemini-1.5-flash
Starting glossary extraction from ACL papers...
Papers directory: /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/papers
Found 5 PDF files

Processing: 2022.acl-long.110.pdf
Extracted 22826 characters
Calling LLM for 2022.acl-long.110.pdf using gemini...
Extracted 25 terms from 2022.acl-long.110.pdf

Processing: 2022.acl-long.117.pdf
...

✓ Glossary saved to: extracted_glossary.json
✓ Total unique terms: 89
```

## 成本估算

Gemini API 非常经济实惠：

**Gemini 1.5 Flash:**
- 输入: $0.075 / 1M tokens (前 2M 免费)
- 输出: $0.30 / 1M tokens (前 2M 免费)
- **5 篇论文预计成本: ~$0.01 USD (或免费)**

**Gemini 1.5 Pro:**
- 输入: $1.25 / 1M tokens (前 2M 免费)
- 输出: $5.00 / 1M tokens (前 2M 免费)
- **5 篇论文预计成本: ~$0.10 USD (或免费)**

## 故障排除

### 错误: "GEMINI_API_KEY environment variable not set"

**解决方案:**
```bash
export GEMINI_API_KEY='your-actual-api-key'
```

### 错误: "google-generativeai package not installed"

**解决方案:**
```bash
pip install google-generativeai
```

### API 配额限制

Gemini 有以下免费配额限制：
- Flash: 15 RPM (requests per minute)
- Pro: 2 RPM

如果遇到限制，脚本会自动重试。你也可以在代码中添加延迟：

```python
import time
time.sleep(5)  # 每个请求之间等待 5 秒
```

## 优势

使用 Gemini 的优势：
- ✅ **免费额度**: 每天 2M tokens 免费
- ✅ **速度快**: Flash 模型响应速度快
- ✅ **质量高**: 1.5 Pro 和 2.0 模型质量很好
- ✅ **大上下文**: 支持 1M+ tokens 上下文
- ✅ **多语言**: 对中文等非英语语言支持好

## 下一步

提取完成后，查看 [USAGE_EXAMPLE.md](USAGE_EXAMPLE.md) 了解如何：
- 合并术语表
- 添加翻译
- 验证和清理结果























