# ACL Paper Glossary Extraction

这个工具从 ACL 会议论文中提取技术术语 (glossary)，用于生成会议演讲的术语表。

## 功能

- 从 PDF 论文中提取文本
- 使用 LLM API 识别领域特定的技术术语
- 支持多种 LLM 提供商（OpenAI、Anthropic 等）
- 生成结构化的术语表 JSON 文件

## 安装

```bash
pip install PyPDF2 google-generativeai openai anthropic
```

## 使用方法

### 1. 设置 API Key

**使用 Google Gemini (推荐，默认):**
```bash
export GEMINI_API_KEY='your-gemini-api-key'
# 可选：指定模型
export MODEL_NAME='gemini-1.5-flash'  # 或 'gemini-1.5-pro' 或 'gemini-2.0-flash-exp'
```

**或使用 OpenAI:**
```bash
export OPENAI_API_KEY='your-openai-api-key'
export LLM_PROVIDER='openai'
export MODEL_NAME='gpt-4o-mini'  # 或 'gpt-4'
```

**或使用 Anthropic:**
```bash
export ANTHROPIC_API_KEY='your-anthropic-api-key'
export LLM_PROVIDER='anthropic'
export MODEL_NAME='claude-3-5-sonnet-20241022'
```

**或使用本地/自定义 LLM 服务:**
```bash
export OPENAI_API_KEY='dummy'  # 某些本地服务需要
export OPENAI_API_BASE='http://localhost:8000/v1'
export LLM_PROVIDER='openai'
export MODEL_NAME='your-model-name'
```

### 2. 准备论文

将 PDF 论文放在 `papers/` 目录下：
```bash
papers/
  ├── 2022.acl-long.110.pdf
  ├── 2022.acl-long.117.pdf
  └── ...
```

### 3. 运行提取

**方式 1: 使用脚本**
```bash
./run_extraction.sh
```

**方式 2: 直接运行 Python**
```bash
python extract_acl_terms_from_paper.py
```

### 4. 查看结果

结果将保存在 `extracted_glossary.json`，格式如下：

```json
{
  "BERT": {
    "term": "BERT",
    "classification_reason": "llm_extracted",
    "confused": false,
    "short_description": "A transformer-based model for NLP pre-training",
    "full_form": "Bidirectional Encoder Representations from Transformers",
    "is_acronym": true,
    "target_translations": {},
    "url": ""
  },
  "attention mechanism": {
    "term": "attention mechanism",
    "classification_reason": "llm_extracted",
    "confused": false,
    "short_description": "A neural network component that weighs the importance of different inputs",
    "full_form": "",
    "is_acronym": false,
    "target_translations": {},
    "url": ""
  }
}
```

## 配置选项

环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_PROVIDER` | `gemini` | LLM 提供商: `gemini`, `openai`, 或 `anthropic` |
| `GEMINI_API_KEY` | - | Google Gemini API 密钥 |
| `OPENAI_API_KEY` | - | OpenAI API 密钥 |
| `ANTHROPIC_API_KEY` | - | Anthropic API 密钥 |
| `MODEL_NAME` | `gemini-1.5-flash` | 使用的模型名称 |
| `OPENAI_API_BASE` | - | 自定义 API 端点（用于本地 LLM） |

### 推荐模型选择

**Google Gemini (推荐):**
- `gemini-1.5-flash` - 快速且经济 (默认)
- `gemini-1.5-pro` - 更高质量
- `gemini-2.0-flash-exp` - 实验性最新模型

**OpenAI:**
- `gpt-4o-mini` - 经济型
- `gpt-4o` - 平衡型
- `gpt-4` - 高质量

**Anthropic:**
- `claude-3-5-sonnet-20241022` - 推荐
- `claude-3-opus-20240229` - 最高质量

## 术语提取标准

脚本使用的 prompt 会提取满足以下条件的术语：

- ✅ 领域特定 (NLP / Speech / ML)
- ✅ 可能在口头演讲中提及
- ✅ 实质性的技术概念
- ✅ 包括缩写及其完整形式

## 文件说明

- `extract_acl_terms_from_paper.py` - 主提取脚本
- `run_extraction.sh` - 便捷运行脚本
- `papers/` - 存放 PDF 论文的目录
- `extracted_glossary.json` - 输出的术语表
- `talk_ids.txt` - 演讲 ID 列表（如果需要批量下载）

## 故障排除

### 问题: "No API key found"
**解决:** 确保设置了正确的环境变量 `OPENAI_API_KEY` 或 `ANTHROPIC_API_KEY`

### 问题: PDF 提取失败
**解决:** 某些 PDF 可能有保护或特殊格式。可以尝试：
- 使用 `pdfplumber` 代替 `PyPDF2`
- 手动转换 PDF 为文本

### 问题: LLM 返回格式错误
**解决:** 
- 尝试更高级的模型 (如 `gpt-4` 而不是 `gpt-4o-mini`)
- 调整 temperature 参数（在代码中修改）
- 检查论文内容是否正确提取

## 扩展功能

如需添加翻译功能，可以在提取后调用翻译 API：

```python
# 在 merge_terms 函数中添加
"target_translations": {
    "zh": translate_to_chinese(term),
    "es": translate_to_spanish(term),
}
```

## 许可

与主项目相同























