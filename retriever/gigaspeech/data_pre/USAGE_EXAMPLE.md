# 使用示例

## 快速开始

### 步骤 1: 设置 API Key

```bash
# 如果使用 OpenAI
export OPENAI_API_KEY='sk-...'

# 如果使用 Anthropic Claude
export ANTHROPIC_API_KEY='sk-ant-...'
export LLM_PROVIDER='anthropic'

# 如果使用本地 LLM (如 vLLM, text-generation-inference)
export OPENAI_API_KEY='dummy'
export OPENAI_API_BASE='http://localhost:8000/v1'
export MODEL_NAME='Qwen/Qwen2.5-7B-Instruct'
```

### 步骤 2: 运行提取

```bash
cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre
./run_extraction.sh
```

### 步骤 3: 查看结果

```bash
cat extracted_glossary.json | jq '.' | head -50
```

## 预期输出

运行脚本后，你将看到类似以下输出：

```
===================================
ACL Paper Glossary Extraction
===================================

Installing dependencies...

Configuration:
  LLM Provider: openai
  Model: gpt-4o-mini

Running glossary extraction...
Starting glossary extraction from ACL papers...
Papers directory: /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/papers
Found 5 PDF files

Processing: 2022.acl-long.110.pdf
Extracted 22826 characters
Calling LLM for 2022.acl-long.110.pdf using openai...
Extracted 25 terms from 2022.acl-long.110.pdf

Processing: 2022.acl-long.117.pdf
Extracted 21872 characters
Calling LLM for 2022.acl-long.117.pdf using openai...
Extracted 28 terms from 2022.acl-long.117.pdf

...

Total terms extracted: 137
Unique terms after merging: 89

✓ Glossary saved to: /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/extracted_glossary.json
✓ Total unique terms: 89

Sample terms:
  - neural machine translation: A machine learning approach to automatic translation
  - transformer: An attention-based neural network architecture
  - BERT: Bidirectional Encoder Representations from Transformers...
```

## 生成的 JSON 格式

```json
{
  "neural machine translation": {
    "term": "neural machine translation",
    "classification_reason": "llm_extracted",
    "confused": false,
    "short_description": "A machine learning approach to automatic translation using neural networks",
    "full_form": "",
    "is_acronym": false,
    "target_translations": {},
    "url": ""
  },
  "BERT": {
    "term": "BERT",
    "classification_reason": "llm_extracted",
    "confused": false,
    "short_description": "A transformer-based model for NLP pre-training",
    "full_form": "Bidirectional Encoder Representations from Transformers",
    "is_acronym": true,
    "target_translations": {},
    "url": ""
  }
}
```

## 成本估算

使用 OpenAI API 的成本估算（基于 gpt-4o-mini）：

- 每篇论文约 15,000 字符输入
- 每篇论文约 2,000 tokens 输出
- 5 篇论文总成本约 $0.05 - $0.10 USD

使用 gpt-4 会更准确但成本更高（约 10-20 倍）。

## 下一步

提取完成后，你可以：

1. **合并到现有 glossary:**
   ```bash
   python merge_glossaries.py extracted_glossary.json ../data/terms/glossary_acl6060.json
   ```

2. **添加翻译:**
   - 手动编辑 `target_translations` 字段
   - 或使用翻译 API 批量添加

3. **验证和清理:**
   - 检查是否有重复或不相关的术语
   - 添加更详细的描述
   - 添加 URL 链接（如 Wikipedia）

## 故障排除

### PyPDF2 无法提取某些 PDF

尝试使用 `pdfplumber`:

```bash
pip install pdfplumber
```

然后修改 `extract_acl_terms_from_paper.py` 中的 `extract_text_from_pdf` 函数：

```python
import pdfplumber

def extract_text_from_pdf(pdf_path: Path, max_pages: int = 20) -> str:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = min(len(pdf.pages), max_pages)
            text_parts = []
            for page_num in range(num_pages):
                text = pdf.pages[page_num].extract_text()
                if text:
                    text_parts.append(text)
            return " ".join(text_parts)[:15000]
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""
```

### LLM 返回的 JSON 格式不正确

如果遇到 JSON 解析错误，可以：

1. 增加 temperature (更确定性的输出)
2. 使用更好的模型 (gpt-4 而不是 gpt-4o-mini)
3. 在 prompt 中添加更多示例
4. 添加重试逻辑

### API 请求限制

如果遇到 rate limit，可以在代码中添加延迟：

```python
import time

def extract_terms_with_llm(paper_content: str, paper_name: str) -> List[Dict]:
    # ... existing code ...
    time.sleep(2)  # 添加 2 秒延迟
    # ... rest of code ...
```























