# siqi_train_term_map 交接文档

## Part 0：（Quick Start）

### 1) 这脚本干什么

从 `input.jsonl` 抽样一部分样本，读取对应 TSV 里的 `English ASR + zh_tokens`，再拼接 jsonl 里的 assistant 中文翻译，用本地 **Qwen3 (vLLM)** 生成 `term_map`（English→中文）。  
生成的 term 会按 clip 分成多份，并注入到每个 `<audio>` 的 user message 后面（`key=value` 一行一个）。

---

### 2) 输入 / 输出

#### 输入

- **jsonl**：每行一个 `dict`，至少有：
  - `audios: list[str]`（用 `audios[0]` 推 `utt_id`）
  - `messages: list[{role, content}]`（user 的 `content` 包含 `<audio>`）
- **tsv**：
  - `parts[0] = utt_id`
  - `parts[3] = English ASR`
  - `parts[-1] = zh_tokens`（字符串形式 Python list，用 `ast.literal_eval`）

#### 输出

- `output.jsonl`
  - 不传 `--limit`：输出全量 records（只改抽中的）
  - 传 `--limit`：只输出抽中的部分

注入格式（当前版本更规范的写法）示例：

```text
<audio>

term_map:
foo=吧
bar=吧
```




### 3) 存在的问题
    count_clip_slots() 数 <audio> 个数 → distribute_term_map_by_clip() 均分 terms/buzz → inject_term_map_by_clip() 追加到 message。
    现在是均分, 并没有按照chunk的长短来分, 用中文分词也不对.
