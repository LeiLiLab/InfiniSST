# Term Map Dataset Construction - 最终版本

## 📌 核心方案

使用`StreamingTermRAGRetriever`进行**滑动窗口检索 + Max Pooling聚合**。

## 🎯 检索流程

### 对每个audio chunk（例如12秒）：

```
Step 1: 滑动窗口切分（chunk_size=2s, hop_size=1s）
├─ Window  0: [0.0s - 2.0s]
├─ Window  1: [1.0s - 3.0s]
├─ Window  2: [2.0s - 4.0s]
...
└─ Window 10: [10.0s - 12.0s]
    ↓
Step 2: 每个窗口检索top-K
    ↓
Step 3: Max Pooling聚合（按term key取最高分）
    ↓
Step 4: 返回top-10候选terms
```

## 💻 代码实现

### 初始化

```python
from agents.streaming_rag_retriever import StreamingTermRAGRetriever

retriever = StreamingTermRAGRetriever(
    index_path="/mnt/gemini/data2/jiaxuanluo/indices/qwen2_audio_term_index_used_terms_lowercase.pkl",
    model_path="/mnt/gemini/data2/jiaxuanluo/models/qwen2_audio_term_level_modal_v2_best.pt",
    base_model_name="Qwen/Qwen2-Audio-7B-Instruct",
    device="cuda:0",
    chunk_size=2.0,              # 2秒窗口
    hop_size=1.0,                # 1秒步长
    top_k=10,                    # 每窗口top-K
    enable_top_n_filter=False,   # 固定返回10个
    score_threshold=0.0,         # 不过滤
)
```

### 检索

```python
import librosa

# 加载音频
audio, sr = librosa.load(audio_path, sr=16000, mono=True)

# 重置状态
retriever.reset()

# 处理音频（内部滑动窗口+max pooling）
retriever.accumulate_audio(audio, force_process=True)

# 获取top-10结果
references = retriever.get_current_references(min_terms=10)

# 提取候选
candidates = [(ref['term'], ref['translation']) for ref in references]
```

## 📊 示例输出

### 输入音频
- 路径: `YOU0000010238/66/0.wav`
- 时长: 12秒
- 内容: "But if someone has a giant stack of them..."

### 检索结果
```python
[
    ("Jessica Valenti", "杰西卡·瓦伦蒂"),
    ("Bear Stearns", "贝尔斯登公司"),
    ("social statement", "社会声明"),
    ("relationship", "关系"),
    ("direction", "方向"),
    ("planning", "计划"),
    ("dramatic", "戏剧性的"),
    ("giant", "巨大的"),
    ("stack", "堆叠"),
    ("chance", "机会"),
]
```

### 最终term_map
```
term_map:
Jessica Valenti=杰西卡·瓦伦蒂
Bear Stearns=贝尔斯登公司
social statement=社会声明
relationship=关系
direction=方向
planning=计划
dramatic=戏剧性的
giant=巨大的
stack=堆叠
chance=机会
```

## 🚀 运行脚本

### 测试模式（10条数据）
```bash
conda activate infinisst
cd /home/jiaxuanluo/InfiniSST
./retriever/gigaspeech/run_term_map_construction.sh --dry-run
```

### 完整处理（SLURM）
```bash
sbatch retriever/gigaspeech/run_term_map_construction.sh
```

### 直接运行
```bash
python retriever/gigaspeech/handle_train_dataset_for_term_map_v2_buzz.py --dry-run
```

## 📁 输入输出

### 输入
- `train_s_zh_baseline.jsonl`: 原始messages
- `train_xl_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv`: 对齐数据
- `glossary_used.json`: 术语表
- `qwen2_audio_term_index_used_terms_lowercase.pkl`: RAG索引
- `qwen2_audio_term_level_modal_v2_best.pt`: RAG模型

### 输出
- `train_s_zh_with_candidates.jsonl`: 增强后的messages

## ⚙️ 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `chunk_size` | 2.0 | 滑动窗口大小（秒） |
| `hop_size` | 1.0 | 滑动窗口步长（秒） |
| `top_k` | 10 | 每窗口检索数量 |
| `RAG_TOP_K` | 10 | 最终返回数量 |
| `MULTIPLE_RANGE` | [1, 4] | 候选采样倍数 |
| `enable_top_n_filter` | False | 禁用动态过滤 |
| `score_threshold` | 0.0 | 不过滤任何候选 |

## 📈 性能指标

- **模型加载**: ~60秒
- **每个audio chunk**: ~0.2-0.5秒
- **预计总时间（12K条）**: 1-2小时

## ✅ 优势

1. **高召回率**: 多窗口检索提高术语发现概率
2. **精确对齐**: 2s窗口与术语时间匹配
3. **质量保证**: Max pooling自动选择最高分
4. **代码复用**: 使用标准StreamingTermRAGRetriever
5. **鲁棒性强**: 部分窗口失败不影响整体

## 📚 相关文档

- **原理详解**: `SLIDING_WINDOW_EXPLANATION.md`
- **更新日志**: `CHANGELOG_term_map_v2_buzz.md`
- **快速参考**: `QUICKREF_term_map_v2_buzz.md`
- **项目总结**: `SUMMARY_term_map_v2_buzz.md`
- **最终版本**: `FINAL_VERSION.md`

## 🔍 故障排除

### CUDA OOM
```python
RAG_BATCH_SIZE = 16  # 减小批处理大小
```

### 检索结果为空
- 检查index和model路径
- 验证glossary加载
- 确认audio文件存在

### 处理速度慢
- 使用`--max-messages 100`测试
- 检查GPU利用率
- 考虑减小audio chunk数量

## 💡 使用建议

1. **首次运行**: 使用`--dry-run`测试10条数据
2. **调试**: 检查前几条输出的term_map质量
3. **生产**: 使用SLURM批量处理
4. **验证**: 随机抽样检查输出结果

---

**版本**: Final v3  
**日期**: 2025-12-26  
**状态**: ✅ Production Ready  
**作者**: InfiniSST Team


















