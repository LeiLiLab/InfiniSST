# RAG Threshold Calibration Script

## 概述

这个脚本用于在 ACL6060 dev 数据集上校准 RAG 的置信度阈值。它通过滑动窗口的方式对音频进行术语检索，并通过分析 precision-recall 曲线来找到最优的置信度阈值。

## 主要改进

### 1. 统一小写处理

所有术语（terms）都统一转换为小写，包括：
- Glossary 加载时
- Ground Truth 提取时
- RAG 检索结果时

这样可以避免大小写不一致导致的匹配问题。

### 2. Ground Truth 提取逻辑

原来的逻辑使用正则表达式匹配，容易出现误判。现在改为：

- 直接从文本中提取 `[term]` 标记的词
- 例如：`So we're going to be covering what [lexical] borrowing is, the [task] that we proposed...`
- 提取出的 GT terms: `{"lexical", "task"}`

实现代码：

```python
matches = re.findall(r'\[([^\]]+)\]', line)
for term in matches:
    term_lower = term.strip().lower()
    if term_lower:
        gt_in_sent.add(term_lower)
```

### 3. 真实 RAG 推理

替换了 mock 数据生成逻辑，现在使用真实的 `TermRAGRetriever` 进行检索：

```python
def get_terms_from_audio_chunk(audio_chunk, sr=16000, rag_retriever=None, rag_top_k=5):
    """
    使用 RAG 模型进行术语检索。
    输入：一个 audio chunk (numpy array)
    输出：一个 list of (term, score)
    """
    # 1. 转换为 tensor
    audio_tensor = torch.tensor(audio_chunk, dtype=torch.float32)
    
    # 2. 使用模型编码音频
    with torch.no_grad():
        audio_inputs = [audio_tensor.numpy()]
        embedding = rag_retriever.model.encode_audio(audio_inputs)
    
    # 3. 在 FAISS index 中搜索最相似的术语
    D, I = rag_retriever.index.search(embedding, rag_top_k)
    
    # 4. 转换为 (term, confidence_score) 格式
    results = []
    for distance, idx in zip(D[0], I[0]):
        term = rag_retriever.term_list[idx].get("term", "").lower()
        confidence = rag_retriever._distance_to_confidence(float(distance))
        results.append((term, confidence))
    
    return results
```

## 配置说明

在运行脚本前，需要配置 `CONFIG` 字典：

```python
CONFIG = {
    # 数据路径
    "wav_dir": "/mnt/taurus/data/siqiouyang/datasets/acl6060/dev/segmented_wavs/gold",
    "en_txt_path": "/mnt/taurus/data/siqiouyang/datasets/acl6060/dev/text/txt/ACL.6060.dev.tagged.en-xx.en.txt",
    "glossary_path": "/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_acl6060.json",
    
    # 滑动窗口参数
    "chunk_duration": 2.0,  # 2秒 chunk
    "chunk_overlap": 1.0,   # 1秒 overlap
    "target_beta": 1.0,     # F-Beta 的 Beta 值 (0.5=重准率, 1=平衡, 2=重召回)
    
    # RAG 配置
    "rag_enabled": True,
    "rag_index_path": "/path/to/your/rag_index.pkl",        # TODO: 修改为你的路径
    "rag_model_path": "/path/to/your/rag_model.pt",        # TODO: 修改为你的路径
    "rag_base_model": "Qwen/Qwen2-Audio-7B-Instruct",
    "rag_device": "cuda:0",  # RAG 模型使用的设备
    "rag_top_k": 10,         # 检索 top-k 术语
    "rag_score_threshold": 0.0,  # 初始阈值设为 0，让所有结果都能被分析
}
```

**重要**：请将 `rag_index_path` 和 `rag_model_path` 修改为你实际的模型路径。

## 使用方法

1. **配置路径**：修改 `CONFIG` 中的路径参数

2. **运行脚本**：
   ```bash
   cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre
   python rag_threshold_sliding_window.py
   ```

3. **查看结果**：
   - **控制台输出**：显示最佳阈值、F-score、Precision、Recall 等指标
   - **score_distribution.png**：正负样本的分数分布图
   - **pr_curve.png**：Precision-Recall 曲线图

## 输出示例

```
正在初始化 RAG Retriever...
✅ RAG Retriever 初始化成功
   - Index: /path/to/rag_index.pkl
   - Model: /path/to/rag_model.pt
   - Device: cuda:0
   - Top-K: 10
   - Score Threshold: 0.0

加载了 6060 个术语 (已转为小写)
加载了 469 个句子的 Ground Truth

正在进行滑动窗口推理...
100%|██████████| 469/469 [03:24<00:00,  2.29it/s]

=== 最佳阈值分析 (Beta=1.0) ===
Optimal Threshold: 0.6523
Best F1-Score: 0.7845
Precision at Best: 0.8123
Recall at Best:    0.7589

=== 相对阈值参考 ===
Noise Mean (mu): 0.3456
Noise Std (sigma): 0.1234
Suggested Threshold (Mean + 2Std): 0.5924
Suggested Threshold (Mean + 3Std): 0.7158
```

## 工作流程

1. **加载数据**
   - 加载 glossary 术语列表（统一小写）
   - 从标注文本中提取 Ground Truth（使用 `[]` 标记）

2. **滑动窗口推理**
   - 对每个音频文件进行滑动窗口切分
   - 对每个 chunk 调用 RAG 模型进行术语检索
   - 对同一术语的多个分数取最大值（max pooling）

3. **统计分析**
   - 收集 Positive samples（GT 中存在的术语）的分数
   - 收集 Negative samples（误报）的分数
   - 计算 Precision-Recall 曲线

4. **阈值优化**
   - 基于 F-beta score 找到最佳阈值
   - 提供相对阈值参考（Mean + 2Std, Mean + 3Std）

## 调优参数

- **target_beta**：控制 Precision/Recall 的权重
  - `beta < 1`：偏向 Precision（减少误报）
  - `beta = 1`：平衡 Precision 和 Recall
  - `beta > 1`：偏向 Recall（减少漏报）

- **chunk_duration** / **chunk_overlap**：控制滑动窗口的大小和重叠
  - 更大的 chunk：更多上下文，但可能包含多个术语
  - 更多的 overlap：更密集的检索，但计算量更大

- **rag_top_k**：每个 chunk 检索的术语数量
  - 更大的 k：可能检索到更多相关术语，但也会增加误报

## 依赖项

确保安装以下依赖：

```bash
pip install numpy matplotlib librosa scikit-learn torch tqdm
```

以及项目依赖的 RAG 模型相关包：
- transformers
- peft
- faiss-gpu（或 faiss-cpu）

## 注意事项

1. **GPU 内存**：RAG 模型会占用显存，确保有足够的 GPU 内存
2. **运行时间**：完整的 dev set（469 个样本）可能需要几分钟到几十分钟，取决于 GPU 性能
3. **路径一致性**：确保 wav 文件和文本文件的顺序一致（按文件名排序）
4. **小写统一**：所有术语比较都基于小写，确保 Ground Truth 和检索结果一致

## 故障排除

如果遇到问题：

1. **RAG 初始化失败**：检查 `rag_index_path` 和 `rag_model_path` 是否正确
2. **CUDA out of memory**：降低 `rag_top_k` 或使用更小的 batch size
3. **文件数量不匹配**：确认 wav 文件和文本行数一致
4. **无 GT 提取**：检查文本文件中是否有 `[term]` 标记

## 参考

- RAG Retriever 实现：`agents/infinisst_omni_vllm_rag.py`
- Term training dataset：`retriever/gigaspeech/build_term_training_dataset.py`




















