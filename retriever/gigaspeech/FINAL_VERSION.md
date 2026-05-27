# 最终版本说明 - Term Map Dataset Construction

## ✅ 最终确定方案

使用`StreamingTermRAGRetriever`的**滑动窗口 + Max Pooling + Hard Negative Mining + 全负样本**进行术语检索。

### 四大核心技术

1. **滑动窗口（2s/1s）**: 提高长音频召回率
2. **Max Pooling**: 聚合多窗口结果，取最高分
3. **Hard Negative Mining**: 优先选择高分干扰项（1-9倍）
4. **全负样本（10%）**: 训练模型拒绝能力

## 核心逻辑

### 对每个audio chunk（可能12秒）：

```python
# 1. 重置状态
retriever.reset()

# 2. 处理整个音频（内部滑动窗口+Max Pooling）
retriever.accumulate_audio(audio, force_process=True)

# 3. 直接访问内部状态获取所有候选（带分数）
all_scores = retriever._term_scores  # Dict[term_lc, score]
scored_candidates = [(source, target, score) for ...]
scored_candidates.sort(key=lambda x: x[2], reverse=True)  # 按分数排序

# 4. Hard Negative Mining
gt_terms = match_gt_terms(text, ...)  # FlashText匹配GT
hard_neg_pool = [t for t in scored_candidates if not_GT(t)]
selected = hard_neg_pool[:num_negatives]  # 取最硬的前N个

# 5. 混合并打乱（防止位置偏见）
final = gt_terms + selected
random.shuffle(final)
```

## 关键参数

```python
# 配置
RAG_TOP_K = 20                   # 每窗口检索20个（扩大硬负例池）
RAG_BATCH_SIZE = 64              # 批处理大小（加速100h数据）
MULTIPLE_RANGE = [1, 9]          # 硬负例采样倍数（增加训练难度）
ALL_NEGATIVE_RATIO = 0.1         # 10%无GT消息添加全负样本

StreamingTermRAGRetriever(
    chunk_size=2.0,              # 滑动窗口大小：2秒
    hop_size=1.0,                # 滑动窗口步长：1秒
    top_k=20,                    # 每窗口检索20个（增加硬负例池）
    enable_top_n_filter=False,   # 禁用动态过滤
    score_threshold=0.0,         # 不过滤任何候选
    batch_size=64,               # 批处理大小
)
```

## 完整流程图

```
Audio Chunk (12s)
    ↓
Sliding Window (2s窗口, 1s步长, 每窗口top-20)
    ├─ Window 0: [0-2s]   → Retrieve top-20 → terms + scores
    ├─ Window 1: [1-3s]   → Retrieve top-20 → terms + scores
    ├─ Window 2: [2-4s]   → Retrieve top-20 → terms + scores
    └─ ... (11个窗口)
    ↓
Max Pooling (按term key聚合，取最高分)
    {
      "jessica valenti": max([0.85, 0.82, ...]) = 0.85,
      "bear stearns": max([0.78, 0.75, ...]) = 0.78,
      "social statement": max([0.76, ...]) = 0.76,
      ...  (可能50-70个候选)
    }
    ↓
按分数排序 (降序)
    [
      ("Jessica Valenti", "杰西卡·瓦伦蒂", 0.85),
      ("Bear Stearns", "贝尔斯登公司", 0.78),
      ("social statement", "社会声明", 0.76),  ← 高分干扰项
      ("relationship", "关系", 0.75),          ← 高分干扰项
      ...
      ("unrelated", "无关词", 0.12),           ← 低分简单负例
    ]
    ↓
Hard Negative Mining
    1. 匹配GT: ["giant", "stack"]
    2. 过滤GT: 剩余候选去除GT
    3. 取前N个: hard_neg_pool[:num_negatives]
       → 确保都是高分干扰项（0.70-0.85分）
    4. 混合并打乱: GT + hard_neg, then shuffle
    ↓
生成term_map
    term_map:
    social statement=社会声明      (0.76, 硬负例)
    giant=巨大的                  (0.85, GT ✓)
    relationship=关系             (0.75, 硬负例)
    stack=堆叠                   (0.78, GT ✓)
    direction=方向                (0.72, 硬负例)
    ↓
添加到<audio>占位符
```

## 为什么这样做

### ✅ 优势

1. **时间对齐精确**: 2s窗口匹配术语出现时间
2. **召回率高**: 多窗口检索提高发现概率
3. **质量保证**: Max pooling自动选择最高分
4. **代码复用**: 使用标准的StreamingTermRAGRetriever
5. **鲁棒性强**: 部分窗口失败不影响整体

### ❌ 不采用直接检索的原因

- 长音频（12s）直接编码效果差
- 无法精确定位术语位置
- 召回率低

## 配置文件

```python
# retriever/gigaspeech/handle_train_dataset_for_term_map_v2_buzz.py

RAG_INDEX_PATH = "/mnt/gemini/data2/jiaxuanluo/indices/qwen2_audio_term_index_used_terms_lowercase.pkl"
RAG_MODEL_PATH = "/mnt/gemini/data2/jiaxuanluo/models/qwen2_audio_term_level_modal_v2_best.pt"
RAG_BASE_MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
RAG_DEVICE = "cuda:0"
RAG_TOP_K = 10
RAG_BATCH_SIZE = 32
```

## 运行方式

```bash
# 测试（10条数据）
./retriever/gigaspeech/run_term_map_construction.sh --dry-run

# 完整处理
sbatch retriever/gigaspeech/run_term_map_construction.sh
```

## 输出示例

### 输入
```json
{
  "role": "user",
  "content": "<audio>"
}
```

### 输出
```json
{
  "role": "user",
  "content": "<audio>\n\nterm_map:\nJessica Valenti=杰西卡·瓦伦蒂\nBear Stearns=贝尔斯登公司\nsocial statement=社会声明\nrelationship=关系\ndirection=方向"
}
```

## 性能预估

### 单GPU
- **TSV加载**: ~30秒
- **Glossary加载**: ~5秒
- **RAG模型加载**: ~60秒
- **每个audio chunk**: ~0.2-0.5秒
- **总时间（12K条）**: 约1-2小时

### 多GPU（4卡并行，推荐）
- **模型加载**: ~60秒（4个GPU并行加载）
- **数据处理**: 自动分片，每GPU处理1/4数据
- **总时间（12K条）**: 约30分钟
- **加速比**: ~4x

## 文档索引

- **原理说明**: `SLIDING_WINDOW_EXPLANATION.md`
- **更新日志**: `CHANGELOG_term_map_v2_buzz.md`
- **项目总结**: `SUMMARY_term_map_v2_buzz.md`
- **快速参考**: `QUICKREF_term_map_v2_buzz.md`
- **详细文档**: `README_term_map_v2_buzz.md`

## 验证检查清单

- [x] 使用StreamingTermRAGRetriever
- [x] 启用滑动窗口（chunk_size=2.0, hop_size=1.0）
- [x] 每窗口检索top-20（扩大硬负例池）
- [x] Max pooling聚合
- [x] **按分数排序**（关键：确保硬负例优先）
- [x] **Hard Negative Mining**（直接取top-N，不随机采样）
- [x] **硬负例倍数1-9**（训练难度>推理难度）
- [x] **全负样本10%**（训练拒绝能力）
- [x] 混合GT + 硬负例
- [x] **random.shuffle**（防止位置偏见）
- [x] 生成term_map格式
- [x] 支持新index格式（带key字段）
- [x] 多GPU并行（4卡）

## 核心优势

✅ **高质量负例**: 都是0.70-0.85分的高分干扰项  
✅ **真实场景模拟**: 训练时适应推理场景  
✅ **泛化能力强**: 模型学会细粒度辨别  
✅ **拒绝能力**: 全负样本训练拒绝高分干扰项  
✅ **可复现**: 去重处理，避免重复计算  
✅ **高训练难度**: 1-9倍硬负例，增强鲁棒性  

---

**版本**: v5 (Final with All-Negative Samples)  
**日期**: 2025-12-26  
**状态**: ✅ Ready for Production  
**理论支持**: Triplet Loss, SimCLR, Curriculum Learning, Rejection Learning

