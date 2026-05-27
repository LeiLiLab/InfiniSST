# 更新日志 - Term Map Dataset Construction

## 2025-12-26 v5 - 增强硬负例 + 全负样本

### 🎯 核心改进

#### 1. 增加硬负例数量（1-9倍）

```python
MULTIPLE_RANGE = [1, 9]  # 原来是[1, 4]
```

**理由**：
- 推理时可能有10-20个高分候选
- 训练难度**略高于**推理难度
- 增强模型鲁棒性

**示例**：
- GT = 2个 → 负例 = 2-18个 → 总共 4-20个terms
- GT = 5个 → 负例 = 5-45个 → 总共 10-50个terms

#### 2. 添加全负样本（10%）

```python
ALL_NEGATIVE_RATIO = 0.1  # 10%的无GT messages
```

**场景**：对于没有GT terms的messages，随机10%添加纯负例term_map

**目的**：训练模型**拒绝高分干扰项**的能力

**示例**：
```
Audio: "It was a sunny day..."  (无术语)

term_map:
relationship=关系    (score: 0.75, 高分但错误)
direction=方向       (score: 0.72, 高分但错误)
planning=计划        (score: 0.68, 高分但错误)
...（共10个高分候选，但都不正确）

Expected: 模型应拒绝全部候选
```

### 实现细节

```python
# Case 1: 有GT - 混合GT和1-9倍硬负例
if num_gt > 0:
    multiple = random.randint(1, 9)
    selected = hard_neg_pool[:num_gt * multiple]
    final = gt_terms + selected
    random.shuffle(final)

# Case 2: 无GT - 10%概率添加全负样本
elif num_gt == 0 and random.random() < 0.1:
    all_negative = hard_neg_pool[:10]  # Top-10高分干扰项
    # 所有候选都不正确，训练模型拒绝能力
```

### 训练样本分布

假设12K数据：

```
有GT terms:     ~9,000 (75%)
  └─ 硬负例:     1-9倍（难度高）

无GT terms:     ~3,000 (25%)
  ├─ 全负样本:   ~300 (10%)   ← 训练拒绝能力
  └─ 无term_map: ~2,700 (90%)  ← 正常翻译
```

### 效果预期

| 能力 | v4 (无全负) | v5 (有全负) |
|------|------------|------------|
| 识别GT | ✓ | ✓ |
| 区分硬负例 | ✓ | ✓ |
| 拒绝全部候选 | ✗ | ✓ |
| 训练难度 | 高 | 极高 |
| 推理鲁棒性 | 高 | 极高 |

详见：`ALL_NEGATIVE_SAMPLES.md`

---

## 2025-12-26 v4 - Hard Negative Mining（硬负例挖掘）

### 🎯 核心改进

引入**Hard Negative Mining**策略，确保训练数据中的负例都是"最难区分的高分干扰项"。

#### 问题分析

滑动窗口后，`_term_scores`可能有50-70个候选词：
- 高分词（0.70-0.85）：很像GT但不是，需要细粒度辨别
- 低分词（0.10-0.30）：完全无关，模型一眼就能排除

**随机采样的缺陷**：
```python
# ❌ 旧方法
sampled = random.sample(all_candidates, n)
# 结果：抽到很多低分简单负例，模型学不到真正的辨别能力
```

#### 解决方案

**按分数排序，直接取top-N硬负例**：
```python
# ✅ 新方法
scored_candidates.sort(key=lambda x: x[2], reverse=True)
hard_neg_pool = [t for t in scored_candidates if not_GT(t)]
selected = hard_neg_pool[:num_negatives]  # 取最硬的前N个
```

### 关键改动

#### 1. 提高RAG_TOP_K到20

```python
RAG_TOP_K = 20  # 每个窗口检索20个词（原来10个）
```

**理由**：
- 扩大硬负例池
- 确保即使部分窗口失败，仍有足够的高分候选
- Max Pooling后能保留更多高质量负例

#### 2. 返回带分数的三元组

```python
# 返回类型改为
Dict[str, List[Tuple[str, str, float]]]
#              ↑     ↑    ↑    ↑
#           source target score
```

#### 3. 硬负例采样逻辑

```python
# A. 按分数排序（已在batch_retrieve_candidates中完成）
scored_rag = rag_candidates.get(audio_path, [])

# B. 过滤GT
hard_neg_pool = [t for t in scored_rag if t[0].lower() not in gt_keys]

# C. 直接取前N个（最硬的）
selected_negatives = hard_neg_pool[:num_negatives]

# D. 与GT混合并打乱（防止位置偏见）
final_candidates = gt_terms + selected_negatives
random.shuffle(final_candidates)
```

### 效果预期

| 维度 | Random Sampling | Hard Negative Mining |
|------|----------------|---------------------|
| 负例质量 | 参差不齐 | 都是高分干扰项 |
| 训练难度 | 简单 | 困难 |
| 泛化能力 | 弱 | 强 ✓ |
| 推理准确率 | 中等 | 高 ✓ |
| 误判率 | 高 | 低 ✓ |

### 理论支持

- **Triplet Loss** (FaceNet, CVPR 2015)
- **Contrastive Learning** (SimCLR, ICML 2020)
- **Curriculum Learning** (Bengio et al., ICML 2009)

详见：`HARD_NEGATIVE_MINING.md`

---

## 2025-12-26 v3 - 使用StreamingTermRAGRetriever滑动窗口 + Max Pooling

### 最终方案

使用`StreamingTermRAGRetriever`的标准流程进行**滑动窗口检索 + Max Pooling聚合**：

#### 检索流程

对于每个audio chunk（可能12秒或更长）：

1. **滑动窗口切分**: 内部以2s窗口、1s步长进行滑动切分
2. **分块检索**: 每个小窗口检索Top-K候选terms
3. **Max Pooling聚合**: 汇总所有小窗口的结果，取得分最高的前10个
4. **混合GT**: 根据GT数量随机采样候选terms作为强负例

#### 实现代码

```python
# 初始化
retriever = StreamingTermRAGRetriever(
    chunk_size=2.0,  # 2s窗口
    hop_size=1.0,    # 1s步长
    top_k=10,
    enable_top_n_filter=False,  # 禁用动态过滤
    ...
)

# 处理每个audio chunk
retriever.reset()  # 重置状态
retriever.accumulate_audio(audio, force_process=True)  # 滑动窗口处理
references = retriever.get_current_references(min_terms=10)  # Max pooling聚合
```

### 为什么需要滑动窗口

- **长音频覆盖**: audio chunk可能长达12s+，单次编码难以覆盖所有内容
- **时间对齐**: 2s窗口与术语出现时间更匹配
- **召回率**: 多窗口检索提高术语召回率
- **Max Pooling**: 自动过滤低分候选，保留最相关的术语

---

## 2025-12-26 v2 - 使用StreamingTermRAGRetriever + retrieve_direct (已废弃)

### 主要改动

1. **使用StreamingTermRAGRetriever**
   - 移除自定义RAG实现
   - 使用标准的`StreamingTermRAGRetriever`类，保持代码一致性
   - 支持更新后的index格式（带`key`字段）

2. **新增retrieve_direct方法**
   - 在`StreamingTermRAGRetriever`中添加了`retrieve_direct()`方法
   - 直接检索，无滑动窗口开销
   - 适用于批量处理场景

### 技术细节

#### StreamingTermRAGRetriever.retrieve_direct()

```python
def retrieve_direct(
    self,
    audio: Union[np.ndarray, List[np.ndarray]],
    top_k: Optional[int] = None,
) -> List[Dict[str, str]]:
    """
    Direct retrieval without sliding window (for batch processing).
    
    Returns:
        List of dicts with:
        - 'key': canonical lowercase term key
        - 'term': original-cased surface form
        - 'translation': target language translation
        - 'score': retrieval score (0-1)
    """
```

**特点**:
- 直接编码整个audio chunk
- 无滑动窗口分割
- Max pooling聚合多个audio的结果
- 按score排序返回top-k

#### 使用方式

```python
# 初始化
retriever = StreamingTermRAGRetriever(
    index_path=RAG_INDEX_PATH,
    model_path=RAG_MODEL_PATH,
    ...
)

# 直接检索
audio, sr = librosa.load(audio_path, sr=16000)
results = retriever.retrieve_direct(audio, top_k=10)

# 结果格式
for ref in results:
    print(f"{ref['term']} = {ref['translation']}")
    print(f"  score: {ref['score']:.4f}")
```

### 配置更新

```python
# 更新后的路径
RAG_INDEX_PATH = "/mnt/gemini/data2/jiaxuanluo/indices/qwen2_audio_term_index_used_terms_lowercase.pkl"
RAG_MODEL_PATH = "/mnt/gemini/data2/jiaxuanluo/models/qwen2_audio_term_level_modal_v2_best.pt"
```

### 性能优化

**之前**:
- 每个audio chunk需要滑动窗口分割
- 多次编码和检索
- 需要reset()和accumulate_audio()

**现在**:
- 每个audio chunk直接编码一次
- 单次检索
- 无状态管理开销

**预期提升**: 处理速度提升约2-3倍

### 兼容性

- ✅ 支持新index格式（带`key`和`term`字段）
- ✅ 支持更新后的glossary
- ✅ 与`StreamingTermRAGRetriever`其他功能兼容
- ✅ 可用于流式和批量场景

### 测试

创建了测试脚本验证新方法：

```bash
# 测试retrieve_direct
python retriever/gigaspeech/test_retrieve_direct.py
```

### 文件变更

1. **agents/streaming_rag_retriever.py**
   - 新增`retrieve_direct()`方法

2. **retriever/gigaspeech/handle_train_dataset_for_term_map_v2_buzz.py**
   - 使用`StreamingTermRAGRetriever`
   - 调用`retrieve_direct()`进行检索
   - 简化代码逻辑

3. **retriever/gigaspeech/test_retrieve_direct.py** (新增)
   - 测试脚本

4. **文档更新**
   - SUMMARY_term_map_v2_buzz.md
   - CHANGELOG_term_map_v2_buzz.md (本文件)

### 下一步

运行测试验证：

```bash
# 1. 测试retrieve_direct方法
conda activate infinisst
python retriever/gigaspeech/test_retrieve_direct.py

# 2. Dry run测试完整流程
./retriever/gigaspeech/run_term_map_construction.sh --dry-run

# 3. 完整处理
./retriever/gigaspeech/run_term_map_construction.sh
```

---

**更新时间**: 2025-12-26
**版本**: v2 (with retrieve_direct)

