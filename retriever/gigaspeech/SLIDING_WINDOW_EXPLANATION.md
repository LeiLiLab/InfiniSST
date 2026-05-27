# 滑动窗口检索原理说明

## 问题场景

每个audio chunk可能长达12秒或更长，如果直接对整个chunk进行一次编码和检索：
- ❌ 长音频编码效果差
- ❌ 术语时间对齐不精确
- ❌ 召回率低

## 解决方案：滑动窗口 + Max Pooling

### 1. 滑动窗口切分

对于一个12秒的audio chunk：

```
Audio Chunk (12s)
├─ Window 0: [0.0s - 2.0s]    → Retrieve top-K
├─ Window 1: [1.0s - 3.0s]    → Retrieve top-K
├─ Window 2: [2.0s - 4.0s]    → Retrieve top-K
├─ Window 3: [3.0s - 5.0s]    → Retrieve top-K
├─ Window 4: [4.0s - 6.0s]    → Retrieve top-K
├─ Window 5: [5.0s - 7.0s]    → Retrieve top-K
├─ Window 6: [6.0s - 8.0s]    → Retrieve top-K
├─ Window 7: [7.0s - 9.0s]    → Retrieve top-K
├─ Window 8: [8.0s - 10.0s]   → Retrieve top-K
├─ Window 9: [9.0s - 11.0s]   → Retrieve top-K
└─ Window 10: [10.0s - 12.0s] → Retrieve top-K
```

参数：
- **窗口大小** (chunk_size): 2.0秒
- **步长** (hop_size): 1.0秒
- **窗口数量**: `(12 - 2) / 1 + 1 = 11`个窗口

### 2. 分块检索

每个窗口独立进行检索：

```python
Window 0 (0-2s):
  - "Jessica Valenti": 0.85
  - "Bear Stearns": 0.72
  - "relationship": 0.68
  - ...

Window 1 (1-3s):
  - "Bear Stearns": 0.78  # 重复出现，分数不同
  - "social statement": 0.75
  - "direction": 0.65
  - ...

Window 2 (2-4s):
  - "Jessica Valenti": 0.82  # 重复出现
  - "planning": 0.70
  - ...
```

### 3. Max Pooling聚合

对所有窗口的结果按**term key**聚合，保留最高分：

```python
聚合前 (所有窗口):
  - "jessica valenti": [0.85, 0.82, 0.79, ...]
  - "bear stearns": [0.72, 0.78, 0.75, ...]
  - "social statement": [0.75, 0.68, ...]
  - "relationship": [0.68, 0.72, ...]
  - "direction": [0.65, 0.70, ...]
  - ...

Max Pooling:
  - "jessica valenti": max([0.85, 0.82, 0.79]) = 0.85
  - "bear stearns": max([0.72, 0.78, 0.75]) = 0.78
  - "social statement": max([0.75, 0.68]) = 0.75
  - "relationship": max([0.68, 0.72]) = 0.72
  - "direction": max([0.65, 0.70]) = 0.70
  - ...
```

### 4. Top-N 过滤

按分数排序，返回前10个：

```python
Top-10 Candidates:
  1. "Jessica Valenti" = "杰西卡·瓦伦蒂" (0.85)
  2. "Bear Stearns" = "贝尔斯登公司" (0.78)
  3. "social statement" = "社会声明" (0.75)
  4. "relationship" = "关系" (0.72)
  5. "direction" = "方向" (0.70)
  6. ...
  10. "..." (...)
```

## 实现代码

### StreamingTermRAGRetriever使用

```python
# 初始化（配置滑动窗口参数）
retriever = StreamingTermRAGRetriever(
    index_path=RAG_INDEX_PATH,
    model_path=RAG_MODEL_PATH,
    chunk_size=2.0,  # 2秒窗口
    hop_size=1.0,    # 1秒步长
    top_k=10,        # 每窗口检索top-10
    enable_top_n_filter=False,  # 禁用动态过滤
    ...
)

# 处理单个audio chunk
import librosa
audio, sr = librosa.load(audio_path, sr=16000, mono=True)

# 重置状态（新的chunk）
retriever.reset()

# 累积音频（内部自动进行滑动窗口切分和检索）
retriever.accumulate_audio(audio, force_process=True)

# 获取聚合后的结果（max pooling + top-N）
references = retriever.get_current_references(min_terms=10)

# 提取结果
for ref in references:
    print(f"{ref['term']} = {ref['translation']}")
    # ref['term']: 原始大小写形式
    # ref['translation']: 目标语言翻译
    # ref['key']: 小写key（内部使用）
```

### 内部处理流程

`StreamingTermRAGRetriever`内部会：

1. **滑动窗口切分** (`_process_new_windows`)
   ```python
   chunks_to_process = []
   for i in range(num_windows):
       start = i * hop_samples
       end = start + chunk_samples
       chunk = audio_buffer[start:end]
       chunks_to_process.append(chunk)
   ```

2. **批量编码和检索** (`_retrieve_and_aggregate`)
   ```python
   for chunk in chunks:
       embedding = model.encode_audio(chunk)
       distances, indices = index.search(embedding, top_k)
       
       # Max pooling更新
       for idx, dist in zip(indices, distances):
           term_key = term_list[idx]['key']
           score = l2_distance_to_score(dist)
           
           if term_key not in term_scores or score > term_scores[term_key]:
               term_scores[term_key] = score
   ```

3. **聚合和过滤** (`get_current_references`)
   ```python
   # 按分数排序
   sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
   
   # 应用score threshold和top-N
   references = []
   for term_key, score in sorted_terms[:min_terms]:
       if score >= score_threshold:
           references.append({
               'key': term_key,
               'term': term_canonical[term_key],
               'translation': translations[term_key]
           })
   ```

## 优势

✅ **更好的时间对齐**: 2s窗口与术语出现时间匹配

✅ **更高的召回率**: 多窗口检索提高术语发现概率

✅ **自动去重**: Max pooling自动处理重复术语

✅ **鲁棒性**: 即使某个窗口检索失败，其他窗口可能成功

✅ **分数可靠**: 取最高分保证检索质量

## 参数调优

### chunk_size (窗口大小)

- **2.0s**: 标准设置，适合大多数场景
- **1.5s**: 更细粒度，适合快速语速
- **3.0s**: 更粗粒度，适合慢速语速

### hop_size (步长)

- **1.0s**: 标准设置（50%重叠）
- **0.5s**: 更多重叠，召回率更高，但计算量大
- **2.0s**: 无重叠，速度快但可能漏检

### top_k (每窗口检索数量)

- **5-10**: 标准设置
- **15-20**: 需要更多候选时

### enable_top_n_filter

- **False**: 始终返回固定数量的terms（推荐用于数据集构造）
- **True**: 基于音频时长动态调整terms数量（推荐用于推理）

## 完整示例

```python
# 假设audio_chunk是12秒长的音频
audio_chunk, sr = librosa.load("chunk_0.wav", sr=16000)
# audio_chunk.shape = (192000,)  # 12s * 16000

retriever.reset()
retriever.accumulate_audio(audio_chunk, force_process=True)
references = retriever.get_current_references(min_terms=10)

print(f"Retrieved {len(references)} terms:")
for i, ref in enumerate(references, 1):
    print(f"{i}. {ref['term']} = {ref['translation']}")
```

输出：
```
Retrieved 10 terms:
1. Jessica Valenti = 杰西卡·瓦伦蒂
2. Bear Stearns = 贝尔斯登公司
3. social statement = 社会声明
4. relationship = 关系
5. direction = 方向
6. planning = 计划
7. dramatic = 戏剧性的
8. giant = 巨大的
9. stack = 堆叠
10. chance = 机会
```

---

**总结**: 滑动窗口 + Max Pooling是处理长音频术语检索的最佳实践，平衡了召回率、精度和计算效率。


















