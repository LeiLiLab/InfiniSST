# 全负样本（All-Negative Samples）训练策略

## 🎯 目标

训练模型**拒绝高分干扰项**的能力，即使RAG召回的候选词分数很高，如果音频中没有对应内容，模型也应该识别出来。

## 💡 核心思想

在真实推理场景中，RAG可能会因为各种原因（音频相似、主题相关等）召回一些高分但实际不正确的候选词。模型需要学会：
1. 识别正确的GT terms（正样本学习）
2. 区分高分干扰项（硬负例学习）
3. **拒绝全部候选**（当音频中没有任何术语时）← 新增

## 📊 训练样本分布

### 类型1：正常样本（有GT terms，~90%）

```
Audio: "But if someone has a giant stack..."

term_map:
giant=巨大的                  (GT ✓)
stack=堆叠                   (GT ✓)
relationship=关系             (硬负例, score: 0.75)
direction=方向                (硬负例, score: 0.72)
planning=计划                 (硬负例, score: 0.68)
...
```

**学习目标**：从混合候选中识别GT

### 类型2：全负样本（无GT terms，随机10%添加term_map）

```
Audio: "It was a sunny day in the park..."

term_map:
social statement=社会声明      (硬负例, score: 0.76)
relationship=关系             (硬负例, score: 0.75)
direction=方向                (硬负例, score: 0.72)
planning=计划                 (硬负例, score: 0.68)
giant=巨大的                  (硬负例, score: 0.65)
...（共10个高分候选）

Expected Output: (empty or reject all)
```

**学习目标**：识别所有候选都不正确，拒绝全部

### 类型3：无term_map（无GT terms，90%）

```
Audio: "It was a sunny day..."

<audio>
(无term_map)
```

**学习目标**：正常翻译，无术语约束

## 🔧 实现细节

### 采样策略

```python
# 配置
ALL_NEGATIVE_RATIO = 0.1  # 10%的无GT messages添加全负样本

# 逻辑
if num_gt == 0:  # 无GT terms
    if random.random() < ALL_NEGATIVE_RATIO:  # 10%概率
        # 添加top-10高分干扰项作为全负样本
        all_negative_terms = hard_neg_pool[:10]
        # 这些都不是正确答案，但分数很高（0.65-0.85）
```

### 候选选择

全负样本使用**最硬的top-10负例**：
- 分数范围：通常0.65-0.85
- 都是RAG认为最相关的词
- 但实际音频中没有对应内容

### 为什么是10%？

**平衡考虑**：
- 太少（<5%）：模型学不到拒绝能力
- 太多（>20%）：影响正常术语学习
- **10%**：既能学到拒绝能力，又不影响主要任务

## 📈 训练效果

### 场景1：推理时遇到无关高分候选

**问题**：
```
Audio: "The weather is nice today"
RAG召回: ["relationship", "planning", "direction"]  # 高分但无关
```

**无全负样本训练**：
- 模型可能误选一个"看起来最像"的词
- 因为训练时从未见过"全部拒绝"的情况

**有全负样本训练**：
- 模型学会识别"所有候选都不对"
- 输出时不会强制选择任何术语

### 场景2：RAG召回质量下降

在实际部署中，某些情况下RAG可能召回质量较差：
- 音频噪声大
- 说话人口音重
- 背景音乐干扰

全负样本训练让模型更鲁棒，不会被迫选择错误术语。

## 🎓 理论支持

### 1. Negative Sampling (Mikolov et al., Word2Vec)

在对比学习中，负样本（尤其是硬负例）对模型辨别能力至关重要。

### 2. Rejection Option in Classification

某些分类任务需要模型具备"拒绝"能力，当置信度不足时不做预测。

### 3. Curriculum Learning

从简单到困难：
- 阶段1：正样本学习（识别GT）
- 阶段2：硬负例学习（区分干扰项）
- 阶段3：全负样本学习（拒绝全部）← 最难

## 📊 样本分布统计

假设12K训练数据：

```
有GT terms:     ~9,000 (75%)
  ├─ 正常样本:   9,000 (100%)
  └─ 硬负例比例: 1-9倍

无GT terms:     ~3,000 (25%)
  ├─ 全负样本:   ~300 (10%)   ← 新增
  └─ 无term_map: ~2,700 (90%)
```

### 全负样本示例

```json
{
  "role": "user",
  "content": "<audio>\n\nterm_map:\nrelationship=关系\ndirection=方向\nplanning=计划\nsocial statement=社会声明\ngiant=巨大的\nstack=堆叠\ndramatic=戏剧性的\nchance=机会\nhappy=快乐\nfuture=未来"
}
```

**关键**：这10个词都是高分候选（0.65-0.85），但音频中**一个都没有**。

## 💡 训练策略调整

### 硬负例数量增加

```python
MULTIPLE_RANGE = [1, 9]  # 原来是[1, 4]
```

**原因**：
- 推理时可能有10-20个高分候选
- 训练时最多9倍硬负例（如GT=2，负例=18）
- 训练难度**略高于**推理难度，增强鲁棒性

### 组合效果

| GT数量 | 硬负例数量（1-9x） | 总候选数 | 学习难度 |
|--------|------------------|---------|---------|
| 2 | 2-18 | 4-20 | 高 |
| 5 | 5-45 | 10-50 | 极高 |
| 0 (全负) | 10 | 10 | 最难（需拒绝全部）|

## 🔍 调试建议

### 监控全负样本比例

```python
# 在训练日志中记录
logger.info(f"All-negative samples: {all_neg_count}/{total_no_gt} ({ratio:.1%})")
```

### 验证模型拒绝能力

在验证集上测试：
```python
# 人工构造全负样本
test_samples = [
    ("audio_with_no_terms", ["irrelevant_term1", "irrelevant_term2", ...])
]

# 检查模型输出
# 期望：不输出任何术语，或明确拒绝
```

## ⚠️ 注意事项

1. **标注质量**：确保标记为"无GT"的样本确实没有术语
2. **比例调整**：可以根据训练效果调整10%比例
3. **推理一致性**：推理时也要允许模型拒绝所有候选
4. **评估指标**：需要增加"拒绝准确率"指标

## 📝 总结

| 维度 | 无全负样本 | 有全负样本（10%）|
|------|-----------|-----------------|
| 正样本学习 | ✓ | ✓ |
| 硬负例学习 | ✓ | ✓ |
| 拒绝能力 | ✗ | ✓ |
| 鲁棒性 | 中 | 高 |
| 训练难度 | 中 | 高 |
| 推理可靠性 | 中 | 高 |

**全负样本是Hard Negative Mining的重要补充，训练模型在所有候选都不正确时的拒绝能力。**

---

**参考资料**:
- Word2Vec: Negative Sampling (Mikolov et al., 2013)
- Learning to Reject with a Fixed Predictor (Geifman & El-Yaniv, 2019)
- Hard Negative Mining for Metric Learning (Schroff et al., FaceNet, 2015)

**更新时间**: 2025-12-26  
**版本**: v5 (with All-Negative Samples)


















