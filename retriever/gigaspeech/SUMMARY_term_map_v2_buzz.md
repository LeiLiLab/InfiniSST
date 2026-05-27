# Term Map Dataset Construction - 项目总结

## 📋 任务完成情况

✅ 已创建完整的数据集构造脚本，用于将RAG候选术语添加到Omni微调数据集中。

## 📁 创建的文件

1. **主脚本**: `retriever/gigaspeech/handle_train_dataset_for_term_map_v2_buzz.py`
   - 完整的数据处理流程
   - 支持批量RAG检索
   - 命令行参数支持（dry-run模式）

2. **文档**: `retriever/gigaspeech/README_term_map_v2_buzz.md`
   - 详细的使用说明
   - 算法流程说明
   - 故障排除指南

3. **测试脚本**: `retriever/gigaspeech/test_term_map_logic.py`
   - 核心逻辑单元测试
   - 无需加载大模型即可验证

## 🔄 数据处理流程

```
输入 JSONL (messages + audios)
    ↓
提取 utter_id (从audio路径)
    ↓
匹配 TSV 行 (获取src_trajectory, tgt_trajectory, src_text, tgt_text)
    ↓
分割 trajectory 为 chunks (按audio数量均分)
    ↓
FlashText 匹配 GT terms (从glossary)
    ↓
RAG 滑动窗口检索 (2s窗口, 1s步长)
    ↓ 每个chunk内部多窗口检索
    ↓ Max Pooling聚合 → top-10 per chunk
    ↓
随机采样混合 (GT terms + GT_size * random[1,4] 个候选terms)
    ↓
生成 term_map 格式
    ↓
添加到 <audio> 占位符后
    ↓
输出增强的 JSONL
```

### RAG检索详解

对于每个audio chunk（可能12秒）：

1. **滑动窗口切分**: 
   - 窗口大小: 2.0秒
   - 步长: 1.0秒
   - 例：12s audio → 约11个窗口

2. **分块检索**: 每个窗口检索top-K候选

3. **Max Pooling**: 对所有窗口的结果按term聚合，取最高分

4. **Top-N过滤**: 返回得分最高的10个terms

**优势**: 提高长音频的术语召回率，时间对齐更精确。

## 📊 示例输出

### 输入示例
```json
{
  "messages": [
    {"role": "system", "content": "You are a professional simultaneous interpreter. You will be given chunks of English audio and you need to translate the audio into Chinese text."},
    {"role": "user", "content": "<audio>"},
    {"role": "assistant", "content": "但如果有人拥有一大叠这样的文件，那他们很"}
  ],
  "audios": ["/mnt/gemini/data1/jiaxuanluo/audio_clips_siqi_v3/YOU0000010238/66/0.wav"]
}
```

### 输出示例
```json
{
  "messages": [
    {"role": "system", "content": "You are a professional simultaneous interpreter. You will be given chunks of English audio and you need to translate the audio into Chinese text."},
    {"role": "user", "content": "<audio>\n\nterm_map:\nsocial statement=社会声明\nlet=让\nrelationship=关系\ndirection=方向\nplanning=计划"},
    {"role": "assistant", "content": "但如果有人拥有一大叠这样的文件，那他们很"}
  ],
  "audios": ["/mnt/gemini/data1/jiaxuanluo/audio_clips_siqi_v3/YOU0000010238/66/0.wav"]
}
```

## 🚀 快速开始

### 1. 测试核心逻辑（无需GPU）
```bash
conda activate infinisst
cd /home/jiaxuanluo/InfiniSST
python retriever/gigaspeech/test_term_map_logic.py
```

### 2. Dry Run测试（处理10条数据）
```bash
conda activate infinisst
cd /home/jiaxuanluo/InfiniSST
python retriever/gigaspeech/handle_train_dataset_for_term_map_v2_buzz.py --dry-run
```

### 3. 完整处理
```bash
conda activate infinisst
cd /home/jiaxuanluo/InfiniSST
python retriever/gigaspeech/handle_train_dataset_for_term_map_v2_buzz.py
```

## ⚙️ 配置参数

可在脚本中修改的关键参数：

```python
# 输入输出路径
INPUT_JSONL = "/mnt/gemini/data1/jiaxuanluo/train_s_zh_baseline.jsonl"
INPUT_TSV = "/mnt/gemini/data1/jiaxuanluo/train_xl_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv"
GLOSSARY_PATH = "/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_used.json"
OUTPUT_JSONL = "/mnt/gemini/data1/jiaxuanluo/train_s_zh_with_candidates.jsonl"

# RAG配置
RAG_INDEX_PATH = "/mnt/gemini/data2/jiaxuanluo/indices/qwen2_audio_term_index_used_terms.pkl"
RAG_MODEL_PATH = "/mnt/gemini/data2/jiaxuanluo/ckpts/qwen2_audio_siqi_contrastive_lora_used_terms_epoch5/checkpoint_epoch_5.pt"
RAG_TOP_K = 10  # 每个chunk检索的候选term数量
RAG_BATCH_SIZE = 32  # 批处理大小

# 候选term采样
MULTIPLE_RANGE = [1, 4]  # 随机倍数范围
```

## 🔍 关键特性

1. **两阶段处理**
   - 第一阶段：收集所有audio路径
   - 第二阶段：批量RAG检索 + 消息增强

2. **智能采样**
   - GT terms保证包含
   - 候选terms根据GT数量动态采样（1-4倍）
   - 随机打乱并去重

3. **错误处理**
   - 缺失audio文件：记录警告，继续处理
   - TSV不匹配：跳过该条数据
   - RAG检索失败：返回空候选列表

4. **内存优化**
   - TSV流式读取 + 索引
   - 批量RAG检索减少内存占用

## 📝 测试结果

所有核心逻辑测试通过：
- ✅ utter_id提取
- ✅ trajectory分割
- ✅ term_map生成
- ✅ glossary加载和匹配（15000个terms）
- ✅ 消息处理流程

## 🛠️ 依赖项

```bash
torch
transformers
peft
faiss-gpu
flashtext
librosa
tqdm
numpy
```

## 📌 注意事项

1. **GPU要求**: RAG检索需要CUDA（默认cuda:0）
2. **内存**: TSV文件较大（~200MB），但已优化为流式处理
3. **随机性**: 使用random.shuffle和random.sample，可设置random.seed()保证可复现
4. **路径格式**: audio路径必须符合 `.../speaker_id/utt_num/chunk.wav` 格式

## 🔧 故障排除

### 如果遇到OOM错误
```python
RAG_BATCH_SIZE = 16  # 减小批处理大小
```

### 如果TSV加载太慢
- 脚本已使用流式读取，正常情况下应该很快
- 可以先用 `--max-messages 100` 测试

### 如果没有候选terms生成
1. 检查glossary是否正确加载
2. 检查RAG index和model路径
3. 使用 `--dry-run` 查看中间结果

## 📞 联系方式

如有问题，请检查：
1. `README_term_map_v2_buzz.md` - 详细文档
2. `test_term_map_logic.py` - 测试脚本
3. 脚本中的日志输出

---

**项目**: InfiniSST Simultaneous Interpretation
**创建日期**: 2025-12-26

