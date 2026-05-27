# 快速参考 - Term Map Dataset Construction

## 🎯 目标
为Omni微调数据集添加RAG检索的候选术语，增强同传模型的术语翻译能力。

## 📦 已创建文件

| 文件 | 说明 |
|------|------|
| `handle_train_dataset_for_term_map_v2_buzz.py` | 主处理脚本 |
| `test_term_map_logic.py` | 单元测试脚本 |
| `run_term_map_construction.sh` | 便捷运行脚本 |
| `README_term_map_v2_buzz.md` | 详细文档 |
| `SUMMARY_term_map_v2_buzz.md` | 项目总结 |
| `QUICKREF_term_map_v2_buzz.md` | 本文件 |

## ⚡ 快速命令

```bash
# 激活环境
conda activate infinisst

# 测试核心逻辑（无需GPU，秒级完成）
cd /home/jiaxuanluo/InfiniSST
python retriever/gigaspeech/test_term_map_logic.py

# Dry Run（处理10条，测试完整流程）
./retriever/gigaspeech/run_term_map_construction.sh --dry-run

# 处理100条（中等规模测试）
./retriever/gigaspeech/run_term_map_construction.sh --max 100

# 完整处理
./retriever/gigaspeech/run_term_map_construction.sh
```

## 📊 输入输出

```
输入:
├── train_s_zh_baseline.jsonl          (原始messages)
├── train_xl_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv  (对齐数据)
├── glossary_used.json                 (术语表)
├── qwen2_audio_term_index_used_terms.pkl  (RAG索引)
└── checkpoint_epoch_5.pt              (RAG模型)

输出:
└── train_s_zh_with_candidates.jsonl   (增强后的messages)
```

## 🔑 核心逻辑

```
<audio> 
  ↓
提取 utter_id
  ↓
查询 TSV → src_trajectory
  ↓
FlashText → GT terms
  ↓
RAG检索 → top-10 candidates
  ↓
采样混合 (GT + random[1,4] × |GT| 个候选)
  ↓
<audio>

term_map:
term1=翻译1
term2=翻译2
```

## 🎨 输出格式示例

```json
{
  "role": "user",
  "content": "<audio>\n\nterm_map:\nJessica Valenti=杰西卡·瓦伦蒂\nBear Stearns=贝尔斯登公司\ndirection=方向"
}
```

## 🔧 常用配置修改

需要修改配置时，编辑 `handle_train_dataset_for_term_map_v2_buzz.py`:

```python
# 减少批处理大小（如果OOM）
RAG_BATCH_SIZE = 16  # 默认32

# 改变候选term数量
RAG_TOP_K = 5  # 默认10

# 改变采样倍数
MULTIPLE_RANGE = [1, 2]  # 默认[1, 4]
```

## 📈 性能估计

- TSV加载: ~30秒 (200MB, 流式处理)
- Glossary加载: ~5秒 (15K terms)
- RAG模型加载: ~1分钟
- 每条数据处理: ~0.1秒 (含RAG检索)
- 总时间: 取决于数据量

**估计**: 12K条数据约需 20-30分钟

## ⚠️ 注意事项

1. **GPU**: 必须有CUDA设备
2. **内存**: 至少16GB RAM推荐
3. **磁盘**: 输出文件约为输入的1.5-2倍
4. **路径**: audio路径格式必须为 `*/speaker_id/utt_num/chunk.wav`

## 🐛 故障排除

| 问题 | 解决方案 |
|------|----------|
| CUDA OOM | 减小 `RAG_BATCH_SIZE` |
| 找不到audio文件 | 检查路径是否正确挂载 |
| TSV不匹配 | 检查utter_id提取逻辑 |
| 没有候选terms | 检查glossary和RAG模型 |

## 📞 获取帮助

1. 查看详细文档: `README_term_map_v2_buzz.md`
2. 查看项目总结: `SUMMARY_term_map_v2_buzz.md`
3. 运行测试: `python test_term_map_logic.py`
4. 查看日志: 脚本会输出详细进度

## ✅ 验证输出

```bash
# 检查输出文件
wc -l /mnt/gemini/data1/jiaxuanluo/train_s_zh_with_candidates.jsonl

# 查看第一条数据
head -n 1 /mnt/gemini/data1/jiaxuanluo/train_s_zh_with_candidates.jsonl | jq .

# 验证term_map格式
grep -o "term_map:" /mnt/gemini/data1/jiaxuanluo/train_s_zh_with_candidates.jsonl | wc -l

# 随机抽样检查
shuf -n 3 /mnt/gemini/data1/jiaxuanluo/train_s_zh_with_candidates.jsonl | jq .
```

---

**最后更新**: 2025-12-26
**状态**: ✅ 已测试并验证


















