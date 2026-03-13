计划：先把你提到的 `qwen3_AuT_BGE_M3_train_lora_unfrozen_text.py` 里“recall hit”的实现定位出来，确认它用什么特征（audio embedding / text embedding / ANN index）和 hit 逻辑。然后读取几行 `DEV_JSONL` 看字段结构，最后新增一个离线评测脚本：过滤空 term 样本、构建 glossary、对 K1 做 sweep，输出 `tsv + png`（recall@K1 曲线）。

[51 tools called]

已经按 `offline_evaluation_desc.md` 做完并跑通了 **GigaSpeech dev（`term_dev_dataset_final.jsonl`）的 Recall@K1 饱和曲线**，产物在下面这个目录：

- **输出目录**：`/mnt/gemini/data2/jiaxuanluo/offline_eval_k1_saturation_gigaspeech_dev/`
- **曲线图（PNG）**：`/mnt/gemini/data2/jiaxuanluo/offline_eval_k1_saturation_gigaspeech_dev/recall_k1_saturation.png`
- **数据表（TSV）**：`/mnt/gemini/data2/jiaxuanluo/offline_eval_k1_saturation_gigaspeech_dev/recall_k1_saturation.tsv`

### 结果（recall@K1）
TSV 里的关键点如下（严格 multi-positive recall：分母=所有 chunk 的正例 term 总数 `1918`）：

- **K1=1**：0.653806  
- **K1=2**：0.817518  
- **K1=3**：0.875391  
- **K1=5**：0.906674  
- **K1=10**：0.935349  
- **K1=15**：0.946298  
- **K1=20**：0.955683  
- **K1=50**：0.973931  
- **K1=100**：0.985401  

**饱和趋势**：K1 从 1→10 提升很大（~0.65→~0.94），**10→20** 仍有提升但开始放缓（~0.94→~0.96），**20 之后边际收益明显变小**。

### 你后续怎么复现/再跑
我新增了 2 个脚本：

- **Python 主脚本**：`documents/code/offline_evaluation/gigaspeech_recall_k1_saturation.py`
- **Slurm 提交脚本（推荐）**：`documents/code/offline_evaluation/sbatch_gigaspeech_recall_k1_saturation.sh`

在登录节点直接跑会遇到 “No CUDA GPUs are available”，所以建议：

```bash
sbatch /home/jiaxuanluo/InfiniSST/documents/code/offline_evaluation/sbatch_gigaspeech_recall_k1_saturation.sh
```

如果你希望我把图再额外画一条“增量提升（Δ recall）随 K1 变化”的曲线，用来更直观看饱和点，我也可以直接补上。