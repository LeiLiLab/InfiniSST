我会按你给的 `bypass_simuleval_rank32_iter_0000452_hf_zh_topk2_ablation.sh` 风格，新建一套“**双 glossary × latency(1/2/3/4) × K2(5/10/15/20)，固定 K1=10**”的 simuleval 运行脚本，再配一个 post-eval 脚本：对每个输出目录的 `instances.log` 调 `stream_laal_term.py`，把 **TERM_ACC** 等指标汇总成一个 TSV。接下来我直接落两个脚本到 `documents/code/simuleval/`。

[3 tools called]

已按你要求实现“**latency(1/2/3/4) × K2(5/10/15/20)，固定 K1=10，双 glossary（acl6060 raw + paper extracted）**跑 simuleval，然后用 `stream_laal_term.py` 汇总 TERM_ACC 到 TSV”的整套脚本。

### 1) 跑 simuleval（双 glossary × latency × K2，固定 K1=10）
- **脚本**：`documents/code/simuleval/bypass_simuleval_rank32_iter_0000452_hf_zh_k1_10_k2_sweep_glossary2.sh`
- **输出根目录**（可在脚本配置区改）：  
  `OUTPUT_BASE="/mnt/gemini/data2/jiaxuanluo/infinisst_omni_vllm_rag_rank32_iter_0000452_hf_zh_k1_10_k2_sweep_glossary2"`
- **目录命名**包含：`g<glossary_tag>_cs<vllm_segment_sec>_lm<latency>_k2<K2>_k1<K1>`，避免不同设置互相覆盖。

运行：

```bash
bash /home/jiaxuanluo/InfiniSST/documents/code/simuleval/bypass_simuleval_rank32_iter_0000452_hf_zh_k1_10_k2_sweep_glossary2.sh
```

### 2) 跑 stream_laal_term 并汇总 TSV（TERM_ACC 等）
- **脚本**：`documents/code/simuleval/streamlaal_summary_k1_10_k2_sweep_glossary2.sh`
- **汇总 TSV 输出**：  
  `/mnt/gemini/data2/jiaxuanluo/infinisst_omni_vllm_rag_rank32_iter_0000452_hf_zh_k1_10_k2_sweep_glossary2/zh/k1_10_k2_sweep_glossary2_streamlaal_summary.tsv`

运行：

```bash
bash /home/jiaxuanluo/InfiniSST/documents/code/simuleval/streamlaal_summary_k1_10_k2_sweep_glossary2.sh
```

如果你希望 TSV 里再加上 `StreamLAAL`/`BLEU` 之外的字段（比如 `eval_rc`、或把 `post_eval_raw_json` 也写进去），我可以再按你的格式偏好补一版。