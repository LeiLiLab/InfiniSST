# Per-Paper Glossary Pipeline：按 Paper 单独跑 RAG SimulEval

本文档说明如何从"**paper PDF 提取 terms**"到"**按 paper 单独跑 SimulEval（每个 talk 用自己 paper 的子 glossary）**"的完整流程。

---

## 目标

- **之前**：5 篇 paper 的 glossary 合并成一个，所有 talk 共用（评估时也是全量 glossary 统一算 TERM_ACC）。
- **现在**：每篇 paper 独立一个子 glossary，SimulEval 推理时每个 talk 只用自己 paper 的 terms；对应的 RAG index 也是 per-paper 的。

---

## Pipeline 流程

### Step 1: 从 Paper PDF 提取 Terms（带翻译）

**脚本**：  
- `retriever/gigaspeech/data_pre/extract_acl_terms_from_paper_v2.py`  
- `documents/data/data_pre/extract_acl_terms_from_paper_v2.py`（两份同步）

**功能**：
- 读取 5 个 ACL paper PDF（`papers/*.pdf`）
- 逐个 PDF → text → 分 chunk 喂给 Gemini
- Gemini 返回 `[{"term": "...", "target_translations": {"zh":"...", "de":"...", "ja":"..."}}]`
- 每个 paper 单独输出一个 glossary JSON（**允许跨 paper 出现重复 term**）

**输出**：
- `extracted_glossaries_by_paper/extracted_glossary__<paper_id>.json`  
  （5 个文件，paper_id 例如 `2022.acl-long.110`）
- `extracted_glossary_lists_by_paper/extracted_glossary_list__<paper_id>.json`
- `extracted_glossary_by_paper_manifest.json`（汇总 term 数量）

**运行**：

```bash
cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre
# 或 cd /home/jiaxuanluo/InfiniSST/documents/data/data_pre

GEMINI_API_KEY=... python3 extract_acl_terms_from_paper_v2.py
```

**关键配置**（环境变量可调）：
- `MODEL_NAME`（默认 `gemini-2.0-flash-exp`）
- `MAX_OUTPUT_TOKENS`（默认 8192）
- `TEXT_CHUNK_CHARS`（默认 25000）
- `MAX_TEXT_CHUNKS`（默认 6）

**注意**：
- Glossary dict key 为小写（例如 `"bert"` → `{"term": "BERT", ...}`）
- `short_description/full_form/is_acronym` 留空（只保留 term + translations）
- `source_paper` 字段为 PDF 文件名（例如 `2022.acl-long.110.pdf`）

---

### Step 2: 准备 Per-Paper 的 SimulEval 输入（子 glossary + 子 source/target list）

**脚本**：  
`documents/code/simuleval/prepare_extracted_glossary_by_paper_inputs.py`

**功能**：
- 读取 Step 1 产出的 5 个子 glossary（从 manifest JSON 或直接扫目录）
- 读取 `dev.source` / `dev.target.zh`（5 行，对应 5 个 talk）
- 按 wav 文件名（例如 `2022.acl-long.110.wav`）和 glossary 的 `source_paper`（例如 `2022.acl-long.110.pdf`）对齐
- 为每个 paper 生成：
  - **子 glossary**（只保留该 paper 的 terms）
  - **子 dev.source**（只保留该 talk 的 wav 行）
  - **子 dev.target**（只保留该 talk 的 reference 行）
- 输出 `paper_inputs_map.json`（包含每个 paper 的文件路径映射）

**输出目录**（默认）：  
`/mnt/gemini/data2/jiaxuanluo/infinisst_omni_vllm_rag_rank32_iter_0000452_hf_zh_k1_10_k2_sweep_glossary2/zh/__paper_inputs__/`

**输出结构**：
```
__paper_inputs__/
├── glossaries/
│   ├── extracted_glossary_with_translations__2022.acl-long.110.json
│   ├── extracted_glossary_with_translations__2022.acl-long.117.json
│   ├── ...（5 个）
├── lists/
│   ├── dev.source__2022.acl-long.110.txt  （1 行）
│   ├── dev.target.zh__2022.acl-long.110.txt  （对应的 reference 行）
│   ├── ...（5 个 paper × 2 种文件）
└── paper_inputs_map.json
```

**运行**：

```bash
python3 /home/jiaxuanluo/InfiniSST/documents/code/simuleval/prepare_extracted_glossary_by_paper_inputs.py \
  --output-dir /mnt/gemini/data2/jiaxuanluo/infinisst_omni_vllm_rag_rank32_iter_0000452_hf_zh_k1_10_k2_sweep_glossary2/zh/__paper_inputs__ \
  --data-root /mnt/taurus/data/siqiouyang/datasets/acl6060 \
  --lang-code zh \
  --extracted-glossary-dir /home/jiaxuanluo/InfiniSST/documents/data/data_pre/extracted_glossaries_by_paper
```

（如果 manifest JSON 已存在，也可以直接用 `--extracted-glossary-manifest`）

---

### Step 3: Build Per-Paper 的 RAG Index

**脚本**：  
`documents/code/simuleval/build_indices_for_extracted_glossary_by_paper.sh`

**功能**：
- 读取 Step 2 产出的 `paper_inputs_map.json`
- 对每个 paper 的子 glossary，调用 `retriever/gigaspeech/run_build_index_v4.sh` 构建 index
- Index 命名规则：  
  `${MODEL_TAG}__${GLOSSARY_TAG}__tr16.pkl`  
  例如：`final_main_result_model_v1__extracted_glossary_with_translations__2022.acl-long.110__tr16.pkl`

**输出目录**（默认）：  
`/mnt/gemini/data2/jiaxuanluo/index_cache_v4/`

**运行**：

```bash
# 跑全部 5 个 paper
bash /home/jiaxuanluo/InfiniSST/documents/code/simuleval/build_indices_for_extracted_glossary_by_paper.sh

# 只 build 一个 paper（测试）
PAPER_IDS_OVERRIDE="2022.acl-long.110" \
bash /home/jiaxuanluo/InfiniSST/documents/code/simuleval/build_indices_for_extracted_glossary_by_paper.sh
```

**配置**（脚本内可改）：
- `RAG_MODEL_PATH`（默认 `/mnt/gemini/data2/jiaxuanluo/final_main_result_model_v1.pt`）
- `INDEX_CACHE_DIR`（默认 `/mnt/gemini/data2/jiaxuanluo/index_cache_v4`）

**注意**：
- 如果 index 已存在会自动跳过（避免重复 build）
- 每个 index 大约需要几分钟（取决于 glossary 大小和 GPU）

---

### Step 4: 按 Paper 跑 SimulEval（latency × K2 sweep，固定 K1=10）

**脚本**：  
`documents/code/simuleval/bypass_simuleval_rank32_iter_0000452_hf_zh_k1_10_k2_sweep_extracted_glossary_by_paper.sh`

**功能**：
- 读取 `paper_inputs_map.json`
- 对每个 paper_id，调用基础 sweep 脚本（`bypass_simuleval_rank32_iter_0000452_hf_zh_k1_10_k2_sweep_glossary2.sh`）
- 通过环境变量 override：
  - `GLOSSARY_PATHS_OVERRIDE`（指向该 paper 的子 glossary）
  - `SRC_LIST_OVERRIDE`（指向该 talk 的 dev.source 子列表）
  - `TGT_LIST_OVERRIDE`（指向该 talk 的 dev.target 子列表）
- 基础脚本会自动根据 glossary 路径找到对应的 index（前提是 Step 3 已 build）

**默认 sweep 配置**（继承自基础脚本）：
- latency_multipliers: `1, 2, 3, 4`
- K2: `5, 10, 15, 20`
- K1: `10`（固定）
- 2 个 glossary 类型（但这个脚本只跑 extracted，如需 acl6060 raw 需单独跑）

**输出目录结构**（示例）：
```
/mnt/gemini/data2/jiaxuanluo/infinisst_omni_vllm_rag_rank32_iter_0000452_hf_zh_k1_10_k2_sweep_glossary2/zh/
├── iter_0000452-hf_gextracted_glossary_with_translations__2022.acl-long.110_cs0.96_hs0.48_lm1_k25_k110/
│   ├── instances.log
│   ├── simuleval.log
│   └── vllm_debug.jsonl
├── iter_0000452-hf_gextracted_glossary_with_translations__2022.acl-long.110_cs0.96_hs0.48_lm1_k210_k110/
│   └── ...
├── ...（5 paper × 4 latency × 4 K2 = 80 个目录）
```

**运行**：

```bash
# 跑全部 5 个 paper × 4 latency × 4 K2（需要很长时间）
bash /home/jiaxuanluo/InfiniSST/documents/code/simuleval/bypass_simuleval_rank32_iter_0000452_hf_zh_k1_10_k2_sweep_extracted_glossary_by_paper.sh

# 只跑一个 paper + 限制 sweep 范围（快速测试）
PAPER_IDS_OVERRIDE="2022.acl-long.110" \
LATENCY_MULTIPLIERS_OVERRIDE="1" \
RAG_K2_VALUES_OVERRIDE="10" \
RESUME_MODE=1 \
bash /home/jiaxuanluo/InfiniSST/documents/code/simuleval/bypass_simuleval_rank32_iter_0000452_hf_zh_k1_10_k2_sweep_extracted_glossary_by_paper.sh
```

**常用环境变量 override**：
- `PAPER_IDS_OVERRIDE`（空格分隔，例如 `"2022.acl-long.110 2022.acl-long.117"`）
- `LATENCY_MULTIPLIERS_OVERRIDE`（例如 `"1 2"`）
- `RAG_K2_VALUES_OVERRIDE`（例如 `"5 10"`）
- `RESUME_MODE=1`（跳过已完成的 run，只跑缺失的）
- `CLEAN_OUTPUT_DIR_OVERRIDE=0`（resume 时推荐设为 0）

---

### Step 5: 汇总 TERM_ACC 等指标到 TSV

**脚本**（如需汇总全部结果）：  
可以复用或修改 `streamlaal_summary_k1_10_k2_sweep_glossary2.sh`，但需要适配"按 paper 拆分输出目录"的结构。

**或者直接用 FBK stream_laal_term.py 手动验证**：
```bash
python /mnt/taurus/home/jiaxuanluo/FBK-fairseq/examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py \
  --simuleval-instances <output_dir>/instances.log \
  --reference /mnt/taurus/data/siqiouyang/datasets/acl6060/dev/text/txt/ACL.6060.dev.en-xx.zh.txt \
  --audio-yaml /mnt/taurus/data/siqiouyang/datasets/acl6060/dev.yaml \
  --sacrebleu-tokenizer zh \
  --latency-unit char \
  --glossary <paper_specific_glossary.json> \
  --term-lang zh
```

**注意**：
- 由于现在是 per-paper 跑的，每个 `instances.log` 只有 **1 条样本**（对应该 talk）
- 如果你想得到"所有 5 个 paper 合并的 TERM_ACC"，需要自己写聚合脚本（把 5 个 paper 的 `TERM_CORRECT/TERM_TOTAL` 加起来）

---

## 完整 Pipeline 快速回顾

```
1. Extract terms from PDFs
   ├─ Input:  papers/*.pdf (5 files)
   └─ Output: extracted_glossaries_by_paper/extracted_glossary__<paper_id>.json (5 files)
              + manifest JSON
              
2. Prepare per-paper inputs
   ├─ Input:  Step 1 的 5 个 glossary + dev.source / dev.target.zh
   └─ Output: __paper_inputs__/glossaries/*.json (5 files)
              __paper_inputs__/lists/dev.source__*.txt (5 files)
              __paper_inputs__/lists/dev.target.zh__*.txt (5 files)
              + paper_inputs_map.json
              
3. Build per-paper RAG indices
   ├─ Input:  Step 2 的 5 个子 glossary
   └─ Output: index_cache_v4/*__extracted_glossary_with_translations__<paper_id>__tr16.pkl (5 files)
   
4. Run SimulEval (per paper)
   ├─ Input:  Step 3 的 5 个 index + Step 2 的 5 对 src/tgt list
   └─ Output: 每个 paper × 4 latency × 4 K2 = 80 个 instances.log
   
5. Summarize results (optional)
   └─ Aggregate TERM_ACC from 80 runs
```

---

## 常见问题

**Q1: 为什么要按 paper 拆分？**  
A: 因为原始 extracted glossary 里不同 paper 的 term 合并到一起，导致每个 talk 在评估时"看到了不属于自己 paper 的 terms"，TERM_ACC 会偏高。现在按 paper 拆分后，每个 talk 只评估自己 paper 里提到的 terms，更接近真实场景。

**Q2: Step 3 的 index build 会不会很慢？**  
A: 每个 paper 的 glossary 大约 10-60 个 term，build 一个 index 大约 1-3 分钟（取决于 GPU 和 model size）。5 个 paper 总共不到 15 分钟。

**Q3: Step 4 跑 SimulEval 要多久？**  
A: 每个 talk 只有 1 条样本（~10 分钟长的 audio），但 4×4=16 种参数组合，5 个 paper 总共 80 个 run。如果每个 run 平均 15 分钟，总时长约 20 小时（可以用 `RESUME_MODE=1` 增量跑）。

**Q4: 怎么和之前的"合并 glossary"结果对比？**  
A: 
- 之前（合并）：`bypass_simuleval_rank32_iter_0000452_hf_zh_k1_10_k2_sweep_glossary2.sh` + `GLOSSARY_PATHS_OVERRIDE` 指向全量 `extracted_glossary_with_translations.json`
- 现在（拆分）：本文档的 Step 4 脚本
- 对比：把两边的 TERM_ACC 汇总到同一个 TSV，看 per-paper 的 TERM_ACC 是否比全量低（符合预期）

---

## 相关文件索引

- **Extract terms**:  
  `retriever/gigaspeech/data_pre/extract_acl_terms_from_paper_v2.py`  
  `documents/data/data_pre/extract_acl_terms_from_paper_v2.py`

- **Prepare inputs**:  
  `documents/code/simuleval/prepare_extracted_glossary_by_paper_inputs.py`

- **Build indices**:  
  `documents/code/simuleval/build_indices_for_extracted_glossary_by_paper.sh`  
  `retriever/gigaspeech/run_build_index_v4.sh`

- **Run SimulEval**:  
  `documents/code/simuleval/bypass_simuleval_rank32_iter_0000452_hf_zh_k1_10_k2_sweep_extracted_glossary_by_paper.sh`  
  `documents/code/simuleval/bypass_simuleval_rank32_iter_0000452_hf_zh_k1_10_k2_sweep_glossary2.sh`（基础脚本）

- **Summarize results**:  
  `documents/code/simuleval/streamlaal_summary_k1_10_k2_sweep_glossary2.sh`（需修改适配 per-paper）

- **说明文档**:  
  `documents/code/simuleval/reeval_streamlaal_by_k1_10_k2_sweep_glossary_extracted_paper.md`（本文档）

