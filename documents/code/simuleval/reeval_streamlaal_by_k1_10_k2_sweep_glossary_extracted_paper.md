@bypass_simuleval_rank32_iter_0000452_hf_zh_k1_10_k2_sweep_glossary2.sh @fix_k1_10_sweep_k2_result.md 
之前通过这个脚本产生了simuleval的数据, 然后通过 @streamlaal_summary_k1_10_k2_sweep_glossary2.sh 生成了streamlaal的结果.


现在要针对paper extracted glossary: /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/extracted_glossary_with_translations.json
进行如下 **SimulEval 运行时（RAG）** 的修改（不是 post-eval / streamlaal 的参数修改）:
glossary里有source_paper字段表示来自哪个paper.
而simuleval时, 有个source list:SRC_LIST, 可以根据id找到对应的paper:
/mnt/taurus/data/siqiouyang/datasets/acl6060$ cat dev.source
/mnt/data/siqiouyang/datasets/acl6060/dev/full_wavs/2022.acl-long.268.wav
/mnt/data/siqiouyang/datasets/acl6060/dev/full_wavs/2022.acl-long.367.wav
/mnt/data/siqiouyang/datasets/acl6060/dev/full_wavs/2022.acl-long.590.wav
/mnt/data/siqiouyang/datasets/acl6060/dev/full_wavs/2022.acl-long.110.wav
/mnt/data/siqiouyang/datasets/acl6060/dev/full_wavs/2022.acl-long.117.wav
现在是整个paper的glossary合到一起作为eval, 现在要改成针对每个talk, 单独用对应paper source的glossary.
相当于拆成5个子glossary, 然后根据source list用对应的子glossary算结果.


可以参照这个脚本:/home/jiaxuanluo/InfiniSST/documents/code/simuleval/bypass_simuleval_rank32_iter_0000452_hf_zh_k1_10_k2_sweep_glossary2_resume_extracted_only.sh

实现我的需求.

---

### 实现方式（SimulEval 运行时按 talk/paper 切换 glossary/index）

现在 `extracted_glossary_with_translations.json` 里每个 term 都带 `source_paper`（例如 `2022.acl-long.110.pdf`）。SimulEval 的输入 `dev.source` 里也能拿到每条样本的 wav（例如 `.../2022.acl-long.110.wav`）。

因此我们把 **SimulEval 推理阶段使用的 glossary/index** 从“全量合并 glossary（单一 index）”改成：

- **按 paper_id（talk id）拆 dev.source/dev.target**（用 wav 文件名提取 `2022.acl-long.110` 这类 id）
- **按 paper_id 拆 extracted glossary**（用 `source_paper` 提取同样的 id）
- **对每个 paper_id 运行 simuleval**（传入该 paper 的 glossary/index + 该 paper 的 source/target 子列表）

### 新增脚本（按 paper 跑 simuleval）

- **准备 per-paper glossary + per-paper dev lists**：`documents/code/simuleval/prepare_extracted_glossary_by_paper_inputs.py`
- **按 paper_id 调基础 sweep 脚本跑 simuleval**：`documents/code/simuleval/bypass_simuleval_rank32_iter_0000452_hf_zh_k1_10_k2_sweep_extracted_glossary_by_paper.sh`
- **基础 sweep 脚本新增 override（供 wrapper 使用）**：`documents/code/simuleval/bypass_simuleval_rank32_iter_0000452_hf_zh_k1_10_k2_sweep_glossary2.sh`
  - 新增环境变量：`SRC_LIST_OVERRIDE`、`TGT_LIST_OVERRIDE`

运行：

```bash
bash /home/jiaxuanluo/InfiniSST/documents/code/simuleval/bypass_simuleval_rank32_iter_0000452_hf_zh_k1_10_k2_sweep_extracted_glossary_by_paper.sh
```

常用 override：

```bash
PAPER_IDS_OVERRIDE="2022.acl-long.268" \
LATENCY_MULTIPLIERS_OVERRIDE="1" \
RAG_K2_VALUES_OVERRIDE="10" \
RESUME_MODE=1 \
bash /home/jiaxuanluo/InfiniSST/documents/code/simuleval/bypass_simuleval_rank32_iter_0000452_hf_zh_k1_10_k2_sweep_extracted_glossary_by_paper.sh
```

输出：

- 每个 paper_id 会生成一套 simuleval 输出目录（目录名里会包含该 paper 的 glossary tag，例如 `..._gextracted_glossary_with_translations__2022.acl-long.268_...`）
- per-paper 生成的 glossary/list 文件默认放在：  
  `/mnt/gemini/data2/jiaxuanluo/infinisst_omni_vllm_rag_rank32_iter_0000452_hf_zh_k1_10_k2_sweep_glossary2/zh/__paper_inputs__/`














脚本命令如下:


你说得对：我刚才做的是“**后处理阶段**按 paper 拆 glossary 去算 TERM”，这并不会改变 **SimulEval/RAG 在推理时用到的 glossary**。你要的是在 **SimulEval 运行时**，让每个 talk 用它自己 paper 的子 glossary（也就意味着 `--rag-index-path/--rag-model-path` 指向“按 paper 构建的 index”），这确实应该改 **simuleval 的参数/运行方式**。  
接下来我会做两件事：先把 extracted glossary 按 `source_paper` 自动拆成多个子 glossary（5 个 talk 对应 5 个 json），然后新增一个 simuleval 运行脚本：对每个 talk 生成对应的 `dev.source/dev.target` 子列表、选择对应的 index，跑完整的 latency×K2 sweep，输出目录里带上 paper_id，方便后续汇总。

[3 tools called]

你说得完全正确：如果目标是“**让 RAG 在 SimulEval 推理阶段就只看到该 talk 对应 paper 的 glossary**”，那就必须在 **SimulEval 运行时**切换 `--rag-index-path`（以及对应 glossary/index），而不是只改后面的 `stream_laal_term.py`。我马上把实现方向切到“按 talk 拆分 dev.source/dev.target + 按 paper 拆分 extracted glossary + 每个 talk 跑一套 simuleval sweep（参数层面变化）”，并在现有脚本上加最小的 override 接口，避免大量复制代码。

[10 tools called]

### 你说得对：已经把实现切到 **SimulEval 运行时参数/输入**（不是 streamlaal）

我已做了这套改动，让你能在 **SimulEval 推理阶段**按 talk/paper 切换 extracted 子 glossary（从而切换对应 `--rag-index-path`）：

- **基础 sweep 脚本支持按 talk 覆盖 source/target list**  
  已在 `documents/code/simuleval/bypass_simuleval_rank32_iter_0000452_hf_zh_k1_10_k2_sweep_glossary2.sh` 新增环境变量：
  - `SRC_LIST_OVERRIDE`
  - `TGT_LIST_OVERRIDE`

- **准备脚本：拆 5 个子 glossary + 拆 5 份 dev.source/dev.target 子列表（严格对齐行号）**  
  新增：`documents/code/simuleval/prepare_extracted_glossary_by_paper_inputs.py`  
  会在输出目录下生成：
  - `glossaries/extracted_glossary_with_translations__<paper_id>.json`
  - `lists/dev.source__<paper_id>.txt`、`lists/dev.target.zh__<paper_id>.txt`
  - `paper_inputs_map.json`

- **SimulEval runner：按 paper_id 逐个调用基础 sweep 脚本（从而改变 simuleval 的输入/参数）**  
  新增：`documents/code/simuleval/bypass_simuleval_rank32_iter_0000452_hf_zh_k1_10_k2_sweep_extracted_glossary_by_paper.sh`

### 运行方式

跑全量 5 个 talk：

```bash
bash /home/jiaxuanluo/InfiniSST/documents/code/simuleval/bypass_simuleval_rank32_iter_0000452_hf_zh_k1_10_k2_sweep_extracted_glossary_by_paper.sh
```

只跑一个 talk（示例）+ 限制 sweep：

```bash
PAPER_IDS_OVERRIDE="2022.acl-long.268" \
LATENCY_MULTIPLIERS_OVERRIDE="1" \
RAG_K2_VALUES_OVERRIDE="10" \
RESUME_MODE=1 \
bash /home/jiaxuanluo/InfiniSST/documents/code/simuleval/bypass_simuleval_rank32_iter_0000452_hf_zh_k1_10_k2_sweep_extracted_glossary_by_paper.sh
```

### 文档已按你的澄清更新

`documents/code/simuleval/reeval_streamlaal_by_k1_10_k2_sweep_glossary_extracted_paper.md` 已改成明确说明这是 **SimulEval 运行时（RAG）修改**。