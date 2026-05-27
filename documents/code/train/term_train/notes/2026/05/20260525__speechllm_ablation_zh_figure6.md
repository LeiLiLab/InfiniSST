# Speech LLM Ablation Figure 6, En-Zh

## Hypothesis

The Speech LLM ablation is clearer as a single En-Zh figure that separates the
older LLM-generated term-map SFT line from the current retriever-style RASST
line.

## Background / Motivation

The older `main_result_tagged.pdf` figure already contains En-Zh rows for
Offline ST, InfiniSST, SLLM+RAG without TM-SFT, and a RASST line backed by the
LLM-generated term-map SFT summary. For the paper ablation, that older RASST
line should be labelled explicitly as `RASST (LLM-generated TM SFT)`. The true
current `RASST` line should come from the En-Zh main result in
`new_main_result_tagged.pdf`.

## What changed vs baseline

- Added `figure_06_speechllm_placeholder/data.tsv` with En-Zh-only plotting
  rows.
- Added `plot_figure_06_speechllm_ablation.py`.
- Regenerated `speechllm_ablation_zh.pdf` and `.png`, and copied them to
  `latex/figures/`.
- Replaced the inline LaTeX placeholder for Figure 6 with the generated PDF.
- Added the blue oracle-term upper-bound line from the current main-result TSV.

## Expected metrics

The figure should preserve the old En-Zh main-result rows for Offline ST,
InfiniSST, SLLM+RAG without TM-SFT, and `RASST (LLM-generated TM SFT)`. The
current `RASST` rows and oracle upper bound should match the En-Zh rows in
`plot/figure_01_main_result_tagged/data.tsv`.

## Verdict

Success. The paper-facing files are:

- `documents/code/train/term_train/reports/EMNLP_26_InfiniSST_RAG_src/latex/figures/speechllm_ablation_zh.pdf`
- `documents/code/train/term_train/reports/EMNLP_26_InfiniSST_RAG_src/latex/figures/speechllm_ablation_zh.png`
