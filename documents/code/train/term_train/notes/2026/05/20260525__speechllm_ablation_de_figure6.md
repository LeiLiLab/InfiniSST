# Speech LLM Ablation Figure 6, En-De

## Hypothesis

The En-Zh Speech LLM ablation has small separations between several lines. The
En-De rows should make the difference between the historical LLM-generated
term-map SFT line and the current retriever-style RASST line easier to see.

## Background / Motivation

The earlier Figure 6 draft used En-Zh only. The user requested a German version.
This refresh keeps the same provenance rule: old rows come from the legacy
`main_result_tagged.pdf` data source, the old RASST line is explicitly labelled
`RASST (LLM-generated TM SFT)`, and the current `RASST` plus oracle upper bound
come from the current main-result TSV behind `new_main_result_tagged.pdf`.

## What changed vs baseline

- Replaced `figure_06_speechllm_placeholder/data.tsv` with En-De rows.
- Changed the plotting script default output stem to `speechllm_ablation_de`.
- Regenerated `speechllm_ablation_de.pdf` and `.png`, and copied them to
  `latex/figures/`.
- Updated `results.tex` and the plot registry to use the En-De Figure 6 asset.

## Expected metrics

The historical line should be labelled `RASST (LLM-generated TM SFT)` and use
the En-De rows from
`documents/code/simuleval/reports/20260521_tagged_acl_main_results_fourline_llmgen_rasst_data.tsv`.
The current `RASST` and oracle upper-bound rows should use En-De
`acl_tagged_raw` rows from
`documents/code/train/term_train/reports/EMNLP_26_InfiniSST_RAG_src/plot/figure_01_main_result_tagged/data.tsv`.

## Verdict

Success. The paper-facing files are:

- `documents/code/train/term_train/reports/EMNLP_26_InfiniSST_RAG_src/latex/figures/speechllm_ablation_de.pdf`
- `documents/code/train/term_train/reports/EMNLP_26_InfiniSST_RAG_src/latex/figures/speechllm_ablation_de.png`
