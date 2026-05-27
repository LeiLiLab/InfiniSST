# Speech LLM Ablation Figure 6 Companion, En-Ja

## Hypothesis

The En-Ja rows may provide an additional view of the Speech LLM term-map
supervision effect after the En-De figure made the comparison more visible than
the original En-Zh draft.

## Background / Motivation

The user requested a Japanese version after the En-De Figure 6 refresh. This
companion keeps the same provenance rule: old rows come from the legacy
`main_result_tagged.pdf` data source, the old RASST line is explicitly labelled
`RASST (LLM-generated TM SFT)`, and the current `RASST` plus oracle upper bound
come from the current main-result TSV behind `new_main_result_tagged.pdf`.

## What changed vs baseline

- Added `figure_06_speechllm_placeholder/data_ja.tsv` with En-Ja rows.
- Updated the plotting script so `--update-paper` can write a requested paper
  stem instead of always writing the En-De filename.
- Regenerated `speechllm_ablation_ja.pdf` and `.png`, and copied them to
  `latex/figures/`.
- Updated the paper-local plot registry and Figure 6 package README to record
  the En-Ja companion asset.

## Expected metrics

The historical line should be labelled `RASST (LLM-generated TM SFT)` and use
the En-Ja rows from
`documents/code/simuleval/reports/20260521_tagged_acl_main_results_fourline_llmgen_rasst_data.tsv`.
The current `RASST` and oracle upper-bound rows should use En-Ja
`acl_tagged_raw` rows from
`documents/code/train/term_train/reports/EMNLP_26_InfiniSST_RAG_src/plot/figure_01_main_result_tagged/data.tsv`.

## Verdict

Success. The paper-facing files are:

- `documents/code/train/term_train/reports/EMNLP_26_InfiniSST_RAG_src/latex/figures/speechllm_ablation_ja.pdf`
- `documents/code/train/term_train/reports/EMNLP_26_InfiniSST_RAG_src/latex/figures/speechllm_ablation_ja.png`
