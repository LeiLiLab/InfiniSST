# Speech LLM Ablation Figure 6, En-Zh Tau Comparison

## Hypothesis

The En-Zh ACL tagged raw tau ablation should be easier to read as a five-line
companion figure: two offline references, InfiniSST, RASST at the main
threshold, and RASST at tau=0.0.

## Background / Motivation

The existing En-Zh Speech LLM ablation figure has six lines and includes older
LLM-generated term-map SFT variants. The user requested a figure following that
style but limited to `Offline ST`, `Offline + GT Terms`, `InfiniSST`,
`RASST (tau=0.78)`, and `RASST (tau=0.0)`.

## What changed vs baseline

- Added `build_data_zh_tau_compare.py` to construct the five-line En-Zh TSV.
- Added `data_zh_tau_compare.tsv` under the paper-local Figure 6 package.
- Made `plot_figure_06_speechllm_ablation.py` validate and plot the methods
  present in a TSV, so existing En-De and En-Ja plots still regenerate.
- Generated `speechllm_ablation_zh_tau_compare.pdf` and `.png` locally and in
  `latex/figures/`.

## Expected metrics

The offline, InfiniSST, and `RASST (tau=0.78)` rows should come from
`figure_01_main_result_tagged/data.tsv`. The `RASST (tau=0.0)` rows should come
from `documents/code/simuleval/reports/20260526_tagged_acl_zh_tau000_vs_main_rasst.tsv`.

## Verdict

Success. The paper-facing files are:

- `documents/code/train/term_train/reports/EMNLP_26_InfiniSST_RAG_src/latex/figures/speechllm_ablation_zh_tau_compare.pdf`
- `documents/code/train/term_train/reports/EMNLP_26_InfiniSST_RAG_src/latex/figures/speechllm_ablation_zh_tau_compare.png`
