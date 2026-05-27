## Hypothesis

The multi-scale inference ablation should be represented as a paper-local figure,
not only as a TSV/report table, because the main contrast is the robustness gap
between full MaxSim windows, largest-window-only inference, and dense one-shot
retrieval.

## Background / Motivation

The source ablation table is
`documents/code/train/term_train/reports/figures/20260525_multiscale_inference_ablation_devraw.tsv`.
It combines the verified multi-scale MaxSim reference, the inference-only
largest-window MaxSim run, and the historical dense 1.92s single-embedding
trained retriever.

## What changed vs baseline

Added a paper-local plotting folder under
`reports/EMNLP_26_InfiniSST_RAG_src/plot/figure_10_multiscale_inference_ablation/`,
generated `multiscale_inference_ablation_devraw.pdf/.png`, inserted the figure
into `latex/sections/results.tex`, and updated the plot registry.

## Expected metrics

The plotted values should match the frozen TSV: full multi-scale MaxSim reaches
98.58 Recall@10 and 98.42 filtered recall on the GS-100k bank; largest-window
only reaches 96.07 and 93.62; dense 1.92s reaches 92.65 and 23.94.

## Verdict

Paper figure and related ablation prose were generated from the frozen TSV and
linked into the paper.
