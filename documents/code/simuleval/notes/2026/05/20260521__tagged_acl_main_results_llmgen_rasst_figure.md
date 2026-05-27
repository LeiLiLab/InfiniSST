# Tagged ACL Main Results With LLM-Generated RASST

## Hypothesis

Replacing the placeholder RASST line with the completed LLM-generated term-map
SFT readout should provide the current four-line tagged ACL comparison while
preserving the same StreamLAAL, BLEU, and tagged raw terminology accuracy axes.

## Background / Motivation

This is the second version of the tagged ACL main-results figure.  It uses the
same offline ST, InfiniSST, and current no term-map SFT SLLM+RAG lines as the
first figure, but replaces the old RASST placeholder line with the completed
`tagged_acl_llmgen_bsz4_tau073` full-grid summary.

## What changed vs baseline

The plotting script now accepts `--rasst-summary-tsv` and parses the selected
`gs=raw` rows from the exact summary TSV at
`/mnt/aries/data7/jiaxuanluo/slm/tagged_acl_llmgen_sft_tau073_full_20260521T050150/__summary__/summary_metrics_llmgen_sft_full.tsv`.
The sibling markdown report remains the human-readable result path.
No eval results are recomputed.

## Expected metrics

The RASST line should match the summary markdown raw rows for zh, de, and ja at
latency multipliers 1 through 4.  Other lines should remain identical to the
first four-line figure.

## Verdict

Generated as `documents/code/simuleval/reports/20260521_tagged_acl_main_results_fourline_llmgen_rasst.png`
and `.pdf`, with plot source rows in
`documents/code/simuleval/reports/20260521_tagged_acl_main_results_fourline_llmgen_rasst_data.tsv`.
