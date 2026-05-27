# Tagged ACL Main Results Four-Line Figure

## Hypothesis

A four-line tagged ACL figure can make the latency-quality tradeoff easier to
read by placing offline ST, InfiniSST, current SLLM+RAG, and prior RASST on the
same StreamLAAL axis for zh, de, and ja.

## Background / Motivation

This analysis visualizes the completed tagged ACL origin-bsz4 tau=0.73 sweep at
`/mnt/gemini/data2/jiaxuanluo/tagged_acl_origin_bsz4_tau073_baseline_20260520T1010_ragreset_full_mp3/full`.
The current line is the no term-map SFT Speech LLM plus retriever RAG readout.

## What changed vs baseline

No model, data, or eval output was changed.  The plotting script parses current
raw-glossary `eval_results.tsv` rows and combines them with supplied offline,
InfiniSST, and previous RASST comparison tables.  RASST is explicitly a
placeholder from previous data until the updated RASST readout is available.

## Expected metrics

The figure should reproduce the user-supplied raw tagged ACL values for the
current SLLM+RAG line and keep StreamLAAL, BLEU, and tagged raw terminology
accuracy on the same axes as the existing main-result figure format.

## Verdict

Generated as `documents/code/simuleval/reports/20260520_tagged_acl_main_results_fourline.png`
and `.pdf`, with plot source rows in
`documents/code/simuleval/reports/20260520_tagged_acl_main_results_fourline_data.tsv`.
