## Hypothesis

Collect the current En-De tagged ACL result surface so decode-cap, cache, SLM, and retriever-threshold comparisons can be discussed without mixing incompatible runs.

## Background / Motivation

Several En-De probes were run to diagnose why RASST improves TERM_ACC but often fails to improve BLEU relative to the verified no-RAG streaming baseline. The user specifically asked to compare cap16, termwrap TM-SFT, old TM-SFT, and related tau/cache settings.

## What changed vs baseline

This is an analysis-only inventory. It does not create new metrics. It indexes verified artifact-backed rows, explicitly marks running rows, and separates prompt-provided legacy references from rerun/readout artifacts.

## Expected metrics

The report should expose which configurations pass the BLEU gate and which only improve TERM_ACC. It should also make clear when a proposed setting, such as old TM-SFT at tau=0.0, does not yet have verified evidence.

## Verdict

Current inventory is written to `documents/code/simuleval/reports/20260525_de_tagged_acl_result_inventory.tsv`. Rows marked `running` are not final metric truth and must be replaced only after their summary artifacts exist.
