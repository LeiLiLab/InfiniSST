# ACL Paper-Extracted Union zh Origin Baseline and New V9 HN1024

## Hypothesis

Using one union paper-extracted glossary should make the ACL paper-extracted zh readout cheaper and cleaner than the old per-paper task split while keeping the strict raw extracted-term denominator fixed.

## Background / Motivation

The previous ACL paper-extracted run evaluated one paper glossary at a time.  This rerun uses the latest zh new_v9 speech LLM with the HN1024 MaxSim retriever and compares it against the old origin/no_tmsft no-RAG SLM baseline for `lm=1..4`.

## What changed vs baseline

- Baseline: old `gigaspeech-zh-s_origin-bsz4` model, RAG disabled, `max_new_tokens=80`.
- RASST: zh new_v9 assistant termtag delay clean model plus HN1024 MaxSim retriever, `tau=0.78`, `top_k=10`, `max_new_tokens=80`.
- Runtime glossaries: union raw, union gs1k, union gs10k.
- TERM metrics: fixed to `acl6060_paper_extracted_union_raw_zh.json` for every setting.

## Expected metrics

Baseline should provide the no-RAG SLM reference for `lm=1..4`.  RASST should improve TERM_ACC on at least one runtime glossary without silently changing the metric denominator; BLEU and StreamLAAL should be inspected as readout metrics rather than used for tau or checkpoint selection.

## Verdict

Pending.  Fill after all baseline and RASST workers produce W&B runs and non-empty `eval_results.tsv` files.
