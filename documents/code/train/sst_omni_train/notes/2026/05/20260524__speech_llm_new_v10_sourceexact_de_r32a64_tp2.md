## Canceled / Do Not Use

This branch is canceled. It incorrectly followed the v13/sourceexact route.
The final zh Speech LLM lineage is old new_v3 r32 -> new_v4 -> new_v5 ->
new_v9, so de must follow that lineage instead. Do not train or report this
setting.

## Hypothesis

Training de Speech LLM on clean source-exact New V10 term-map data should avoid
the BLEU and TERM_ACC regressions caused by fuzzy GT pollution in the old New V9
de data.

## Background / Motivation

Old de New V9 used GT terms derived from LLM-generated term maps.  The clean
New V10 dataset instead uses source chunk ASR exact matching and exact future
target evidence before retriever term maps, LLM variants, no-GT-zero, and
assistant target tags are applied.

## What changed vs baseline

- Dataset:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_new_v10_sourceexact_de_20260524/train_s_de_new_v10_sourceexact_llmvariant_no_gt_zero_termtag_boundary.jsonl`
- Validation:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_new_v10_sourceexact_de_20260524/new_v10_sourceexact_de_summary.json`
- LoRA rank/alpha: r32/a64, TP=2, EP=4, one epoch.

## Expected metrics

Quick tagged-ACL de raw lm=1..4 should recover BLEU relative to polluted New V9
while preserving or improving real term adoption.

## Verdict

Pending.
