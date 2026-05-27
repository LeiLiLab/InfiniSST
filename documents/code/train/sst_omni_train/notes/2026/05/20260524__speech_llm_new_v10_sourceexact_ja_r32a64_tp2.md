## Canceled / Do Not Use

This branch is canceled. It incorrectly followed the v13/sourceexact route.
The final zh Speech LLM lineage is old new_v3 r32 -> new_v4 -> new_v5 ->
new_v9, so ja must follow that lineage instead. Do not train or report this
setting.

## Hypothesis

Training ja Speech LLM on clean source-exact New V10 term-map data should avoid
the GT pollution and hallucinated assistant tagging seen in the old New V9 ja
pipeline.

## Background / Motivation

Old ja New V9 used GT terms derived from LLM-generated term maps.  The clean
New V10 dataset instead uses source chunk ASR exact matching and exact future
target evidence before retriever term maps, LLM variants, no-GT-zero, and
assistant target tags are applied.

## What changed vs baseline

- Dataset:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_new_v10_sourceexact_ja_20260524/train_s_ja_new_v10_sourceexact_llmvariant_no_gt_zero_termtag_boundary.jsonl`
- Validation:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_new_v10_sourceexact_ja_20260524/new_v10_sourceexact_ja_summary.json`
- LoRA rank/alpha: r32/a64, TP=2, EP=4, one epoch.

## Expected metrics

Quick tagged-ACL ja raw lm=1..4 should reduce noisy term-map over-adoption
relative to polluted New V9 while preserving strict term adoption.

## Verdict

Pending.
