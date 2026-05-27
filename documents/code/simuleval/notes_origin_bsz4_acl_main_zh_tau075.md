# Origin bsz4 ACL main zh eval tau075

## Hypothesis

The original `gigaspeech-zh-s_origin-bsz4` Speech LLM should expose how much of
the current ACL main zh performance comes from the Speech LLM rather than the
shared RAG retriever when tau-filtered retrieval is enabled.

## Background / Motivation

The current ACL main v2r32 sweep is complete. We need the same 5-paper x 4-lm x
3-glossary grid for the historical origin Speech LLM so tau-filtered RAG
behavior can be compared under identical evaluation and glossary settings.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/simuleval_eval/runs/djcp4rmt
- Diff:
  - Speech LLM: new_v3/v2 variants -> `/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-zh-s_origin-bsz4`
  - RAG retriever: unchanged current main-result retriever
  - eval grid: 5 ACL papers x 4 latency multipliers x raw/gs1k/gs10k
  - RAG tau: `0.75`
  - compute: Taurus 3-GPU one-setting jobs

## Expected metrics

The run should show whether tau0.75 filtering helps the origin SLM avoid noisy
term maps or instead removes useful context. BLEU may be more sensitive than
TERM_ACC if sparse term maps shift generation quality on individual papers.

## Verdict

PENDING: update after all tau0.75 origin main-result jobs finish and the
aggregate summary is generated.
