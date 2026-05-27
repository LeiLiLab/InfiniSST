# Origin bsz4 ACL main zh eval tau0

## Hypothesis

The original `gigaspeech-zh-s_origin-bsz4` Speech LLM should provide a stable
pre-TCM comparison point for the current ACL main zh results when paired with
the same RAG retriever, timeline mode, and full MaxSim window family.

## Background / Motivation

The current ACL main v2r32 sweep is complete. We need the same 5-paper x 4-lm x
3-glossary grid for the historical origin Speech LLM so BLEU, StreamLAAL,
TERM_ACC, RealAdopt, and FCR can be compared under the same RAG/eval pipeline.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/simuleval_eval/runs/3fic89wn
- Diff:
  - Speech LLM: new_v3/v2 variants -> `/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-zh-s_origin-bsz4`
  - RAG retriever: unchanged current main-result retriever
  - eval grid: 5 ACL papers x 4 latency multipliers x raw/gs1k/gs10k
  - RAG tau: `0.0`
  - compute: Taurus 3-GPU one-setting jobs

## Expected metrics

The run should establish the origin-SLM floor under the current retriever. We
expect lower term-aware metrics than v2r32 if the newer SLM learned better
constraint following, while StreamLAAL should remain comparable because the
latency multipliers and decoding pipeline are unchanged.

## Verdict

PENDING: update after all tau0 origin main-result jobs finish and the aggregate
summary is generated.
