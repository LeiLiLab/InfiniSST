# Tagged ACL New V9 HN1024 Tau0.0 Raw zh lm1-lm4 Aries GPU01 Sequential

## Hypothesis

Lowering the HN1024 retrieval threshold from tau `0.78` to tau `0.0` on tagged
ACL zh should expose the effect of permissive term-map retrieval while holding
the Speech LLM, raw tagged glossary, top-k, and timeline lookback fixed.

## Background / Motivation

The current main-result RASST zh rows use the New V9 assistant term-tag SFT
model with the HN1024 retriever and raw tagged ACL glossary.  This ablation runs
the same tagged ACL zh readout for `lm=1,2,3,4` with `RAG_SCORE_THRESHOLD=0.0`
on Aries, then compares the verified outputs against the main-result RASST rows
in `documents/code/simuleval/reports/20260524_main_result_data.tsv`.

## What changed vs baseline

- Speech LLM: unchanged New V9 assistant term-tag delay-clean HF export.
- Retriever checkpoint: unchanged HN1024 `lh1b88kw`.
- Runtime glossary and scoring glossary: unchanged raw tagged ACL glossary
  `acl6060_tagged_gt_raw_min_norm2.json`.
- Retrieval top-k: unchanged `10`.
- Timeline lookback: unchanged `1.92s`.
- Threshold: changed from tau `0.78` to tau `0.0`.
- Dataset/readout: tagged ACL `zh`, `lm=1,2,3,4`, full corpus, raw glossary.
- Execution: direct detached Aries run on GPU pair `0,1`, sequential across LMs.
- Storage: output and logs use `/mnt/taurus/data2/jiaxuanluo` because
  `/mnt/gemini/data1` was close to full at planning time.
- Temp: short `/mnt/taurus/data2/jiaxuanluo/tmp/jx_tacl0` because Aries `/tmp`
  had only about 2.3GB free during preflight; the temp path is still short
  enough for vLLM IPC.

## Expected metrics

Each LM should produce a verified `eval_results.tsv`, `instances.log`, and
`term_adoption.json`.  W&B logging is enabled if stable, but the completion
criterion is the verified TSV/log artifacts.

## Verdict

`lm1` completed generation on Aries GPU pair `0,1`.  The first pass produced
`instances.log`, `metrics.tsv`, and `scores.tsv` but did not produce the
standard `eval_results.tsv`; offline scoring was recovered from the completed
`instances.log` without rerunning generation.  The remaining `lm=2,3,4` readouts
were split into parallel Aries events:

- `20260526T033545__simuleval__tagged_acl_new_v9_hn1024_tau000_raw_zh_lm234_aries4567_parallel`
  for `lm2/lm3`.
- `20260526T034148__simuleval__tagged_acl_new_v9_hn1024_tau000_raw_zh_lm4_aries23_parallel`
  for `lm4`.
