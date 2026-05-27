# Tagged ACL New V9 HN1024 Tau0.0 Raw zh lm2-lm4 Aries GPU4567 Parallel

## Hypothesis

Lowering the HN1024 retrieval threshold from tau `0.78` to tau `0.0` on tagged
ACL zh should expose the effect of permissive term-map retrieval while holding
the Speech LLM, raw tagged glossary, top-k, and timeline lookback fixed.

## Background / Motivation

The original `20260526T0210` readout started as a sequential Aries run for
`lm=1,2,3,4`; `lm=1` ran on GPU pair `0,1`.  Aries GPU pairs `4,5` and `6,7`
were confirmed idle, so this event parallelizes `lm=2,3` with two concurrent
workers.  GPU pair `2,3` was also confirmed idle, so `lm4` was split into
`20260526T034148__simuleval__tagged_acl_new_v9_hn1024_tau000_raw_zh_lm4_aries23_parallel`.

The final comparison target is the main-result RASST zh rows in
`documents/code/simuleval/reports/20260524_main_result_data.tsv`.

## What changed vs baseline

- Speech LLM: unchanged New V9 assistant term-tag delay-clean HF export.
- Retriever checkpoint: unchanged HN1024 `lh1b88kw`.
- Runtime glossary and scoring glossary: unchanged raw tagged ACL glossary
  `acl6060_tagged_gt_raw_min_norm2.json`.
- Retrieval top-k: unchanged `10`.
- Timeline lookback: unchanged `1.92s`.
- Threshold: changed from tau `0.78` to tau `0.0`.
- Dataset/readout: tagged ACL `zh`, `lm=2,3`, full corpus, raw glossary.
- Execution: direct detached Aries run on GPU pairs `4,5` and `6,7`, with
  `MAX_PARALLEL=2`; the parent scheduler shell was stopped after `lm2/lm3`
  startup verification so it will not queue a duplicate `lm4`.
- Storage: output and logs use `/mnt/taurus/data2/jiaxuanluo` because
  `/mnt/gemini/data1` was close to full at planning time.
- Temp: short `/mnt/taurus/data2/jiaxuanluo/tmp/jx_tacl0p` because Aries `/tmp`
  had only about 2.3GB free during the preceding preflight; the temp path is
  still short enough for vLLM IPC.

## Expected metrics

Each LM should produce a verified `eval_results.tsv`, `instances.log`, and
`term_adoption.json`.  W&B logging is enabled if stable, but the completion
criterion is the verified TSV/log artifacts.

## Verdict

Completed and verified for `lm2` and `lm3`.  Both produced `eval_results.tsv`,
`instances.log`, and `term_adoption.json`.  The parent scheduler shell was
stopped after startup so this event did not queue a duplicate `lm4`; `lm4` is
handled by `20260526T034148__simuleval__tagged_acl_new_v9_hn1024_tau000_raw_zh_lm4_aries23_parallel`.
