# Tagged ACL New V9 HN1024 Tau0.0 Raw zh lm4 Aries GPU23

## Hypothesis

Lowering the HN1024 retrieval threshold from tau `0.78` to tau `0.0` on tagged
ACL zh should expose the effect of permissive term-map retrieval while holding
the Speech LLM, raw tagged glossary, top-k, and timeline lookback fixed.

## Background / Motivation

The original `20260526T0210` readout started as a sequential Aries run for
`lm=1,2,3,4`; `lm=1` is running on GPU pair `0,1`.  The follow-up
`20260526T033545` event started `lm=2` and `lm=3` on GPU pairs `4,5` and `6,7`.
GPU pair `2,3` was also confirmed idle, so this event runs the remaining
`lm=4` readout immediately on `2,3`.

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
- Dataset/readout: tagged ACL `zh`, `lm=4`, full corpus, raw glossary.
- Execution: direct detached Aries run on GPU pair `2,3`; RAG is placed on the
  second VLLM GPU via `RAG_GPU_OVERRIDE=cuda:1`.
- Storage: output and logs use `/mnt/taurus/data2/jiaxuanluo` because
  `/mnt/gemini/data1` was close to full at planning time.
- Temp: short `/mnt/taurus/data2/jiaxuanluo/tmp/jx_tacl04`; the temp path is
  short enough for vLLM IPC.

## Expected metrics

The readout should produce a verified `eval_results.tsv`, `instances.log`, and
`term_adoption.json`.  W&B logging is enabled if stable, but the completion
criterion is the verified TSV/log artifacts.

## Verdict

Completed and verified.  `lm4` produced `eval_results.tsv`, `instances.log`,
and `term_adoption.json` on Aries GPU pair `2,3`.
