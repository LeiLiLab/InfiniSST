# Tagged ACL New V9 HN1024 Tau0.78 Raw zh lm1 Aries GPU01

## Hypothesis

New V9 assistant-side term-tag SFT with HN1024 retrieval at tau `0.78` should
provide the current zh tagged ACL RASST main-result lm1 raw readout when output
`<term>` tags are stripped before scoring.

## Background / Motivation

This is the first setting of the zh tagged ACL RASST main-result sweep:
`lm=1`, `lang=zh`, raw tagged ACL glossary.  The full target panel is
`lm=1,2,3,4`, but this event intentionally launches only lm1 first on Aries
GPU `0,1` so startup, W&B logging, and output paths can be verified before
launching the remaining settings.

## What changed vs baseline

- Speech LLM: New V9 assistant term-tag delay-clean HF export.
- Retriever: HN1024 `lh1b88kw` checkpoint.
- Threshold: tau `0.78`, top-k `10`, timeline lookback `1.92s`.
- Dataset/readout: tagged ACL `zh`, `lm=1`, raw glossary.
- Metric denominator: fixed raw tagged ACL glossary
  `acl6060_tagged_gt_raw_min_norm2.json`.
- Runtime glossary: same raw tagged ACL glossary.
- Output-side `<term>` markers are stripped before BLEU, StreamLAAL, and term
  metrics.
- Execution: direct detached Aries run on GPU `0,1`; outputs/logs/temp/cache
  are placed under `/mnt/gemini/data1` because `/mnt/aries/data7` is full.

## Expected metrics

One W&B run should be logged under family `tagged_acl_new_v9_hn1024_tau078`
with BLEU, StreamLAAL, TERM_ACC, REAL_TERM_ADOPT, TERM_FCR, and
SOURCE_TERM_SENT_FCR for `zh/lm1/raw`.

## Verdict

Completed and verified on Aries GPU `0,1`.  First launch failed before
generation because the original `/mnt/gemini/data1/...` temp path exceeded the
vLLM ZeroMQ IPC path limit.  Retry1 used the short temp path but failed during
vLLM EngineCore startup with the Aries `ShmRingBuffer` shared memory issue.
Retry2 used the short temp path plus explicit safer vLLM multi-GPU flags
(`VLLM_DISABLE_CUSTOM_ALL_REDUCE=1`, `VLLM_MOE_USE_DEEP_GEMM=0`,
`VLLM_USE_FUSED_MOE_GROUPED_TOPK=0`) and completed successfully.

Verified artifacts: `eval_results.tsv`, `instances.log`, and
`term_adoption.json`.  W&B logging succeeded as `simuleval_eval/5fhcqity`.
