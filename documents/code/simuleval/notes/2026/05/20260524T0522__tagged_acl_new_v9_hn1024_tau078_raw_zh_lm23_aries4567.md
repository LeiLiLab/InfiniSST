# Tagged ACL New V9 HN1024 Tau0.78 Raw zh lm2-lm3 Aries GPU4567

## Hypothesis

New V9 assistant-side term-tag SFT with HN1024 retrieval at tau `0.78` should
provide the current zh tagged ACL RASST main-result readouts for `lm=2` and
`lm=3` under the fixed raw tagged denominator.

## Background / Motivation

This continues the zh tagged ACL main-result sweep after launching `lm=1`.
The requested settings are `lm=2` and `lm=3`, both `lang=zh` and raw glossary.
They are run concurrently on Aries GPU pairs `4,5` and `6,7`; the existing
`lm=1` run stays on GPU `0,1`.

## What changed vs baseline

- Speech LLM: New V9 assistant term-tag delay-clean HF export.
- Retriever: HN1024 `lh1b88kw` checkpoint.
- Threshold: tau `0.78`, top-k `10`, timeline lookback `1.92s`.
- Dataset/readout: tagged ACL `zh`, `lm=2,3`, raw glossary.
- Metric denominator: fixed raw tagged ACL glossary
  `acl6060_tagged_gt_raw_min_norm2.json`.
- Runtime glossary: same raw tagged ACL glossary.
- Output-side `<term>` markers are stripped before BLEU, StreamLAAL, and term
  metrics.
- Execution: direct detached Aries run with `lm=2` on GPU `4,5` and `lm=3` on
  GPU `6,7`; outputs/logs/temp/cache are placed under `/mnt/gemini/data1`.

## Expected metrics

Two W&B runs should be logged under family `tagged_acl_new_v9_hn1024_tau078`,
one for `zh/lm2/raw` and one for `zh/lm3/raw`, each with BLEU, StreamLAAL,
TERM_ACC, REAL_TERM_ADOPT, TERM_FCR, and SOURCE_TERM_SENT_FCR.

## Verdict

Completed and verified on Aries GPU `4,5` for `lm=2` and GPU `6,7` for
`lm=3`.  Verified artifacts exist for both settings: `eval_results.tsv`,
`instances.log`, and `term_adoption.json`.

W&B logging succeeded as `simuleval_eval/hplut7h5` for `lm=2` and
`simuleval_eval/a7bqd6nu` for `lm=3`.
