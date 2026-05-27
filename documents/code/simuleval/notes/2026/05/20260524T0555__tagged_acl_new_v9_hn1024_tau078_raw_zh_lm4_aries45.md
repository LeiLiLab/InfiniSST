# Tagged ACL New V9 HN1024 Tau0.78 Raw zh lm4 Aries GPU45

## Hypothesis

New V9 assistant-side term-tag SFT with HN1024 retrieval at tau `0.78` should
provide the current zh tagged ACL RASST main-result readout for `lm=4` under the
fixed raw tagged denominator.

## Background / Motivation

This completes the requested zh tagged ACL main-result sweep after launching
`lm=1`, then `lm=2,3`.  The requested setting here is `lang=zh`, `lm=4`, raw
glossary.  Current Aries inspection showed GPU `4,5` idle while GPU `2,3` were
occupied by an unrelated medicine run.

## What changed vs baseline

- Speech LLM: New V9 assistant term-tag delay-clean HF export.
- Retriever: HN1024 `lh1b88kw` checkpoint.
- Threshold: tau `0.78`, top-k `10`, timeline lookback `1.92s`.
- Dataset/readout: tagged ACL `zh`, `lm=4`, raw glossary.
- Metric denominator: fixed raw tagged ACL glossary
  `acl6060_tagged_gt_raw_min_norm2.json`.
- Runtime glossary: same raw tagged ACL glossary.
- Output-side `<term>` markers are stripped before BLEU, StreamLAAL, and term
  metrics.
- Execution: direct detached Aries run on GPU `4,5`; outputs/logs/temp/cache
  are placed under `/mnt/gemini/data1`.

## Expected metrics

One W&B run should be logged under family `tagged_acl_new_v9_hn1024_tau078` for
`zh/lm4/raw`, with BLEU, StreamLAAL, TERM_ACC, REAL_TERM_ADOPT, TERM_FCR, and
SOURCE_TERM_SENT_FCR.

## Verdict

Running as detached Aries direct process `direct_pid_1146047` on GPU `4,5`.
Startup is verified: `run_meta.txt` exists, the cached HN1024 MaxSim index is
used, vLLM TP=2 loaded the 15-shard HF export, and SimulEval entered the `0/5`
sample progress loop.  Update after `eval_results.tsv`, `instances.log`,
`term_adoption.json`, and W&B logging are verified.
