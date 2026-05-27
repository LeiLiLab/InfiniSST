# HN256 best-secondary fixed-denominator dev100k + held-out eval, fast chunks

## Hypothesis

HN256 may be a better middle point than HN1024 if it preserves more recall while
retaining some hard-negative score calibration benefit.  The comparison should
use the same fixed strict raw denominator protocol as no-HN and HN1024.

## Background / Motivation

The conservative taurus2 run `fwdr8y3c` underused GPU memory with
`query_chunk=128` and `text_chunk=1024`.  This run keeps the same checkpoint and
eval protocol, but increases score chunks to use more of GPUs 4,5 and relies on
the eval-code optimization that skips duplicate no-term-noise logits when the
dev set has no no-term rows.

Checkpoint:

`/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8192_gc256_wr1000k_m0.0_maxsim_mfa_variantE_hn256_tcmoff_ep6_v3_smallest_dense_normAGGR_gpu01234567_taurus_resume_latest_best_eval_acl6060_recallat10.pt`

## What changed vs baseline

- Checkpoint: HN256 `lrdx14pm` best-secondary checkpoint, step 1200.
- Tau grid: `0.50..0.90` at stride `0.01`.
- Dev readout: raw/base, gs10k, and gs100k only.
- Held-out readouts: ACL6060, tagged ACL6060, and strict medicine with raw/base,
  gs1k, and gs10k.
- Metrics denominator: fixed raw/strict term universe; retriever glossary size
  changes only the candidate bank.
- Compute: taurus two-GPU eval on CUDA devices `4,5`.
- Score chunks: `query_chunk=256`, `text_chunk=4096`.

## Expected metrics

The output should identify HN256 tau values for strict raw-included dev-drop
budgets `<0.5 pp`, `<1.0 pp`, and `<1.5 pp`, then compare held-out gs10k
recall and micro precision against no-HN and HN1024 at matched dev-drop budgets.

## Verdict

Completed successfully as W&B `ykwbip03`.

- Runtime: 838.63s on taurus allocation `45300`, CUDA devices `4,5`.
- Checkpoint: HN256 `lrdx14pm` best-secondary checkpoint, step 1200.
- Dev raw/gs10k/gs100k unfiltered recall@10: `99.0289 / 98.8061 / 98.4877`.
- Strict raw-included dev-drop selected tau:
  - `<0.5 pp`: tau `0.69`, max drop `0.4776 pp`.
  - `<1.0 pp`: tau `0.76`, max drop `0.9471 pp`.
  - `<1.5 pp`: tau `0.80`, max drop `1.4327 pp`.
- Held-out gs10k R/P:
  - tau `0.69`: ACL `92.06 / 9.35`, tagged ACL `97.92 / 9.98`,
    medicine `93.11 / 10.83`.
  - tau `0.76`: ACL `89.84 / 12.85`, tagged ACL `96.96 / 12.94`,
    medicine `89.53 / 15.77`.
  - tau `0.80`: ACL `86.34 / 21.51`, tagged ACL `94.80 / 19.37`,
    medicine `85.55 / 23.58`.

Important compute caveat: this eval path is still rank0-only for scoring in
`eval_only`; the log reports `score_device=cuda:0`.  Increasing score chunks
uses more memory on GPU4, but GPU5 is not a real shard until the eval scorer is
rewritten to partition queries or banks across ranks.
