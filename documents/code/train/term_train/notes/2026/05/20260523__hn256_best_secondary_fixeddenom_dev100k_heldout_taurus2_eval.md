# HN256 best-secondary fixed-denominator dev100k + held-out eval

## Hypothesis

HN256 may be a better middle point than HN1024 if it preserves more recall while
retaining some hard-negative score calibration benefit.  The comparison should
use the same fixed strict raw denominator protocol as no-HN and HN1024.

## Background / Motivation

The current comparison report covers no-HN and HN1024.  The HN256 continuation
run `lrdx14pm` produced a best-secondary checkpoint at step 1200:

`/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8192_gc256_wr1000k_m0.0_maxsim_mfa_variantE_hn256_tcmoff_ep6_v3_smallest_dense_normAGGR_gpu01234567_taurus_resume_latest_best_eval_acl6060_recallat10.pt`

This eval replaces the aborted dev-1M attempt `wziaorub`.  For HN256, dev
selection only uses raw/base plus gs10k and gs100k; dev-1M is intentionally not
scored.

## What changed vs baseline

- Checkpoint: HN256 `lrdx14pm` best-secondary checkpoint, step 1200.
- Tau grid: `0.50..0.90` at stride `0.01`.
- Dev readout: raw/base, gs10k, and gs100k only.
- Held-out readouts: ACL6060, tagged ACL6060, and strict medicine with raw/base,
  gs1k, and gs10k.
- Metrics denominator: fixed raw/strict term universe; retriever glossary size
  changes only the candidate bank.
- Compute: taurus two-GPU eval on CUDA devices `4,5`.

## Expected metrics

The output should identify HN256 tau values for strict raw-included dev-drop
budgets `<0.5 pp`, `<1.0 pp`, and `<1.5 pp`, then compare held-out gs10k
recall and micro precision against no-HN and HN1024 at matched dev-drop budgets.

## Verdict

Aborted before producing final metrics. This run used conservative scoring
chunks (`query_chunk=128`, `text_chunk=1024`) and was replaced by the fast-chunk
run
`20260523T0358__retriever_eval__hn256_best_secondary_fixeddenom_dev100k_heldout_taurus2_fastchunks`
after confirming taurus GPUs 4,5 had enough memory headroom.
