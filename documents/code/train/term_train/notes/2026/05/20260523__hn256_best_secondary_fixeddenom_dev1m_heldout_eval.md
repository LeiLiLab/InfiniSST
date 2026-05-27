# HN256 best-secondary fixed-denominator dev-1M + held-out eval

## Hypothesis

HN256 may keep more of the useful score-shaping behavior of mined hard
negatives than no-HN while avoiding the recall loss seen in HN1024.  The
fixed-denominator eval should show whether HN256 is a better middle point.

## Background / Motivation

The current comparison report covers no-HN and HN1024.  The HN256 continuation
run `lrdx14pm` produced a new best-secondary checkpoint at step 1200:

`/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8192_gc256_wr1000k_m0.0_maxsim_mfa_variantE_hn256_tcmoff_ep6_v3_smallest_dense_normAGGR_gpu01234567_taurus_resume_latest_best_eval_acl6060_recallat10.pt`

This eval adds HN256 to the same fixed strict raw/base denominator protocol used
for the no-HN vs HN1024 report.

## What changed vs baseline

- Checkpoint: HN256 `lrdx14pm` best-secondary checkpoint, step 1200.
- Tau grid: `0.50..0.90` at stride `0.01`.
- Dev readout attempted: raw/base, gs10k, gs100k, and gs1M.
- Held-out readouts: ACL6060, tagged ACL6060, and strict medicine with raw/base,
  gs1k, and gs10k.
- Metrics denominator: fixed raw/strict term universe; retriever glossary size
  changes only the candidate bank.

## Expected metrics

The output should identify HN256 tau values for strict raw-included dev-drop
budgets `<0.5 pp`, `<1.0 pp`, and `<1.5 pp`, then compare held-out gs10k
recall and micro precision against no-HN and HN1024 at matched dev-drop budgets.

## Verdict

Aborted before producing final metrics. This dev-1M attempt was intentionally
stopped and replaced by
`20260523T0341__retriever_eval__hn256_best_secondary_fixeddenom_dev100k_heldout_taurus2`,
because HN256 only needs dev gs10k/gs100k scoring for this comparison.
