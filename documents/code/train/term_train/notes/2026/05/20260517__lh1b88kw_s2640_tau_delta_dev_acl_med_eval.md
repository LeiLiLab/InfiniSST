# lh1b88kw Secondary Checkpoint Tau-Delta Eval

## Hypothesis

The `lh1b88kw` secondary checkpoint can support a practical retrieval
filter threshold selected from dev recall retention: choose the largest tau
whose maximum recall drop across dev base / gs10k / gs100k is within 0.5pp, 1.0pp,
or 1.5pp from the tau=0.0 raw recall@10 baseline.

## Background / Motivation

Slurm job `45227` timed out before completing all requested epochs, but the
run produced a secondary-best checkpoint around step 2640.  The previous tau
sweep launchers split dev and ACL into separate root-level scripts and did not
include medicine, which makes provenance and later lookup difficult.  This run
records the calibration sweep and the held-out domain readouts as one managed
eval event.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/lh1b88kw
- Baseline checkpoint: `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt`
- Diff: eval-only readout, no training. Dev is evaluated on base / gs10k / gs100k and is the only source for tau selection. ACL6060 and medicine are evaluated on base / gs1k / gs10k using the selected tau candidates only for readout, not selection.
- Tau grid: `0.70..0.90` in 0.01 increments. Tau `0.0` is represented by raw `recall@10` / `recall@10_gs*`.
- Selection rule: for each delta in `0.5pp`, `1.0pp`, `1.5pp`, pick the largest tau whose maximum dev recall drop across base / gs10k / gs100k is at most that delta. Precision is reported alongside the selected tau values but is not used to choose tau.

## Expected metrics

Expect stricter deltas to select lower tau values and preserve more recall,
while looser deltas select higher tau values with fewer forwarded candidates
and lower no-term noise.  ACL6060 and medicine should be reported after dev
selection only as held-out readouts.

## Verdict

Completed successfully.  Dev-only max-drop recall-retention calibration selects
progressively stricter tau values for the three drop budgets; ACL6060 and
medicine are readout-only results in W&B run `4g108a3w`.
