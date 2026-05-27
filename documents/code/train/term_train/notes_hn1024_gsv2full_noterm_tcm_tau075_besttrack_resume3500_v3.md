# No-Term TCM tau0.75 Best-Track Resume v3

## Hypothesis

Continuing the selected `T_alpha=0.64, T_beta=0.85, pos_w=1, neg_w=4`
checkpoint from the 100k-full-eval best step should reveal whether the useful
signal is continued dense recall or the deployment-facing tau0.75 frontier.
Tracking tau0.75 filtered recall and filtered micro precision directly should
produce checkpoints that are more relevant for downstream retrieval than dense
recall alone.

## Background / Motivation

The previous continuing-training run `tau6iuo3` selected best checkpoints by
dense `recall@10` on 10k and 100k glossaries.  That misses the two deployment
signals we now care about:

- `eval_dev/topk10_filtered_recall@tau_0p75_gs10000`, reflecting positive-side
  TCM pressure at the operating threshold.
- `eval_dev/topk10_filtered_precision_micro@tau_0p75_gs10000`, reflecting
  negative-side TCM pressure and how many forwarded top-10 candidates are true
  positives.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/tau6iuo3
- Supporting baseline/control URLs:
  - TCM-off control: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/us4obwe3
  - p1n4 weight scout: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/aamk3dok
  - n64 threshold scout: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/iie3967j
- Diff:
  - resume checkpoint: `q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_mfa_ntcm_final_v3_n64_p1n4_s2000_aries-8gpu_smallest_dense_smoke4650_best_eval_dev_full_recallat10_gs100000.pt`
  - expected resume step: 3500
  - target max step: 6000
  - save path stem changed to keep this best-tracking run separate
  - primary best metric: `eval_dev/topk10_filtered_recall@tau_0p75_gs10000`
  - secondary best metric: `eval_dev/topk10_filtered_precision_micro@tau_0p75_gs10000`
  - best trackers reset on resume so the first improvement under the new metric
    keys writes fresh checkpoints.

## Expected metrics

Prefer checkpoints that improve tau0.75 filtered recall without collapsing
filtered micro precision.  If the two best trackers disagree, compare both
saved checkpoints on 10k/100k and downstream simuleval before picking a final
export.

## Verdict

PENDING: update after the resume run finishes and the tau0.75 best checkpoints
are evaluated.
