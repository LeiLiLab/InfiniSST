# Scheduled HN256 stop and HN512 resume

## Hypothesis

Let `gsjheh6r` continue for three more hours, then pause it and resume the
existing HN512 ablation from its latest checkpoint so Taurus GPUs are reused
without manual intervention.

## Background / Motivation

The current HN256 run is `gsjheh6r`, resumed from step 1200 with ACL metric
reset. The HN512 run is `5fwrs7rh`, previously paused after the latest
checkpoint at step 320 was saved.

## What changed vs baseline

This is a maintenance timer, not a new training recipe. At timer fire time it
will:

- terminate the `gsjheh6r` process group;
- annotate `gsjheh6r` as paused in W&B;
- update and register the HN256 manifest as paused;
- launch HN512 from the latest checkpoint with `hard_neg_k_per_sample=512`,
  `grad_cache_chunk_size=256`, `batch_size=8190`, and GPU list `0,1,2,3,4,5`;
- wait for HN512 W&B init and update the scheduled HN512 resume manifest if a
  run id appears.

## Expected metrics

No metric should be produced by the timer itself. The HN512 resumed training
should continue using the original dev-primary checkpoint metric
`eval_dev/recall@10_gs10000` and secondary readout
`eval_acl6060/recall@10`.

## Verdict

SCHEDULED. Timer is armed for a three-hour delay from launch.
