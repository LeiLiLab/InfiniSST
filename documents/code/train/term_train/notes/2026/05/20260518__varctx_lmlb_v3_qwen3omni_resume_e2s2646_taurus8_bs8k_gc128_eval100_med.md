# Qwen3-Omni Varctx576 Resume From Epoch 2

## Hypothesis

Continuing `lh1b88kw` from the epoch-2 checkpoint should let the original
Qwen3-Omni audio plus BGE-M3 text retriever keep improving after the crashed
Aries run stopped before convergence.  Evaluating every 100 optimizer steps
should give a denser convergence trace than the source run's 80-step cadence
while preserving the same training semantics.

## Background / Motivation

The source W&B run `lh1b88kw` used the varctx576 GSV2-full GSDedup training
data, k=1024 per-sample hard negatives, MaxSim MFA, TCM disabled, global batch
8192, GradCache chunk 128, and a six-epoch schedule.  It wrote the requested
resume checkpoint:

`/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_epoch_2.pt`

Local checkpoint inspection found `epoch=2`, `global_step=2646`,
`best_metric_key=eval_dev/recall@10_gs10000`, and restored best dev value
`0.9901305437088013`.  ACL is held-out readout and must not be used for
checkpoint or hyperparameter selection.

## What changed vs baseline

- Source run: `lh1b88kw`.
- Resume checkpoint: epoch 2 / global step 2646 from the original Qwen3-Omni
  varctx576 run.
- Compute: `aries` source run -> `taurus` resume run.
- Eval interval: source launcher `80` steps -> resume launcher `100` steps.
- Medicine domain inline eval is enabled with the varctx576 medicine dataset
  and medicine gs10k glossary.
- ACL eval uses the min-normalized-length-2 backfilled gs10k glossary for
  comparable held-out readout.
- Checkpoint selection is dev-only: primary best metric remains
  `eval_dev/recall@10_gs10000`; secondary best uses `eval_dev/recall@10` so
  held-out ACL does not write a secondary best checkpoint.

## Expected metrics

The first post-resume inline eval should occur at step 2700 because the
checkpoint resumes from global step 2646 and the interval is 100.  Dev recall
should be tracked as the model-selection signal; ACL and medicine should be
interpreted as readouts only.

## Verdict

PENDING: fill after the resumed run finishes and W&B is synced.
