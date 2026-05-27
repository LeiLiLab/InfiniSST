# Multi-Scale Inference Ablation: Dense 1.92s Trained Retriever

## Hypothesis

A retriever trained to encode the 1.92s speech context as one dense embedding
should underperform the main multi-scale MaxSim retriever, because it cannot
localize multiple term mentions inside the context and does not receive
MFA-localized MaxSim supervision.

## Background / Motivation

The historical dense single-embedding training run is W&B `r5l4780c`
(`sweep_tpool_sweep_tp_cls_tfpool_bs12k`).  Its config does not enable
`use_maxsim`; it uses `pooling_type=transformer`, `text_pooling=cls`, and the
default fixed 1.92s audio context.  The run crashed after the early sweep
window, but W&B and checkpoint artifacts record a best checkpoint at step 198.

## What changed vs baseline

This readout evaluates checkpoint:

```text
/mnt/aries/data4/jiaxuanluo/train_outputs/sweep_text_pooling/sweep_tp_cls_tfpool_bs12k_best.pt
```

with `USE_MAXSIM=false`, `MFA_SUPERVISED=false`, `FIXED_SECONDS=1.92`,
`TEXT_POOLING=cls`, and the same dev raw fixed-denominator protocol used for
the existing 1.92s context ablation row.  The current MaxSim references are
`q2fus6f1` (multi-scale MaxSim) and `y454004y` (only largest MaxSim window).

## Expected metrics

Metrics should be read from the W&B run created by this eval and mirrored into
the manifest after completion.  The main reporting keys are
`eval_dev/recall@10`, `eval_dev/recall@10_gs10000`, and
`eval_dev/recall@10_gs100000` under the fixed raw ctx1.92 denominator.

## Verdict

Completed as W&B `740c7y40`.  Under the ctx1.92 dev fixed-raw protocol,
the dense single-embedding checkpoint from `r5l4780c` reached
`eval_dev/recall@10=0.9760`, `eval_dev/recall@10_gs10000=0.9597`, and
`eval_dev/recall@10_gs100000=0.9265`.  The tau-0.75 filtered recall is much
lower (`0.2394`), consistent with a dense one-shot model producing high
precision but poor recall after thresholding.
