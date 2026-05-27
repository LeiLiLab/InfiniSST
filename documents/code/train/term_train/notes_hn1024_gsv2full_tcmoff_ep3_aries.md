# Full GSV2 k1024 TCM-off warm start

- **Family / data / task**: `sst_ood_hardneg` / `3variant_gsv2full_gsfix_mfa` / `train`
- **Variant tag**: `hn1024_gsv2full_tcmoff_ep3`
- **Launcher**: `documents/code/train/term_train/run_mfa_smallest_dense_hn1024_gsv2full_tcmoff_ep3_8gpu_aries.sh`
- **Primary baseline run**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/ly6sc2mr
- **Same-k historical baseline**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/fma3wmh2

## Hypothesis

A full GSV2 k=1024 TCM-off warm start should give a cleaner base for TCM
calibration than continuing from a partially trained TCM run. Training for three
epochs should establish strong dense retrieval while preserving AdamW momentum
for a fourth-epoch TCM continuation.

## Background / Motivation

The TCM threshold/weight search should be judged on dev metrics, not ACL6060.
This run creates a shared source checkpoint for all continuing runs using the
full GSV2 gsrepaired data line and a dev 10k glossary sampled from P31 terms not
used during training.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/ly6sc2mr
- Diff:
  - hparam `hard_neg_k_per_sample`: `4096` -> `1024`
  - hparam `num_gpus`: `6` -> `8`
  - hparam `batch_size`: `6144` -> `12288`
  - hparam `epochs`: scout `1` -> warm-start `3`
  - hparam `scheduler_epochs`: unset -> `4`, so the third-epoch checkpoint resumes with non-zero LR
  - hparam `tcm_pos_loss_weight` / `tcm_neg_loss_weight`: `0.0` / `0.0` unchanged
  - eval selection: ACL6060 metrics disabled; best checkpoints track `eval_dev/*`
  - eval glossary: `wiki_glossary_nlp_ai_cs.json` -> untrained P31 dev sample at gs10000

## Expected metrics

The run is useful if dev `recall@10_gs10000` and tau-filtered recall improve
over the early same-k TCM-off scout while keeping no-term noise reasonable at
`tau=0.80`. The key output is the epoch-2 checkpoint with optimizer and
scheduler state for the 16-run TCM continuation sweep.

## Verdict

PENDING: update after the baseline warm-start finishes and the checkpoint audit
confirms AdamW moment state is present.
