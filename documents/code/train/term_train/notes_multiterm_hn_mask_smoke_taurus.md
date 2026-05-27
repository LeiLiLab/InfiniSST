# Multi-Term HN Mask Smoke - MFA term-scoped positives

- **Family / data / task**: `sst_ood_hardneg` / `3variant_gsv2p020_oldwiki_mfa` / `smoke`
- **Variant tag**: `mt_hnmask_smoke_taurus2`
- **Launcher**: `documents/code/train/term_train/run_multiterm_hn_mask_smoke_taurus.sh`
- **Baseline run candidates**:
  - same-k scout: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/fma3wmh2
  - GSV2p020 oldwiki k=4096 reference: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/yx52spnl
  - historical strong k=512 reference: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/tys70s0y

## Hypothesis

If multi-term chunk positives are the source of the train-inference mismatch,
then term-scoped MFA positives plus chunk-positive HN masking should run without
co-chunk GT terms entering per-sample hard negatives, while preserving ordinary
single-term positive supervision.

## Background / Motivation

The ACL expansion audit showed chunks where a valid ACL term such as `realm`
co-occurs with generic terms like `model`. The previous training path grouped
all rows from the same chunk as positives, but MFA scoring used the current
row's single term span for every positive column. Per-sample HN mining also
excluded only the anchor term, so other GT terms from the same chunk could be
mined as negatives.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/fma3wmh2
- Diff:
  - code: add `_chunk_positive_term_ids` per row and carry padded
    `positive_term_ids` / `positive_term_mask` through collate and train step
  - HN mining: `hard_neg_k_per_sample` excludes every known GT term for the
    speech chunk, not only the anchor `term_id`
  - loss masking: per-sample HN columns whose term is a known co-chunk GT term
    are false-negative masked
  - MFA positives: `mfa_positive_scope=auto`, which resolves to term-level
    positives when `mfa_supervised_maxsim` is active
  - smoke hparams: 1 GPU, `train_limit=20000`, `hard_neg_k_per_sample=32`,
    `max_steps=20`, with one dev/ACL eval at step 20

## Expected metrics

This is a functional smoke, not a scientific training point. Expected signals:

- WandB run initializes successfully and records `task:smoke`.
- `train/hn_false_positive_masked_count` stays near zero after mining-time
  exclusion; nonzero values would indicate defensive loss masking caught a leak.
- `train/cochunk_neutral_count` is nonzero when a batch contains multi-term
  chunks, showing term-scoped MFA is neutralizing same-chunk different-term rows.
- Dev/ACL eval at step 4 should complete so the run can be compared structurally
  with the selected baselines, but metric values are not used as model evidence.

## Verdict

Pending smoke completion.
