# WandB Tag Hygiene Fail-Fast

## Hypothesis

Experiment launches should not reach `wandb.init` before discovering that a
structured tag is longer than W&B's 64-character limit.

## Background / Motivation

Several retriever train/eval launches have failed during W&B initialization
because `data:*` or related structured tags exceeded 64 characters.  This wastes
GPU allocation startup time and creates partially started attempts that must be
cleaned up manually.

## What changed vs baseline

- `experiment_event.py` now validates static W&B tag candidates from manifest
  metadata and launcher exports before register/launch.
- `qwen3_glossary_neg_train.py` now uses the central `wandb_tags.py` helper to
  shorten overlong tags deterministically before `wandb.init`.
- If W&B initialization still fails while `--enable_wandb` is set, the training
  script raises and aborts instead of continuing locally.

## Expected metrics

No model metrics are affected.  The expected behavior change is operational:
overlong tags are either caught by `experiment_event.py` before launch or
shortened deterministically and recorded in W&B config.

## Verdict

Implemented and smoke-checked with `py_compile`, a valid manifest register, and
a synthetic overlong `data_tag` validation failure.
