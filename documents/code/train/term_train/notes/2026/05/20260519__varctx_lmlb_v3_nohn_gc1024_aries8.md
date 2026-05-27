# Variable-context retriever no-HN ablation, gc1024

## Hypothesis

Turning off per-sample hard negatives should isolate how much of `lh1b88kw` comes
from the HN objective versus the variable-context data, MaxSim MFA setup, and
Qwen3-Omni/BGE-M3 encoders. With HN disabled, GradCache chunks can be increased
from `128` to `1024` to reduce overhead. The current launch uses taurus hold
allocation `45269` with the explicit GPU list `0,1,2,3,4,5,6,7`.

## Background / Motivation

Source run `lh1b88kw` used the balanced 2.88s/3.84s/4.80s/5.76s GSV2-full
GSDedup variable-context dataset with global batch `8192`,
`hard_neg_k_per_sample=1024`, `grad_cache_chunk_size=128`, TCM-off, MaxSim MFA,
and six epochs on Aries.

## What changed vs baseline

- Source run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/lh1b88kw
- Ablation:
  - `hard_neg_k_per_sample`: `1024` -> `0`
  - `grad_cache_chunk_size`: `128` -> `1024`
  - GPU list: `0,1,2,3,4,5,6,7`
  - global batch remains exactly `8192` via `PER_GPU_BATCH=1024` on 8 GPUs.
- Protocol guardrail:
  - checkpoint selection stays dev-only: primary `eval_dev/recall@10_gs10000`,
    secondary `eval_dev/recall@10`.
  - ACL remains inline readout only and is not used to select the winner.

## Expected metrics

This should quantify the recall and tau-retention cost of removing HN while
keeping data, encoders, batch size, MaxSim MFA, and TCM-off settings aligned with
`lh1b88kw`. If HN mainly improves disambiguation, dev/ACL gs10k and filtered
recall should drop more than base recall.

## Verdict

PENDING: update after training finishes. Compare against `lh1b88kw` using
WandB at-best-step bundles, with ACL treated as held-out readout only.
