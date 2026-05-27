# HN depth scout — `k=4096`, `TCM off`, `smallest + dense`, 8 GPU

Aggressive HN-depth scout after the TCM pivot. This run tests whether a much
deeper per-sample hard-negative budget still improves the current
`smallest + dense + normAGGR` retriever once TCM is removed entirely.

- **Family / data / task**: `sst_ood_hardneg` / `3variant_1m_mfa` / `train`
- **Variant tag**: `hnps_k4096_tcmoff_smallest_dense_normAGGR_8gpu_scout`
- **Launcher**: `documents/code/train/term_train/run_mfa_smallest_dense_hnps_k4096_tcmoff_8gpu_aries.sh`
- **Primary baseline run**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/tys70s0y

## Hypothesis

If the current OOD bottleneck is still insufficient hard-negative depth, then
`k=4096` should improve early dev/ACL `gs10000` metrics beyond the `k=1024`
candidate. If the recipe has already entered diminishing returns, `k=4096`
should instead reveal itself through slower steps and unstable/noisy gains.

## Background / Motivation

We do not yet know where the HN sweet spot lies under `smallest + dense`. The
historical `k=512` reference (`tys70s0y`) and the older `k=1024` anchor
(`r0xi5xkt`) both suggest that going deeper can help, but neither answers
whether the current recipe keeps benefiting once HN depth is pushed several
times further. This run is the first deliberate test of that deeper regime with
TCM entirely out of the way.

## What changed vs baseline

- **Baseline run URL**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/tys70s0y
- **Diff**:
  - hparam `tcm_loss_weight`: `1.0` -> `0.0`
  - hparam `tcm_pos_loss_weight`: legacy shared branch -> `0.0`
  - hparam `tcm_neg_loss_weight`: legacy shared branch -> `0.0`
  - hparam `hard_neg_k_per_sample`: `512` -> `4096`
  - hparam `num_gpus`: `6` -> `8`
  - hparam `per_gpu_batch`: `2048` -> `1536` (global `batch_size=12288` preserved)
  - hparam `grad_cache_chunk_size`: `256` -> `128` (memory safety for the deeper scout)
  - hparam `epochs`: historical `3` -> scout `1`
  - hparam `max_steps`: full run -> `120`
  - hparam `max_train_seconds`: full run -> `12600`
  - data / recipe: unchanged `smallest + dense + normAGGR`
  - code: use the new HN-depth common launcher so only HN depth moves semantically

## Expected metrics

- no OOM in the first `40` steps
- `train/step_time_ms <= 90000`
- by step `80`, dev metrics should not show obvious collapse relative to the `k=1024` scout
- by step `120`, `eval_acl6060/recall@10_gs10000 >= 0.65`
- by step `120`, `eval_dev/topk10_filtered_recall@tau_0p80_gs10000 >= 0.55`

## Verdict

FAILED (resource limit): the run OOMed at step `1` during the per-sample
MaxSim contraction `torch.einsum("bwd,bkd->bwk", speech_embs, hn_embs)` inside
`_maxsim_score_mfa_per_sample`. Even with `grad_cache_chunk_size=128`, each
rank tried to allocate another `12.00 GiB` on top of ~`41.2 GiB` already in
use, leaving only ~`5.7 GiB` free on the A6000s. This means `k=4096` is not
feasible under the current `8 x 1536`, `smallest + dense` recipe without
additional memory-reduction changes.
