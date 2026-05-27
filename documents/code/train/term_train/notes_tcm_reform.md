# tcm_reform + hnps_k512: TCM weight x5 + relaxed margins, stacked on per-sample HN k=512

Stacks the TCM reform (5x weight, relaxed margins) on top of the A1-validated per-sample HN k=512 recipe. A1 (`tys70s0y`) proved per-sample HN > pool HN at matched steps; this run tests whether adding the TCM reform on top yields further OOD gains. No longer an isolated single-axis ablation — this is a combined best-of-both run.

- **Family / data / task**: `sst_ood_hardneg` / `3variant_1m_mfa` / `train`
- **Variant tag**: `hnps_k512_tcm5x_relaxed_smallest_dense_normAGGR_8gpu`
- **Launcher**: `documents/code/train/term_train/run_mfa_smallest_dense_tcm_reform_8gpu_aries.sh`
- **Baseline runs**:
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/zv28ve3q (43848, original recipe)
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/tys70s0y (A1/hnps_k512, per-sample HN validated)

## Hypothesis

Two proven/promising axes stacked:
1. **Per-sample HN k=512** (A1/tys70s0y validated): converges ~2x faster, +0.026 gs10000 at matched steps vs pool k=64.
2. **TCM reform**: At 43848 convergence, `train/tcm_pos_viol_rate=0.232`, `train/tcm_neg_viol_rate=0.125`. Roughly 23% of positives still sit below beta=0.85 and 12.5% of hard-negatives still sit above alpha=0.25, yet TCM total share <2% of loss. Relaxing beta=0.85->0.70 and alpha=0.25->0.40 narrows the violators to the truly broken pairs, and raising weight 1.0->5.0 lifts TCM's effective loss share into ~8-12%.

Combined expected: +1.0 to +2.0pp on `best_secondary/metric_value` over `zv28ve3q`.

## Background / Motivation

A1 (`tys70s0y`) validated per-sample HN k=512 > pool k=64 (faster convergence, +0.026 gs10000 at matched steps). Now we stack the TCM reform on the better HN recipe to push further, rather than testing TCM in isolation on the weaker pool-HN baseline.

## What changed vs baseline

- **Baseline run URLs**:
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/zv28ve3q (43848)
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/tys70s0y (A1/hnps_k512)
- **Diff vs zv28ve3q (43848)**:
  - hparam `TCM_LOSS_WEIGHT`: 1.0 -> 5.0
  - hparam `TCM_WARMUP_STEPS`: 0 -> 200 (linear ramp; 43883 with no warmup collapsed at step 280)
  - hparam `TCM_POS_THRESHOLD`: 0.85 -> 0.70 (relax: easier for positives to pass)
  - hparam `TCM_NEG_THRESHOLD`: 0.25 -> 0.40 (relax: easier for negatives to pass)
  - hparam `HARD_NEG_K`: 64 -> 0 (disable pool mode)
  - hparam `HARD_NEG_K_PER_SAMPLE`: 0 -> 512 (per-sample HN, validated in A1)
  - hparam `GRAD_CACHE_CHUNK_SIZE`: 256 -> 512 (speed; memory fits in 48GB A6000)
  - hparam `NUM_GPUS`: 8 (same as baseline)
  - hparam `PER_GPU_BATCH`: 1536 (same; `BATCH_SIZE=12288` preserved)
  - MFA: unchanged (smallest + dense grid)
  - data / code: `--tcm_warmup_steps` added to `qwen3_glossary_neg_train.py`

## Expected metrics

- `best/metric_value >= 0.94` (eval_acl6060/recall@10_gs1000; 43848=0.9395, tys70s0y=0.9287)
- `best_secondary/metric_value >= 0.89` (eval_acl6060/recall@10_gs10000; 43848=0.8744, tys70s0y=0.8419@520; target: beat both)
- `eval_acl6060/noterm_noise@top10_tau_0p80_gs10000 <= 2.0` (43848=1.47)
- Final `train/tcm_pos_viol_rate <= 0.08` and `train/tcm_neg_viol_rate <= 0.05` (3x reduction, evidence the relaxation worked)
- Sanity: `train/pos_sim >= 0.80` throughout (43848 plateaued at 0.87; large drop would indicate the 5x TCM overshadowing InfoNCE)
- `step_time_ms <= 45000` (8 GPUs per-sample HN k=512, grad_cache_chunk=512)

## Risk notes

- **43883 (failed, no warmup)**: 5x TCM from step 0 collapsed embeddings by step 280 (loss=9.39, recall@10=0.095, pos_sim=0.545/neg_sim=0.480 — gap 0.065). The TCM gradient overwhelmed InfoNCE before any embedding structure formed.
- This run adds 200-step linear warmup (TCM weight ramps 0 -> 5.0 over steps 0-200). InfoNCE has exclusive control for the first ~100 steps, then TCM gradually joins. By step 200, tys70s0y showed recall@10 > 0.93 at the same recipe, so embeddings should be well-separated before TCM reaches full strength.
- If `train/pos_sim` drops below 0.80 or `train/loss_infonce` starts climbing after warmup completes, the 5x TCM is still too aggressive; fallback: weight=3.0.
- The relaxed alpha=0.40 is a large relaxation; with `r0xi5xkt`-style runs showing `neg_sim` often <0.10, the practical effect of the alpha change may be small, so most of the signal will come from beta relaxation + weight scale.

## Verdict

<!-- Filled by scripts/tools/finalize_wandb_run.py after the run reaches state=finished. -->
