# hnps_k512: per-sample HN k=512, stacked on smallest+dense MFA + aggressive term_id normalization

Isolates the hard-negative-depth axis on top of 43848's proven smallest+dense+normAGGR recipe. 43848 (`zv28ve3q`) delivered `best/recall@10_gs1000=0.9395`, `best_secondary/recall@10_gs10000=0.8744` at step=1240 with pool-HN k=64, but lost -1.7pp on gs10000 vs the prior per-sample k=1024 variantE run (`r0xi5xkt`, best_sec=0.8915). This run restores the deeper HN mining but keeps the dense MFA recipe, expecting to stack the two improvements.

- **Family / data / task**: `sst_ood_hardneg` / `3variant_1m_mfa` / `train`
- **Variant tag**: `hnps_k512_smallest_dense_normAGGR_6gpu`
- **Launcher**: `documents/code/train/term_train/run_mfa_smallest_dense_hnps_k512_6gpu_aries.sh`
- **Baseline run**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/zv28ve3q (43848)
- **Cross-reference baseline**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/r0xi5xkt (prior per-sample k=1024 variantE, crashed at step 2160 with best_sec=0.8915)

## Hypothesis

Per-sample HN mining with k=512 tailors the hardest 512 negatives per anchor (vs pool k=64 shared across all anchors in the mini-batch), which should restore the -1.7pp gap on gs10000 that 43848 introduced when it swapped per-sample k=1024 for pool k=64. Stacked on the smallest+dense MFA recipe (which 43848 validated as non-harmful on gs1000), we expect `best_secondary/metric_value >= 0.90`, i.e. +2.6pp over 43848 and +0.85pp over `r0xi5xkt`.

## Background / Motivation

43848 result (finalized 2026-04-23T06:38Z): STATUS:SUCCESS on absolute thresholds but delta_gs10000=-0.0171 vs `r0xi5xkt`. Interpretation: the MFA window-selection change (hard_max -> smallest + dense grid) is NOT the OOD bottleneck on gs10000; the pool-HN k=64 is. Pool-HN shares the same 64 candidates across the whole mini-batch, so close-acoustic distractors for a specific anchor are often absent from the pool, and the model never learns to separate them. Per-sample HN fills that gap.

We choose k=512 rather than k=1024 because:
- Step-time scales roughly linearly in HN-depth for the max-sim contrastive call (dominated by `[B, K, D]` matmul in the loss).
- k=1024 would push step time from ~42s to ~65s on 6 GPUs, squeezing 3 epochs past the 28h SLURM cap.
- k=512 still mines the tail-of-negatives regime that pool-64 can't reach, at ~1.5x pool-k=64 step cost.

## What changed vs baseline

- **Baseline run URL**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/zv28ve3q
- **Diff**:
  - hparam `HARD_NEG_K`: 64 -> 0 (disable pool mode)
  - hparam `HARD_NEG_K_PER_SAMPLE`: 0 -> 512
  - hparam `NUM_GPUS`: 8 -> 6
  - hparam `PER_GPU_BATCH`: 1536 -> 2048 (effective `BATCH_SIZE=12288` preserved)
  - code: `qwen3_glossary_neg_train.py` HEAD unchanged from 43848 submission point
  - data: `/mnt/gemini/data1/jiaxuanluo/term_train_3variant_1m_mfa.jsonl` unchanged

## Expected metrics

- `best/metric_value  >= 0.95` (eval_acl6060/recall@10_gs1000; 43848=0.9395, r0xi5xkt=0.9411)
- `best_secondary/metric_value >= 0.90` (eval_acl6060/recall@10_gs10000; 43848=0.8744, r0xi5xkt=0.8915)
- `eval_acl6060/noterm_noise@top10_tau_0p80_gs10000 <= 2.0` (43848=1.47, well within budget)
- `step_time_ms <= 45000` (6 GPUs * PER_GPU_BATCH 2048 * per-sample k=512; 43848 ran ~21-25 s/it at 8 GPUs pool-64)

## Verdict

<!-- Filled by scripts/tools/finalize_wandb_run.py after the run reaches state=finished. -->
