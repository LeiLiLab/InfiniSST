# C1 — λ-only ablation: lift TCM_LOSS_WEIGHT 1→5 on top of tys70s0y (A1)

Clean isolation of `tcm_loss_weight` starting from the A1 (tys70s0y) baseline. Every other hyper-parameter is held identical to tys70s0y by the launcher [`run_mfa_smallest_dense_lambda5_6gpu_aries.sh`](run_mfa_smallest_dense_lambda5_6gpu_aries.sh) (diff is a single λ bump plus a 100-step safety warmup ramp).

- **Family / data / task**: `sst_ood_hardneg` / `3variant_1m_mfa` / `train`
- **Variant tag**: `hnps_k512_lambda5_smallest_dense_normAGGR_6gpu`
- **Launcher**: [`documents/code/train/term_train/run_mfa_smallest_dense_lambda5_6gpu_aries.sh`](run_mfa_smallest_dense_lambda5_6gpu_aries.sh)
- **Baseline run (tys70s0y / A1)**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/tys70s0y
- **SLURM job**: `43901` (aries, 6 GPUs, submitted 2026-04-24T00:02Z)
- **WandB run**: `ll5a6p9k` — https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/ll5a6p9k
- **Warmup sanity @ step 20** (2026-04-24T00:14Z): `total_loss = 9.305 = infonce 9.131 + 1.0 × (L_tcm_pos 0.094 + L_tcm_neg 0.080)` → effective `λ(step=20) = 5 × 20/100 = 1.0`,ramp 生效,与 tys70s0y @ step 20 (`infonce=9.08 neg_sim=0.511`) 的无 warmup baseline 在同一起点,早期梯度零尖峰。

## Hypothesis

在 `tys70s0y` 的 α=0.25 / β=0.85 架构下,把 `tcm_loss_weight` 从 1 抬到 5 会让 TCM neg-side 的梯度压力变强,把 `train/tcm_neg_viol_rate` 从 tys70s0y 稳态 ~0.21 进一步压低到 ~0.15,neg_sim_mean 从 +0.12 压到 ≤ +0.05,同时:

- `eval_acl6060/topk10_filtered_recall@tau_0p80_gs10000` 在 `best_secondary` checkpoint 相对 tys70s0y 不降(≥ 持平),最好 +0.5pp 以上(当前 tys70s0y = 0.7845)。
- `eval_acl6060/recall@10_gs10000` 不降(tys70s0y best_secondary = 0.8419)。
- 训练不崩:`train/loss_infonce` 稳态 ≤ 2.5,`train/tcm_loss_weight` 曲线在 step 100 之后等于 5。

A2 (we6ssrn7) 的失败证明放宽 α=0.40 让 per-sample HN 挖出来的硬 neg 从 TCM 视野里撤走、导致 noise 反向炸。C1 反其道而行,**保留 α=0.25**,只用更大的 λ 把已经暴露在 TCM 下的 20% 硬 neg 再压下去。

## Background / Motivation

**Post-mortem of A2 (we6ssrn7, retired)**: A2 同改 α 0.25→0.40、β 0.85→0.70、λ 1→5,还加 warmup=200,并把 GPU 6→8、`grad_cache_chunk` 256→512。经过读码验证([`qwen3_glossary_neg_train.py:2124`](qwen3_glossary_neg_train.py) `global_text_embs = all_gather_with_grad(text_embs)`),GPU 数和 `grad_cache_chunk_size` 对 InfoNCE/TCM 看到的 global neg 池构造**零影响**,所以 A2 的真实变量仍然是 α/β/λ 三个同改。A2 `noise@tau0.80_gs10k` 从 tys70s0y 1.57 炸到 5.35,`best/gs1000` 从 step 120 peak 0.8295 衰退到 step 240 的 0.7535,verdict 判定为 TCM 反效果 + embedding 腐蚀。

**zv28ve3q (pool HN k=64) vs tys70s0y (per-sample HN k=512) 对照诊断** (α/β/λ 三者完全相同):

| step 500 | zv28ve3q (pool k=64) | tys70s0y (per-sample k=512) |
|---|---|---|
| pos_viol_rate | 0.312 | 0.311 |
| neg_viol_rate | **0.041** | **0.208** |
| pos_sim_mean | 0.844 | 0.845 |
| neg_sim_mean | **−0.098** | **+0.122** |
| gs_you S_neg_top1 p95 (offline) | 0.969 | 0.952 |

pos 侧完全对称,neg 侧差异全部来自 HN 策略:per-sample HN k=512 把 neg 池硬度抬高,neg_sim_mean +0.22 offset,自然让 20% 的 neg 落在 α=0.25 右边。**这是健康稳态,不是训练失败**。每个 chunk 的 top-1 尾巴(决定 τ=0.80 filtered recall)per-sample 更低,所以 tys 的 retrieval 反而好。

**结论锁定**:α=0.25 在 per-sample HN k=512 下是对的,不要为了降 viol_rate 去放宽。先单独抬 λ。

## What changed vs baseline

- **Baseline run URL**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/tys70s0y
- **Diff** (相对 [`run_mfa_smallest_dense_hnps_k512_6gpu_aries.sh`](run_mfa_smallest_dense_hnps_k512_6gpu_aries.sh)):
  - hparam `tcm_loss_weight`: 1.0 → **5.0**(**唯一的 ablation 变量**)
  - hparam `tcm_warmup_steps`: 0 → **100**(safety ramp, 不计入 ablation 变量;step 0 时 `λ × (L_pos+L_neg) ≈ 5 × 0.18 = 0.9` ≈ InfoNCE(~9.2)的 10%,100 步线性 ramp 防止早期梯度尖峰)
  - naming / `VERSION` / `WANDB_EXP_NAME` / `EXTRA_WANDB_TAGS` / `BASELINE_RUN_IDS=tys70s0y` / `NOTES_FILE` / `MASTER_PORT=29965`(避开 tys70s0y 风格的端口冲突)
  - code: `qwen3_glossary_neg_train.py` HEAD 未变(A1、A2 已经用过同一提交点 `tcm_warmup_steps` 的实现)
  - data: `/mnt/gemini/data1/jiaxuanluo/term_train_3variant_1m_mfa.jsonl` 不变
  - α / β / HN 策略 / k=512 / GPU=6 / PER_GPU_BATCH=2048 / `grad_cache_chunk_size=256` / LR=1.7e-4 / T=0.07 / epochs=3 / MFA smallest+dense / eval 配置 **全部逐字保留**

## Expected metrics

| metric | baseline (tys70s0y @ best_secondary step=520) | expected @ C1 best_secondary |
|---|---|---|
| `best_secondary/metric_value` (`eval_acl6060/recall@10_gs10000`) | 0.8419 | **≥ 0.842** (no worse) |
| `best/metric_value` (`eval_acl6060/recall@10_gs1000`) | 0.9287 | **≥ 0.925** (no worse) |
| `eval_acl6060/topk10_filtered_recall@tau_0p80_gs10000` | 0.7845 | **≥ 0.785** (best outcome ≥ 0.790, +0.5pp) |
| `eval_acl6060/noterm_noise@top10_tau_0p80_gs10000` | 2.78 | **≤ 2.5** (strict improvement, direct target of λ bump) |
| `train/tcm_neg_viol_rate` (stable @ step 500+) | 0.21 | **0.12 – 0.16** |
| `train/tcm_pos_viol_rate` (stable @ step 500+) | 0.27 | 0.24 – 0.30 (≈unchanged; λ lift hits both sides but pos is already near floor) |
| `train/neg_sim_mean` (stable @ step 500+) | +0.12 | +0.03 – +0.08 |
| `train/loss_infonce` (stable) | 2.15 | 2.0 – 2.5 (健康) |
| `train/step_time_ms` | ~40000 | ~40000 (TCM weight change is free) |

**Falsifiability**:
- 若 `noise@tau0.80_gs10000` ≥ tys70s0y 的 2.78 或 `best_secondary` < 0.835 → λ=5 无效,C1 失败,放弃 TCM 调参直接用 tys70s0y 冲 SST 下游。
- 若 `loss_infonce` > 4 持续 200 步以上、或 `pos_sim_mean` 跌破 0.7 → embedding 退化,立即 kill。

## Verdict

<!-- Filled by scripts/tools/finalize_wandb_run.py after the run reaches state=finished,
     then agent reads BOTH best/step and best_secondary/step bundles via
     `python documents/code/general/wandb_tool.py --project qwen3_rag compare <C1_id> tys70s0y --preset retriever_eval retriever_train --at-best-step --anchor-metric both`
     and writes a two-bundle comparison table here. -->
