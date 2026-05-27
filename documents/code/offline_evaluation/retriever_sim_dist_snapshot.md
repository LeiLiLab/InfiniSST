# Retriever Pos/Neg Sim-Distribution Snapshots (smallest+dense MFA family)

Central registry of honest S_pos / S_neg similarity distributions for the
recent retriever checkpoints that use `MFA_WINDOW_SELECTION=smallest` +
`MAXSIM_WINDOWS="2 3 4 5 6 7 8 10 12 16 20 24"` (stride=2).

Purpose: pick `TCM_NEG_THRESHOLD (alpha)` and `TCM_POS_THRESHOLD (beta)`
off the data, instead of guessing. Each snapshot prints per-domain
percentiles and spits out OOD-robust alpha/beta candidates.

## Scripts

| Purpose | Path |
|---|---|
| **Pos/Neg distribution (this snapshot's evaluator)** | `documents/code/offline_evaluation/retriever_posneg_dist.py` |
| — imports encoders + model build from ↓ | |
| Threshold-sweep evaluator (source of encoders) | `documents/code/offline_evaluation/threshold_sweep_maxsim.py` |
| Older per-domain sim-dist script (hard-coded to Config C) | `documents/code/offline_evaluation/retriever_sim_distribution.py` |

`retriever_posneg_dist.py` is intentionally narrow:

- **Unbiased `S_pos`**: computes `cos-sim(chunk, GT term)` directly for
  every `(has_term chunk, GT term)` pair. The hist in
  `threshold_sweep_maxsim.py::_plot_score_histogram` only collects
  positives that made it into top-10 — it silently drops
  `(1 − recall@10)` of the hardest positives, which is exactly the slice
  `TCM_POS_THRESHOLD (beta)` is supposed to penalize.
- **Rich `S_neg` tail**: top-K negs per chunk (K=50) instead of top-10,
  so the 90–99th percentiles are meaningful for `alpha`.
- **Per-domain buckets**: `gs_pod` / `gs_you` / `gs_aud` / `wiki_synth` /
  `acl6060`. OOD threshold robustness needs the worst-domain tail, not
  the average.
- **No sweep machinery** (P/R/F1 curves, chunk-detection TSVs, tau sweep):
  cut for clarity. Those live in `threshold_sweep_maxsim.py` when needed.

## Checkpoints being snapshotted

Source of truth: WandB project `qwen3_rag`, tag `family:sst_ood_hardneg`.
Each row is the `_best_acl6060_gs10000.pt` (i.e. `best_secondary/step`)
artifact for a finished/successful run that used the smallest+dense MFA
window recipe.

| WandB run | Variant | best_secondary/step | `recall@10_gs10000` @ step | Ckpt path | Snapshot |
|---|---|---|---|---|---|
| [`zv28ve3q`](https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/zv28ve3q) | variantE baseline + pool HN k=64 | 1240 | 0.8744 | `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_clean_gc_wr1000k_m0.0_maxsim_mfa_variantE_hardneg_k64_tcm_ep3_cold_smallest_dense_normAGGR_best_acl6060_gs10000.pt` | done (43899) |
| [`tys70s0y`](https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/tys70s0y) | hnps_k512 (per-sample HN k=512, same recipe) | 520 | 0.8419 | `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_clean_gc_wr1000k_m0.0_maxsim_mfa_variantE_hardneg_ps_k512_tcm_ep3_cold_smallest_dense_normAGGR_6gpu_best_acl6060_gs10000.pt` | done (43900) |
| [`ll5a6p9k`](https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/ll5a6p9k) (C1) | hnps_k512 + λ=5 + warmup=100 (only λ changed vs tys70s0y) | _TBD_ | _TBD_ | `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_clean_gc_wr1000k_m0.0_maxsim_mfa_variantE_hardneg_ps_k512_tcm_lambda5_wu100_ep3_cold_smallest_dense_normAGGR_6gpu_best_acl6060_gs10000.pt` | pending (wait for training to finish; ETA ~24h from 2026-04-24T00:02Z) |

Notes:
- `zv28ve3q` / `tys70s0y` both trained with `α = 0.25`, `β = 0.85`, `λ = 1.0`.
- `tys70s0y` was killed at step 680 after A1 validated per-sample HN > pool HN. Its `best_secondary` is mid-epoch.
- A2 (`we6ssrn7`, β=0.70 α=0.40 λ=5 + TCM warmup=200, 8 GPUs) killed at step 260 on 2026-04-23 — `status:failed`. Matched-step @240: −11pp gs1000, −19pp gs10000 vs `tys70s0y`; `noise@τ0.80_gs10k` went the wrong way (5.35 vs 1.57). A2 retired as "too many simultaneous variables".
- C1 (`ll5a6p9k`, λ=5 with warmup=100, α/β/HN/GPU/grad_cache 全部保持 tys70s0y 配置逐字未变) replaces A2 as the clean λ-only isolation. Pending histogram will sit next to `tys70s0y` for direct compare: expect `S_neg_top1` tail at p95 drop from tys70s0y ~0.95 to ~0.90 on `gs_you` / `acl6060` if λ=5 is doing real work.

## Output artifacts (per checkpoint)

Each snapshot job writes into `<output_dir>`:

```
raw_sims.npz                         flat arrays: {domain}__S_pos, {domain}__S_neg_top1,
                                     {domain}__S_neg_top5, {domain}__S_neg_top10,
                                     {domain}__S_neg_pure_top1, {domain}__S_neg_all_topk
posneg_percentiles.tsv               per (domain, kind): n, mean, std, p1,p5,p10,p25,p50,p75,p90,p95,p99
hist_posneg_<domain>.png             S_pos vs S_neg_top1 density, with
                                      p10(S_pos), p90/p95(S_neg), current α/β marked
hist_posneg_global.png               3 panels (S_pos, S_neg_top1, S_neg_pure_top1) across domains
alpha_beta_suggest.txt               OOD-robust α/β candidates:
                                      α = max_domain(p90 S_neg_top1) — penalize top ~10% hardest negs
                                      β = min_domain(p10 S_pos)     — penalize bottom ~10% positives
```

## Launchers (this snapshot)

| WandB run | SLURM launcher | Output dir |
|---|---|---|
| `zv28ve3q` | `documents/code/offline_evaluation/run_hist_zv28ve3q_secondary_aries.sh` | `/mnt/gemini/data2/jiaxuanluo/sim_dist_snapshot/zv28ve3q_secondary/` |
| `tys70s0y` | `documents/code/offline_evaluation/run_hist_tys70s0y_secondary_aries.sh` | `/mnt/gemini/data2/jiaxuanluo/sim_dist_snapshot/tys70s0y_secondary/` |
| `ll5a6p9k` (C1) | _TBD_ — clone `run_hist_tys70s0y_secondary_aries.sh`, swap CKPT path to C1's `_best_acl6060_gs10000.pt` and output dir to `ll5a6p9k_secondary/`. Submit after C1 training finishes. | `/mnt/gemini/data2/jiaxuanluo/sim_dist_snapshot/ll5a6p9k_secondary/` (pending) |

The `_taurus.sh` variants still exist as backup in the same dir; aries variant
is just `sed 's/partition=taurus/partition=aries/'` on top.

## SLURM job IDs

Submission history:
- 43893/43894: threshold-sweep version — cancelled (bias in pos-score collection + extraneous sweep work).
- 43895/43896: posneg-dist on **taurus** — cancelled while PENDING (aries freed up after 43892 kill).
- 43897/43898: posneg-dist on aries — FAILED in 3s with `SyntaxError: invalid syntax` at `threshold_sweep_maxsim.py:778` (stray `kan xa` text). Fixed by removing that line.
- **43899/43900: posneg-dist on aries — COMPLETED (2026-04-23T23:22Z, ~2min each).** Output artifacts live at `/mnt/gemini/data2/jiaxuanluo/sim_dist_snapshot/{zv28ve3q,tys70s0y}_secondary/`.
- **C1 (`ll5a6p9k`) pending**: queued for snapshot after SLURM 43901 (training) finishes.

| WandB run | SLURM job | Status | Log |
|---|---|---|---|
| `zv28ve3q` | 43899 (aries) | COMPLETED | `/mnt/gemini/data1/jiaxuanluo/logs/43899_posneg_zv28ve3q_sec.*` |
| `tys70s0y` | 43900 (aries) | COMPLETED | `/mnt/gemini/data1/jiaxuanluo/logs/43900_posneg_tys70s0y_sec.*` |
| `ll5a6p9k` (C1) | pending on training 43901 | blocked — submit after ckpt exists | — |

After `43901` (C1 training) finishes, submit the C1 snapshot with:

```bash
cd documents/code/offline_evaluation
cp run_hist_tys70s0y_secondary_aries.sh run_hist_ll5a6p9k_secondary_aries.sh
# Edit: CKPT -> .../q3rag_..._tcm_lambda5_wu100_..._best_acl6060_gs10000.pt
#       OUT_DIR -> /mnt/gemini/data2/jiaxuanluo/sim_dist_snapshot/ll5a6p9k_secondary/
#       job-name suffix ll5a6p9k_sec
sbatch run_hist_ll5a6p9k_secondary_aries.sh
```

After snapshot jobs finish:
```bash
cat /mnt/gemini/data2/jiaxuanluo/sim_dist_snapshot/<run_id>_secondary/alpha_beta_suggest.txt
column -ts $'\t' /mnt/gemini/data2/jiaxuanluo/sim_dist_snapshot/<run_id>_secondary/posneg_percentiles.tsv | less -S
# Key pngs for "is α=0.40 / β=0.70 right?":
#   hist_posneg_acl6060.png   (OOD)
#   hist_posneg_gs_pod.png    (in-distribution podcast)
#   hist_posneg_global.png    (side-by-side)
```
