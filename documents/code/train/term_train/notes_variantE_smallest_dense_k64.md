# Variant E, MFA smallest-covering, dense window grid, HN pool k=64, aggressive term_id normalization

Back-fill notes for WandB run `zv28ve3q` (SLURM 43848), which was launched before the experiment_tracking schema flags were threaded through the aries launcher. The launcher already carried a full prose justification inline; this file lifts that into the schema-validated sections so the run can be finalized under the rules.

- **WandB run**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/zv28ve3q
- **Launcher**: `documents/code/train/term_train/run_mfa_smallest_dense_k1024_aries.sh`
- **Family / data / task**: `sst_ood_hardneg` / `3variant_1m_mfa` / `train`
- **Variant tag**: `hnpool_k64_smallest_dense_normAGGR`

## Hypothesis

Forcing the MFA-supervised max-sim gradient through the *smallest* window that still covers the ground-truth term span, combined with a *denser* window grid `{2,3,4,5,6,7,8,10,12,16,20,24}` so that every sample has <=1 frame of context leakage, should strip podcast/lecture contextual acoustics from the positive embedding and reduce the acoustic-domain share of the DEV->ACL6060 OOD gap. Target: `eval_acl6060/recall@10_gs10000 >= 0.85`, `noterm_noise@top10_tau_0p80_gs10000 <= 2.0`.

## Background / Motivation

Variant E baseline (per-sample HN k=1024, hard_max MFA) peaked at `eval_acl6060/recall@10_gs10000 ≈ 0.9085` (from launcher commentary; peak checkpoint is per-sample variant E ep=5 on aries pre-43827). ACL6060 OOD decomposition (no-TCM tsweep baseline) showed total DEV->ACL pos_sim shift = -10.4pp, of which -8.4pp is "acoustic" (seen-in-train terms still drop 8.4pp DEV->ACL) and only -3.9pp is vocabulary novelty. ~80% of the OOD drop is audio-domain, not word-identity. The prior hard_max argmax biased the MFA gradient through the widest covering window on ~30% of samples, baking in neighboring contextual acoustics. This run isolates the MFA window-selection change on top of the proven variantE recipe.

## What changed vs baseline

- **Baseline run URL**: variantE per-sample k=1024 ep=5 (aries), to be confirmed via WandB query in finalizer. Short ref: `43827`-lineage.
- **Diff vs that baseline**:
  - `MAXSIM_WINDOWS`: `6 10 16 24` -> `2 3 4 5 6 7 8 10 12 16 20 24`
  - `mfa_window_selection`: `hard_max` -> `smallest`
  - `HARD_NEG_K`: pool k=64 (isolated the HN strategy change; per_sample k=1024 was ~3x slower per step)
  - `HARD_NEG_K_PER_SAMPLE`: 1024 -> 0
  - `term_id_normalize`: `default` -> `aggressive` (prevents near-variant surface forms like "proposition"/"propositions" from being mined as hard negatives)
  - `EPOCHS`: 5 -> 3 (compute-constrained at smallest+dense, W_total 76 -> 230)
  - Compute: `PER_GPU_BATCH=1536`, `NUM_GPUS=8`, `BATCH_SIZE=12288`, `GRAD_CACHE_CHUNK_SIZE=256`
- **Data**: `/mnt/gemini/data1/jiaxuanluo/term_train_3variant_1m_mfa.jsonl` unchanged.
- **Code**: launched against `qwen3_glossary_neg_train.py` at whatever HEAD was live on 2026-04-22T14:42:27Z.

## Expected metrics

- `eval_acl6060/recall@10_gs1000  >= 0.90` (primary `best/metric_value`)
- `eval_acl6060/recall@10_gs10000 >= 0.85` (secondary `best_secondary/metric_value`)
- `eval_acl6060/noterm_noise@top10_tau_0p80_gs10000 <= 2.0`
- `step_time_ms`: target 55-65 s/it at dense grid; acceptable if <=70 s/it

## Verdict

STATUS:FAILED; best@step=1240; recall@10_gs1000=0.9395; recall@10_gs10000@primary=0.8744; recall@10_gs10000@secondary(step=1240)=0.8744; filt@tau0.80_gs1000=0.7736; filt@tau0.80_gs10000=0.7736; noise@tau0.80_gs10000=2.07; tcm_viol(pos/neg)=0.211/0.040; baseline=r0xi5xkt; delta_gs1000=-0.0016; delta_gs10000=-0.0171
