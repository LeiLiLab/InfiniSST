# OOD-shift attribution: smallest-MFA + pool-k64 vs aggrNorm alone

Status: first-pass finding, 2026-04-22, based on `run_inline_eval_sweep_ckpts_taurus.sh` job `43854`.

## Motivation

Run 43848 (smallest MFA + dense window grid + pool hard-negatives k=64 + `term_id_normalize=aggressive`) showed a visibly smaller
`eval_acl6060/topk10_filtered_recall@tau_0p80_gs10000` − `eval_dev/topk10_filtered_recall@tau_0p80_gs10000`
gap than run 43849 (conventional MFA + per-sample hard-negatives k=1024 + `term_id_normalize=aggressive`).
We need to know whether the gap shrinkage is driven by:

- **(H1) `term_id_normalize=aggressive`** (false-negative HN filter, the only
  bug fix landed between 43827 and 43848/43849), or
- **(H2) smallest-MFA + dense grid + pool-k=64 HN** (the other changes
  unique to 43848).

Earlier historical checkpoints (43827 baseline, variantE pool-k=64 full-
epoch best) never logged `sweep@tau_0p80` inline, so their numbers could
not be compared directly. We re-ran the training-time `run_sample_eval`
code path offline via `qwen3_glossary_neg_train.py --eval_only --resume`
on a single taurus A6000, holding DEV + ACL jsonl + eval hyperparams
constant. Small absolute drift vs the wandb value (~0.02) is expected
(A100-trained / A6000-reeval bf16 kernel mismatch); the relative
comparison across ckpts on the same eval hardware is apples-to-apples.

## Checkpoints actually evaluated

All ckpts are .pt files under `/mnt/gemini/home/jiaxuanluo/train_outputs/`;
43848 / 43850 were snapshotted to `snapshots/20260422_*` so ongoing
training wouldn't overwrite them mid-eval.

**Valid (fully trained) ckpts**

| tag | norm | step | epoch | HN strategy | MFA selection |
|---|---|---|---|---|---|
| `43827_snap_step1320` | none | 1320 | 3 | per-sample k=1024 | hard_max |
| `ps_k1024_cold_ep5_best` | none | 1960 | 4 | per-sample k=1024 | hard_max |
| `variantE_k64_ep5_best` | none | 2160 | 5 | pool k=64 | hard_max |
| `43849_ps_k1024_normAGGR_best` | aggressive | 600 | 2 | per-sample k=1024 | hard_max |
| `43848_smallest_k64_normAGGR_snap` | aggressive | 640 | 2 | pool k=64 | **smallest + dense grid** |

**Discarded ckpts (file on disk is an overwritten early snapshot, not the
trained model)**

| tag | disk-state |
|---|---|
| `variantE_k128_ep5_best` | step 120 epoch 0, r@10≈0.01 — untrained |
| `variantE_k256_ep5_best` | step 200 epoch 0 — untrained |
| `variantE_k1024_ep5_best` | step 40 epoch 0 — untrained |
| `43850_NOhardneg_normAGGR_snap` | step 80 epoch 0 — 43850 still very early |

The pool-k sweep ckpts (k=128/256/1024) were overwritten by failed
restart attempts before the first meaningful eval — they cannot be used
as variantE-family comparators. 43850 needs another ~300 steps before it
is worth re-snapshotting.

## Results (offline inline-eval re-run; ACL gs10000 bank)

From `summary_20260422_212202.tsv`:

| tag | norm | step | DEV r@10 gs10k | ACL r@10 gs10k | **gap r@10 gs10k** | DEV sweep@0.80 R gs10k | ACL sweep@0.80 R gs10k | **gap sweep@0.80 R gs10k** |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 43827_snap_step1320        | none        | 1320 | 0.9469 | 0.8760 | **−0.0709** | 0.8710 | 0.6510 | **−0.2200** |
| ps_k1024_cold_ep5_best     | none        | 1960 | 0.9474 | 0.8930 | **−0.0544** | 0.8180 | 0.5120 | **−0.3060** |
| variantE_k64_ep5_best      | none        | 2160 | 0.9497 | 0.9085 | **−0.0412** | 0.8590 | 0.5800 | **−0.2790** |
| 43849_ps_k1024_normAGGR    | aggressive  |  600 | 0.9354 | 0.8264 | **−0.1090** | 0.8580 | 0.5940 | **−0.2640** |
| 43848_smallest_k64_normAGGR|aggressive  |  640 | 0.9267 | 0.8419 | **−0.0848** | 0.8760 | 0.7400 | **−0.1360** |

### Key reading — sweep@0.80 R gs10000 OOD gap

- Three pre-aggrNorm baselines: gap in **[−0.22, −0.31]** regardless of
  HN strategy (ps-k=1024 or pool-k=64).
- **43849 (post-aggrNorm, ps-k=1024, default MFA): gap = −0.264** — no
  meaningful change vs pre-aggrNorm at comparable training exposure.
- **43848 (post-aggrNorm + smallest MFA + dense grid + pool-k=64):
  gap = −0.136** — roughly half the pre-aggrNorm gap, at nearly the
  same training step as 43849.

### Interpretation

- `term_id_normalize=aggressive` alone does NOT close the OOD gap
  (43849 stays in the pre-aggrNorm range).
- The OOD-gap reduction observed at 43848 is therefore driven by the
  **smallest-MFA + dense window grid + pool-k=64 HN** bundle, not by
  aggressive term-id normalization.
- Smallest-MFA forces the gradient through the narrowest covering
  window, stripping surrounding-context acoustics out of the
  positive-audio-to-term similarity — that is directly what an OOD
  corpus (ACL-6060 vs 3-variant training) benefits from, so the
  direction is mechanistically consistent.

### Caveats

1. 43849 was killed at step 600, 43848 is still training (snapshot at
   step 640). 43827 / variantE_k64 / ps_k1024_cold are fully trained
   (1320 / 2160 / 1960 steps). **The cleanest comparison pair is
   43849@600 vs 43848@640** (same training exposure, same aggrNorm,
   only MFA+HN differ). That pair shows gap −0.264 → −0.136.
2. On-disk offline numbers differ from wandb (~0.01–0.03) because
   training ran on aries/A100 and the re-eval ran on taurus/A6000
   (bf16 kernel drift). This bias is constant across ckpts here.
3. We cannot separately attribute the 43848 improvement to "smallest
   MFA alone" vs "pool-k=64 HN alone" without a companion run. Scheme B
   (default MFA + pool-k=64 + aggrNorm, bs=12288) is recommended as the
   next ablation — cheapest compute for the missing isolation.

## Artifacts

- Per-ckpt eval logs: `/mnt/gemini/data1/jiaxuanluo/offline_eval/inline_sweep_tau0p80/<tag>.log`
- Aggregated TSV: `/mnt/gemini/data1/jiaxuanluo/offline_eval/inline_sweep_tau0p80/summary_20260422_212202.tsv`
- Sweep runner: `documents/code/offline_evaluation/run_inline_eval_sweep_ckpts_taurus.sh`
- Single-ckpt runner: `documents/code/offline_evaluation/inline_eval_retriever.sh`
- TSV parser: `documents/code/offline_evaluation/parse_inline_eval_results.py`
- Submitted as SLURM job `43854` on taurus partition, 1 × A6000, ~11 min total.
