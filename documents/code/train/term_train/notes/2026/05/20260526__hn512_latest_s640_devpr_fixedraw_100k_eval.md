# HN512 latest step-640 dev fixed-raw PR eval

## Hypothesis

The HN512 `bkcnqlg9` latest checkpoint at step 640 may be mature enough to add a fourth HN-depth curve to Figure 5 if its dev fixed-raw precision-recall curve sits between HN256 and HN1024 or otherwise clarifies the HN-depth trend.

## Background / Motivation

Figure 5 currently uses dev-only fixed-raw-denominator curves for no-HN, HN256, and HN1024. The current HN512 latest checkpoint is:

`/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8190_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8190_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn512_tcmoff_ep6_v3_smallest_dense_normAGGR_gpu234567_taurus_latest.pt`

This eval checks whether the available HN512 checkpoint is figure-worthy before spending more Aries time on an 8-GPU resume.

## What changed vs baseline

- Checkpoint: HN512 `bkcnqlg9` latest, step 640.
- Scope: dev-only readout; ACL/tagged ACL/medicine are disabled.
- Denominator: `fixed_raw`.
- Retriever banks: raw/base, gs10k, and gs100k from the dev 1M glossary source.
- Tau grid: `0.50..0.90` by `0.01`.
- Execution: detached `srun` step inside Aries allocation `45310`, GPU `2`.

## Expected metrics

HN512 should be considered reasonable for Figure 5 only if its fixed-raw recall-drop curve is monotonic and interpretable relative to HN256/HN1024, rather than looking like an undertrained or noisy partial point. The useful range is the moderate-precision region emphasized by the current HN1024 figure.

## Verdict

COMPLETED. The second detached `srun` step loaded the HN512 latest checkpoint with `[RESUME] ... epoch=4 step=640`, initialized W&B run `iz1x2v3o`, and completed the dev-only fixed-raw readout at `2026-05-26 01:04:11 UTC`.

Key readout:

- `eval_dev/recall@10`: `99.036932`
- `eval_dev/recall@10_gs10000`: `98.774278`
- `eval_dev/recall@10_gs100000`: `98.256922`
- Near 1pp raw-reference recall drop:
  - raw/base: tau `0.74`, precision `14.689778`, drop `1.026744`
  - gs10k: tau `0.74`, precision `13.209895`, drop `1.026744`
  - gs100k: tau `0.72`, precision `10.328320`, drop `0.986946`

Figure decision: do not replace the current Figure 5 paper asset with this HN512 checkpoint. A candidate plot was generated for inspection, but the HN512 step-640 curve sits clearly below the HN256/HN1024 curves in the Figure 5 precision window, so it reads as an undertrained checkpoint rather than a clean HN-depth ablation. Resume training instead.
