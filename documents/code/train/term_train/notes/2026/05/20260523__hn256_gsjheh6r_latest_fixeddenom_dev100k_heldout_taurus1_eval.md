# HN256 gsjheh6r latest fixed-denominator dev100k + held-out eval

## Hypothesis

The later `gsjheh6r` latest checkpoint may give a better HN256 readout than the
older `lrdx14pm` best-secondary checkpoint used in the current HN report.

## Background / Motivation

The existing report uses HN256 eval run `ykwbip03`, which loaded the
`lrdx14pm` best-secondary step-1200 checkpoint.  A newer checkpoint was saved
by the ACL-metric-reset continuation run `gsjheh6r`:

`/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8190_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8190_gc256_wr1000k_m0.0_maxsim_mfa_variantE_hn256_tcmoff_ep6_v3_smallest_dense_normAGGR_gpu012367_taurus_step1200_aclmetric_reset_latest.pt`

## What changed vs baseline

- Checkpoint: replace `lrdx14pm` best-secondary with `gsjheh6r` latest.
- Eval protocol remains fixed-denominator:
  - dev tau selection uses raw/base, gs10k, and gs100k.
  - ACL6060, tagged ACL6060, and strict medicine are held-out readouts.
  - held-out candidate banks include raw/base, gs1k, and gs10k.
- Tau grid remains `0.50..0.90` at stride `0.01`.
- Dev-1M is still skipped for HN256.
- Compute: Taurus GPU 6, one eval process.

## Expected metrics

Compare the selected strict raw-included `<0.5 pp`, `<1.0 pp`, and `<1.5 pp`
dev-drop tau values and held-out raw/1k/10k R/P against the previous HN256
`ykwbip03` readout.  The result should indicate whether the HN256 row in the
main fixed-denominator report should switch from `lrdx14pm` to `gsjheh6r`.

## Verdict

SUCCESS.  W&B run: `8h9q0v4t`.

The `gsjheh6r` latest checkpoint was loaded as epoch 3 / step 1440:

`/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8190_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8190_gc256_wr1000k_m0.0_maxsim_mfa_variantE_hn256_tcmoff_ep6_v3_smallest_dense_normAGGR_gpu012367_taurus_step1200_aclmetric_reset_latest.pt`

It is a better HN256 readout than the older `ykwbip03` / `lrdx14pm`
checkpoint.  Use `8h9q0v4t` as the current HN256 line for the
fixed-denominator report.

Dev unfiltered recall:

| bank | recall@10 |
|---|---:|
| raw/base | 99.17 |
| gs10k | 98.94 |
| gs100k | 98.53 |

Strict raw-included dev selection:

| budget | tau | max drop | raw drop | gs10k drop | gs100k drop | dev gs100k R/P |
|---|---:|---:|---:|---:|---:|---:|
| <0.5 pp | 0.70 | 0.4378 | 0.4378 | 0.2467 | 0.0716 | 98.46 / 10.26 |
| <1.0 pp | 0.76 | 0.9233 | 0.9233 | 0.6925 | 0.3582 | 98.17 / 12.51 |
| <1.5 pp | 0.80 | 1.4088 | 1.4088 | 1.1780 | 0.7720 | 97.76 / 15.82 |

Held-out gs10k readout at the strict raw-included operating points:

| setting | ACL gs10k R/P | tagged gs10k R/P | medicine gs10k R/P |
|---|---:|---:|---:|
| HN256 tau 0.70, <0.5 pp | 93.23 / 9.48 | 98.30 / 10.03 | 93.81 / 11.22 |
| HN256 tau 0.76, <1.0 pp | 91.89 / 12.33 | 97.64 / 12.62 | 90.61 / 15.56 |
| HN256 tau 0.80, <1.5 pp | 87.97 / 20.94 | 95.89 / 19.35 | 86.71 / 22.56 |

Compared with the old HN256 row, the recall-first `<0.5 pp` operating point
moves from tau `0.69` to `0.70`, and held-out gs10k recall improves on all
three domains: ACL `92.06 -> 93.23`, tagged ACL `97.92 -> 98.30`, medicine
`93.11 -> 93.81`, with similar precision.

Report artifacts updated:

- `documents/code/train/term_train/reports/20260522_nohn_vs_hn1024_fixeddenom_eval_report.md`
- `documents/code/train/term_train/reports/figures/20260523_dev_gs100k_pr_nohn_hn256_hn1024.png`
- `documents/code/train/term_train/reports/figures/20260523_heldout_lt0p5_raw1k10k_delta_nohn_hn256_hn1024.png`
- `documents/code/train/term_train/reports/figures/20260523_heldout_tradeoff_by_budget_summary.png`

First attempt note: the initial launch was stopped because W&B init rejected the
long `data:*` tag.  The launcher was updated to use the shorter
`DATA_TAG=vctx576_hn256_latest_fixeddenom` before relaunch.
