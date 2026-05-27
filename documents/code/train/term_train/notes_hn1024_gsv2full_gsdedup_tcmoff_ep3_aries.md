# Full GSV2 k1024 TCM-off with GigaSpeech MFA-event dedup

- **Family / data / task**: `sst_ood_hardneg` / `3variant_gsv2full_gsfix_mfa_gsdedup` / `train`
- **Variant tag**: `hn1024_gsv2full_gsdedup_tcmoff_ep3`
- **Launcher**: `documents/code/train/term_train/run_mfa_smallest_dense_hn1024_gsv2full_gsdedup_tcmoff_ep3_8gpu_aries.sh`
- **Primary baseline run**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/ly6sc2mr
- **Top same-family candidates from pre-flight**: `5np0cxmq`, `us4obwe3`, `058tdx9a`

## Hypothesis

Keeping one random row per absolute GigaSpeech MFA term event should remove
overlap-induced duplicate positive gradients without changing the wiki-synth
portion of the training set. This should improve the fairness of the
GigaSpeech-vs-wiki mixture while preserving the term-spotting behavior learned
by the original full GSV2 TCM-off baseline.

## Background / Motivation

GigaSpeech training chunks are 1.92s windows with 0.96s stride so phrase terms
that cross the base unit boundary are not missed. After MFA supervision, the
same absolute term event can appear in adjacent chunks and receive duplicate
loss mass. The deduped data line tests whether the previous strong retriever
metrics partly depended on this overlap weighting.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/ly6sc2mr
- Diff:
  - data: `/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired.jsonl` -> `/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup.jsonl`
  - data preprocessing: non-`wiki_synth_` rows are grouped by `(utter_id, term_key, chunk_idx * 0.96 + mfa_start, chunk_idx * 0.96 + mfa_end)` and one row is randomly kept per group with seed `20260509`
  - data size: total `6,575,566` -> `4,961,904`; GigaSpeech `3,681,831` -> `2,068,169`; wiki synth unchanged at `2,893,735`
  - hparams: same as `hn1024_gsv2full_tcmoff_ep3` (`hard_neg_k_per_sample=1024`, `TCM=off`, `batch_size=12288`, `epochs=3`, `scheduler_epochs=4`)
  - eval selection: unchanged dev gs10000 primary and tau0.80 filtered recall secondary

## Expected metrics

The run is useful if dev `recall@10_gs10000` remains close to the original
TCM-off baseline while no-term noise and tau-filtered precision do not degrade.
A small recall drop is acceptable if the deduped data gives a cleaner baseline
for later context/window ablations.

## Verdict

PENDING: update after training finishes and compare best-step dev bundles
against `ly6sc2mr`, `5np0cxmq`, and `us4obwe3`.
