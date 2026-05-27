# HN depth scout — `k=4096`, `TCM off`, partial GSV2 voice-pool data

- **Family / data / task**: `sst_ood_hardneg` / `3variant_gsv2p020_mfa` / `train`
- **Variant tag**: `hn4096_gsv2p020_fbmax`
- **Launcher**: `documents/code/train/term_train/run_mfa_smallest_dense_hn4096_gsv2p020_tcmoff_6gpu_taurus.sh`
- **Primary baseline run**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/iaiyi1m8

## Hypothesis

Keeping the proven `k=4096`, TCM-off retriever recipe fixed, replacing the wiki
synthetic speech portion with partial GigaSpeech-v2 speaker-pool TTS should
improve speaker robustness and OOD dense-domain filtering, especially on ACL
10k-bank retrieval.

## Background / Motivation

Run `iaiyi1m8` established the current `k=4096`, batch-6k, TCM-off recipe as
the HN-depth winner, but it still carries fixed-threshold noise. The current
axis is speaker diversity rather than threshold calibration: use completed local
TTS shards `0-20` first, do not wait for shard `21` or teammate shards `22-31`,
and keep the result namespaced so the full speaker-pool experiment can be
repeated cleanly later.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/iaiyi1m8
- Diff:
  - train JSONL: `/mnt/gemini/data1/jiaxuanluo/term_train_3variant_1m_mfa.jsonl` -> `/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_partial0_20_clean_mfa.jsonl`
  - wiki synthetic speech source: original 3-variant MFA -> partial GSV2 clean-only TTS shards `0-20`, aligned into 20 MFA shards
  - partial MFA rows now include `mfa_term_start_in_chunk`, `mfa_term_end_in_chunk`, and `mfa_term_duration`, matching the baseline training data contract
  - original GigaSpeech rows are read from the baseline MFA-enriched train JSONL while skipping old `wiki_synth_*` rows, so the GigaSpeech portion also keeps `mfa_term_*`
  - MFA MaxSim fallback changed: rows with no covering window now fall back to standard MaxSim over all windows instead of the longest window
  - hparam `hard_neg_k_per_sample`: `4096` (unchanged)
  - hparam `num_gpus`: `8` -> `6`
  - hparam `per_gpu_batch`: `768` -> `1024`
  - hparam `batch_size`: `6144` (unchanged)
  - hparam `max_steps`: `400` (unchanged)
  - hparam `grad_cache_chunk_size`: `256` (unchanged)
  - hparam `tcm_loss_weight`: `0.0` (unchanged; TCM remains fully off)
  - no noise augmentation added in this run

## Expected metrics

Compare against `iaiyi1m8` with:

```bash
python documents/code/general/wandb_tool.py --project qwen3_rag compare iaiyi1m8 <new_run_id> \
  --preset retriever_eval retriever_train --at-best-step --anchor-metric both
```

The run is a winner if partial voice-pool data improves DEV and/or ACL
`topk10_filtered_recall@tau_0p80_gs10000` at matched `step=400` without a
disproportionate increase in `noterm_noise@top10_tau_0p80_gs10000`.

## Verdict

PENDING: update after the run finishes and compare with `iaiyi1m8` at best-step
and matched `step=400`.
