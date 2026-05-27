# HN depth scout - `k=4096`, `TCM off`, full GSV2 0-31 data

- **Family / data / task**: `sst_ood_hardneg` / `3variant_gsv2full_mfa` / `train`
- **Variant tag**: `hn4096_gsv2full_fbmax`
- **Launcher**: `documents/code/train/term_train/run_mfa_smallest_dense_hn4096_gsv2full_tcmoff_6gpu_taurus.sh`
- **Primary baseline run**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/iaiyi1m8
- **Partial GSV2 baseline run**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/yx52spnl

## Hypothesis

Using the full clean GSV2 speaker-pool TTS set for wiki_synth shards 0-31 should
recover the wiki/OOD coverage lost by the partial scout while preserving the
speaker diversity gains from GSV2. With the MFA no-cover fallback fixed to
standard MaxSim, this should outperform the partial GSV2 scout and be
competitive with or better than `iaiyi1m8` on ACL 10k dense retrieval.

## Background / Motivation

Run `yx52spnl` showed that partial GSV2 data can train normally after the
fallback fix, but it was a near-tie / slight loser because only shards 0-20 were
available at the time. The teammate handoff now provides shards 22-31, and local
shard 21 is also present. This run replaces the wiki_synth portion with the full
clean GSV2 0-31 set instead of mixing in old wiki rows as a temporary patch.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/iaiyi1m8
- Diff:
  - train JSONL: `/mnt/gemini/data1/jiaxuanluo/term_train_3variant_1m_mfa.jsonl` -> `/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired.jsonl`
  - wiki synthetic speech source: old 3-variant TTS -> full clean GSV2 speaker-pool TTS shards `0-31`
  - MFA/wiki build input: reuse existing partial `0-20` MFA output, then add local shard `21` plus teammate merged shards `22-31` as an `extra21_31` MFA suffix
  - extra `21-31` raw TTS merge: `1,030,799` rows, with `--global-idx-offset 2061613` so `utter_id` numbering stays compatible with the full `0-31` dataset
  - GigaSpeech repair: `246,792` empty-term/no-term chunks dropped; `79,513` non-empty missing MFA spans recovered from SQLite/TextGrid; `480` remaining unmatched rows dropped
  - final active training data after wiki-rank filtering and GigaSpeech cleanup: `3,681,831` GigaSpeech rows + `2,567,670` clean GSV2 wiki rows = `6,249,501` active rows
  - validation: active rows have non-empty term keys and all required `mfa_term_start_in_chunk`, `mfa_term_end_in_chunk`, and `mfa_term_duration` fields; wiki `utter_id` duplicate count is zero
  - MFA MaxSim fallback remains fixed: rows with no covering window fall back to standard MaxSim over all windows
  - hparam `hard_neg_k_per_sample`: `4096` (unchanged)
  - hparam `num_gpus`: `8` -> `6`
  - hparam `per_gpu_batch`: `768` -> `1024`
  - hparam `batch_size`: `6144` (unchanged)
  - hparam `max_steps`: `400` (unchanged)
  - hparam `grad_cache_chunk_size`: `256` (unchanged)
  - hparam `tcm_loss_weight`: `0.0` (unchanged; TCM remains fully off)

## Expected metrics

Compare against `iaiyi1m8` and `yx52spnl` with:

```bash
python documents/code/general/wandb_tool.py --project qwen3_rag compare iaiyi1m8 yx52spnl <new_run_id> \
  --preset retriever_eval retriever_train --at-best-step --anchor-metric both
```

The run is a winner if it improves ACL `recall@10_gs10000` and
`topk10_filtered_recall@tau_0p80_gs10000` over `yx52spnl` and matches or exceeds
`iaiyi1m8` without a disproportionate increase in
`noterm_noise@top10_tau_0p80_gs10000`.

## Verdict

SUCCESS: full GSV2 gsfix is a winner vs `iaiyi1m8` and `yx52spnl` on ACL/dev
dense recall and tau-filtered recall at the best checkpoint, with a modest ACL
tau-filter no-term-noise increase; use the step-320 checkpoint for export.
