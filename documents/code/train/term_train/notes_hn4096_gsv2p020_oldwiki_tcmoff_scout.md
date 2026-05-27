# HN depth scout - `k=4096`, `TCM off`, partial GSV2 + old wiki supplement

- **Family / data / task**: `sst_ood_hardneg` / `3variant_gsv2p020_oldwiki_mfa` / `train`
- **Variant tag**: `hn4096_gsv2p020_oldwiki_fbmax`
- **Launcher**: `documents/code/train/term_train/run_mfa_smallest_dense_hn4096_gsv2p020_oldwiki_tcmoff_6gpu_taurus.sh`
- **Primary baseline run**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/iaiyi1m8
- **Previous partial GSV2 run**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/yx52spnl

## Hypothesis

The partial GSV2 clean-only scout underperformed because it reduced the active
wiki_synth portion by about one third. Keeping the GSV2 speaker-diverse rows
and supplementing the missing active wiki count from the old clean wiki data
should recover ACL/OOD term coverage while preserving the speaker-diversity
benefit.

## Background / Motivation

Run `yx52spnl` confirmed that the MFA fallback fix (`no-cover -> MaxSim over all
windows`) prevents collapse, but the run was only a near-tie / slight loser vs
`iaiyi1m8` on ACL best-step metrics. Data audit showed that runtime filters
made `yx52spnl` train on 1,684,911 active wiki rows, while `iaiyi1m8` trained on
2,583,564 active wiki rows. This run matches the baseline active wiki count by
copying old clean wiki rows into the partial GSV2 train set.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/iaiyi1m8
- Diff:
  - train JSONL: `/mnt/gemini/data1/jiaxuanluo/term_train_3variant_1m_mfa.jsonl` -> `/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_partial0_20_oldwiki_clean_mfa.jsonl`
  - partial GSV2 rows: keep all rows from `/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_partial0_20_clean_mfa.jsonl`
  - old clean wiki supplement: add 898,653 rows from baseline old wiki clean data, prioritizing terms missing from the partial active wiki set
  - active runtime composition after `wiki_rank=1000000`, `noisy_ratio=0.0`: 3,929,103 GigaSpeech rows + 2,583,564 clean wiki rows = 6,512,667 total, matching `iaiyi1m8`
  - wiki active unique terms: 967,492 in the supplemented mix vs 886,404 in `iaiyi1m8` and 634,941 in `yx52spnl`
  - MFA MaxSim fallback remains fixed: rows with no covering window fall back to standard MaxSim over all windows
  - hparam `hard_neg_k_per_sample`: `4096` (unchanged)
  - hparam `num_gpus`: `8` -> `6`
  - hparam `per_gpu_batch`: `768` -> `1024`
  - hparam `batch_size`: `6144` (unchanged)
  - hparam `max_steps`: `400` (unchanged)
  - hparam `grad_cache_chunk_size`: `256` (unchanged)
  - hparam `tcm_loss_weight`: `0.0` (unchanged; TCM remains fully off)

## Expected metrics

Compare against both `iaiyi1m8` and `yx52spnl` with:

```bash
python documents/code/general/wandb_tool.py --project qwen3_rag compare iaiyi1m8 yx52spnl <new_run_id> \
  --preset retriever_eval retriever_train --at-best-step --anchor-metric both
```

The run is a winner if it keeps the DEV gains from `yx52spnl` and recovers or
beats `iaiyi1m8` on ACL `recall@10_gs10000` and
`topk10_filtered_recall@tau_0p80_gs10000` without increasing fixed-threshold
noise disproportionately.

## Verdict

PENDING: update after the run finishes and compare with `iaiyi1m8` and
`yx52spnl` at best-step and matched `step=400`.
