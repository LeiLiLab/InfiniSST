# HN depth + TCM scout - `k=1024`, GSV2p020 oldwiki, candidate-aware TCM

- **Family / data / task**: `sst_ood_hardneg` / `3variant_gsv2p020_oldwiki_mfa` / `train`
- **Variant tag**: `hn1024_gsv2p020_oldwiki_tcm_topk32`
- **Launcher**: `documents/code/train/term_train/run_mfa_smallest_dense_hn1024_gsv2p020_oldwiki_tcm_topk32_8gpu_aries.sh`
- **Baseline run candidates**:
  - same `k=1024` HN-depth scout: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/fma3wmh2
  - same GSV2p020 data line at `k=4096`: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/yx52spnl
  - historical strong `k=512` reference: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/tys70s0y

## Hypothesis

`k=4096` is too expensive for sweep work and shows overfitting/noise pressure
when trained longer. Moving the speaker-diverse GSV2p020 oldwiki recipe to
`k=1024` should restore one-epoch speed while retaining most of the dense-bank
retrieval benefit. A light candidate-aware TCM branch over only the hardest
top-32 negatives should improve threshold calibration without overwhelming the
InfoNCE objective.

## Background / Motivation

The `mlcepvil` continuation confirmed that a fully trained `k=4096` run can
improve filtered recall, but it takes about 1060 steps per epoch and begins to
look like an expensive endpoint rather than a useful sweep point. The earlier
`k=1024` scout was much cheaper, but it used the older training data recipe.
This run combines the faster HN depth with the multi-speaker GSV2p020 oldwiki
training set and tests candidate-aware TCM in the same launcher.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/fma3wmh2
- Diff:
  - train JSONL: `/mnt/gemini/data1/jiaxuanluo/term_train_3variant_1m_mfa.jsonl` -> `/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_partial0_20_oldwiki_clean_mfa.jsonl`
  - hparam `hard_neg_k_per_sample`: `1024` (unchanged vs same-k scout; `4096` -> `1024` vs GSV2p020 oldwiki reference)
  - hparam `num_gpus`: `8`
  - hparam `per_gpu_batch`: `1536`
  - hparam `batch_size`: `12288`
  - hparam `grad_cache_chunk_size`: `256` -> `512`
  - hparam `max_steps`: `200` in the old `k=1024` scout / `400` in GSV2p020 oldwiki reference -> `530`, about one full epoch at this batch size
  - hparam `tcm_pos_loss_weight`: `0.0` -> `0.10`
  - hparam `tcm_neg_loss_weight`: `0.0` -> `0.50`
  - hparam `tcm_loss_form`: `squared_hinge` -> `hinge`
  - hparam `tcm_reduction`: `mean_viol`
  - hparam `tcm_neg_scope`: `all` -> `topk`
  - hparam `tcm_neg_topk`: `0` -> `32`
  - hparam `tcm_pos_threshold`: `0.76`
  - hparam `tcm_neg_threshold`: `0.80`
  - hparam `tcm_warmup_steps`: `0` -> `100`

## Expected metrics

Compare against `fma3wmh2`, `yx52spnl`, and `tys70s0y` with:

```bash
python documents/code/general/wandb_tool.py --project qwen3_rag compare fma3wmh2 yx52spnl tys70s0y <new_run_id> \
  --preset retriever_eval retriever_train --at-best-step --anchor-metric both
```

The run is useful if DEV `topk10_filtered_recall@tau_0p80_gs10000` improves over
the old `k=1024` scout and approaches the GSV2p020 oldwiki `k=4096` reference
with lower runtime and controlled `noterm_noise@top10_tau_0p80_gs10000`. ACL is
kept as one-shot validation, not as the source of TCM parameter selection.

## Verdict

CANCELLED on 2026-04-25 before any scientific read. The initial threshold
choice was invalid for a separation-style TCM objective:
`tcm_pos_threshold=0.76` was below `tcm_neg_threshold=0.80`, so the auxiliary
loss did not enforce positives above negatives with a margin. Do not use WandB
run `kakdzlx8` as evidence for TCM.
