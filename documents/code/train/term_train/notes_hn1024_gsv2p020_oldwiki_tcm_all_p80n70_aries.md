# HN depth + TCM scout - `k=1024`, GSV2p020 oldwiki, all-scope TCM

- **Family / data / task**: `sst_ood_hardneg` / `3variant_gsv2p020_oldwiki_mfa` / `train`
- **Variant tag**: `hn1024_gsv2p020_oldwiki_tcmall_p80n70`
- **Launcher**: `documents/code/train/term_train/run_mfa_smallest_dense_hn1024_gsv2p020_oldwiki_tcm_all_p80n70_8gpu_aries.sh`
- **Baseline run candidates**:
  - same `k=1024` HN-depth scout: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/fma3wmh2
  - same GSV2p020 data line at `k=4096`: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/yx52spnl
  - historical strong `k=512` reference: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/tys70s0y

## Hypothesis

Most ACL boundary failures appear to be `clear_noise`, not true false
negatives, so the model should be able to separate positives from high-scoring
noise with a modest absolute-margin TCM objective. With evaluation centered at
`tau=0.75`, `tcm_pos_threshold=0.80` and `tcm_neg_threshold=0.70` provide a
small margin around the deployment threshold without making TCM dominate the
InfoNCE objective.

## Background / Motivation

The previous `topk32` setup was cancelled before scientific use because
`tcm_pos_threshold` was below `tcm_neg_threshold`, which did not enforce a
separation objective. This replacement tests all-scope TCM over the full
combined negative set: in-batch negatives plus per-sample hard negatives.
Compared with `topk32`, all-scope TCM is a better first control because it does
not assume the correct boundary width before we have a dev-selected tau curve.

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
  - hparam `tcm_neg_scope`: `all`
  - hparam `tcm_neg_topk`: `0`
  - hparam `tcm_pos_threshold`: `0.80`
  - hparam `tcm_neg_threshold`: `0.70`
  - eval tau sweep: `0.85, 0.80, 0.75, 0.70`
  - hparam `tcm_warmup_steps`: `0` -> `100`

## Expected metrics

Compare against `fma3wmh2`, `yx52spnl`, and `tys70s0y` with:

```bash
python documents/code/general/wandb_tool.py --project qwen3_rag compare fma3wmh2 yx52spnl tys70s0y <new_run_id> \
  --preset retriever_eval retriever_train --at-best-step --anchor-metric both
```

Primary selection should use dev metrics around `tau=0.75` and `tau=0.80`.
The run is useful if DEV `topk10_filtered_recall@tau_0p75_gs10000` and
`topk10_filtered_recall@tau_0p80_gs10000` improve over the old `k=1024` scout
while fixed-threshold no-term noise stays controlled. ACL6060 remains one-shot
validation, not the source of the threshold choice.

## Verdict

PENDING: update after the run finishes or shows an early recall/noise failure.
