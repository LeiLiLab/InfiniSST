# Speech LLM Training: zh Cap16 Denoise-Budget Short-Tag r32/a32 Taurus4

## Hypothesis
The new zh cap16 denoise-budget short-tag data should close part of the gap between zh medicine RASST and offline+GT by matching the cleaner de/ja SLM data recipe.

## Background / Motivation
The current zh medicine main row descends from the new_v9 SLM branch, while German and Japanese later moved to cap16 HN1024 tau `0.78`, denoise-budget sampling, no-GT emptying, and short `<t>...</t>` assistant tags. The latest verified zh baseline W&B run for that older branch is `2g6kan5y`.

## What changed vs baseline
- Parent data event: `20260526T0437__data_prepare__zh_cap16_denoise_budget_ttag`.
- Train JSONL: `/mnt/data1/jiaxuanluo/speech_llm_zh_cap16_denoise_budget_20260526/zh/hn1024_tau078_cap16_denoise_budget_ttag_v1/train_s_zh_retriever_hn1024_tau078_cap16_denoise_budget_ttag_exactboundary.jsonl`.
- Dev JSONL: `/mnt/data1/jiaxuanluo/speech_llm_zh_cap16_denoise_budget_20260526/zh/hn1024_tau078_cap16_denoise_budget_ttag_v1/dev_s_zh_retriever_hn1024_tau078_cap16_denoise_budget_ttag_exactboundary_first355.jsonl`.
- LoRA is r32/a32.
- Taurus4 topology is `NPROC=4`, `EP=2`, `TP=2`, `sequence_parallel=true`, `GLOBAL_BATCH_SIZE=4`, `MAX_LENGTH=3072`.
- Host GPU allocation is fixed to `0,1,2,3`, which were idle at submission preflight.
- Checkpoints, logs, HF staging, and local HF cache are on `/mnt/data1` because `/mnt/gemini/data1` and `/mnt/gemini/data2` are full.
- HF export starts with `HF_EXPORT_SWIFT_EXTRA_ARGS="--device_map auto"` to avoid the single-A6000 export OOM seen in the ja retry.
- Runtime eval must use `--strip-output-tags term_t`.

## Expected metrics
The immediate goal is a usable Chinese cap16-denoise short-tag checkpoint and HF export. First gates should be tagged ACL raw zh and medicine hard/raw zh with HN1024, tau `0.78`, omit-empty term maps, and short-tag stripping.

## Verdict
Startup verified. Training is running on Taurus GPUs `0,1,2,3` with W&B run `ccgjhu4r`.

- W&B: `https://wandb.ai/luojiaxuan1215-johns-hopkins-university/sst_omni/runs/ccgjhu4r`
- Train log: `/mnt/data1/jiaxuanluo/logs/speech_llm_zh_cap16_denoise_budget_ttag_r32a32_ep1_taurus4/train_keep1.0_r32_20260526_142437.log`
- Run dir: `/mnt/data1/jiaxuanluo/slm/speech_llm_zh_cap16_denoise_budget_ttag_r32a32_ep1_taurus4/keep1.0_r32/v0-20260526-142452`
