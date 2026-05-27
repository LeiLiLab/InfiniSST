# Speech LLM Training: ja Cap16 Denoise-Budget Short-Tag r32/a32 Taurus4 Retry1

## Hypothesis
The Japanese cap16 denoise-budget short-tag data is valid; the prior Taurus4 attempt failed from GPU contention after W&B init, not from malformed training data.

## Background / Motivation
The first Taurus4 attempt selected GPUs `4,5,6,7` and failed with CUDA OOM during Megatron MoE layer initialization because those GPUs were concurrently occupied by DE tagged ACL simuleval/vLLM workers. This retry fixes the allocation to GPUs `0,1,2,3`, which were idle at submission time.

## What changed vs baseline
- Parent data event: `20260525T1506__data_prepare__ja_cap16_denoise_budget_ttag`.
- Failed prior training event: `20260525T1513__speech_llm_train__ja_cap16_denoise_budget_ttag_r32a32_ep1_taurus4`.
- Failed prior W&B run: `c3xxgy7s`.
- Train JSONL: `/mnt/gemini/data1/jiaxuanluo/speech_llm_ja_cap16_denoise_budget_20260525/ja/hn1024_tau078_cap16_denoise_budget_ttag_v1/train_s_ja_retriever_hn1024_tau078_cap16_denoise_budget_ttag_exactboundary.jsonl`.
- Dev JSONL: `/mnt/gemini/data1/jiaxuanluo/speech_llm_ja_cap16_denoise_budget_20260525/ja/hn1024_tau078_cap16_denoise_budget_ttag_v1/dev_s_ja_retriever_hn1024_tau078_cap16_denoise_budget_ttag_exactboundary_first355.jsonl`.
- LoRA remains r32/a32.
- Taurus4 topology remains `NPROC=4`, `EP=2`, `TP=2`, `sequence_parallel=true`, `GLOBAL_BATCH_SIZE=4`, `MAX_LENGTH=3072`.
- Host GPU allocation is fixed to `0,1,2,3`.
- Runtime eval must use `--strip-output-tags term_t`.

## Expected metrics
The immediate goal is a usable Japanese cap16-denoise short-tag checkpoint and HF export. First gate should be tagged ACL raw Japanese with HN1024, tau `0.78`, omit-empty term maps, and short-tag stripping, comparing BLEU recovery and TERM_ACC against the existing Japanese cap16 and no-RAG readouts.

## Verdict
Training completed and wrote the Megatron checkpoint under `v2-20260525-235251`, but the initial post-training HF export failed because a stale stage directory already existed at `/mnt/data1/jiaxuanluo/hf_export_stage/speech_llm_ja_cap16_denoise_budget_ttag_r32a32_ep1_taurus4/v2-20260525-235251-hf.stage.31`.

HF export retry submitted detached on Taurus GPU `6` at `2026-05-25T18:38:24Z`. The retry used a fresh writable stage root because the previous stage-root parents are root-owned and not writable by this user.

That retry failed after `Merge LoRA` while loading HF checkpoint shards onto a single A6000: `torch.OutOfMemoryError` at shard `9/15`, with GPU 0 at `47.51 GiB` used. Retry2 is submitted at `2026-05-25T18:54:44Z` on Taurus GPUs `2,3,4,5` with `HF_EXPORT_SWIFT_EXTRA_ARGS="--device_map auto"` so HF loading can shard across visible GPUs.

Retry2 completed at `2026-05-25T19:23:25Z`. HF validation passed for both NFS output and Taurus local cache: 15 safetensor shards, approximately `66G`.

- W&B: `https://wandb.ai/luojiaxuan1215-johns-hopkins-university/sst_omni/runs/wkoonqux`
- Train log: `/mnt/gemini/data1/jiaxuanluo/logs/speech_llm_ja_cap16_denoise_budget_ttag_r32a32_ep1_taurus4/train_keep1.0_r32_20260525_235237.log`
- Run dir: `/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_ja_cap16_denoise_budget_ttag_r32a32_ep1_taurus4/keep1.0_r32/v2-20260525-235251`
- HF output dir: `/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_ja_cap16_denoise_budget_ttag_r32a32_ep1_taurus4/keep1.0_r32/v2-20260525-235251-hf`
- Taurus local HF cache: `/mnt/data1/jiaxuanluo/slm_local_cache/ja_tagged_acl_20260525/cap16_denoise_ttag/v2-20260525-235251-hf`
- HF export retry stage root: `/mnt/data1/jiaxuanluo/slm_local_cache/hf_export_stage_retry/speech_llm_ja_cap16_denoise_budget_ttag_r32a32_ep1_taurus4`
- HF export retry log: `/mnt/gemini/data1/jiaxuanluo/logs/export_ja_cap16_denoise_ttag_hf_20260525T183729Z/export.out`
- HF export retry2 log: `/mnt/gemini/data1/jiaxuanluo/logs/export_ja_cap16_denoise_ttag_hf_retry2_20260525T185444Z/export.out`
