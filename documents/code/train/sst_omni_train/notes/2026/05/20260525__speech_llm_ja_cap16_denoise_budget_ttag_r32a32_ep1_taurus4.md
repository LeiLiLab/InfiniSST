# Speech LLM Training: ja Cap16 Denoise-Budget Short-Tag r32/a32 Taurus4

## Hypothesis
The Japanese cap16 denoise-budget short-tag data should give the SLM the same runtime-noise robustness and reduced tag overhead as the German cap16 denoise branch.

## Background / Motivation
The previous Japanese cap16 SLM branch used retriever HN1024 term maps capped at 16 entries. German cap16 was upgraded to denoise-budget short-tag data before training. This run applies that same data policy to Japanese and uses only four Taurus GPUs as requested.

## What changed vs baseline
- Parent data event: `20260525T1506__data_prepare__ja_cap16_denoise_budget_ttag`.
- Train JSONL: `/mnt/gemini/data1/jiaxuanluo/speech_llm_ja_cap16_denoise_budget_20260525/ja/hn1024_tau078_cap16_denoise_budget_ttag_v1/train_s_ja_retriever_hn1024_tau078_cap16_denoise_budget_ttag_exactboundary.jsonl`.
- Dev JSONL: `/mnt/gemini/data1/jiaxuanluo/speech_llm_ja_cap16_denoise_budget_20260525/ja/hn1024_tau078_cap16_denoise_budget_ttag_v1/dev_s_ja_retriever_hn1024_tau078_cap16_denoise_budget_ttag_exactboundary_first355.jsonl`.
- LoRA is r32/a32.
- Taurus4 topology: `NPROC=4`, `EP=2`, `TP=2`, `sequence_parallel=true`, `GLOBAL_BATCH_SIZE=4`, `MAX_LENGTH=3072`.
- Training is launched through a GPU-idle watcher because Taurus is currently occupied by tagged-ACL batch evals.
- Runtime eval must use `--strip-output-tags term_t`.

## Expected metrics
The immediate goal is a usable Japanese cap16-denoise short-tag checkpoint and HF export. First gate should be tagged ACL raw Japanese with HN1024, tau `0.78`, omit-empty term maps, and short-tag stripping, comparing BLEU recovery and TERM_ACC against the existing Japanese cap16 and no-RAG readouts.

## Verdict
Failed. Detached Taurus watcher PID `2229131` selected host GPUs `4,5,6,7` and launched training at `2026-05-25T15:23:13Z`. W&B initialized as run `c3xxgy7s`, but Megatron failed during MoE layer initialization with CUDA OOM before the first checkpoint. At failure time the selected GPUs were also occupied by tagged ACL de cap16 denoise simuleval/vLLM workers, so this is a GPU-contention/OOM launch failure rather than a data-prep failure.

- W&B: `https://wandb.ai/luojiaxuan1215-johns-hopkins-university/sst_omni/runs/c3xxgy7s`
- Train log: `/mnt/gemini/data1/jiaxuanluo/logs/speech_llm_ja_cap16_denoise_budget_ttag_r32a32_ep1_taurus4/train_keep1.0_r32_20260525_232314.log`
- Failed run dir: `/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_ja_cap16_denoise_budget_ttag_r32a32_ep1_taurus4/keep1.0_r32/v1-20260525-232327`
