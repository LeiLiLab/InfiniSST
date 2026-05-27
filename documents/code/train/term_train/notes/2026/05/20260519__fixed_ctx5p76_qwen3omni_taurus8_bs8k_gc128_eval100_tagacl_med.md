## Hypothesis

A fixed 5.76s retriever trained with the same Qwen3-Omni/BGE-M3 recipe as `lh1b88kw` will separate the effect of always using the longest speech context from the effect of variable context assignment. If variable context is useful beyond context length, it should remain competitive against this fixed-long-context control.

## Background / Motivation

The current comparison table has fixed 1.92s, fixed 3.84s, and variable 2.88/3.84/4.80/5.76s results. Because longer speech context can independently improve recall, the variable run needs a fixed 5.76s control before interpreting any variable-context advantage.

## What changed vs baseline

- Parent/source run: `lh1b88kw`.
- Train/dev/tagged-ACL/medicine JSONLs are rebuilt as fixed `5.76s` contexts under `ctx5p76` paths.
- Paper-extracted ACL is not evaluated in this run; tagged ACL is the held-out ACL readout.
- Inline eval runs every 100 steps.
- Checkpoint selection is dev-only: primary `eval_dev/recall@10_gs10000`, secondary `eval_dev/recall@10`.

## Expected metrics

Report dev, tagged ACL, and medicine base/1k/10k recall from WandB at best dev-selected checkpoints. ACL and medicine are readouts only and must not choose the checkpoint or hyperparameters.

## Verdict

Paused by user request after the step-400 eval/checkpoint so the run can be
resumed later. W&B run `zseptpl0` was stopped intentionally, not due to a model
failure. Primary resume checkpoint:
`/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_ctx5p76_bs8k_gc128_eval100_tagacl_med_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_q3o_taurus8_best.pt`.
Secondary dev-recall checkpoint:
`/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_ctx5p76_bs8k_gc128_eval100_tagacl_med_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_q3o_taurus8_best_eval_dev_recallat10.pt`.

For resume speed, `grad_cache_chunk_size=256` is possible but risky on 48GB
A6000s because the gc128 run already peaked around 35.7GB/GPU. Prefer trying
`192` first as the safer compromise.
