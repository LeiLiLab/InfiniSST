# Variable-context retriever no-HN ablation, gc256 retry

## Hypothesis

Turning off per-sample hard negatives should isolate how much of `lh1b88kw` comes
from the HN objective versus the variable-context data, MaxSim MFA setup, and
Qwen3-Omni/BGE-M3 encoders. `gc1024` and `gc512` both OOMed on taurus, so this
retry uses `grad_cache_chunk_size=256` while keeping the same no-HN ablation and
exact 8k global batch.

## Background / Motivation

Source run `lh1b88kw` used the balanced 2.88s/3.84s/4.80s/5.76s GSV2-full
GSDedup variable-context dataset with global batch `8192`,
`hard_neg_k_per_sample=1024`, `grad_cache_chunk_size=128`, TCM-off, MaxSim MFA,
and six epochs. Failed attempts: `ioclpbnq` (`gc1024`) and `ibf4rs6a` (`gc512`).

## What changed vs baseline

- Source run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/lh1b88kw
- Retry:
  - `hard_neg_k_per_sample`: `1024` -> `0`
  - `grad_cache_chunk_size`: `128` -> `256`
  - GPU list: `0,1,2,3,4,5,6,7`
  - global batch remains exactly `8192` via `PER_GPU_BATCH=1024` on 8 GPUs.
- Metric tracking:
  - this interrupted run was launched with primary `eval_dev/recall@10_gs10000`
    and secondary `eval_dev/recall@10`.
  - after interruption, the reusable HN-setting launchers were updated so future
    resumes/runs use secondary `eval_acl6060/recall@10` by user request.

## Expected metrics

This should quantify the recall and tau-retention cost of removing HN while
keeping data, encoders, batch size, MaxSim MFA, and TCM-off settings aligned with
`lh1b88kw`.

## Verdict

INTERRUPTED: training was manually cancelled on 2026-05-20 to free the taurus
GPUs for another script. The run reached step 240 and wrote the primary best
checkpoint:

`/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8192_gc256_wr1000k_m0.0_maxsim_mfa_variantE_nohn_tcmoff_ep6_v3_smallest_dense_normAGGR_gpu01234567_aries_best.pt`

Resume later from that checkpoint rather than treating this interrupted run as
final.
