# Fixed 5.76s Qwen3-Omni Retriever Resume, Aries 6GPU

## Hypothesis

Resuming the fixed `5.76s` context control from `zseptpl0` should continue the
same HN1024/Qwen3-Omni/BGE-M3 training line without changing the data or
selection protocol. The Aries run uses the available six GPUs
`0,1,2,3,6,7`; because the DDP training path requires equal per-rank batches,
the effective batch is `1365 * 6 = 8190`, the closest lower value to the
original target batch `8192`.

## Background / Motivation

The taurus run `zseptpl0` was intentionally paused after the step-400
eval/checkpoint so the user could resume later. That run is the fixed-longest
context control for the variable-context source run `lh1b88kw`; it keeps ACL as
held-out readout and uses dev metrics for checkpoint selection.

## What changed vs baseline

- Direct parent run: `zseptpl0`.
- Resume checkpoint:
  `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_ctx5p76_bs8k_gc128_eval100_tagacl_med_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_q3o_taurus8_best.pt`
- Compute changes:
  - node/partition: Aries hold allocation `45290`
  - GPU list: `0,1,2,3,6,7`
  - port: `20042`
  - target global batch: `8192`
  - effective equal-rank global batch: `8190`
  - grad cache chunk: `128`
  - workers: `0`
- Unchanged protocol:
  - fixed/eval audio context: `5.76s`
  - hard negatives: `hard_neg_k_per_sample=1024`
  - eval cadence: every `100` train steps
  - primary checkpoint selection: `eval_dev/recall@10_gs10000`
  - secondary saved metric: `eval_dev/recall@10`
  - tagged ACL and medicine are readouts only

## Expected metrics

Use WandB at-best-step bundles after the run starts or finishes. The immediate
startup check should confirm `[RESUME]` loads the parent checkpoint and that the
training log reports `requested_global=8190 effective_global=8190`.

## Verdict

PAUSED on `2026-05-24T01:24:00+00:00` by user request to free Aries GPUs for a
later resume. W&B run `jyb2u787` was running inside Aries hold allocation
`45290`; the training process received SIGTERM and exited cleanly from
torchrun.

The last completed eval/save point before termination was step `1300`. Training
had progressed to at least step `1360`, but the resume-safe latest checkpoint is:
`/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8190_t=0.07_3var_gsv2full_gsdedup_ctx5p76_bs8190_gc128_eval100_tagacl_med_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_q3o_resume_s400_gpu012367_aries6_latest.pt`

Best primary checkpoint remains:
`/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8190_t=0.07_3var_gsv2full_gsdedup_ctx5p76_bs8190_gc128_eval100_tagacl_med_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_q3o_resume_s400_gpu012367_aries6_best.pt`

Best secondary checkpoint remains:
`/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8190_t=0.07_3var_gsv2full_gsdedup_ctx5p76_bs8190_gc128_eval100_tagacl_med_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_q3o_resume_s400_gpu012367_aries6_best_eval_dev_recallat10.pt`
