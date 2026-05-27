# TCM Anchor Baseline Lock

This locks the source artifacts for the distribution-anchored TCM sweep without
making this markdown an authoritative metric store. Exact metrics should be
queried from W&B history at the recorded steps.

## Baseline Artifacts

- Primary source run: `us4obwe3`
- Primary source step: `2650`
- Primary checkpoint:
  `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_r3auto1m_bs12k_smallest_dense_normAGGR_8gpu_aries_best.pt`
- 1M eval run for the same exported checkpoint: `pacexfw7`
- Superseding continuation run for reference only: `la8vkt47`

## Metric Query Commands

Use W&B history reads, not last-step summary keys:

```bash
python documents/code/general/wandb_tool.py --project qwen3_rag history us4obwe3 \
  --keys best/metric_value best/step eval_dev/recall@10_gs10000 eval_dev/step \
  --eval-rows-only

python documents/code/general/wandb_tool.py --project qwen3_rag history pacexfw7 \
  --keys eval_dev/recall@10_gs1000000 eval_dev/step \
  --eval-rows-only
```

