# Fixed 3.84s Context Step-1040 Readout

## Hypothesis

The fixed 3.84s context retriever checkpoint selected by dev at step 1040 should
provide a direct context-length ablation against the current variable-context
main result when evaluated on the same recall table shape: dev, ACL6060, and
medicine with base / 1k / 10k glossary banks.

## Background / Motivation

W&B run `dxwrgbln` trained the full GSV2 k1024 TCM-off retriever with fixed
3.84s GigaSpeech contexts.  The run was cancelled before convergence, but its
dev-best checkpoint at step 1040 was saved as `_best.pt`.  This eval-only event
uses that saved checkpoint instead of trying to resume training.

## What changed vs baseline

- Source run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/dxwrgbln
- Source checkpoint: `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_mfa_gsdedup_ctx384_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_bs12k_smallest_dense_normAGGR_8gpu_aries_best.pt`
- Diff: eval-only fixed 3.84s context readout.  No model weights are changed.
- Dev uses `/mnt/gemini/data1/jiaxuanluo/term_dev_dataset_m4.jsonl` with P31 dev filler glossary at 1k and 10k.
- ACL6060 uses the fixed 3.84s ACL JSONL and the min-norm-2 backfilled ACL gs10k glossary.
- Medicine uses the `context_duration_tag=3p84` subset filtered from the variable-context medicine readout set.

## Expected metrics

Report raw `recall@10` as `base`, `recall@10_gs1000` as `1k`, and
`recall@10_gs10000` as `10k` for dev, ACL6060, and medicine.  This run is a
readout for context-length ablation, not a new checkpoint-selection event.

## Verdict

Completed successfully in W&B run `hsttrl08`.  The fixed 3.84s step-1040
checkpoint produced a full dev / ACL6060 / medicine base-1k-10k recall readout;
use W&B history for the authoritative metric values.
