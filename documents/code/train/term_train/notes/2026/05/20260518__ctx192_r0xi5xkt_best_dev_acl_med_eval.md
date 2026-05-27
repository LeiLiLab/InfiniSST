# Fixed 1.92s Context r0xi5xkt Best-Checkpoint Readout

## Hypothesis

The fixed 1.92s context retriever checkpoint from `r0xi5xkt` should provide the
short-context endpoint for the context-length ablation table when evaluated on
the same dev / ACL6060 / medicine base-1k-10k recall shape as the fixed 3.84s
and variable-context readouts.

## Background / Motivation

The checkpoint
`/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_clean_gc_wr1000k_m0.0_maxsim_mfa_variantE_hardneg_per_sample_k1024_tcm_ep5_cold_best.pt`
comes from W&B run `r0xi5xkt`, a legacy fixed-1.92s run using per-sample
hard negatives and TCM training.  This eval-only event records the checkpoint
under the current lineage workflow and evaluates it with the same domain/glossary
readout shape used by the current context-length ablations.

## What changed vs baseline

- Source run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/r0xi5xkt
- Source checkpoint: `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_clean_gc_wr1000k_m0.0_maxsim_mfa_variantE_hardneg_per_sample_k1024_tcm_ep5_cold_best.pt`
- Diff: eval-only fixed 1.92s context readout.  No model weights are changed.
- Dev uses `/mnt/gemini/data1/jiaxuanluo/term_dev_with_wiki_synth_normalized.jsonl`.
- ACL6060 uses the original fixed-1.92s ACL JSONL and the min-norm-2 backfilled ACL gs10k glossary.
- Medicine uses a newly generated fixed-1.92s medicine eval package at `/mnt/gemini/home/jiaxuanluo/medicine_eval_ctx1p92`.

## Expected metrics

Report raw `recall@10` as `base`, `recall@10_gs1000` as `1k`, and
`recall@10_gs10000` as `10k` for dev, ACL6060, and medicine.  This run is a
readout for context-length ablation, not a new checkpoint-selection event.

## Verdict

Completed successfully in W&B run `rpbmrhnq`.  This is a legacy fixed-1.92s
per-sample-HN/TCM checkpoint readout, not the smallest-dense MFA baseline; use
the separate `ctx192_conv5_s1650_eval` event for the paper context-length
ablation.
