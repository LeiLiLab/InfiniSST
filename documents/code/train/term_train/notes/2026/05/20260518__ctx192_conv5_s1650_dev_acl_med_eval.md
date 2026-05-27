# Fixed 1.92s Smallest-Dense MFA Conv5 Step-1650 Readout

## Hypothesis

The fixed 1.92s smallest-dense MFA conv5 checkpoint from `7xu2b4so` provides
the correct short-context baseline for the context-length ablation table when
evaluated on the same dev / ACL6060 / medicine base-1k-10k recall shape as the
fixed 3.84s and variable-context readouts.

## Background / Motivation

The earlier `r0xi5xkt` checkpoint is a legacy per-sample-HN/TCM run and is not
the smallest-dense MFA baseline.  This event instead evaluates the conv5
smallest-dense MFA checkpoint selected by the dev filtered-recall secondary
tracker from W&B run `7xu2b4so`.

## What changed vs baseline

- Source run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/7xu2b4so
- Source checkpoint: `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_mfa_gsdedup_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_conv5_bs12k_smallest_dense_normAGGR_8gpu_aries_best_eval_dev_topk10_filtered_recallattau_0p80_gs10000.pt`
- Source anchor: `best_secondary/step=1650`, metric `eval_dev/topk10_filtered_recall@tau_0p80_gs10000`.
- Diff: eval-only fixed 1.92s context readout.  No model weights are changed.
- Dev uses `/mnt/gemini/data1/jiaxuanluo/term_dev_with_wiki_synth_normalized.jsonl`.
- ACL6060 uses the original fixed-1.92s ACL JSONL and the min-norm-2 backfilled ACL gs10k glossary.
- Medicine uses the fixed-1.92s medicine package at `/mnt/gemini/home/jiaxuanluo/medicine_eval_ctx1p92`.

## Expected metrics

Report raw `recall@10` as `base`, `recall@10_gs1000` as `1k` when available,
and `recall@10_gs10000` as `10k` for dev, ACL6060, and medicine.  For dev, the
fixed-1.92s base GT bank is larger than 1k terms, so the script may skip gs1k
as a non-expanding bank; treat that as base-equivalent or report N/A depending
on the final table convention.

## Verdict

Completed successfully as W&B run `e3kljn9e`.  This is the fixed 1.92s
smallest-dense MFA conv5 readout from source run `7xu2b4so` at
`best_secondary/step=1650`; use W&B run `e3kljn9e` as the metric source for the
dev / ACL6060 / medicine base-1k-10k recall table.  Dev gs1k was skipped because
the fixed-1.92s GT bank already has 1852 terms.
