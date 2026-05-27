# Retriever data ablation dev figure

## Hypothesis

Adding Wiki-synthetic terminology supervision to the GigaSpeech-derived
retriever training data improves development-set term retrieval beyond
GigaSpeech-only acoustic term supervision.

## Background / Motivation

The paper main retriever uses GigaSpeech-derived speech-text supervision plus
Wiki-synthetic terminology examples. The user provided a GigaSpeech-only
checkpoint for a data ablation:

`/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_gsonly_bs8192_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_4gpu_aries_best_eval_acl6060_recallat10.pt`

ACL is treated only as held-out readout/provenance. The figure uses the main
development readout for the ablation.

## What changed vs baseline

Added a paper-local figure package under
`plot/figure_09_retriever_data_ablation/` and inserted the figure into the
Results ablation section.

The frozen TSV compares:

- `lh1b88kw`: main retriever, GigaSpeech + Wiki-synthetic training data.
- `g49qabuf`: GigaSpeech-only ablation run, event
  `20260525T1248__retriever_train__varctx576_hn1024_gsonly_tcmoff_ep6_aries4`.

## Expected metrics

The comparison should show a small gain in unfiltered dev Recall@10 and a
larger gain under higher score thresholds.

## Verdict

Completed. On the main dev GS-10k Recall@10 metric, the main retriever reaches
0.9901 while the GigaSpeech-only ablation reaches 0.9830. At tau 0.80 on the
same dev GS-10k bank, filtered recall is 0.9791 vs 0.9586.
