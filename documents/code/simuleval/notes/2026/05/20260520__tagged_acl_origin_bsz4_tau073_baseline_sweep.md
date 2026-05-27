# Tagged ACL origin-bsz4 baseline sweep, tau=0.73

## Hypothesis

The baseline streaming SLM checkpoints can use tau-filtered MaxSim term maps
from the variable-context retriever without changing the Speech LLM weights.
The strict tagged ACL glossary is the fixed denominator for term metrics.

## Background / Motivation

This is the main tagged ACL readout before returning to medicine.  It evaluates
the existing no-special-term-map streaming checkpoints for zh, ja, and de under
latency multipliers 1, 2, 3, and 4.  Retrieval uses the lh1b88kw variable-context
retriever with timeline mode: one retrieval per vLLM generation step, current
window length `lm * 0.96s`, plus a 1.92s look-back, filtered at tau=0.73.

## What changed vs baseline

- Baseline Speech LLM checkpoints:
  - `/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-zh-s_origin-bsz4`
  - `/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-ja-s_origin-bsz4`
  - `/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-de-s_origin-bsz4`
- Retriever checkpoint:
  - `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt`
- Retrieval glossaries: tagged raw, tagged+filler 1k, tagged+filler 10k.
- Metric glossary: fixed tagged raw terms only for all term metrics.
- FCR policy: term-map-gated source/ref negative sentence policy.

## Expected metrics

Expect larger retrieval banks to reduce precision/noise but preserve recall due
to tau=0.73.  The baseline SLM may not fully exploit term maps; this sweep is
the zero-shot baseline for later term-map SFT comparisons.

## Verdict

Pending smoke/full sweep completion.

