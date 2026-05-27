## Hypothesis

The German exact GT-term-wrapped TM-SFT SLM should improve tagged-ACL raw terminology when used as the RASST generator with the HN1024 retriever.  This run reads out all latency multipliers with one same-lm batch process per multiplier.

## Background / Motivation

The model comes from `20260525T0055__speech_llm_train__tmsft_gttermwrap_exact_de_r32a32_ep4_aries8` / W&B `sf6nw09x`.  The previous Taurus lm=2 waiter was superseded by this Aries all-lm readout so that `lm=1,2,3,4` can run concurrently on NVLink 2-GPU pairs.

## What changed vs baseline

- Model: `/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_tmsft_gttermwrap_exact_de_r32a32_ep4_aries8/keep1.0_r32/v0-20260525-083705-hf`
- Runtime glossary and scoring denominator: `acl6060_tagged_gt_raw_min_norm2`
- Retriever: HN1024 MaxSim checkpoint
- RAG threshold: `tau=0.78`
- Batch settings: `max_new_tokens=80`, `VLLM_LIMIT_AUDIO=128`, `VLLM_MAX_MODEL_LEN=12288`, `max_num_seqs=5`, five talks per lm.
- Aries GPU pairs: `0,1`, `2,3`, `4,5`, `6,7`.

## Expected metrics

Each lm should emit exactly one `eval_results.tsv`, plus `instances.log` and `instances.strip_term.log` with five rows.  The launcher prints a `RESULT` line as soon as each lm finishes and writes a merged summary TSV under `__summary__`.

## Verdict

Completed and verified from per-lm artifacts.  Each `lm=1,2,3,4` produced one
`eval_results.tsv`; `instances.log` and `instances.strip_term.log` each contain
five rows per lm with `TERM_TOTAL=935`.

Metrics:

| lm | BLEU | StreamLAAL | StreamLAAL_CA | TERM_ACC | TERM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 20.6877 | 823.8605 | 1728.1666 | 0.7872 | 736 / 935 |
| 2 | 28.6427 | 1569.8803 | 1732.9080 | 0.8642 | 808 / 935 |
| 3 | 31.2239 | 2105.1139 | 1739.2763 | 0.8663 | 810 / 935 |
| 4 | 32.2681 | 2539.9186 | 1599.2362 | 0.8727 | 816 / 935 |

Summary TSV:
`/mnt/gemini/data1/jiaxuanluo/tagged_acl_tmsft_gttermwrap_exact_de_hn1024_tau078_batch_20260525T0413_tagacl_tmsftgtwrap_de_hn1024_tau078_lm1to4_aries/tmsft_gttermwrap_exact_de_r32a32_ep4_hn1024_tau078_batch_max80/__summary__/summary_de_lm1to4_manual.tsv`

Protocol caveat: current `batched_vllm_rag_eval.py` emits explicit
`term_map:\nNONE` when retriever references are empty after thresholding or
filtering.  It does not omit the `term_map` block.

The delegate launcher reports a shell parse error after writing each per-lm
`eval_results.tsv`; this did not affect the verified TSV/log artifacts, but the
summary was merged manually from those artifacts and recorded in the manifest.
