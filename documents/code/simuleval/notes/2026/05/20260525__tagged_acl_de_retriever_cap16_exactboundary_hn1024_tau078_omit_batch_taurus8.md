# Tagged ACL De Retriever Cap16 Exact-Boundary, HN1024 Tau0.78, Omit Empty Term Map

## Hypothesis

German Speech LLM fine-tuned with retriever-recalled cap16 noisy term maps and exact-boundary assistant `<term>` wrapping may recover BLEU relative to the previous tagged-term variants while preserving terminology gains.

## Background / Motivation

The previous German runs showed that `term_map:\nNONE` on empty retrieval prompts can hurt generation, especially at low latency. This readout uses `empty_term_map_policy=omit` so empty retrieval chunks match the training-time prompt shape more closely.

## What changed vs baseline

- Model: `/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_de_retriever_cap16_exactboundary_r32a32_ep4_taurus8/keep1.0_r32/v1-20260525-141908-hf`
- Parent training event: `20260525T0620__speech_llm_train__de_retriever_cap16_exactboundary_r32a32_ep4_taurus8`
- Dataset: tagged ACL raw German readout.
- Retriever: HN1024, `tau=0.78`, `top_k=10`, lookback 1.92s.
- Batch eval: same-lm five-talk batch, `max_new_tokens=80`, `VLLM_LIMIT_AUDIO=128`, `VLLM_MAX_MODEL_LEN=12288`.
- Empty term-map behavior: `omit`.
- Launch ordering prioritizes `lm=4` first on GPU pair `6,7`; the remaining LMs use `2,3`, `4,5`, and `0,1`.

## Expected metrics

The main gate is whether `lm=4` restores BLEU toward the verified German no-RAG baseline while keeping TERM_ACC clearly above no-RAG. Secondary readout covers `lm=1,2,3` under the same protocol.

## Verdict

Completed with verified per-LM eval artifacts. The detached launcher produced all four `eval_results.tsv` files and each LM has 5 rows in both `instances.log` and `instances.strip_term.log`.

The launcher exited nonzero only in the post-eval summary step because its Python `Path.glob` pattern used `de/**_lm{lm}_*/eval_results.tsv`, which is invalid in Python 3.12 because `**` must be a full path component. The launcher was patched to `de/*_lm{lm}_*/eval_results.tsv`, and the summary was regenerated from the verified eval artifacts.

Results:

| lm | BLEU | StreamLAAL | StreamLAAL_CA | TERM_ACC |
| --- | ---: | ---: | ---: | ---: |
| 1 | 23.6820 | 1045.8916 | 1238.3764 | 0.8225 (769/935) |
| 2 | 29.9921 | 1667.3327 | 1333.1941 | 0.8588 (803/935) |
| 3 | 32.2752 | 2242.5082 | 1025.0184 | 0.8471 (792/935) |
| 4 | 32.5343 | 2778.3129 | 1347.4342 | 0.8749 (818/935) |

Summary TSV: `/mnt/gemini/data1/jiaxuanluo/tagged_acl_de_retriever_cap16_exactboundary_hn1024_tau078_omit_batch_20260525T070739_de_cap16_hn1024_tau078_omit_lm4first_taurus8/de_retriever_cap16_exactboundary_hn1024_tau078_omit_batch_max80/__summary__/summary_de_lm1to4.tsv`.
