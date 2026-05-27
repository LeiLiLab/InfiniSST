## Hypothesis

The older strong En-Zh Speech LLM trained from the source-final TCM term-map data may recover the stronger Zh BLEU/term-adoption behavior when paired with the current HN1024 retriever at the fixed deployment threshold tau=0.78.

## Background / Motivation

Recent NewV9/NewV10 probes made it unclear whether the low BLEU came from the retriever, the batch evaluator, or the newer Speech LLM data recipe. This readout keeps the ACL tagged raw denominator and HN1024 retriever fixed, but swaps in the older strong Zh Speech LLM export.

## What changed vs baseline

- Speech LLM is `/mnt/gemini/data2/jiaxuanluo/speech_llm_tcm_filtered_wiki100kgt_tau075_v2_sourcefinal_gtzh/keep1.0_r32/v0-20260507-103419-hf`.
- Retriever is the HN1024 MaxSim checkpoint used in the current tagged-ACL launcher.
- Evaluation is En-Zh only, `lm=2`, ACL tagged raw glossary, `tau=0.78`, fixed `max_new_tokens=80`.
- Batch shape is same-lm batch eval with five talks and TP=2.

## Expected metrics

BLEU should be closer to the older strong Zh readout than to the degraded NewV9/NewV10 probes, while TERM_ACC should remain above the no-RAG InfiniSST baseline.

## Verdict

Success. The two-GPU Taurus run completed with TP=2 on physical GPUs 5,6 and the HN1024 retriever sharing visible `cuda:0` with one TP shard. Verified artifacts:

- `eval_results.tsv`: `/mnt/gemini/data1/jiaxuanluo/tagged_acl_oldzh_v2_lm2_hn1024_tau078_batch_20260525T0430_tagacl_oldzh_v2_lm2_hn1024_tau078_batch_taurus56_v1mp0/oldzh_v2_sourcefinal_gtzh_hn1024_tau078_batch_max80/zh/dtagacl_oldzh_v2_hn1024_tau078_m80_lm2_k10_th0.78_gacl6060_tagged_gt_raw_min_norm2/eval_results.tsv`
- `instances.log`: 5 rows.
- `instances.strip_term.log`: 5 rows.
- W&B eval run: `simuleval_eval/v1lguaw6`.

Readout: BLEU 47.7021, StreamLAAL 1711.9045, StreamLAAL_CA 1214.7829, TERM_ACC 0.8719 (776 / 890), TERM_FCR 0.0922. This supports the hypothesis that the older strong Zh Speech LLM remains a viable high-BLEU RASST path under the current HN1024/tau=0.78 batch evaluator.
