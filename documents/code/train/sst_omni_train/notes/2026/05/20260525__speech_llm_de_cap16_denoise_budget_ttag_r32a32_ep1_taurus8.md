# Speech LLM Training: de Cap16 Denoise-Budget Short-Tag r32/a32

## Hypothesis
The de cap16 denoising-budget data should teach the SLM to ignore unsupported runtime terms while still adopting supported terms.  Replacing `<term>...</term>` with `<t>...</t>` should reduce generation overhead at low latency without changing the semantic supervision.

## Background / Motivation
Prior de tagged-ACL runs showed that retriever recall was high but sentence-aligned noise was also high, so BLEU did not consistently improve despite higher TERM_ACC.  The next SLM should be trained on a distribution closer to inference: bounded term maps, no-GT chunks with occasional omitted maps, score/noise dropout, and assistant supervision only for GT target translations that actually appear in the future assistant span.

## What changed vs baseline
- Parent data event: `20260525T1225__data_prepare__de_cap16_denoise_budget_ttag`.
- Training JSONL: `/mnt/gemini/data1/jiaxuanluo/speech_llm_de_cap16_denoise_budget_20260525/de/hn1024_tau078_cap16_denoise_budget_ttag_v1/train_s_de_retriever_hn1024_tau078_cap16_denoise_budget_ttag_exactboundary.jsonl`.
- Dev JSONL: `/mnt/gemini/data1/jiaxuanluo/speech_llm_de_cap16_denoise_budget_20260525/de/hn1024_tau078_cap16_denoise_budget_ttag_v1/dev_s_de_retriever_hn1024_tau078_cap16_denoise_budget_ttag_exactboundary_first355.jsonl`.
- LoRA stays r32/a32; max length stays 3072; EP stays 4; global batch size stays 8.
- Eval for this model must strip both legacy and short markers with `--strip-output-tags term_t`.

## Expected metrics
First gate is tagged ACL raw de, HN1024, dev-calibrated tau, short-cache/omit runtime setting.  The desired behavior is BLEU not below verified InfiniSST/no-RAG at lm=4 and improved lm=1 stability, while TERM_ACC stays clearly above no-RAG.

## Verdict
Pending.  Training is submitted through a Taurus 8-GPU idle watcher because GPU 0/6/7 were occupied at submission time.
