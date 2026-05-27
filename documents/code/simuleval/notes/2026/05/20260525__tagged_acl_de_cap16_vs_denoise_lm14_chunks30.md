# Tagged ACL De Cap16 vs Denoise lm1/lm4 Chunks30

## Hypothesis
Fixing the runtime cache to `max_cache_chunks=30` and `keep_cache_chunks=30`, while matching the SFT system prompt and empty term-map policy, should clarify whether the BLEU drop is caused by runtime prompt/cache mismatch rather than the cap16 or denoise SLM itself.

## Background / Motivation
The de cap16 and cap16-denoise SLMs were both trained with the `given chunks of English audio` system prompt and omitted empty `term_map` blocks. Recent checks showed runtime prompt wording had drifted from the training JSONL, and earlier cache experiments mixed seconds-based and chunk-based settings. This readout standardizes the runtime shape before deciding which curve is usable.

## What changed vs baseline
- Dataset: tagged ACL raw En-De, five talks, lm=1 and lm=4.
- Retriever: HN1024, tau=0.78, top-k=10, timeline lookback 1.92s.
- Models:
  - cap16: `/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_de_retriever_cap16_exactboundary_r32a32_ep4_taurus8/keep1.0_r32/v1-20260525-141908-hf`
  - cap16-denoise: `/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_de_cap16_denoise_budget_ttag_r32a32_ep1_taurus6/keep1.0_r32/v0-20260525-203735-hf`
- Runtime prompt: `given_chunks`, matching both training JSONLs.
- Empty term-map policy: `omit`.
- Cache: `max_cache_chunks=30`, `keep_cache_chunks=30`; seconds overrides set to 0.
- Decode cap: `max_new_tokens = 20 * lm`.
- Scoring strips `<term>` for cap16 and both `<term>`/`<t>` for cap16-denoise.

## Expected metrics
This is a diagnostic readout, not a new calibration pass. The immediate comparison is batch vs serial under the same model, lm, cache, prompt, and max-token settings. lm=4 should show whether BLEU can recover toward the no-RAG gate; lm=1 should show whether low-latency degradation is structural or runtime-induced.

## Verdict
Pending. Results should be read from the registered eval artifacts and per-run summary TSVs, not from chat history.
