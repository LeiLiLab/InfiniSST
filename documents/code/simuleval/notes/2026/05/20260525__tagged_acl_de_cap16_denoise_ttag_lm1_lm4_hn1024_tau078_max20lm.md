# Tagged ACL De Eval: Cap16 Denoise-Budget Short-Tag SLM

## Hypothesis
The denoising-budget short-tag SLM should recover BLEU relative to the previous cap16 model while preserving terminology gains from HN1024 retrieval.  The first gate is lm=1 and lm=4 because they expose the low-latency and high-latency failure modes most clearly.

## Background / Motivation
The model was trained from `20260525T1236__speech_llm_train__de_cap16_denoise_budget_ttag_r32a32_ep1_taurus6` on short `<t>...</t>` supervision.  This readout waits for the HF export to finish, then evaluates the tagged ACL raw de benchmark with the same HN1024 retriever and short-cache runtime setting used in the strongest cap16 readout.

## What changed vs baseline
- SLM: `/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_de_cap16_denoise_budget_ttag_r32a32_ep1_taurus6/keep1.0_r32/v0-20260525-203735-hf`.
- Runtime retriever: HN1024, tau=0.78, top-k=10.
- Empty runtime term maps are omitted.
- Runtime cache uses 40/20 seconds and 8/4 chunks.
- Decode cap is `max_new_tokens = 20 * lm`, so lm=1 uses 20 and lm=4 uses 80.
- Offline scoring strips both legacy `<term>` and short `<t>` markers with `--strip-output-tags term_t`.

## Expected metrics
The immediate acceptance signal is lm=4 BLEU above the verified no-RAG gate while retaining high TERM_ACC.  lm=1 should improve over the prior low-BLEU cap16 readout without collapsing TERM_ACC.

## Verdict
Pending.
