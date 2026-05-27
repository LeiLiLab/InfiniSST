## Hypothesis

Batch-vLLM and serial SimulEval may produce different de/lm4 metrics for the
`de_c16_denoise_ttag_r32a32_ep1` Speech LLM. This readout uses serial SimulEval
with the same model, HN1024 retriever, tau, max-new-token budget, and empty-map
policy as the current batch readout.

## Background / Motivation

The batch readout for `de_c16_denoise_ttag_r32a32_ep1` completed for lm=1 and
lm=4 only. The lm=4 BLEU remained below the target, so we need to rule out a
batch/serial evaluator discrepancy before treating the result as model behavior.

## What changed vs baseline

- SLM: `/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_de_cap16_denoise_budget_ttag_r32a32_ep1_taurus6/keep1.0_r32/v0-20260525-203735-hf`
- Dataset: tagged ACL raw de, five full dev talks.
- Retriever: HN1024 checkpoint with tau=0.78, top-k=10, lookback=1.92s.
- Empty term-map policy: `omit`.
- Strip policy: `term_t`, removing both `<term>...</term>` and `<t>...</t>`.
- Decode cap: `MAX_NEW_TOKENS=80`.
- Effective cache target: chunks 8/4, matching the batch readout.

## Expected metrics

If batch/serial behavior matches, lm=4 should be near the batch BLEU
32.6869 and TERM_ACC 0.8513. A large BLEU jump would indicate evaluator/runtime
differences rather than only SLM behavior.

## Verdict

Canceled at user request before rerun. The first two-GPU serial attempt failed
before generation because the serial RAG path placed the retriever and vLLM on
the same visible GPU, leaving only 3.85 GiB free for vLLM startup. No metrics
from this canceled event should be used.
