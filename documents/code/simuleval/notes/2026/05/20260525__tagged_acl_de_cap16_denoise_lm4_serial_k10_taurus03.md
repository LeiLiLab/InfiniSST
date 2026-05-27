## Hypothesis

The cap16-denoise tagged-term de SLM should be checked with the original serial SimulEval path for tagged ACL raw lm=4, because batch-vLLM and serial evaluation may differ in generation/cache behavior.

## Background / Motivation

The user requested a Taurus-only non-batch eval for En-De tagged ACL raw, lm=4, denoise SLM, k=10. A previous two-GPU serial attempt failed before generation because the retriever and vLLM shared one visible GPU. This run exposes three GPUs so vLLM TP=2 can use the first two visible devices and the MaxSim retriever can use the third.

## What changed vs baseline

- Dataset: tagged ACL raw de, five full dev talks.
- SLM: Taurus-local cap16-denoise HF cache.
- Retriever: HN1024, top-k 10, tau 0.78, lookback 1.92s.
- Eval path: serial SimulEval, not batch-vLLM.
- GPU layout: physical Taurus GPUs 0,1,2; vLLM TP=2 plus auto-selected MaxSim retriever on visible `cuda:2`.
- Empty term-map policy: `omit`; strip policy: `term_t`; decode cap: `MAX_NEW_TOKENS=80`.

## Expected metrics

The run should produce exactly one `eval_results.tsv` for de/lm=4 and five rows in both `instances.log` and `instances.strip_term.log`.

## Verdict

Pending.
