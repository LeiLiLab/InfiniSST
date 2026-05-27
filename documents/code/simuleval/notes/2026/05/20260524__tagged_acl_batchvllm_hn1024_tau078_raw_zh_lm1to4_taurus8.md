# Tagged ACL zh raw batched-vLLM HN1024 tau0.78

## Hypothesis

The standalone batched-vLLM driver should produce comparable zh tagged ACL raw metrics to the serial SimulEval path while using one shared 8-GPU vLLM instance for all `lm=1,2,3,4` streams.

## Background / Motivation

The serial zh New V9 tagged ACL raw readout has already run each latency multiplier separately.  This event tests whether the same five-talk full-corpus input can be evaluated by batching independent `(talk, lm)` streams through one vLLM process.

## What changed vs baseline

- New non-invasive driver: `documents/code/simuleval/src/batched_vllm_rag_eval.py`.
- Existing serial launchers and `agents/infinisst_omni_vllm_maxsim_rag.py` are not modified.
- Speech LLM: zh New V9 assistant-term-tag-delay HF export.
- Retriever: HN1024 `lh1b88kw`, tau `0.78`, top-k `10`, timeline lookback `1.92s`.
- Runtime and metric glossary: fixed raw tagged ACL `acl6060_tagged_gt_raw_min_norm2.json`.
- Output-side `<term>...</term>` markers are stripped before scoring.

## Expected metrics

The four rows for `zh/lm=1,2,3,4/raw` should be close to the serial reference rows.  Exact StreamLAAL may differ because this prototype is not the original SimulEval event loop.

## Verdict

Completed on Taurus 8 GPUs.  The shared-vLLM run finished all five tagged-ACL talks for `lm=1,2,3,4` in one process and wrote W&B runs `t0vwd0j0`, `3dd8ugqc`, `7cdgkq0j`, `fs9t8i3k`.

The metrics are not serial-equivalent: BLEU is higher for all four lms, TERM_ACC is higher for lm1/lm2/lm4 and slightly lower for lm3, and StreamLAAL is much lower for all four lms.  Treat this as a useful throughput prototype and debugging signal, not as a drop-in replacement for the serial SimulEval result until the batching driver's scheduling and latency accounting are validated.

Comparison report: `documents/code/simuleval/reports/20260524_tagged_acl_batchvllm_hn1024_tau078_raw_zh_lm1to4_compare.md`.
