## Hypothesis

Batching independent streaming eval streams through one shared vLLM can improve GPU utilization without changing the existing serial SimulEval path.

## Background / Motivation

The current agent launches one SimulEval process per latency multiplier and uses `max_num_seqs=1`, so lm=1/2/3/4 serial eval repeatedly loads the same Speech LLM and retriever.  For raw-glossary runs with the same language, SLM, retriever, and glossary, the streams are independent and can be decoded concurrently.

## What changed vs baseline

Added a standalone prototype:

- `documents/code/simuleval/src/batched_vllm_rag_eval.py`
- `documents/code/simuleval/launchers/2026/05/20260524__batched_vllm_rag_eval.sh`

It keeps the existing serial launchers untouched.  The new driver loads one vLLM and one MaxSim retriever/index, schedules multiple `(sample, lm)` streams, writes per-lm `instances.log`, and then reuses `offline_streamlaal_eval.py`.

## Expected metrics

Metrics should be close to the serial path for the same inputs, model, retriever, tau, and glossary.  Minor latency differences are expected because this driver is not the original SimulEval event loop; it is intended first as a throughput prototype.

## Verdict

Ready for dry-run/static validation.  Do not replace serial main-result scripts until one small run has been compared against the serial output.
