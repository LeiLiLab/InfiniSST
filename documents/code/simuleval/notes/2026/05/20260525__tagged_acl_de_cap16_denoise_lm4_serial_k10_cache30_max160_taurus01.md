## Hypothesis

The non-batch En-De tagged ACL raw lm=4 RASST readout for the cap16-denoise
term-tagged Speech LLM should be evaluated with the same cache budget requested
for the current comparison: `max_cache_chunks=30`, `keep_cache_chunks=30`, and
`max_new_tokens=lm*40=160`.

## Background / Motivation

The previous serial launch was canceled before producing metrics because it used
the older `max_new_tokens=80` and `8/4` cache setting. This run keeps the
serial SimulEval path but aligns the cache and generation cap with the requested
protocol. It uses two Taurus GPUs; vLLM uses TP=2 and MaxSim shares visible
`cuda:1` explicitly.

## What changed vs baseline

- Dataset/readout: tagged ACL raw En-De, `lm=4`.
- Method: cap16-denoise term-tagged Speech LLM + HN1024 MaxSim retriever.
- Driver: non-batch serial SimulEval.
- RAG top-k: `10`.
- Tau: `0.78`.
- Empty term map policy: `omit`.
- Cache: `max_cache_chunks=30`, `keep_cache_chunks=30`.
- Generation cap: `max_new_tokens=160`.
- GPUs: Taurus physical `0,1`; vLLM TP=2; retriever shares visible `cuda:1`.

## Expected metrics

This is a readout/check run. The goal is to compare against batch and older
serial settings, especially whether lm=4 BLEU recovers when the serial cache and
token budget match the intended protocol.

## Verdict

Pending.
