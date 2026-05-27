# En-De Batch vs Serial BLEU Difference Root Cause

Date: 2026-05-25

## Short Answer

The batch evaluator is not a drop-in replacement for serial SimulEval. It is a
throughput simulator with its own prompt/state scheduler, batched stochastic
vLLM calls, and hand-written `instances.log` assembly. BLEU can therefore move
even when source lists, references, and row counts match.

## Verified Evidence

Existing En-De no-RAG lm4 evidence already shows a large driver gap:

| run | protocol | max_new_tokens | BLEU | TERM_ACC | note |
| --- | --- | ---: | ---: | ---: | --- |
| `20260524T160830` | serial SimulEval | 40 | 33.3008 | 0.6909 | verified serial baseline |
| `20260525T132816` | serial SimulEval | 80 | 32.9602 | 0.6930 | serial with batch max-token budget |
| `20260524T2338` | batch | 40 | 30.0243 | 0.6738 | original batch no-RAG |
| `20260525T135847` | batch | 40 | 29.6990 | 0.6695 | batch rerun with serial33 budget |
| `20260525T115605` | batch | 80 | 30.6587 | 0.6492 | serial-compatible no-RAG prompt, still batch scheduling |

So the gap is not explained only by `max_new_tokens`. Even after removing the
most obvious no-RAG prompt mismatch, batch remains far below serial for de/lm4.

## Code-Level Differences

### 1. Prompt equivalence was initially broken for no-RAG

The no-RAG diagnosis found:

- serial: `0/899` LLM input prompts contained `term_map`.
- batch: `899/899` LLM input prompts contained `term_map`.

This explains the first no-RAG discrepancy, but it is not the whole story,
because later batch no-RAG with serial-compatible prompt still stayed low.

### 2. Batch uses multi-stream stochastic decoding

Serial SimulEval calls vLLM one request at a time through the agent action loop.
The batch driver calls:

```python
outputs = llm.generate(prepared, sampling_params=sampling_params, use_tqdm=False)
```

where `prepared` may contain several active streams. Current de batch launchers
typically use `max_num_seqs=5`, `scheduler_batch_size=5`, and round-robin stream
scheduling. The decoding is stochastic (`temperature=0.6`, `top_p=0.95`,
`top_k=20`), so the same seed does not guarantee identical outputs between a
multi-sequence request and serial single-sequence requests.

### 3. Batch bypasses SimulEval output assembly

Serial writes via `WriteAction(content=translation, finished=...)` and lets
SimulEval construct `instances.log`. Batch manually appends generated chunks and
then writes JSONL rows itself. For German, it inserts spaces between chunk
outputs when needed. This is a compatibility shim, not the same code path.

### 4. Audio increment handling is not identical

Serial pads short increments before vLLM:

```python
if len(increment) < 15360:
    increment = np.pad(increment, ...)
```

Batch sends the exact final increment from `last_vllm_samples:cursor_samples`.
This is secondary to scheduling/prompt differences, but it is another
non-equivalence.

### 5. RAG batch retrieval is another separate path

Batch can call `retrieve_timeline_batch`; serial calls the normal timeline
retrieval per stream. They should be logically close, but this is not the same
execution path and should not be assumed identical without a per-call reference
diff.

## Current de_c16_denoise_ttag Status

Verified batch rows for `de_c16_denoise_ttag_r32a32_ep1` currently exist only
for lm=1 and lm=4:

| lm | max_new_tokens | BLEU | StreamLAAL | StreamLAAL_CA | TERM_ACC | correct/total |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 20 | 25.8806 | 1062.7271 | 957.1833 | 0.8000 | 748/935 |
| 4 | 80 | 32.6869 | 2724.8139 | 714.2452 | 0.8513 | 796/935 |

No verified lm=2/3 rows were found for this exact model/setting.

## Practical Conclusion

For paper-facing main results, do not mix serial no-RAG baselines with
round-robin batch RASST rows as if they were protocol-equivalent. Either:

1. use serial for all paper-facing de points, or
2. use a strict batch serial-compat mode for all compared systems:
   `schedule_mode=serial_by_lm`, `max_num_seqs=1`, `scheduler_batch_size=1`,
   prompt-compatible empty-map policy, and preferably greedy first for
   equivalence debugging.

Round-robin batch is still useful for fast exploration, but its BLEU is a
different runtime protocol.
