# Medicine No-RAG vLLM Override Probe

## Hypothesis

PSC-oriented vLLM memory overrides may reduce OOM risk on V100, but they can
change streaming context and therefore quality.  A long `zh lm=4` medicine
sample should expose whether the change is harmless enough before using it for
baseline collection.

## Background / Motivation

The PSC launcher currently uses:

- `VLLM_TP_SIZE_OVERRIDE=2`
- `VLLM_MAX_MODEL_LEN_OVERRIDE=8192`
- `VLLM_LIMIT_AUDIO_OVERRIDE=8`
- `VLLM_DISABLE_CUSTOM_ALL_REDUCE=1`
- `GPU_MEMORY_UTILIZATION_OVERRIDE=0.80`
- `MAX_CACHE_SECONDS_OVERRIDE=4.0`
- `KEEP_CACHE_SECONDS_OVERRIDE=4.0`

The original medicine no-RAG defaults are:

- `VLLM_TP_SIZE_OVERRIDE=2`
- `VLLM_MAX_MODEL_LEN_OVERRIDE=32768`
- `VLLM_LIMIT_AUDIO_OVERRIDE` unset, so the agent uses `max_cache_chunks`
- `VLLM_DISABLE_CUSTOM_ALL_REDUCE=0`
- `GPU_MEMORY_UTILIZATION_OVERRIDE=0.80`
- `MAX_CACHE_SECONDS_OVERRIDE=80.0`
- `KEEP_CACHE_SECONDS_OVERRIDE=60.0`

## What changed vs baseline

Run a Taurus probe on `sample=605000`, `lang=zh`, `lm=4`:

1. `orig80`: original cache/model-length behavior.
2. `psc4s`: PSC memory-reduced behavior.
3. `psc_limit8_keep80`: keep PSC-oriented `max_model_len=8192`,
   `limit_audio=8`, and `disable_custom_all_reduce=1`, but restore
   `MAX_CACHE_SECONDS=80.0` / `KEEP_CACHE_SECONDS=60.0`.

Both use the same origin no-RAG model, same source sample, same glossary builder,
same two GPUs, same generation seed, and same post-eval.

## Expected metrics

The output should include per-variant StreamLAAL/TERM_ACC TSVs and a
sentence-level diff TSV comparing the resegmented predictions.  The key checks
are:

- whether `psc4s` completes without OOM,
- whether `TERM_ACC` materially changes,
- how many sentence predictions change,
- which changed sentences alter term exact-hit counts.

## Verdict

Completed on Taurus for `sample=605000`, `lang=zh`, `lm=4`.

`psc4s` is too aggressive as a quality-preserving default:

| variant | wall min | BLEU | StreamLAAL | TERM_ACC | TERM_CORRECT | TERM_FCR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `orig80` | 36.400 | 42.201 | 2318.013 | 0.6567 | 551/839 | 0.1646 |
| `psc4s` | 10.333 | 41.583 | 2475.030 | 0.6031 | 506/839 | 0.1091 |
| `psc_limit8_keep80` | 10.500 | 43.233 | 2411.149 | 0.6400 | 537/839 | 0.1399 |

Sentence-level diffs vs `orig80`:

| right variant | changed sentences | changed sentence rate | changed term-hit sentences | term-correct delta |
| --- | ---: | ---: | ---: | ---: |
| `psc4s` | 455/486 | 0.9362 | 102 | -45 |
| `psc_limit8_keep80` | 382/486 | 0.7860 | 65 | -14 |

Conclusion: do not use `MAX_CACHE_SECONDS=4.0` / `KEEP_CACHE_SECONDS=4.0`
for the medicine no-RAG baseline.  Keep the PSC-oriented vLLM limits
(`max_model_len=8192`, `limit_audio=8`, `disable_custom_all_reduce=1`) but
restore the original streaming cache defaults (`80.0` / `60.0`) unless a V100
smoke proves a stricter cache is required.  Taurus A6000 success does not prove
V100-32 fit; the PSC launch still needs a V100 smoke because the engine used
about 38 GB per active A6000 at `gpu_memory_utilization=0.80`.
