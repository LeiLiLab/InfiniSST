## Hypothesis

Run a serial SimulEval readout for En-De medicine hardraw at `lm=1` to compare against the batch result under the same cache and prompt policy.

## Background / Motivation

The current medicine hardraw main-result row for En-De RASST uses cap16-denoise batch-vLLM. The user requested a serial run with `max_cache_chunks=30`, `keep_cache_chunks=30`, `empty_term_map_policy=omit`, and `max_new_tokens=40` on Taurus.

## What changed vs baseline

- Driver: serial SimulEval via `eval_density_unified.sh`, not same-LM batch-vLLM.
- Dataset: medicine hardraw, five selected samples.
- Language: `de`.
- Latency multiplier: `lm=1`.
- Decode/cache policy: `max_new_tokens=40`, `max_cache_chunks=30`, `keep_cache_chunks=30`.
- Prompt policy: omit empty term maps, tagged-term output stripping for scoring.

## Expected metrics

This is a diagnostic comparison against the batch medicine de lm1 row. The exact metric target is not assumed before the serial run completes.

## Verdict

Submitted on Taurus. Await `eval_results.tsv`, `instances.log`, and `instances.strip_term.log` validation.
