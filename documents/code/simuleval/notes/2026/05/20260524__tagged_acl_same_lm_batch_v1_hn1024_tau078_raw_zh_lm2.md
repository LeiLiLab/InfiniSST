# Tagged ACL Same-LM Batch V1 zh lm2 raw

## Hypothesis

Batching the five tagged-ACL talks for a fixed latency multiplier (`lm=2`) in one shared vLLM process should preserve serial SimulEval metrics while improving throughput.

## Background / Motivation

The earlier mixed-lm batch-vLLM prototype improved BLEU/TERM/LAAL but was not serial-equivalent because it mixed `lm=1..4` scheduling and changed generation trajectories.  This event isolates one latency multiplier to keep the same speech chunk cadence, cache policy, token budget, retriever, glossary, and model as the serial zh lm2 raw result.

## What changed vs baseline

- Runs only `zh`, `lm=2`, raw tagged ACL glossary.
- Uses the same New V9 zh Speech LLM and HN1024 retriever at tau `0.78`.
- Uses fixed `max_new_tokens=40`, temperature/top-p/top-k, and cache seconds from the serial-compatible check.
- Enables batched MaxSim timeline retrieval for the five ready streams in each vLLM batch.
- Writes a compare report against the existing serial lm2 output.

## Expected metrics

Final BLEU, TERM_ACC, REAL_TERM_ADOPT, TERM_FCR, and StreamLAAL should be close to the serial lm2 raw result.  Prompt/reference alignment should match except for harmless runtime prompt truncation in logs.

## Verdict

Two validation runs completed.  The first full audio-encoder batch changed retriever scores near tau and is not serial-equivalent.  The exact variant keeps batched Whisper feature extraction but runs the Qwen audio encoder per stream; it matches serial retrieval distribution exactly and gives near-serial metrics for zh lm=2 raw, but generation is not token-identical because batched vLLM sampling changes continuation trajectories.  Use the exact variant for same-lm accelerated eval; do not use full encoder batch for main results.
