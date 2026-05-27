## Hypothesis

Running medicine hardraw En-De lm=3 and lm=4 on newly freed Taurus GPU pairs 0,1 and 2,3 should complete the high-latency medicine readout faster without interfering with the currently running lm=1,2 wave on GPU pairs 4,5 and 6,7.

## Background / Motivation

The earlier combined Taurus job started medicine hardraw lm=1,2 on GPU pairs 4,5 and 6,7. GPUs 0-3 later became idle. This launcher runs only lm=3,4 into an independent output root to avoid concurrent writes to the combined job output directory.

## What changed vs baseline

- Uses the same cap16-denoise tagged-term de SLM and HN1024 retriever as the combined job.
- Uses the same prepared medicine five-sample input lists and fixed hardraw glossary.
- Uses same-lm batch-vLLM, `empty_term_map_policy=omit`, `rag_prompt_policy=given_chunks`, `strip_output_tags=term_t`, cache chunks 30/30, and `max_new_tokens=20*lm`.
- Uses Taurus GPU pairs 0,1 and 2,3 only.

## Expected metrics

Each lm should produce `eval_results.tsv`, `instances.log`, and `instances.strip_term.log` with five rows.

## Verdict

Pending. This is an acceleration run; the previous combined parent orchestrator was paused to prevent duplicate lm=3,4 launches and shared-memory cleanup conflicts.
