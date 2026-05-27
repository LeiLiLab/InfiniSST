## Hypothesis

The cap16-denoise tagged-term de SLM should provide a consistent batch-vLLM readout for medicine hardraw de and fill the missing tagged ACL raw lm=2,3 points under the same Taurus-only runtime protocol used for recent cap16-denoise checks.

## Background / Motivation

The current de main-result sweep needs medicine hardraw lm=1,2,3,4 with the cap16-denoise SLM, and the tagged ACL raw cap16-denoise curve has lm=1,4 but needs lm=2,3 filled under matching batch settings.

## What changed vs baseline

- Uses the Taurus-local HF cache for `speech_llm_de_cap16_denoise_budget_ttag_r32a32_ep1_taurus6`.
- Uses HN1024 MaxSim retrieval with tau 0.78 and top-k 10.
- Uses same-lm batch-vLLM with five streams per lm.
- Uses `empty_term_map_policy=omit`, `rag_prompt_policy=given_chunks`, `strip_output_tags=term_t`.
- Uses cache chunks fixed at 30/30 and decode cap `max_new_tokens=20*lm`.
- Runs only on Taurus GPU pairs 4,5 and 6,7 in waves.

## Expected metrics

The medicine run is a new readout and should produce one `eval_results.tsv` per lm with five rows in both `instances.log` and `instances.strip_term.log`.

For tagged ACL raw, lm=2 and lm=3 should be directly comparable to the existing cap16-denoise lm=1,4 Taurus batch results.

## Verdict

Pending. The launcher should be treated as submitted/running until all six lm summaries and the top-level `.success` marker are present.
