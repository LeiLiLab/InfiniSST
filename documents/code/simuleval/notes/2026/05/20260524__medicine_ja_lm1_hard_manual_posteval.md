## Hypothesis

The missing medicine hardraw JA InfiniSST point is caused by a missing
hard-manual StreamLAAL/TERM post-eval TSV, not by a missing generation run.

## Background / Motivation

The JA lm1 no-RAG generation under
`medicine_norag_baseline_abbrev_restored_batched_20260524_ja_lm1_aries01`
finished successfully and wrote five `instances.log` rows, but the canonical
main-results builder only accepts hard-manual post-eval TSVs that also pass the
five-instance check.

## What changed vs baseline

Run only the offline hard-manual StreamLAAL/TERM scoring step for JA lm1 using
the existing generation artifacts, combined medicine inputs, and
`medicine_hard_manual_glossary_streamlaal_20260524.json`.

## Expected metrics

One TSV should be written at the JA lm1 setting directory:
`eval_results_streamlaal_term.hard_llm_manual_check.tsv`.

## Verdict

Completed. JA lm1 now has a five-instance hard-manual post-eval TSV with
BLEU=17.8462, StreamLAAL=1256.0678, StreamLAAL_CA=2057.8372, and
TERM_ACC=0.3150 (229 / 727).
