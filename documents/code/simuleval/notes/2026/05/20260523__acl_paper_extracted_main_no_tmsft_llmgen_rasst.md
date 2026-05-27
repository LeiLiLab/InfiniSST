# ACL Paper-Extracted Main Result: No-TM-SFT vs LLM-Generated RASST

## Hypothesis

Paper-extracted ACL terminology readouts should preserve the main tagged-ACL
comparison shape for the two speech-LLM lines that matter here: no-TM-SFT SLM
and LLM-generated term-map RASST.  Runtime glossary size may affect retrieval,
but terminology metrics must be scored against each paper's strict extracted
raw glossary.

## Background / Motivation

The current main-result table under
`documents/code/simuleval/reports/20260521_tagged_acl_main_results_fourline_llmgen_rasst_data.tsv`
contains tagged ACL rows only.  This event prepares the corresponding
paper-extracted ACL grid for:

- no-TM-SFT SLM: `gigaspeech-{lang}-s_origin-bsz4`
- LLM-generated term-map RASST: `gigaspeech-{lang}-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4`

Grid: 5 ACL papers x 3 glossary sizes x 3 languages x 4 latency multipliers,
for both model lines.

## What changed vs baseline

- New portable launcher supports Taurus and PSC path overrides.
- Each single-paper eval uses the runtime glossary requested by `raw`, `gs1k`,
  or `gs10k`.
- Each single-paper eval sets `EVAL_GLOSSARY_PATH_OVERRIDE` to the same paper's
  extracted raw glossary, so TERM_ACC / adoption / FCR denominators do not
  follow runtime glossary size.
- Results are aggregated across 5 papers with micro term counts and macro BLEU /
  StreamLAAL.

## Expected metrics

Track BLEU, StreamLAAL, TERM_ACC, REAL_TERM_ADOPT, and TERM_FCR for each
method/language/latency/glossary-size setting.  Use the raw extracted glossary
as the fixed denominator for all three runtime glossary sizes.

## Verdict

Planned.  Submit only after the PSC quick smoke eval succeeds.
