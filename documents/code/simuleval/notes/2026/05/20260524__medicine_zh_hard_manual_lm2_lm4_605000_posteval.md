# Medicine zh hard-manual post-eval patch

## Hypothesis

The existing zh `lm=2` generation can be re-scored against the current
hard-manual medicine glossary without rerunning the LLM, and zh `lm=4` can be
reported on five samples by combining the new Aries four-sample run with the
existing Taurus `sample605000` artifact.

## Background / Motivation

The earlier zh `lm=2` metric was produced from the previous glossary setup, not
the hard-manual term list.  The Aries completion run intentionally skipped
`lm=4/sample605000` because that sample had already been generated on Taurus.

## What changed vs baseline

- Re-run StreamLAAL + term miss export for zh `lm=2` with
  `medicine_hard_manual_glossary_streamlaal_20260524.json`.
- Re-run StreamLAAL + term miss export for Taurus zh `lm=4/sample605000`
  using the same hard-manual glossary.
- Build a new zh `lm=4` five-sample directory by concatenating Aries
  `404/545006/596001/606` with Taurus `605000`, then post-evaluate that merged
  artifact.

## Expected metrics

The output should contain hard-manual `TERM_ACC`, StreamLAAL, and term-miss TSVs
for zh `lm=2`, Taurus zh `lm=4/sample605000`, and the combined zh `lm=4`
five-sample artifact.

## Verdict

Succeeded.  The hard-manual post-eval artifacts were written for zh `lm=2`,
Taurus zh `lm=4/sample605000`, and the merged zh `lm=4` five-sample directory.
Use the TSVs in the output directories as the metric source of record.
