## Hypothesis

For the strict medicine readout, a source term should become a GT positive only
when it has an exact match in the MFA word intervals.

## Background / Motivation

The no-fallback clean data removes unmatched source labels such as
`Dosimetristen`, but still keeps `char_proportional` terms. Those terms are
present in the source sentence text but not exactly matched in the MFA word
sequence, so their timing is estimated from character offsets. This can still
place a term into a chunk whose MFA text does not contain it.

## What changed vs baseline

Use the same ESO renewed translation path and MFA TextGrid path, but set:

```text
--unmatched-term-policy drop
--allowed-locate-methods mfa_exact
```

The output glossary is built only from retained MFA-exact GT terms plus the
medicine wiki filler/backfill.

## Expected metrics

Term rows will be fewer than the no-fallback data, and recall should be
interpreted as a stricter MFA-exact target-set readout.

## Verdict

Data preparation succeeded. The strict MFA-only dataset has 11,071 rows and
2,408 term rows. All retained term rows are `mfa_exact`; 204 unmatched source
annotations and 241 `char_proportional` annotations were dropped. The resulting
glossary has 570 retained medicine GT terms plus medicine wiki filler/backfill,
and no dropped-only term appears in the gs10000 bank.
