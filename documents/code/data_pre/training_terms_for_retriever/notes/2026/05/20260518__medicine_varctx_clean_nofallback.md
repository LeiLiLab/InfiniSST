## Hypothesis

Medicine eval should only use terms that can be located in the English source
audio/text. Unmatched source labels should not become ground-truth positives via
sentence-center fallback.

## Background / Motivation

The ESO medicine source has entries where `terms[].term` is a target-language
glossary label rather than the spoken English surface form. Example:
`Dosimetristen` is attached to an English sentence containing `dosimetrists`.
The previous preprocessing path assigned such unmatched labels to a compact
span at the sentence center, which can produce GT positives that do not appear
in the chunk text or audio.

## What changed vs baseline

`prepare_medicine_variable_context.py` now defaults to
`--unmatched-term-policy drop`. Terms that cannot be located by MFA exact match
or sentence-text char-proportional match are omitted and written to a dropped
terms audit JSON. The legacy behavior remains available with
`--unmatched-term-policy center_fallback`.

## Expected metrics

The cleaned dataset should have no `sentence_center_fallback` rows. Medicine
recall should be recomputed against this cleaned target set before comparing
domain performance.

## Verdict

Data preparation succeeded. The clean dataset has 11,141 rows, 3,094 term rows,
and no `sentence_center_fallback` rows. Term-row locate methods are
`mfa_exact=2393` and `char_proportional=701`. The dropped-term audit contains
204 unmatched source labels, including the `Dosimetristen` / `klinische
Physiker` case from sample 596001 sentence 260.
