# Medicine ja lm2 StreamLAAL with hard manual terms

## Hypothesis

The manually checked hard-term glossary should give the strict term readout for
the completed Japanese no-RAG baseline without changing the generated
hypotheses.

## Background / Motivation

The generated `ja/lm=2` no-RAG hypotheses were first scored against the full
restored ESO term set. This analysis rescoring uses the manually checked hard
term glossary:

```text
/home/jiaxingxu/rag-sst/eso-dataset/outputs_hard_terms_llm_judge_with_hypothesis_v2/hard_medicine_glossary.from_outputs_v2_terms.llm_judge_with_hypothesis_manual_check.json
```

The glossary is extracted into a minimal StreamLAAL-compatible term list under
the existing output root, preserving all `215` entries.

## What changed vs baseline

Only the glossary used for StreamLAAL / TERM_ACC changes. The `instances.log`,
references, source text, and audio yaml are unchanged.

## Expected metrics

Write separate hard-term metrics and miss files with the
`hard_llm_manual_check` suffix so the previous full-term results are preserved.

## Verdict

Success.

Extracted glossary stats:

```text
input_entries=215
output_entries=215
unique_term_casefold=212
missing_ja_translation=0
```

Hard-term StreamLAAL / TERM_ACC:

```text
BLEU=23.8731
StreamLAAL=2143.40 ms
StreamLAAL_CA=2923.88 ms
TERM_ACC=0.3604
TERM_CORRECT=262
TERM_TOTAL=727
miss_occurrences=465
unique_missed_term_translations=188
```
