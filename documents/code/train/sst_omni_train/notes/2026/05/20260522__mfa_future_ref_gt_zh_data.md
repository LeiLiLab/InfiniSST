## Hypothesis

MFA-grounded source term matching plus future-reference exact target filtering
should produce a clean `gt_terms_by_chunk` source for later Speech LLM term-map
SFT experiments.

## Background / Motivation

The previous `src_trajectory` source-match proxy did not use GigaSpeech MFA
timestamps and is deprecated.  For V13 we need GT labels tied to actual audio
chunk timelines.

## What changed vs baseline

- Source-side GT uses GigaSpeech MFA TextGrid word intervals.
- A glossary term is matched as a normalized word n-gram.
- A matched term is assigned to a chunk when its MFA span overlaps that audio
  chunk timeline.
- Target-side GT requires exact substring match of the target translation in
  assistant messages from the current audio response through the end of the
  conversation.
- Existing user term maps are stripped to `term_map:NONE`; this dataset only
  validates GT labels, not retriever term-map construction.
- Rows missing MFA alignment are explicitly dropped and counted.

## Expected metrics

- Zero future-reference validation violations.
- Zero MFA-overlap validation violations.
- `term_map:NONE` in every audio user chunk.
- Non-empty train output; dev may be empty if the existing dev split lacks MFA
  index coverage.

## Verdict

FAILED / DO NOT USE. Train output was written, but validation failed on an
edge-boundary MFA overlap case.  The user redirected the workflow to build
`source_chunk_asr_by_chunk` first and generate `gt_terms_by_chunk` from that
source chunk text plus the 100k retriever glossary.
