## Hypothesis

The En-De tagged ACL lm=2 BLEU drop is not caused by the retriever returning a
different term map.  It is caused by the current NewV9 SLM using the same
retrieved term map more aggressively than the older TM-SFT SLM.

## Background / Motivation

The main figure shows that offline ST benefits strongly from GT terminology,
while the current RASST curve improves terminology accuracy but does not recover
BLEU for En-De.  This analysis aligns the DE lm=2 runtime outputs to ACL source
sentences using the MFA-derived `audio.yaml` sentence offsets.

## What changed vs baseline

This is an analysis-only event.  It compares:

- current NewV9 RASST DE lm=2,
- older TM-SFT + HN1024 DE lm=2,
- verified origin no-RAG DE lm=2.

The script re-segments the five full-talk predictions into sentence rows with
`mwerSegmenter`, aligns runtime term maps by source sentence intervals, and
exports sentence-level examples with source, reference, hypotheses, term maps,
term adoption, false-copy markers, and local chrF.

## Expected metrics

No new model metrics are produced.  The expected output is an evidence table and
case report explaining the BLEU/TERM_ACC tradeoff.

## Verdict

The retrieved term maps are identical between NewV9 RASST and TM-SFT + HN1024
for all 1,795 lm=2 chunks.  NewV9 has higher TERM_ACC/REAL_TERM_ADOPT, but it
also emits many `<term>` spans and has a higher term-map false-copy rate.  The
case report shows concrete sentence-level BLEU-risk modes: over-forced terms
from adjacent chunks, non-gold term copies, incomplete sentence realizations,
and malformed leftover tag fragments.
