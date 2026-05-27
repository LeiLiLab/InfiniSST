## Hypothesis

The En-De `lm=2` BLEU drop can be diagnosed by placing old InfiniSST,
current no-RAG rerun, origin-SLM+HN1024, TM-SFT+HN1024, and clean NewV9
RASST+HN1024 on the same BLEU-vs-term-accuracy plane.

## Background / Motivation

The historical `acl_tagged_raw / InfiniSST / de / lm=2` row was not reproduced
by the current same-lm batch no-RAG rerun. A compact setting-level figure makes
the tradeoff visible without mixing it into the main paper figure yet.

## What changed vs baseline

This is an analysis-only artifact. It reads the canonical main-result TSV for
old reference rows and reads verified event manifests for the new reruns.

## Expected metrics

The generated TSV should expose source provenance for every point and should
mark the old InfiniSST row as an unreproduced historical reference.

## Verdict

Completed. The generated TSV contains six unique setting points:

- Offline ST reference
- Old InfiniSST prompt row
- Current InfiniSST/no-RAG rerun
- Origin de SLM + HN1024
- Old TM-SFT de SLM + HN1024
- Clean NewV9 de SLM + HN1024

The old InfiniSST row is marked as `old_unreproduced_reference` and rendered as
a red X because the current no-RAG same-lm batch rerun did not reproduce it.
The plot shows the expected tradeoff: HN1024 retrieval recovers term accuracy,
TM-SFT adds BLEU over the origin SLM, and clean NewV9 has the highest term
accuracy but the lowest BLEU among the HN1024 points.
