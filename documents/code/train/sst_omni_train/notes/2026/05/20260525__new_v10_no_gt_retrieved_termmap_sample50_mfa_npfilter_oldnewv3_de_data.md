# New V10 Sample50 No-GT Retrieved Term Map Data: de

## Hypothesis

Full V10 preserved too many retrieved terms on no-GT chunks.  Sampling roughly
half of no-GT retrieved term-map chunks should restore exposure closer to the
TM-SFT training distribution while still avoiding the NewV9 all-empty no-GT
term-map pathology.

## Background / Motivation

The full V10 de build preserved `206,314` no-GT retrieved term-map entries with
a no-GT nonempty rate of `0.8728`.  The aligned TM-SFT source distribution is
much sparser: `79,906` no-GT entries and nonempty rate `0.4916`.

## What changed vs baseline

- Source remains the clean NewV9 `stage2_gtbackfill` JSONL.
- Chunks with GT terms keep the GT-backfilled term map unchanged.
- Chunks without GT terms use deterministic hash sampling at the term level:
  each retrieved `source=target` term-map line is independently kept with
  `keep_prob=0.5`; chunks with no kept terms become `term_map:NONE`.
- Assistant `<term>` target wrapping is then applied with the same NewV9/V10
  settings.

## Expected metrics

This data should be less term-map dense than full V10 and should be easier for
the SLM to handle without BLEU loss.  The first gate remains de/lm2 tagged ACL
raw with HN1024 and tau `0.79`, compared against the verified no-RAG BLEU
baseline `30.0676`.

## Verdict

Built and validated with term-level Bernoulli sampling.

- Rows: `12,500`; chunks: `71,730`.
- no-GT chunks: `28,842`.
- no-GT entries: `206,314 -> 102,842`, keep rate `0.4985`.
- no-GT average entries/chunk: `7.1532 -> 3.5657`.
- TM-SFT reference no-GT average entries/chunk: `2.7705`.
- no-GT nonempty rate: `0.8728 -> 0.8066`.
- TM-SFT reference no-GT nonempty rate: `0.4916`.
- GT term coverage remains `83,193 / 83,193 = 1.0`.
- Assistant tag validation: malformed tags `0`, Latin word-cut tags `0`.

Important caveat: term-level 50% sampling halves the entry count but does not
halve the no-GT nonempty chunk rate.  This is expected because multi-term
chunks usually retain at least one term.
