## Hypothesis

Keeping retriever-filtered term maps on no-GT chunks should reduce the clean
NewV9 SLM's over-sensitivity to term-map presence while preserving the GT-backed
assistant `<term>` salience signal.

## Background / Motivation

The clean de NewV9 data zeroed term maps on all chunks whose
`gt_terms_by_chunk[i]` was empty. The de tagged-ACL readout then showed high
TERM_ACC but weak BLEU. The likely failure mode is that the SLM learned a sharp
distributional contrast between GT chunks with dense term maps and no-GT chunks
with `term_map:NONE`.

## What changed vs baseline

This repair branches from the existing de NewV9 `stage2_gtbackfill` artifact.
It skips the `zero_no_gt_termmap_chunks.py` step and runs assistant target
wrapping directly on the GT-backfilled retriever term maps:

- GT chunks keep their GT-backfilled term maps.
- No-GT chunks keep the translated retriever term maps already present in
  `stage2_gtbackfill`.
- Assistant `<term>` wrapping is still driven only by `gt_terms_by_chunk`, not
  by no-GT retrieved filler terms.

## Expected metrics

Train one de SLM with the same NewV9 training configuration. First eval gate is
tagged ACL raw `de/lm=2`, HN1024, `tau=0.79`, `max_new_tokens=80`. The target is
BLEU at or above the verified no-RAG baseline while keeping TERM_ACC clearly
above no-RAG.

## Verdict

Completed successfully.

- Rows: 12,500 train / 355 dev.
- Chunks: 71,730 total; 42,888 GT chunks; 28,842 no-GT chunks.
- No-GT retrieved term-map entries preserved: 206,314.
- No-GT chunks with nonempty term maps: 25,172 / 28,842 (0.8728).
- GT terms in term map: 83,193 / 83,193 (1.0000).
- Assistant tag validation: 0 malformed tags, 0 Latin word-cut tags.

Spot-check samples confirm no-GT chunks now contain retrieved term-map entries
while assistant `<term>` wrapping remains driven by `gt_terms_by_chunk`.
