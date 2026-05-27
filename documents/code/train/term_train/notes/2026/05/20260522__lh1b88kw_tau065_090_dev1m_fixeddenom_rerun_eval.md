## Hypothesis

The HN1024 `lh1b88kw` checkpoint should retain more recall than no-HN at the
same dev max-recall-drop budgets once the tau grid is aligned to `0.65..0.90`
and the dev retriever bank is expanded to 1M.

## Background / Motivation

The first HN1024 dev-1M attempt (`ywz40glz`) was aborted because running it
concurrently with no-HN made both eval processes grow beyond 200GB RSS.  The
no-HN eval completed, so this rerun executes HN1024 alone to avoid the
concurrent memory peak.

ACL, tagged ACL, and medicine remain disabled.  This is still a dev-only
calibration/readout for the HN vs no-HN comparison.

## What changed vs baseline

- Checkpoint: `lh1b88kw` HN1024 best-secondary checkpoint.
- Tau grid: `0.65..0.90`, stride `0.01`.
- Dev retriever banks: gs10k, gs100k, and gs1M from the P31 untrained 1M
  glossary source.
- Metrics denominator: fixed raw/strict dev positives; retriever glossary size
  changes only the candidate bank.
- Execution: rerun HN1024 alone on one GPU after the no-HN eval finished.

## Expected metrics

The output should allow HN1024 and no-HN to be compared by matched max dev
recall-drop budgets `<0.5pp`, `<1.0pp`, and `<1.5pp`, including raw tau `0.0`
recall and tau-filtered precision/recall for base, gs10k, gs100k, and gs1M.

## Verdict

Finished successfully as W&B run `31xmxmdp`.

The HN1024 dev-1M rerun completed in `1176.88s` on taurus GPU7.  The key dev
unfiltered recalls were base `0.9921`, gs10k `0.9896`, gs100k `0.9858`, and
gs1M `0.9778`.  The run used GPU chunked scoring (`query_chunk=64`,
`text_chunk=4096`) but still materialized full CPU logits; peak observed RSS was
about `256GB`, so 1M should be treated as a heavy stress-test until the eval
path is converted to streaming/top-k-only aggregation.

For raw/base-included max-drop, HN1024 selects tau `0.72`, `0.77`, and `0.82`
under `<0.5pp`, `<1.0pp`, and `<1.5pp`.  For expanded-bank-only max-drop
(gs10k/gs100k/gs1M), it selects tau `0.75`, `0.80`, and `0.83`.  These results
should be compared against no-HN `evcgcdlu`; the report keeps the two selection
surfaces separate because no-HN does not satisfy `<0.5pp` when raw/base is
included.
