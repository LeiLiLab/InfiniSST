## Hypothesis

The HN1024 `lh1b88kw` checkpoint should retain higher recall at stricter tau
than no-HN when both models are compared at the same dev max-recall-drop
budget, especially with a 1M retriever bank.

## Background / Motivation

The existing main result uses `lh1b88kw` at tau 0.73, while the no-HN ablation
needs a fairer comparison at matched dev recall-drop budgets.  This run expands
the tau sweep to `0.65..0.90` and adds the dev gs1M candidate bank to check
whether HN still gives a cleaner recall/precision tradeoff under the same fixed
denominator protocol.

ACL, tagged ACL, and medicine are disabled in this run.  They remain held-out
readouts and are not used to choose tau.

## What changed vs baseline

- Checkpoint: `lh1b88kw` HN1024 best-secondary checkpoint used by the current
  main retriever result.
- Tau grid: `0.65..0.90`, stride `0.01`.
- Dev retriever banks: gs10k, gs100k, and gs1M from the P31 untrained 1M
  glossary source.
- Metrics denominator: fixed raw/strict dev positives; retriever glossary size
  changes only the candidate bank.
- Held-out readouts: disabled.

## Expected metrics

The main readout is the HN1024 tau corresponding to no-HN-like max dev recall
drop budgets `<0.5pp`, `<1.0pp`, and `<1.5pp`, plus raw tau `0.0` dev recall
for base, gs10k, gs100k, and gs1M.

## Verdict

Aborted as W&B run `ywz40glz` before completion.  The concurrent no-HN and
HN1024 dev-1M evals each grew past 200GB RSS because the current eval path
materializes full CPU logits for the 1M bank.  To avoid a node-level OOM, the
HN1024 run was stopped and no metrics from this run should be used.

The HN1024 dev-1M comparison should be rerun after the no-HN run finishes, or
after adding a streaming/top-k-only eval path that avoids retaining the full
query-by-glossary logits tensor.
