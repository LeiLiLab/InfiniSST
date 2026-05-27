## Hypothesis

Fixed 5.76s train and eval speech contexts are needed as the control for the variable-context `lh1b88kw` family. This isolates the benefit of variable context assignment from the recall gain that may come simply from always using the longest context.

## Background / Motivation

The current variable run mixes 2.88s, 3.84s, 4.80s, and 5.76s contexts. Existing fixed-context readouts cover 1.92s and 3.84s, but there is no fixed 5.76s data path for a direct comparison against the variable run.

## What changed vs baseline

- Rebuild GigaSpeech train rows from the deduplicated MFA source with only `DURATION_SECS=5.76`.
- Drop train rows whose 5.76s context cannot be rebuilt instead of writing legacy 1.92s fallback rows.
- Rebuild dev speech rows from latency multiplier `m=6`, equivalent to 5.76s.
- Rebuild tagged ACL6060 context rows only; paper-extracted ACL is intentionally not part of this data bundle.
- Rebuild medicine context rows with explicit `UNMATCHED_TERM_POLICY=drop` and recorded drop counters.

## Expected metrics

This data-prep event should produce non-empty train/dev/tagged-ACL/medicine JSONLs whose diagnostics show only the `5p76` duration bucket and no missing required outputs. Any filtered train rows must have explicit counters in the stats JSON. Metric effects belong to the downstream retriever training run, not this data-prep note.

## Verdict

PENDING: update after the wrapper finishes and the stats/diagnostic JSON files have been checked.
