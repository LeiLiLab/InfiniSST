# Context-Length Devraw Fixed-Denominator 100k Eval

## Hypothesis

The context-length ablation should be compared under one dev-only metric
protocol: fixed raw dev denominator, shared Wiki 100k prefix bank, and no
held-out ACL or medicine readout.

## Background / Motivation

Earlier context-length readouts mixed denominator choices because they did not
always pass an explicit `EVAL_METRICS_GLOSSARY`.  The no-HN/HN-256/HN-1024
fixed-denominator readouts used the full dev raw glossary file as the metrics
universe.  This eval reuses that protocol for fixed 1.92s, fixed 3.84s, fixed
5.76s, and the variable-context main checkpoint.

## What changed vs baseline

- Dev metrics glossary is fixed to
  `documents/code/train/term_train/reports/figures/20260524_dev_raw_glossary_from_term_dev_varctx576.json`.
- Runtime retrieval bank uses the shared p31 dev Wiki prefix file
  `wiki_p31_untrained_rank1000000_sample100000.json`.
- Eval sizes are `1000 10000 100000`; raw/base is also logged.
- ACL, tagged ACL, and medicine evals are disabled.

## Expected metrics

The reported dev recall should be lower than any eval that silently uses a
smaller base bank as the metric universe.  The 100k value is the key check for
whether fixed-context and variable-context rows are directly comparable.

## Verdict

Success for the per-context fixed-raw protocol.  The valid W&B runs are:
`9mff3bc4` (1.92s), `k0odyh1h` (3.84s), `d988vg46` (5.76s), and `q2fus6f1`
(variable context).  Summary tables are in
`documents/code/train/term_train/reports/20260525_context_ablation_devraw_fixeddenom_100k.md`
and
`documents/code/train/term_train/reports/figures/20260525_context_ablation_devraw_fixeddenom_100k.tsv`.

The earlier diagnostic run `s8o0es7g` is invalid for the final table: it used
the variable-context 974-term metrics glossary on the old 1.92s dev file, which
left many rows without positives and made recall artificially low.
