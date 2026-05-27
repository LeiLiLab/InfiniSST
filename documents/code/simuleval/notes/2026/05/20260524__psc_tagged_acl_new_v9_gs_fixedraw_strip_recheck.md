# PSC Tagged ACL New V9 zh gs1k/gs10k Strip Recheck

## Hypothesis

The PSC `gs1k` and `gs10k` generation outputs are usable, but their original
`eval_results.tsv` files scored raw `instances.log` with assistant-side
`<term>` markers still present.  Re-running only offline evaluation with
`--strip-output-tags term` should recover normal zh tagged ACL BLEU while
keeping the fixed raw tagged glossary denominator.

## Background / Motivation

The PSC Slurm jobs completed generation and offline eval outputs, then failed
at the W&B logging stage.  Inspection showed the PSC source tree was stale:
`offline_streamlaal_eval.py` did not yet support `--strip-output-tags`.  A
single `gs1k/lm2` post-hoc recheck raised BLEU from about `31.0` to about
`48.7`, confirming the low BLEU was a scoring artifact.

## What changed vs baseline

- No generation is rerun.
- The same PSC `instances.log` files are rescored into
  `eval_results.strip_term_recheck.tsv`.
- Output-side `<term>` tags are stripped before BLEU, StreamLAAL, and term
  metrics.
- Metric glossary remains the fixed raw tagged ACL glossary
  `acl6060_tagged_gt_raw_min_norm2.json`.
- Runtime glossary labels remain `gs1k` and `gs10k`.

## Expected metrics

Eight strip-rechecked rows should be produced for `zh`:

- `gs1k`: `lm=1,2,3,4`
- `gs10k`: `lm=1,2,3,4`

Each row should include BLEU, StreamLAAL, TERM_ACC, REAL_TERM_ADOPT, and
TERM_FCR.

## Verdict

Completed.  All eight PSC `gs1k`/`gs10k` rows were rescored from the original
generation `instances.log` files with `--strip-output-tags term`; no generation
was rerun.  The combined PSC strip summary is:

`/ocean/projects/cis260009p/jluo7/InfiniSST_psc_eval/outputs/tagged_acl_new_v9_hn1024_tau078_gs_fixedraw/20260524T0520_psc_tagacl_newv9_hn1024_tau078_gs1k_gs10k_fixedraw_zh/__summary__/strip_recheck_summary.tsv`

W&B post-hoc metric runs:

- `simuleval_eval/b6p445cl`: `gs1k`, `lm=1..4`
- `simuleval_eval/lv8d2i9r`: `gs10k`, `lm=1..4`

Merged raw + `gs1k` + `gs10k` report:

- `documents/code/simuleval/reports/20260524_tagged_acl_new_v9_hn1024_tau078_zh_raw_gs_fixedraw_data.tsv`
- `documents/code/simuleval/reports/20260524_tagged_acl_new_v9_hn1024_tau078_zh_raw_gs_fixedraw_summary.md`

Update on 2026-05-24: the `tagged_raw` / `lm=1` row was replaced with the
same-lm batch max256 readout from `simuleval_eval/kolja8vr`: BLEU 44.41,
TERM_ACC 85.39, REAL_ADOPT 89.40, TERM_FCR 13.01, StreamLAAL 1236.72.
