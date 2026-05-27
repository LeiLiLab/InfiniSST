# PSC Tagged ACL New V9 HN1024 Tau0.78 zh gs1k/gs10k Fixed-Raw

## Hypothesis

New V9 termtag-delay speech LLM with the HN1024 retriever at tau=0.78 should
produce the zh tagged ACL RASST main-result rows for runtime glossary sizes
gs1k and gs10k, while keeping term metrics comparable by using the fixed raw
tagged ACL glossary as the denominator.

## Background / Motivation

The raw tagged ACL main-result row is being handled separately on Aries.  This
PSC run is only for the larger runtime glossary conditions needed for the zh
main table.  The PSC-side New V9 HF export and HN1024 checkpoint are already in
the project storage area.

## What changed vs baseline

- Runtime glossary kinds are `gs1k` and `gs10k`; `raw` is intentionally not run.
- Language is `zh`.
- LM settings are `1 2 3 4`.
- Speech LLM is New V9 termtag-delay.
- Retriever is HN1024 `lh1b88kw`, tau `0.78`, top-k `10`, timeline lookback
  `1.92s`.
- Metric glossary denominator is fixed to
  `acl6060_tagged_gt_raw_min_norm2.json` via `EVAL_GLOSSARY_FOLLOWS_KIND=0`.
- Jobs are submitted directly as full jobs without a smoke gate, per the
  updated execution target.

## Expected metrics

Expected output is eight completed zh rows:

- `gs1k`: `lm=1,2,3,4`
- `gs10k`: `lm=1,2,3,4`

Each row should include BLEU, StreamLAAL, TERM_ACC, REAL_TERM_ADOPT, and
TERM_FCR using the fixed raw tagged denominator.

## Verdict

Generation completed for all eight PSC jobs, but the original Slurm jobs exited
`FAILED` after eval because W&B logging was misconfigured in the initial PSC
wrapper.  A second issue made the first `eval_results.tsv` BLEU values invalid:
the PSC source tree was stale and scored raw `instances.log` files with
assistant-side `<term>` tags still present.

Post-hoc strip recheck is complete.  Use
`eval_results.strip_term_recheck.tsv` for the PSC `gs1k`/`gs10k` rows, not the
original `eval_results.tsv`.

Summary artifacts:

- PSC strip summary: `/ocean/projects/cis260009p/jluo7/InfiniSST_psc_eval/outputs/tagged_acl_new_v9_hn1024_tau078_gs_fixedraw/20260524T0520_psc_tagacl_newv9_hn1024_tau078_gs1k_gs10k_fixedraw_zh/__summary__/strip_recheck_summary.tsv`
- Merged report TSV: `documents/code/simuleval/reports/20260524_tagged_acl_new_v9_hn1024_tau078_zh_raw_gs_fixedraw_data.tsv`
- Merged report MD: `documents/code/simuleval/reports/20260524_tagged_acl_new_v9_hn1024_tau078_zh_raw_gs_fixedraw_summary.md`
- W&B post-hoc runs: `simuleval_eval/b6p445cl` for `gs1k`, and
  `simuleval_eval/lv8d2i9r` for `gs10k`.
