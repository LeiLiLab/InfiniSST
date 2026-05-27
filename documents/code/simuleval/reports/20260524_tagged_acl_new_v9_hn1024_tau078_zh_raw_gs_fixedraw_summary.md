# Tagged ACL zh New V9 HN1024 Tau0.78 Raw/GS Fixed-Raw Summary

All rows use the fixed raw tagged ACL glossary denominator.  `tagged_gs1k` and `tagged_gs10k` rows are post-hoc strip rechecks from PSC generation outputs; raw rows are verified raw runs.

| runtime glossary | lm | BLEU | StreamLAAL | TERM_ACC | REAL_ADOPT | TERM_FCR | source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| tagged_raw | 1 | 44.4102 | 1236.7205 | 85.39 | 89.40 | 13.01 | Taurus same-lm batch max256 raw |
| tagged_raw | 2 | 49.6099 | 1861.8699 | 88.88 | 90.37 | 12.41 | Aries raw |
| tagged_raw | 3 | 50.1545 | 2342.6068 | 89.89 | 91.08 | 8.07 | Aries raw |
| tagged_raw | 4 | 50.8145 | 2785.5040 | 90.00 | 92.12 | 9.40 | Aries raw |
| tagged_gs1k | 1 | 46.9162 | 1320.2466 | 86.29 | 89.84 | 10.32 | PSC strip recheck |
| tagged_gs1k | 2 | 48.7430 | 1926.7497 | 89.10 | 90.48 | 12.76 | PSC strip recheck |
| tagged_gs1k | 3 | 49.8246 | 2427.2251 | 88.09 | 89.99 | 7.95 | PSC strip recheck |
| tagged_gs1k | 4 | 50.2348 | 2823.8464 | 89.10 | 91.36 | 9.33 | PSC strip recheck |
| tagged_gs10k | 1 | 46.1979 | 1373.0813 | 84.61 | 88.11 | 9.38 | PSC strip recheck |
| tagged_gs10k | 2 | 49.0672 | 1913.4310 | 88.54 | 90.29 | 8.87 | PSC strip recheck |
| tagged_gs10k | 3 | 49.8589 | 2426.7215 | 88.99 | 91.25 | 8.32 | PSC strip recheck |
| tagged_gs10k | 4 | 50.3517 | 2828.4538 | 88.65 | 91.37 | 10.28 | PSC strip recheck |

Generated artifacts:
- TSV: `/home/jiaxuanluo/InfiniSST/documents/code/simuleval/reports/20260524_tagged_acl_new_v9_hn1024_tau078_zh_raw_gs_fixedraw_data.tsv`
- PSC strip summary: `/ocean/projects/cis260009p/jluo7/InfiniSST_psc_eval/outputs/tagged_acl_new_v9_hn1024_tau078_gs_fixedraw/20260524T0520_psc_tagacl_newv9_hn1024_tau078_gs1k_gs10k_fixedraw_zh/__summary__/strip_recheck_summary.tsv`

Caveat: initial PSC `eval_results.tsv` files are known-bad for BLEU because they scored unstripped `<term>` tags. Use `eval_results.strip_term_recheck.tsv` for PSC gs rows.
