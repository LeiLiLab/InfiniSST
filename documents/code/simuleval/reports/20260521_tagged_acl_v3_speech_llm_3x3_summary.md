# Tagged ACL V3 Speech LLM 3x3 Summary

Run stamp: `20260521T024609`

Output root:
`/mnt/aries/data7/jiaxuanluo/slm/tagged_acl_v3_speech_llm_3x3_20260521T024609`

| setting | variant | W&B | BLEU | TERM_ACC | REAL_ADOPT | TERM_FCR | StreamLAAL |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| zh lm2 raw | real | `0mjv1k4g` | 46.71 | 78.88 | 78.81 | 7.01 | 1738.9 |
| zh lm2 raw | tagged | `d23qtuy1` | 46.85 | 79.66 | 79.67 | 7.79 | 1840.9 |
| zh lm2 raw | adv | `hq0l9nmh` | 46.87 | 79.89 | 79.10 | 7.27 | 1779.5 |
| de lm3 raw | real | `3n4pz80k` | 24.15 | 65.78 | 65.54 | 13.44 | 1597.1 |
| de lm3 raw | tagged | `e1x87ce2` | 27.92 | 68.88 | 70.85 | 11.08 | 1748.1 |
| de lm3 raw | adv | `ft5rzerh` | 24.92 | 66.42 | 69.07 | 11.79 | 1630.0 |
| ja lm1 gs10k | real | `93ibu1d5` | 5.39 | 48.72 | 51.14 | 11.59 | -5.5 |
| ja lm1 gs10k | tagged | `bzkxz3ph` | 13.75 | 62.55 | 66.42 | 16.34 | 1078.2 |
| ja lm1 gs10k | adv | `j7dmvgbr` | 11.14 | 58.09 | 61.69 | 14.59 | 705.7 |

## Readout

- `zh lm2 raw`: all variants are close. `adv` has the highest TERM_ACC and BLEU, but the spread is small.
- `de lm3 raw`: `tagged` is the clear winner among V3 variants. It improves BLEU, TERM_ACC, REAL_ADOPT, and TERM_FCR relative to `real`.
- `ja lm1 gs10k`: `tagged` recovers the most from the noisy 10k setting in BLEU/TERM_ACC/REAL_ADOPT, but it also has the highest TERM_FCR.
- `adv` helps relative to `real` in `de` and `ja`, but is weaker than `tagged` on the two targeted failure settings.

Metric source of truth remains W&B plus the generated `eval_results.tsv` files.
