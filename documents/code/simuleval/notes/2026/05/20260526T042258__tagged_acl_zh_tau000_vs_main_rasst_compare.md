# Tagged ACL zh Tau0.0 vs Main RASST Comparison

## Hypothesis

Tau `0.0` should preserve or slightly improve BLEU by increasing term-map
coverage, but may lower term accuracy by admitting noisy retrieved terms.

## Background / Motivation

The user requested a tagged ACL `zh` tau `0.0` ablation for `lm=1,2,3,4` and a
comparison against the current main-result RASST rows.

## What changed vs baseline

This analysis reads verified tau `0.0` `eval_results.tsv` artifacts from:

- `20260526T0210__simuleval__tagged_acl_new_v9_hn1024_tau000_raw_zh_lm1to4_aries01_seq`
- `20260526T033545__simuleval__tagged_acl_new_v9_hn1024_tau000_raw_zh_lm234_aries4567_parallel`
- `20260526T034148__simuleval__tagged_acl_new_v9_hn1024_tau000_raw_zh_lm4_aries23_parallel`

It compares them to `acl_tagged_raw / RASST / zh` rows in
`documents/code/simuleval/reports/20260524_main_result_data.tsv`.

## Expected metrics

The output report records BLEU, StreamLAAL, StreamLAAL_CA, and TERM_ACC deltas
for `lm=1,2,3,4`.

## Verdict

Completed.  The comparison TSV is
`documents/code/simuleval/reports/20260526_tagged_acl_zh_tau000_vs_main_rasst.tsv`.
