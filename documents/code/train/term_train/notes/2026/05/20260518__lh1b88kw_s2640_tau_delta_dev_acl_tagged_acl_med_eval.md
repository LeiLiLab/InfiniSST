# lh1b88kw Secondary Checkpoint Tau-Delta Eval With Tagged ACL

## Hypothesis

The `lh1b88kw` secondary checkpoint can be read out on the existing
extracted-paper ACL glossary, the newly prepared tagged ACL glossary, and
medicine without changing the dev-only tau selection rule.

## Background / Motivation

The prior tau-delta run `4g108a3w` included dev, extracted-paper ACL6060, and
medicine readouts, but did not include the separate tagged ACL glossary from
`documents/data/data_pre/glossary_acl6060.json`.

## What changed vs baseline

- Baseline eval event: `20260517T1922__retriever_eval__lh1b88kw_s2640_tau_delta_dev_acl_med`
- Baseline W&B run: `4g108a3w`
- Data-prep event for tagged ACL: `20260518T1359__data_prepare__acl6060_tagged_varctx_lmlb2p88_5p76`
- Tagged ACL JSONL: `/mnt/gemini/home/jiaxuanluo/acl6060_dev_offline_eval_tagged_glossary_varctx2p88_3p84_4p80_5p76/acl6060_tagged_dev_dataset.jsonl`
- Tagged ACL glossary: `/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_union_gs10000_min_norm2_backfill.json`
- Dev remains the only source for tau selection. Extracted-paper ACL, tagged ACL, and medicine are held-out readouts only.

## Expected metrics

Expect tagged ACL base / gs1k / gs10k recall to differ from extracted-paper ACL
because the positive term set and no-term rows are derived from tagged glossary
annotations rather than paper-extracted glossary matching.

## Verdict

Completed successfully in W&B run `mry7kesp` from direct aries GPU7 launch
`direct_lh1b88kw_tagacl_gpu5_20260518T161730`.  The tagged ACL readout was
added without changing the dev-only tau selection rule; final metric truth
lives in W&B.
