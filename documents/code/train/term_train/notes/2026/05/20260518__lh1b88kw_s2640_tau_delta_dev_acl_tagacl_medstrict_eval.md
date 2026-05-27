## Hypothesis

Re-run the lh1b88kw step-2640 tau-delta readout with the cleaned strict
MFA-only medicine dataset, because the prior medicine readout included fallback
and char-proportional positives that should not be used for offline retriever
evaluation.

## Background / Motivation

The current tau=0.0 readout at step 2640 is:

```text
domain           base    1k      10k     100k
dev              0.9920  0.9917  0.9897  0.9861
ACL6060 paper    0.9918  0.9743  0.9591  -
tagged ACL       0.9921  0.9876  0.9800  -
medicine strict  0.9522  0.9489  0.9348  -
```

Medicine strict uses data-prep event
`20260518T1812__data_prepare__medicine_varctx_clean_mfa_exact_only`.

## What changed vs baseline

Use the same checkpoint and dev-only tau selection protocol as the previous
tagged ACL/medicine tau-delta readout, but point medicine JSONL/glossary to:

- `/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/medicine_dev_dataset.jsonl`
- `/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/medicine_glossary_gt_plus_medicine_wiki_gs10000.json`

The tau grid is 0.70 to 0.90 at 0.01 stride. Tau selection uses dev only.
ACL6060 paper, tagged ACL, and medicine strict are held-out readouts.

## Expected metrics

Report tau=0.0 recall plus per-tau filtered recall/precision for dev, ACL6060
paper, tagged ACL, and strict medicine. Select tau using the previously stated
mean dev recall pp drop and max dev recall pp drop criteria.

## Verdict

Pending run.
