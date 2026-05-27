# Medicine gs10k PR Extra Tau Sweep for New V5 no-GT-zero old-new_v3 R32

## Hypothesis

The previous one-talk medicine gs10k PR sweep suggests HN1024 tau `0.78` is the
best operating point among the evaluated HN1024 thresholds.  We add HN1024 tau
`0.80` and HN256 tau `0.70` / `0.76` to check whether the intermediate HN1024
threshold or the newer HN256 checkpoint gives a better downstream Speech LLM
tradeoff.

## Background / Motivation

This is an incremental extension of
`20260523__medicine_gs10k_pr_sweep_new_v5_no_gt_zero_oldnewv3_r32.md`.

The Speech LLM is fixed to
`speech-llm-new_v5-no-gt-zero-oldnewv3-r32a64-tp2-aries2_keep1.0_r32`.
The eval remains medicine `sample_404`, language `zh`, `lm=2`, runtime
medicine gs10k glossary, and fixed per-talk strict/raw denominator.

Retriever operating points use fixed-denominator readouts from the no-HN vs HN
report and W&B:

| setting | tau | medicine gs10k P/R source |
|---|---:|---|
| HN1024 | 0.80 | `ry8osg4u`: 20.00 / 86.88 |
| HN256 | 0.70 | `8h9q0v4t`: 11.22 / 93.81 |
| HN256 | 0.76 | `8h9q0v4t`: 15.56 / 90.61 |

## What changed vs baseline

- Adds one HN1024 intermediate threshold, tau `0.80`.
- Adds HN256 latest checkpoint at tau `0.70` and `0.76`.
- Keeps Speech LLM, medicine talk, language, lm, runtime glossary, metric
  denominator, top-k, and timeline lookback unchanged.

## Expected metrics

Report BLEU, TERM_ACC, REAL_ADOPT, TERM_FCR, SOURCE_TERM_SENT_FCR, and
StreamLAAL for the three added settings.  Compare primarily against the earlier
HN1024 tau `0.78` and tau `0.82` rows.

## Verdict

Completed on medicine sample 404 (`zh`, `lm=2`, runtime gs10k glossary, fixed
strict/raw denominator).  W&B runs:

- HN1024 tau `0.80`: `f9hkum1p`
- HN256 tau `0.70`: `sd4y1s3m`
- HN256 tau `0.76`: `gmh8oflv`

Summary artifacts:

- `/mnt/gemini/data1/jiaxuanluo/medicine_gs10k_pr_extra_tau_new_v5_no_gt_zero_oldnewv3_r32_20260523T1810/summary_medicine_gs10k_pr_extra_tau_metrics.md`
- `/mnt/gemini/data1/jiaxuanluo/medicine_gs10k_pr_extra_tau_new_v5_no_gt_zero_oldnewv3_r32_20260523T1810/summary_medicine_gs10k_pr_extra_tau_metrics.tsv`

Quick readout: HN1024 tau `0.80` gives the best BLEU among these three added
rows, while HN256 tau `0.76` is the best HN256 point and closes most of the
TERM_ACC / REAL_ADOPT gap relative to HN1024.  HN256 tau `0.70` preserves more
retriever recall but is weaker downstream on this one-talk check.
