# Medicine gs10k PR Sweep for New V5 no-GT-zero old-new_v3 R32

## Hypothesis

The same Speech LLM can show different terminology behavior depending on the
retriever operating point.  We sweep no-HN and HN1024 retrievers at three
dev-recall-drop tau budgets and evaluate the hardest quick readout: medicine
with the gs10k runtime glossary.

## Background / Motivation

The current Speech LLM winner is
`speech-llm-new_v5-no-gt-zero-oldnewv3-r32a64-tp2-aries2_keep1.0_r32`.
Its tagged ACL `zh lm=2 raw` quick eval reached BLEU 48.20, TERM_ACC 90.00%,
REAL_ADOPT 90.19%, TERM_FCR 7.53%, and StreamLAAL 1663.60.

Retriever operating points use fixed raw-domain denominator readouts from the
no-HN vs HN1024 report:

| setting | tau | medicine gs10k P/R |
|---|---:|---|
| no-HN | 0.61 | 10.34 / 94.10 |
| no-HN | 0.70 | 14.38 / 91.03 |
| no-HN | 0.73 | 17.23 / 88.50 |
| HN1024 | 0.72 | 10.90 / 92.69 |
| HN1024 | 0.78 | 16.21 / 89.00 |
| HN1024 | 0.82 | 25.35 / 83.55 |

## What changed vs baseline

- Speech LLM fixed to New V5 no-GT-zero old-new_v3 r32/a64.
- Eval domain fixed to medicine, language `zh`, `lm=2`, one talk `sample_404`.
- Runtime glossary fixed to medicine gs10k.
- Metric glossary fixed to the per-talk strict/raw medicine terms generated from
  the old medicine strict source, rather than following the gs10k runtime bank.
- Retriever changes across six `(model, tau)` settings only.

## Expected metrics

Report BLEU, TERM_ACC, REAL_ADOPT, TERM_FCR, and StreamLAAL for all six
settings.  This sweep is a quick sensitivity readout, not a final full-domain
medicine table.

## Verdict

Completed on medicine `sample_404`, `zh`, `lm=2`, runtime medicine gs10k,
fixed per-talk strict/raw denominator.

Summary artifact:

`/mnt/gemini/data1/jiaxuanluo/medicine_gs10k_pr_sweep_new_v5_no_gt_zero_oldnewv3_r32_20260523T0710/summary_medicine_gs10k_pr_sweep_metrics.md`

| Retriever | Tau | Report P/R | W&B | BLEU | TERM_ACC | REAL_ADOPT | SOURCE_SENT_FCR | StreamLAAL |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| no-HN | 0.61 | 10.34 / 94.10 | dfyziwtq | 42.39 | 88.06 | 89.13 | 13.19 | 1838.2 |
| no-HN | 0.70 | 14.38 / 91.03 | 4wicgr7b | 42.78 | 87.31 | 88.94 | 15.38 | 1765.4 |
| no-HN | 0.73 | 17.23 / 88.50 | wcfwm4t2 | 42.00 | 89.18 | 91.14 | 10.99 | 1932.3 |
| HN1024 | 0.72 | 10.90 / 92.69 | a0uib4pp | 41.43 | 86.94 | 88.74 | 10.44 | 1866.9 |
| HN1024 | 0.78 | 16.21 / 89.00 | rzx4ttz4 | 42.35 | 90.30 | 93.30 | 12.64 | 1946.7 |
| HN1024 | 0.82 | 25.35 / 83.55 | 6ffq1ifl | 43.40 | 86.19 | 87.21 | 9.34 | 1925.0 |

Quick readout: the best one-talk TERM_ACC is HN1024 at tau 0.78, not the
highest-precision tau 0.82.  The high-precision/low-recall point loses too much
coverage for this Speech LLM setting.  Among no-HN points, tau 0.73 gives the
best TERM_ACC/REAL_ADOPT on this talk despite lower reported retriever recall.

Operational notes:

- The first no-HN run started on two GPUs before the current eval script's
  stricter `VLLM_TP_SIZE+1` GPU check.  The HN resume used
  `RAG_GPU_OVERRIDE=cuda:1` to reproduce the two-GPU shared-RAG layout requested
  for taurus GPUs 4,5.
- `no-HN tau=0.73` and `HN1024 tau=0.82` completed SimulEval inference but hit
  a transient shell parse error before offline eval.  Their TSV/W&B rows were
  recovered from the existing `instances.log`; inference was not rerun.
