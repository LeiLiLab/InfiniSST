## Hypothesis

Strict MFA-exact medicine evaluation should remove both sentence-center fallback
labels and char-proportional timing estimates from the target set.

## Background / Motivation

The no-fallback clean run `6dxdrrl8` removed target-language source labels that
could not be located at all, but still retained terms located only by
sentence-text character interpolation. The strict readout uses only terms with
exact matches in the MFA word intervals.

## What changed vs baseline

Use data-prep event
`20260518T1812__data_prepare__medicine_varctx_clean_mfa_exact_only`:

- JSONL:
  `/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/medicine_dev_dataset.jsonl`
- glossary:
  `/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/medicine_glossary_gt_plus_medicine_wiki_gs10000.json`

## Expected metrics

Report base, gs1000, and gs10000 recall on 2,408 strict MFA-exact term rows.

## Verdict

Finished as W&B run `614b3nbi`.

Strict MFA-exact medicine recall@10 at step 2640:

- base: 0.9522, 115 misses / 2,408 term rows
- gs1000: 0.9489, 123 misses / 2,408 term rows
- gs10000: 0.9348, 157 misses / 2,408 term rows

Compared with the no-fallback run `6dxdrrl8`, removing char-proportional
positives improves recall by about +2.01 pp base, +1.97 pp gs1000, and
+1.88 pp gs10000. Compared with the original fallback-contaminated medicine
data, the strict readout is about +7.11 pp base, +7.07 pp gs1000, and +7.44 pp
gs10000.
