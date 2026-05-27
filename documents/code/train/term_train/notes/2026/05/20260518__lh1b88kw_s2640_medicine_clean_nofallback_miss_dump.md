## Hypothesis

The previous medicine recall gap was inflated by source labels that could not be
located in the English audio/text and were kept through sentence-center
fallback. Re-evaluating on the cleaned no-fallback medicine target set should
give a more interpretable domain readout and miss list.

## Background / Motivation

The source medicine preprocessing previously treated every JSONL `term` as an
original positive. For unmatched source labels, it assigned a fallback span at
the sentence center. This produced positives such as `dosimetristen` in chunks
whose text did not contain the term.

## What changed vs baseline

Use the clean medicine data-prep event
`20260518T1741__data_prepare__medicine_varctx_clean_nofallback`:

- JSONL:
  `/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_nofallback/medicine_dev_dataset.jsonl`
- glossary:
  `/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_nofallback/medicine_glossary_gt_plus_medicine_wiki_gs10000.json`
- dropped-term audit:
  `/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_nofallback/medicine_dev_dataset_dropped_terms.json`

## Expected metrics

Recall should be reported for base, gs1000, and gs10000 on the cleaned target
set. Miss dumps should no longer include `sentence_center_fallback` positives.

## Verdict

Finished as W&B run `6dxdrrl8`.

Clean medicine recall@10 at step 2640:

- base: 0.9321, 210 misses / 3,094 term rows
- gs1000: 0.9292, 219 misses / 3,094 term rows
- gs10000: 0.9160, 260 misses / 3,094 term rows

This removes the previous `sentence_center_fallback` positives from the target
set. Compared with the old fallback-contaminated data, recall improves by
approximately +5.10 pp base, +5.10 pp gs1000, and +5.56 pp gs10000.
