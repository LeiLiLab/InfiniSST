# Speech LLM New V5 no-GT-zero old-new_v3 r32/a64 TP2

## Hypothesis

New V5 combines the stronger old `new_v3` speech-LLM term-map data, LLM-variant target-translation augmentation, and the no-GT-zero rule.  It should preserve New V4's adoption pressure while reducing noise on chunks where no reference-matched GT term is present.

## Background / Motivation

V16 no-GT-zero improved the quick tagged ACL `zh lm=2 raw` readout over V16.  New V5 tests whether the same rule helps the stronger old-`new_v3` rank-32 line.

## What changed vs baseline

- Base data: New V4 old-`new_v3` LLM-variant cache-only data.
- Data ablation: no-GT chunks are rewritten to `term_map:NONE`.
- LoRA: r32/a64.
- Parallelism: TP=2 with sequence parallel, matching the successful New V4 retry.
- Max length: 3072.

## Expected metrics

Primary downstream readout is tagged ACL `zh lm=2 raw`.  A useful model should improve `TERM_ACC` and `REAL_ADOPT` without a large BLEU regression.

## Verdict

Training process failed after writing the iteration-1000 MCore checkpoint because
`/mnt/aries/data7` ran out of space while TensorBoard was appending events.
The checkpoint itself is present and is being exported to HF on data6 for
downstream quick eval.
