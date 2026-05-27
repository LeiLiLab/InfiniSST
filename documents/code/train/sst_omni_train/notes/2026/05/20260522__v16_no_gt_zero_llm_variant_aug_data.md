## Hypothesis

Zeroing term maps on no-GT chunks can test whether no-term false positives teach the Speech LLM to treat term maps as background noise and reduce real adoption.

## Background / Motivation

V16 uses LLM-variant target-translation augmentation on top of the V13 retriever-timeline data. Its original training data still has dense retrieved term maps on chunks without GT terms.

## What changed vs baseline

Starting from the V16 JSONL, only chunks with empty `gt_terms_by_chunk[i]` are rewritten to `term_map:NONE`. Chunks with any GT terms keep the original V16 term_map and assistant target unchanged.

## Expected metrics

Training this data should reduce no-term noise exposure. The desired outcome is higher REAL_ADOPT / TERM_ACC without large BLEU loss on zh lm2 raw. This data-prep event itself only produces JSONL and stats.

## Verdict

Data generated successfully. Train split: avg term_map entries/chunk drops from 9.05 to 4.63; no-GT nonempty term_map rate drops from 87.75% to 0%; with-GT term maps are unchanged.
