# V15 term translation marker augmentation data

## Hypothesis

Replacing a subset of retrieved GT term translations with marked variants in
both `term_map` and assistant references should increase the Speech LLM's
pressure to adopt term-map values instead of relying on its default translation
prior.

## Background / Motivation

Recent quick evals suggest retriever recall is mostly sufficient, while
`REAL_ADOPT` remains the bottleneck.  V13 aligns the retriever timeline with
inference.  V15 keeps the same V13 retriever term maps and adds an adoption
stress signal for exact GT terms.

## What changed vs baseline

- Input data: V13 lm1..6 retriever timeline JSONL.
- Only GT terms that already appear in the current chunk's `term_map` are
  eligible.
- The canonical target translation must appear as an exact substring in the
  current/future assistant text.
- A marked target variant is written atomically to:
  - the current `term_map` line,
  - the matching `gt_terms_by_chunk` entry,
  - the first matching future assistant occurrence.
- Non-GT retriever entries and missed GT terms are not changed.

## Expected metrics

This data should improve downstream `REAL_ADOPT` and `TERM_ACC` if the model can
learn to copy/adopt term-map values.  BLEU may drop if the marker signal is too
strong, so the first eval should be a quick tagged ACL check before broader
sweeps.

## Verdict

Data build completed successfully.

Train data:
`/mnt/gemini/data1/jiaxuanluo/speech_llm_v15_marker_aug_retriever_timeline_zh_lh1b88kw_tau073_20260522/train_s_zh_v15_marker_aug_tau073_k10_minctx2p88.jsonl`

Summary:
`/mnt/gemini/data1/jiaxuanluo/speech_llm_v15_marker_aug_retriever_timeline_zh_lh1b88kw_tau073_20260522/v15_marker_aug_summary.json`

Key stats:

- GT terms: `39468`
- GT terms in current term map: `33544` (`84.99%`)
- augmented terms: `16781`
- augmented / GT terms: `42.52%`
- augmented / GT terms in current term map: `50.03%`
- missing future-reference exact matches among selected candidates: `6`
