## Hypothesis

Restricting Speech LLM term-map SFT data to `merge_multiplier=1..6` and using
the same timeline retrieval policy as inference should reduce train/inference
mismatch and improve REAL_ADOPT.

## Background / Motivation

The clean GT source has `merge_multiplier=1..12`, but the current agent
inference is evaluated at `lm=1..4`.  Very long chunks have denser GT terms and
may teach the Speech LLM to over-trust dense term maps.  The retriever itself
was trained on roughly `3..6 * 0.96s` contexts, so low-latency chunks should
wait until enough timeline buffer is available instead of padding.

## What changed vs baseline

- Input GT source:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_train_srcchunk_asr_100k_future_ref_gt_zh_20260522/train_s_zh_srcchunk_asr_100k_future_ref_gt_termmap_none.jsonl`.
- Keep only rows with `merge_multiplier` in `[1, 6]`.
- Use `lookback_sec=1.92`, `min_context_sec=2.88`, and
  `max_context_sec=5.76` so contexts stay within the retriever training range.
- For `lm=1`, the first two chunks have `term_map:NONE`; chunk 3 retrieves over
  the first three chunks, then keeps only evidence overlapping chunk 3.
- For `lm=2`, the first chunk has `term_map:NONE`; chunk 2 retrieves over the
  first two chunks.
- For `lm=3..6`, each user message is already long enough, so retrieval uses
  the current chunk only with no lookback.
- Use `tau=0.73`, `top_k=10`, no GT backfill.

## Expected metrics

- `rows_filtered_by_merge_multiplier` should correspond to `lm>=7` rows.
- Main diagnostic is `gt_term_recall` in generated term maps.
- Early lm=1/2 chunks should have lower non-empty term-map rate by design.

## Verdict

Build succeeded.

Output:
`/mnt/gemini/data1/jiaxuanluo/speech_llm_v13_lm1to6_retriever_timeline_zh_lh1b88kw_tau073_20260522/train_s_zh_v13_lm1to6_retriever_timeline_tau073_k10_minctx2p88.jsonl`

Key stats:

- rows: `6237`
- chunks: `52318`
- filtered `lm>=7` rows: `6263`
- GT terms: `39468`
- GT terms hit in term_map: `33544`
- GT term recall: `84.99%`
- GT chunk any-hit rate: `91.76%`
- GT chunk all-hit rate: `79.57%`
- nonempty term_map rate: `92.28%`
- no-GT nonempty term_map rate: `87.75%`
- average term_map entries/chunk: `9.05`

Bucket recall:

- `lm1`: `88.19%`
- `lm2to4`: `85.35%`
- `lm5to6`: `82.62%`

Validation: sampled `lm>=3` chunks have `context_sec == chunk_duration_sec`,
confirming that current chunks long enough for the retriever are evaluated
without lookback.
