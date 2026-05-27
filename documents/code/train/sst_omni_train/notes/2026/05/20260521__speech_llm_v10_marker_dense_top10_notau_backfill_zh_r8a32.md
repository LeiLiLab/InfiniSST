## Hypothesis

Marker-augmenting GT target translations in both term_map and assistant targets should create a stronger literal-copy signal than XML source-term tags, improving REAL_TERM_ADOPT and TERM_ACC.

## Background / Motivation

The remaining gap is not retriever recall; it is Speech LLM adoption of the provided target translation.  V10 keeps the V9 dense top10 no-tau distribution but wraps exact-reference GT target translations with random marker strings during SFT so the model must learn to copy the term-map target string exactly.

## What changed vs baseline

- Train data: V10 marker-augmented dense top10 no-tau retriever term maps.
- Marker policy: deterministic random wrappers around exact-reference GT target translations; assistant references are modified only by exact substring replacement.
- Inference eval remains plain term_map format, matching deployment.
- LoRA: rank 8, alpha 32.
- Base model: Qwen3-Omni MCore initial checkpoint.

## Expected metrics

On quick tagged ACL zh lm2/raw, V10 should improve REAL_TERM_ADOPT over V9 if the marker curriculum transfers to plain inference term maps.  A drop would indicate marker overfitting or output-distribution mismatch.

## Verdict

Pending.
