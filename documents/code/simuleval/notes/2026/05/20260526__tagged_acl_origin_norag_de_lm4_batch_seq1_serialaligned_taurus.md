## Hypothesis

The remaining batch/serial gap may come from batch scheduler semantics rather than only cache/decode parameters. This run keeps the batch evaluator but forces one sequence at a time with `max_num_seqs=1`, `scheduler_batch_size=1`, and `schedule_mode=serial_by_lm`.

## Background / Motivation

The serial-aligned batch rerun on Aries still underperformed the serial verified lm4 gate. That run aligned fixed `max_new_tokens=40`, `80s/60s` cache, and `vllm_limit_audio=20`, but still used batch scheduling over five streams. This test checks whether making the batch evaluator effectively single-sequence recovers the serial result.

## What changed vs baseline

- Host: Taurus, GPUs 2,3.
- Method: InfiniSST/no-RAG En-De tagged ACL lm4.
- Batch engine is still used.
- `max_num_seqs=1`, `scheduler_batch_size=1`, `schedule_mode=serial_by_lm`.
- Fixed `max_new_tokens=40`.
- Cache remains `max_cache_seconds=80`, `keep_cache_seconds=60`.
- `vllm_limit_audio=20`.

## Expected metrics

If batch scheduling is the main source of mismatch, BLEU should move closer to the verified serial gate of 33.3008. If it remains around the previous batch result, the gap likely comes from deeper generation or prompt construction differences.

## Verdict

Pending.
