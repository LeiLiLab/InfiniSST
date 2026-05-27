## Hypothesis

The En-De lm4 serial baseline that reached the validated 33.3 BLEU point used
`MAX_NEW_TOKENS=40`. A fresh batch readout with the same lm4 generation budget
and old serial-compatible context settings tests whether the batch driver itself
is responsible for the lower BLEU.

## Background / Motivation

The current batch baseline table uses lm4 with `MAX_NEW_TOKENS=80`, while the
older validated serial lm4 baseline used `MAX_NEW_TOKENS=40`. There is also an
older batch max40 readout from 20260524, but this event is a clean rerun on Aries
GPU 6,7 with a new manifest and output root.

## What changed vs baseline

- Method: InfiniSST/origin En-De Speech LLM, no RAG.
- Dataset: ACL6060 tagged raw glossary denominator.
- Latency multiplier: lm4 only.
- Protocol: same-lm batch vLLM readout with five talks in one batch.
- Generation cap: `MAX_NEW_TOKENS=40`, matching the old validated serial lm4 run.
- vLLM/cache: `VLLM_MAX_MODEL_LEN=16384`, `VLLM_LIMIT_AUDIO=128`, cache 80/60s.
- W&B logging is disabled for this standalone rerun; artifacts and manifest are
  the source of truth.

## Expected metrics

If the old 33.3 BLEU result was mainly due to the lm4 max-new-token setting, this
batch max40 rerun should improve relative to the current batch max80 baseline.
If it stays near the previous batch max40 readout, the issue is likely the batch
driver/scheduling behavior rather than only the cap.

## Verdict

Pending.
