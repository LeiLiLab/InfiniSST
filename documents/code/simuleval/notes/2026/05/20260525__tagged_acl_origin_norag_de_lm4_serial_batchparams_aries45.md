## Hypothesis

The previously validated En-De lm4 InfiniSST/no-RAG serial baseline used a smaller
generation cap than the current lm4 batch readout. Re-running serial lm4 with the
batch cap and vLLM context settings will determine whether the BLEU gap is caused
by batch scheduling or by parameter mismatch.

## Background / Motivation

The validated serial lm4 baseline from 20260524 used `MAX_NEW_TOKENS=40`.
The current no-RAG batch lm4 baseline uses `MAX_NEW_TOKENS=80`,
`VLLM_LIMIT_AUDIO=128`, `VLLM_MAX_MODEL_LEN=12288`, and cache window 80/60s.
This run keeps the serial SimulEval path but matches the lm4 batch generation
budget and vLLM limits as closely as the legacy serial launcher allows.

## What changed vs baseline

- Language: En-De.
- Dataset/glossary: ACL6060 tagged raw denominator.
- Method: InfiniSST/origin SLM, no RAG.
- Latency multiplier: lm4.
- Protocol: serial SimulEval baseline agent.
- Generation cap: `MAX_NEW_TOKENS=80`, matching current batch lm4.
- vLLM limits: `VLLM_LIMIT_AUDIO=128`, `VLLM_MAX_MODEL_LEN=12288`.
- Cache window: `MAX_CACHE_SECONDS=80`, `KEEP_CACHE_SECONDS=60`.

## Expected metrics

If the old 33.3 BLEU point depended mainly on the smaller cap or model-length
setting, this serial batch-parameter run may shift toward the current batch lm4
BLEU. If it remains close to 33.3, the batch driver/scheduling path is likely the
main source of the gap.

## Verdict

Pending.
