# Export origin bsz4 streaming baseline HF checkpoint to mcore

## Hypothesis

Converting the existing pure streaming zh baseline checkpoint from HF format to Megatron-core format lets the all-GT term-map SFT start from the real streaming baseline instead of the raw Qwen3-Omni mcore base.

## Background / Motivation

The existing baseline checkpoint is:

`/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-zh-s_origin-bsz4`

It is already used by SimulEval scripts as the origin streaming Speech LLM baseline, but the current SFT training wrapper uses `megatron sft`, which expects an mcore checkpoint. This conversion keeps the training recipe unchanged while making initialization provenance explicit.

## What changed vs baseline

- Input HF checkpoint:
  - `/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-zh-s_origin-bsz4`
- Output mcore checkpoint:
  - `/mnt/gemini/data2/jiaxuanluo/gigaspeech_zh_s_origin_bsz4_mcore`
- Conversion command uses `swift export --model <HF> --to_mcore true --torch_dtype bfloat16`.
- No training happens in this event.

## Expected metrics

No metrics are expected. Success means the mcore directory contains `args.json`, `latest_checkpointed_iteration.txt`, and at least one `iter_*` checkpoint directory.

## Verdict

DEPRECATED before launch: do not convert this baseline for SFT. The pure-streaming HF checkpoint should be evaluated zero-shot with oracle term maps, while all-GT SFT should start from the initial Qwen3-Omni mcore base.
