# New V4 old-new_v3 r32/a64 TP2 retry

## Hypothesis

The intended New V4 LLM-variant augmentation on old `new_v3` r32/a64 should be
trained with the same memory-safe model-parallel shape used by earlier working
speech-LLM runs: `TP=2` and sequence parallel enabled.

## Background / Motivation

The first 2-GPU r32/a64 launch (`sst_omni/5g2apva3`) used `TP=1` and failed at
the first training step with CUDA OOM in fused vocab-parallel cross entropy.
The older working r32/r64 launchers used tensor parallelism for the language
model path, so this retry keeps two GPUs but changes the parallel shape.

## What changed vs baseline

- Same New V4 train/dev JSONL as the cancelled/failed r32 run.
- LoRA rank/alpha remains `32/64`.
- Compute remains two Aries GPUs.
- Parallelism changes from `TP=1, sequence_parallel=false` to
  `TP=2, sequence_parallel=true`.
- Output root is versioned separately under
  `/mnt/aries/data7/jiaxuanluo/slm/speech_llm_new_v4_llm_variant_aug_oldnewv3_zh_r32a64_tp2_aries2`.

## Expected metrics

Primary readout remains tagged ACL `zh lm=2 raw`, plus one-paper
`2022.acl-long.110` extracted-glossary reference.  The retry is only meant to
recover the intended training run; it does not change the data hypothesis.

## Verdict

Pending training and eval.
