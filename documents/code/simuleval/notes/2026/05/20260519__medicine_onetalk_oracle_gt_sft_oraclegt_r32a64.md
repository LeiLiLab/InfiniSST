# Oracle term_map readout: all-GT SFT oraclegt_r32a64 on medicine one-talk

## Hypothesis

The all-GT term_map SFT model should use clean oracle terminology evidence more reliably than the pure streaming baseline, while reducing unsupported term copying.

## Background / Motivation

The origin bsz4 zero-shot baseline can already read oracle term maps, but it still has non-trivial false-copy behavior. This readout evaluates the newly trained all-GT zh SFT checkpoint on the same one-talk medicine setup for a direct sanity check.

Parent training run:

- `sst_omni/3h4wm92o`

## What changed vs baseline

- Baseline readout:
  - `simuleval_eval/x818tsc6`
  - model `/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-zh-s_origin-bsz4`
- This readout uses the all-GT SFT HF export:
  - `/mnt/gemini/data2/jiaxuanluo/speech_llm_oracle_gt_termmap_zh_r32a64_taurus4/keep1.0_r32/v1-20260519-105111-hf`
- Evaluation is otherwise the same:
  - medicine sample 404
  - latency multiplier 2
  - strict MFA-only translated GT terms
  - oracle term-map injection
  - no retriever

## Expected metrics

Expected behavior is similar or higher `TERM_ACC` and `REAL_TERM_ADOPT` than zero-shot baseline, with lower sentence-level false-copy behavior. Exact values are recorded in W&B, not in this notes file.

## Verdict

Completed. Exact metrics are recorded in WandB run `2r9bts4j`; compared with the zero-shot baseline readout `x818tsc6`, the all-GT SFT model keeps similar oracle term accuracy while substantially reducing false-copy behavior on this medicine one-talk sanity check.
