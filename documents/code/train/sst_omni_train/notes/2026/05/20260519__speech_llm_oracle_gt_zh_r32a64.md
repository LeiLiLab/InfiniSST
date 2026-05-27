# Speech LLM all-GT term_map SFT: zh oracle upper bound

## Hypothesis

Training the Speech LLM with all available ground-truth term maps should provide an upper-bound check for whether the model can use terminology evidence at all. If this setting does not materially improve term adoption under oracle evaluation, the bottleneck is likely the Speech LLM training/interface rather than retriever recall.

## Background / Motivation

The current retriever best checkpoint can provide high-quality term candidates, but downstream Speech LLM gains have been smaller than expected. Before tuning noisy retriever term maps, we need a clean oracle setting where each speech chunk receives only MFA-aligned ground-truth terms, and non-term chunks receive `term_map:NONE`.

Data-prep parent manifest:

- `20260519T0010__data_prepare__oracle_gt_termmap_zh_sft`

## What changed vs baseline

- Baseline behavior: existing zh streaming Speech LLM checkpoints were not trained specifically for clean all-GT term maps.
- This run fine-tunes Qwen3-Omni with LoRA on oracle GT term-map SFT data.
- Training data:
  - `/mnt/gemini/data1/jiaxuanluo/speech_llm_oracle_gt_termmap_zh_20260519/train_s_zh_v4_ner_baseline_aligned_rate1p0_k20_oracle_gt_termmap_none.jsonl`
- Validation data:
  - `/mnt/gemini/data1/jiaxuanluo/speech_llm_oracle_gt_termmap_zh_20260519/dev_s_zh_v4_ner_baseline_aligned_freq_k20_oracle_gt_termmap_none.jsonl`
- Default initialization:
  - `/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct-v2/`
- The pure-streaming HF baseline `/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-zh-s_origin-bsz4` is evaluated zero-shot with oracle term maps instead of being converted to mcore and further SFT-tuned.
- Training config:
  - Taurus 4 GPU
  - LoRA rank 32, alpha 64
  - max length 3072
  - 1 epoch

## Expected metrics

Primary follow-up is oracle term-map SimulEval/offline readout on dev, ACL, and medicine with `TERM_ACC`, `realAdopt`, sentence-level `FCR`, BLEU, and StreamLAAL. Expected result is higher `TERM_ACC` and `realAdopt` than the pure streaming baseline, with low `FCR` when the term map is clean.

## Verdict

Training completed for one epoch from the initial mcore Qwen3-Omni base, and the final checkpoint was exported to HF for downstream oracle SimulEval. Use the HF export at `/mnt/gemini/data2/jiaxuanluo/speech_llm_oracle_gt_termmap_zh_r32a64_taurus4/keep1.0_r32/v1-20260519-105111-hf` for the next readout; exact training metrics live in W&B run `3h4wm92o`.
