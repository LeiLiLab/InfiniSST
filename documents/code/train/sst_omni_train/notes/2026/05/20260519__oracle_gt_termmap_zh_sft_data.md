# Oracle GT Term Map SFT Data For Speech LLM

## Hypothesis

An all-GT term-map SFT dataset should isolate whether the Speech LLM can learn to use clean terminology evidence before we introduce retriever noise.

## Background / Motivation

Existing zh Speech LLM checkpoints mostly measure pure streaming translation or noisy term-map training.  To test the upper bound, each chunk with `gt_terms_by_chunk` should receive exactly those terms in `term_map`, while chunks without GT terms should receive `term_map:NONE`.

## What changed vs baseline

- Baseline run URL: N/A; data-prep event derived from existing zh SFT JSONL.
- Diff:
  - train input: `/mnt/gemini/data/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_final.jsonl`
  - dev input: `/mnt/gemini/data1/jiaxuanluo/dev_s_zh_v4_ner_baseline_aligned_freq_k20.jsonl`
  - overwrite every user audio message with GT-only `term_map` or `term_map:NONE`
  - preserve assistant targets and audio paths
  - filter rows that lack `gt_terms_by_chunk`; this is explicitly counted in stats because oracle term-map data cannot be reconstructed without GT metadata

## Expected metrics

This is a data-prep event.  Expected output is train/dev JSONL ready for Megatron SFT, plus stats showing chunk count, GT chunk ratio, and term-map size distribution.

## Verdict

SUCCESS: built oracle-GT train/dev SFT JSONL. Train output has 12,318 conversations, 68,450 audio chunks, GT chunk ratio 75.43%, and 124,843 GT terms; 182 train rows without `gt_terms_by_chunk` were explicitly filtered. Dev output has 303 conversations, 829 audio chunks, GT chunk ratio 69.96%, and 1,152 GT terms; 52 dev rows without `gt_terms_by_chunk` were explicitly filtered.
