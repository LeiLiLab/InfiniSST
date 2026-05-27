## Hypothesis

Increasing the corrected new_v2 speech LLM LoRA capacity from rank 32 to rank 64 may improve term adoption under larger ACL glossaries while preserving the GT-translation fix and deployment-like TCM-RAG term maps.

## Background / Motivation

The rank-32 new_v2 run (`5e1iu7zo`) fixed two data issues from `wog7tt7u`: it used the baseline-final source JSONL and forced retrieved GT terms to use the chunk-specific `gt_terms_by_chunk` zh translation. One-paper eval showed the strongest gain at `gs10k, tau=0`, suggesting the data fix helps but the speech LLM may still be capacity-limited for larger term maps.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/sst_omni/runs/5e1iu7zo
- Diff: same corrected new_v2 dataset, same TCM-RAG checkpoint, same max length and 1 epoch; change LoRA rank/alpha from 32/32 to 64/64. Use 8 Taurus GPUs with global batch 8 and half the per-epoch iteration count to keep one epoch comparable.
- Historical comparison target: old v4_ner HF checkpoint `/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-zh-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4` remains a non-WandB historical baseline for SimulEval only.

## Expected metrics

Primary overnight check is ACL one-paper `2022.acl-long.110`, `lm=1`, raw/gs1k/gs10k, especially `gs10k` at `tau=0` and `tau=0.75`. Expect rank64 to improve TERM_ACC/TERM_ADOPTION over rank32 without materially increasing TERM_FCR.

## Verdict

Pending overnight rank64 training and one-paper SimulEval.
