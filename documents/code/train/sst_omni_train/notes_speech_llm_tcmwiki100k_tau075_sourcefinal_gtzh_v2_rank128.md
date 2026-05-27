## Hypothesis

Increasing the corrected new_v2 speech LLM LoRA capacity to rank 128 may further improve term adoption for large ACL glossaries, but it may also overfit or increase false-copy noise relative to rank 32/64.

## Background / Motivation

The corrected rank-32 new_v2 run (`5e1iu7zo`) recovered the data source and GT translation alignment issues from `wog7tt7u`. Its one-paper eval improved most clearly at `gs10k, tau=0`, so rank128 is a capacity stress test on the same fixed data rather than a change to retrieval or term-map construction.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/sst_omni/runs/5e1iu7zo
- Diff: same corrected new_v2 dataset, same TCM-RAG checkpoint, same max length and 1 epoch; change LoRA rank/alpha from 32/32 to 128/128. Use 8 Taurus GPUs with global batch 8 and half the per-epoch iteration count to keep one epoch comparable.
- Historical comparison target: old v4_ner HF checkpoint `/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-zh-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4` remains a non-WandB historical baseline for SimulEval only.

## Expected metrics

Primary overnight check is ACL one-paper `2022.acl-long.110`, `lm=1`, raw/gs1k/gs10k, especially `gs10k` at `tau=0` and `tau=0.75`. Rank128 should beat rank32/rank64 on adoption to justify the extra capacity; a higher TERM_FCR would be evidence against using it.

## Verdict

Pending overnight rank128 training and one-paper SimulEval.
