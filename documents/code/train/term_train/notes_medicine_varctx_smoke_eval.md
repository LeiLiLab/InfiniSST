## Hypothesis
The new medicine eval JSONL should load through the retriever eval path and emit `eval_medicine/*` metrics with the same glossary-scale machinery used for dev and ACL6060.

## Background / Motivation
Medicine is a new cross-domain readout built from ESO full WAVs, sentence-level terminology annotations, and MFA TextGrid word intervals on taurus. The smoke run validates data shape, audio chunk readability, medicine-specific glossary loading, and W&B metric names before the next full training launch uses this eval.

## What changed vs baseline
  - Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/lh1b88kw
  - Diff: Eval-only smoke on a 256-row medicine subset from `/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76/medicine_dev_dataset.jsonl`; adds `eval_medicine` to the training script without changing model weights.

## Expected metrics
The run should finish without data-loader, TextGrid-derived audio, glossary, or W&B logging errors. Metric values are not used as a model-selection result because this is a smoke subset.

## Verdict
Smoke eval passed on the 256-row medicine subset: `eval_medicine` loaded audio chunks and the 10k medicine glossary, logged metrics to W&B run `iiuqrsv1`, and exited cleanly.
