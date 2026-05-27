## Hypothesis
Encoder choice may change cross-domain retriever recall under the same varctx576 training/eval setup.

## Background / Motivation
The current v3 retriever uses Qwen3-Omni audio features and BGE-M3 text embeddings. We need controlled text-side and audio-side encoder ablations while keeping data, hard-negative depth, GradCache, and eval protocol fixed.

## What changed vs baseline
  - Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag
  - Diff: only the encoder preset/model changes per launcher: BGE-large-en-v1.5, multilingual-E5-large with retrieval prefix, Whisper-medium.en, or WavLM-large-plus. All launchers reuse the v3 varctx576 data and medicine/dev/ACL eval settings.

## Expected metrics
Primary metric remains dev recall@10 gs10000; secondary remains ACL6060 recall@10. Medicine eval is logged as an additional cross-domain readout.

## Verdict
Pending.
