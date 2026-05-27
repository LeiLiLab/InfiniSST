## Hypothesis

Training the speech LLM on TCM-retrieved term maps built from the same source JSONL as the v4_ner final baseline, while forcing all GT translations to match `gt_terms_by_chunk`, should recover positive supervision lost in v1 and reduce wrong-GT translation noise under `tau=0.75` inference.

## Background / Motivation

The first TCM-wiki100k SLM run (`wog7tt7u`) used a cleaned source JSONL with fewer GT chunks and rebuilt term maps that kept retriever/glossary translations when a GT term was already retrieved. Dataset comparison showed this removed many positives and introduced GT translation mismatches, explaining weaker term adoption despite deployment-like tau filtering.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/sst_omni/runs/wog7tt7u
- Diff: use `/mnt/gemini/data/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_final.jsonl` as the source JSONL instead of the cleaned data; keep the same strongest TCM RAG checkpoint, wiki100k+GigaSpeech-GT glossary, `tau=0.75`, and `k=min(10, ceil(duration_sec * 5))`; in `tcm_filtered_with_gt_backfill`, any retrieved GT term keeps its retrieved order position but uses the chunk-specific `gt_terms_by_chunk` zh translation.
- Historical comparison target: the old v4_ner HF checkpoint `/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-zh-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4` is still historical debt because it is not a schema-compliant WandB run.

## Expected metrics

Primary check is corrected ACL one-paper/per-paper SimulEval with raw/gs1k/gs10k glossaries and `RAG_SCORE_THRESHOLD=0.75`. Expect v2 to improve TERM_ADOPTION/TERM_ADOPTION_MICRO over `wog7tt7u` and narrow the gap to the old v4_ner baseline without increasing TERM_FCR materially.

## Verdict

Pending v2 data build, training, and targeted SimulEval evaluation.
