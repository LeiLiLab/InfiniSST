# New V9 de/ja data on old-new_v3-equivalent line

## Hypothesis

The accepted zh New V9 recipe should transfer to de/ja if we first recreate the same old-new_v3 style data distribution: dense retriever term maps, cap 20, GT backfill, no-GT-zero, and assistant-side `<term>` target tagging.

## Background / Motivation

The zh main model was not built directly from the legacy LLM-generated `v4_ner_baseline` JSONL.  Its data lineage is:

1. old `new_v3` retriever-SFT data:
   `sourcefinal_tcmwiki100kgt`, `tau=0.75`, `d9`, `k20`, post-filter cap, GT override/backfill
2. New V4:
   LLM-variant target-translation replacement on the old `new_v3` data
3. New V5:
   no-GT chunks set to `term_map:NONE`
4. New V9:
   assistant-side target translations wrapped as `<term>...</term>`, with local rewrite and boundary-prefix delay

The de/ja legacy JSONL files do not contain `gt_terms_by_chunk`, unlike the zh source JSONL.  Therefore the de/ja pipeline adds an explicit Stage 0: derive conservative chunk-level GT terms from existing LLM-generated `term_map` entries whose target translations are supported by current/future assistant text.

## What changed vs baseline

- Inputs:
  - `/mnt/gemini/data1/jiaxuanluo/train_s_de_v4_ner_baseline_aligned_rate1.0_k20_final.jsonl`
  - `/mnt/gemini/data1/jiaxuanluo/train_s_ja_v4_ner_baseline_aligned_rate1.0_k20_final.jsonl`
- Stage 0:
  - derive `gt_terms_by_chunk` from existing language-specific term maps
  - extract a top-100k language-specific train glossary from those term maps
- Stage 1/2:
  - run the old TCM MaxSim retriever with `tau=0.75`, `retrieval_density=9`, `max_top_k=20`
  - rebuild term maps with `tcm_filtered_with_gt_backfill`, cap 20
- Stage 3:
  - New V4 LLM natural target-translation variants
- Stage 4:
  - New V5 no-GT-zero
- Stage 5:
  - New V9 assistant `<term>...</term>` tagging with local rewrite and boundary-prefix delay

## Expected metrics

Data diagnostics should be comparable to the zh New V9 line:

- derived GT chunk rate should be nontrivial for de/ja
- rebuilt term-map density should be dense and capped at 20
- no-GT-zero should remove term maps only from chunks with no derived GT
- assistant tag rate should be high over candidate GT terms after min-length and pronoun/source-token filtering

## Verdict

Pending.
