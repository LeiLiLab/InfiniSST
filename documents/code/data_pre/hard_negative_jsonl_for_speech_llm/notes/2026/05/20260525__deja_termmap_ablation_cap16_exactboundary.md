## Hypothesis

The German and Japanese Speech LLM data should follow the stronger Zh-style construction: capped term maps, exact assistant-side term tags, and a controlled comparison between LLM-generated negatives and retriever-recalled terms. This should improve TERM_ACC without repeating the uncapped term-map density that likely damaged BLEU.

## Background / Motivation

The historical De/Ja TM-SFT JSONLs contain embedded LLM-generated term maps but no `gt_terms_by_chunk`, and their term maps can be very dense. This data-prep event derives exact GT terms from supported target strings, caps term-map density to 16 terms per chunk, and builds a matched retriever-recalled branch using the HN1024 MaxSim retriever.

## What changed vs baseline

- Branch A keeps old LLM-generated term maps but caps each chunk to 16 entries, preserving exact-derived GT terms first.
- Branch B rebuilds user-side term maps from HN1024 retriever outputs with `tcm_filtered_with_gt_backfill` and the same cap.
- Both branches use exact target `<term>...</term>` wrapping with Latin-boundary checks.
- The only non-exact repair allowed is boundary-only prefix/suffix repair across adjacent assistant chunks.

## Expected metrics

The retriever-recalled branch should reduce noisy term-map exposure compared with old uncapped TM-SFT. It should keep TERM_ACC above no-RAG while recovering BLEU relative to the current De/Ja RASST rows.

## Verdict

Success. Built De/Ja LLM-generated and HN1024 retriever-recalled term-map branches under `/mnt/gemini/data1/jiaxuanluo/speech_llm_deja_termmap_ablation_cap16_exactboundary_20260525`.
All four final train/dev JSONLs passed validation with `max_termmap_entries=16`, `malformed_tag_messages=0`, and `latin_boundary_cut_messages=0`.
Train term-map chunk rates are: De LLM-gen 0.7642, De retriever 0.9329, Ja LLM-gen 0.7409, Ja retriever 0.9034.
These data are ready for matched downstream SFT runs; downstream SFT/eval is not part of this data-prep verdict.
