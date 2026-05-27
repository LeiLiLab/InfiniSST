## Hypothesis

Canonical exact-span GT should provide a clean de Speech LLM SFT base without exploding OpenAI calls: source terms must pass MFA exact matching, old-new_v3 source candidate filtering, and exact future assistant support from the legacy de term-map translation lexicon.

## Background / Motivation

The source-union OpenAI rewrite variants produced too many candidate rewrite calls. We first need a clean and tractable GT base, then can apply capped LLM variant augmentation only to selected GT terms.

## What changed vs baseline

- Source glossary is wiki100k plus de old-new_v3 noun/entity source candidates.
- Legacy term_map is used only as a translation lexicon and exact future-span filter.
- Stage A skips OpenAI per-candidate rewrite and keeps the exact supported target span.
- Downstream old-new_v3 TCM retriever term_map, GT backfill, no-GT-zero, and assistant `<term>` tags remain unchanged.

## Expected metrics

Data-prep should finish much faster than OpenAI-per-candidate variants. GT terms should be cleaner than the first de New V9 run and much denser than pure wiki100k source matching.

## Verdict

Pending data-prep completion and validation.

## Runtime note

The first Stage C1 retriever shard00 attempted to start on GPU6 and failed with CUDA OOM because a concurrent JA vLLM eval was already occupying that GPU. Shard00 was relaunched on GPU2, and a recovery launcher was added to wait for shard00/shard01, merge the source-copy retriever outputs, and resume the standard builder from Stage C2.

## Monitor completion

- Final train JSONL: `/mnt/gemini/data1/jiaxuanluo/speech_llm_new_v9_mfa_npfilter_srcunion_lexexact_oldnewv3_de_20260524/train_s_de_new_v9_mfa_openai_rewrite_oldnewv3.jsonl`
- Train rows: `12500`
- Final dev JSONL: `/mnt/gemini/data1/jiaxuanluo/speech_llm_new_v9_mfa_npfilter_srcunion_lexexact_oldnewv3_de_20260524/dev_s_de_new_v9_mfa_openai_rewrite_oldnewv3_first355.jsonl`
- Dev rows: `355`
- Summary: `/mnt/gemini/data1/jiaxuanluo/speech_llm_new_v9_mfa_npfilter_srcunion_lexexact_oldnewv3_de_20260524/new_v9_mfa_openai_rewrite_oldnewv3_de_summary.json`
