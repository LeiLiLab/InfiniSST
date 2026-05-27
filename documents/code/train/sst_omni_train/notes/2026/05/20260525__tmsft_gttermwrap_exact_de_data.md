# TM-SFT Exact GT Term Wrapping Data, De, 2026-05-25

## Hypothesis

Directly wrapping supported GT target translations in the historical German
TM-SFT training set should test whether tagged term supervision helps without
changing the SLM recipe through LLM variants, no-GT-zero, or retriever rebuilt
term maps.

## Background / Motivation

The historical TM-SFT German model used:

`/mnt/gemini/data1/jiaxuanluo/train_s_de_v4_ner_baseline_aligned_rate1.0_k20_final.jsonl`

That JSONL stores legacy term-map entries inside each user message but does not
store `gt_terms_by_chunk` as a top-level field. This data event derives
chunk-level GT terms from the embedded `term_map:` only when the target
translation appears in current/future assistant text, then wraps exact future
assistant occurrences with `<term>...</term>`.

## What changed vs baseline

- Input data remains the historical German TM-SFT training JSONL.
- User prompts and embedded term maps are unchanged.
- No retriever rebuild is run.
- No OpenAI/LLM variant augmentation is run.
- No no-GT term-map zeroing is run.
- Assistant targets are modified only by exact `<term>` wrapping of supported GT
  target translations.
- Exact wrapping uses Latin/digit boundary checks to avoid broken tags inside a
  larger word.

## Expected metrics

This data should support a quick Aries 8-GPU SFT branch whose first readout is
German tagged ACL raw `lm=2` with HN1024 and `tau=0.79`. The intended comparison
is against the verified no-RAG German `lm=2` BLEU baseline and the existing
TM-SFT + HN1024 reference.

## Verdict

Completed.

- Train rows: 12,500; train chunks: 71,730.
- Embedded term-map entries: 400,416.
- Exact GT terms selected from future assistant text: 86,516.
- Assistant `<term>` exact replacements: 77,821.
- Rows with assistant tags: 11,968.
- Tagged assistant messages: 39,665.
- Dev rows: 355; dev chunks: 1,946; dev exact GT terms: 2,449; dev
  replacements: 2,199.
- Malformed tag messages: 0.
- Latin word-cut tag messages: 0.

Unwrapped selected GT terms are counted explicitly, primarily because a longer
overlapping term was already wrapped or because the exact boundary check would
split a Latin/digit word.
