# Streaming no-RAG lm2 baseline for restored ESO medicine samples

## Hypothesis

The streaming no-RAG Qwen3-Omni baseline should expose which medicine_gt terms
are already easy for the speech LLM and which remain hard without retrieval or
oracle term injection.

## Background / Motivation

This run executes the student-facing medicine_gt strict-filter baseline locally
because aries GPU 6 and 7 are currently available. The input data is the
restored abbreviation / exact-match ESO medicine test directory:

```text
/mnt/taurus/home/jiaxingxu/rag-sst/eso-dataset/outputs_v2_abbrev_exact_match_abbrev_restored/test
```

The target coverage is three languages (`zh`, `de`, `ja`), latency multiplier
`2`, and five medicine samples (`404 545006 596001 605000 606`). To avoid
reloading the 30B speech LLM for every individual sample, each setting is one
`(language, lm=2)` run containing all five samples.
The primary output for filtering is still one hypothesis row per sample in the
per-setting `instances.log`.

Outputs are written under:

```text
/mnt/gemini/data1/jiaxuanluo/medicine_norag_baseline_abbrev_restored_batched_20260522
```

## What changed vs baseline

This is no-RAG streaming generation only:

- no retriever
- no oracle term map injection
- sample-specific source/target lists prepared from the restored ESO v2 test
  directory, then concatenated into a five-sample batch per language
- sample-specific medicine_gt glossary files are retained for exact-match
  metadata; the batched glossary is used only for bookkeeping and output naming

The model checkpoints are the origin bsz4 speech LLM exports:

- zh: `/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-zh-s_origin-bsz4`
- de: `/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-de-s_origin-bsz4`
- ja: `/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-ja-s_origin-bsz4`

## Expected metrics

This event is primarily a hypothesis-generation baseline. The launcher records
per `(language, lm=2)` wall-clock runtime in `timing.tsv` and extracts
per-sample hypotheses from `instances.log` into `hypotheses.tsv`. Term acceptance decisions
should be made later from exact-match plus manual review rather than from this
notes file.

## Verdict

Running. Final pass/fail and timing summary should be filled after the detached
launcher completes.
