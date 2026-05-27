# Tagged ACL V3 Speech LLM 3x3 Probe

## Hypothesis

The V3 speech-LLM SFT variants should improve robustness on the known tagged ACL failure modes: sparse early term maps for `de lm3/raw` and noisy dense term maps for `ja lm1/gs10k`.

## Background / Motivation

Previous tagged ACL readouts showed that retriever strict-term coverage was high, but the speech LLM could still collapse due to no-term early chunks or excessive noisy term maps. This probe compares three V3 SFT variants on three targeted settings instead of rerunning the full language/latency/glossary grid.

## What changed vs baseline

- Baseline run URL: existing origin-bsz4 tagged ACL pipeline.
- Diff: replace the origin-bsz4 streaming SLM with one of three V3 SFT HF exports:
  - real retriever timeline term maps
  - tagged term-map formatting
  - real retriever plus adversarial/noisy distractors
- Settings:
  - `zh`, `lm=2`, `gs=raw`
  - `de`, `lm=3`, `gs=raw`
  - `ja`, `lm=1`, `gs=10k`

## Expected metrics

Track BLEU, TERM_ACC, REAL_TERM_ADOPT, TERM_FCR, StreamLAAL. The important comparisons are against the known origin-bsz4 failures in `de lm3/raw` and `ja lm1/gs10k`.

## Verdict

Completed on `2026-05-21`. All 9 targeted settings produced `eval_results.tsv`
under `/mnt/aries/data7/jiaxuanluo/slm/tagged_acl_v3_speech_llm_3x3_20260521T024609`
and were logged to W&B `simuleval_eval`.

Operational note: the first all-8 wrapper used process-count concurrency and
reused a busy GPU pair after early settings finished, causing one `tagged ja`
startup OOM. The launcher was patched with `RUN_TAGS_OVERRIDE` and
`SKIP_COMPLETED_OVERRIDE`; missing settings were completed with filtered direct
runs on free GPU pairs.

High-level readout:

| setting | best variant by TERM_ACC | note |
| --- | --- | --- |
| `zh lm2 raw` | `adv` | very small spread; all three around 79% TERM_ACC |
| `de lm3 raw` | `tagged` | strongest gain among V3 variants; better BLEU/TERM_ACC and lower FCR than `real` |
| `ja lm1 gs10k` | `tagged` | best BLEU/TERM_ACC among V3 variants, but FCR rises under noisy 10k glossary |

The table above is a summary only; metric source of truth is W&B plus the
generated TSVs in the event output directory.
