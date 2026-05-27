# Tagged ACL LLM-Generated Term-Map SFT Full Sweep

## Hypothesis

The earlier LLM-generated term-map SFT checkpoints may preserve more term adoption
than the new V3 robustness-first SFT variants.  Run the same tagged ACL
`tau=0.73` pipeline as the origin-bsz4 sweep, replacing only the speech LLM
checkpoint for each language.

## Background / Motivation

The V3 real-retriever SFT probe underperformed the no-term-map-SFT origin model
on the main tagged ACL TERM_ACC readouts.  Before abandoning term-map SFT, this
event evaluates the older LLM-generated term-map SFT checkpoints on the full
main grid.

## What changed vs baseline

- Baseline launcher:
  `documents/code/simuleval/launchers/2026/05/20260520__tagged_acl_origin_bsz4_tau073_sweep_taurus45269.sh`
- Diff: use language-specific LLM-generated term-map SFT HF checkpoints:
  - zh: `/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-zh-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4`
  - ja: `/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-ja-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4`
  - de: `/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-de-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4`
- Retriever and eval protocol remain the tagged ACL `lh1b88kw`, `tau=0.73`,
  timeline lookback `1.92s` setup.
- Full grid: `zh ja de` x `lm=1 2 3 4` x `raw gs1k gs10k`.

## Expected metrics

Track BLEU, TERM_ACC, REAL_TERM_ADOPT, TERM_FCR, and StreamLAAL.  The main
comparison is against the origin-bsz4 full sweep, especially the known sentinel
settings `zh lm2 raw`, `de lm3 raw`, and `ja lm1 gs10k`.

## Verdict

Success.  The full 36-setting grid completed and was logged to W&B family
`tagged_acl_llmgen_bsz4_tau073`; SQLite sync completed for all 36 runs.

Summary artifacts:

- `/mnt/aries/data7/jiaxuanluo/slm/tagged_acl_llmgen_sft_tau073_full_20260521T050150/__summary__/summary_metrics_llmgen_sft_full.tsv`
- `/mnt/aries/data7/jiaxuanluo/slm/tagged_acl_llmgen_sft_tau073_full_20260521T050150/__summary__/summary_metrics_llmgen_sft_full.md`

One setting, `zh lm2 gs10k`, finished locally but did not create its W&B run
during the original wrapper execution, so it was backfilled from its completed
`eval_results.tsv` using `documents/code/offline_evaluation/wandb_eval_logger.py`
as W&B run `c0818g3a`.
