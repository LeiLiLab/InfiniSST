# Medicine remaining oracle GT SimulEval: all-GT SFT r32a64

## Hypothesis

The all-GT term_map SFT checkpoint should preserve the one-talk oracle gains on the remaining ESO medicine samples, with true false-copy rate lower than the zero-shot origin_bsz4 baseline.

## Background / Motivation

Sample 404 showed that the origin_bsz4 baseline can read oracle term maps zero-shot, but the all-GT SFT checkpoint reduces sentence-level false-copy behavior.  The FCR definition has been tightened to count only true false copies: the source sentence does not contain the term, the reference does not contain the target translation, and the hypothesis contains the target translation.

## What changed vs baseline

- Baseline readout: origin_bsz4 zero-shot oracle term_map on medicine sample 404.
- SFT checkpoint: `/mnt/gemini/data2/jiaxuanluo/speech_llm_oracle_gt_termmap_zh_r32a64_taurus4/keep1.0_r32/v1-20260519-105111-hf`.
- Eval scope: remaining strict MFA-only medicine samples `596001`, `606`, and `545006`.
- Term map: sentence-aligned oracle GT terms from strict medicine JSONL/glossary.
- Metric update: `TERM_FCR` now uses `term_map_sentence_true_false_copy`.

## Expected metrics

Expect TERM_ACC and REAL_TERM_ADOPT to remain close to the sample-404 oracle readout.  True TERM_FCR should be lower than the old strict term-map FCR because supported source terms and reference paraphrases are no longer counted as false-copy negatives.

## Verdict

Pending.
