# Medicine lm2-4 oracle GT SimulEval: all-GT SFT r32a64

## Hypothesis

Increasing the latency multiplier should trade latency for translation quality on the strict medicine oracle term-map readout.  Because this is all-GT term_map mode, the run measures the Speech LLM's ability to use correct terminology evidence rather than retriever recall.

## Background / Motivation

The lm2 readout over four ESO medicine samples is complete.  We need the same reusable pipeline for lm1, lm2, lm3, and lm4 so BLEU is computed as corpus BLEU over the four talks, while term metrics are pooled from the per-talk sentence-level outputs.

## What changed vs baseline

- Baseline run scope: existing lm2 medicine oracle-GT SFT readout.
- SFT checkpoint: `/mnt/gemini/data2/jiaxuanluo/speech_llm_oracle_gt_termmap_zh_r32a64_taurus4/keep1.0_r32/v1-20260519-105111-hf`.
- Samples: `404`, `596001`, `606`, `545006`.
- Latency multipliers: `1`, `2`, `3`, `4`.
- Term map: sentence-aligned oracle GT terms from strict medicine JSONL/glossary.
- Aggregation: corpus BLEU/StreamLAAL from concatenated instances; TERM_ACC, REAL_TERM_ADOPT, and TERM_FCR pooled from per-talk count columns.

## Expected metrics

Expect lm3/lm4 to improve BLEU and possibly TERM_ACC relative to lm2, with higher StreamLAAL.  True TERM_FCR should remain in the same range if the model is not over-copying oracle terms.

## Verdict

Pending.
