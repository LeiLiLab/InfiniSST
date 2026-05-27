## Hypothesis

Using glossary-derived oracle term maps with a fixed medicine-GT denominator should provide the clean Panel A upper-bound readout for whether the all-GT SFT Speech LLM can use terminology when the relevant terms are explicitly available.

## Background / Motivation

The earlier medicine oracle-GT readout used ESO `sentence.terms` to construct oracle term maps.  Manual inspection of sample `596001` showed that `sentence.terms` misses valid glossary terms such as `dose-volume histogram` and `patient model`.  Panel A should therefore use the glossary as the source of truth and keep the denominator fixed to curated `medicine_gt` terms, so the table measures model behavior rather than changes in glossary coverage.

## What changed vs baseline

- Baseline run family: `speech_llm_oracle_gt_sft_readout`
- Baseline behavior: oracle term map from ESO `sentence.terms`, strict per-sample translated glossary.
- New behavior: `TERM_SOURCE=glossary_match`; oracle and metric glossary are both derived by source/reference matching against the translated medicine glossary.
- Fixed denominator: `GLOSSARY_SOURCE_FILTER=medicine_gt`.
- Samples: ESO medicine `404`, `596001`, `606`, `545006`.
- Latency multipliers: `1 2 3 4`.
- Model: `/mnt/gemini/data2/jiaxuanluo/speech_llm_oracle_gt_termmap_zh_r32a64_taurus4/keep1.0_r32/v1-20260519-105111-hf`.

## Expected metrics

TERM totals should increase for cases where `sentence.terms` missed valid glossary terms, especially sample `596001`.  TERM_ACC may decrease relative to the legacy oracle table because the corrected denominator includes previously unexposed terms.  BLEU and StreamLAAL should be comparable to the legacy all-GT SFT readout because the model and generation settings are unchanged.

## Verdict

Completed and corrected.  Four-sample Panel A fixed-GT oracle readout finished for `lm=1,2,3,4`; aggregate metrics are in `/mnt/gemini/data2/jiaxuanluo/medicine4_panelA_fixedgt_oraclegt_sft_r32a64_lm_sweep_20260519/zh/summary_lm_sweep.tsv`.  The corrected W&B aggregate run is `simuleval_eval/w1wlbv3j`; the earlier aggregate run `simuleval_eval/j225eij7` used the wrong FCR policy and is superseded.

Metric fix: Panel A `TERM_FCR` now uses the explicit sentence-level oracle term_map as the candidate set (`term_map_source_ref_negative_sentence`).  It no longer uses full fixed-glossary source/ref negatives or runtime prompt lookback terms as false-copy candidates.
