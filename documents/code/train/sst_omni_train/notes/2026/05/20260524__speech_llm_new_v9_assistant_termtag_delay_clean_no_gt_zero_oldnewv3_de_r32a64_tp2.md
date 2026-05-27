# New V9 assistant term-tag delay-clean data, de

## Hypothesis

The accepted zh New V9 recipe should transfer to German if the de training data follows the same old-new_v3 -> LLM variant -> no-GT-zero -> assistant term-tag lineage, with stricter tag safety for Latin-script text.

## Background / Motivation

The zh New V9 model became the current main Speech LLM line because assistant-side term tags improved terminology adoption after metric-time tag stripping.  The de legacy SFT JSONL did not contain `gt_terms_by_chunk`, so the de data was first rebuilt into an old-new_v3-equivalent form by deriving GT terms from existing term maps and future assistant support, then applying the same New V4/New V5/New V9 stages.

## What changed vs baseline

- Language: German.
- Data source: `/mnt/gemini/data1/jiaxuanluo/speech_llm_new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3_de_20260524`.
- Training JSONL: `train_s_de_new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3.jsonl`.
- Dev JSONL: `dev_s_de_new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3_first355.jsonl`.
- LoRA: rank 32, alpha 64.
- Parallelism: 4 GPUs, tensor parallel 2, expert parallel 4.
- Safety fix: assistant tag rewriting now rejects nested tag regions and rewrite spans that split Latin-letter or digit boundaries.

## Expected metrics

Primary downstream readouts should improve or stabilize German terminology adoption relative to no term-map SFT, especially in raw/sparse-term starts.  Metric scripts must strip `<term>` and `</term>` before BLEU, TERM_ACC, REAL_ADOPT, FCR, and StreamLAAL.

## Verdict

Pending.
