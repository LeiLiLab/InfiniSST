## Hypothesis

Clean de/ja Speech LLM SFT should follow the final zh lineage but rebuild GT from MFA source evidence instead of polluted term_map-derived labels.

## Background / Motivation

The previous de/ja New V9 branch used `derive_gt_terms_from_termmap_matches.py` plus fuzzy/local wrapping, which can create bad GT terms and assistant tags. This branch rejects that data as polluted and uses GigaSpeech MFA word timestamps as the source-side evidence.

## What changed vs baseline

- Input remains the raw de/ja SFT JSONL.
- Legacy `term_map` content is cleared before GT construction.
- Source candidates come from MFA word n-gram exact matches against source-only wiki100k.
- The de/ja TSV is used only to map `audio_clips_siqi_*` ids to GigaSpeech `opus:start:n_frames`; source text still comes from MFA TextGrid words.
- OpenAI is used once per candidate to return an exact assistant `reference_span` and an uncommon target translation; the span is replaced directly.
- old-new_v3 TCM retriever (`tau=0.75`, density 9, cap 20) supplies term_map filler only; GT labels come only from MFA+OpenAI.
- no-GT chunks are zeroed to `term_map:NONE`.
- assistant target translations are wrapped with `<term>...</term>` using exact wrap plus adjacent-boundary-only repair.

## Expected metrics

This is expected to improve de/ja realAdopt relative to the rejected polluted de/ja New V9 branch and reduce BLEU degradation from source==target or malformed tags.

## Verdict

Pending data-prep validation and train/eval.
