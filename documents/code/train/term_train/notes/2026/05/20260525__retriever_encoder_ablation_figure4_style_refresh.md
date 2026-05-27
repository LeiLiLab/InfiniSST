## Hypothesis

Figure 4 should match the paper's restrained plotting style and avoid unexplained abbreviations.

## Background / Motivation

The previous retriever encoder ablation used saturated colors and the labels `GS-10k` and `GS-100k`, which were too visually loud and unclear in the paper figure.

## What changed vs baseline

- Replotted the encoder ablation with a restrained blue/gray palette and line-style differences.
- Replaced the plotted bank labels with `GigaSpeech-10k` and `GigaSpeech-100k`.
- Replaced the ambiguous legend label `Main` with the full encoder name `Qwen3-Omni-AuT + BGE-M3`.
- Updated the Figure 4 caption and discussion to state that the GigaSpeech-expanded banks add distractor terms only to the runtime retrieval candidate bank while keeping the development raw-glossary denominator fixed.
- Expanded the later glossary-bank ablation text from `GS-*` to `GigaSpeech-*` for consistency.

## Expected metrics

No metric values should change. The cached TSV remains the input truth for the plot.

## Verdict

Success. Figure 4 was regenerated from the cached TSV and copied into the paper figure directory.
