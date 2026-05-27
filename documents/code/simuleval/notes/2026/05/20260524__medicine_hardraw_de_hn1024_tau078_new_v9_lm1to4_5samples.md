## Hypothesis

The de New V9 speech LLM should improve hard-medicine strict-term adoption when paired with the HN1024 retriever at tau 0.78 under lm 1, 2, 3, and 4.

## Background / Motivation

This is the de counterpart of the zh New V9 hard-medicine raw strict-glossary readout.  It uses five medicine samples and keeps the runtime/eval glossary fixed to the manually checked hard medicine glossary.

## What changed vs baseline

- Speech LLM: de New V9 assistant-term-tag delay/no-gt-zero old-new-v3 r32/a64 HF export.
- Retriever: HN1024 `lh1b88kw`, tau 0.78, top-k 10, timeline mode with 1.92s lookback.
- Glossary: `/mnt/gemini/data1/jiaxuanluo/medicine_hard_manual_glossary_streamlaal_20260524.json`.
- Evaluation: lm 1, 2, 3, and 4; five samples per lm; output-side `<term>` tags stripped before scoring.

## Expected metrics

Report BLEU, TERM_ACC, REAL_TERM_ADOPT, TERM_FCR, and StreamLAAL for each lm on the combined five-sample de medicine set.

## Verdict

Running.
