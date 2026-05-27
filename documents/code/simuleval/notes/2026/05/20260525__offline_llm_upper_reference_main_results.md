## Hypothesis

Adding Siqi's offline full-context LLM outputs gives reviewer-facing horizontal references for the ACL tagged raw and medicine hardraw main-result figures.

## Background / Motivation

The offline directory at `/mnt/data/siqiouyang/runs/infinisst_rag/offline` contains `baseline` outputs for full-context decoding and `glossary` outputs where oracle glossary terms are supplied. These should be parsed into the canonical main-result TSV instead of leaving medicine offline rows as missing placeholders.

## What changed vs baseline

- Parsed BLEU from each offline `scores.tsv`.
- Recomputed TERM_ACC against the fixed ACL tagged raw and medicine hardraw glossaries using the same source-term and target-substring rule as `stream_laal_term.py`.
- Added `Offline + GT terms` as a second no-latency horizontal-reference method.
- Regenerated `new_main_result_tagged.pdf` and `medicine_main_result.pdf`.
- Updated the paper captions and main-results discussion to describe the horizontal offline references.

## Expected metrics

The TSV should have unique `(dataset, method, lang, lm)` keys. Offline rows should have `lm=NA`, `StreamLAAL=NA`, and `StreamLAAL_CA=NA`, because they are full-context references plotted as horizontal lines.

## Verdict

Success. The canonical TSV and both main-result figures were regenerated from the updated script. The paper text now describes the two offline horizontal references and no longer says medicine offline rows are unavailable.
