## Hypothesis

Replacing the medicine hardraw En-De RASST lm=1,2,4 rows with verified serial promptfix readouts gives the paper figure a stronger BLEU/latency tradeoff while keeping the same hardraw terminology denominator.

## Background / Motivation

The previous medicine hardraw En-De rows came from batch readouts. New serial promptfix cache30/max40lm readouts finished for lm=1,2,4 and were selected for the canonical main-result table. The lm=3 batch row remains unchanged.

## What changed vs baseline

- Updated `medicine_hardraw / RASST / de / lm=1,2,4` in the canonical TSV.
- Synced the paper-local figure data snapshots.
- Regenerated `medicine_main_result.pdf` and `medicine_main_result.png`.

## Expected metrics

- lm=1: BLEU 22.6187, StreamLAAL 1023.7187, TERM_ACC 0.7264.
- lm=2: BLEU 26.7696, StreamLAAL 1838.9518, TERM_ACC 0.7821.
- lm=4: BLEU 28.9154, StreamLAAL 2782.5832, TERM_ACC 0.8238.

## Verdict

Success. TSV uniqueness and updated-row numeric validation passed for the canonical TSV and both paper-local data snapshots. The paper medicine figure was regenerated and copied into `latex/figures/`.
