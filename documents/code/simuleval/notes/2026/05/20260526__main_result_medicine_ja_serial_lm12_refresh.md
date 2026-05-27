## Hypothesis

For JA medicine hardraw, the verified serial promptfix readouts for lm=1 and
lm=2 improve BLEU over the prior batch rows while preserving the RASST
terminology advantage.  Updating only these two rows keeps lm=3 and lm=4 on
their currently verified batch outputs because no matching serial medicine
artifacts are available yet.

## Background / Motivation

The canonical main-result TSV and the paper-local medicine figure package were
using JA medicine batch RASST rows from the 20260525 Taurus sweep.  Later Aries
serial readouts completed for lm=1 and lm=2 with five `instances.log` rows and
five `instances.strip_term.log` rows each.

## What changed vs baseline

- Replaced `medicine_hardraw / RASST / ja / lm=1` with the serial promptfix
  vllm-audio-128/cache30/max40lm row.
- Replaced `medicine_hardraw / RASST / ja / lm=2` with the serial promptfix
  vllm-audio-128/cache30/max40lm row.
- Left JA lm=3 and lm=4 unchanged because only the previous batch artifacts are
  currently verified for medicine hardraw.
- Regenerated `medicine_main_result.pdf` and `.png` from the paper-local figure
  package.

## Expected metrics

JA medicine RASST lm=1 should become BLEU 19.3213, TERM_ACC 0.7515.
JA medicine RASST lm=2 should become BLEU 25.1412, TERM_ACC 0.8086.

## Verdict

Completed.  The canonical TSV, paper-local figure data snapshot, and paper
figure artifacts were updated.  TSV uniqueness and numeric parsing validation
passed for the canonical TSV and both main-result figure snapshots.
