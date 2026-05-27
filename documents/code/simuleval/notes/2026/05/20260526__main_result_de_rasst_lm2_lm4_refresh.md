## Hypothesis

The En-De ACL tagged raw RASST main-result curve should use the verified serial lm=2 readout and the verified promptfix audioauto/cache30 lm=4 readout selected by the user.

## Background / Motivation

The de lm=2 batch row underperformed an older verified serial readout in BLEU, while the de lm=4 promptfix audioauto/cache30 serial readout improved BLEU over the previous main-result row. The user requested replacing exactly these two main-result points.

## What changed vs baseline

- Updated `acl_tagged_raw / RASST / de / lm=2` from the batch row to the old serial cache30/max40lm row.
- Updated `acl_tagged_raw / RASST / de / lm=4` from the batch row to the serial promptfix audioauto/cache30 row.
- Left de lm=1 and lm=3 unchanged.
- Refreshed the paper-local Figure 1 data snapshot and regenerated `new_main_result_tagged.pdf/png`.

## Expected metrics

- de lm=2: BLEU 31.5758, StreamLAAL 1666.9229, StreamLAAL_CA 2414.3860, TERM_ACC 0.8299 (776/935).
- de lm=4: BLEU 34.7098, StreamLAAL 2702.2864, StreamLAAL_CA 3933.0219, TERM_ACC 0.8481 (793/935).

## Verdict

Success. The canonical main-result TSV, Figure 1 paper-local data snapshot, and paper-facing `new_main_result_tagged.pdf/png` were updated. TSV uniqueness and numeric validation passed for 87 rows.
