## Hypothesis

The En-De ACL tagged raw RASST main-result curve should use the selected reruns produced under the new cache-chunks max strategy for lm=1, lm=2, and lm=4.

## Background / Motivation

The user selected the lm=1 promptfix audioauto/cache30 row, the lm=2 old serial cache30/max40lm row, and the lm=4 promptfix audioauto/cache30 row for the main result. The shared rationale is to report the rerun points under the new cache chunks max strategy.

## What changed vs baseline

- Updated `acl_tagged_raw / RASST / de / lm=1` from the previous serial row to the promptfix audioauto/cache30 row.
- Kept `acl_tagged_raw / RASST / de / lm=2` on the old serial cache30/max40lm row.
- Kept `acl_tagged_raw / RASST / de / lm=4` on the promptfix audioauto/cache30 row.
- Updated the TSV note/reason for de lm=1, lm=2, and lm=4 to start with: `rerun using new cache chunks max strategy`.
- Refreshed the paper-local Figure 1 data snapshot and regenerated `new_main_result_tagged.pdf/png`.

## Expected metrics

- de lm=1: BLEU 26.4180, StreamLAAL 1026.3858, StreamLAAL_CA 1532.6696, TERM_ACC 0.8107 (758/935).
- de lm=2: BLEU 31.5758, StreamLAAL 1666.9229, StreamLAAL_CA 2414.3860, TERM_ACC 0.8299 (776/935).
- de lm=4: BLEU 34.7098, StreamLAAL 2702.2864, StreamLAAL_CA 3933.0219, TERM_ACC 0.8481 (793/935).

## Verdict

Success. The canonical main-result TSV, Figure 1 paper-local data snapshot, and paper-facing `new_main_result_tagged.pdf/png` were updated. TSV uniqueness and numeric validation passed for 87 rows.
