## Hypothesis

The new ACL6060 DE/JA InfiniSST baseline outputs under
`documents/code/train/term_train/reports/figures/acl6060_de_ja_results` should
replace the older user-supplied DE/JA baseline rows in the canonical main-result
TSV.

## Background / Motivation

The paper Figure 1 uses `acl_tagged_raw` main-result rows.  The previous DE/JA
InfiniSST rows were mostly prompt-supplied reusable values.  The new folders
contain talk-level `instances.log` files for `seg960`, `seg1920`, `seg2880`, and
`seg3840`, corresponding to `lm=1..4`.

## What changed vs baseline

Post-evaluated the new DE/JA `instances.log` files with
`offline_streamlaal_eval.py --mode acl6060` using the fixed ACL6060 source/ref
files and `acl6060_tagged_gt_raw_min_norm2.json`.  Replaced only
`dataset=acl_tagged_raw, method=InfiniSST, lang in {de,ja}, lm=1..4`.

## Expected metrics

DE/JA rows should keep the tagged raw denominators: DE `TERM_TOTAL=935`, JA
`TERM_TOTAL=940`.  Figure 1 should refresh the InfiniSST baseline curves only.

## Verdict

Completed.  The canonical TSV, Figure 1 package TSV, and paper figure PDF/PNG
were refreshed from post-evaluated artifacts.
