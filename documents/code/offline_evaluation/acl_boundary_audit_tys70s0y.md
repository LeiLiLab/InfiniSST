# ACL Boundary Audit for `tys70s0y`

Goal: inspect ACL6060 `gs10000` boundary cases for the `tys70s0y` best-secondary
checkpoint before changing TCM, and answer whether top-10 near-threshold
non-GT candidates are mostly false negatives / very similar terms or genuine
noise.

## Script and outputs

- Audit script: `documents/code/offline_evaluation/audit_acl_boundary_samples.py`
- Model: `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_clean_gc_wr1000k_m0.0_maxsim_mfa_variantE_hardneg_ps_k512_tcm_ep3_cold_smallest_dense_normAGGR_6gpu_best_acl6060_gs10000.pt`
- Output dir: `/mnt/gemini/data2/jiaxuanluo/acl_boundary_audit/tys70s0y_secondary_gs10000`

Main artifacts:

- Full ACL dump: `/mnt/gemini/data2/jiaxuanluo/acl_boundary_audit/tys70s0y_secondary_gs10000/acl6060_gs10000_top10_dump.jsonl`
- Boundary subset: `/mnt/gemini/data2/jiaxuanluo/acl_boundary_audit/tys70s0y_secondary_gs10000/acl6060_boundary_samples.jsonl`
- Boundary TSV: `/mnt/gemini/data2/jiaxuanluo/acl_boundary_audit/tys70s0y_secondary_gs10000/acl6060_boundary_samples.tsv`
- Summary JSON: `/mnt/gemini/data2/jiaxuanluo/acl_boundary_audit/tys70s0y_secondary_gs10000/acl6060_boundary_summary.json`
- Representative samples: `/mnt/gemini/data2/jiaxuanluo/acl_boundary_audit/tys70s0y_secondary_gs10000/acl6060_boundary_report.md`

Audit settings:

- Boundary band: top non-GT score in `[0.75, 0.85]`
- Reference tau: `0.80`
- Same retriever config as `tys70s0y` / `run_hist_tys70s0y_secondary_aries.sh`:
  `temperature=0.07`, MaxSim windows `2 3 4 5 6 7 8 10 12 16 20 24`,
  `pooling=transformer`, `sparse_weight=0.0`

## Headline counts

From `acl6060_boundary_summary.json`:

- ACL chunks: `3583` total, `605` with GT term
- Boundary rows: `397`
- Boundary auto labels:
  - `likely_false_negative`: `6`
  - `very_similar_term`: `68`
  - `clear_noise`: `323`

Subgroup breakdown:

- `gt_missing_or_outranked`: `105` rows
  - `clear_noise`: `89`
  - `very_similar_term`: `16`
  - `likely_false_negative`: `0`
- `gt_in_top10_near_tau`: `76` rows
  - `clear_noise`: `63`
  - `very_similar_term`: `13`
- `other_boundary`: `266` rows
  - `clear_noise`: `211`
  - `very_similar_term`: `49`
  - `likely_false_negative`: `6`

## What the samples show

### 1. True false-negative / alias-collision cases exist, but they are rare

The only recurring clean false-negative family in this audit was:

- `Transformer` vs `transformer encoder`

This accounts for all `6` `likely_false_negative` rows in the current
heuristic pass. These examples usually still have GT at rank 1, so they are
not the dominant source of `gt_missing_or_outranked` failures.

### 2. There is a real but minority band of semantically near terms

Representative families in the `very_similar_term` bucket:

- `morphological` vs `morphology`
- `query` vs `Conjunctive query`
- `Language model` vs `Large language model` / `Language module`
- `Transformer` vs `Vision transformer`
- `F1 score` vs `CLEVER score`
- `reference` vs `Reference frame`

These are real semantic-neighbor or term-family competitors, but they are not
the majority of the boundary set.

### 3. Most ACL boundary negatives are still plain noise

The important result is the `gt_missing_or_outranked` subset:

- `89 / 105` (`84.8%`) are tagged `clear_noise`
- `16 / 105` (`15.2%`) are tagged `very_similar_term`
- `0 / 105` are `likely_false_negative`

So the cases where the model is actually at risk of dropping recall after
thresholding are mostly not explained by mislabeled negatives or trivial alias
collisions.

Representative clear-noise examples from the audit include:

- `counterfactual` vs `Artificial`
- `utterance` vs `AUTOSAR`
- `task-oriented dialogue` vs `InterACT`
- `GLUE benchmark` vs `Max Tegmark`
- `RGF` vs `How Data Happened`

## Interpretation for the TCM discussion

This audit weakens the hypothesis that ACL `gs10000` boundary failures are
mostly fake negatives from a dense in-domain glossary.

What it supports instead:

1. There is a small alias/near-neighbor slice that a smarter candidate-aware
   treatment could help.
2. But the dominant mass of boundary mistakes is still genuine high-scoring
   noise.
3. Therefore, it still makes sense to discuss score calibration / threshold
   shaping, because the main problem has not disappeared into label noise.

What it does **not** support:

- "Most `gt_missing_or_outranked` ACL boundary negatives are really false
  negatives."

That statement is not consistent with the current sample audit.

## Caveats

- Heuristics are intentionally lightweight and conservative.
- `small_gap_to_gt` is recorded as a flag, but it is not enough by itself to
  call something "very similar".
- Some semantic-neighbor cases may still be undercounted, but the gap is too
  large to reverse the main conclusion: the boundary set is mostly noise, not
  aliases.
