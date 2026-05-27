# ACL Boundary Audit for `zv28ve3q`

Goal: rerun the same ACL6060 `gs10000` boundary-sample audit used for
`tys70s0y`, but on the `zv28ve3q` best-secondary checkpoint (the same snapshot
anchored at the `best_secondary/step=1240` family), then compare whether this
older pool-HN model is a better basis for TCM threshold selection.

## Script and outputs

- Audit script: `documents/code/offline_evaluation/audit_acl_boundary_samples.py`
- Dense tau sweep helper: `documents/code/offline_evaluation/sweep_tau_from_top10_dump.py`
- Model:
  `/mnt/gemini/home/jiaxuanluo/train_outputs/snapshots/20260422_q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_clean_gc_wr1000k_m0.0_maxsim_mfa_variantE_hardneg_k64_tcm_ep3_cold_smallest_dense_normAGGR_best_acl6060_gs10000.pt`
- Output dir:
  `/mnt/gemini/data2/jiaxuanluo/acl_boundary_audit/zv28ve3q_secondary_gs10000`

Main artifacts:

- Full ACL dump:
  `/mnt/gemini/data2/jiaxuanluo/acl_boundary_audit/zv28ve3q_secondary_gs10000/acl6060_gs10000_top10_dump.jsonl`
- Boundary subset:
  `/mnt/gemini/data2/jiaxuanluo/acl_boundary_audit/zv28ve3q_secondary_gs10000/acl6060_boundary_samples.jsonl`
- Boundary TSV:
  `/mnt/gemini/data2/jiaxuanluo/acl_boundary_audit/zv28ve3q_secondary_gs10000/acl6060_boundary_samples.tsv`
- Summary JSON:
  `/mnt/gemini/data2/jiaxuanluo/acl_boundary_audit/zv28ve3q_secondary_gs10000/acl6060_boundary_summary.json`
- Representative samples:
  `/mnt/gemini/data2/jiaxuanluo/acl_boundary_audit/zv28ve3q_secondary_gs10000/acl6060_boundary_report.md`
- Dense tau sweep TSV:
  `/mnt/gemini/data2/jiaxuanluo/acl_boundary_audit/zv28ve3q_secondary_gs10000/dense_tau_sweep.tsv`
- Dense tau sweep JSON:
  `/mnt/gemini/data2/jiaxuanluo/acl_boundary_audit/zv28ve3q_secondary_gs10000/dense_tau_sweep.json`

Audit settings:

- Boundary band: top non-GT score in `[0.75, 0.85]`
- Reference tau: `0.80`
- Same retriever config family as the previous audit:
  `temperature=0.07`, MaxSim windows `2 3 4 5 6 7 8 10 12 16 20 24`,
  `pooling=transformer`, `sparse_weight=0.0`

## Headline counts

From `acl6060_boundary_summary.json`:

- ACL chunks: `3583` total, `605` with GT term
- Boundary rows: `381`
- Boundary auto labels:
  - `likely_false_negative`: `2`
  - `very_similar_term`: `53`
  - `clear_noise`: `326`

Subgroup breakdown:

- `gt_missing_or_outranked`: `104` rows
  - `clear_noise`: `95`
  - `very_similar_term`: `9`
  - `likely_false_negative`: `0`
- `gt_in_top10_near_tau`: `62` rows
  - `clear_noise`: `52`
  - `very_similar_term`: `10`
- `other_boundary`: `257` rows
  - `clear_noise`: `214`
  - `very_similar_term`: `41`
  - `likely_false_negative`: `2`

## Dense tau sweep

At the same local grid `0.72..0.88`, the dense tau sweep again keeps
`tau*=0.80` under the "no more than 0.005 filtered-recall drop" rule:

- `tau=0.80`: filtered recall `0.7835`, micro precision `0.1866`,
  no-term noise `2.62`
- `tau=0.82`: noise improves to `1.55`, but filtered recall drops to `0.7256`
  which is too large a loss under the selected rule

So `zv28ve3q` does not expose a new better fixed deployment tau either.

## Interpretation

The qualitative picture is the same as `tys70s0y`, but slightly *more*
noise-dominated:

1. False negatives are even rarer here (`2` vs `6`).
2. The `gt_missing_or_outranked` subset is more noise-heavy:
   `95 / 104 = 91.3%` clear noise.
3. The "very similar term" slice is smaller than in `tys70s0y`
   (`53` vs `68` overall; `9` vs `16` in `gt_missing_or_outranked`).

That means `zv28ve3q` does **not** strengthen the case for tuning TCM from
alias-heavy / false-negative-heavy evidence. If anything, it pushes the story
further toward "the boundary problem is genuine high-scoring noise."
