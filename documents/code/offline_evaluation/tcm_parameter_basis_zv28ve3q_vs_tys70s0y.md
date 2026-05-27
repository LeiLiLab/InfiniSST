# TCM Parameter Basis: `zv28ve3q` vs `tys70s0y`

Compare the two ACL6060 `gs10000` boundary audits to decide which checkpoint is
the better basis for choosing TCM thresholds / weights.

Inputs:

- `tys70s0y` audit: `documents/code/offline_evaluation/acl_boundary_audit_tys70s0y.md`
- `zv28ve3q` audit: `documents/code/offline_evaluation/acl_boundary_audit_zv28ve3q.md`
- Machine-readable comparison:
  `/mnt/gemini/data2/jiaxuanluo/acl_boundary_audit/zv28ve3q_vs_tys70s0y_tau0p80.{json,md}`

## Same-tau comparison (`tau=0.80`)

Locked-tau metrics from the two dense sweeps:

| model | filtered recall | micro precision | no-term noise |
|---|---:|---:|---:|
| `tys70s0y` | `0.7868` | `0.2042` | `2.25` |
| `zv28ve3q` | `0.7835` | `0.1866` | `2.62` |

Boundary-label totals:

| model | boundary rows | clear noise | very similar | likely false negative |
|---|---:|---:|---:|---:|
| `tys70s0y` | `397` | `323` | `68` | `6` |
| `zv28ve3q` | `381` | `326` | `53` | `2` |

Critical subgroup `gt_missing_or_outranked`:

| model | total rows | clear noise | very similar | likely false negative |
|---|---:|---:|---:|---:|
| `tys70s0y` | `105` | `89` | `16` | `0` |
| `zv28ve3q` | `104` | `95` | `9` | `0` |

## Decision

`tys70s0y` is the better basis for TCM parameter selection.

Why:

1. At the same deployment tau, it has slightly better filtered recall and
   materially lower no-term noise.
2. Its boundary failures are still mostly real noise, but it retains a larger
   "very similar term" slice (`16` vs `9` in `gt_missing_or_outranked`), which
   is more informative for deciding how aggressively a candidate-aware TCM
   should shape near-top competitors.
3. It is also the direct baseline for the current TCM-v2 branch (`E1` / `E2`),
   so calibrating from `tys70s0y` avoids switching to a different HN regime
   (`pool k=64` instead of per-sample `k=512`) while tuning the TCM knobs.

## What `zv28ve3q` still tells us

`zv28ve3q` is still useful as a sanity-check model:

- it confirms that even the pool-HN checkpoint is not dominated by false
  negatives at the ACL boundary,
- and it reinforces the conclusion that the main problem remains genuine
  high-scoring noise.

But it is *less* suitable as the primary TCM-parameter reference checkpoint,
because its boundary set is more noise-dominated and less representative of the
current training recipe we are actively modifying.
