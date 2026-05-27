# no-HN vs HN1024/HN256 Fixed-Denominator Retriever Eval

Date: 2026-05-22; HN256 latest checkpoint updated 2026-05-23

This report supersedes the earlier legacy evals that mixed retriever glossary
size with the metrics denominator.  All runs here use
`eval_metric_denominator=fixed_raw`: within each eval domain, that domain's
strict raw/base term universe stays as the recall/precision denominator, while
gs10k / gs100k / gs1M only change the retriever candidate bank.

Important denominator caveat: fixed raw is per domain here, not one shared
ACL6060 denominator across paper ACL, tagged ACL, and medicine.  For example,
HN256 latest run `8h9q0v4t` logs `fixed_metric_denominator=1`, but its raw
metrics universes differ: paper ACL has `metrics_bank_terms=97`, tagged ACL has
`metrics_bank_terms=238`, and medicine has `metrics_bank_terms=570`.  The
tagged ACL numbers are therefore fixed-denominator within tagged ACL, but they
are not using the paper ACL 97-term glossary as the denominator.  This likely
explains why tagged ACL recall is consistently higher than the other two
domains.

## Runs

| role | W&B run | checkpoint | scope |
|---|---|---|---|
| no-HN tau 0.61 held-out readout | `zji769ve` | `40fgbr2y` best-secondary, step 1600 | dev raw/10k/100k + ACL/tagged/medicine raw/1k/10k, single tau `0.61` |
| no-HN held-out readout | `9esujv2w` | `40fgbr2y` best-secondary, step 1600 | dev raw/10k/100k + ACL/tagged/medicine raw/1k/10k |
| HN1024 held-out readout | `ry8osg4u` | `lh1b88kw` best-secondary, step 2640 | dev raw/10k/100k + ACL/tagged/medicine raw/1k/10k |
| no-HN low-tau dev-1M sweep | `e8t8zdtd` | `40fgbr2y` best-secondary, step 1600 | dev raw/10k/100k/1M only, tau `0.50..0.70` |
| no-HN dev-1M sweep | `evcgcdlu` | `40fgbr2y` best-secondary, step 1600 | dev raw/10k/100k/1M only |
| HN1024 dev-1M sweep | `31xmxmdp` | `lh1b88kw` best-secondary, step 2640 | dev raw/10k/100k/1M only |
| HN256 held-out readout | `8h9q0v4t` | `gsjheh6r` latest, step 1440 | dev raw/10k/100k + ACL/tagged/medicine raw/1k/10k, tau `0.50..0.90` |

Selection rule: choose the highest tau whose max dev recall drop is below the
given budget.  no-HN now has a combined dev-only tau surface from `0.50..0.90`
at stride `0.01`; HN1024 remains `0.65..0.90` at stride `0.01`; HN256 uses
`0.50..0.90` at stride `0.01` on dev raw/10k/100k only.  Tau `0.0` means
unfiltered `recall@10`.  When combining no-HN runs, each drop is computed
against that run's own unfiltered recall; the low/high sweeps differ by only
about `0.008 pp` on raw recall.

Reviewer-facing interpretation of the budgets:

- `<0.5 pp` is the recall-first operating point.  This is the strict setting to
  use when missing a true term is considered much worse than showing a few extra
  terms to the downstream speech LLM.
- `<1.0 pp` and `<1.5 pp` are sensitivity points, not post-hoc alternatives.
  They show whether HN's precision/thresholdability benefit is stable when the
  system is allowed to spend more recall for cleaner candidates.
- The main decision should therefore be stated conditionally: no-HN is the
  recall-first gs10k choice; HN1024 is the precision-oriented / smaller-bank
  choice.  This avoids claiming one global winner from a single tau budget.

## Dev-1M Raw Recall

Unfiltered recall is not higher for HN1024.  The gap is small until gs1M, where
HN1024 is about `0.21 pp` lower.

| run | raw | gs10k | gs100k | gs1M |
|---|---:|---:|---:|---:|
| no-HN `e8t8zdtd` low-tau | 99.2518 | 98.9653 | 98.5912 | 97.9863 |
| no-HN `evcgcdlu` | 99.2439 | 98.9812 | 98.5833 | 97.9943 |
| HN1024 `31xmxmdp` | 99.2120 | 98.9573 | 98.5753 | 97.7794 |
| HN - no-HN `evcgcdlu` | -0.0319 | -0.0239 | -0.0080 | -0.2149 |

## Raw-Included Selection

This is the strictest calibration surface: raw/base, gs10k, and gs100k all
count toward max dev drop.  Adding gs1M does not change the selected tau here
because the raw/base drop dominates.

| run | budget | tau | max drop | raw drop | gs10k drop | gs100k drop | gs1M drop |
|---|---:|---:|---:|---:|---:|---:|---:|
| no-HN `e8t8zdtd` | <0.5 | 0.61 | 0.4537 | 0.4537 | 0.2308 | 0.0876 | 0.0318 |
| no-HN `e8t8zdtd` | <1.0 | 0.69 | 0.9869 | 0.9869 | 0.7004 | 0.3661 | 0.1035 |
| no-HN `evcgcdlu` | <1.5 | 0.73 | 1.4804 | 1.4804 | 1.2178 | 0.8278 | 0.4696 |
| HN1024 `31xmxmdp` | <0.5 | 0.72 | 0.4378 | 0.4378 | 0.2229 | 0.0398 | 0.0000 |
| HN1024 `31xmxmdp` | <1.0 | 0.77 | 0.9153 | 0.9153 | 0.6606 | 0.3025 | 0.0398 |
| HN1024 `31xmxmdp` | <1.5 | 0.82 | 1.4167 | 1.4167 | 1.1620 | 0.7800 | 0.2865 |
| HN256 `8h9q0v4t` | <0.5 | 0.70 | 0.4378 | 0.4378 | 0.2467 | 0.0716 | n/a |
| HN256 `8h9q0v4t` | <1.0 | 0.76 | 0.9233 | 0.9233 | 0.6925 | 0.3582 | n/a |
| HN256 `8h9q0v4t` | <1.5 | 0.80 | 1.4088 | 1.4088 | 1.1780 | 0.7720 | n/a |

At this strict setting, lower tau does find a no-HN `<0.5 pp` operating point:
tau `0.61`.  The important difference is calibration shape: HN1024 reaches the
same strict recall-retention budget at tau `0.72`, while no-HN must lower tau by
about `0.11`.

| run/budget | tau | gs100k R/P | gs1M R/P | gs1M kept |
|---|---:|---:|---:|---:|
| no-HN <0.5 | 0.61 | 98.50 / 10.18 | 97.95 / 9.82 | 9.98 |
| no-HN <1.0 | 0.69 | 98.23 / 12.32 | 97.88 / 10.12 | 9.68 |
| no-HN <1.5 | 0.73 | 97.76 / 15.16 | 97.52 / 10.83 | 9.03 |
| HN1024 <0.5 | 0.72 | 98.54 / 10.19 | 97.78 / 9.79 | 9.99 |
| HN1024 <1.0 | 0.77 | 98.27 / 12.26 | 97.74 / 9.98 | 9.81 |
| HN1024 <1.5 | 0.82 | 97.80 / 16.96 | 97.49 / 11.56 | 8.45 |
| HN256 <0.5 | 0.70 | 98.46 / 10.26 | n/a | n/a |
| HN256 <1.0 | 0.76 | 98.17 / 12.51 | n/a | n/a |
| HN256 <1.5 | 0.80 | 97.76 / 15.82 | n/a | n/a |

HN256 was intentionally not evaluated on dev-1M.  Its calibration is limited to
dev raw/10k/100k because the 1M path is expensive and did not change the
selection story for no-HN/HN1024.

HN256 compute note: `8h9q0v4t` used `query_chunk=256` and `text_chunk=4096`,
but the eval-only path is still rank0-only for scoring.  The log reports
`score_device=cuda:0` for dev/ACL/tagged/medicine, so GPU5 is not a true eval
shard yet.  Using both GPUs efficiently would require partitioning queries or
the text bank across ranks instead of only increasing per-rank chunks.

## Expanded-Bank-Only Selection

If max-drop is measured only over expanded retriever banks, the lower sweep
moves no-HN's `<0.5 pp` point to tau `0.66`.  The compact table below uses
`gs10k/gs100k` so HN256 can be compared directly; HN256 was not rerun on dev-1M.

| run | budget | tau | max drop | gs10k drop | gs100k drop | gs100k R/P |
|---|---:|---:|---:|---:|---:|---:|
| no-HN `e8t8zdtd` | <0.5 | 0.66 | 0.4935 | 0.4935 | 0.2069 | 98.38 / 11.10 |
| no-HN `evcgcdlu` | <1.0 | 0.71 | 0.9551 | 0.9551 | 0.5731 | 98.01 / 13.57 |
| no-HN `evcgcdlu` | <1.5 | 0.75 | 1.4884 | 1.4884 | 1.0904 | 97.49 / 17.01 |
| HN256 `8h9q0v4t` | <0.5 | 0.73 | 0.4696 | 0.4696 | 0.2229 | 98.30 / 10.97 |
| HN256 `8h9q0v4t` | <1.0 | 0.78 | 0.9472 | 0.9472 | 0.5492 | 97.98 / 14.01 |
| HN256 `8h9q0v4t` | <1.5 | 0.81 | 1.3212 | 1.3212 | 0.9153 | 97.61 / 16.86 |
| HN1024 `31xmxmdp` | <0.5 | 0.75 | 0.4776 | 0.4776 | 0.1831 | 98.39 / 11.06 |
| HN1024 `31xmxmdp` | <1.0 | 0.80 | 0.9631 | 0.9631 | 0.5890 | 97.99 / 14.90 |
| HN1024 `31xmxmdp` | <1.5 | 0.83 | 1.4008 | 1.4008 | 1.0188 | 97.56 / 18.12 |

This is the cleanest evidence for HN's score-shaping effect: HN256 and HN1024
both support much higher tau than no-HN at the same expanded-bank recall budget.
The gain is mainly precision/thresholdability, not raw recall.  The historical
no-HN/HN1024 dev-1M values are left out of this compact table because they were
not materially different for selection: the max drop was dominated by gs10k,
and gs1M did not change the selected tau.

## Dev-Only PR Curves

This view ignores held-out domains and compares the three models on dev only.
The x-axis is filtered micro precision and the y-axis is filtered recall as tau
sweeps.  no-HN combines `e8t8zdtd` for tau `0.50..0.70` and `evcgcdlu` for tau
`0.71..0.90`; HN256 uses `8h9q0v4t`; HN1024 uses `31xmxmdp` for tau
`0.65..0.90` plus `ifs45d6j` for the tau `0.91..0.99` right-tail extension.

Fixed-raw common-reference view.  The previous absolute-PR plots are kept as
diagnostic-only history and removed from this report.  Tau selection should not
use a per-bank recall ceiling.  The plots below use each model's own raw-dev
maximum filtered recall as the common reference and show recall drop in pp.
Lower is better.  The dashed horizontal line is the `1.0 pp` drop budget; the
small red dot marks the interpolated HN1024 threshold where it exactly reaches
that budget.

![Dev-only fixed-reference PR loss, three banks](figures/20260523_dev_pr_fixedraw_commonbase_threepanel_nohn_hn256_hn1024.pdf)

The useful operating range is now clearer: HN1024 reaches the `1.0 pp` recall
drop line at precision roughly `13.7` on gs100k, and the comparable raw/gs10k
points are still in the `17.5..18.5` precision range.  The far-right region
past this point is not needed for calibration because recall loss accelerates.

HN1024 tau choice from dev-only PR:

| HN1024 tau | raw P/R/drop | gs10k P/R/drop | gs100k P/R/drop | read |
|---:|---:|---:|---:|---|
| 0.78 | 17.94 / 98.19 / 0.91 | 16.82 / 98.19 / 0.91 | 13.04 / 98.18 / 0.92 | conservative, all-bank drop stays below 1 pp |
| 1.0pp crossing | 18.50 / 98.10 / 1.00, tau 0.788 | 17.48 / 98.10 / 1.00, tau 0.788 | 13.70 / 98.10 / 1.00, tau 0.787 | exact interpolated drop budget |
| 0.79 | 18.61 / 98.08 / 1.02 | 17.62 / 98.08 / 1.02 | 13.93 / 98.07 / 1.03 | midpoint, just crosses 1 pp drop |
| 0.80 | 19.29 / 97.99 / 1.11 | 18.41 / 97.99 / 1.11 | 14.90 / 97.99 / 1.11 | best sweet point if targeting moderate precision |

Here `drop` is recall drop in pp from HN1024's raw-dev maximum filtered recall,
not from each bank's own maximum.  This is the correct common-reference view:
gs100k no longer looks artificially safest.  If the selection rule is a hard
all-bank `<1 pp` recall-drop budget, tau `0.78` is the only clean choice in
this grid range.  If continuous tau is allowed, the exact drop-budget operating
point is about tau `0.787..0.788`; for a two-decimal deployment setting, this
rounds to tau `0.79`, but tau `0.78` is the safer rounded choice because tau
`0.79` is already slightly above the budget on all three banks.

## Held-Out Readout From 100k Sweep

I added the missing no-HN held-out readout at tau `0.61` as run `zji769ve`.
This lets the strict raw-included `<0.5 pp` dev-drop operating points be
compared directly:

- no-HN `e8t8zdtd`: tau `0.61`, max dev drop `0.4537 pp`.
- HN256 `8h9q0v4t`: tau `0.70`, max dev drop `0.4378 pp`.
- HN1024 `31xmxmdp`: tau `0.72`, max dev drop `0.4378 pp`.

Held-out gs10k values use the fixed strict raw/base denominator.  Delta rows
are `HN - no-HN`, in percentage points.

Strict raw-included `<0.5 pp` dev-drop:

| setting | ACL gs10k R/P | tagged gs10k R/P | medicine gs10k R/P |
|---|---:|---:|---:|
| no-HN tau 0.61 | 94.63 / 9.52 | 98.07 / 9.91 | 94.10 / 10.34 |
| HN1024 tau 0.72 | 93.11 / 9.45 | 97.44 / 9.96 | 92.69 / 10.90 |
| HN256 tau 0.70 | 93.23 / 9.48 | 98.30 / 10.03 | 93.81 / 11.22 |
| HN1024 - no-HN | R -1.52 / P -0.06 | R -0.63 / P +0.05 | R -1.41 / P +0.56 |
| HN256 - no-HN | R -1.40 / P -0.04 | R +0.23 / P +0.12 | R -0.29 / P +0.88 |

At the matched strict `<0.5 pp` dev-drop point, HN1024 still is not a recall
win on held-out readouts.  It is roughly precision-neutral on ACL/tagged ACL
and gains only `+0.56 pp` micro precision on medicine, while losing
`0.6..1.5 pp` recall.

Existing held-out nearest `~1.0 pp` dev-drop pair:

The exact strict `<1.0 pp` dev-selected pair is no-HN tau `0.69` vs HN1024 tau
`0.77`, but the older no-HN held-out grid did not include tau `0.69`.  The
available held-out comparison is the previous nearest pair, no-HN tau `0.70`
vs HN1024 tau `0.78`.

| setting | ACL gs10k R/P | tagged gs10k R/P | medicine gs10k R/P |
|---|---:|---:|---:|
| no-HN tau 0.70 | 93.40 / 12.17 | 97.34 / 12.34 | 91.03 / 14.38 |
| HN1024 tau 0.78 | 91.42 / 12.73 | 96.60 / 12.81 | 89.00 / 16.21 |
| HN256 tau 0.76 | 91.89 / 12.33 | 97.64 / 12.62 | 90.61 / 15.56 |
| HN1024 - no-HN | R -1.98 / P +0.56 | R -0.73 / P +0.47 | R -2.03 / P +1.82 |
| HN256 - no-HN | R -1.51 / P +0.16 | R +0.30 / P +0.28 | R -0.42 / P +1.18 |

Strict raw-included `<1.5 pp` dev-drop:

| setting | ACL gs10k R/P | tagged gs10k R/P | medicine gs10k R/P |
|---|---:|---:|---:|
| no-HN tau 0.73 | 91.71 / 15.85 | 96.48 / 15.36 | 88.50 / 17.23 |
| HN1024 tau 0.82 | 86.05 / 22.02 | 94.48 / 20.01 | 83.55 / 25.35 |
| HN256 tau 0.80 | 87.97 / 20.94 | 95.89 / 19.35 | 86.71 / 22.56 |
| HN1024 - no-HN | R -5.66 / P +6.17 | R -2.00 / P +4.65 | R -4.94 / P +8.11 |
| HN256 - no-HN | R -3.74 / P +5.09 | R -0.59 / P +3.99 | R -1.79 / P +5.33 |

Across all matched or nearest held-out pairs, HN1024 does not produce a recall
win.  Precision only becomes visibly larger in the looser `<1.5 pp` regime,
where the recall cost is also large.

For reference, the compact delta view is:

| matched dev-drop pair | HN ACL gs10k delta | HN tagged gs10k delta | HN medicine gs10k delta |
|---|---:|---:|---:|
| no-HN tau 0.70 vs HN tau 0.78, about 1.0 pp drop | R -1.98 / P +0.56 | R -0.73 / P +0.47 | R -2.03 / P +1.82 |
| no-HN tau 0.73 vs HN tau 0.82, about 1.4 pp drop | R -5.66 / P +6.17 | R -2.00 / P +4.65 | R -4.94 / P +8.11 |
| no-HN tau 0.61 vs HN256 tau 0.70, about 0.5 pp drop | R -1.40 / P -0.04 | R +0.23 / P +0.12 | R -0.29 / P +0.88 |
| no-HN tau 0.70 vs HN256 tau 0.76, about 1.0 pp drop | R -1.51 / P +0.16 | R +0.30 / P +0.28 | R -0.42 / P +1.18 |
| no-HN tau 0.73 vs HN256 tau 0.80, about 1.5 pp drop | R -3.74 / P +5.09 | R -0.59 / P +3.99 | R -1.79 / P +5.33 |

### Held-Out Raw/1k/10k Detail

The tables below show the same held-out operating points with raw, gs1k, and
gs10k candidate banks.  Each cell is `recall / micro precision`.

Strict raw-included `<0.5 pp` dev-drop:

| setting | ACL raw | ACL 1k | ACL 10k | tagged raw | tagged 1k | tagged 10k | medicine raw | medicine 1k | medicine 10k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no-HN tau 0.61 | 97.90 / 26.94 | 97.66 / 12.19 | 94.63 / 9.52 | 98.76 / 15.74 | 98.73 / 12.53 | 98.07 / 9.91 | 94.44 / 11.62 | 94.39 / 11.32 | 94.10 / 10.34 |
| HN256 tau 0.70 | 95.74 / 33.12 | 95.68 / 14.64 | 93.23 / 9.48 | 98.61 / 18.13 | 98.58 / 14.05 | 98.30 / 10.03 | 94.02 / 12.97 | 94.02 / 12.63 | 93.81 / 11.22 |
| HN1024 tau 0.72 | 96.09 / 31.91 | 96.03 / 14.45 | 93.11 / 9.45 | 98.10 / 17.97 | 98.10 / 13.93 | 97.44 / 9.96 | 93.23 / 12.91 | 93.19 / 12.46 | 92.69 / 10.90 |

This `<0.5 pp` table is the strongest held-out evidence for HN's useful
precision effect.  On raw+gs1k candidate banks, HN1024's average precision gain
over no-HN is `+2.22 pp`, while the average recall drop is `-1.19 pp`.  The
new HN256 checkpoint shows the same pattern with better recall retention:
`+2.53 pp` precision and `-0.87 pp` recall on raw+gs1k.  This benefit still
does not carry over strongly to gs10k: HN256 averages `+0.32 pp` precision at
`-0.49 pp` recall cost, while HN1024 averages only `+0.19 pp` precision at
roughly `-1.19 pp` recall cost.

![Held-out raw/1k/10k delta at strict <0.5 pp](figures/20260523_heldout_lt0p5_raw1k10k_delta_nohn_hn256_hn1024.png)

Strict raw-included `<1.0 pp` dev-drop:

| setting | ACL raw | ACL 1k | ACL 10k | tagged raw | tagged 1k | tagged 10k | medicine raw | medicine 1k | medicine 10k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no-HN tau 0.70 | 93.93 / 42.10 | 93.93 / 25.14 | 93.40 / 12.17 | 97.52 / 24.11 | 97.52 / 20.98 | 97.34 / 12.34 | 91.11 / 15.55 | 91.11 / 15.36 | 91.03 / 14.38 |
| HN256 tau 0.76 | 92.76 / 47.04 | 92.76 / 29.10 | 91.89 / 12.33 | 97.69 / 25.31 | 97.69 / 21.78 | 97.64 / 12.62 | 90.61 / 17.20 | 90.61 / 17.03 | 90.61 / 15.56 |
| HN1024 tau 0.78 | 92.12 / 44.49 | 92.12 / 27.06 | 91.42 / 12.73 | 96.63 / 25.09 | 96.63 / 22.03 | 96.60 / 12.81 | 89.00 / 18.43 | 89.00 / 18.13 | 89.00 / 16.21 |

Strict raw-included `<1.5 pp` dev-drop:

| setting | ACL raw | ACL 1k | ACL 10k | tagged raw | tagged 1k | tagged 10k | medicine raw | medicine 1k | medicine 10k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no-HN tau 0.73 | 91.83 / 47.12 | 91.83 / 32.11 | 91.71 / 15.85 | 96.50 / 27.54 | 96.50 / 25.15 | 96.48 / 15.36 | 88.50 / 18.21 | 88.50 / 18.05 | 88.50 / 17.23 |
| HN256 tau 0.80 | 88.15 / 54.45 | 88.15 / 41.63 | 87.97 / 20.94 | 95.89 / 31.03 | 95.89 / 28.53 | 95.89 / 19.35 | 86.71 / 23.86 | 86.71 / 23.78 | 86.71 / 22.56 |
| HN1024 tau 0.82 | 86.05 / 52.01 | 86.05 / 40.23 | 86.05 / 22.02 | 94.48 / 30.86 | 94.48 / 29.27 | 94.48 / 20.01 | 83.55 / 26.91 | 83.55 / 26.71 | 83.55 / 25.35 |

### Budget Sensitivity

The compact view below averages deltas over the three held-out domains.  The
`raw+1k` group averages raw and gs1k candidate banks; the `10k` group averages
gs10k only.  In the plot, solid lines are precision gain and dashed lines are
recall cost; solid above dashed means the precision gain is larger than the
recall loss.

| budget | model | bank group | mean R delta | mean P delta |
|---|---|---|---:|---:|
| <0.5 | HN256 | raw+1k | -0.87 | +2.53 |
| <0.5 | HN256 | 10k | -0.49 | +0.32 |
| <0.5 | HN1024 | raw+1k | -1.19 | +2.22 |
| <0.5 | HN1024 | 10k | -1.19 | +0.18 |
| <1.0 | HN256 | raw+1k | -0.50 | +2.37 |
| <1.0 | HN256 | 10k | -0.54 | +0.54 |
| <1.0 | HN1024 | raw+1k | -1.60 | +2.00 |
| <1.0 | HN1024 | 10k | -1.58 | +0.95 |
| <1.5 | HN256 | raw+1k | -2.03 | +5.85 |
| <1.5 | HN256 | 10k | -2.04 | +4.80 |
| <1.5 | HN1024 | raw+1k | -4.25 | +6.30 |
| <1.5 | HN1024 | 10k | -4.20 | +6.31 |

![Held-out tradeoff by budget](figures/20260523_heldout_tradeoff_by_budget_summary.png)

This makes the decision boundary more defensible.  For raw/gs1k candidate
banks, HN's precision gain exceeds its recall loss across all three budgets.
For gs10k, the new HN256 checkpoint is closer to no-HN under `<0.5 pp` and
`<1.0 pp`, but the precision gain is still small.  The precision gain clearly
exceeds recall cost only in the loose `<1.5 pp` regime.  So HN remains a
reasonable precision/small-bank ablation, while no-HN remains the conservative
recall-first gs10k choice.

## 1M Eval Cost

The current 1M path is only GPU-chunked, not fully streaming.  `_score_eval_logits`
uses `eval_score_query_chunk=64` and `eval_score_text_chunk=4096`, but then
returns `torch.cat(out_rows, dim=0)`, so the full `[12564, 1000000]` CPU logits
are materialized.

Observed peak RSS:

| run | peak RSS | wall time |
|---|---:|---:|
| no-HN low-tau `e8t8zdtd` | about 259M KiB RSS, ~247GiB | 1115s runtime / 1164s launcher elapsed |
| no-HN `evcgcdlu` | about 255GB | 1193s |
| HN1024 `31xmxmdp` | about 256GB | 1177s |

For occasional stress tests this is acceptable on taurus.  For regular 1M dev
calibration, the eval path should be changed to streaming/top-k-only aggregation
so recall, filtered recall, precision, and tau sweeps are accumulated without
keeping full logits.

## Interpretation

The updated tau grid makes the calibration surface clearer:

- no-HN is still at least tied on unfiltered dev raw/10k/100k recall and is
  better on dev gs1M.
- no-HN does have a strict `<0.5 pp` dev point after extending tau downward:
  tau `0.61`, not `none`.
- HN runs have better thresholdability: they reach comparable recall budgets at
  higher tau, especially strict `<0.5 pp` tau `0.70` for HN256 and `0.72` for
  HN1024 vs no-HN tau `0.61`.
- On dev-only gs100k PR curves, HN1024 is the small same-precision recall winner
  around `15..20` precision, by roughly `+0.05..+0.20 pp` recall.
- On held-out raw/gs1k candidate banks, HN's precision gain is usually larger
  than its recall loss across `<0.5`, `<1.0`, and `<1.5` budgets.  The new
  HN256 checkpoint gives the cleaner version of this story because it keeps
  much more recall than the older HN256 readout.
- On held-out gs10k, HN does not beat no-HN under recall-first budgets.  The
  precision gain is too small at `<0.5` and `<1.0`; it becomes large only at
  `<1.5`, where recall cost is also large.

Recommendation: frame the result as an operating-point tradeoff, not a single
global winner.  no-HN is the conservative recall-first gs10k main line; HN1024
is the precision-oriented / small-bank variant, and the latest HN256 checkpoint
is the most balanced HN point.
