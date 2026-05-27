# HN fixed-denominator report denominator audit and dev PR update

## Hypothesis

The HN ablation report should distinguish per-domain fixed denominators from a
shared ACL6060 denominator, and should include dev-only PR curves to evaluate
thresholdability independent of held-out domains.

## Background / Motivation

The tagged ACL gs10k recall is higher than ACL and medicine.  The report needed
to verify whether tagged ACL was using the same strict raw ACL6060 glossary as
paper ACL, and needed additional raw/gs1k held-out readouts plus a dev-only
precision-recall view for no-HN, HN256, and HN1024.

## What changed vs baseline

- Added a denominator caveat: fixed raw is per eval domain, not a shared ACL6060
  denominator.
- Added held-out raw/gs1k/gs10k detail tables.
- Added a dev-only gs100k precision-recall curve and matched-precision anchor
  table.
- Added a held-out `<0.5 pp` raw/gs1k/gs10k delta plot showing that HN's
  precision gain is concentrated in smaller candidate banks.
- Added a budget-sensitivity plot/table across `<0.5`, `<1.0`, and `<1.5`
  dev-drop budgets to avoid relying on a single recall-retention cutoff.
- Updated the dev-only PR curve to extend the x-axis through precision `30`
  after adding the HN1024 tau `0.91..0.99` right-tail sweep.
- Replaced the superseded absolute-PR visualizations with fixedraw
  common-reference loss plots split by bank: raw dev approximately 1k-bank
  scale, gs10k, and gs100k.
- Added a paper-style three-panel matplotlib figure for the fixedraw
  common-reference loss plots, with smaller HN1024 `1 pp` crossing markers.
  The canonical figure artifact is PDF, with a PNG preview retained for quick
  inspection.
- Replaced the older HN256 report row with latest checkpoint eval `8h9q0v4t`,
  loaded from `gsjheh6r` latest step 1440.
- Added matched-precision tau/readout anchors for dev gs100k.
- Kept HN256 dev selection limited to raw/gs10k/gs100k, without dev-1M.

## Expected metrics

The report should show that HN gives higher tau / thresholdability on dev, but
held-out recall-first comparisons still do not clearly beat no-HN.

## Verdict

Completed; updated again on 2026-05-23 with HN256 latest run `8h9q0v4t`.  The
report clarifies that tagged ACL uses its own 238-term raw tagged denominator,
not the paper ACL 97-term denominator.  The dev-only gs100k PR curve still gives
HN1024 a small same-precision recall edge around the `15..20` precision band,
while the latest HN256 row is closer to no-HN than the older HN256 readout.

For strict raw-included `<0.5 pp`, HN256 now selects tau `0.70` and gives
held-out gs10k ACL/tagged/medicine `93.23/9.48`, `98.30/10.03`, and
`93.81/11.22`.  Relative to no-HN tau `0.61`, that is much less damaging than
the older HN256 row: ACL `R -1.40 / P -0.04`, tagged `R +0.23 / P +0.12`,
medicine `R -0.29 / P +0.88`.

The held-out `<0.5 pp` delta plot now shows HN256's raw/gs1k precision gain is
`+2.53 pp` for `-0.87 pp` recall, while HN1024 is `+2.22 pp` for `-1.19 pp`.
On gs10k, HN256 is still only a modest precision tradeoff: `+0.32 pp`
precision for `-0.49 pp` recall.  The budget-sensitivity summary keeps the same
main interpretation: HN is useful as a precision/small-bank ablation, while
no-HN remains the conservative recall-first gs10k choice.

Matched-precision dev gs100k anchors sharpen the model-selection story.  After
adding HN1024 run `ifs45d6j` for tau `0.91..0.99`, HN1024 has the highest
interpolated recall at precision anchors `10/12/15/16/17/18/20/22`, with
corresponding tau values approximately
`0.704/0.766/0.801/0.811/0.820/0.829/0.845/0.860`.  no-HN retakes the lead at
precision `25/28/30`, where HN1024 needs tau about `0.880/0.899/0.911` and the
recall cost becomes larger.  This supports choosing HN1024 as the
precision-aware retriever in the useful moderate-precision region, while
keeping no-HN as the recall-first baseline and HN256 as the balanced middle
point.

The multi-bank overview is now also available as one three-panel figure, with
raw dev, gs10k, and gs100k aligned horizontally for paper-friendly placement.
HN1024's story is strongest on gs100k.  On raw dev and gs10k, HN1024 still
gives visible score-shaping, but the same-precision winner is more mixed near
the low-precision and 22+ edges.  For HN1024 tau selection, tau `0.78` is the
cleaner hard all-bank `<1 pp` recall-drop choice, while tau `0.80` is the
better dev-PR sweet point because it keeps all three dev banks around `98%`
recall and lands in the useful moderate-precision range.

Correction on 2026-05-23: the first multi-bank tau table reported recall drop
relative to each bank's own maximum recall.  That made gs100k look artificially
low-drop because its bank-expansion ceiling is already lower.  The corrected
common-reference artifacts add `fixedraw_commonbase` TSV/figures, where drop is
measured from each model's own raw-dev maximum filtered recall.  Under this
metric, HN1024 tau `0.78` is the clean `<1 pp` all-bank recall-drop point, while
tau `0.80` is a dev-PR sweet-point tradeoff with about `1.11 pp`
common-reference drop.

Follow-up cleanup: removed the old absolute-PR / per-bank-ceiling figures and
derived TSVs from the report artifacts.  The fixedraw common-reference plots now
have no yellow band and mark the HN1024 `drop=1.0 pp` crossing directly:
raw tau `0.788` at P/R `18.50/98.10`, gs10k tau `0.788` at `17.48/98.10`, and
gs100k tau `0.787` at `13.70/98.10`.  For a two-decimal deployment tau, `0.78`
is the conservative rounded setting because `0.79` is already slightly above
the all-bank `1 pp` budget.

Figure polishing update: moved the raw-dev tau label to the right side of the
crossing marker, shrank the red marker, and exported the three-panel paper
figure as PDF.
