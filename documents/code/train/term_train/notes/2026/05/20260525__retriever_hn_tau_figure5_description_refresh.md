## Hypothesis

Figure 5 should describe only the curves that are currently shown and make the threshold-selection dashed line explicit.

## Background / Motivation

The paper text still mentioned an HN512 placeholder even though the current Figure 5 data and PDF include only no-HN, HN256, and HN1024. The dashed line also needed to be explained as the dev-only recall-drop threshold point used to set the deployment operating region.

## What changed vs baseline

- Removed the HN512 placeholder wording from the Figure 5 caption and discussion.
- Clarified that the dashed vertical line marks the HN1024 threshold where recall drops by 1 percentage point from the raw-bank reference.
- Stated that this recall-preserving point places the selected deployment threshold near tau=0.78.

## Expected metrics

No metric values or figure pixels should change. This is a paper-text clarification over the existing fixed-raw-denominator PR-curve figure.

## Verdict

Success. The paper builds after the Figure 5 caption and discussion update.
