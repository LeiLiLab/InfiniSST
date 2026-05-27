# ACL6060 eval glossary normalized-length filter readout

## Hypothesis

Dropping optional glossary expansion terms whose normalized text has fewer than
two characters should remove spurious transcript matches such as `a+ -> a`
without harming real ACL6060 extracted terms such as `qa`, `crf`, or `asr`.

## Background / Motivation

ACL6060 gs10k evaluation builds chunk positives from the active glossary and
the speech chunk transcript. The scaled ACL glossary is a union of per-paper
extracted terms plus wiki padding, so very short normalized padding terms can
inflate chunk-level positives.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/lh1b88kw
- Diff: one-shot eval-only readout from the current best-secondary checkpoint;
  compare `eval_glossary_match_min_norm_chars=1` against `2` on the same
  ACL6060 variable-context dataset and same ACL gs10k glossary.

## Expected metrics

Base-bank recall should remain essentially unchanged. The gs10k label counts
and filtered recall may move if previous metrics were receiving credit from
one-character normalized wiki padding terms.

## Verdict

The normalized-length filter removes the obvious one-character padding terms.
With a backfilled gs10k bank, ACL6060 base recall is unchanged at 0.9912 and
gs10k recall moves from 0.9463 to 0.9457. The gs10k chunk-positive label count
drops from 2940 to 2751, indicating that the old rule inflated positives through
short normalized padding terms while retrieval quality is effectively stable.
