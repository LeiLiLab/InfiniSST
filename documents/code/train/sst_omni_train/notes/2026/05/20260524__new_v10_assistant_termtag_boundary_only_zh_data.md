## Hypothesis

New V9's assistant-side term salience supervision is useful, but the broad local fuzzy rewrite fallback introduces noisy labels.  New V10 keeps exact wrapping and only repairs target translations split exactly across adjacent assistant-message boundaries.

## Background / Motivation

The original boundary-delay idea was for terms split at streaming chunk boundaries, for example previous assistant ends with a target prefix and the current assistant starts with the suffix.  The New V9 implementation also allowed arbitrary SequenceMatcher local rewrite.  In zh train data, that produced 2,498 non-boundary rewrite events in addition to 2,093 true boundary-delay events.

## What changed vs baseline

Baseline input is New V5 no-GT-zero oldnewv3 zh:

`/mnt/gemini/data1/jiaxuanluo/speech_llm_new_v5_no_gt_zero_llm_variant_aug_oldnewv3_zh_20260522/train_s_zh_new_v5_no_gt_zero_llm_variant_aug_oldnewv3_cacheonly.jsonl`

New V10 uses `wrap_assistant_term_targets.py --rewrite-boundary-only`, so exact target occurrences are still wrapped, but non-boundary fuzzy rewrites are skipped.  User messages and term maps remain unchanged.

## Expected metrics

This is data preparation only.  The expected data-side effect is that assistant tag replacements drop by approximately the prior non-boundary rewrite count while preserving exact wraps and true adjacent-boundary repairs.

## Verdict

Generated successfully.  Train output has 12,500 rows and 68,450 chunks.  New V10 keeps 78,389 exact assistant tag replacements and 2,988 adjacent-boundary-only repairs, for 81,377 total assistant tags.  Compared with the New V9 audit, the broad non-boundary fuzzy rewrite path is removed; 11,747 candidate rewrite cases were skipped because they did not satisfy the exact adjacent-boundary split.

Output:

`/mnt/gemini/data1/jiaxuanluo/speech_llm_new_v10_assistant_termtag_boundary_only_no_gt_zero_oldnewv3_zh_20260524/train_s_zh_new_v10_assistant_termtag_boundary_only_no_gt_zero_oldnewv3.jsonl`
