# New V5 no-GT-zero data

## Hypothesis

Applying the no-GT-zero rule to New V4 should reduce noisy term-map supervision on chunks without GT terms while keeping New V4's LLM-variant adoption signal on GT chunks.

## Background / Motivation

The V16 no-GT-zero quick eval improved `TERM_ACC` and `REAL_ADOPT` over V16.  New V5 applies the same rule to the stronger old-`new_v3` + LLM-variant data line.

## What changed vs baseline

- Input: New V4 cache-only LLM-variant JSONL.
- Rule: if `gt_terms_by_chunk[i]` is empty, rewrite user chunk `i` as `term_map:NONE`; otherwise keep the original term map unchanged.
- Output: New V5 train/dev JSONL under `/mnt/gemini/data1/jiaxuanluo/speech_llm_new_v5_no_gt_zero_llm_variant_aug_oldnewv3_zh_20260522`.

## Expected metrics

The useful signal is lower noisy term conditioning and better downstream `REAL_ADOPT` on tagged ACL.

## Verdict

Success.  Train data has 12,500 rows / 68,705 chunks.  The no-GT-zero rule removed 135,011 term-map entries from 16,815 no-GT chunks, reducing average entries per chunk from 14.50 to 12.54.  182 legacy train rows lacked `gt_terms_by_chunk` and were explicitly kept unchanged with `--missing-gt-policy keep_unchanged`.  Dev rows also lack GT fields and are kept unchanged for trainer validation plumbing only.
