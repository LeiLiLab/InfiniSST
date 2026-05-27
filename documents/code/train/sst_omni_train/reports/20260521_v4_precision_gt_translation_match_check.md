# V4 Precision Term-map GT Translation Match Check

This check validates whether GT terms from the source-match 100k glossary have
target translations that exactly appear in the SFT assistant/reference text.

Dataset:

- train: `/mnt/gemini/data1/jiaxuanluo/speech_llm_v4_precision_termmap_zh_lh1b88kw_tau073_srcmatch100k_20260521/train_s_zh_v4_precision_termmap_lh1b88kw_tau073_srcmatch100k.jsonl`
- dev: `/mnt/gemini/data1/jiaxuanluo/speech_llm_v4_precision_termmap_zh_lh1b88kw_tau073_srcmatch100k_20260521/dev_s_zh_v4_precision_termmap_lh1b88kw_tau073_srcmatch100k.jsonl`

Definitions:

- `full_ref exact`: target translation is an exact substring of all assistant
  outputs in the same row.
- `local4 exact`: target translation is an exact substring of assistant outputs
  in the current chunk plus the next few assistant turns, matching the builder's
  local target window.
- `term_map GT`: term-map entries whose source key is present in
  `gt_terms_by_chunk`.

## Summary

| split | GT terms | GT full_ref exact | GT local4 exact | term_map GT entries | term_map GT full_ref exact | term_map GT local4 exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 172,726 | 47.51% | 45.07% | 93,034 | 63.65% | 60.85% |
| dev | 1,902 | 43.01% | 41.96% | 1,044 | 57.76% | 56.13% |

No-space normalization barely changes the result, so this is not mostly a
spacing issue.

## Interpretation

The source-match 100k glossary is too broad to be used directly as GT
term-map supervision.  Many entries are everyday words or dictionary/Wikipedia
translations that do not match the SFT reference wording.  If these entries are
placed into the term map, the model is trained either to ignore term_map or to
copy translations that conflict with the reference.

This is likely one cause of low zh TERM_ACC after retriever-SFT: no-TM-SFT often
uses the benchmark-preferred reference wording, while TM-SFT sees alternative
target translations and learns weaker exact-adoption behavior.

## Frequent Full-reference Mismatches

Train, all GT terms:

- `one -> 人们`: 1,807
- `like -> 点赞`: 1,653
- `think -> 想一想`: 1,292
- `people -> 人们`: 996
- `well -> 井`: 787
- `time -> 时间`: 748
- `want -> 需要`: 687
- `right -> 权利`: 646
- `back -> 后面`: 626
- `way -> 方式`: 622

Train, GT entries actually placed in term_map:

- `one -> 人们`: 501
- `people -> 人们`: 406
- `like -> 点赞`: 394
- `think -> 想一想`: 359
- `back -> 后面`: 272
- `time -> 时间`: 269
- `things -> 事情`: 223
- `way -> 方式`: 212
- `thing -> 东西`: 189
- `well -> 井`: 186

## Examples

`want -> 需要`

- local reference: `我想 引用一段文字。`
- issue: source term is valid, but the glossary translation `需要` does not match
  the reference wording.

`quote -> 名言`, `text -> 文本`

- local reference: `引用一段文字。`
- issue: the reference uses `引用` / `文字`, not the glossary target forms.

`things -> 事情`

- local reference: `她会做些我本意不想让她做的事。`
- issue: the source word is present, but the reference uses `事`, not `事情`.

## Recommendation

Do not train the current V4 precision data as the main retriever-SFT line.
Regenerate a stricter version that only treats a source-match term as GT if its
target translation exact-matches the local target window or full row reference.

Suggested next dataset:

- keep realistic retriever term maps for noise exposure;
- backfill or prioritize only reference-supported GT terms;
- demote unmatched source-match terms to ordinary retrieved/noisy entries, not
  GT supervision;
- keep `tagged` and `adv` as follow-up variants after the reference-supported
  base version is validated.
