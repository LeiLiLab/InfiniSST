# New V6 assistant term-tag SFT, zh, r32/a64, TP2

Purpose: strengthen term adoption on top of the current best New V5 setting.

Base setting:

- New V5 data: `new_v5_no_gt_zero_llm_variant_aug_oldnewv3`
- user input / term_map: unchanged from New V5
- no-GT chunks: remain `term_map:NONE`
- assistant target text: exact GT target translations are wrapped as `<term>...</term>`

Data:

- root: `/mnt/gemini/data1/jiaxuanluo/speech_llm_new_v6_assistant_termtag_no_gt_zero_oldnewv3_zh_20260523`
- train: `train_s_zh_new_v6_assistant_termtag_no_gt_zero_oldnewv3.jsonl`
- dev: `dev_s_zh_new_v6_assistant_termtag_no_gt_zero_oldnewv3.jsonl`
- summary: `new_v6_assistant_termtag_no_gt_zero_oldnewv3_summary.json`

Build policy:

- tag template: `<term>{translation}</term>`
- minimum target length after whitespace removal: 2 chars
- max tags per row: 16
- exact substring only; if the target translation is not found in future assistant text, it is skipped
- legacy rows without `gt_terms_by_chunk` are kept unchanged

Train stats:

- rows: 12,500
- candidate GT terms: 124,843
- candidate GT terms after min length: 114,833
- assistant tag replacements: 91,808
- tag rate over all GT terms: 73.54%
- tag rate after min length: 79.95%

Dev note:

The oldnewv3 dev JSONL does not carry usable `gt_terms_by_chunk`, so dev is kept
unchanged and is only an SFT-loss placeholder.  Downstream SimulEval is the real
selection signal.

Eval note:

This model may emit `<term>...</term>` tags.  SimulEval metrics should strip
`<term>` and `</term>` before BLEU / TERM_ACC / REAL_ADOPT scoring, or report
both raw and tag-stripped outputs.
