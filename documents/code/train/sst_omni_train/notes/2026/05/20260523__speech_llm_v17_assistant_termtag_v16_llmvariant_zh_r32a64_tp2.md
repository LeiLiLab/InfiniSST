# V17 assistant term-tag SFT, zh, r32/a64, TP2

Purpose: stronger term-adoption supervision on top of V16 LLM-variant data.

Compared with V16:

- user input is unchanged
- retriever `term_map` is unchanged
- `gt_terms_by_chunk` is unchanged
- only assistant target text is changed: exact GT target translations are wrapped as `<term>...</term>`

Data:

- root: `/mnt/gemini/data1/jiaxuanluo/speech_llm_v17_assistant_termtag_v16_llmvariant_zh_20260523`
- train: `train_s_zh_v17_assistant_termtag_v16_llmvariant_tau073_k10_minctx2p88.jsonl`
- dev: `dev_s_zh_v17_assistant_termtag_v16_llmvariant_tau073_k10_minctx2p88_first200.jsonl`
- summary: `v17_assistant_termtag_v16_llmvariant_summary.json`

Build policy:

- source data: V16 LLM-variant retriever-timeline data
- tag template: `<term>{translation}</term>`
- minimum target length after whitespace removal: 2 chars
- max tags per row: 16
- exact substring only; if the target translation is not found in future assistant text, it is skipped

Train stats:

- rows: 6,237
- chunks: 52,318
- candidate GT terms: 39,468
- candidate GT terms after min length: 36,709
- assistant tag replacements: 33,461
- tag rate over all GT terms: 84.78%
- tag rate after min length: 91.15%

Important eval note:

This model may emit `<term>...</term>` tags.  Fast eval should strip `<term>` and
`</term>` before BLEU / TERM_ACC / REAL_ADOPT scoring, or run both raw and
tag-stripped scoring to see whether tags leak at inference.
