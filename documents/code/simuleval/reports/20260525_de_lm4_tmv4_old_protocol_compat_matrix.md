# En-De lm4 Old Protocol Compatibility Matrix

Date: 2026-05-25

## Scope

This diagnostic readout checks whether the earlier de/lm=4 RASST BLEU recovery
can be reproduced by running the old TM-SFT / LLM-generated term-map SLM with
HN1024 under protocol settings close to the old run.

Manifest:

`documents/code/simuleval/manifests/2026/05/20260525T113241__simuleval__tagged_acl_tmv4_de_lm4_old_protocol_compat_matrix_aries.json`

## Result

The old protocol knobs do not recover BLEU above the verified no-RAG lm=4 gate.
The best matrix setting is:

- `tau=0.73`, `max_new_tokens=80`, short `40/20s` cache with `8/4` chunk
  limits, `empty_term_map_policy=omit`.
- BLEU `32.7702`, TERM_ACC `0.8492` (`794/935`).

This is better than the current tau=0.78 old-cache rerun (`32.5332`) but still
below the verified no-RAG gate (`33.3008`) and below the current cap16 selected
short-cache result (`33.4820`, TERM_ACC `0.8674`).

## Interpretation

The historical old LLMGEN/TM-SFT de/lm=4 row was BLEU `33.2847` and TERM_ACC
`0.8396`, but this compatibility run cannot reproduce that value in the current
batch evaluator. The closest old-protocol setting with `term_map:\nNONE` reaches
only BLEU `32.2613`.

So the previous apparent BLEU improvement was not explained by simply restoring
old tau, old cache length, max_new_tokens=40, or NONE-vs-omit empty maps. It was
most likely tied to the old serial runner/protocol details and the lower old
origin+RAG comparison baseline. Against the verified no-RAG gate, old TM-SFT +
HN1024 still does not clear lm=4.

## Validation

All four matrix rows wrote:

- one `eval_results.tsv`
- five `instances.log` rows
- five `instances.strip_term.log` rows

Headline `TERM_ACC` is valid. Optional `REAL_TERM_ADOPT` and `TERM_FCR` are
`N/A` in these rows because the batch offline scorer searched for
`/home/jiaxuanluo/InfiniSST/documents/code/offline_sst_eval/compute_sentence_term_adoption.py`
inside the Aries child process. That optional post-hoc adoption step is separate
from the FBK-based headline `TERM_ACC` calculation.

Full table:

`documents/code/simuleval/reports/20260525_de_lm4_tmv4_old_protocol_compat_matrix.tsv`
