# En-De lm4 TERM_ACC Scoring Policy Check

Date: 2026-05-25

## Scope

Check whether `TERM_ACC` is computed with the same policy for the En-De lm4
serial no-RAG result and the En-De lm4 same-LM batch no-RAG result.

Serial event:

- `20260524T160830__simuleval__tagged_acl_origin_norag_de_lm4_raw_rerun`
- Eval TSV:
  `/mnt/gemini/data1/jiaxuanluo/tagged_acl_origin_norag_de_lm4_raw_rerun_20260524T160830_tagacl_origin_norag_de_lm4_raw_rerun/origin_norag/de/gigaspeech-de-s_origin-bsz4_gacl6060_tagged_gt_raw_min_norm2_cs3.84_hs0.48_lm4_k210_k110_th0p0/eval_results.tsv`

Batch event:

- `20260524T2338__simuleval__tagged_acl_origin_norag_de_lm4_batch_max40_aries23`
- Eval TSV:
  `/mnt/gemini/data1/jiaxuanluo/tagged_acl_origin_norag_de_lm4_batch_max40_aries23_20260524T2338_tagacl_origin_norag_de_lm4_batch_max40_aries23/origin_norag_de_lm4_batch_max40/de/dtagacl_origin_norag_batch_max40_lm4_k0_th0.0_gacl6060_tagged_gt_raw_min_norm2/eval_results.tsv`

## Verdict

The `TERM_ACC` denominator policy is consistent for these two TSVs.

With `--source-reference` available, the scorer counts a glossary occurrence
only when both conditions are true:

1. The source sentence contains the English source term.
2. The target reference sentence contains the target-language translation.

The numerator then requires the prediction to contain the same target
translation.

This matches the intended policy: source-term match plus reference-translation
match defines the ground-truth term denominator.

## Code Evidence

`offline_streamlaal_eval.py` calls FBK `stream_laal_term.py` for headline
`TERM_ACC` and parses `TERM_ACC`, `CORRECT_TERMS`, and `TOTAL_TERMS` from that
tool's stdout.

The called tool is:

`/mnt/taurus/home/jiaxuanluo/FBK-fairseq/examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py`

In that tool:

- `parse_references(...)` attaches source reference sentences when
  `--source-reference` is passed.
- `source_contains(...)` applies normalized case-folded matching with word
  boundaries for ASCII-like source terms.
- `compute_term_accuracy(...)` does:
  - `source_has_term = source_contains(source_ref, source_term)` when source
    references exist.
  - `target_has_term = target_term in ref`.
  - If both are true, increment `total_terms`.
  - If the target term is also in the prediction, increment `correct_terms`.

## Artifact Evidence

Both serial and batch evals use:

- mode: `acl6060`
- lang: `de`
- glossary:
  `/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_raw_min_norm2.json`
- source text: equivalent 468 lines
- reference text: equivalent 468 lines
- audio yaml: equivalent after normalizing `wav` to basename

Source/ref equality checks:

| file | equality | sha256 |
| --- | --- | --- |
| source text | equal | `aa37f443c0e1c6d23fac0ef285230c29e9a25e7941d45cb796c622a5c15452d1` |
| ref text | equal | `fd6b1de66cd0aa5a9219a79aebaedc15d5117dff5339cd9cf7613e951a311160` |
| audio yaml normalized by wav basename | equal | `f31dbbc7c05b29a9880d3d70a9d4cd3d4255c5fb4e66a87d847b42d4b607feca` |

An independent denominator count over the raw tagged glossary, source text, and
German references found:

- unique target translations loaded by scorer-style de-duplication: 230
- source/reference sentence rows: 468
- ground-truth denominator: 935
- sentences with at least one counted term: 384

Both eval TSVs report `TERM_TOTAL=935`, matching the independent denominator
count.

## Metrics

| mode | TERM_ACC | TERM_CORRECT | TERM_TOTAL |
| --- | ---: | ---: | ---: |
| serial | 0.6909 | 646 | 935 |
| batch | 0.6738 | 630 | 935 |

The different `TERM_ACC` comes from different predictions, not from a different
denominator or scoring policy.

## Caveat

`TERM_ACC` is not the same as the repo-local `TERM_ADOPTION` or
`REAL_TERM_ADOPT` columns. In the batch run, the optional
`compute_sentence_term_adoption.py` step did not run because
`offline_streamlaal_eval.py` looked for the repo under
`/home/jiaxuanluo/InfiniSST` instead of the actual Aries path. That is why the
batch TSV has `TERM_ADOPTION=N/A` and `TERM_FCR_MODE=unknown`. This does not
affect headline `TERM_ACC`, because `TERM_ACC` had already been computed by
`stream_laal_term.py`.
