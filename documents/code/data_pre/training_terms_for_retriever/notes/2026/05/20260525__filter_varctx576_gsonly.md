# Filter varctx576 retriever train data to GigaSpeech only

## Hypothesis

The mixed varctx576 retriever training JSONL can be filtered to a valid
GigaSpeech-only ablation dataset by dropping rows whose `utter_id`,
`context_build`, or audio path identify `wiki_synth`.

## Background / Motivation

The downstream retriever ablation needs the current HN1024 varctx576 recipe but
without synthetic wiki training rows.  The source stats report both GigaSpeech
and `wiki_synth` domains, so a streaming JSONL filter is sufficient and avoids
rebuilding the expensive audio context dataset.

## What changed vs baseline

- Input: mixed varctx576 train JSONL.
- Output: GigaSpeech-only train JSONL under `/mnt/gemini/home/jiaxuanluo`.
- Script: `documents/code/data_pre/training_terms_for_retriever/filter_gigaspeech_only_jsonl.py`.
- Filtering is fail-fast for unknown domains.

## Expected metrics

The stats JSON should report all `wiki_synth` rows dropped, zero unknown-domain
rows, and kept rows matching the GigaSpeech count from the source stats.

## Verdict

Completed.  The filter read 7,554,926 rows, kept 4,661,221 GigaSpeech rows,
dropped 2,893,705 `wiki_synth` rows, and reported zero JSON errors or unknown
domains.  Output:
`/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2full_gsdedup_varctx576_gsonly.jsonl`.
