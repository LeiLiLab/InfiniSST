# DE lm=1 Tagged-ACL RASST Low-BLEU Diagnosis

## Scope

This note diagnoses the low `lm=1` row from:

`20260525T0413__simuleval__tagged_acl_tmsft_gttermwrap_exact_de_lm1to4_hn1024_tau078_batch_aries8`

The run uses the German exact GT-term-wrapped TM-SFT SLM, HN1024 retriever,
tagged ACL raw glossary, `tau=0.78`, `max_new_tokens=80`,
`VLLM_LIMIT_AUDIO=128`, and `VLLM_MAX_MODEL_LEN=12288`.

## Observed Metrics

| lm | BLEU | StreamLAAL | StreamLAAL_CA | TERM_ACC |
| --- | ---: | ---: | ---: | ---: |
| 1 | 20.6877 | 823.8605 | 1728.1666 | 0.7872 (736/935) |
| 2 | 28.6427 | 1569.8803 | 1732.9080 | 0.8642 (808/935) |
| 3 | 31.2239 | 2105.1139 | 1739.2763 | 0.8663 (810/935) |
| 4 | 32.2681 | 2539.9186 | 1599.2362 | 0.8727 (816/935) |

## Main Finding

The low `lm=1` row is dominated by one talk, `2022.acl-long.367`.
For this talk, the `lm=1` output over-generates and enters a late decode loop:

- `lm=1` prediction length: 2707 words
- reference length: 1327 words
- length ratio: 2.04
- eval-log AS-WER: 167.925
- repeated 8-gram rate: 0.538

The same talk is normal for the larger latency multipliers:

| lm | pred words | ref words | ratio | repeated 8-gram rate |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2707 | 1327 | 2.04 | 0.538 |
| 2 | 1284 | 1327 | 0.97 | 0.000 |
| 3 | 1219 | 1327 | 0.92 | 0.000 |
| 4 | 1328 | 1327 | 1.00 | 0.000 |

The repeated tail is a German/Chinese mixed loop:

`Daher verwenden wir eine Rei儿Bemerkungen，以实现位置编码。`

This is not copied from the retriever. Across `lm=1`, retrieved references have
zero Chinese characters, while 31 LLM output chunks contain Chinese characters.
The first mixed Chinese output appears at segment 693 of instance 1, after the
model has already repeated the German position-encoding phrase many times.

## Retrieval / Prompt Context

`lm=1` makes far more LLM calls with much smaller audio increments:

| lm | LLM calls | empty retrieved refs | empty-ref rate | average refs per call |
| --- | ---: | ---: | ---: | ---: |
| 1 | 3588 | 1726 | 0.481 | 0.93 |
| 2 | 1795 | 589 | 0.328 | 1.47 |
| 3 | 1199 | 279 | 0.233 | 1.96 |
| 4 | 899 | 162 | 0.180 | 2.46 |

For empty retrieval, current `batched_vllm_rag_eval.py` emits explicit
`term_map:\nNONE`; it does not omit the term-map block. Thus `lm=1` sees many
more no-term-map turns, plus a much longer chat history, which makes repeated
continuations more likely.

## Training Contamination Check

I checked the parent TM-SFT + exact GT-term-wrap training data:

- source JSONL:
  `/mnt/gemini/data1/jiaxuanluo/train_s_de_v4_ner_baseline_aligned_rate1.0_k20_final.jsonl`
- final train JSONL:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_tmsft_gttermwrap_exact_de_20260525/train_s_de_tmsft_gttermwrap_exact.jsonl`
- final dev JSONL:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_tmsft_gttermwrap_exact_de_20260525/dev_s_de_tmsft_gttermwrap_exact_first355.jsonl`

The abnormal mixed tail from `lm=1` is not present in the source, train, or dev
data. Exact searches for `位置编码`, `Rei儿`, `以实现位置编码`,
`Bemerkungen，以`, `儿Bemerkungen`, and `Reihe von Bemerkungen` all return
zero hits.

The training data does contain a tiny amount of pre-existing CJK noise, but it
is unchanged by the exact GT-term-wrap transform:

| data | rows | user CJK chars | user msgs with CJK | assistant CJK chars | assistant msgs with CJK |
| --- | ---: | ---: | ---: | ---: | ---: |
| source train | 12500 | 9 | 4 | 1 | 1 |
| final train | 12500 | 9 | 4 | 1 | 1 |
| final dev | 355 | 0 | 0 | 0 | 0 |

This is worth cleaning for hygiene, but it is too small and too different from
the repeated `位置编码` tail to explain the `lm=1` collapse by itself.

The stronger training/eval mismatch is the empty term-map format:

| data | user messages | term-map headers | `term_map:\nNONE` | audio-only user messages |
| --- | ---: | ---: | ---: | ---: |
| source train | 71730 | 54813 | 0 | 16917 |
| final train | 71730 | 54813 | 0 | 16917 |
| final dev | 1946 | 1493 | 0 | 453 |

Training empty chunks are represented as bare `<audio>` messages with no
`term_map:` header. Evaluation empty retrievals are represented as explicit
`term_map:\nNONE`. Since `lm=1` has 1726 empty-retrieval calls, this mismatch is
a more plausible trigger for the observed low-latency repetition failure than
large-scale training contamination.

## Interpretation

This does not look like a global batch-eval bug:

- all four `eval_results.tsv` files are present;
- each `instances.log` and `instances.strip_term.log` has five rows;
- only one `lm=1` talk shows severe repetition;
- the repeated phrase is from the same talk's content, not another sample;
- retrieved references are not Chinese-polluted.
- the exact abnormal mixed-language tail is absent from the parent train/dev
  data;
- the final exact GT-term-wrap data has the same tiny CJK noise count as its
  source JSONL, so the wrapping transform did not introduce new CJK
  contamination.

The likely cause is a low-latency streaming failure mode: with `lm=1`, the model
is called every small audio increment, often with empty retrieved refs and a long
history. On `2022.acl-long.367`, the generator gets stuck near a
position-encoding phrase and keeps emitting it through the tail of the audio.

## Recommended Follow-Up

Do not mix this row with a different prompt policy. If changing empty-retrieval
behavior from `term_map:\nNONE` to omitted term-map, rerun all comparable rows
under the same policy.

For diagnosis, rerun `de/lm=1` only with one of:

- same settings but serial/non-batch, to test stochastic or batch sensitivity;
- empty retrieval omits the term-map block, to match training if training omitted
  term maps for empty chunks;
- a repetition guard that suppresses identical normalized chunk outputs after a
  small repeat count, then rerun all paper-facing rows if the guard is adopted.

## Empty-Term-Map Omission Rerun

I ran the second diagnostic option with the same DE tagged-ACL RASST settings
but changed empty retrieval prompts from explicit `term_map:\nNONE` to omitting
the term-map text block.

Event:
`20260525T0518__simuleval__tagged_acl_tmsft_gtwrap_de_lm1_omit_emptytm_hn1024_tau078_batch_aries01`

Result:

| policy | lm | BLEU | StreamLAAL | StreamLAAL_CA | TERM_ACC |
| --- | ---: | ---: | ---: | ---: | ---: |
| `term_map:\nNONE` | 1 | 20.6877 | 823.8605 | 1728.1666 | 0.7872 (736/935) |
| omit empty term map | 1 | 24.0580 | 899.4127 | 1515.8970 | 0.8171 (764/935) |

The pathological `2022.acl-long.367` talk is repaired by this prompt-policy
change:

| policy | pred words | ref words | ratio | repeated 8-gram rate | mixed tail |
| --- | ---: | ---: | ---: | ---: | --- |
| `term_map:\nNONE` | 2707 | 1327 | 2.04 | 0.539 | yes |
| omit empty term map | 1386 | 1327 | 1.04 | 0.000 | no |

This confirms that explicit `term_map:\nNONE` is a material trigger for the
`lm=1` decode loop. The overall BLEU remains below `lm=2/3`, so this is not the
only source of low-latency degradation; it is specifically a fix for the severe
looping failure mode.
