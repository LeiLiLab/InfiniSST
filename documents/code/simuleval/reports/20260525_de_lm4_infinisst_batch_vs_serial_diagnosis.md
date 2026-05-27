# En-De lm4 InfiniSST Batch vs Serial Diagnosis

Date: 2026-05-25

## Scope

Investigate why the same no-retriever En-De `lm=4` InfiniSST setting differs
between the serial SimulEval rerun and the same-LM batched run.

Serial event:

- `20260524T160830__simuleval__tagged_acl_origin_norag_de_lm4_raw_rerun`
- W&B run: `3upoqej5`
- Eval TSV: `/mnt/gemini/data1/jiaxuanluo/tagged_acl_origin_norag_de_lm4_raw_rerun_20260524T160830_tagacl_origin_norag_de_lm4_raw_rerun/origin_norag/de/gigaspeech-de-s_origin-bsz4_gacl6060_tagged_gt_raw_min_norm2_cs3.84_hs0.48_lm4_k210_k110_th0p0/eval_results.tsv`

Batch event:

- `20260524T2338__simuleval__tagged_acl_origin_norag_de_lm4_batch_max40_aries23`
- W&B run: `ixmu9jhv`
- Eval TSV: `/mnt/gemini/data1/jiaxuanluo/tagged_acl_origin_norag_de_lm4_batch_max40_aries23_20260524T2338_tagacl_origin_norag_de_lm4_batch_max40_aries23/origin_norag_de_lm4_batch_max40/de/dtagacl_origin_norag_batch_max40_lm4_k0_th0.0_gacl6060_tagged_gt_raw_min_norm2/eval_results.tsv`

## Result Difference

| mode | BLEU | TERM_ACC | TERM_CORRECT | TERM_TOTAL | StreamLAAL | StreamLAAL_CA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| serial | 33.30075964509326 | 0.6909 | 646 | 935 | 2824.4372171749956 | 4100.570443638962 |
| batch | 30.024288610842845 | 0.6738 | 630 | 935 | 2675.3593704178256 | 882.8662155653666 |

The denominator is identical (`TERM_TOTAL=935`), but generated hypotheses differ
for all five talks.

## Data Alignment Checks

- Source lists are byte-identical:
  - serial `dev.source.portable` sha256:
    `b0b9484da51930226cc6a8985129df6efa53f899cec310c790f196fb4581b50b`
  - batch `source.portable.list` sha256:
    `b0b9484da51930226cc6a8985129df6efa53f899cec310c790f196fb4581b50b`
- Both runs have five instance rows.
- References match by row index for all five rows.
- `source_length` matches by row index.
- Total vLLM input calls match: 899 serial and 899 batch.

Per-row generation comparison:

| idx | audio id | same ref | source length ms | serial words | batch words | same prediction |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 0 | `2022.acl-long.268.wav` | yes | 737440.0 | 1623 | 1626 | no |
| 1 | `2022.acl-long.367.wav` | yes | 695019.6875 | 1153 | 1326 | no |
| 2 | `2022.acl-long.590.wav` | yes | 577237.3125 | 1145 | 1154 | no |
| 3 | `2022.acl-long.110.wav` | yes | 703007.375 | 1564 | 1562 | no |
| 4 | `2022.acl-long.117.wav` | yes | 729013.6875 | 1772 | 1758 | no |

## Root Cause Evidence

The two runs are not prompt-equivalent in no-RAG mode.

Serial runtime prompt, first vLLM input:

- `term_map_count=0`
- `system_count=2`
- Prompt prefix:
  `You are a professional simultaneous interpreter. Your task is to translate English audio chunks into accurate and fluent German.`

Batch runtime prompt, first vLLM input:

- `term_map_count=2`
- `system_count=1`
- Prompt prefix:
  `You are a professional simultaneous interpreter. Your task is to translate English audio chunks into accurate and fluent German. Use the 'term_map' as a reference for terminology if provided.`
- User turn appends:
  `term_map:\nNONE`

Across runtime logs:

- Serial: `0/899` `llm_input` records contain `term_map`.
- Batch: `899/899` `llm_input` records contain `term_map`.
- Batch also logs 899 `rag` records with `rag_disabled=true` and empty
  references, so the retriever is disabled, but the prompt still carries the
  empty terminology scaffold.

Code paths:

- Serial no-RAG prompt is selected in
  `agents/infinisst_omni_vllm_rag_v4.py`: when `rag_enabled` is false, the
  system text does not include the term-map instruction, and `term_map:NONE` is
  only appended if RAG is enabled or real references exist.
- Batch prompt is selected in
  `documents/code/simuleval/src/batched_vllm_rag_eval.py`: it always adds the
  term-map instruction and always appends `term_map:NONE` when no references are
  available, including `--disable-rag`.

This is enough to change sampled decoding even with the same model, `lm`,
seed, max-new-tokens, temperature, top-p, and top-k.

## Additional Non-Equivalence

The vLLM scheduling is also different.

- Serial agent creates vLLM with `max_num_seqs=1`.
- Batch launcher sets `MAX_NUM_SEQS_OVERRIDE=5` and
  `SCHEDULER_BATCH_SIZE_OVERRIDE=5`.
- Batch driver calls `llm.generate(prepared, ...)` for multiple active streams
  in one request.

The decoding is stochastic (`temperature=0.6`, `top_p=0.95`, `top_k=20`).
For vLLM, a fixed seed does not make a multi-sequence batched request
bit-equivalent to five independent serial requests, because scheduling and RNG
consumption order can differ.

There is a possible smaller final-chunk difference:

- Serial pads any increment shorter than 15360 samples before vLLM.
- Batch sends the exact final increment from `last_vllm_samples:cursor_samples`.

This is secondary to the observed prompt mismatch and batched stochastic
scheduling, but it should be checked if prompt-compatible greedy decoding still
differs.

## Verdict

The batch run should not be used as a drop-in replacement for the serial lm4
no-RAG main result. The result difference is explained by driver
non-equivalence, primarily prompt mismatch in no-RAG mode and secondarily
batched stochastic vLLM scheduling.

## Recommended Follow-Up

Create a serial-compatibility mode for `batched_vllm_rag_eval.py`:

1. In `--disable-rag`, use the exact serial no-RAG system prompt.
2. In `--disable-rag`, suppress `term_map:NONE`.
3. Run with `--schedule-mode serial_by_lm`, `--scheduler-batch-size 1`, and
   `--max-num-seqs 1` for equivalence testing.
4. First test greedy decoding (`temperature=0`) to separate prompt/audio
   equivalence from stochastic RNG effects.
5. If greedy still differs after prompt/schedule compatibility, inspect final
   increment padding and SimulEval source buffering.
