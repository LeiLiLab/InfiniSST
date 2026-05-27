## Hypothesis

The clean de New V9 Speech LLM rebuilt from MFA/source-filtered GT should be usable as the German main-line tagged ACL raw-glossary readout with HN1024 at tau 0.78.

## Background / Motivation

The earlier de New V9 data path was rejected because GT construction used noisy term-map/fuzzy matching. This eval uses the newly trained clean de model:

`/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_new_v9_mfa_npfilter_srcunion_lexexact_oldnewv3_de_r32a64_tp2_taurus8/keep1.0_r32/v0-20260525-001708-hf`

## What changed vs baseline

Compared with the previous de New V9 tagged ACL run, the Speech LLM checkpoint comes from the clean `mfa_npfilter_srcunion_lexexact` training set. The retriever and eval protocol stay fixed:

- language: `de`
- latency multipliers: `1 2 3 4`
- runtime glossary: raw tagged ACL glossary
- metric denominator: fixed raw tagged ACL glossary
- retriever: HN1024 `lh1b88kw`
- tau: `0.78`
- timeline lookback: `1.92s`
- max new tokens: `80`
- strip assistant-only `<term>...</term>` tags before metrics

## Expected metrics

The run should produce BLEU, TERM_ACC, REAL_TERM_ADOPT, TERM_FCR, and StreamLAAL for `lm=1,2,3,4`. Eval uses the same-lm batch path: each `lm` runs all five ACL talks together in one vLLM process, instead of launching one process per talk. The main diagnostic is whether clean de SFT recovers term adoption without the polluted-GT artifacts.

## Verdict

Completed.

The de clean New V9 MFA/source-filtered model exported correctly and the tagged ACL raw-glossary batch eval completed for `lm=1,2,3,4` with `max_new_tokens=80`. The first `lm=1` attempt hit vLLM's 64-audio prompt limit, then succeeded after rerun with `VLLM_LIMIT_AUDIO_OVERRIDE=128`. The `lm=2/3/4` wrapper emitted a shell EOF after writing `eval_results.tsv`, so those runs may not have W&B run ids, but their TSV metrics are present.

The initial same-lm batch post-eval produced negative `StreamLAAL` for de because the batch writer concatenated German chunk outputs without inserting word boundaries. This is invalid for word-level latency: the delay trace counts words per generated chunk, while `prediction.split()` after raw concatenation merged words across chunk edges. The batch evaluator is now fixed to insert chunk-boundary spaces for `lang_code=de` and fail fast if prediction units and delay units disagree. Existing de runtime outputs were repaired from `llm_output` records and rescored under `wordfix_eval/`.

Summary:

| lm | BLEU | TERM_ACC | REAL_ADOPT | TERM_FCR | StreamLAAL | TERM |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 26.25 | 84.28 | 86.83 | 21.57 | 1015.7 | 788/935 |
| 2 | 29.93 | 85.99 | 87.70 | 20.16 | 1671.3 | 804/935 |
| 3 | 30.81 | 85.56 | 85.88 | 16.45 | 2157.6 | 800/935 |
| 4 | 31.70 | 84.92 | 87.26 | 14.61 | 2718.3 | 794/935 |

Artifacts:

- Summary TSV: `/mnt/gemini/data1/jiaxuanluo/tagged_acl_same_lm_batch_v1_mfa_npfilter_hn1024_tau078_raw_de_lm1to4_20260524T1738_tagacl_newv9_mfa_npfilter_de_batch_mt80/new_v9_mfa_npfilter_lexexact_oldnewv3_de_r32a64_hn1024_tau078_same_lm_batch_v1/__summary__/summary_de_raw_lm1to4_same_lm_batch_v1_max80_wordfix.tsv`
- Summary MD: `/mnt/gemini/data1/jiaxuanluo/tagged_acl_same_lm_batch_v1_mfa_npfilter_hn1024_tau078_raw_de_lm1to4_20260524T1738_tagacl_newv9_mfa_npfilter_de_batch_mt80/new_v9_mfa_npfilter_lexexact_oldnewv3_de_r32a64_hn1024_tau078_same_lm_batch_v1/__summary__/summary_de_raw_lm1to4_same_lm_batch_v1_max80_wordfix.md`
- Main-result TSV: `documents/code/simuleval/reports/20260524_main_result_data.tsv`
- Paper figure: `documents/code/train/term_train/reports/EMNLP_26_InfiniSST_RAG_src/latex/figures/new_main_result_tagged.pdf`
