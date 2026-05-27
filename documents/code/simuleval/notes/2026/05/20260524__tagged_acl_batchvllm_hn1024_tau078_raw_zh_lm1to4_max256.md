## Hypothesis

After serial-compatible validation, the batched-vLLM driver can run the zh
tagged ACL raw-glossary sweep with higher throughput while preserving BLEU and
term metrics.  `StreamLAAL_CA` is not used as a pass/fail metric for this batch
path.

## Background / Motivation

The batch prototype initially produced better metrics because its decode/cache
environment differed from serial SimulEval.  The validation path now fixes TP,
RAG, cache, decode, prompt construction, and character-level instance output
normalization.  This run changes only the throughput schedule and increases
`max_new_tokens` to avoid truncating `<term>...</term>`-heavy outputs.

## What changed vs baseline

- Reuses the New V9 zh speech LLM and HN1024 retriever at tau=0.78.
- Uses raw tagged ACL glossary and evaluates `lm=1,2,3,4` in one shared vLLM.
- Keeps vLLM TP=2 plus a separate retriever GPU.
- Uses `schedule_mode=round_robin`, `max_num_seqs=8`, `scheduler_batch_size=8`.
- Sets `max_new_tokens=256` instead of the serial-compatibility value `40`.
- Keeps `MAX_CACHE_SECONDS=80` and `KEEP_CACHE_SECONDS=60`.
- Uses `VLLM_LIMIT_AUDIO=96`; the initial 64-audio attempt failed on lm=1
  once the long cache exceeded vLLM's per-prompt audio item limit.

## Expected metrics

BLEU, TERM_ACC, REAL_TERM_ADOPT, TERM_FCR, and StreamLAAL should be close to
serial-compatible output for the same model/RAG setting.  `StreamLAAL_CA` may
change because the batched driver has different wall-clock compute behavior and
is not treated as a reliable reported metric here.

## Verdict

Success.  The serial-compatible validation first matched serial exactly on
BLEU, TERM_ACC, REAL_TERM_ADOPT, TERM_FCR, and StreamLAAL.  The max-new-token
rerun then completed with `max_new_tokens=256` after raising
`VLLM_LIMIT_AUDIO` from 64 to 96.

| lm | BLEU | TERM_ACC | REAL_ADOPT | TERM_FCR | StreamLAAL |
|---:|---:|---:|---:|---:|---:|
| 1 | 44.04 | 85.62% | 88.90% | 10.98% | 1243 |
| 2 | 49.83 | 88.88% | 91.02% | 12.41% | 1860 |
| 3 | 49.15 | 87.87% | 90.15% | 9.01% | 2370 |
| 4 | 50.35 | 90.22% | 93.31% | 7.41% | 2773 |

Summary:
`/mnt/gemini/data1/jiaxuanluo/tagged_acl_batchvllm_hn1024_tau078_raw_zh_lm1to4_max256_20260524T1311_tagacl_batchvllm_hn1024_tau078_raw_zh_lm1to4_max256_audio96_taurus675/new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078_batchvllm_max256/zh/summary_metrics_max256_audio96.md`
