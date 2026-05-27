## Hypothesis

The batched-vLLM driver can be made serial-compatible when TP, decode params,
cache policy, RAG checkpoint/glossary, and scheduling are pinned to the serial
SimulEval setting.

## Background / Motivation

The first batched prototype improved zh tagged ACL metrics too much versus
serial.  Inspection showed that the prototype used different decode defaults,
larger `max_num_seqs`, TP=8, fixed cache chunks, and round-robin scheduling.
This validation run intentionally removes those degrees of freedom before
treating batch output as an acceleration path.

## What changed vs baseline

- Uses the standalone batched driver, but `schedule_mode=serial_by_lm`.
- Pins `VLLM_TP_SIZE=2`, `max_num_seqs=1`, `scheduler_batch_size=1`,
  `VLLM_ENFORCE_EAGER=1`, and serial decode params.
- Uses serial cache policy: `MAX_CACHE_SECONDS=80`, `KEEP_CACHE_SECONDS=60`.
- Runs zh tagged ACL raw glossary, HN1024 retriever, tau=0.78, lm=2.
- Produces a prompt/output/delay alignment report against the serial lm=2 run.

## Expected metrics

Metrics should be close to the serial lm=2 reference.  Prompt and retrieval
alignment should match until generation diverges.  Any remaining gap must be
explained before using round-robin batch results as main-result truth.

## Verdict

Success.  With serial-compatible TP/env/decode/cache settings, the batched
driver exactly matches the serial lm=2 run on BLEU, TERM_ACC, REAL_TERM_ADOPT,
TERM_FCR, and StreamLAAL.  `StreamLAAL_CA` differs and is excluded from the
acceptance check.

Alignment report:
`/mnt/gemini/data1/jiaxuanluo/tagged_acl_batchvllm_serialcompat_hn1024_tau078_raw_zh_lm2_20260524T1159_tagacl_batchvllm_serialcompat_hn1024_tau078_raw_zh_lm2_taurus3/compare_serialcompat_lm2.md`
