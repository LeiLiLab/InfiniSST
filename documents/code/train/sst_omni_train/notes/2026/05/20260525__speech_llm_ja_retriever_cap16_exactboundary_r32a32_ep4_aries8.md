# Speech LLM Ja Retriever Cap16 Exact-Boundary, r32a32, Aries8

## Hypothesis

Japanese Speech LLM fine-tuning with HN1024 retriever-recalled term maps capped to 16 entries and exact assistant-side `<term>` wrapping should improve lm=1 stability by reducing negative term density while preserving enough retrieval exposure for terminology control.

## Parent data

Parent data event:

`20260525T0348__data_prepare__deja_termmap_ablation_cap16_exactboundary`

## Data

- Training JSONL:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_deja_termmap_ablation_cap16_exactboundary_20260525/ja/retriever_hn1024_tau078_cap16_exactboundary/train_s_ja_retriever_hn1024_tau078_cap16_gttermwrap_exactboundary.jsonl`
- Dev JSONL:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_deja_termmap_ablation_cap16_exactboundary_20260525/ja/retriever_hn1024_tau078_cap16_exactboundary/dev_s_ja_retriever_hn1024_tau078_cap16_gttermwrap_exactboundary_first355.jsonl`
- Validation:
  - train rows: 12500
  - train chunks: 53352
  - train term_map chunks: 48197
  - train term_map chunk rate: 0.9033775678512521
  - max term_map entries: 16
  - tagged rows: 11091
  - malformed tag messages: 0
  - latin boundary cut messages: 0

## Training config

The launcher follows the De cap16 exact-boundary setup:

- LoRA `r32a32`
- 8 GPUs on Aries
- EP=4, TP=1
- global batch size 8, micro batch size 1
- one epoch
- `max_length=3072` to avoid strict preprocessing failures from long cap16 rows

## Expected eval

After training and HF export, first run Japanese tagged ACL raw RASST at `lm=2` with HN1024, tau=0.78, batch eval, `max_new_tokens=80`, `VLLM_LIMIT_AUDIO=128`, and `VLLM_MAX_MODEL_LEN=12288`. If lm=2 is usable, continue with `lm=1,3,4`.

## Verdict

Queued as an Aries 8-GPU idle-watched job. The watcher waits until GPUs 0-7 all fall below the configured memory threshold before starting training.
