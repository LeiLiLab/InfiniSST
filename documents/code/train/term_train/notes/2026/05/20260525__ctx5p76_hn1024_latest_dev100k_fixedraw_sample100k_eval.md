# Fixed 5.76s HN1024 Latest Dev100k Fixed-Raw Eval, Sample100k

## Hypothesis

The paused fixed `5.76s` HN1024 Qwen3-Omni checkpoint should provide the
longest-context control needed for dev-only comparison against variable context.
The eval keeps the raw dev denominator fixed and changes only the retriever bank
size from base to `1k`, `10k`, and `100k`.

## Background / Motivation

The first `20260525T005027` launch was cancelled because it supplied the 1M
wiki-term file; the eval implementation encodes the entire supplied file before
slicing to requested bank sizes. This corrected run points directly to the dev
100k glossary file.

## What changed vs baseline

- Parent run: `jyb2u787`.
- Checkpoint:
  `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8190_t=0.07_3var_gsv2full_gsdedup_ctx5p76_bs8190_gc128_eval100_tagacl_med_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_q3o_resume_s400_gpu012367_aries6_latest.pt`
- Eval dataset:
  `/mnt/gemini/home/jiaxuanluo/term_dev_dataset_ctx5p76_new_version.jsonl`
- Retriever expansion glossary:
  `/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample100000.json`
- Eval sizes: `1000 10000 100000`.
- Metric denominator: `fixed_raw`.
- Scope: dev only; no ACL, tagged ACL, or medicine readout.

## Expected metrics

Report dev `recall@10`, `recall@10_gs1000`, `recall@10_gs10000`, and
`recall@10_gs100000` from W&B/logs. This is a diagnostic control, not a new
checkpoint-selection run.

## Verdict

SUCCESS as W&B run `yw4ykulb`. The corrected run used the 100k glossary file
and completed dev-only eval at checkpoint step `1300`.

| domain | base | 1k | 10k | 100k |
| --- | ---: | ---: | ---: | ---: |
| dev | 0.9964 | 0.9964 | 0.9958 | 0.9933 |

The metric denominator was fixed raw dev (`eval_metric_denominator=fixed_raw`),
with `3574` label-positive dev samples and `3520` text-match positives for all
expanded banks.
