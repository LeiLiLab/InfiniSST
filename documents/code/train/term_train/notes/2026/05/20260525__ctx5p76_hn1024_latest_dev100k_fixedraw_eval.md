# Fixed 5.76s HN1024 Latest Dev100k Fixed-Raw Eval

## Hypothesis

The paused fixed `5.76s` HN1024 Qwen3-Omni checkpoint should provide the
longest-context control needed for dev-only comparison against variable context.
The eval uses the raw dev denominator while changing only the retriever bank
size from base to `1k`, `10k`, and `100k`.

## Background / Motivation

The training run `jyb2u787` was paused after its latest eval checkpoint at step
`1300`. Its in-run eval only covered `1k` and `10k`; this one-shot eval adds the
same dev `100k` bank so the fixed `5.76s` line can be compared directly with the
existing variable-context dev table.

## What changed vs baseline

- Parent run: `jyb2u787`.
- Checkpoint:
  `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8190_t=0.07_3var_gsv2full_gsdedup_ctx5p76_bs8190_gc128_eval100_tagacl_med_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_q3o_resume_s400_gpu012367_aries6_latest.pt`
- Eval dataset:
  `/mnt/gemini/home/jiaxuanluo/term_dev_dataset_ctx5p76_new_version.jsonl`
- Retriever expansion glossary:
  `/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample1000000.json`
- Eval sizes: `1000 10000 100000`.
- Metric denominator: `fixed_raw`.
- Scope: dev only; no ACL, tagged ACL, or medicine readout.

## Expected metrics

Report dev `recall@10`, `recall@10_gs1000`, `recall@10_gs10000`, and
`recall@10_gs100000` from W&B/logs. This is a diagnostic control, not a new
checkpoint-selection run.

## Verdict

CANCELLED on `2026-05-25T01:05:22+00:00`. This launch used the 1M dev glossary
file with `eval_glossary_sizes=1000 10000 100000`; the current eval
implementation encodes the entire supplied wiki-term file before slicing to the
requested sizes. The run was stopped and replaced by a rerun that points
directly to the 100k glossary file.
