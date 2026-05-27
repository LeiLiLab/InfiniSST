# Handoff: Variable-Context Retriever Training And ACL/Medicine Eval

Date: 2026-05-17

This note summarizes the current state so a new Codex window can continue
without reconstructing the whole thread.

## Current Running Training

- Slurm job: `45227`
- Partition/node: `aries`
- Job name: `q3_varctx_v3_bs8k_gc128`
- W&B run: `lh1b88kw`
- W&B URL:
  `https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/lh1b88kw`
- Launcher:
  `documents/code/train/term_train/run_mfa_smallest_dense_hn1024_gsv2full_gsdedup_varctx_lmlb_v3_tcmoff_ep6_8gpu_aries.sh`
- Current checkpoint path tracked by the training run:
  `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries.pt`
- Current best-secondary checkpoint:
  `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt`
- Last observed best secondary:
  `eval_acl6060/recall@10=0.9912` at step `1840`.
- Best metric config:
  - Primary: `eval_dev/recall@10_gs10000`
  - Secondary: `eval_acl6060/recall@10`
- Important: the running job started before the latest code changes. It will not
  automatically pick up new defaults such as
  `EVAL_GLOSSARY_MATCH_MIN_NORM_CHARS=2`.

## Main Data Paths

Training data:

- `/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_varctx2p88_3p84_4p80_5p76.jsonl`
- Earlier diagnostic count: `7,554,926` rows.
- Context lengths are balanced among `2.88`, `3.84`, `4.80`, `5.76` seconds.
- These lengths correspond to original `0.96/1.92/2.88/3.84s` plus the
  `1.92s` inference look-back.

Dev eval:

- `/mnt/gemini/home/jiaxuanluo/term_dev_dataset_varctx2p88_3p84_4p80_5p76_new_version.jsonl`
- Dev gs10k glossary:
  `/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample10000.json`

ACL6060 eval:

- `/mnt/gemini/home/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary_varctx2p88_3p84_4p80_5p76/acl6060_dev_dataset.jsonl`
- Original ACL gs10k glossary:
  `/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/glossary_acl6060_gt_union_gs10000.json`
- Min-normalized-length-2 backfilled ACL gs10k glossary:
  `/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_gt_union_gs10000_min_norm2_backfill.json`

Medicine eval:

- Dataset:
  `/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76/medicine_dev_dataset.jsonl`
- GT plus medicine-wiki gs10k glossary:
  `/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76/medicine_glossary_gt_plus_medicine_wiki_gs10000.json`
- Medicine glossary source:
  `documents/code/data_pre/glossary_scale/wiki_glossary_medicine.json`
- Enriched medicine glossary:
  `documents/code/data_pre/glossary_scale/wiki_glossary_medicine_enriched.json`

## Data Leakage Filter

The train/eval term-key filter is now explicitly controlled and defaults to
non-strict mode.

- Python flag:
  `--strict_train_eval_term_filter`
- Launcher env:
  `STRICT_TRAIN_EVAL_TERM_FILTER=false`
- Configured exclusion glossaries in the v3 launcher:
  - `documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs_enriched.json`
  - `/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76/medicine_glossary_gt_plus_medicine_wiki_gs10000.json`
- These paths are passed/recorded, but no train rows are removed unless
  `STRICT_TRAIN_EVAL_TERM_FILTER=true`.
- Rationale: eval positives are chunk-conditioned. Extra glossary terms only
  become positives if they are found in `chunk_src_text` or match the sample's
  term metadata.

## Eval Positive Construction

Core code:

- `documents/code/train/term_train/qwen3_glossary_neg_train.py`
- `_build_glossary_positive_indices(...)` maps each sample to active-bank terms.
- It uses:
  - exact sample term metadata match
  - n-gram matching against `chunk_src_text`
- Expanded gs10k glossary terms are therefore positives only if they are
  actually present in the speech chunk transcript.

## Normalized-Length Filter

Problem found:

- Some ACL wiki-padding glossary terms normalize to one-character strings.
- Examples filtered now:
  `C`, `I*`, `J`, `F`, `H`, `A+`, `B*`, `E`, `F*`, `B`, `P`, `M`, `R`
- With old normalization, `A+` could match ordinary transcript token `a`, and
  `I*` could match ordinary token `i`.

Implemented control:

- Python flag:
  `--eval_glossary_match_min_norm_chars`
- Launcher env:
  `EVAL_GLOSSARY_MATCH_MIN_NORM_CHARS`
- New default: `2`
- Old behavior can be reproduced with:
  `EVAL_GLOSSARY_MATCH_MIN_NORM_CHARS=1`

Files changed:

- `documents/code/train/term_train/qwen3_glossary_neg_train.py`
- `documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh`

Implementation details:

- `eval_glossary_match_min_norm_chars=2` prevents transcript-text positive
  matching for normalized terms shorter than two chars.
- It also filters optional expansion candidates before embedding them.
- Base/paper-extracted terms like `qa`, `crf`, `asr` remain valid because their
  normalized length is at least two.

## ACL6060 Short-Term Filter Readout

Checkpoint used:

- `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt`

Runs:

- Old behavior, min chars 1:
  - W&B id: `evobq3ek`
  - URL:
    `https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/evobq3ek`
- Min chars 2, no backfill:
  - W&B id: `pootuyap`
  - URL:
    `https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/pootuyap`
  - Caveat: bank shrank to `9988` because 13 entries were removed.
- Min chars 2, gs10k backfilled:
  - W&B id: `58z87ec3`
  - URL:
    `https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/58z87ec3`
  - This is the fair comparison to old behavior because bank size remains
    `10000`.

Main comparison from W&B history, step `1840`:

| setting | W&B run | bank | base r@10 | gs10k r@10 | gs10k any-positive | gs10k text-match-positive | tau0.80 filtered R | tau0.80 P_micro | tau0.80 no-term noise |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| min=1 old | `evobq3ek` | 10000 | 0.9912 | 0.9463 | 2940 | 2554 | 0.9119 | 0.1661 | 2.1118 |
| min=2 backfill | `58z87ec3` | 10000 | 0.9912 | 0.9457 | 2751 | 2268 | 0.9113 | 0.1678 | 2.2652 |

Conclusion:

- Base recall is unchanged.
- gs10k recall changes by only `-0.0006`.
- gs10k chunk-positive labels drop from `2940` to `2751`.
- Text-match positives drop from `2554` to `2268`.
- The filter removes noisy one-character padding matches while keeping
  retrieval quality effectively stable.

## W&B/Tracking Notes

Repo rule:

- Use `documents/code/general/wandb_tool.py` for run reads and comparisons.
- Do not quote cross-run metrics from raw `run.summary` snapshots when a best
  checkpoint comparison is needed.

Useful commands:

```bash
python documents/code/general/wandb_tool.py --project qwen3_rag show lh1b88kw \
  --summary-regex '^(best|best_secondary|eval_acl6060/)'

python documents/code/general/wandb_tool.py --project qwen3_rag history evobq3ek \
  --keys _step eval_acl6060/recall@10 eval_acl6060/recall@10_gs10000 \
  eval_acl6060/gs10000_label_any_positive \
  eval_acl6060/gs10000_label_text_match_positive \
  eval_acl6060/topk10_chunk_any_positive_filtered_recall@tau_0p80_gs10000 \
  eval_acl6060/topk10_filtered_precision_micro@tau_0p80_gs10000 \
  eval_acl6060/noterm_noise@top10_tau_0p80_gs10000 \
  --samples 10
```

The readout runs were synced into SQLite with:

```bash
python documents/code/general/wandb_tool.py --project qwen3_rag db-sync \
  --runs evobq3ek pootuyap 58z87ec3 --best-bundles
```

## Direct Taurus Eval Notes

- Direct GPU eval works on taurus with:
  `/usr/bin/env CUDA_VISIBLE_DEVICES=<gpu> ...`
- The login shell may have `CUDA_VISIBLE_DEVICES` readonly/empty; use
  `/usr/bin/env` rather than simple assignment in the shell.
- For eval-only, set `--hard_neg_k_per_sample 0`; otherwise the script can try
  to initialize a negative bank from empty train samples and fail with:
  `AssertionError: No valid terms found for negative bank`.

## Validation Already Run

After changing code defaults:

```bash
/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python -m py_compile \
  documents/code/train/term_train/qwen3_glossary_neg_train.py

bash -n documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh

bash -n documents/code/train/term_train/run_mfa_smallest_dense_hn1024_gsv2full_gsdedup_varctx_lmlb_v3_tcmoff_ep6_8gpu_aries.sh
```

## Follow-Up Recommendations

1. For future eval/training launchers, let
   `EVAL_GLOSSARY_MATCH_MIN_NORM_CHARS=2` be the default.
2. For ACL gs10k eval, prefer the backfilled glossary path so bank size remains
   exactly `10000`.
3. Do not enable `STRICT_TRAIN_EVAL_TERM_FILTER` by default. Use it only for a
   conservative leakage ablation.
4. When reporting `lh1b88kw` or related runs, refresh from W&B history via
   `wandb_tool.py`, not markdown notes or stale SQLite rows.
5. The current running Aries job will not pick up the new default. Relaunch if
   the active run must use the min-normalized-length-2 behavior during inline
   evals.
