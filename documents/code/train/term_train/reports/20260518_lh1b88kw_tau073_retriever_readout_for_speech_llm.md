# lh1b88kw tau=0.73 retriever readout for speech LLM background

## Short Takeaway

Use `tau=0.73` as the recall-preserving retriever gate for downstream speech
LLM exploration.

- Retriever checkpoint: `lh1b88kw` secondary checkpoint at step 2640.
- Context: `lm=1,2,3,4` with 1.92s lookback-derived variable-context eval
  windows (`2.88/3.84/4.80/5.76s`, eval fixed audio `5.76s`).
- Selection rule: choose tau from dev only; ACL, tagged ACL, and medicine are
  held-out readouts.
- Reason: downstream speech LLM can remove some noisy candidates, but a missed
  glossary term cannot be recovered.

## Checkpoint

```text
/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt
```

Key eval config:

```text
audio_encoder_preset=qwen3-omni
audio_model_id=Atotti/Qwen3-Omni-AudioTransformer
audio_feature_extractor_id=openai/whisper-large-v3
text_encoder_preset=bge-m3
text_model_id=BAAI/bge-m3
text_pooling=cls
use_maxsim=true
maxsim_windows=2 3 4 5 6 7 8 10 12 16 20 24
maxsim_stride=2
mfa_window_selection=smallest
term_id_normalize=aggressive
eval_glossary_match_min_norm_chars=2
hard_neg_k_per_sample=0
```

## Dev Calibration

Raw tau=0.0 dev mean recall@10 = `98.928%`.

| bank | raw recall@10 |
| --- | ---: |
| base | 99.204 |
| gs10k | 98.973 |
| gs100k | 98.607 |

Selected tau: `0.73`.

| metric | value |
| --- | ---: |
| max dev recall drop | 0.486 pp |
| mean dev recall drop | 0.273 pp |
| dev P_micro mean | 12.68 |
| dev P_macro mean | 15.30 |

W&B source for dev + paper ACL sweep: `4g108a3w`.

## Readout Tables

Values are `Recall / P_micro / noise`. Recall and precision are percentages.
Noise is `noterm_noise@top10_tau_*`, average kept candidates on no-term chunks.

### ACL6060 Paper-Extracted Glossary

| tau | base | gs1k | gs10k |
| --- | ---: | ---: | ---: |
| 0.73 | 95.86 / 34.06 / 0.57 | 96.50 / 15.99 / 2.53 | 95.80 / 9.90 / 8.08 |

W&B: `4g108a3w`.

### Tagged ACL Glossary

| tau | base | gs1k | gs10k |
| --- | ---: | ---: | ---: |
| 0.73 | 97.92 / 19.08 / 1.01 | 97.95 / 14.96 / 1.88 | 97.90 / 10.15 / 7.19 |
| 0.75 | 97.52 / 21.31 / 0.73 | 97.54 / 17.38 / 1.26 | 97.67 / 10.68 / 5.49 |
| 0.78 | 96.60 / 25.08 / 0.46 | 96.63 / 22.03 / 0.70 | 97.19 / 12.88 / 3.04 |

W&B: `nrxiasfm`.

### Medicine Strict MFA-Only Glossary

| tau | base | gs1k | gs10k |
| --- | ---: | ---: | ---: |
| 0.73 | 92.57 / 13.49 / 2.34 | 92.52 / 13.06 / 2.57 | 92.15 / 11.41 / 3.73 |
| 0.75 | 91.11 / 14.93 / 1.83 | 91.11 / 14.53 / 1.95 | 90.86 / 12.75 / 2.74 |
| 0.78 | 89.00 / 18.43 / 1.14 | 89.00 / 18.13 / 1.18 | 89.00 / 16.21 / 1.53 |

W&B: `qjy4m1x9`.

Note: medicine gs10k has `9994` active terms after
`eval_glossary_match_min_norm_chars=2` filters six expansion terms.

## Data And Glossary Paths

### Dev Calibration

| role | path |
| --- | --- |
| dev JSONL | `/mnt/gemini/home/jiaxuanluo/term_dev_dataset_varctx2p88_3p84_4p80_5p76_new_version.jsonl` |
| dev glossary | `/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample1000000.json` |
| dev banks | `base`, `gs10000`, `gs100000` |

### ACL6060 Paper-Extracted Glossary

| role | path |
| --- | --- |
| eval JSONL | `/mnt/gemini/home/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary_varctx2p88_3p84_4p80_5p76/acl6060_dev_dataset.jsonl` |
| eval glossary | `/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_gt_union_gs10000_min_norm2_backfill.json` |
| banks | `base`, `gs1000`, `gs10000` |

### Tagged ACL Glossary

| role | path |
| --- | --- |
| eval JSONL | `/mnt/gemini/home/jiaxuanluo/acl6060_dev_offline_eval_tagged_glossary_varctx2p88_3p84_4p80_5p76/acl6060_tagged_dev_dataset.jsonl` |
| eval glossary | `/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_union_gs10000_min_norm2_backfill.json` |
| source tagged glossary | `/home/jiaxuanluo/InfiniSST/documents/data/data_pre/glossary_acl6060.json` |
| source processing script | `/home/jiaxuanluo/InfiniSST/documents/data/data_pre/handle_tagged_glossary.py` |
| banks | `base`, `gs1000`, `gs10000` |

### Medicine Strict MFA-Only

| role | path |
| --- | --- |
| eval JSONL | `/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/medicine_dev_dataset.jsonl` |
| eval glossary | `/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/medicine_glossary_gt_plus_medicine_wiki_gs10000.json` |
| data-prep manifest | `documents/code/data_pre/training_terms_for_retriever/manifests/2026/05/20260518T1812__data_prepare__medicine_varctx_clean_mfa_exact_only.json` |
| renewed translation source | `/home/jiaxingxu/rag-sst/eso-dataset/outputs_v2/test` |
| MFA TextGrid source | `/home/jiaxingxu/rag-sst/eso-dataset/mfa_v1/textgrids` |
| banks | `base`, `gs1000`, `gs10000` |

Medicine cleaning rule:

```text
unmatched term policy = drop
allowed locate methods = mfa_exact only
```

The strict dataset keeps only terms with MFA-exact matches, so target-language
or fallback-only term labels do not become offline-eval positives.

## Eval Launchers And Provenance

| purpose | launcher | W&B | manifest |
| --- | --- | --- | --- |
| dev calibration + paper ACL readout | `documents/code/train/term_train/launchers/2026/05/20260517__lh1b88kw_s2640_tau_delta_dev_acl_med_aries1_eval.sh` | `4g108a3w` | see note below |
| tagged ACL tau3 readout | `documents/code/train/term_train/launchers/2026/05/20260518__lh1b88kw_s2640_tagged_acl_tau073_075_078_aries1_eval.sh` | `nrxiasfm` | `documents/code/train/term_train/manifests/2026/05/20260518T2115__retriever_eval__lh1b88kw_s2640_tagged_acl_tau073_075_078.json` |
| medicine strict tau3 readout | `documents/code/train/term_train/launchers/2026/05/20260518__lh1b88kw_s2640_medicine_tau073_075_078_strict_aries1_eval.sh` | `qjy4m1x9` | `documents/code/train/term_train/manifests/2026/05/20260518T2052__retriever_eval__lh1b88kw_s2640_medicine_tau073_075_078_strict.json` |
| combined dev/ACL/tagged/medicine strict launcher | `documents/code/train/term_train/launchers/2026/05/20260518__lh1b88kw_s2640_tau_delta_dev_acl_tagged_acl_med_aries1_eval.sh` | superseded for tau3 readouts | `documents/code/train/term_train/manifests/2026/05/20260518T2037__retriever_eval__lh1b88kw_s2640_tau_delta_dev_acl_tagacl_medstrict.json` |

Notes:

| purpose | notes |
| --- | --- |
| dev + paper ACL + old medicine tau-delta | `documents/code/train/term_train/notes/2026/05/20260517__lh1b88kw_s2640_tau_delta_dev_acl_med_eval.md` |
| tagged ACL tau3 | `documents/code/train/term_train/notes/2026/05/20260518__lh1b88kw_s2640_tagged_acl_tau073_075_078_eval.md` |
| medicine strict tau3 | `documents/code/train/term_train/notes/2026/05/20260518__lh1b88kw_s2640_medicine_tau073_075_078_strict_eval.md` |

Log files for latest tau3 readouts:

```text
/mnt/gemini/data1/jiaxuanluo/logs/direct_lh1b88kw_tagged_acl_tau3_taurus45269_20260518T211619.err
/mnt/gemini/data1/jiaxuanluo/logs/direct_lh1b88kw_medicine_tau3_strict_taurus45269_20260518T210225.err
```

## Reproduction Commands

Use taurus hold job `45269` if it is still alive and idle.

Tagged ACL:

```bash
cd /mnt/taurus/home/jiaxuanluo/InfiniSST

RUN_ID=direct_lh1b88kw_tagged_acl_tau3_taurus45269_$(date -u +%Y%m%dT%H%M%S)
LOG_DIR=/mnt/gemini/data1/jiaxuanluo/logs
TMP_DIR=/mnt/gemini/data1/jiaxuanluo/tmp/${RUN_ID}
LAUNCH=/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/launchers/2026/05/20260518__lh1b88kw_s2640_tagged_acl_tau073_075_078_aries1_eval.sh

mkdir -p "$LOG_DIR" "$TMP_DIR"
srun --jobid=45269 --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST \
  env RUN_STAMP="$RUN_ID" MASTER_PORT=30385 LOCAL_TMP_DIR="$TMP_DIR" \
  NUM_WORKERS=0 SELECT_CLEAN_GPUS=true TCM_SWEEP_THRESHOLDS="0.73 0.75 0.78" \
  bash "$LAUNCH" \
  > "${LOG_DIR}/${RUN_ID}.out" \
  2> "${LOG_DIR}/${RUN_ID}.err"
```

Medicine strict:

```bash
cd /mnt/taurus/home/jiaxuanluo/InfiniSST

RUN_ID=direct_lh1b88kw_medicine_tau3_strict_taurus45269_$(date -u +%Y%m%dT%H%M%S)
LOG_DIR=/mnt/gemini/data1/jiaxuanluo/logs
TMP_DIR=/mnt/gemini/data1/jiaxuanluo/tmp/${RUN_ID}
LAUNCH=/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/launchers/2026/05/20260518__lh1b88kw_s2640_medicine_tau073_075_078_strict_aries1_eval.sh

mkdir -p "$LOG_DIR" "$TMP_DIR"
srun --jobid=45269 --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST \
  env RUN_STAMP="$RUN_ID" MASTER_PORT=30384 LOCAL_TMP_DIR="$TMP_DIR" \
  NUM_WORKERS=0 SELECT_CLEAN_GPUS=true TCM_SWEEP_THRESHOLDS="0.73 0.75 0.78" \
  bash "$LAUNCH" \
  > "${LOG_DIR}/${RUN_ID}.out" \
  2> "${LOG_DIR}/${RUN_ID}.err"
```

## Speech LLM Exploration Notes

- Treat `tau=0.73` as the default retrieval threshold for recall-preserving
  speech LLM training/eval.
- Prefer reporting both ACL paper-extracted glossary and tagged ACL glossary:
  they behave differently in precision/noise but both preserve high recall.
- Keep medicine strict MFA-only as the cross-domain stress readout; do not mix
  in fallback or non-MFA-matched medicine positives.
- If the speech LLM is robust to noise, `gs10k` is reasonable because recall is
  still high; if prompt budget or hallucinated adoption is a concern, compare
  `gs1k` vs `gs10k` downstream rather than tightening tau first.
- Do not use ACL/tagged ACL/medicine to reselect tau; they are readouts after
  dev calibration.
