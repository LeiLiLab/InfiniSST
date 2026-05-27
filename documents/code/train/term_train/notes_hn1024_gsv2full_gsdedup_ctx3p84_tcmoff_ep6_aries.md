# Full GSV2 k1024 TCM-off with 3.84s GigaSpeech context

- **Family / data / task**: `sst_ood_hardneg` / `3variant_gsv2full_gsfix_mfa_gsdedup_ctx384` / `train`
- **Variant tag**: `hn1024_gsv2full_gsdedup_ctx384_tcmoff_ep6`
- **Launcher**: `documents/code/train/term_train/run_mfa_smallest_dense_hn1024_gsv2full_gsdedup_ctx3p84_tcmoff_ep6_8gpu_aries.sh`
- **Data builder**: `documents/code/data_pre/training_terms_for_retriever/run_expand_gsv2full_gsdedup_gsctx3p84.sh`
- **Primary comparison baseline**: `ah9u1bao` is the direct 1.92s GSV2-dedup TCM-off ep3 run; `7xu2b4so` is the strongest same-family GSV2-dedup continuation baseline from WandB pre-flight.

## Hypothesis

Expanding GigaSpeech retriever chunks from 1.92s to 3.84s should give the
audio encoder more lexical context around each MFA term event. If the current
1.92s windows are too tight for some phrase terms or coarticulation patterns,
the wider context should improve dev recall without requiring TCM.

## Background / Motivation

The current deduped full-GSV2 run keeps one random row per absolute GigaSpeech
MFA term event, but each real-speech row still uses a 1.92s audio chunk. This
run tests whether a wider real-speech context is beneficial while preserving
the same hard-negative depth, MaxSim MFA supervision, dev selection setting,
and wiki-synth mixture.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/ah9u1bao
  - Secondary same-family baseline URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/7xu2b4so
- Diff:
  - data: `/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup.jsonl` -> `/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_gsctx3p84.jsonl`
  - data preprocessing: for every existing GigaSpeech `(utter_id, chunk_idx)` group, cut a 3.84s context window from the original GigaSpeech opus using the MFA SQLite index; include every known term event whose MFA span overlaps that wider window so newly covered terms are positives, not hard-negative false negatives
  - wiki-synth policy: `wiki_synth_` rows are recut to real 3.84s chunks from the original TTS WAVs and wiki-synth MFA TextGrids inferred from each 1.92s chunk path
  - audio length: train and inline eval fixed waveform length `1.92s` -> `3.84s`
  - eval data: dev uses the existing latency-multiplier-4 JSONL `/mnt/gemini/data1/jiaxuanluo/term_dev_dataset_m4.jsonl`; ACL uses a regenerated extracted-paper JSONL with 3.84s chunks and 1.92s stride from cached MFA TextGrids/full-wave sources
  - hparams: `epochs=3` -> `epochs=6`; `scheduler_epochs=4` -> `scheduler_epochs=6`; effective batch remains `8 * 1536 = 12288`; `hard_neg_k_per_sample=1024`; `TCM=off`
  - memory control: `grad_cache_chunk_size=512` -> `256` to keep peak activation memory reasonable with the doubled audio time axis
  - eval selection: unchanged dev gs10000 primary and tau0.80 filtered recall secondary

## Expected metrics

The run is useful if dev `recall@10_gs10000` improves over the 1.92s context
line or if tau0.80 filtered recall improves without a material recall drop.
The main failure mode to watch is higher no-term/noise behavior from the wider
speech context.

## Verdict

PENDING: update after training finishes and compare best-step dev bundles using
`wandb_tool.py compare --at-best-step` with both primary and secondary anchors.
