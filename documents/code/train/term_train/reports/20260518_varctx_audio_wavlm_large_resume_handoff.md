# Varctx WavLM-Large Resume Handoff

Date: 2026-05-18 UTC

## Goal

Resume the varctx576 retriever training with `microsoft/wavlm-large` as the
audio encoder and BGE-M3 as the text encoder.  The resume seed is the current
WavLM best checkpoint from run `qou1rg53`.  The resumed run should evaluate
every 100 training steps, not every 200 steps as in the source run.

This is a continuation of the WavLM speech-encoder ablation, not a text-encoder
swap.  Keep the WavLM-specific stability fixes:

- WavLM model id: `microsoft/wavlm-large`
- WavLM `get_input_embeddings()` shim for PEFT checkpointing
- WavLM `config.layerdrop=0.0`
- WavLM retriever DDP `find_unused_parameters=True`
- dynamic MFA/MaxSim frame-time mapping

## Source Run State

- Source W&B run: `qou1rg53`
- Source W&B URL:
  `https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/qou1rg53`
- Source Slurm job: `45265`
- Source job name: `q3_vctx_wavl_fu_b8g128`
- Source launcher:
  `documents/code/train/term_train/launchers/2026/05/20260518__varctx_lmlb_v3_audio_wavlm_large_taurus8_bs8k_gc128_eval200_layerdrop0_findunused_full.sh`
- Source manifest:
  `documents/code/train/term_train/manifests/2026/05/20260518T0247__retriever_train__varctx_audio_wavlm_large_taurus8_bs8k_gc128_eval200_layerdrop0_findunused_full.json`
- Source notes:
  `documents/code/train/term_train/notes/2026/05/20260518__varctx_lmlb_v3_audio_wavlm_large_taurus8_bs8k_gc128_eval200_layerdrop0_findunused_full.md`
- Current status at handoff creation:
  `qou1rg53` is running and had reached at least step 200.  It completed inline
  dev, ACL, and medicine eval at step 200 and wrote the primary and secondary
  best checkpoints.

The source run used `EVAL_STEPS_SAMPLE=200`.  The resume run should override
this to `100`.

## Checkpoint Artifact

Do not include the checkpoint model in the data package.  Transfer or reference
it separately:

```text
/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_aud_wavl_fu_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_dev100Tau1_eval200_taurus8_best.pt
```

At handoff time this file exists and is about 2.8GB.  Because the source run is
still training, the `_best.pt` path can be overwritten if a later eval improves
the primary metric.  If exact step-200 reproducibility matters, snapshot this
file under a versioned name before transferring it.  If the goal is simply to
resume from the latest best WavLM checkpoint, copy the path above after the
source run reaches the desired handoff point.

The source secondary checkpoint is:

```text
/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_aud_wavl_fu_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_dev100Tau1_eval200_taurus8_best_eval_acl6060_recallat10.pt
```

Use the primary `_best.pt` above for `RESUME` unless explicitly doing an ACL
readout-only continuation.  ACL is held-out readout and must not select the
resume checkpoint for the paper protocol.

## Control Files

Use the source WavLM launcher as the starting point, but create new resume
control files instead of editing the source event in place:

- Source launcher to copy:
  `documents/code/train/term_train/launchers/2026/05/20260518__varctx_lmlb_v3_audio_wavlm_large_taurus8_bs8k_gc128_eval200_layerdrop0_findunused_full.sh`
- New launcher suggestion:
  `documents/code/train/term_train/launchers/2026/05/20260518__varctx_lmlb_v3_audio_wavlm_large_taurus8_bs8k_gc128_resume_best_eval100_full.sh`
- New notes suggestion:
  `documents/code/train/term_train/notes/2026/05/20260518__varctx_lmlb_v3_audio_wavlm_large_taurus8_bs8k_gc128_resume_best_eval100_full.md`
- New manifest suggestion:
  `documents/code/train/term_train/manifests/2026/05/20260518Txxxx__retriever_train__varctx_audio_wavlm_large_taurus8_bs8k_gc128_resume_best_eval100_full.json`

Before launching, copy the source manifest and update at least:

- `event_id`
- `variant`
- `status`
- `wandb_run_id`
- `slurm_job_id`
- `launcher_path`
- `notes_path`
- `command_template`
- `parent_run_ids`
- `artifacts`
- `metadata.eval_steps_sample`
- `metadata.resume_checkpoint`

Do not relaunch with the old running manifest.

## Required Resume Configuration

Set or verify these launcher environment values:

```bash
export AUDIO_ENCODER_PRESET="wavlm-large"
export AUDIO_ENCODER_TYPE="wavlm"
export AUDIO_MODEL_ID="microsoft/wavlm-large"
export AUDIO_FEATURE_EXTRACTOR_ID="microsoft/wavlm-large"
export AUDIO_INPUT_DTYPE="fp32"

export TEXT_ENCODER_PRESET="bge-m3"
export TEXT_MODEL_ID="BAAI/bge-m3"
export TEXT_INPUT_PREFIX=""
export TEXT_POOLING="cls"

export NUM_GPUS="8"
export PER_GPU_BATCH="1024"
export GRAD_CACHE_CHUNK_SIZE="128"
export MAX_STEPS="0"
export EPOCHS="6"
export SCHEDULER_EPOCHS="6"

export RESUME="/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_aud_wavl_fu_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_dev100Tau1_eval200_taurus8_best.pt"
export RESET_SCHEDULER="false"
export RESET_BEST_ON_RESUME="false"
export RESUME_COSINE_DECAY_TO_MAX_STEPS="false"

export EVAL_STEPS_SAMPLE="100"
export EVAL_SAMPLE_LIMIT="100"
export ACL_EVAL_SAMPLE_LIMIT="0"
export MEDICINE_EVAL_SAMPLE_LIMIT="0"
export EVAL_GLOSSARY_MATCH_MIN_NORM_CHARS="2"
export TCM_SWEEP_THRESHOLDS="0.75"
```

Recommended naming overrides:

```bash
export VARIANT_TAG="vctx576_aud_wavl_fu_resume_t8_b8k_g128"
export VERSION="3var_gsv2full_gsdedup_varctx576_aud_wavl_fu_bs8k_gc128_resume_best_eval100_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_taurus8"
export WANDB_EXP_NAME="variantE_varctx576_aud_wavl_fu_resume_best_taurus8_bs8k_gc128_dev100Tau1_eval100_full"
export EXTRA_WANDB_TAGS="variant:vctx576_aud_wavl_fu_resume_t8_b8k_g128 compute:taurus-8gpu"
export BASELINE_RUN_IDS="lh1b88kw qou1rg53"
```

Important resume behavior:

- The checkpoint stores `global_step` and `epoch`.  For the current step-200
  best checkpoint, the script will restore `global_step=200` and start from the
  next epoch recorded in the checkpoint.
- With `RESET_BEST_ON_RESUME=false`, the script restores the previous best
  tracker if the metric keys match.  A new `_best.pt` under the new save path
  will only be written after the resumed run improves over the restored best.
- If the new run must produce a fresh best artifact even without improvement,
  set `RESET_BEST_ON_RESUME=true`, but then document that the resumed run's
  best tracker is no longer a continuous tracker from `qou1rg53`.
- The first eval after resume with `EVAL_STEPS_SAMPLE=100` should occur at the
  next multiple of 100 after the restored global step.  If resuming exactly from
  step 200, expect the next inline eval at step 300, not immediately at step
  200.

## Launch Command

After creating the new resume manifest and notes, launch through the event
wrapper:

```bash
python documents/code/general/experiment_event.py launch \
  documents/code/train/term_train/manifests/2026/05/<new_wavlm_resume_manifest>.json \
  -- sbatch --parsable \
  documents/code/train/term_train/launchers/2026/05/20260518__varctx_lmlb_v3_audio_wavlm_large_taurus8_bs8k_gc128_resume_best_eval100_full.sh
```

After W&B init appears, write the new W&B run id and Slurm job id back into the
new manifest, re-register it, then sync:

```bash
python documents/code/general/experiment_event.py register \
  documents/code/train/term_train/manifests/2026/05/<new_wavlm_resume_manifest>.json
python documents/code/general/wandb_tool.py --project qwen3_rag db-sync --runs <new_run_id>
```

If W&B init fails, cancel the Slurm job.  Do not let an untracked resume run
continue.

## Transfer Package

The data package already exists and should be treated as data/control context,
not as a model-checkpoint package:

```text
/mnt/gemini/home/jiaxuanluo/transfer_packages/varctx_bgel_resume_data_20260517T232142Z.tar.zst
```

Path list:

```text
/mnt/gemini/home/jiaxuanluo/transfer_packages/varctx_bgel_resume_data_20260517T232142Z.paths.txt
```

The package currently includes the shared varctx train/dev/ACL/medicine data
and glossary artifacts.  It was originally named for the BGE-large handoff, but
the data artifacts are the same WavLM resume needs.  Do not add the WavLM
checkpoint to this package; list and transfer the checkpoint separately.

The `.sha256` file currently exists but is empty at handoff time, so regenerate
or verify checksum before relying on it:

```bash
sha256sum /mnt/gemini/home/jiaxuanluo/transfer_packages/varctx_bgel_resume_data_20260517T232142Z.tar.zst \
  > /mnt/gemini/home/jiaxuanluo/transfer_packages/varctx_bgel_resume_data_20260517T232142Z.tar.zst.sha256
```

Restore with original absolute mount layout:

```bash
tar -I zstd -xf varctx_bgel_resume_data_20260517T232142Z.tar.zst -C /
```

If extracting under another root, the archive creates `mnt/gemini/...` and
`mnt/taurus/...` relative to that root; then rewrite launchers and JSONL
`chunk_audio_path` values consistently.

## Data Artifacts

The JSONL files contain absolute `chunk_audio_path` values.  Preserve the same
mount paths on the target cluster or rewrite the JSONL paths.

| Role | Current path | Notes |
| --- | --- | --- |
| Train JSONL | `/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_varctx2p88_3p84_4p80_5p76.jsonl` | mandatory |
| Train audio | `/mnt/gemini/home/jiaxuanluo/term_train_audio_chunks_gsv2full_gsdedup_varctx2p88_3p84_4p80_5p76/` | mandatory |
| Dev JSONL | `/mnt/gemini/home/jiaxuanluo/term_dev_dataset_varctx2p88_3p84_4p80_5p76_new_version.jsonl` | dev sampled 100 inline |
| Dev audio | `/mnt/gemini/home/jiaxuanluo/term_dev_audio_chunks_varctx_m3/` | mandatory |
| ACL JSONL | `/mnt/gemini/home/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary_varctx2p88_3p84_4p80_5p76/acl6060_dev_dataset.jsonl` | held-out readout |
| ACL audio | `/mnt/gemini/home/jiaxuanluo/acl6060_dev_offline_eval_varctx2p88_3p84_4p80_5p76/audio_chunks/` | mandatory |
| Medicine JSONL | `/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76/medicine_dev_dataset.jsonl` | held-out/readout |
| Medicine audio | `/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76/audio_chunks/` | mandatory |
| Dev glossary | `/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample10000.json` | dev gs1k/gs10k |
| ACL glossary | `/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_gt_union_gs10000_min_norm2_backfill.json` | min-norm-2 backfilled gs10k |
| Medicine glossary | `/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76/medicine_glossary_gt_plus_medicine_wiki_gs10000.json` | medicine gs10k |
| Train exclusion glossary | `/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs_enriched.json` | used by current launcher |

## Path Rewrite On New Cluster

Fastest route: recreate mount-compatible paths with symlinks or bind mounts:

- `/mnt/taurus/home/jiaxuanluo/InfiniSST` -> target repo root.
- `/mnt/gemini/home/jiaxuanluo` -> target data root.
- `/mnt/gemini/data1/jiaxuanluo/logs` -> target log directory.

If mount-compatible paths are not possible, edit:

- Resume launcher: `REPO_ROOT`, SBATCH partition/GPU/memory/time/log paths,
  `BASE_LAUNCHER`, `NOTES_FILE`, `RESUME`, and glossary paths.
- Base launcher:
  `TRAIN_JSONL`, `DEV_JSONL`, `ACL_DEV_JSONL`, `MEDICINE_DEV_JSONL`,
  `EVAL_WIKI_GLOSSARY`, `MEDICINE_EVAL_WIKI_GLOSSARY`,
  `TRAIN_EXCLUDE_TERM_GLOSSARIES`, and the final sourced common launcher path.
- Common launcher:
  `CONDA_PREFIX`, `PYTHONPATH`, Hugging Face cache roots, `SCRIPT_PATH`,
  `SAVE_DIR`, and `WANDB_API_KEY`.
- JSONLs: rewrite every `chunk_audio_path` if audio roots move.

Minimal JSONL rewrite pattern:

```bash
python - <<'PY'
from pathlib import Path
repls = {
    "/mnt/gemini/home/jiaxuanluo": "/NEW/DATA/ROOT",
}
for p in [
    Path("/NEW/DATA/ROOT/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_varctx2p88_3p84_4p80_5p76.jsonl"),
    Path("/NEW/DATA/ROOT/term_dev_dataset_varctx2p88_3p84_4p80_5p76_new_version.jsonl"),
    Path("/NEW/DATA/ROOT/acl6060_dev_offline_eval_extracted_paper_glossary_varctx2p88_3p84_4p80_5p76/acl6060_dev_dataset.jsonl"),
    Path("/NEW/DATA/ROOT/medicine_eval_varctx2p88_3p84_4p80_5p76/medicine_dev_dataset.jsonl"),
]:
    tmp = p.with_suffix(p.suffix + ".tmp")
    with p.open("r", encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
        for line in src:
            for old, new in repls.items():
                line = line.replace(old, new)
            dst.write(line)
    tmp.replace(p)
PY
```

## Environment

- 8 GPUs with memory comparable to Taurus A6000.
- CPU memory target from the launcher: 320G.
- Python environment must provide torch distributed, transformers, peft, wandb,
  and repo dependencies used by `qwen3_glossary_neg_train.py`.
- Hugging Face cache/model entries needed:
  `microsoft/wavlm-large`, `BAAI/bge-m3`.
- If the target cluster has no internet, pre-transfer Hugging Face cache entries
  or set `HF_HOME`/`HF_HUB_CACHE` to an existing cache.

## Startup Checks

Before `sbatch`:

```bash
bash -n documents/code/train/term_train/launchers/2026/05/20260518__varctx_lmlb_v3_audio_wavlm_large_taurus8_bs8k_gc128_resume_best_eval100_full.sh
python -m json.tool documents/code/train/term_train/manifests/2026/05/<new_wavlm_resume_manifest>.json >/dev/null
test -f /mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_aud_wavl_fu_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_dev100Tau1_eval200_taurus8_best.pt
test -f /mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_varctx2p88_3p84_4p80_5p76.jsonl
test -d /mnt/gemini/home/jiaxuanluo/term_train_audio_chunks_gsv2full_gsdedup_varctx2p88_3p84_4p80_5p76
test -f /mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_gt_union_gs10000_min_norm2_backfill.json
```

At startup, confirm logs print:

```text
[TRAIN] RESUME=/mnt/gemini/home/jiaxuanluo/train_outputs/..._best.pt
[TRAIN] Audio encoder: preset=wavlm-large type=wavlm model=microsoft/wavlm-large
[TRAIN] Text encoder: preset=bge-m3 model=BAAI/bge-m3 input_prefix='' pooling=cls
[TRAIN] eval_wiki_glossary=... sizes=1000 10000 eval_steps=100
[AUDIO_ENCODER] disabled WavLM layerdrop (0.1 -> 0.0) for DDP GradCache stability.
[DDP] WavLM retriever uses find_unused_parameters=True for GradCache multi-forward stability.
```

After W&B init succeeds, confirm the new W&B config shows:

```text
resume = /mnt/gemini/home/jiaxuanluo/train_outputs/..._best.pt
eval_steps_sample = 100
audio_model_id = microsoft/wavlm-large
text_model_id = BAAI/bge-m3
baseline_run_ids includes qou1rg53
```

## Reporting And Calibration Notes

- Do not use ACL to choose tau, checkpoint, threshold, hyperparameters, or the
  variant winner.
- Dev remains the selection/calibration split.  ACL and medicine are held-out
  readouts.
- Cross-run metric tables should be generated through `wandb_tool.py compare`
  or `db-compare --refresh --anchor-metric both`; do not quote loop-logged
  `run.summary[...]` values directly.
- The source run's step-200 eval is useful operational evidence that the WavLM
  training path is healthy, but the resumed run should get its own manifest,
  notes, W&B run id, and SQLite sync.
