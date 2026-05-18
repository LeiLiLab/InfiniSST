# Varctx Multilingual-E5-Large Handoff

Date: 2026-05-18 UTC

## Goal

Run the varctx576 retriever training on the new cluster with
`intfloat/multilingual-e5-large` as the text encoder, using the E5 retrieval
prefix and training to convergence.

This is no longer a BGE-large resume run.  The previous BGE-large epoch-0
checkpoint is packaged as a reference artifact, but it should not be passed as
`RESUME` for the E5 run because its `text_model_state_dict` belongs to
`BAAI/bge-large-en-v1.5` and is not compatible with
`intfloat/multilingual-e5-large`.

## Control Files

- E5 launcher to use as the starting point:
  `documents/code/train/term_train/launchers/2026/05/20260517__varctx_lmlb_v3_text_multilingual_e5_large_aries8_gc256_fast_eval.sh`
- E5 notes:
  `documents/code/train/term_train/notes/2026/05/20260517__varctx_lmlb_v3_text_multilingual_e5_large_aries8_gc256_dev100_tau1_eval.md`
- Prior E5 manifest, for provenance/reference only:
  `documents/code/train/term_train/manifests/2026/05/20260517T2021__retriever_train__varctx_text_multilingual_e5_large_aries8_gc256_dev100_tau1_eval.json`
- Prior E5 W&B run:
  `om6fnv90`
- Prior E5 status:
  failed from `SIGHUP` in a direct `nohup` run at about step 167; no E5 model
  loading failure, CUDA OOM, or Python exception was observed.
- Baseline/control run:
  `lh1b88kw`
- Running BGE sibling/reference:
  `mhukv2bi`

Before launching on the new cluster, create a new pending manifest by copying
the prior E5 manifest, changing `event_id`, `status`, `wandb_run_id`,
`slurm_job_id`, `cwd`, `command`, and cluster-specific paths.  Do not relaunch
with the old failed manifest as-is.

## Required E5 Configuration

Set or verify these launcher environment values:

```bash
export TEXT_ENCODER_PRESET="multilingual-e5-large"
export TEXT_MODEL_ID="intfloat/multilingual-e5-large"
export TEXT_INPUT_PREFIX="query: "
export TEXT_POOLING="mean"
export GRAD_CACHE_CHUNK_SIZE="256"
export EVAL_STEPS_SAMPLE="100"
export EVAL_SAMPLE_LIMIT="100"
export ACL_EVAL_SAMPLE_LIMIT="0"
export MEDICINE_EVAL_SAMPLE_LIMIT="0"
export EVAL_GLOSSARY_MATCH_MIN_NORM_CHARS="2"
export MAX_STEPS="0"
export EPOCHS="6"
export SCHEDULER_EPOCHS="6"
unset RESUME
```

Important details:

- `TEXT_INPUT_PREFIX="query: "` is required for E5 retrieval-mode term text.
- `TEXT_POOLING="mean"` matches the E5 pooling recipe.
- `MAX_STEPS=0` removes the earlier 2000-step smoke cap and lets the run train
  by epoch count.
- The previous direct E5 run accidentally logged with eval interval `240`; the
  handoff run should force `EVAL_STEPS_SAMPLE=100`.
- ACL remains held-out readout only.  Do not choose checkpoint, threshold, tau,
  or variant winner from ACL.

## Launch Command

After creating the new manifest and fixing cluster paths, launch through the
event wrapper:

```bash
python documents/code/general/experiment_event.py launch \
  documents/code/train/term_train/manifests/2026/05/<new_e5_manifest>.json \
  -- sbatch --parsable \
  documents/code/train/term_train/launchers/2026/05/20260517__varctx_lmlb_v3_text_multilingual_e5_large_aries8_gc256_fast_eval.sh
```

After W&B init appears, write the new W&B run id and Slurm job id back into the
new manifest, re-register it, then sync:

```bash
python documents/code/general/experiment_event.py register \
  documents/code/train/term_train/manifests/2026/05/<new_e5_manifest>.json
python documents/code/general/wandb_tool.py --project qwen3_rag db-sync --runs <new_run_id>
```

Avoid direct `nohup` launch for this handoff unless wrapped in `setsid` or a
real allocation.  The prior E5 attempt failed because the `torchrun` parent
received `SIGHUP`.

## Transfer Package

The data package created for transfer is:

```text
/mnt/gemini/home/jiaxuanluo/transfer_packages/varctx_bgel_resume_data_20260517T232142Z.tar.zst
```

It contains the dataset/audio/glossary artifacts listed below plus the old BGE
checkpoint reference.  `zstd -t` already passed locally and reported the tar
stream size as `667160637440` bytes.  The sha256 file is being generated
separately after this document update.

Path list file:

```text
/mnt/gemini/home/jiaxuanluo/transfer_packages/varctx_bgel_resume_data_20260517T232142Z.paths.txt
```

Restore with original absolute mount layout:

```bash
tar -I zstd -xf varctx_bgel_resume_data_20260517T232142Z.tar.zst -C /
```

If extracting under another root, the archive creates `mnt/gemini/...` and
`mnt/taurus/...` relative to that root; then rewrite launchers and JSONL
`chunk_audio_path` values accordingly.

## Data Artifacts

Transfer the repo state plus these data artifacts.  The JSONL files contain
absolute `chunk_audio_path` values, so either preserve the same mount paths on
the new cluster or rewrite the JSONL paths consistently.

| Role | Current path | Notes |
| --- | --- | --- |
| BGE checkpoint reference | `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_txt_bgel_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_dev100Tau1_eval240_taurus8_smoke2000_epoch_0.pt` | Packaged, but do not use as E5 `RESUME` |
| Train JSONL | `/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_varctx2p88_3p84_4p80_5p76.jsonl` | mandatory |
| Train audio | `/mnt/gemini/home/jiaxuanluo/term_train_audio_chunks_gsv2full_gsdedup_varctx2p88_3p84_4p80_5p76/` | mandatory; subdirs include `2p88`, `3p84`, `4p8`, `5p76`, `wiki_synth` |
| Dev JSONL | `/mnt/gemini/home/jiaxuanluo/term_dev_dataset_varctx2p88_3p84_4p80_5p76_new_version.jsonl` | mandatory |
| Dev audio | `/mnt/gemini/home/jiaxuanluo/term_dev_audio_chunks_varctx_m3/` | mandatory |
| ACL JSONL | `/mnt/gemini/home/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary_varctx2p88_3p84_4p80_5p76/acl6060_dev_dataset.jsonl` | held-out readout |
| ACL audio | `/mnt/gemini/home/jiaxuanluo/acl6060_dev_offline_eval_varctx2p88_3p84_4p80_5p76/audio_chunks/` | mandatory; subdirs `2p88`, `3p84`, `4p8`, `5p76` |
| Medicine JSONL | `/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76/medicine_dev_dataset.jsonl` | mandatory |
| Medicine audio | `/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76/audio_chunks/` | mandatory |
| Dev glossary | `/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample10000.json` | dev gs10k |
| ACL glossary | `/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_gt_union_gs10000_min_norm2_backfill.json` | min-norm-2 backfilled gs10k |
| Medicine glossary | `/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76/medicine_glossary_gt_plus_medicine_wiki_gs10000.json` | medicine gs10k |
| NLP/AI/CS glossary | `/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs_enriched.json` | optional eval/readout glossary |

## Path Rewrite On New Cluster

Fastest route: recreate mount-compatible paths with symlinks or bind mounts so
the JSONLs keep working:

- `/mnt/taurus/home/jiaxuanluo/InfiniSST` -> target repo root.
- `/mnt/gemini/home/jiaxuanluo` -> target data root.
- `/mnt/gemini/data1/jiaxuanluo/logs` -> target log directory.

If mount-compatible paths are not possible, edit these before launch:

- E5 launcher: `REPO_ROOT`, SBATCH partition/GPU/memory/time/log paths,
  `BASE_LAUNCHER`, `NOTES_FILE`, and `ACL_EVAL_WIKI_GLOSSARY`.
- E5 launcher defaults: force `MAX_STEPS=0`, `EVAL_STEPS_SAMPLE=100`,
  `TEXT_ENCODER_PRESET=multilingual-e5-large`,
  `TEXT_MODEL_ID=intfloat/multilingual-e5-large`,
  `TEXT_INPUT_PREFIX="query: "`, and `TEXT_POOLING=mean`.
- Base launcher:
  `TRAIN_JSONL`, `DEV_JSONL`, `ACL_DEV_JSONL`, `MEDICINE_DEV_JSONL`,
  `EVAL_WIKI_GLOSSARY`, `MEDICINE_EVAL_WIKI_GLOSSARY`,
  `NLP_AI_CS_EVAL_GLOSSARY`, `TRAIN_EXCLUDE_TERM_GLOSSARIES`, and the final
  `source ...run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh` path.
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

- 8 GPUs with memory comparable to Aries/Taurus A6000 runs.
- CPU memory target from the launcher: 320G.
- Python environment must provide torch distributed, transformers, peft,
  wandb, and repo dependencies used by `qwen3_glossary_neg_train.py`.
- Model downloads or caches needed:
  `Atotti/Qwen3-Omni-AudioTransformer`, `openai/whisper-large-v3`,
  `intfloat/multilingual-e5-large`.
- If the target cluster has no internet, pre-transfer Hugging Face cache
  entries or set `HF_HOME`/`HF_HUB_CACHE` to an existing cache.

## Startup Checks

Before `sbatch`:

```bash
bash -n documents/code/train/term_train/launchers/2026/05/20260517__varctx_lmlb_v3_text_multilingual_e5_large_aries8_gc256_fast_eval.sh
python -m json.tool documents/code/train/term_train/manifests/2026/05/<new_e5_manifest>.json >/dev/null
test -f /mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_varctx2p88_3p84_4p80_5p76.jsonl
test -d /mnt/gemini/home/jiaxuanluo/term_train_audio_chunks_gsv2full_gsdedup_varctx2p88_3p84_4p80_5p76
test -f /mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_gt_union_gs10000_min_norm2_backfill.json
```

At startup, confirm the log prints:

```text
[TRAIN] Text encoder: preset=multilingual-e5-large model=intfloat/multilingual-e5-large input_prefix='query: ' pooling=mean
[TRAIN] eval_wiki_glossary=... sizes=1000 10000 eval_steps=100
[TRAIN] RESUME=<none>
```

After launch, watch logs until W&B init succeeds.  If W&B init fails, cancel
the job instead of allowing an untracked training run to continue.
