# InfiniSST Infrastructure Reference

> Experiments: use the WandB MCP + `.cursor/rules/experiment_tracking.mdc`.
> This file only covers infrastructure that is NOT per-run (storage, shared
> upstream data, env, pitfalls). Per-run config / metrics / hypothesis /
> git snapshot live in WandB.

## 1. Storage conventions

| alias | cross-node path | local path on owning node | typical use |
|---|---|---|---|
| taurus data | `/mnt/taurus/data` | `/mnt/data` (taurus) | shared datasets (siqi's gigaspeech etc.) |
| taurus data2 | `/mnt/taurus/data2` | `/mnt/data2` (taurus) | training outputs, model checkpoints |
| aries data4 | `/mnt/aries/data4` | `/mnt/data4` (aries) | aries-local scratch, some MFA outputs |
| aries data6 | `/mnt/aries/data6` | `/mnt/data6` (aries) | wiki MFA (two sub-roots) |
| gemini data1 | `/mnt/gemini/data1` | `/mnt/data1` (gemini) | term_train JSONLs, SQLite indices |
| gemini data2 | `/mnt/gemini/data2` | `/mnt/data2` (gemini) | simuleval outputs, HF model caches |

Rule of thumb: when a job runs on **aries**, write cross-node inputs/outputs as `/mnt/taurus/...` or `/mnt/gemini/...` (NFS-mounted there). `/home/jiaxuanluo` is **not** NFS; use `/mnt/taurus/home/jiaxuanluo/...` when an aries job needs the repo source. Always use the fully-qualified `/mnt/<node>/...` form — never node-local shortcuts.

## 2. Shared upstream data (multi-run, not per-run)

- **Gigaspeech MFA**
  - SQLite index (term -> utter_id/time): `/mnt/gemini/data1/jiaxuanluo/gigaspeech_mfa_index/gigaspeech_mfa_index.sqlite`
  - Source TextGrids (siqi's): `/mnt/taurus/data/siqiouyang/datasets/gigaspeech/textgrids`
- **Wiki-synth MFA** — `WIKI_MFA_BASES` lookup order in `documents/code/data_pre/enrich_jsonl_with_mfa_timestamps.py`:
  1. `/mnt/aries/data4/jiaxuanluo/MFA/3variant`
  2. `/mnt/aries/data6/jiaxuanluo/MFA/1third_aries`
  3. `/mnt/aries/data6/jiaxuanluo/MFA/aries`

  Layout under each root: `work/shard_{00..19}/mfa_output/utt_<numeric>.TextGrid`. 20 shards x 149936 utterances (`WIKI_SHARD_SIZE=149936`, `WIKI_NUM_SHARDS=20`); `utter_id = wiki_synth_<numeric>`, `shard_idx = numeric // 149936` (capped at 19).
- **ACL6060 dev (eval inputs)**
  - Audio manifest: `/mnt/taurus/data/siqiouyang/datasets/acl6060/dev.yaml`
  - Paper order (for combining per-paper instances): `/mnt/taurus/data/siqiouyang/datasets/acl6060/dev.source`
  - zh reference: `/mnt/taurus/data/siqiouyang/datasets/acl6060/dev/text/txt/ACL.6060.dev.en-xx.zh.txt`
  - Full-corpus glossary: `retriever/gigaspeech/data_pre/glossary_acl6060.json`
  - Per-paper glossaries: `documents/data/data_pre/extracted_glossaries_by_paper/extracted_glossary__<paper>.json`
  - Per-paper manifest: `documents/data/data_pre/extracted_glossary_by_paper_manifest.json`
- **mwerSegmenter binary**: `/mnt/taurus/home/jiaxuanluo/mwerSegmenter` (prepend to `PATH` for offline eval).

## 3. Conda environments

| env | path | role |
|---|---|---|
| `spaCyEnv` | `/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv` | retriever / term training; simuleval; offline eval. **Required** for `stream_laal_term.py`, `sacrebleu`, `mwerSegmenter`. |
| `base` | `/home/jiaxuanluo/miniconda3` | default shell; does **not** have simuleval. |

## 4. Known pitfalls (so we don't re-learn them)

1. **aries wandb init timeout**: aries network can't reach wandb; training jobs must set `WANDB_MODE=offline` (see `documents/code/train/sst_omni_train/run_adversarial_train_sbatch.sh`).
2. **Docker GPU isolation on aries**: use `--gpus "device=${ALLOCATED_GPUS}"` (hardware isolation), NOT `--gpus all` with `CUDA_VISIBLE_DEVICES` soft isolation — the latter OOMs on shared nodes.
3. **vLLM P2P corruption on taurus**: set `VLLM_DISABLE_CUSTOM_ALL_REDUCE=1` when TP >= 2.
4. **Offline eval Python**: `run_one_density_eval.sh` invokes `python3` directly; must have `spaCyEnv` prepended to PATH (fixed in `documents/code/simuleval/run_phase5_model_eval.sh`).
5. **`eval_density_unified.sh` `set -u` bug**: `RAG_CONFIDENCE_HEAD_THRESHOLD_OVERRIDE` is referenced without a default (line ~340). Add a `:- ` default before running under `set -u`, or the combine stage will crash after all per-paper simuleval runs succeed.
