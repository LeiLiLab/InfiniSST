# General Context & Data-File Locations

Canonical index of large on-disk artifacts (datasets, caches, checkpoints,
indices) and the scripts that produce or consume them. Paths are always
written in the **cross-node** form (`/mnt/taurus/...`, `/mnt/aries/...`,
`/mnt/gemini/...`) so they remain valid regardless of which node a job is
launched on — never use node-local shortcuts (`/home`, `/mnt/data`, `/mnt/data1`,
...) in scripts, configs, or docs.

> When you add a new large artifact, append an entry in the relevant
> section so it stays discoverable without archaeology.

---

## 1. Storage conventions

### 1.1 Partition aliases

| alias | cross-node path | local path on owning node | typical use |
|---|---|---|---|
| taurus data | `/mnt/taurus/data` | `/mnt/data` (taurus) | shared datasets (Siqi's gigaspeech, ACL6060 dev, etc.) |
| taurus data2 | `/mnt/taurus/data2` | `/mnt/data2` (taurus) | my training outputs; HF model exports |
| taurus home | `/mnt/taurus/home/jiaxuanluo` | `/home/jiaxuanluo` (taurus) | repo source; must use this form when invoking from aries |
| aries data3 | `/mnt/aries/data3` | `/mnt/data3` (aries) | aries-local scratch (after data4 ran out) |
| aries data4 | `/mnt/aries/data4` | `/mnt/data4` (aries) | aries-local scratch; some MFA outputs |
| aries data6 | `/mnt/aries/data6` | `/mnt/data6` (aries) | wiki-synth MFA (two sub-roots) |
| gemini data1 | `/mnt/gemini/data1` | `/mnt/data1` (gemini) | term-train JSONLs; SQLite indices |
| gemini data2 | `/mnt/gemini/data2` | `/mnt/data2` (gemini) | simuleval outputs; HF model caches |

**Home dir is not NFS.** A job running on aries that needs this repo
must reference `/mnt/taurus/home/jiaxuanluo/InfiniSST/...`, never
`/home/jiaxuanluo/InfiniSST/...`.

### 1.2 Conda environments

| env | path | role |
|---|---|---|
| `spaCyEnv` | `/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv` | retriever / term training; simuleval; offline eval. **Required** for `stream_laal_term.py`, `sacrebleu`, `mwerSegmenter`. |
| `base`     | `/home/jiaxuanluo/miniconda3` | default shell; does **not** have simuleval |

`mwerSegmenter` binary: `/mnt/taurus/home/jiaxuanluo/mwerSegmenter` (prepend
to `PATH` for offline eval).

---

## 2. Term-retriever training data (Qwen3-RAG / Config C)

### 2.1 Merged 3-variant JSONLs

`utter_id` prefix `wiki_synth_` distinguishes wiki samples from gigaspeech.

| file | size | contents |
|---|---|---|
| `/mnt/gemini/data1/jiaxuanluo/term_train_3variant_1m.jsonl` | 2.8 GB | 3-variant merged, no MFA timestamps |
| `/mnt/gemini/data1/jiaxuanluo/term_train_3variant_1m_mfa.jsonl` | 3.7 GB | above + `mfa_term_{start,end,duration}` from enrich step |
| `/mnt/gemini/data1/jiaxuanluo/term_train_3variant_1m_phoneme.jsonl` | 3.2 GB | above variant with phoneme annotations |
| `/mnt/gemini/data1/jiaxuanluo/term_train_3variant_1m_stats.json` | <1 KB | per-source counts |

Per-row keys in `_mfa.jsonl`:
`term, term_key, chunk_src_text, utter_id, chunk_idx, chunk_audio_path, p31_rank, mfa_term_start_in_chunk, mfa_term_end_in_chunk, mfa_term_duration`.

### 2.2 Dev JSONLs (retriever training + eval)

| file | rows | chunks | has_term% | note |
|---|---:|---:|---:|---|
| `/mnt/gemini/data1/jiaxuanluo/term_dev_with_wiki_synth_normalized.jsonl` | 4646 | 3164 | 77.0% | **primary dev** for retriever eval |
| `/mnt/gemini/data1/jiaxuanluo/term_dev_with_wiki_leftover_normalized.jsonl` | — | — | — | variant kept for reference |
| `/mnt/gemini/data1/jiaxuanluo/term_dev_expanded_wiki1m2m.jsonl` | — | — | — | expanded dev (wiki 1M–2M) |

Per-chunk keys: `utter_id, chunk_idx, chunk_audio_path, chunk_src_text, term, term_key`.
`has_term = any non-empty term_key for (utter_id, chunk_idx)`.

### 2.3 Dev audio chunks (1.92s @ 16 kHz WAV slices)

| dir | files | source |
|---|---:|---|
| `/mnt/gemini/data1/jiaxuanluo/term_dev_audio_chunks/` | ~3k | original dev (POD/YOU/wiki_synth chunks) |

### 2.4 Gigaspeech MFA artifacts

| artifact | path | size |
|---|---|---|
| MFA SQLite index (term → utter_id / time lookup) | `/mnt/gemini/data1/jiaxuanluo/gigaspeech_mfa_index/gigaspeech_mfa_index.sqlite` | 2.2 GB |
| Source TextGrids (Siqi's) | `/mnt/taurus/data/siqiouyang/datasets/gigaspeech/textgrids` | ~8.24M files |

### 2.5 Gigaspeech-XL TSV

| file | size | rows | note |
|---|---:|---:|---|
| `/mnt/taurus/data/siqiouyang/datasets/gigaspeech/train_xl_case_ft-qwen2.5-32b-instruct_marked_mfa_punc_asr.tsv` | 1.6 GB | 1,365,025 | Columns: `id, audio, n_frames, speaker, src_text, tgt_text, src_lang, tgt_lang, trajectory`. `audio = <opus_file>:<offset_samples>:<duration_samples>` at 16 kHz. Id prefix distribution: YOU 542624, POD 434789, AUD 387612. |

### 2.6 Wiki-synth MFA artifacts

Enrichment code looks up TextGrids in these roots **in order** (see
`documents/code/data_pre/enrich_jsonl_with_mfa_timestamps.py` `WIKI_MFA_BASES`):

1. `/mnt/aries/data4/jiaxuanluo/MFA/3variant`
2. `/mnt/aries/data6/jiaxuanluo/MFA/1third_aries`
3. `/mnt/aries/data6/jiaxuanluo/MFA/aries`

Each root: `<root>/work/shard_{00..19}/mfa_output/utt_<numeric>.TextGrid`
and `<root>/work/shard_{00..19}/mfa_input/utt_<numeric>.lab`
(`WIKI_SHARD_SIZE=149936`, `WIKI_NUM_SHARDS=20`; `utter_id = wiki_synth_<numeric>`).

Per-shard intermediate JSONLs (already enriched):
`/mnt/gemini/data1/jiaxuanluo/wiki_synth_mfa_output_aries/wiki_synth_train_shard_{02,10..19}.jsonl`
(11 shards present; `.../wiki_synth_mfa_p31_output_aries/` sibling is empty).

### 2.7 Scripts that produce / consume these

| script | role |
|---|---|
| `documents/code/data_pre/enrich_jsonl_with_mfa_timestamps.py` | enriches merged JSONL with MFA term timestamps |
| `documents/code/data_pre/run_enrich_jsonl_mfa.sh` | sbatch wrapper: `term_train_3variant_1m.jsonl` → `term_train_3variant_1m_mfa.jsonl` |
| `documents/code/train/term_train/run_3variant_1m_aries_gc12k_maxsim_mfa.sh` | aries 8-GPU training (`--mfa_supervised_maxsim`) |
| `retriever/gigaspeech/modal/build_index_multi_gpu.py` | build `gigaspeech_mfa_index.sqlite` |

---

## 3. Retriever (Config C) checkpoints

| role | path | note |
|---|---|---|
| **Config C MaxSim retriever** | `/mnt/aries/data4/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_clean_gc_wr1000k_m0.1_maxsim_mfa_final_C_best_acl6060_gs10000.pt` | frozen; used by `StreamingMaxSimRetriever` and all downstream SimulEval runs |
| SST-eval retriever ref | `/mnt/taurus/data/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs10752_t=0.03_3var_clean_gc_wr1000k_m0.1_maxsim_sp07_best_acl6060_gs10000.pt` | referenced as `RAG_MODEL` in `run_phase5_model_eval.sh` / `run_one_density_eval.sh` families |

> Confidence-head training / caching / integration has been fully removed
> from this repo (2026-04-18). The retriever emits top-K cosine candidates
> directly; any future calibration / filtering logic should be baked into
> the retriever training objective, not a separate post-hoc head.

---

## 4. Speech-LLM (Qwen3-Omni) training data

### 4.1 Current Qwen3-Omni SST datasets (density ablation family)

Produced by `documents/code/data_pre/hard_negative_jsonl_for_speech_llm/rebuild_termmap.py`:

| file | notes |
|---|---|
| `/mnt/gemini/data1/jiaxuanluo/adversarial/train_maxsim_varlen_d5_cap.jsonl` | control for density=5, `--max_terms 20` |
| `/mnt/gemini/data1/jiaxuanluo/adversarial/train_maxsim_varlen_d5_cap_adv.jsonl` | experiment: `_cap` + adversarial term-map perturbations |
| `/mnt/gemini/data1/jiaxuanluo/adversarial/train_cleaned_with_retriever_results_varlen_adv.jsonl` | adversarially-perturbed source that `rebuild_termmap.py` reads |
| `/mnt/gemini/data1/jiaxuanluo/adversarial/trivial_terms.json` | Phase 1 output (zero-shot prior matches retriever) |
| `/mnt/gemini/data1/jiaxuanluo/adversarial/adversarial_alternatives.json` | Phase 2 output (LLM-generated alternatives) |

### 4.2 Phase 6 knobs (added 2026-04-18; not yet used in a successful run)

`rebuild_termmap.py` accepts:
- `--no_gt_max_terms N` – cap term_map length on chunks without GT terms
- `--empty_prob_no_gt P` – override empty-termmap prob on no-GT chunks
- `--empty_prob_has_gt P` – override on has-GT chunks

### 4.3 HF model exports

Under `/mnt/taurus/data2/jiaxuanluo/speech_llm_density_ablation/<tag>/r<rank>/v*-hf/`:

| tag | notes |
|---|---|
| `d5_cap/r16/v0-20260418-125814-hf` | Phase 4 control |
| `d5_cap_adv/r16/v1-20260418-170914-hf` | Phase 4 experiment; **current recommended default** (TERM_ACC +3.48pp vs control) |

Older density-ablation runs live under
`/mnt/aries/data4/jiaxuanluo/train_outputs/` (mostly pruned to `_best.pt`).

---

## 5. Evaluation artifacts

### 5.1 ACL6060 dev assets (inputs for eval)

| file | role |
|---|---|
| `/mnt/taurus/data/siqiouyang/datasets/acl6060/dev.yaml` | audio manifest |
| `/mnt/taurus/data/siqiouyang/datasets/acl6060/dev.source` | paper order for combining per-paper instances |
| `/mnt/taurus/data/siqiouyang/datasets/acl6060/dev/text/txt/ACL.6060.dev.en-xx.zh.txt` | zh reference |
| `retriever/gigaspeech/data_pre/glossary_acl6060.json` | full-corpus ACL6060 glossary |
| `documents/data/data_pre/extracted_glossaries_by_paper/extracted_glossary__<paper>.json` | per-paper extracted glossaries |
| `documents/data/data_pre/extracted_glossary_by_paper_manifest.json` | manifest listing the 5 eval papers |

### 5.2 SimulEval outputs (per-density, per-paper)

Root: `/mnt/gemini/data2/jiaxuanluo/density_eval_maxsim_fixed/zh/`

Naming: `d<tag>_lm<L>_k<K>_g<glossary>_pp<paper_id>/` for per-paper runs;
`d<tag>_lm<L>_k<K>_per_paper_combined/` for the combined
`instances.log`, `runtime_omni_vllm_maxsim_rag_combined_lm<L>.jsonl`,
`eval_results_by_paper.tsv`.

Key recent combined results:

| label | tag | tsv |
|---|---|---|
| control (density=5) | `d5_cap_lm1_k10_per_paper_combined` | `eval_results_by_paper.tsv` |
| experiment d5_cap_adv | `d5_cap_adv_lm1_k10_per_paper_combined` | `eval_results_by_paper.tsv` |

### 5.3 Phase 5 / 6 orchestration artifacts

- Orchestration dir: `documents/data/phase456_orchestration/`
  - `REPORT.md` — final human-readable report (A/B deltas, decision gate, next step)
  - `phase5_decision.json` — gate output (experiment TERM_FCR vs threshold)
  - `logs/`, `eval_control.log`, `eval_experiment.log` — run logs

---

## 6. Known pitfalls (so we don't re-learn them)

1. **aries wandb init timeout.** Aries network can't reach wandb; training
   jobs must set `WANDB_MODE=offline`
   (see `documents/code/train/sst_omni_train/run_adversarial_train_sbatch.sh`).
2. **Docker GPU isolation on aries.** Use `--gpus "device=${ALLOCATED_GPUS}"`
   (hardware isolation). Avoid `--gpus all` + `CUDA_VISIBLE_DEVICES` soft
   isolation — the latter OOMs on shared nodes.
3. **vLLM P2P corruption on taurus.** Set `VLLM_DISABLE_CUSTOM_ALL_REDUCE=1`
   when TP ≥ 2.
4. **Offline eval Python.** `run_one_density_eval.sh` invokes `python3`
   directly; `spaCyEnv` must be prepended to `PATH`
   (fixed in `documents/code/simuleval/run_phase5_model_eval.sh`).
5. **Always default new `_OVERRIDE` env vars in shell scripts.** Use
   `:- "<default>"` so `set -u` runs don't crash when the caller doesn't
   export the variable. This has bitten us multiple times.
6. **vLLM TP=2 deadlock when running two TP=2 simuleval jobs concurrently
   on the same aries node.** Cross-NUMA GPU groups hang during init under
   `enforce_eager`. Workaround: shard lm=1,2 and lm=3,4 into sequential
   SLURM jobs rather than firing them simultaneously.
7. **aries `/mnt/data4` full.** Redirect `HF_HOME`, `TORCH_HOME`,
   `XDG_CACHE_HOME`, and output dirs to `/mnt/aries/data3` when training
   on aries.
8. **Portable paths in source lists.** `prepare_extracted_glossary_by_paper_inputs.py`
   must rewrite `/mnt/data/...` prefixes to `/mnt/taurus/data/...` via
   `_portable_path()`; otherwise aries runs crash when reading
   `dev.source` on another node.
