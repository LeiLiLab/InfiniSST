# simuleval — streaming evaluation

Stabilized streaming-eval recipes and module-level findings for the
Speech LLM. For active debugging notes, see `dev_journal.md`.

---

## Stabilized eval pipeline (Phase 5 spec)

For an A/B comparison between Speech LLM variants on ACL6060 dev:

```
export DENSITY_TAG=<tag>                # e.g. 5_cap, 5_cap_adv
export MODEL_NAME=<absolute HF dir>
bash documents/code/simuleval/run_phase5_model_eval.sh
```

Fixed settings inside the wrapper:

- latency multiplier `LM=1`, topk `K=10`
- `RAG_RETRIEVE_STRIDE_SEC=1.92` (stride = window, no sliding overlap)
- 5 eval papers listed in
  `documents/data/data_pre/extracted_glossary_by_paper_manifest.json`
  (papers 110 / 117 / 268 / 367 / 590)
- `spaCyEnv` prepended to `PATH` so the offline-eval step finds
  `simuleval`, `sacrebleu`, `stream_laal_term.py`
- `VLLM_DISABLE_CUSTOM_ALL_REDUCE=1` (TP≥2 P2P workaround on taurus)
- offline eval mode `extracted_by_paper` → emits per-paper extracted
  glossary metrics *and* the full-corpus glossary metrics in the same
  `eval_results_by_paper.log`

Output:

```
/mnt/gemini/data2/jiaxuanluo/density_eval_maxsim_fixed/zh/d<tag>_lm1_k10_per_paper_combined/
  eval_results_by_paper.log     # JSON (bleu, stream_laal, term_*, + full run stream_laal_term output)
  eval_results_by_paper.tsv     # tab-separated summary row
  instances.log                 # merged per-paper instances
  runtime_omni_vllm_maxsim_rag_combined_lm1.jsonl
```

The full-glossary TERM_ACC / FCR numbers live inside
`full_run_stream_laal_term_output` (parse with Python; see
`simuleval/dev_journal.md` for the snippet).

---

## Known pitfalls (in this module; see also `general/` Tier-3 §7)

1. **Offline-eval Python.** `run_one_density_eval.sh` invokes `python3`
   directly; `run_phase5_model_eval.sh` must prepend `spaCyEnv` to `PATH`
   or the combine step fails with `ModuleNotFoundError: simuleval`.
2. **`set -u` trap.** `RAG_CONFIDENCE_HEAD_THRESHOLD_OVERRIDE` must have a
   `:- "0.0"` default; new `_OVERRIDE` vars need the same treatment.
3. **vLLM TP=2 deadlock.** Running two TP=2 jobs on the same aries node
   concurrently hangs during init under `enforce_eager`. Shard across
   sequential SLURM jobs.
4. **aries co-tenancy slowdown.** When the node is shared with a running
   training job, per-iter throughput can drop 4x even with
   `--gpus "device=${ALLOCATED_GPUS}"` hardware isolation (PCIe /
   memory bandwidth contention). Watch `squeue -p aries` before
   submitting.
5. **Portable paths.** `prepare_extracted_glossary_by_paper_inputs.py`
   rewrites `/mnt/data/...` → `/mnt/taurus/data/...`; keep this when
   adding new source lists or aries runs crash reading `dev.source`.

---

## Adversarial copy-faith track: 3-way final result (2026-03-28)

Context: Phase 4 tested whether adversarially rewriting term-map values
+ references on "trivial" terms could force the Speech LLM to copy from
the retriever instead of defaulting to its zero-shot prior. We evaluated
three checkpoints under the stabilized pipeline above.

| metric | d5 (no cap) | d5_cap | d5_cap_adv |
|---|---:|---:|---:|
| BLEU | **43.18** | 42.58 | 40.81 |
| AS-WER avg (5 papers) | **55.34** | 56.28 | 59.43 |
| StreamLAAL (ms) | 1187.9 | 1165.3 | 1103.2 |
| TERM_ACC per-paper (374) | 77.81% (291) | 74.87% (280) | **78.34%** (293) |
| TERM_ACC full (1518) | **74.97%** (1138) | 74.44% (1130) | 74.51% (1131) |
| TCR | 91.14% (72/79) | 87.34% (69/79) | 91.14% (72/79) |
| TERM_FCR | 0.210% | 0.230% | 0.220% |

Source JSON (all three merged):
`documents/data/phase456_orchestration/phase5_3way_summary.json`.

### Takeaways

1. The real lever is the **`--max_terms 20` cap**, not the adversarial
   signal. Dropping the cap (d5 no-cap) recovers BLEU (+0.60),
   per-paper TERM_ACC (+2.94pp) and TCR (+3.80pp) in one step, with
   no training changes.
2. **Adversarial rewrite is a band-aid for the cap.** It restores
   per-paper TERM_ACC and TCR, but pays BLEU -1.78 and AS-WER +3.15 vs
   the cap baseline. Net-net it is *worse* than no-cap on BLEU (-2.37)
   and AS-WER (+4.09) while full-glossary TERM_ACC stays flat (±0.5pp).
   Not worth it as a drop-in default.
3. **Full-glossary TERM_ACC (1518 occ.) is flat across all three.**
   Adversarial only changes *whether retrieved terms get copied*; it
   does not fix *whether the correct term is retrieved*. The retriever
   is the ceiling on full TERM_ACC, not the LLM's copy behavior.
4. **BLEU drop is not a reference-file artefact.** The reference
   (`ACL.6060.dev.en-xx.zh.txt`) was not modified; adversarial
   perturbation only touched training-time `gt_terms_by_chunk[].zh`
   and training-time assistant messages. The drop is explained by
   AS-WER going up (+3.15) and emit-time dropping (-85ms): less fluent,
   earlier emission → lower BLEU.

### Recommended next steps (not yet scheduled)

1. Re-train `d5_cap` with a less aggressive cap (`--max_terms 30` or
   `--max_terms 25`) to find the BLEU ↔ per-paper-TERM_ACC Pareto
   knee. Single-knob ablation; the data above predicts the cleanest win.
2. Push on the **retriever** to lift full-glossary TERM_ACC. See
   `train/term_train/` for current recipes (Config C, confidence head,
   notermaug_v1). The LLM-side cannot move this metric.
3. If the adversarial ratio sweep is revisited, submit to a
   **non-contended** aries node (the 2026-03-28 half-ratio attempt was
   aborted after the aries node ran 4.4x slower than the reference
   `43715` run due to co-tenancy). The `perturb-prob 0.25` recipe is
   preserved in `data_pre/dev_journal.md` and is bit-exactly
   reproducible with `--seed 42`.
