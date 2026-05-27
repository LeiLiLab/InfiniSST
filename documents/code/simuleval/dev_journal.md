# simuleval / dev_journal.md

Active notebook for streaming evaluation debugging, configuration tweaks, and
per-run analysis. Distill stabilized findings into `simuleval/README.md` when
a run concludes.

---

## 2026-03-28 — paper-110-only rescore of cached baselines (old SLM vs d5_r16)

Context: 8h audit-rank-neg exploration. Full report:
`documents/data/phase456_orchestration/REPORT_audit_and_rank_ablation.md`.

### What changed

Added `documents/code/tools/compute_paper110_metrics_from_cache.py`. It reads
an existing combined 5-paper `instances.log`, subsets `dev.yaml` + `dev.zh.txt`
to only entries whose `wav == 2022.acl-long.110.wav`, and re-invokes
`offline_streamlaal_eval.py --mode extracted_by_paper` on that subset. This
lets us compare models on a single paper without re-running SimulEval (saves
~35 min per model).

### Paper-110, lm=1, per-paper extracted glossary

| Model | BLEU | TERM_ACC | TERM n | StreamLAAL (ms) | StreamLAAL_CA (ms) | TERM_FCR |
|---|---|---|---|---|---|---|
| old_slm | 45.10 | 66.67% | 36/54 | 1222 | 1664 | 0.0020 |
| d5_r16  | 45.73 | 61.11% | 33/54 | 1183 | 1661 | 0.0016 |

TCR is `N/A` at paper-110 granularity (only full-5-paper TCR is logged today).

Cached artifacts (instances.log + subsetted yaml/ref + offline eval TSV) are
archived at `/mnt/taurus/data2/jiaxuanluo/8h_audit_rank_neg/cached_paper110/{old_slm,d5_r16}/`.

### Findings

- d5_r16 gains +0.63 BLEU, shaves -39 ms StreamLAAL, -3 ms CA, and -0.0004
  FCR vs old_slm on paper-110.
- d5_r16 loses 5.56pp TERM_ACC (3 fewer of 54 terms) vs old_slm on paper-110.
- This per-paper view is the baseline future r=32/r=64 ablation runs must
  beat; current 2x A6000 recipe cannot train r=32+ (see training dev_journal
  rank-ablation note).

---

## 2026-03-28 — 3-way compare: d5 / d5_cap / d5_cap_adv (Phase 5 re-read)

### Setup

All three runs evaluated under the same Phase 5 config:
- `eval_density_unified.sh` → `run_one_density_eval.sh`
- latency multiplier LM=1, topk K=10, stride=window (no sliding)
- 5 ACL6060 papers (110 / 117 / 268 / 367 / 590), combined via `eval_results_by_paper.log`
- offline eval: `offline_streamlaal_eval.py --mode extracted_by_paper`
  (per-paper extracted glossary **AND** full-corpus glossary, both reported)
- reference file untouched: `ACL.6060.dev.en-xx.zh.txt`, `glossary_acl6060.json`

Combined results root:
`/mnt/gemini/data2/jiaxuanluo/density_eval_maxsim_fixed/zh/d{5,5_cap,5_cap_adv}_lm1_k10_per_paper_combined/eval_results_by_paper.log`

### Numbers

| metric | d5 (no cap) | d5_cap | d5_cap_adv |
|---|---:|---:|---:|
| BLEU | **43.18** | 42.58 | 40.81 |
| StreamLAAL (ms) | 1187.9 | 1165.3 | 1103.2 |
| StreamLAAL_CA (ms) | 1677.7 | 1645.7 | 1586.8 |
| AS-WER avg (5 papers) | **55.34** | 56.28 | 59.43 |
| TERM_ACC per-paper (374 occ.) | 77.81% (291) | 74.87% (280) | **78.34%** (293) |
| TERM_ACC full (1518 occ.) | **74.97%** (1138) | 74.44% (1130) | 74.51% (1131) |
| TCR | 91.14% (72/79) | 87.34% (69/79) | 91.14% (72/79) |
| TERM_FCR | 0.210% (190/89274) | 0.230% (202/89274) | 0.220% (199/89274) |

Per-paper AS-WER (110 / 117 / 268 / 367 / 590):

| model | 110 | 117 | 268 | 367 | 590 |
|---|---:|---:|---:|---:|---:|
| d5 | 56.01 | 51.48 | 63.04 | 53.49 | 52.67 |
| d5_cap | 55.07 | 54.49 | 64.38 | 52.58 | 54.89 |
| d5_cap_adv | 60.99 | 56.16 | 68.78 | 53.49 | 57.74 |

Snapshot written to
`documents/data/phase456_orchestration/phase5_3way_summary.json`.

### Reading

- `--max_terms 20` cap (d5 → d5_cap): loses **-2.94pp** per-paper TERM_ACC
  and 0 BLEU cost. TCR drops -3.80pp.
- Adversarial rewrite on top of cap (d5_cap → d5_cap_adv): reverses per-paper
  TERM_ACC (+3.47pp) and TCR (+3.80pp), but **BLEU drops -1.78 and AS-WER
  worsens +3.15** on average.
- Full-glossary TERM_ACC (1518 occ.) is **flat across all three** (74.5% ± 0.3).
  Adversarial does not expand overall term coverage; it only rescues
  recall on the 374 paper-relevant occurrences that the cap demoted.
- BLEU drop is not a reference-file problem. Reference (`ACL.6060.dev.en-xx.zh.txt`)
  is untouched and read-only; adversarial perturbation only touched the
  **training-time** `gt_terms_by_chunk[].zh` + training assistant messages.
  The drop is explained by AS-WER regression (+3.15) — the model emits
  earlier and less fluently.

### Interpretation (vs d5 no-cap baseline, not vs cap baseline)

d5_cap_adv vs d5:
- Per-paper TERM_ACC: +0.53pp (78.34 vs 77.81) — noise
- Full TERM_ACC: -0.46pp (74.51 vs 74.97) — noise
- TCR: 0 (both 91.14%)
- FCR: +0.01pp
- BLEU: **-2.37**
- AS-WER: **+4.09**
- StreamLAAL: -84ms (faster emission)

Net: adversarial + cap is not better than the no-cap baseline; it just
compensates for the damage the cap introduced, and pays a BLEU/AS-WER bill.

### Follow-up — half-ratio attempt aborted (2026-03-28 23:44)

The half-ratio run (`d5_cap_adv_half`, perturb-prob 0.25, job 43725) was
`scancel`-ed at iter 150/585. aries was 4.4x slower than the 43715 reference
(67s/iter vs 15s/iter) due to shared-node contention; projected ETA 10–11h.
Loss trajectory matched 43715 at the same iter, so the training itself was
fine, only the wall-clock was bad.

Decision: **stop pursuing the adversarial track** based on the 3-way data
above. No need to wait for a full half-ratio result — best-case it would
trade some BLEU back for per-paper TERM_ACC that already lives in the
no-cap baseline anyway.

### Final conclusion (3-way based)

The real lever is the `--max_terms` cap, not the adversarial signal.

1. Dropping the cap (going to no-cap `d5`) recovers BLEU (+0.60),
   per-paper TERM_ACC (+2.94pp) and TCR (+3.80pp) in one step, with
   no training changes.
2. Adversarial rewrite on top of the cap (`d5_cap_adv`) is a band-aid:
   it restores per-paper TERM_ACC / TCR but pays BLEU -1.78 and AS-WER
   +3.15 vs the cap baseline; net-net it is **worse than no-cap** on
   BLEU (-2.37) and AS-WER (+4.09) while full-glossary TERM_ACC stays
   flat (±0.5pp). So adversarial is not worth it as a drop-in default.
3. Full-glossary (1518 occurrences) TERM_ACC is **flat across all three**
   variants (74.5% ± 0.3). Adversarial only changes *whether retrieved
   trivial terms get copied*; it does not fix *whether the correct term
   is retrieved in the first place*. Retrieval is the ceiling on full
   TERM_ACC, not the LLM's copy behavior.

Recommended next experiments (not yet scheduled):

1. Re-train `d5_cap` with a **less aggressive cap** (e.g. `--max_terms 30`)
   to find the knee of the BLEU / per-paper-TERM_ACC Pareto curve. This
   is a single-knob ablation that the above data predicts will be the
   cleanest win.
2. Push on the **retriever** to lift full-glossary TERM_ACC, since the
   LLM-side (adversarial or not) cannot move it. See
   `train/term_train/` for current retriever recipes.
3. If the half-ratio adversarial run is revisited, submit it to a
   **non-contended aries node** (watch squeue first; avoid co-tenancy with
   training or TP=2 simuleval jobs). The recipe is preserved (see
   `data_pre/dev_journal.md` 2026-03-28 entry; rerunning
   `apply_adversarial_perturbation.py --perturb-prob 0.25 --seed 42`
   reproduces the JSONL bit-exactly).

Final artifacts:
- 3-way summary JSON:
  `documents/data/phase456_orchestration/phase5_3way_summary.json`
- Distilled Tier-2 writeup: `documents/code/simuleval/README.md`
  (see next commit).
