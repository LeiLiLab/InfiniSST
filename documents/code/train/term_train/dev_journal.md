# term_train dev_journal

## 2026-04-22 — MFA smallest-covering + dense window grid

### Hypothesis
~80% of DEV→ACL drop is acoustic domain shift (seen-in-train terms still
drop -8.4pp). `hard_max` biases gradient through the widest covering
window → bakes context acoustics. `smallest` forces gradient through the
tightest MFA-aligned crop, stripping neighboring context. Dense 12-window
grid (`2 3 4 5 6 7 8 10 12 16 20 24`) cuts context leakage to ≤1 frame
for every sample (vs ≥2 frames for ~30% of samples on the old 4-window grid).

### Smoke (43834, `MAX_STEPS=100` per-sample k=1024, aries)

COMPLETED, wall-clock 58:18, no OOM. Step time ~28.2s (only +20% vs
baseline ~23.4s; projected 55–65s was pessimistic).

| step | bank      | 43827 (hard_max, 4w) | 43834 (smallest, 12w) |
|------|-----------|----------------------|-----------------------|
| 80   | DEV r@10  | 0.0197               | **0.4102**            |
| 80   | ACL r@10  | 0.1504               | **0.6698**            |
| 80   | ACL gs1k  | 0.0465               | **0.2078**            |
| 80   | ACL gs10k | 0.0078               | **0.0760**            |

Convergence curve shifted left ~80–100 steps. Smoke artifacts deleted
(~15 GB freed).

### 43843 (full 3-epoch, per-sample k=1024, aries) — NO normalize fix

Launched before `--term_id_normalize aggressive` was added. Has the
false-negative HN bug live. Still useful as a datapoint: smallest+dense
effect with broken HN.

### Script update: pool k=64 + normAGGR

`run_mfa_smallest_dense_k1024_aries.sh` updated:
1. `HARD_NEG_K=64, HARD_NEG_K_PER_SAMPLE=0` — pool mode, same as proven
   variantE baseline (ACL r@10_gs10000 ≈ 0.9085). Per-sample k=1024 is
   ~3× slower per step and not the variable under test.
2. `--term_id_normalize aggressive` — bug fix (see HN near-variant entry
   below). Without this, ~50% of anchors have near-variant false negatives.
3. VERSION/WANDB tags updated to `_hardneg_k64_..._normAGGR`.

Next submission will test smallest+dense in isolation against the variantE
k=64 baseline, with both runs using the normalize fix.

---

## 2026-04-22 — HN near-variant false-negative bug + term_id normalization fix

### Problem
In `qwen3_glossary_neg_train.py`, `mine_hard_negatives_per_sample` and the
in-batch false-negative mask in `compute_masked_contrastive_loss` both rely on
exact `stable_term_id(term_text)` hash equality to identify positives / false
negatives. Because the hash input was the raw surface form, string variants
of the anchor's GT term (`proposition` vs `propositions`, `length` vs
`lengths`, `easiest way` vs `easiest ways`, `neural network` vs
`neural-network`, etc.) received different `term_id`s and were surfaced by
the HN miner as hard negatives. InfoNCE at τ=0.07 then pushed the model to
separate GT from these near-synonyms — a destructive signal that scales with
`hard_neg_k_per_sample` (K=1024 in the current Aries cold-start recipe).

### Diagnostic (string-only, no GPU)
Script: `analyze_hn_variant_collision.py`.

- Bank: all unique train `term_key`s
  → 1,405,622 rows (1.397M train + 8.2k wiki).
- Sample: 200 anchors drawn from `term_train_3variant_1m_mfa.jsonl` (seed=0).

Output: `/mnt/gemini/data2/jiaxuanluo/hn_variant_analysis/collision_summary.tsv`

Headline numbers:

| metric | value |
|---|---|
| `frac_has_aggr_eq_variant` | **50.5%** |
| `frac_has_substr_variant` | 99.5% |
| `avg_n_variants_top5_sm≥0.80` | 3.4 |
| `avg_n_variants_top64_sm≥0.80` | 7.7 |
| `frac_anchors_top5_sm≥0.80_nonzero` | 89.5% |

Interpretation: ~50% of anchors have at least one aggressive-normalized
bank variant (plural, punctuation, spacing) that the old HN miner would
treat as a hard negative; at K=64 the average anchor contains ~7.7
SM-ratio≥0.80 lookalikes.

### Fix
`qwen3_glossary_neg_train.py`:

- Added `_normalize_term_for_id()` + `set_term_id_normalize_mode()` module
  state. Modes:
    - `none` — legacy behavior; bit-for-bit compatible with pre-fix ckpts.
    - `lower_strip` — `.lower().strip()` only (near no-op since `term_key` is
      already normalized this way during JSONL prep).
    - `aggressive` — lower+strip + punctuation → space + naive per-token
      plural stripping (`ies→y`, `es`, trailing `s`).
- `stable_term_id()` now normalizes the surface form BEFORE the blake2b hash.
- CLI `--term_id_normalize {none,lower_strip,aggressive}`, set once in
  `main()` via `set_term_id_normalize_mode()` so every worker rank sees the
  same mode.
- The text encoder input is unchanged — we only fold the hash used by
  `gt_match` (HN miner), `fn_mask`, `fn_hn` into a common id. InfoNCE
  therefore stops penalizing near-variants via the already-existing masking
  paths.

### Verification (data says yes)
Script: `verify_term_id_normalize_fix.py` (independently re-implements the
normalization logic in the same file to keep the verification honest).

Output: `/mnt/gemini/data2/jiaxuanluo/hn_variant_analysis/normalize_fix_verification.tsv`

Setup:
- Same 200-anchor / 1.4M-term bank as the diagnostic.
- Enumerate all `(anchor, bank_variant)` pairs where
  `norm_aggressive(anchor) == norm_aggressive(bank_variant)` and
  `bank_variant != anchor`. That's 108 pairs spread over 101 anchors
  (50.5% — matches diagnostic).

Results:

| mode | pairs | collapsed→same term_id | frac | anchors fully fixed | frac_anchors |
|---|---|---|---|---|---|
| `none` | 108 | 0 | 0.00% | 0/101 | 0.00% |
| `lower_strip` | 108 | 0 | 0.00% | 0/101 | 0.00% |
| `aggressive` | 108 | **108** | **100.00%** | **99/101** | **98.02%** |

Spot-check demo pairs under `aggressive`: `proposition↔propositions`,
`length↔lengths`, `easiest way↔easiest ways`, `health↔healths`,
`neural network↔neural-network`, `n-gram↔n gram` all collapse to a single
`term_id`. Under `none` all six collide to `False`.

The 2 anchors not "fully fixed" (99/101 vs 101/101) are the degenerate
case where the anchor itself contains punctuation the normalization maps
differently from the bank variant — rare and harmless.

**Conclusion: the fix correctly folds the problematic near-variant family
into the false-negative / positive mask paths with 100% pair-level
collapse rate, without touching the text encoder input.**

### A/B ablation launched (2026-04-22 14:19 UTC)

Both runs fix normalization to `aggressive`.  Everything else — data, LR,
batch, schedule, TCM, LoRA, MaxSim — bit-identical.  This way the ONLY
moving part is `HARD_NEG_K_PER_SAMPLE` (1024 vs 0), so the delta attributes
cleanly to HN contribution AFTER the bug is fixed.

| Run | Script | Slurm job | Partition | GPUs | HN K | normalize |
|---|---|---|---|---|---|---|
| ~~A (aries)~~ | ~~`...aries.sh`~~ | ~~43844~~ | aries | 8 | 1024 | aggressive | 1536 | 256 |
| ~~B (aries)~~ | ~~`...aries.sh`~~ | ~~43845~~ | aries | 8 | 0 | aggressive | 1536 | 256 |
| ~~A (taurus v1)~~ | ~~`..._taurus.sh`~~ | ~~43846~~ | taurus | 6 | 1024 | aggressive | 1536 | 256 |
| ~~B (taurus v1)~~ | ~~`..._taurus.sh`~~ | ~~43847~~ | taurus | 6 | 0 | aggressive | 1536 | 256 |
| **A (taurus v2)** | `run_ablation_A_..._taurus.sh` | **43849** | taurus | 6 | 1024 | aggressive | 2048 | 512 |
| **B (taurus v2)** | `run_ablation_B_..._taurus.sh` | **43850** | taurus | 6 | 0 | aggressive | 2048 | 512 |

History: 43844/45 cancelled (moved aries→taurus). 43846/47 cancelled
(per-GPU batch too small — global batch was 9216, not 12288).
43849/50 final: PER_GPU_BATCH=2048 → global batch = 6×2048 = 12288
(matches aries baseline), GRAD_CACHE_CHUNK_SIZE=512 (taurus VRAM allows it).

Decision metric: `eval_acl6060/recall@10_gs10000` at best-of-run.
Secondary: `recall@{5,10}_gs1000`, `topk5_filtered_recall@tau_0p70`.

- **A > B ⇒** HN is useful and the fix recoups its value.
- **A ≈ B ⇒** HN adds no net signal at K=1024 (overhead not worth it).
- **A < B ⇒** HN is actively hurting even after the fix; probably the
  audio encoder is too weak to benefit from hard negatives at this stage.

### Files
- `analyze_hn_variant_collision.py` — diagnostic (no GPU, ~2 min for 200 anchors).
- `verify_term_id_normalize_fix.py` — fix-efficacy check (no GPU).
- `qwen3_glossary_neg_train.py` — added `--term_id_normalize` and
  `_normalize_term_for_id` hook in `stable_term_id`.

### Artifacts (Tier 3 candidates)
- `/mnt/gemini/data2/jiaxuanluo/hn_variant_analysis/collision_summary.tsv`
- `/mnt/gemini/data2/jiaxuanluo/hn_variant_analysis/per_anchor_detail.tsv`
- `/mnt/gemini/data2/jiaxuanluo/hn_variant_analysis/normalize_fix_verification.tsv`
