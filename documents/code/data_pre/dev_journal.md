# data_pre / dev_journal.md

Active notebook for data-generation pipelines (hard-negative JSONL, MFA
enrichment, adversarial perturbation, term-map rebuild). Distill stabilized
recipes into `data_pre/README.md` once the pipeline is frozen.

---

## 2026-03-28 — Adversarial half-ratio (d5_cap_adv_half)

Goal: halve the adversarial rewrite rate to check whether the BLEU / AS-WER
regression seen in d5_cap_adv (see `simuleval/dev_journal.md` 3-way compare)
can be dialed down without losing per-paper TERM_ACC recovery.

### Knob

`documents/code/data_pre/adversarial/apply_adversarial_perturbation.py`
  `--perturb-prob` default 0.5. Full Phase 4 run used 0.5.
  For half-ratio we set 0.25.

### Smoke test (500 conversations)

```
p=0.50 (existing)   p=0.25 (new)
candidate_terms_in_adv_map   2414   2414     # deterministic
targeted_for_perturbation    1179    575
actually_perturbed           1046    514     # ratio 0.491 ≈ 0.5 ✓
```

Smoke artifacts removed per hygiene rule.

### Full-scale build

Cmd:

```
python3 documents/code/data_pre/adversarial/apply_adversarial_perturbation.py \
  --perturb-prob 0.25 \
  --output-jsonl /mnt/gemini/data1/jiaxuanluo/adversarial/train_cleaned_with_retriever_results_varlen_adv_half.jsonl \
  --stats-json   /mnt/gemini/data1/jiaxuanluo/adversarial/perturbation_stats_half.json
```

Stats (`perturbation_stats_half.json`):

| key | p=0.5 (existing) | p=0.25 (half) |
|---|---:|---:|
| conversations_total | 12499 | 12499 |
| candidate_terms_in_adv_map | 60850 | 60850 |
| targeted_for_perturbation | 30467 | 15171 |
| skipped_by_prob | 30383 | 45679 |
| actually_perturbed | 26934 | **13465** |
| miss_fraction | 0.116 | 0.112 |

`actually_perturbed` is exactly 50.0% of the p=0.5 run. Miss fraction
< 40% soft threshold.

### term_map rebuild (same cap config as d5_cap_adv)

```
python3 documents/code/data_pre/hard_negative_jsonl_for_speech_llm/rebuild_termmap.py \
  --input_jsonl  /mnt/gemini/data1/jiaxuanluo/adversarial/train_cleaned_with_retriever_results_varlen_adv_half.jsonl \
  --output_jsonl /mnt/gemini/data1/jiaxuanluo/adversarial/train_maxsim_varlen_d5_cap_adv_half.jsonl \
  --density_coeff 5 --max_terms 20 --seed 42
```

Output: 39.4 MB (same size as d5_cap / d5_cap_adv — structurally identical,
only term_map values and a subset of assistant-reference contents differ).

Rebuild stats match d5_cap_adv 1-to-1 on chunk counts / GT fractions / empty
prob / cap histogram (as expected — only term text changed).

### Files produced

| path | size | role |
|---|---:|---|
| `/mnt/gemini/data1/jiaxuanluo/adversarial/train_cleaned_with_retriever_results_varlen_adv_half.jsonl` | 188 MB | half-ratio adversarial source JSONL |
| `/mnt/gemini/data1/jiaxuanluo/adversarial/train_maxsim_varlen_d5_cap_adv_half.jsonl` | 39.4 MB | training JSONL for d5_cap_adv_half |
| `/mnt/gemini/data1/jiaxuanluo/adversarial/perturbation_stats_half.json` | 409 B | audit trail |

Downstream training: aries job 43725, logged in
`train/sst_omni_train/dev_journal.md`.

### 2026-03-28 23:44 — aborted, artifacts deleted

Job 43725 was `scancel`-ed at iter 150/585 (aries was 4.4x slower than the
43715 reference due to co-tenant contention). Per the smoke-test-and-destroy
hygiene rule, the unused full-scale artifacts were removed to keep
gemini/data1 (99% full) clean:

- `/mnt/gemini/data1/jiaxuanluo/adversarial/train_cleaned_with_retriever_results_varlen_adv_half.jsonl`
- `/mnt/gemini/data1/jiaxuanluo/adversarial/train_maxsim_varlen_d5_cap_adv_half.jsonl`
- `/mnt/gemini/data1/jiaxuanluo/adversarial/perturbation_stats_half.json`

The recipe is preserved (this journal entry + the 0.25 cmd above); rerunning
those two commands reproduces the half-ratio JSONL bit-exactly (deterministic
seed 42).
