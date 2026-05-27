# Medicine zh lm4 gs10k Low TERM_ACC Investigation

## Question

The glossary-bank ablation row below looked suspiciously low:

```text
Medicine zh lm=4 runtime_bank=gs10k TERM_ACC=79.94 538/673
```

This report checks whether the drop is caused by a scoring/denominator issue or by the gs10k runtime bank changing generation behavior.

## Inputs Checked

- Ablation row:
  `documents/code/simuleval/reports/20260525_glossary_bank_ablation_zh_fixedraw_data.tsv`
- gs10k lm4 re-posteval result:
  `/mnt/gemini/data1/jiaxuanluo/psc_medicine_gs_reposteval_fixedraw_20260525/gs10k_lm4/eval_results.localraw.tsv`
- raw lm4 reference result:
  `/mnt/gemini/data1/jiaxuanluo/medicine_hardraw_hn1024_tau078_new_v9_batch_20260524T0242/zh/dmedhard5_new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau0p78_raw_lm4_k10_th0.78_ghard_medicine_glossary_raw_llm_judge_manual_zh215_unique212_ppmedicine5_hardraw/eval_results.tsv`
- Miss-diff debug directory:
  `/mnt/gemini/data1/jiaxuanluo/glossary_bank_debug_medicine_gs10k_lm4_20260525`

## Result

This is not a fixed-denominator or post-eval mix-up. Both rows use the same strict raw denominator:

| setting | BLEU | StreamLAAL | TERM_ACC | REAL_TERM_ADOPT | TERM_FCR |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw lm4 | 44.3591 | 2821.6752 | 85.14, 573/673 | 92.15 | 25.96 |
| gs10k lm4 | 44.1033 | 2836.7997 | 79.94, 538/673 | 88.63 | 17.81 |

The net drop is exactly 35 term hits:

```text
raw misses      100
gs10k misses    135
common misses    76
new gs10k misses 59
fixed by gs10k   24
net drop         35
```

BLEU is essentially unchanged, so this is not a broad translation-quality collapse. The term behavior changed: REAL_TERM_ADOPT drops from 92.15 to 88.63, while false-copy rate improves. Under the strict exact raw target denominator, the larger gs10k bank trades fewer false copies for more missed exact target forms.

## Main Cause

The gs10k runtime bank injects many more terms and near-duplicate translation variants than the raw bank. The term-map size distribution changed sharply:

| setting | mean terms/sentence | nonzero sentences | p50 | p95 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw lm4 | 1.72 | 944 | 1 | 6 | 14 |
| gs10k lm4 | 7.13 | 1208 | 5 | 21 | 42 |

Among the 59 new gs10k-only misses, the top contributors are:

| term | strict target | new misses |
| --- | --- | ---: |
| Radiation Oncologist | 放射肿瘤科医生 | 10 |
| dose | 剂量 | 7 |
| HER2 | HER2 | 4 |
| radiotherapy | 放疗 | 3 |
| triple-negative disease | 三阴性疾病 | 3 |

For the 59 new misses, gs10k's term map still contained the exact source or target in 36 cases. That means many misses are not simple retrieval absence; they are generation choices under a noisier/conflicting term map.

## Representative Failure

`Radiation Oncologist => 放射肿瘤科医生` is the cleanest example.

In the gs10k runs, the term map often contains both:

```text
radiation oncologist  => 放射肿瘤科医生
radiation oncologists => 放射肿瘤医生
```

The model then outputs `放射肿瘤医生`, which is plausible Chinese but does not match the strict raw denominator target `放射肿瘤科医生`.

Example from sentence 335:

```text
source: ... the radiation oncologist will decide whether to treat ...
raw hyp: 基本上，我们会有一位患者来到我们科室，放射肿瘤科医生会决定是否进行治疗...
gs10k hyp: 基本上，会有患者来到我们科室，放射肿瘤医生会决定是否进行治疗...
reference target: 放射肿瘤科医生
```

This pattern accounts for 10 new exact-match misses by itself.

Other observed patterns:

- `dose => 剂量`: gs10k often retrieves compound entries such as `dose distribution => 剂量分布` but omits or weakens the simple raw target in generation.
- `HER2 => HER2`: some gs10k term maps contain `anti-her2 therapy => 抗HER2治疗` or drug-name terms but not the literal `HER2`, and the generated sentence drops the exact token.
- `radiotherapy => 放疗`: gs10k sometimes pushes valid variants such as `放射治疗`, which the strict raw target scorer marks wrong when it expects `放疗`.

## Interpretation

The low TERM_ACC is real under the current metric policy. It is a bank-quality and exact-target-conflict issue, not an eval denominator bug.

For the paper, this supports the glossary-bank ablation story: expanding the runtime bank can reduce false copies, but noisy near-duplicate entries and target variants can hurt strict raw-denominator TERM_ACC.

If we want gs10k to improve strict TERM_ACC, the next controlled fix should be a bank-side conflict filter or raw-priority injection rule, for example:

- collapse singular/plural or near-duplicate source variants against raw glossary entries;
- remove filler entries whose normalized source overlaps a raw term but whose target differs;
- when both raw and filler variants are retrieved, force the raw-denominator target to take precedence in the injected term map.

Those changes affect generation and require a rerun; post-eval alone cannot recover the exact-target misses.
