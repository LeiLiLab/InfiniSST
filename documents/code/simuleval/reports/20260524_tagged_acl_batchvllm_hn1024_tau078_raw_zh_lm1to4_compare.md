# Tagged ACL zh Raw Batch-vLLM Prototype Compare

Date: 2026-05-24

Setting: New V9 zh SLM, HN1024 retriever, tau=0.78, tagged ACL raw glossary, lm=1..4 in one shared 8-GPU vLLM process.

Important caveat: this is a throughput prototype, not a serial SimulEval-equivalent result.  The deltas below show that scheduling and latency accounting differ from the current serial launcher.

| lm | Batch BLEU | Serial BLEU | Delta | Batch TERM_ACC | Serial TERM_ACC | Delta pp | Batch REAL | Serial REAL | Delta pp | Batch FCR | Serial FCR | Delta pp | Batch LAAL | Serial LAAL | Delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 46.80 | 43.35 | +3.45 | 90.67 | 81.91 | +8.76 | 95.12 | 84.37 | +10.75 | 17.89 | 9.35 | +8.54 | 926 | 1116 | -190 |
| 2 | 50.50 | 49.61 | +0.89 | 91.01 | 88.88 | +2.13 | 94.36 | 90.37 | +3.98 | 12.77 | 12.41 | +0.35 | 1530 | 1862 | -332 |
| 3 | 51.57 | 50.15 | +1.41 | 89.66 | 89.89 | -0.23 | 92.42 | 91.08 | +1.34 | 9.32 | 8.07 | +1.24 | 1775 | 2343 | -567 |
| 4 | 52.01 | 50.81 | +1.20 | 91.35 | 90.00 | +1.35 | 93.67 | 92.12 | +1.55 | 9.12 | 9.40 | -0.28 | 2255 | 2786 | -531 |

Artifacts:

- Output root: `/mnt/gemini/data1/jiaxuanluo/tagged_acl_batchvllm_hn1024_tau078_raw_zh_lm1to4_20260524T1109_tagacl_batchvllm_hn1024_tau078_raw_zh_taurus8/new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078_batchvllm`
- W&B runs: `t0vwd0j0`, `3dd8ugqc`, `7cdgkq0j`, `fs9t8i3k`
- Launcher: `documents/code/simuleval/launchers/2026/05/20260524__tagged_acl_batchvllm_hn1024_tau078_raw_zh_lm1to4_taurus8.sh`

