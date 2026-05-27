# Full training run: Variant C (InfoNCE + TCM λ=0.1)

## Event

Submitted a full-budget retraining of ablation Variant C so its "no wall-time
cap" performance can be compared against the eventual ablation winner.

## Rationale

2h-budget ablation showed C (TCM only) and D (HCL+TCM) tied on
`eval_acl6060/recall@10_gs1000` (both 0.9349) but C lagged slightly on
`recall@10_gs10000` (0.8450 vs 0.8512).  D benefited from ~15 additional
minutes of training, so the gap may be compute-confounded.  A 5-epoch C run
isolates TCM's effect at production compute budget and sets a reference
ceiling before committing to HCL+TCM.

## Files affected

| Path | Change |
|------|--------|
| `/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_tcm_full_C_aries.sh` | NEW - full-run launcher cloned from ablation script |

## Key configuration deltas vs. 2h ablation launcher

| Knob | Ablation | Full-run |
|------|----------|---------|
| `MAX_TRAIN_SECONDS` | 7200 | 0 (disabled) |
| `EPOCHS` | 99 (never reached) | 5 |
| `SAVE_STEPS` | 999999 (disabled) | 100 |
| `EVAL_STEPS_SAMPLE` | 40 | 33 |
| `KEEP_CHECKPOINTS` | 2 | 5 |
| `#SBATCH --time` | 2:45:00 | 4-00:00:00 |
| `VERSION` suffix | `_2h` | `full_C_tcm_l01_ep5` |
| `MASTER_PORT` | 29963 | 29973 |

Everything else (LR=1.7e-4, temperature=0.07, batch=12288, wiki_rank=1M,
maxsim windows, MFA supervision, TCM thresholds 0.7/0.4, loss form squared
hinge, reduction mean_viol, margin=0) is kept identical to the ablation
recipe so the final full-run curve stays directly comparable.

## Job id

SLURM job `43750` (partition `aries`).  Train from scratch.  Output paths:

- Checkpoint: `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_clean_gc_wr1000k_m0.0_maxsim_mfa_full_C_tcm_l01_ep5.pt`
- Logs:      `/mnt/gemini/data1/jiaxuanluo/logs/43750_q3_full_C_tcm.{out,err}`
- WandB:     project `qwen3_rag`, run `full_q3rag_..._full_C_tcm_l01_ep5`

## Queue state at submission

```
43748  q3abl_B       RUNNING   34:32   aries
43749  q3abl_A       PENDING   (Resources)
43750  q3_full_C_tcm PENDING   (Priority)
```

Full-C will start after A finishes.
