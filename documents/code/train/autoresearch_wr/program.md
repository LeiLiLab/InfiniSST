# autoresearch — dual-tower retrieval model

This is an autonomous experiment loop for a **dual-tower contrastive retrieval model** (Qwen3-Audio speech encoder + BGE-M3 text encoder). The goal is to find the best hyperparameter configuration, primarily **WIKI_RANK** (how many Wikipedia synthetic terms to include in training, ranked by P31 type rarity).

Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) for non-LM contrastive training on a Slurm cluster.

## Setup

To set up a new experiment session:

1. **Read the in-scope files** for full context:
   - `program.md` — this file, your instructions.
   - `train.sh` — the Slurm training script you modify. Contains all hyperparameters.
   - `train.py` — the Python training code. **Do not modify.**
   - `experiment.py` — the run orchestrator. **Do not modify.**
2. **Initialize results.tsv**: Create `results.tsv` with just the header row (if not exists):
   ```
   commit	acl6060_r10_gs10k	dev_r10_gs10k	memory_gb	wiki_rank	status	description
   ```
3. **Verify data exists**: Check that `/mnt/gemini/data1/jiaxuanluo/term_train_v1_0.jsonl` exists and is non-empty.
4. **Confirm and go**: Confirm setup looks good, then kick off the experimentation.

## Model Architecture (context for the agent)

This is NOT a standard language model. It is a **dual-tower retrieval system**:

- **Speech tower**: Qwen3-Audio encoder with LoRA adapters. Encodes 1.92s audio chunks into a 1024-dim embedding.
- **Text tower**: BGE-M3 text encoder with LoRA adapters. Encodes terminology strings ("machine learning", "Kubernetes") into a 1024-dim embedding.
- **Training objective**: Masked multi-positive InfoNCE contrastive loss. Given a batch of (audio, term) pairs, the model learns to match audio chunks to the correct term.
- **Evaluation**: Recall@K over a glossary of terms. For each audio chunk, retrieve the K nearest terms from a bank of N candidates. Success = correct term is in top K.

Key metric: `eval_acl6060/recall@10_gs10000` — recall@10 on ACL conference talks with a bank of 10,000 terms. **Higher is better.**

## Training Data

The training data (`term_train_v1_0.jsonl`) contains two sources:
- **Gigaspeech** (~3.9M entries): Real human speech, `p31_rank = -1` (always included).
- **Wiki_synth** (~5.4M entries): Synthetic TTS audio of Wikipedia terms, each with a `p31_rank` field (0 = rarest type like "widget toolkit", 4.5M = most common like "human").

**WIKI_RANK** controls how many wiki_synth terms to include: only entries with `p31_rank < WIKI_RANK` are loaded. The rest (common/homogeneous terms) are excluded.

Each wiki_synth entry has an `audio_type` field: `"clean"` (original TTS) or `"noisy"` (TTS with additive noise augmentation). **CLEAN_RATIO** controls the proportion of clean vs noisy wiki_synth entries:
- `-1` (default): keep all entries unchanged (current behavior: 1 clean + 1 noisy per term).
- `0.0`: noisy only (drop all clean entries).
- `1.0`: clean only (drop all noisy entries).
- `0.5`: randomly keep ~50% of clean entries and ~50% of noisy entries.
- Any value in `[0.0, 1.0]`: keep `CLEAN_RATIO` fraction of clean and `1 - CLEAN_RATIO` fraction of noisy.

Gigaspeech entries (real speech, no `audio_type` field) are always kept regardless of CLEAN_RATIO.

| WIKI_RANK | Wiki terms included | Approx total dataset |
|-----------|-------------------|---------------------|
| 100000    | ~196K (rare only) | ~4.1M               |
| 500000    | ~786K             | ~4.7M               |
| 1000000   | ~1.96M            | ~5.9M               |
| 2000000   | ~3.5M             | ~7.4M               |
| 4500000   | ~5.4M (all)       | ~9.3M               |

Previous experiments showed that including ALL wiki data (WIKI_RANK=4500000) degrades eval because the data becomes dominated by common types (human names, biological taxa). The sweet spot likely lies in the 500K-2M range.

## Experimentation

Each experiment runs on **8x NVIDIA A100 80GB GPUs** via Slurm on the taurus partition. Training runs for a **fixed time budget of 30 minutes** (wall clock training time, excluding model loading/compilation). You launch it as:

```bash
python experiment.py > run.log 2>&1
```

For the very first experiment (if the dataset rebuild job 43116 hasn't completed yet):
```bash
python experiment.py --dependency 43116 > run.log 2>&1
```

**What you CAN modify:**
- `train.sh` — this is the only file you edit. Fair game: WIKI_RANK, CLEAN_RATIO, LR, PER_GPU_BATCH, TEMPERATURE, TIME_BUDGET, EVAL_STEPS_SAMPLE, TEXT_LR, LORA_RANK, etc.

**What you CANNOT modify:**
- `train.py` — the Python training code is fixed.
- `experiment.py` — the orchestrator is fixed.
- Data files on disk.

**The goal: get the highest `acl6060_r10_gs10k`.** Since the time budget is fixed at 30 minutes, you don't need to worry about training time. Everything in `train.sh` is fair game.

**VRAM constraint**: 8x A100 80GB. The current PER_GPU_BATCH=512 works reliably. Going higher will likely OOM. If you increase model capacity (e.g. higher LORA_RANK), you may need to reduce PER_GPU_BATCH. **Do NOT set PER_GPU_BATCH above 512** — it will OOM.

**Suggested WIKI_RANK sweep** (start with these, then refine):
- Phase 1: 250000, 500000, 1000000, 2000000
- Phase 2: Narrow in around the best from Phase 1

After finding the best WIKI_RANK, you may also try:
- **CLEAN_RATIO**: -1 (baseline, all), 0.0 (noisy only), 1.0 (clean only), 0.5 (half each)
- LR: 5e-5, 1e-4, 2e-4
- TEMPERATURE: 0.02, 0.03, 0.05
- PER_GPU_BATCH: 256, 384, 512 (affects in-batch negatives)
- LORA_RANK: 16, 32, 64 (with matching LORA_ALPHA = 2x)

## Output format

When `experiment.py` finishes, it prints a summary like:

```
---
acl6060_r10_gs10k: 0.4523
dev_r10_gs10k:     0.7812
memory_gb:         63.9
wiki_rank:         1000000
clean_ratio:       -1.0
total_steps:       500
training_seconds:  1800.3
status:            ok
log_file:          /mnt/gemini/data1/jiaxuanluo/logs/autoresearch/43200_ar_wr.out
```

You can extract the key metric:
```bash
grep "^acl6060_r10_gs10k:" run.log
```

If the run crashed:
```bash
grep "^status:" run.log
# will show "status: crash"
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 8 columns:

```
commit	acl6060_r10_gs10k	dev_r10_gs10k	memory_gb	wiki_rank	clean_ratio	status	description
```

1. git commit hash (short, 7 chars)
2. acl6060_r10_gs10k achieved — use 0.000000 for crashes
3. dev_r10_gs10k achieved — use 0.000000 for crashes
4. peak memory in GB, round to .1f — use 0.0 for crashes
5. wiki_rank used
6. clean_ratio used (-1 = all, 0.0-1.0 = filtered)
7. status: `keep`, `discard`, or `crash`
8. short text description of what this experiment tried

Example:

```
commit	acl6060_r10_gs10k	dev_r10_gs10k	memory_gb	wiki_rank	clean_ratio	status	description
a1b2c3d	0.452300	0.781200	63.9	1000000	-1	keep	baseline WIKI_RANK=1M
b2c3d4e	0.461500	0.793100	63.9	500000	-1	keep	reduce WIKI_RANK to 500K
c3d4e5f	0.443200	0.768900	63.9	1000000	0.0	discard	noisy only CLEAN_RATIO=0
d4e5f6g	0.000000	0.000000	0.0	250000	-1	crash	WIKI_RANK=250K with LR=5e-4 (OOM)
```

## The experiment loop

LOOP FOREVER:

1. Look at the git state and results so far.
2. Edit `train.sh` with an experimental idea (change WIKI_RANK, LR, etc.).
3. `git add train.sh && git commit -m "description of change"`
4. Run the experiment: `python experiment.py > run.log 2>&1`
5. Read out the results: `grep "^acl6060_r10_gs10k:\|^status:" run.log`
6. If status is "crash", run `tail -n 50 run.log` to debug. If fixable, fix and re-run. Otherwise log as crash and move on.
7. Record the results in results.tsv (do NOT commit results.tsv — leave it untracked).
8. If acl6060_r10_gs10k **improved** (higher), "advance" the branch (keep the commit).
9. If acl6060_r10_gs10k is equal or worse, `git reset --hard HEAD~1` to discard the change.

**IMPORTANT: The metric is HIGHER IS BETTER.** Keep changes that increase acl6060_r10_gs10k. Discard changes that decrease it.

**Timeout**: Each experiment takes ~35 minutes total (30min training + ~5min startup/eval). If a run exceeds 60 minutes, kill it and treat as failure.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human. The human might be asleep. You are autonomous. If you run out of WIKI_RANK values to try, move on to other hyperparameters (LR, temperature, batch size). If you've exhausted those, try combinations. The loop runs until the human interrupts you.
