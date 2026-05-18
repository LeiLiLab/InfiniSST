# Run Notes Template

Every training / eval run MUST pass `--notes_file <path>` pointing at a filled-in copy of this template. Empty or placeholder sections are a hard error: the training / eval script will refuse to launch.

Copy this file, rename it (e.g. `docs/runs/2026-04-23_sst_omni_d7_cap_hypothesis.md`), fill in every section, then pass it to the launcher.

Do NOT remove or rename sections — they are parsed by `wandb_eval_logger.py` / the training entrypoint to validate that the run is documentable.

---

## Hypothesis

<!-- One to three sentences stating what you expect to happen and why.
     Example:
     Increasing density from 5 to 7 while keeping max_terms=20 will increase TERM_ACC by >=1pp
     over the d5_cap baseline without harming TERM_FCR, because the extra terms saturate the
     attention budget on high-salience utterances. -->

## Background / Motivation

<!-- Why now? What motivated this experiment?
     Link the preceding experiments / observations / user requests. -->

## What changed vs baseline

- **Baseline run URL**: https://wandb.ai/<entity>/<project>/runs/<id>
- **Diff**:
  - hparam `X`: A -> B
  - data: `/mnt/...` -> `/mnt/...`
  - code: (brief summary or commit range)

## Expected metrics

<!-- Concrete, falsifiable expectation. Example:
     TERM_ACC: baseline 71.2 -> expected >=72.2 (+1pp)
     TERM_FCR: baseline 4.3  -> expected <= 4.3 (no worse)
     BLEU:     baseline 28.1 -> expected >=28.0 (no worse) -->

## Verdict

<!-- Filled in by agent AFTER the run finishes. One sentence, plus Top-level metrics.
     Example:
     SUCCESS: TERM_ACC 72.4 (+1.2pp vs d5_cap), TERM_FCR 4.1 (-0.2), BLEU 28.0 (flat). Promote as new default. -->
