# PSC Medicine New V9 zh gs1k/gs10k Fixed-Raw Eval

## Hypothesis

Using medicine runtime glossaries expanded to `gs1k` and `gs10k` should improve
retrieval coverage for hard medicine terms while keeping TERM metrics
comparable by scoring against the fixed hardraw glossary denominator.

## Background / Motivation

The current medicine main-result gap is `zh`, `lm=1,2,3,4`, runtime glossary
`gs1k/gs10k`, with strict fixed raw glossary denominator.  PSC already has the
New V9 HF export, HN1024 `lh1b88kw` checkpoint, Apptainer runtime, FBK
StreamLAAL tooling, and the five ESO medicine samples staged under the project
eval root.

## What changed vs baseline

- Runtime glossaries are `hardraw + medicine gt/wiki translated filler`.
- Fixed denominator remains the hardraw manual glossary:
  `hard_medicine_glossary_raw_llm_judge_manual_zh215_unique212.json`.
- The launcher runs one five-sample combined SimulEval process per `lm`.
- PSC submissions use 4x V100-32 with TP=4 and the safer `80/60` cache setting.
- A detached monitor sends Slack heartbeat messages every 30 minutes.

## Expected metrics

The run should produce eight rows:

- `gs1k`: `lm=1,2,3,4`
- `gs10k`: `lm=1,2,3,4`

Each row should include BLEU, StreamLAAL, TERM_ACC, REAL_TERM_ADOPT, TERM_FCR,
and paths to `instances.log` / `eval_results.tsv`.

## Verdict

Pending PSC execution.
