## Hypothesis
Running DE tagged ACL lm=2/3 first, then JA medicine lm=1/2/3/4, keeps Aries GPU usage ordered and avoids competing vLLM loads.

## Background / Motivation
Aries GPU 4-7 are available for the immediate DE readout. JA medicine should start only after the DE readout finishes and after the JA HF model has been staged locally.

## What changed vs baseline
This orchestration event chains two standalone SimulEval readouts and relies on the separate JA local-cache staging job.

## Expected metrics
DE produces tagged ACL lm=2/3 summaries. JA produces medicine hardraw lm=1/2/3/4 summaries after the local cache is complete.

## Verdict
Pending.
