# `tys70s0y` Dense Tau Lock

Lock the real deployment tau before any TCM-v2 retraining.

## Inputs and outputs

- Source checkpoint audit dump:
  `/mnt/gemini/data2/jiaxuanluo/acl_boundary_audit/tys70s0y_secondary_gs10000/acl6060_gs10000_top10_dump.jsonl`
- Sweep script:
  `documents/code/offline_evaluation/sweep_tau_from_top10_dump.py`
- Sweep TSV:
  `/mnt/gemini/data2/jiaxuanluo/acl_boundary_audit/tys70s0y_secondary_gs10000/dense_tau_sweep.tsv`
- Sweep JSON:
  `/mnt/gemini/data2/jiaxuanluo/acl_boundary_audit/tys70s0y_secondary_gs10000/dense_tau_sweep.json`

Tau grid evaluated on ACL6060 `gs10000`: `0.72 0.74 0.76 0.78 0.80 0.82 0.84 0.86 0.88`

Selection rule:

1. treat `tau=0.80` as the current operating point,
2. allow at most `0.005` absolute filtered-recall drop vs that baseline,
3. among eligible taus, pick the one with the lowest no-term noise,
4. tie-break by higher micro precision, then higher tau.

## Result

Locked deployment tau:

- `tau* = 0.80`
- downstream TCM-v2 neg threshold: `tau* - 0.02 = 0.78`

Why `0.80` stayed optimal under the chosen rule:

- `tau=0.80`: filtered recall `0.7868`, micro precision `0.2042`,
  no-term noise `2.25`
- `tau=0.82`: noise drops to `1.27`, but filtered recall falls to `0.7455`
  (too large a drop to keep under the `0.005` tolerance)
- lower taus recover recall, but noise rises sharply (`3.63` at `0.78`,
  `5.28` at `0.76`)

So for this checkpoint the existing deployment tau is already the best
noise-vs-recall compromise in the local `0.72..0.88` band under the
"do not materially harm filtered recall" rule.
