# Context Ablation Devraw Fixed-Denominator 100k

Protocol: dev-only, `eval_metric_denominator=fixed_raw`, shared p31 dev Wiki 100k prefix bank, per-context raw metrics glossary. ACL/tagged ACL/medicine disabled.

| context | W&B run | metrics terms | base | 1k | 10k | 100k |
|---|---|---:|---:|---:|---:|---:|
| 1.92 fixed | `9mff3bc4` | 1852 | 0.9811 | n/a | 0.9747 | 0.9518 |
| 3.84 fixed | `k0odyh1h` | 955 | 0.9865 | 0.9862 | 0.9831 | 0.9758 |
| 5.76 fixed | `d988vg46` | 867 | 0.9964 | 0.9964 | 0.9958 | 0.9933 |
| variable | `q2fus6f1` | 974 | 0.9920 | 0.9920 | 0.9897 | 0.9858 |

`gs1k` is absent for 1.92 fixed because its raw/base bank already has 1852 terms, so the eval script skips `gs1000`. The earlier diagnostic run `s8o0es7g` is invalid for the table because it applied the variable-context 974-term metrics glossary to the old 1.92s dev file, leaving many rows without positives.
