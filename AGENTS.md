# InfiniSST Agent Local Rules

This file records repo-local rules that must be followed by Codex agents in
`/home/jiaxuanluo/InfiniSST`.

## SimulEval / vLLM Temporary Directory Rule

For any SimulEval, vLLM, or streaming-eval launcher, keep `TMPDIR` and
`EVAL_TMPDIR` short.  vLLM creates Unix-domain IPC sockets under the temporary
directory, and Linux `sockaddr_un` paths are limited to about 107 bytes.  Long
experiment output paths can therefore fail with `ZMQError` or IPC path-length
errors even when the GPUs and model are fine.

Required default:

```bash
EVAL_TMPDIR="${EVAL_TMPDIR:-/tmp/jx_<short_slug>}"
mkdir -p "${EVAL_TMPDIR}"
export TMPDIR="${EVAL_TMPDIR}"
```

Do not default `TMPDIR` or `EVAL_TMPDIR` to deep paths under
`/mnt/gemini/data1/.../logs/...`, `OUTPUT_BASE`, W&B directories, or per-run
artifact directories.  If a launcher already defines `EVAL_TMPDIR`, verify that
the resolved path is short before launching.

When retrying a failed eval after this issue, record the retry reason in the
manifest, for example: `shorten EVAL_TMPDIR to avoid vLLM ZMQ IPC path >107
chars`.
