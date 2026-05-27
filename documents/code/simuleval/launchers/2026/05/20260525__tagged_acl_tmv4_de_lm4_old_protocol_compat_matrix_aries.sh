#!/usr/bin/env bash
set -euo pipefail

# Submit four detached de/lm4 TM-SFT + HN1024 protocol-compatibility probes on
# Aries. The child launcher validates five-row eval artifacts and writes one
# summary TSV per config.

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
cd "${ROOT_DIR}"

RUN_STAMP="${RUN_STAMP_OVERRIDE:-20260525T113241_tmv4_de_lm4_oldproto_aries}"
EVENT_ID="${EVENT_ID_OVERRIDE:-20260525T113241__simuleval__tagged_acl_tmv4_de_lm4_old_protocol_compat_matrix_aries}"
PROBE_LAUNCHER="${PROBE_LAUNCHER_OVERRIDE:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260525__tagged_acl_de_cap16_lm4_decode_cache_probe.sh}"
NOTES_FILE="${NOTES_FILE_OVERRIDE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260525__tagged_acl_tmv4_de_lm4_old_protocol_compat_matrix.md}"

MODEL_NAME="/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-de-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4"
OUT_ROOT_BASE="/mnt/gemini/data1/jiaxuanluo/tagged_acl_tmv4_de_lm4_old_protocol_compat_20260525T113241"
LOG_ROOT_BASE="/mnt/gemini/data1/jiaxuanluo/logs/tagged_acl_tmv4_de_lm4_old_protocol_compat_20260525T113241"

mkdir -p "${OUT_ROOT_BASE}" "${LOG_ROOT_BASE}"
test -x "${PROBE_LAUNCHER}" || { echo "[ERROR] missing probe launcher: ${PROBE_LAUNCHER}" >&2; exit 2; }
test -s "${NOTES_FILE}" || { echo "[ERROR] missing notes: ${NOTES_FILE}" >&2; exit 2; }

cat > "${LOG_ROOT_BASE}/matrix.tsv" <<'EOF'
config_id	gpu_pair	tau	max_new_tokens	max_cache_seconds	keep_cache_seconds	max_cache_chunks	keep_cache_chunks	empty_term_map_policy	eval_tmpdir
tmv4_tau073_oldcache_none_mt40_c80k60_ch20k15	0,1	0.73	40	80	60	20	15	none_block	/dev/shm/jxop0
tmv4_tau073_oldcache_omit_mt40_c80k60_ch20k15	2,3	0.73	40	80	60	20	15	omit	/dev/shm/jxop1
tmv4_tau073_shortcache_omit_mt80_c40k20_ch8k4	4,5	0.73	80	40	20	8	4	omit	/dev/shm/jxop2
tmv4_tau078_oldcache_omit_mt80_c80k60_ch20k15	6,7	0.78	80	80	60	20	15	omit	/dev/shm/jxop3
EOF

echo "[SUBMIT] event_id=${EVENT_ID}" | tee "${LOG_ROOT_BASE}/submit.out"
date -u +"[SUBMIT] utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "${LOG_ROOT_BASE}/submit.out"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,pstate,name --format=csv,noheader,nounits \
  | tee "${LOG_ROOT_BASE}/prelaunch_gpus.csv"
df -h /mnt/gemini/data1 /dev/shm /tmp | tee "${LOG_ROOT_BASE}/prelaunch_df.txt" || true

{
  echo -e "config_id\tgpu_pair\tpid\tpid_file\touter_out\touter_err\tout_root\tlog_root"
} > "${LOG_ROOT_BASE}/pids.tsv"

tail -n +2 "${LOG_ROOT_BASE}/matrix.tsv" | while IFS=$'\t' read -r config_id gpu_pair tau max_new max_cache keep_cache max_chunks keep_chunks empty_policy eval_tmpdir; do
  cfg_out="${OUT_ROOT_BASE}/${config_id}"
  cfg_log="${LOG_ROOT_BASE}/${config_id}"
  mkdir -p "${cfg_out}" "${cfg_log}" "${eval_tmpdir}"
  cmd=(
    "ROOT_DIR_OVERRIDE='${ROOT_DIR}'"
    "RUN_STAMP_OVERRIDE='${RUN_STAMP}_${config_id}'"
    "CONFIG_ID_OVERRIDE='${config_id}'"
    "GPU_PAIR_OVERRIDE='${gpu_pair}'"
    "LM_VALUE_OVERRIDE='4'"
    "MODEL_NAME_OVERRIDE='${MODEL_NAME}'"
    "MODEL_LABEL_OVERRIDE='tmv4_de_lm4_oldproto_hn1024'"
    "OUT_ROOT_OVERRIDE='${cfg_out}'"
    "OUTPUT_BASE_OVERRIDE='${cfg_out}/tmv4_de_lm4_oldproto_hn1024'"
    "LOG_ROOT_OVERRIDE='${cfg_log}'"
    "EVAL_TMPDIR_ROOT_OVERRIDE='${eval_tmpdir}'"
    "MAX_NEW_TOKENS_VALUE_OVERRIDE='${max_new}'"
    "MAX_CACHE_SECONDS_VALUE_OVERRIDE='${max_cache}'"
    "KEEP_CACHE_SECONDS_VALUE_OVERRIDE='${keep_cache}'"
    "MAX_CACHE_CHUNKS_VALUE_OVERRIDE='${max_chunks}'"
    "KEEP_CACHE_CHUNKS_VALUE_OVERRIDE='${keep_chunks}'"
    "RAG_SCORE_THRESHOLD_VALUE_OVERRIDE='${tau}'"
    "EMPTY_TERM_MAP_POLICY_VALUE_OVERRIDE='${empty_policy}'"
    "EXPECTED_INSTANCE_ROWS_OVERRIDE='5'"
    "NOTES_FILE_OVERRIDE='${NOTES_FILE}'"
    "POLL_SECS_OVERRIDE='15'"
    "PYTHON_BIN_OVERRIDE='/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python'"
    "bash '${PROBE_LAUNCHER}'"
  )
  launch_cmd="${cmd[*]}"
  printf '%s\n' "${launch_cmd}" > "${cfg_log}/command.sh"
  chmod +x "${cfg_log}/command.sh"
  setsid bash -lc "${launch_cmd}" > "${cfg_log}/outer.out" 2> "${cfg_log}/outer.err" < /dev/null &
  pid="$!"
  echo "${pid}" > "${cfg_log}/outer.pid"
  echo -e "${config_id}\t${gpu_pair}\t${pid}\t${cfg_log}/outer.pid\t${cfg_log}/outer.out\t${cfg_log}/outer.err\t${cfg_out}\t${cfg_log}" \
    | tee -a "${LOG_ROOT_BASE}/pids.tsv"
done

echo "[SUBMITTED] pids=${LOG_ROOT_BASE}/pids.tsv"
