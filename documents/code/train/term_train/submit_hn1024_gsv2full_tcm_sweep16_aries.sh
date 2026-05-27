#!/bin/bash
# Submit the full chain:
#   1. k1024 full GSV2 TCM-off 3-epoch warm start
#   2. optimizer/scheduler checkpoint audit
#   3. 16-run TCM pair/weight continuation sweep
#   4. 16-run dev fullbank eval-only array

set -euo pipefail

ROOT="/mnt/taurus/home/jiaxuanluo/InfiniSST"
BASELINE="${ROOT}/documents/code/train/term_train/run_mfa_smallest_dense_hn1024_gsv2full_tcmoff_ep3_8gpu_aries.sh"
AUDIT="${ROOT}/documents/code/train/term_train/audit_hn1024_gsv2full_tcmoff_ep3_checkpoint_aries.sh"
SWEEP="${ROOT}/documents/code/train/term_train/run_mfa_smallest_dense_hn1024_gsv2full_tcm_sweep16_8gpu_aries.sh"
FULL_EVAL="${ROOT}/documents/code/train/term_train/run_mfa_smallest_dense_hn1024_gsv2full_tcm_sweep16_fullbank_eval_1gpu_aries.sh"
GLOSS_10K="/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample10000.json"
GLOSS_1M="/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample1000000.json"

check_file() {
    local path="$1"
    if [ ! -f "${path}" ]; then
        echo "[FATAL] missing file: ${path}" >&2
        exit 1
    fi
}

check_tag_len() {
    local tag="$1"
    local n="${#tag}"
    if [ "${n}" -lt 1 ] || [ "${n}" -gt 64 ]; then
        echo "[FATAL] WandB tag length ${n} invalid: ${tag}" >&2
        exit 1
    fi
}

for path in "${BASELINE}" "${AUDIT}" "${SWEEP}" "${FULL_EVAL}" "${GLOSS_10K}" "${GLOSS_1M}"; do
    check_file "${path}"
done

check_tag_len "family:sst_ood_hardneg"
check_tag_len "task:train"
check_tag_len "task:eval"
check_tag_len "data:3variant_gsv2full_gsfix_mfa"
check_tag_len "status:running"
check_tag_len "compute:aries-8gpu"
check_tag_len "compute:aries-1gpu"
check_tag_len "variant:hn1024_gsv2full_tcmoff_ep3"
for p in p85_n70 p80_n60 p75_n50 p70_n40; do
    for w in 1 2 4 8; do
        check_tag_len "variant:tcm_${p}_w${w}"
        check_tag_len "variant:fullbank_tcm_${p}_w${w}"
    done
done

echo "[SUBMIT] baseline: ${BASELINE}"
baseline_job="$(sbatch --parsable "${BASELINE}")"
echo "[SUBMIT] baseline_job=${baseline_job}"

echo "[SUBMIT] audit after baseline"
audit_job="$(sbatch --parsable --dependency=afterok:${baseline_job} "${AUDIT}")"
echo "[SUBMIT] audit_job=${audit_job}"

echo "[SUBMIT] sweep after audit"
sweep_job="$(sbatch --parsable --dependency=afterok:${audit_job} "${SWEEP}")"
echo "[SUBMIT] sweep_job=${sweep_job}"

echo "[SUBMIT] fullbank eval after sweep"
full_eval_job="$(sbatch --parsable --dependency=afterok:${sweep_job} "${FULL_EVAL}")"
echo "[SUBMIT] full_eval_job=${full_eval_job}"

echo "[SUBMIT] chain=${baseline_job} -> ${audit_job} -> ${sweep_job} -> ${full_eval_job}"
