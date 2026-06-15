#! /bin/bash

set -euo pipefail

export SFT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_PATH="$(dirname "$(dirname "${SFT_PATH}")")"
export WORKSPACE_PATH="$(dirname "${REPO_PATH}")"
export SRC_FILE="${SFT_PATH}/benchmark_pi05_inference_throughput.py"
export SUMMARY_FILE="${SFT_PATH}/summarize_pi05_inference_throughput.py"

BATCH_LIST="${BATCH_LIST:-1 2 4 8 16 32 64}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
DEVICE_IDS="${DEVICE_IDS:-all}"
WARMUP_STEPS="${WARMUP_STEPS:-3}"
MEASURE_STEPS="${MEASURE_STEPS:-20}"
BREAKDOWN_STEPS="${BREAKDOWN_STEPS:-3}"
SYNC_CUDA_TIMERS="${SYNC_CUDA_TIMERS:-True}"
COMPUTE_VALUES="${COMPUTE_VALUES:-True}"
MODE="${MODE:-train}"
MACHINE_LABEL="${MACHINE_LABEL:-$(hostname)}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

RESULTS_DIR="${RESULTS_DIR:-${WORKSPACE_PATH}/codex_notes/inference_throughput}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/pi05_inference_throughput_${MACHINE_LABEL}_$(date +'%Y%m%d-%H%M%S')}"
OUTPUT_CSV="${OUTPUT_CSV:-${LOG_DIR}/pi05_inference_throughput_raw.csv}"
PER_STEP_CSV="${PER_STEP_CSV:-}"
BATCH_SUMMARY_CSV="${BATCH_SUMMARY_CSV:-${LOG_DIR}/pi05_inference_throughput_batch_summary.csv}"
SUMMARY_CSV="${SUMMARY_CSV:-${LOG_DIR}/pi05_inference_throughput_summary.csv}"
SUMMARY_MD="${SUMMARY_MD:-${LOG_DIR}/pi05_inference_throughput_summary.md}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/run.log}"

mkdir -p "${LOG_DIR}"

echo "Writing inference sweep results to ${OUTPUT_CSV}" | tee -a "${RUN_LOG}"
echo "machine_label=${MACHINE_LABEL}" | tee -a "${RUN_LOG}"
echo "gpu_list=${GPU_LIST}" | tee -a "${RUN_LOG}"
echo "device_ids=${DEVICE_IDS}" | tee -a "${RUN_LOG}"
echo "batch_list=${BATCH_LIST}" | tee -a "${RUN_LOG}"
echo "compute_values=${COMPUTE_VALUES}" | tee -a "${RUN_LOG}"
if [[ -n "${PER_STEP_CSV}" ]]; then
  echo "per_step_csv=${PER_STEP_CSV}" | tee -a "${RUN_LOG}"
fi
if [[ -n "${EXTRA_OVERRIDES}" ]]; then
  echo "extra_overrides=${EXTRA_OVERRIDES}" | tee -a "${RUN_LOG}"
fi

batch_list_csv="${BATCH_LIST// /,}"
batch_list_hydra="[${batch_list_csv}]"
if [[ -n "${PER_STEP_CSV}" ]]; then
  per_step_override="benchmark.inference.per_step_csv=${PER_STEP_CSV}"
else
  per_step_override="benchmark.inference.per_step_csv=null"
fi
read -r -a extra_overrides <<< "${EXTRA_OVERRIDES}"
set +e
CUDA_VISIBLE_DEVICES="${GPU_LIST}" python "${SRC_FILE}" \
  --config-path "${SFT_PATH}/config" \
  --config-name benchmark_pi05_inference_throughput \
  runner.logger.log_path="${LOG_DIR}" \
  benchmark.output_dir="${LOG_DIR}" \
  benchmark.output_csv="${OUTPUT_CSV}" \
  benchmark.warmup_steps="${WARMUP_STEPS}" \
  benchmark.measure_steps="${MEASURE_STEPS}" \
  benchmark.sync_cuda_timers="${SYNC_CUDA_TIMERS}" \
  benchmark.inference.device_ids="${DEVICE_IDS}" \
  benchmark.inference.batch_list="${batch_list_hydra}" \
  benchmark.inference.breakdown_steps="${BREAKDOWN_STEPS}" \
  benchmark.inference.compute_values="${COMPUTE_VALUES}" \
  benchmark.inference.mode="${MODE}" \
  benchmark.inference.machine_label="${MACHINE_LABEL}" \
  "${per_step_override}" \
  "${extra_overrides[@]}" \
  2>&1 | tee -a "${RUN_LOG}"
rc=${PIPESTATUS[0]}
set -e
if [[ "${rc}" -ne 0 ]]; then
  echo "inference sweep failed with rc=${rc}; attempting to summarize completed rows" | tee -a "${RUN_LOG}"
fi

if [[ -f "${OUTPUT_CSV}" ]]; then
  python "${SUMMARY_FILE}" \
    --input "${OUTPUT_CSV}" \
    --batch-summary-csv "${BATCH_SUMMARY_CSV}" \
    --summary-csv "${SUMMARY_CSV}" \
    --summary-md "${SUMMARY_MD}" \
    2>&1 | tee -a "${RUN_LOG}"
else
  echo "No raw CSV produced; skipping summary." | tee -a "${RUN_LOG}"
fi

echo "raw_csv=${OUTPUT_CSV}" | tee -a "${RUN_LOG}"
echo "batch_summary_csv=${BATCH_SUMMARY_CSV}" | tee -a "${RUN_LOG}"
echo "summary_csv=${SUMMARY_CSV}" | tee -a "${RUN_LOG}"
echo "summary_md=${SUMMARY_MD}" | tee -a "${RUN_LOG}"
