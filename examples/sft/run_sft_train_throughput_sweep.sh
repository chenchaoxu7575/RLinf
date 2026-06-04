#! /bin/bash

set -euo pipefail

export SFT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_PATH="$(dirname "$(dirname "${SFT_PATH}")")"
export WORKSPACE_PATH="$(dirname "${REPO_PATH}")"
export SRC_FILE="${SFT_PATH}/benchmark_sft_train_throughput.py"

MODES="${MODES:-expert_only full_model action_out_proj_only}"
WORKER_BACKEND="${WORKER_BACKEND:-local}"
ACTOR_GPUS_LIST="${ACTOR_GPUS_LIST:-1}"
MICRO_BATCH_LIST="${MICRO_BATCH_LIST:-1 2 4 8 16}"
GRAD_ACCUM_LIST="${GRAD_ACCUM_LIST:-1}"
WARMUP_STEPS="${WARMUP_STEPS:-3}"
MEASURE_STEPS="${MEASURE_STEPS:-20}"
SYNC_CUDA_TIMERS="${SYNC_CUDA_TIMERS:-True}"
PEAK_TFLOPS_PER_GPU="${PEAK_TFLOPS_PER_GPU:-}"
PROFILE_TARGET_MFU="${PROFILE_TARGET_MFU:-0.40}"
PROFILE_STEPS="${PROFILE_STEPS:-3}"
PROFILE_TRACE_DIR="${PROFILE_TRACE_DIR:-}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

RESULTS_DIR="${RESULTS_DIR:-${WORKSPACE_PATH}/codex_notes/training_throughput}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/sft_pi05_train_throughput_$(date +'%Y%m%d-%H%M%S')}"
OUTPUT_CSV="${OUTPUT_CSV:-${LOG_DIR}/sft_pi05_train_throughput.csv}"
PER_STEP_OUTPUT_CSV="${PER_STEP_OUTPUT_CSV:-${LOG_DIR}/sft_pi05_train_throughput_steps.csv}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/run.log}"

mkdir -p "${LOG_DIR}"

if [[ -z "${PEAK_TFLOPS_PER_GPU}" ]]; then
  echo "ERROR: PEAK_TFLOPS_PER_GPU is required because MFU is mandatory." | tee -a "${RUN_LOG}"
  exit 1
fi

echo "Writing sweep results to ${OUTPUT_CSV}" | tee -a "${RUN_LOG}"
echo "worker_backend=${WORKER_BACKEND}" | tee -a "${RUN_LOG}"
echo "peak_tflops_per_gpu=${PEAK_TFLOPS_PER_GPU}" | tee -a "${RUN_LOG}"
if [[ -n "${EXTRA_OVERRIDES}" ]]; then
  echo "extra_overrides=${EXTRA_OVERRIDES}" | tee -a "${RUN_LOG}"
fi

for mode in ${MODES}; do
  for actor_gpus in ${ACTOR_GPUS_LIST}; do
    if [[ "${WORKER_BACKEND}" == "local" && "${actor_gpus}" -ne 1 ]]; then
      echo "ERROR: benchmark.worker_backend=local is single-GPU only; set WORKER_BACKEND=fsdp for actor_gpus=${actor_gpus}." | tee -a "${RUN_LOG}"
      exit 1
    fi

    actor_gpu_end=$((actor_gpus - 1))
    actor_placement="0-${actor_gpu_end}"

    for micro_batch_size in ${MICRO_BATCH_LIST}; do
      for grad_accum in ${GRAD_ACCUM_LIST}; do
        global_batch_size=$((micro_batch_size * actor_gpus * grad_accum))
        read -r -a extra_overrides <<< "${EXTRA_OVERRIDES}"

        echo "backend=${WORKER_BACKEND} mode=${mode} actor_gpus=${actor_gpus} micro_batch_size=${micro_batch_size} global_batch_size=${global_batch_size} grad_accum=${grad_accum}" | tee -a "${RUN_LOG}"

        python "${SRC_FILE}" \
          --config-path "${SFT_PATH}/config" \
          --config-name benchmark_sft_pi05_train_throughput \
          runner.logger.log_path="${LOG_DIR}" \
          cluster.component_placement.actor="${actor_placement}" \
          actor.micro_batch_size="${micro_batch_size}" \
          actor.global_batch_size="${global_batch_size}" \
          benchmark.worker_backend="${WORKER_BACKEND}" \
          benchmark.mode="${mode}" \
          benchmark.warmup_steps="${WARMUP_STEPS}" \
          benchmark.measure_steps="${MEASURE_STEPS}" \
          benchmark.sync_cuda_timers="${SYNC_CUDA_TIMERS}" \
          benchmark.output_dir="${LOG_DIR}" \
          benchmark.output_csv="${OUTPUT_CSV}" \
          benchmark.per_step_output_csv="${PER_STEP_OUTPUT_CSV}" \
          benchmark.profile.enabled=True \
          benchmark.profile.with_flops=True \
          benchmark.profile.profile_steps="${PROFILE_STEPS}" \
          benchmark.profile.peak_tflops_per_gpu="${PEAK_TFLOPS_PER_GPU}" \
          benchmark.profile.target_mfu="${PROFILE_TARGET_MFU}" \
          benchmark.profile.trace_dir="${PROFILE_TRACE_DIR}" \
          "${extra_overrides[@]}" \
          2>&1 | tee -a "${RUN_LOG}"
      done
    done
  done
done
