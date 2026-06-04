#! /bin/bash

set -euo pipefail

export EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_PATH="$(dirname "$(dirname "${EMBODIED_PATH}")")"
export WORKSPACE_PATH="$(dirname "${REPO_PATH}")"
export SRC_FILE="${EMBODIED_PATH}/benchmark_dsrl_train_throughput.py"

ACTOR_GPUS_LIST="${ACTOR_GPUS_LIST:-1}"
MICRO_BATCH_LIST="${MICRO_BATCH_LIST:-64 32 16}"
GRAD_ACCUM_LIST="${GRAD_ACCUM_LIST:-1}"
UPDATE_EPOCH="${UPDATE_EPOCH:-32}"
ACTION_CHUNK_SIZE="${ACTION_CHUNK_SIZE:-50}"
ROBOT_CONTROL_HZ="${ROBOT_CONTROL_HZ:-30.0}"
WARMUP_STEPS="${WARMUP_STEPS:-1}"
MEASURE_STEPS="${MEASURE_STEPS:-3}"
SYNC_CUDA_TIMERS="${SYNC_CUDA_TIMERS:-True}"
PEAK_TFLOPS_PER_GPU="${PEAK_TFLOPS_PER_GPU:-}"
PROFILE_TARGET_MFU="${PROFILE_TARGET_MFU:-0.40}"
PROFILE_STEPS="${PROFILE_STEPS:-3}"
PROFILE_TRACE_DIR="${PROFILE_TRACE_DIR:-}"
RUN_FIRST_SUCCESS_ONLY="${RUN_FIRST_SUCCESS_ONLY:-True}"
WAIT_FOR_GPUS_IDLE="${WAIT_FOR_GPUS_IDLE:-False}"
GPU_IDLE_MEM_MB="${GPU_IDLE_MEM_MB:-5000}"
GPU_CHECK_INTERVAL_S="${GPU_CHECK_INTERVAL_S:-300}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

RESULTS_DIR="${RESULTS_DIR:-${WORKSPACE_PATH}/codex_notes/training_throughput}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/dsrl_pi05_train_throughput_$(date +'%Y%m%d-%H%M%S')}"
OUTPUT_CSV="${OUTPUT_CSV:-${LOG_DIR}/dsrl_pi05_train_throughput.csv}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/run.log}"

mkdir -p "${LOG_DIR}"

if [[ -z "${PEAK_TFLOPS_PER_GPU}" ]]; then
  echo "ERROR: PEAK_TFLOPS_PER_GPU is required because MFU is mandatory." | tee -a "${RUN_LOG}"
  exit 1
fi

echo "Writing sweep results to ${OUTPUT_CSV}" | tee -a "${RUN_LOG}"
echo "peak_tflops_per_gpu=${PEAK_TFLOPS_PER_GPU}" | tee -a "${RUN_LOG}"
if [[ -n "${EXTRA_OVERRIDES}" ]]; then
  echo "extra_overrides=${EXTRA_OVERRIDES}" | tee -a "${RUN_LOG}"
fi

is_true() {
  [[ "$1" == "True" || "$1" == "true" || "$1" == "1" ]]
}

wait_for_gpus_idle() {
  local actor_gpus="$1"
  local actor_gpu_end=$((actor_gpus - 1))
  if ! is_true "${WAIT_FOR_GPUS_IDLE}"; then
    return 0
  fi

  while true; do
    local busy_gpus
    busy_gpus="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F',' -v end="${actor_gpu_end}" -v limit="${GPU_IDLE_MEM_MB}" '
          {
            gsub(/ /, "", $1);
            gsub(/ /, "", $2);
            if ($1 <= end && $2 >= limit) {
              printf "%s:%sMB ", $1, $2;
            }
          }')"
    if [[ -z "${busy_gpus}" ]]; then
      echo "GPUs 0-${actor_gpu_end} idle enough for benchmark." | tee -a "${RUN_LOG}"
      return 0
    fi
    echo "Waiting for GPUs 0-${actor_gpu_end}; busy=${busy_gpus}; sleep=${GPU_CHECK_INTERVAL_S}s" | tee -a "${RUN_LOG}"
    sleep "${GPU_CHECK_INTERVAL_S}"
  done
}

for actor_gpus in ${ACTOR_GPUS_LIST}; do
  actor_gpu_end=$((actor_gpus - 1))
  actor_placement="0-${actor_gpu_end}"
  wait_for_gpus_idle "${actor_gpus}"
  run_succeeded_for_actor_gpus="False"

  for micro_batch_size in ${MICRO_BATCH_LIST}; do
    for grad_accum in ${GRAD_ACCUM_LIST}; do
      global_batch_size=$((micro_batch_size * actor_gpus * grad_accum))
      read -r -a extra_overrides <<< "${EXTRA_OVERRIDES}"

      echo "actor_gpus=${actor_gpus} micro_batch_size=${micro_batch_size} global_batch_size=${global_batch_size} grad_accum=${grad_accum}" | tee -a "${RUN_LOG}"

      if python "${SRC_FILE}" \
        --config-path "${EMBODIED_PATH}/config" \
        --config-name benchmark_dsrl_pi05_train_throughput \
        runner.logger.log_path="${LOG_DIR}" \
        cluster.component_placement.actor="${actor_placement}" \
        actor.micro_batch_size="${micro_batch_size}" \
        actor.global_batch_size="${global_batch_size}" \
        actor.model.num_action_chunks="${ACTION_CHUNK_SIZE}" \
        algorithm.update_epoch="${UPDATE_EPOCH}" \
        benchmark.warmup_steps="${WARMUP_STEPS}" \
        benchmark.measure_steps="${MEASURE_STEPS}" \
        benchmark.sync_cuda_timers="${SYNC_CUDA_TIMERS}" \
        benchmark.output_csv="${OUTPUT_CSV}" \
        benchmark.profile.enabled=True \
        benchmark.profile.with_flops=True \
        benchmark.profile.profile_steps="${PROFILE_STEPS}" \
        benchmark.profile.peak_tflops_per_gpu="${PEAK_TFLOPS_PER_GPU}" \
        benchmark.profile.target_mfu="${PROFILE_TARGET_MFU}" \
        benchmark.profile.trace_dir="${PROFILE_TRACE_DIR}" \
        benchmark.producer.robot_control_hz="${ROBOT_CONTROL_HZ}" \
        benchmark.producer.action_chunk_size="${ACTION_CHUNK_SIZE}" \
        "${extra_overrides[@]}" \
        2>&1 | tee -a "${RUN_LOG}"; then
        run_succeeded_for_actor_gpus="True"
        if is_true "${RUN_FIRST_SUCCESS_ONLY}"; then
          break 2
        fi
      else
        status=$?
        echo "benchmark failed with status=${status}; trying next configured micro_batch_size if available" | tee -a "${RUN_LOG}"
        if ! is_true "${RUN_FIRST_SUCCESS_ONLY}"; then
          exit "${status}"
        fi
      fi
    done
  done

  if [[ "${run_succeeded_for_actor_gpus}" != "True" ]]; then
    echo "ERROR: no DSRL benchmark candidate succeeded for actor_gpus=${actor_gpus}" | tee -a "${RUN_LOG}"
    exit 1
  fi
done
