#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROJECT_PARENT=$(cd "${PROJECT_ROOT}/.." && pwd)
PROFILE_BASE="${RLINF_PROFILE_BASE:-/tmp/rlinf_pub}"
DEFAULT_DATASET_ROOT="${PROFILE_BASE}/datasets/DreamZero-DROID-Data-subset"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

DOCKER_BIN="${DOCKER_BIN:-docker}"
if ! command -v "${DOCKER_BIN}" >/dev/null 2>&1; then
    if [[ -x "${PROJECT_PARENT}/env/bin/docker" ]]; then
        DOCKER_BIN="${PROJECT_PARENT}/env/bin/docker"
    fi
fi

IMAGE="rlinf:embodied-dreamzero-videoloader"
DOCKER_GPUS="${DOCKER_GPUS:-all}"
DATASET_ROOT="${DEFAULT_DATASET_ROOT}"
OUTPUT_DIR="${PROFILE_BASE}/results/dreamzero_videoloader_profile/${TIMESTAMP}"
HF_CACHE_DIR="${RLINF_PROFILE_HF_HOME:-${PROFILE_BASE}/hf_cache}"
TOKENIZER_PATH="google/umt5-xxl"
MODEL_PATH=""
CHUNK_INDEX=0
MAX_EPISODES=1000
HF_ENDPOINT_VALUE="${HF_ENDPOINT:-}"
SKIP_BUILD=0
SKIP_DOWNLOAD=0
SKIP_TIMELINE=0
CHECK_ONLY=0
EMIT_TORCH_TRACE=0
RUN_DATALOADER_PROFILE=1
RUN_SYSTEM_PROFILE=0
METRICS_WARMUP=20
METRICS_STEPS=200
METRICS_REPEATS=3
TIMELINE_WARMUP=5
TIMELINE_STEPS=30
SYSTEM_WARMUP=10
SYSTEM_STEPS=80
SYSTEM_REPEATS=1
SYSTEM_WORKLOAD="none"
SYSTEM_SLEEP_MS=0
SYSTEM_PROFILE_DEVICE="auto"
SYSTEM_COPY_BATCH_TO_DEVICE=0
SYSTEM_SYNC_STEP=1
SYSTEM_CUDA_MATMUL_SIZE=2048
SYSTEM_CUDA_MATMUL_ITERS=1
SYSTEM_CUDA_MATMUL_DTYPE="bfloat16"
NUM_WORKERS=4
PREFETCH_FACTOR=8
MICRO_BATCH_SIZE=1
MIN_DATASET_FREE_GB=200
MIN_OUTPUT_FREE_GB=20
ALLOW_HOME_WRITES=0

usage() {
    cat <<EOF
Usage: bash scripts/run_dreamzero_videoloader_profile.sh [options]

Options:
  --dataset-root PATH       Host dataset directory (default: ${DEFAULT_DATASET_ROOT}).
  --output-dir PATH         Host output directory (default: ${OUTPUT_DIR}).
  --hf-cache-dir PATH       Host Hugging Face cache directory (default: ${HF_CACHE_DIR}).
  --tokenizer-path PATH     Local tokenizer path or HF id (default: google/umt5-xxl).
  --model-path PATH         Optional local model path mounted read-only.
  --chunk-index N           DROID chunk to download (default: 0).
  --max-episodes N          Metadata max episodes (default: 1000).
  --hf-endpoint URL         Optional HF_ENDPOINT for download/tokenizer.
  --image NAME              Docker image tag.
  --docker-gpus SPEC        Docker --gpus value (default: ${DOCKER_GPUS}).
  --skip-build              Do not build Docker image.
  --skip-download           Do not download/resume dataset subset.
  --skip-timeline           Only run CSV metrics.
  --check-only              Build/check dataloader with one batch only.
  --emit-torch-trace        Run a separate torch profiler timeline pass after nsys.
  --system-profile          Also run visible step-stall profiling.
  --system-only             Run visible step-stall profiling only.
  --num-workers N           DataLoader workers (default: 4).
  --prefetch-factor N       DataLoader prefetch factor (default: 8).
  --micro-batch-size N      Micro batch size (default: 1).
  --metrics-warmup N        Metrics warmup batches (default: 20).
  --metrics-steps N         Metrics measured batches (default: 200).
  --metrics-repeats N       Metrics repeats per backend (default: 3).
  --timeline-warmup N       Timeline warmup batches (default: 5).
  --timeline-steps N        Timeline measured batches (default: 30).
  --system-warmup N         System-profile warmup steps (default: 10).
  --system-steps N          System-profile measured steps (default: 80).
  --system-repeats N        System-profile repeats (default: 1).
  --system-workload NAME    none|sleep|cuda_matmul (default: none).
  --system-sleep-ms N       Sleep workload duration in ms (default: 0).
  --system-profile-device D Device for H2D/workload (default: auto).
  --system-copy-batch       Copy each batch to device before workload.
  --system-no-step-sync     Do not CUDA synchronize at system step end.
  --system-cuda-matmul-size N
  --system-cuda-matmul-iters N
  --system-cuda-matmul-dtype float32|float16|bfloat16
  --min-dataset-free-gb N   Required free GB on dataset filesystem (default: 200).
  --min-output-free-gb N    Required free GB on output filesystem (default: 20).
  --allow-home-writes       Permit dataset/output paths under /home.
  -h, --help                Show this help.
EOF
}

path_under_home() {
    case "$1" in
        /home|/home/*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

require_free_space_gb() {
    local path=$1
    local required_gb=$2
    local available_kb
    local required_kb

    mkdir -p "${path}"
    available_kb=$(df -Pk "${path}" | awk 'NR == 2 {print $4}')
    required_kb=$((required_gb * 1024 * 1024))
    if [[ -n "${available_kb}" && "${available_kb}" -lt "${required_kb}" ]]; then
        echo "Not enough free space for ${path}: need ${required_gb}G, available $((available_kb / 1024 / 1024))G" >&2
        exit 1
    fi
}

guard_home_quota() {
    local uid
    local quota_line
    local used_kb
    local quota_kb

    if [[ "${ALLOW_HOME_WRITES}" -eq 1 ]]; then
        return
    fi
    if ! path_under_home "${DATASET_ROOT}" && ! path_under_home "${OUTPUT_DIR}" && ! path_under_home "${HF_CACHE_DIR}"; then
        return
    fi

    if command -v lfs >/dev/null 2>&1; then
        uid=$(id -u)
        quota_line=$(lfs quota -u "${uid}" /home 2>/dev/null | awk '$1 == "/home" {print}')
        if [[ -n "${quota_line}" ]]; then
            used_kb=$(awk '{print $2}' <<<"${quota_line}")
            quota_kb=$(awk '{print $3}' <<<"${quota_line}")
            if [[ "${quota_kb}" -gt 0 && $((used_kb * 100 / quota_kb)) -ge 90 ]]; then
                echo "/home quota is already above 90%; refusing to write dataset/results/cache under /home." >&2
                echo "Use paths under /tmp or pass --allow-home-writes explicitly." >&2
                exit 1
            fi
        fi
    fi

    echo "Refusing to write dataset/results/cache under /home by default." >&2
    echo "Use /tmp-backed paths or pass --allow-home-writes explicitly." >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-root)
            DATASET_ROOT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --hf-cache-dir)
            HF_CACHE_DIR="$2"
            shift 2
            ;;
        --tokenizer-path)
            TOKENIZER_PATH="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --chunk-index)
            CHUNK_INDEX="$2"
            shift 2
            ;;
        --max-episodes)
            MAX_EPISODES="$2"
            shift 2
            ;;
        --hf-endpoint)
            HF_ENDPOINT_VALUE="$2"
            shift 2
            ;;
        --image)
            IMAGE="$2"
            shift 2
            ;;
        --docker-gpus)
            DOCKER_GPUS="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=1
            shift
            ;;
        --skip-download)
            SKIP_DOWNLOAD=1
            shift
            ;;
        --skip-timeline)
            SKIP_TIMELINE=1
            shift
            ;;
        --check-only)
            CHECK_ONLY=1
            shift
            ;;
        --emit-torch-trace)
            EMIT_TORCH_TRACE=1
            shift
            ;;
        --system-profile)
            RUN_SYSTEM_PROFILE=1
            shift
            ;;
        --system-only)
            RUN_DATALOADER_PROFILE=0
            RUN_SYSTEM_PROFILE=1
            shift
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --prefetch-factor)
            PREFETCH_FACTOR="$2"
            shift 2
            ;;
        --micro-batch-size)
            MICRO_BATCH_SIZE="$2"
            shift 2
            ;;
        --metrics-warmup)
            METRICS_WARMUP="$2"
            shift 2
            ;;
        --metrics-steps)
            METRICS_STEPS="$2"
            shift 2
            ;;
        --metrics-repeats)
            METRICS_REPEATS="$2"
            shift 2
            ;;
        --timeline-warmup)
            TIMELINE_WARMUP="$2"
            shift 2
            ;;
        --timeline-steps)
            TIMELINE_STEPS="$2"
            shift 2
            ;;
        --system-warmup)
            SYSTEM_WARMUP="$2"
            shift 2
            ;;
        --system-steps)
            SYSTEM_STEPS="$2"
            shift 2
            ;;
        --system-repeats)
            SYSTEM_REPEATS="$2"
            shift 2
            ;;
        --system-workload)
            SYSTEM_WORKLOAD="$2"
            shift 2
            ;;
        --system-sleep-ms)
            SYSTEM_SLEEP_MS="$2"
            shift 2
            ;;
        --system-profile-device)
            SYSTEM_PROFILE_DEVICE="$2"
            shift 2
            ;;
        --system-copy-batch)
            SYSTEM_COPY_BATCH_TO_DEVICE=1
            shift
            ;;
        --system-no-step-sync)
            SYSTEM_SYNC_STEP=0
            shift
            ;;
        --system-cuda-matmul-size)
            SYSTEM_CUDA_MATMUL_SIZE="$2"
            shift 2
            ;;
        --system-cuda-matmul-iters)
            SYSTEM_CUDA_MATMUL_ITERS="$2"
            shift 2
            ;;
        --system-cuda-matmul-dtype)
            SYSTEM_CUDA_MATMUL_DTYPE="$2"
            shift 2
            ;;
        --min-dataset-free-gb)
            MIN_DATASET_FREE_GB="$2"
            shift 2
            ;;
        --min-output-free-gb)
            MIN_OUTPUT_FREE_GB="$2"
            shift 2
            ;;
        --allow-home-writes)
            ALLOW_HOME_WRITES=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ "${CHECK_ONLY}" -eq 1 && "${MAX_EPISODES}" -gt 8 ]]; then
    MAX_EPISODES=8
fi

guard_home_quota
require_free_space_gb "${DATASET_ROOT}" "${MIN_DATASET_FREE_GB}"
require_free_space_gb "${OUTPUT_DIR}" "${MIN_OUTPUT_FREE_GB}"
require_free_space_gb "${HF_CACHE_DIR}" 5

DATASET_ROOT=$(mkdir -p "${DATASET_ROOT}" && cd "${DATASET_ROOT}" && pwd)
OUTPUT_DIR=$(mkdir -p "${OUTPUT_DIR}" && cd "${OUTPUT_DIR}" && pwd)
HF_CACHE_DIR=$(mkdir -p "${HF_CACHE_DIR}" && cd "${HF_CACHE_DIR}" && pwd)

DOCKER_RUN_ARGS=(
    --rm
    --gpus "${DOCKER_GPUS}"
    --ipc=host
    --ulimit memlock=-1
    --ulimit stack=67108864
    -v "${PROJECT_ROOT}:/workspace/RLinf"
    -v "${DATASET_ROOT}:/data/droid"
    -v "${OUTPUT_DIR}:/outputs"
    -v "${HF_CACHE_DIR}:/hf_cache"
    -e "HF_HOME=/hf_cache"
    -e "HF_HUB_ENABLE_HF_TRANSFER=1"
    -e "RLINF_DREAMZERO_DOWNLOAD_OK=1"
)

if [[ -n "${HF_TOKEN:-}" ]]; then
    DOCKER_RUN_ARGS+=(-e "HF_TOKEN=${HF_TOKEN}")
fi
if [[ -n "${HF_ENDPOINT_VALUE}" ]]; then
    DOCKER_RUN_ARGS+=(-e "HF_ENDPOINT=${HF_ENDPOINT_VALUE}")
fi
if [[ -e "${TOKENIZER_PATH}" ]]; then
    DOCKER_RUN_ARGS+=(-v "${TOKENIZER_PATH}:${TOKENIZER_PATH}:ro")
fi
if [[ -n "${MODEL_PATH}" && -e "${MODEL_PATH}" ]]; then
    DOCKER_RUN_ARGS+=(-v "${MODEL_PATH}:${MODEL_PATH}:ro")
fi

run_in_container() {
    "${DOCKER_BIN}" run "${DOCKER_RUN_ARGS[@]}" "${IMAGE}" bash -lc "set -eo pipefail; $1"
}

profile_env_prefix() {
    printf '%s' 'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; '
}

prefetch_tokenizer() {
    if [[ -e "${TOKENIZER_PATH}" ]]; then
        return
    fi
    echo "[tokenizer] ensuring ${TOKENIZER_PATH} is cached in ${HF_CACHE_DIR}"
    run_in_container "source switch_env dreamzero && python -c \"from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('${TOKENIZER_PATH}')\""
}

if [[ "${SKIP_BUILD}" -eq 0 ]]; then
    echo "[build] ${IMAGE}"
    "${DOCKER_BIN}" build \
        -f "${PROJECT_ROOT}/docker/Dockerfile.dreamzero_videoloader" \
        -t "${IMAGE}" \
        "${PROJECT_ROOT}"
fi

if [[ "${SKIP_DOWNLOAD}" -eq 0 ]]; then
    echo "[download] DROID subset to ${DATASET_ROOT}"
    run_in_container "source switch_env dreamzero && cd /workspace/RLinf && bash scripts/download_dreamzero_droid_subset.sh --local-dir /data/droid --chunk-index ${CHUNK_INDEX} --max-episodes ${MAX_EPISODES}"
fi

echo "[metadata] generating /data/droid/metadata.json"
run_in_container "source switch_env dreamzero && export PYTHONPATH=/workspace/RLinf:/opt/dreamzero:\${PYTHONPATH:-} && cd /workspace/RLinf && python toolkits/lerobot/generate_dreamzero_metadata.py --preset oxe_droid --dataset-root /data/droid --output-metadata /data/droid/metadata.json --merge --max-episodes ${MAX_EPISODES}"

prefetch_tokenizer

BASE_PROFILE_ARGS="--dataset-root /data/droid --metadata-json /data/droid/metadata.json --tokenizer-path ${TOKENIZER_PATH} --num-workers ${NUM_WORKERS} --prefetch-factor ${PREFETCH_FACTOR} --micro-batch-size ${MICRO_BATCH_SIZE}"
if [[ -n "${MODEL_PATH}" ]]; then
    BASE_PROFILE_ARGS="${BASE_PROFILE_ARGS} --model-path ${MODEL_PATH}"
fi

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
    echo "[check] dataloader check-only"
    run_in_container "source switch_env dreamzero && $(profile_env_prefix)export PYTHONPATH=/workspace/RLinf:/opt/dreamzero:\${PYTHONPATH:-} && cd /workspace/RLinf && python toolkits/lerobot/profile_dreamzero_videoloader.py ${BASE_PROFILE_ARGS} --check-only --output-csv /outputs/check_only.csv --output-step-csv /outputs/check_only_steps.csv"
    echo "[done] outputs: ${OUTPUT_DIR}"
    exit 0
fi

echo "[metrics] pyav vs torchcodec"
if [[ "${RUN_DATALOADER_PROFILE}" -eq 1 ]]; then
    run_in_container "source switch_env dreamzero && $(profile_env_prefix)export PYTHONPATH=/workspace/RLinf:/opt/dreamzero:\${PYTHONPATH:-} && cd /workspace/RLinf && python toolkits/lerobot/profile_dreamzero_videoloader.py ${BASE_PROFILE_ARGS} --mode metrics --profile-target dataloader --backends pyav torchcodec --warmup-steps ${METRICS_WARMUP} --steps ${METRICS_STEPS} --repeats ${METRICS_REPEATS} --output-csv /outputs/metrics.csv --output-step-csv /outputs/metrics_steps.csv"
fi

if [[ "${RUN_DATALOADER_PROFILE}" -eq 1 && "${SKIP_TIMELINE}" -eq 0 ]]; then
    mkdir -p "${OUTPUT_DIR}/nsys" "${OUTPUT_DIR}/torch_traces"
    for backend in pyav torchcodec; do
        echo "[timeline] backend=${backend}"
        run_in_container "source switch_env dreamzero && $(profile_env_prefix)export PYTHONPATH=/workspace/RLinf:/opt/dreamzero:\${PYTHONPATH:-} && cd /workspace/RLinf && nsys_help=\$(nsys profile --help 2>&1 || true) && NSYS_ARGS='-t cuda,nvtx,osrt -f true' && if grep -q -- '--trace-fork-before-exec' <<<\"\${nsys_help}\"; then NSYS_ARGS=\"\${NSYS_ARGS} --trace-fork-before-exec=true\"; fi && if grep -q -- '--sample' <<<\"\${nsys_help}\"; then NSYS_ARGS=\"\${NSYS_ARGS} --sample=cpu\"; fi && if grep -q -- '--cpuctxsw' <<<\"\${nsys_help}\"; then NSYS_ARGS=\"\${NSYS_ARGS} --cpuctxsw=process-tree\"; fi && nsys profile \${NSYS_ARGS} -o /outputs/nsys/${backend} python toolkits/lerobot/profile_dreamzero_videoloader.py ${BASE_PROFILE_ARGS} --mode timeline --profile-target dataloader --backends ${backend} --warmup-steps ${TIMELINE_WARMUP} --steps ${TIMELINE_STEPS} --repeats 1 --enable-ranges --output-csv /outputs/timeline_${backend}.csv --output-step-csv /outputs/timeline_${backend}_steps.csv"
        if [[ "${EMIT_TORCH_TRACE}" -eq 1 ]]; then
            echo "[torch-trace] backend=${backend}"
            run_in_container "source switch_env dreamzero && $(profile_env_prefix)export PYTHONPATH=/workspace/RLinf:/opt/dreamzero:\${PYTHONPATH:-} && cd /workspace/RLinf && python toolkits/lerobot/profile_dreamzero_videoloader.py ${BASE_PROFILE_ARGS} --mode timeline --profile-target dataloader --backends ${backend} --warmup-steps ${TIMELINE_WARMUP} --steps ${TIMELINE_STEPS} --repeats 1 --enable-ranges --emit-torch-trace --torch-trace-dir /outputs/torch_traces --output-csv /outputs/torch_trace_${backend}.csv --output-step-csv /outputs/torch_trace_${backend}_steps.csv"
        fi
    done
fi

if [[ "${RUN_SYSTEM_PROFILE}" -eq 1 ]]; then
    SYSTEM_ARGS="${BASE_PROFILE_ARGS} --profile-target system --step-workload ${SYSTEM_WORKLOAD} --synthetic-sleep-ms ${SYSTEM_SLEEP_MS} --profile-device ${SYSTEM_PROFILE_DEVICE} --cuda-matmul-size ${SYSTEM_CUDA_MATMUL_SIZE} --cuda-matmul-iters ${SYSTEM_CUDA_MATMUL_ITERS} --cuda-matmul-dtype ${SYSTEM_CUDA_MATMUL_DTYPE}"
    if [[ "${SYSTEM_COPY_BATCH_TO_DEVICE}" -eq 1 ]]; then
        SYSTEM_ARGS="${SYSTEM_ARGS} --copy-batch-to-device"
    fi
    if [[ "${SYSTEM_SYNC_STEP}" -eq 0 ]]; then
        SYSTEM_ARGS="${SYSTEM_ARGS} --no-step-sync"
    fi

    SYSTEM_LABEL="${SYSTEM_WORKLOAD}_${SYSTEM_SLEEP_MS}ms"
    echo "[system-metrics] pyav vs torchcodec workload=${SYSTEM_WORKLOAD} sleep_ms=${SYSTEM_SLEEP_MS}"
    run_in_container "source switch_env dreamzero && $(profile_env_prefix)export PYTHONPATH=/workspace/RLinf:/opt/dreamzero:\${PYTHONPATH:-} && cd /workspace/RLinf && python toolkits/lerobot/profile_dreamzero_videoloader.py ${SYSTEM_ARGS} --mode metrics --backends pyav torchcodec --warmup-steps ${SYSTEM_WARMUP} --steps ${SYSTEM_STEPS} --repeats ${SYSTEM_REPEATS} --output-csv /outputs/system_metrics_${SYSTEM_LABEL}.csv --output-step-csv /outputs/system_steps_${SYSTEM_LABEL}.csv"

    if [[ "${SKIP_TIMELINE}" -eq 0 ]]; then
        mkdir -p "${OUTPUT_DIR}/nsys"
        for backend in pyav torchcodec; do
            echo "[system-timeline] backend=${backend} workload=${SYSTEM_WORKLOAD} sleep_ms=${SYSTEM_SLEEP_MS}"
            run_in_container "source switch_env dreamzero && $(profile_env_prefix)export PYTHONPATH=/workspace/RLinf:/opt/dreamzero:\${PYTHONPATH:-} && cd /workspace/RLinf && nsys_help=\$(nsys profile --help 2>&1 || true) && NSYS_ARGS='-t cuda,nvtx,osrt -f true' && if grep -q -- '--trace-fork-before-exec' <<<\"\${nsys_help}\"; then NSYS_ARGS=\"\${NSYS_ARGS} --trace-fork-before-exec=true\"; fi && if grep -q -- '--sample' <<<\"\${nsys_help}\"; then NSYS_ARGS=\"\${NSYS_ARGS} --sample=cpu\"; fi && if grep -q -- '--cpuctxsw' <<<\"\${nsys_help}\"; then NSYS_ARGS=\"\${NSYS_ARGS} --cpuctxsw=process-tree\"; fi && nsys profile \${NSYS_ARGS} -o /outputs/nsys/system_${SYSTEM_LABEL}_${backend} python toolkits/lerobot/profile_dreamzero_videoloader.py ${SYSTEM_ARGS} --mode timeline --backends ${backend} --warmup-steps ${TIMELINE_WARMUP} --steps ${TIMELINE_STEPS} --repeats 1 --enable-ranges --output-csv /outputs/system_timeline_${SYSTEM_LABEL}_${backend}.csv --output-step-csv /outputs/system_timeline_${SYSTEM_LABEL}_${backend}_steps.csv"
        done
    fi
fi

echo "[done] outputs: ${OUTPUT_DIR}"
