#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROJECT_PARENT=$(cd "${PROJECT_ROOT}/.." && pwd)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

DOCKER_BIN="${DOCKER_BIN:-docker}"
if ! command -v "${DOCKER_BIN}" >/dev/null 2>&1; then
    if [[ -x "${PROJECT_PARENT}/env/bin/docker" ]]; then
        DOCKER_BIN="${PROJECT_PARENT}/env/bin/docker"
    fi
fi

IMAGE="${RLINF_WM_NSYS_IMAGE:-rlinf:embodied-dreamzero-videoloader}"
DATASET_ROOT="${RLINF_WM_DATASET_ROOT:-/tmp/rlinf_pub/datasets/DreamZero-DROID-Data-subset-80ep}"
MODEL_ROOT="${RLINF_WM_MODEL_ROOT:-/tmp/rlinf_pub/models/DreamZero-DROID}"
TOKENIZER_ROOT="${RLINF_WM_TOKENIZER_ROOT:-/tmp/rlinf_pub/models/umt5-xxl}"
HF_HOME_ROOT="${RLINF_WM_HF_HOME:-/tmp/rlinf_pub/hf_home}"
OUTPUT_DIR="${RLINF_WM_NSYS_OUTPUT_DIR:-${PROJECT_ROOT}/codex_notes/wm_videoloader/results/worldmodel_videoloader_cpu_nsys/${TIMESTAMP}}"

MAX_STEPS="${RLINF_WM_NSYS_MAX_STEPS:-12}"
NUM_WORKERS="${RLINF_WM_NSYS_NUM_WORKERS:-4}"
PREFETCH_FACTOR="${RLINF_WM_NSYS_PREFETCH_FACTOR:-8}"
GLOBAL_BATCH_SIZE="${RLINF_WM_NSYS_GLOBAL_BATCH_SIZE:-8}"
MICRO_BATCH_SIZE="${RLINF_WM_NSYS_MICRO_BATCH_SIZE:-1}"
DOCKER_GPUS="${RLINF_WM_NSYS_DOCKER_GPUS:-all}"
EXPORT_SQLITE="${RLINF_WM_NSYS_EXPORT_SQLITE:-1}"
GPU_MONITOR="${RLINF_WM_NSYS_GPU_MONITOR:-1}"
NSYS_BASENAME="${RLINF_WM_NSYS_BASENAME:-worldmodel_videoloader_cpu}"
NSYS_CPU_SAMPLE="${RLINF_WM_NSYS_CPU_SAMPLE:-0}"
NSYS_CPUCTXSW="${RLINF_WM_NSYS_CPUCTXSW:-0}"
NSYS_PYTHON_SAMPLING="${RLINF_WM_NSYS_PYTHON_SAMPLING:-0}"
NSYS_WAIT_PRIMARY="${RLINF_WM_NSYS_WAIT_PRIMARY:-1}"
PROFILE_MODE="${RLINF_WM_NSYS_PROFILE_MODE:-outer}"
PROFILE_RANKS="${RLINF_WM_NSYS_PROFILE_RANKS:-0,1}"
PROFILE_CAPTURE_NAME="${RLINF_WM_NSYS_PROFILE_CAPTURE_NAME:-sft.worker.profile_window}"
PROFILE_FINALIZE_SLEEP="${RLINF_WM_NSYS_PROFILE_FINALIZE_SLEEP:-20}"
DRY_RUN=0
SKIP_PREFLIGHT=0

usage() {
    cat <<EOF
Usage: bash scripts/run_worldmodel_videoloader_nsys.sh [options]

Runs the DreamZero WorldModel SFT script under Nsight Systems with CPU + NVTX
capture suitable for inspecting DataLoader worker video loading.

Options:
  --output-dir PATH        Result directory (default: ${OUTPUT_DIR})
  --dataset-root PATH      DROID subset root (default: ${DATASET_ROOT})
  --model-root PATH        DreamZero model root (default: ${MODEL_ROOT})
  --tokenizer-root PATH    UMT5 tokenizer root (default: ${TOKENIZER_ROOT})
  --hf-home PATH           HF cache mounted into container (default: ${HF_HOME_ROOT})
  --image NAME             Docker image (default: ${IMAGE})
  --docker-gpus SPEC       Docker --gpus value (default: ${DOCKER_GPUS})
  --max-steps N            Training steps to profile (default: ${MAX_STEPS})
  --num-workers N          DataLoader workers per rank (default: ${NUM_WORKERS})
  --prefetch-factor N      DataLoader prefetch factor (default: ${PREFETCH_FACTOR})
  --global-batch-size N    Global batch size (default: ${GLOBAL_BATCH_SIZE})
  --micro-batch-size N     Micro batch size per rank (default: ${MICRO_BATCH_SIZE})
  --rank-profile           Profile only selected actor worker ranks via RLinf cluster.nsight
  --profile-ranks LIST     Comma-separated ranks for --rank-profile (default: ${PROFILE_RANKS})
  --no-gpu-monitor         Do not write monitor/gpu.csv
  --no-sqlite              Do not export the .nsys-rep to sqlite after profile
  --skip-preflight         Skip local nvidia-smi/ps snapshots
  --dry-run                Print the command plan without running
  -h, --help               Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --dataset-root)
            DATASET_ROOT="$2"
            shift 2
            ;;
        --model-root)
            MODEL_ROOT="$2"
            shift 2
            ;;
        --tokenizer-root)
            TOKENIZER_ROOT="$2"
            shift 2
            ;;
        --hf-home)
            HF_HOME_ROOT="$2"
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
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --prefetch-factor)
            PREFETCH_FACTOR="$2"
            shift 2
            ;;
        --global-batch-size)
            GLOBAL_BATCH_SIZE="$2"
            shift 2
            ;;
        --micro-batch-size)
            MICRO_BATCH_SIZE="$2"
            shift 2
            ;;
        --rank-profile)
            PROFILE_MODE="rank"
            shift
            ;;
        --profile-ranks)
            PROFILE_RANKS="$2"
            shift 2
            ;;
        --no-gpu-monitor)
            GPU_MONITOR=0
            shift
            ;;
        --no-sqlite)
            EXPORT_SQLITE=0
            shift
            ;;
        --skip-preflight)
            SKIP_PREFLIGHT=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
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

require_path() {
    local label=$1
    local path=$2
    if [[ ! -e "${path}" ]]; then
        echo "Missing ${label}: ${path}" >&2
        exit 1
    fi
}

require_path "dataset root" "${DATASET_ROOT}"
require_path "metadata.json" "${DATASET_ROOT}/metadata.json"
require_path "model root" "${MODEL_ROOT}"
require_path "tokenizer root" "${TOKENIZER_ROOT}"
mkdir -p "${OUTPUT_DIR}/nsys" "${OUTPUT_DIR}/step_csv" "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/monitor" "${HF_HOME_ROOT}"

DOCKER_RUN_ARGS=(
    --rm
    --gpus "${DOCKER_GPUS}"
    --ipc=host
    --ulimit memlock=-1
    --ulimit stack=67108864
    -v "${PROJECT_ROOT}:/workspace/RLinf"
    -v "${DATASET_ROOT}:/data/droid:ro"
    -v "${MODEL_ROOT}:/models/DreamZero-DROID:ro"
    -v "${TOKENIZER_ROOT}:/models/umt5-xxl:ro"
    -v "${HF_HOME_ROOT}:/hf_home"
    -v "${OUTPUT_DIR}:/outputs"
    -e "PYTHONDONTWRITEBYTECODE=1"
    -e "EMBODIED_PATH=/workspace/RLinf/examples/sft"
    -e "PYTHONPATH=/workspace/RLinf:/opt/dreamzero"
    -e "DREAMZERO_PATH=/opt/dreamzero"
    -e "RLINF_SFT_PROFILE=1"
    -e "RLINF_SFT_PROFILE_DIR=/outputs/step_csv"
    -e "RLINF_SFT_PROFILE_CAPTURE_WINDOW=1"
    -e "RLINF_SFT_PROFILE_CAPTURE_START_STEP=0"
    -e "RLINF_SFT_PROFILE_CAPTURE_END_STEP=${MAX_STEPS}"
    -e "RLINF_SFT_PROFILE_CAPTURE_NAME=${PROFILE_CAPTURE_NAME}"
    -e "RLINF_SFT_PROFILE_FINALIZE_SLEEP_SEC=${PROFILE_FINALIZE_SLEEP}"
    -e "RLINF_DREAMZERO_VIDEOLOADER_PROFILE=1"
    -e "TOKENIZERS_PARALLELISM=false"
    -e "NO_ALBUMENTATIONS_UPDATE=1"
    -e "HF_HOME=/hf_home"
    -e "HYDRA_FULL_ERROR=1"
    -w "/workspace/RLinf"
)

TRAIN_ARGS=(
    "examples/sft/train_vla_sft.py"
    "--config-path" "/workspace/RLinf/examples/sft/config"
    "--config-name" "droid_sft_dreamzero_14b"
    "runner.max_steps=${MAX_STEPS}"
    "runner.log_interval=1"
    "runner.save_interval=-1"
    "runner.val_check_interval=-1"
    "runner.logger.log_path=/outputs/logs"
    "data.train_data_paths=/data/droid"
    "data.num_workers=${NUM_WORKERS}"
    "data.prefetch_factor=${PREFETCH_FACTOR}"
    "data.video_backend=torchcodec"
    "actor.global_batch_size=${GLOBAL_BATCH_SIZE}"
    "actor.micro_batch_size=${MICRO_BATCH_SIZE}"
    "actor.model.model_path=/models/DreamZero-DROID"
    "actor.model.tokenizer_path=/models/umt5-xxl"
    "+actor.model.metadata_json_path=/data/droid/metadata.json"
)

case "${PROFILE_MODE}" in
    outer|rank)
        ;;
    *)
        echo "Unsupported profile mode: ${PROFILE_MODE}" >&2
        exit 1
        ;;
esac

if [[ "${PROFILE_MODE}" == "rank" ]]; then
    PROFILE_RANKS_LIST="[${PROFILE_RANKS// /}]"
    RANK_NSYS_ARGS="-t cuda,nvtx,osrt -f true --show-output=true --trace-fork-before-exec=true --osrt-threshold=1000 --capture-range=nvtx --nvtx-capture=${PROFILE_CAPTURE_NAME} --capture-range-end=stop-shutdown --kill=none --wait=all"
    TRAIN_ARGS+=(
        "+cluster.nsight.enabled=true"
        "+cluster.nsight.worker_groups=ActorGroup"
        "+cluster.nsight.worker_ranks=${PROFILE_RANKS_LIST}"
        "+cluster.nsight.options.t=cuda\\,nvtx\\,osrt"
        "+cluster.nsight.options.f=true"
        "+cluster.nsight.options.show-output=true"
        "+cluster.nsight.options.trace-fork-before-exec=true"
        "+cluster.nsight.options.osrt-threshold=1000"
        "+cluster.nsight.options.capture-range=nvtx"
        "+cluster.nsight.options.nvtx-capture=${PROFILE_CAPTURE_NAME}"
        "+cluster.nsight.options.capture-range-end=stop-shutdown"
        "+cluster.nsight.options.kill=none"
        "+cluster.nsight.options.wait=all"
    )
    if [[ "${NSYS_CPU_SAMPLE}" == "1" ]]; then
        RANK_NSYS_ARGS="${RANK_NSYS_ARGS} --sample=cpu"
        TRAIN_ARGS+=("+cluster.nsight.options.sample=cpu")
    fi
    if [[ "${NSYS_CPUCTXSW}" == "1" ]]; then
        RANK_NSYS_ARGS="${RANK_NSYS_ARGS} --cpuctxsw=process-tree"
        TRAIN_ARGS+=("+cluster.nsight.options.cpuctxsw=process-tree")
    fi
    if [[ "${NSYS_PYTHON_SAMPLING}" == "1" ]]; then
        RANK_NSYS_ARGS="${RANK_NSYS_ARGS} --python-sampling=true"
        TRAIN_ARGS+=("+cluster.nsight.options.python-sampling=true")
    fi
fi

write_preflight() {
    if [[ "${SKIP_PREFLIGHT}" -eq 1 ]]; then
        return
    fi
    {
        date
        echo
        echo "[docker ps]"
        "${DOCKER_BIN}" ps --format '{{.ID}} {{.Status}} {{.Names}}' || true
        echo
        echo "[nvidia-smi]"
        nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw,power.limit --format=csv,noheader,nounits || true
        echo
        echo "[top cpu]"
        ps -eo pid,ppid,pcpu,pmem,cmd --sort=-pcpu | head -n 80 || true
    } > "${OUTPUT_DIR}/preflight.txt"
}

MONITOR_CONTAINER="wm_videoloader_nsys_monitor_${TIMESTAMP}"
start_gpu_monitor() {
    if [[ "${GPU_MONITOR}" -ne 1 ]]; then
        return
    fi
    "${DOCKER_BIN}" run -d --rm \
        --name "${MONITOR_CONTAINER}" \
        --gpus "${DOCKER_GPUS}" \
        -v "${OUTPUT_DIR}/monitor:/monitor" \
        "${IMAGE}" \
        bash -lc "nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw,power.limit --format=csv,noheader,nounits -l 1 > /monitor/gpu.csv" >/dev/null
}

stop_gpu_monitor() {
    if [[ "${GPU_MONITOR}" -ne 1 ]]; then
        return
    fi
    "${DOCKER_BIN}" stop "${MONITOR_CONTAINER}" >/dev/null 2>&1 || true
}

run_profile() {
    local inner_script
    local train_args_quoted

    printf -v train_args_quoted '%q ' "${TRAIN_ARGS[@]}"
    inner_script=$(cat <<'INNER'
set -eo pipefail
source switch_env dreamzero 2>/dev/null || true
cd /workspace/RLinf

nsys_help=$(nsys profile --help 2>&1 || true)
NSYS_ARGS="-t cuda,nvtx,osrt -f true --show-output=true"
if grep -q -- "--trace-fork-before-exec" <<<"${nsys_help}"; then
    NSYS_ARGS="${NSYS_ARGS} --trace-fork-before-exec=true"
fi
if [[ "__NSYS_CPU_SAMPLE__" == "1" ]] && grep -q -- "--sample" <<<"${nsys_help}"; then
    NSYS_ARGS="${NSYS_ARGS} --sample=cpu"
fi
if [[ "__NSYS_CPUCTXSW__" == "1" ]] && grep -q -- "--cpuctxsw" <<<"${nsys_help}"; then
    NSYS_ARGS="${NSYS_ARGS} --cpuctxsw=process-tree"
fi
if [[ "__NSYS_PYTHON_SAMPLING__" == "1" ]] && grep -q -- "--python-sampling" <<<"${nsys_help}"; then
    NSYS_ARGS="${NSYS_ARGS} --python-sampling=true"
fi
if grep -q -- "--osrt-threshold" <<<"${nsys_help}"; then
    NSYS_ARGS="${NSYS_ARGS} --osrt-threshold=1000"
fi
if [[ "__NSYS_WAIT_PRIMARY__" == "1" ]] && grep -q -- "--wait" <<<"${nsys_help}"; then
    NSYS_ARGS="${NSYS_ARGS} --wait=primary"
fi

printf '%s\n' "${NSYS_ARGS}" > /outputs/nsys/nsys_args.txt
nsys profile ${NSYS_ARGS} -o "/outputs/nsys/__NSYS_BASENAME__" \
    /opt/venv/dreamzero/bin/python __TRAIN_ARGS__
INNER
)
    inner_script=${inner_script//__NSYS_BASENAME__/${NSYS_BASENAME}}
    inner_script=${inner_script//__NSYS_CPU_SAMPLE__/${NSYS_CPU_SAMPLE}}
    inner_script=${inner_script//__NSYS_CPUCTXSW__/${NSYS_CPUCTXSW}}
    inner_script=${inner_script//__NSYS_PYTHON_SAMPLING__/${NSYS_PYTHON_SAMPLING}}
    inner_script=${inner_script//__NSYS_WAIT_PRIMARY__/${NSYS_WAIT_PRIMARY}}
    inner_script=${inner_script//__TRAIN_ARGS__/${train_args_quoted}}
    "${DOCKER_BIN}" run "${DOCKER_RUN_ARGS[@]}" "${IMAGE}" bash -lc "${inner_script}"
}

run_rank_profile() {
    local inner_script
    local train_args_quoted

    printf -v train_args_quoted '%q ' "${TRAIN_ARGS[@]}"
    inner_script=$(cat <<'INNER'
set -eo pipefail
source switch_env dreamzero 2>/dev/null || true
cd /workspace/RLinf

cat > /outputs/nsys/nsys_args.txt <<'NSYS_ARGS'
rank_profile=true
worker_groups=ActorGroup
worker_ranks=__PROFILE_RANKS__
nsys profile __RANK_NSYS_ARGS__
NSYS_ARGS

/opt/venv/dreamzero/bin/python __TRAIN_ARGS__
INNER
)
    inner_script=${inner_script//__PROFILE_RANKS__/${PROFILE_RANKS}}
    inner_script=${inner_script//__RANK_NSYS_ARGS__/${RANK_NSYS_ARGS}}
    inner_script=${inner_script//__TRAIN_ARGS__/${train_args_quoted}}
    "${DOCKER_BIN}" run "${DOCKER_RUN_ARGS[@]}" "${IMAGE}" bash -lc "${inner_script}"
}

export_sqlite() {
    if [[ "${EXPORT_SQLITE}" -ne 1 ]]; then
        return
    fi
    if [[ "${PROFILE_MODE}" == "rank" ]]; then
        "${DOCKER_BIN}" run --rm \
            -v "${OUTPUT_DIR}:/outputs" \
            "${IMAGE}" \
            bash -lc "shopt -s nullglob; reps=(/outputs/logs/droid_sft_dreamzero/nsights/*.nsys-rep); if (( \${#reps[@]} == 0 )); then echo 'No rank-profile nsys reports found' >&2; exit 1; fi; for rep in \"\${reps[@]}\"; do nsys export --type sqlite --force-overwrite=true -o \"\${rep%.nsys-rep}.sqlite\" \"\${rep}\"; done"
    else
        "${DOCKER_BIN}" run --rm \
            -v "${OUTPUT_DIR}:/outputs" \
            "${IMAGE}" \
            bash -lc "nsys export --type sqlite --force-overwrite=true -o /outputs/nsys/${NSYS_BASENAME}.sqlite /outputs/nsys/${NSYS_BASENAME}.nsys-rep"
    fi
}

echo "[plan] output: ${OUTPUT_DIR}"
echo "[plan] max_steps=${MAX_STEPS} num_workers=${NUM_WORKERS} prefetch_factor=${PREFETCH_FACTOR}"
echo "[plan] profile_mode=${PROFILE_MODE}"
if [[ "${PROFILE_MODE}" == "rank" ]]; then
    echo "[plan] profile_ranks=${PROFILE_RANKS}"
    echo "[plan] nsys output: ${OUTPUT_DIR}/logs/droid_sft_dreamzero/nsights/rlinf_nsight_ActorGroup_<rank>_<pid>.nsys-rep"
else
    echo "[plan] nsys output: ${OUTPUT_DIR}/nsys/${NSYS_BASENAME}.nsys-rep"
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '[dry-run] docker run args:\n'
    printf '  %q' "${DOCKER_BIN}" run "${DOCKER_RUN_ARGS[@]}" "${IMAGE}"
    printf '\n[dry-run] train args:\n'
    printf '  %q' "${TRAIN_ARGS[@]}"
    printf '\n'
    exit 0
fi

write_preflight
trap stop_gpu_monitor EXIT
start_gpu_monitor
if [[ "${PROFILE_MODE}" == "rank" ]]; then
    run_rank_profile
else
    run_profile
fi
stop_gpu_monitor
export_sqlite

echo "[done] ${OUTPUT_DIR}"
