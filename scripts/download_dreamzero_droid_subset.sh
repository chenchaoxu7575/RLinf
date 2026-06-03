#!/usr/bin/env bash
set -euo pipefail

REPO_ID="GEAR-Dreams/DreamZero-DROID-Data"
CHUNK_INDEX=0
PROFILE_BASE="${RLINF_PROFILE_BASE:-/tmp/rlinf_pub}"
LOCAL_DIR="${PROFILE_BASE}/datasets/DreamZero-DROID-Data-subset"
HF_ENDPOINT_VALUE="${HF_ENDPOINT:-}"
EXTRA_INCLUDE=()
MAX_EPISODES=0
VIDEO_KEYS=(
    "observation.images.exterior_image_1_left"
    "observation.images.exterior_image_2_left"
    "observation.images.wrist_image_left"
)
META_FILES=(
    "meta/episodes.jsonl"
    "meta/info.json"
    "meta/modality.json"
    "meta/relative_horizon_stats_dreamzero.json"
    "meta/relative_stats.json"
    "meta/relative_stats_dreamzero.json"
    "meta/stats.json"
    "meta/tasks.jsonl"
    "relative_stats_dreamzero.json"
)

usage() {
    cat <<EOF
Usage: bash scripts/download_dreamzero_droid_subset.sh [options]

Options:
  --local-dir PATH       Output dataset directory (default: ${LOCAL_DIR}).
  --repo-id ID           Hugging Face dataset repo id (default: ${REPO_ID}).
  --chunk-index N        Chunk index to download (default: ${CHUNK_INDEX}).
  --max-episodes N      Download the first N episodes from the chunk; 0 downloads the full chunk (default: 0).
  --hf-endpoint URL      Optional HF_ENDPOINT override.
  --include PATTERN      Additional huggingface-cli include pattern.
  -h, --help             Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --local-dir)
            LOCAL_DIR="$2"
            shift 2
            ;;
        --repo-id)
            REPO_ID="$2"
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
        --include)
            EXTRA_INCLUDE+=("$2")
            shift 2
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

if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "huggingface-cli is required. Install with: pip install -U huggingface_hub" >&2
    exit 1
fi

if [[ "${LOCAL_DIR}" == /data/droid* && "${RLINF_DREAMZERO_DOWNLOAD_OK:-0}" != 1 ]]; then
    echo "Refusing container download to ${LOCAL_DIR} without RLINF_DREAMZERO_DOWNLOAD_OK=1." >&2
    echo "Use scripts/run_dreamzero_videoloader_profile.sh so host disk checks run first." >&2
    exit 1
fi

CHUNK_DIR=$(printf "chunk-%03d" "${CHUNK_INDEX}")
mkdir -p "${LOCAL_DIR}"

if [[ "${MAX_EPISODES}" -gt 0 && "${MAX_EPISODES}" -lt 1000 ]]; then
    FILES=("${META_FILES[@]}")
    for episode_index in $(seq 0 $((MAX_EPISODES - 1))); do
        episode_name=$(printf "episode_%06d" "${episode_index}")
        FILES+=("data/${CHUNK_DIR}/${episode_name}.parquet")
        for video_key in "${VIDEO_KEYS[@]}"; do
            FILES+=("videos/${CHUNK_DIR}/${video_key}/${episode_name}.mp4")
        done
    done
else
    INCLUDES=(
        "meta/*"
        "relative_stats_dreamzero.json"
        "data/${CHUNK_DIR}/*"
        "videos/${CHUNK_DIR}/**"
    )
    INCLUDES+=("${EXTRA_INCLUDE[@]}")
fi

echo "[download] repo=${REPO_ID}"
echo "[download] chunk=${CHUNK_DIR}"
echo "[download] local_dir=${LOCAL_DIR}"
if [[ "${MAX_EPISODES}" -gt 0 && "${MAX_EPISODES}" -lt 1000 ]]; then
    echo "[download] first_episodes=${MAX_EPISODES}"
fi

if [[ "${MAX_EPISODES}" -gt 0 && "${MAX_EPISODES}" -lt 1000 && "${#EXTRA_INCLUDE[@]}" -eq 0 ]]; then
    if [[ -n "${HF_ENDPOINT_VALUE}" ]]; then
        echo "[download] HF_ENDPOINT=${HF_ENDPOINT_VALUE}"
        export HF_ENDPOINT="${HF_ENDPOINT_VALUE}"
    fi
    python - "${REPO_ID}" "${LOCAL_DIR}" "${FILES[@]}" <<'PY'
import sys
from huggingface_hub import hf_hub_download

repo_id = sys.argv[1]
local_dir = sys.argv[2]
files = sys.argv[3:]
for path in files:
    print(f"[download] {path}", flush=True)
    hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=path,
        local_dir=local_dir,
    )
PY
else
    CMD=(
        huggingface-cli download "${REPO_ID}"
        --repo-type dataset
        --local-dir "${LOCAL_DIR}"
        --include "${INCLUDES[@]}"
    )
    if [[ -n "${HF_ENDPOINT_VALUE}" ]]; then
        echo "[download] HF_ENDPOINT=${HF_ENDPOINT_VALUE}"
        HF_ENDPOINT="${HF_ENDPOINT_VALUE}" "${CMD[@]}"
    else
        "${CMD[@]}"
    fi
fi

echo "[verify] checking local LeRobot subset structure"
test -f "${LOCAL_DIR}/meta/info.json"
test -f "${LOCAL_DIR}/meta/episodes.jsonl"
test -d "${LOCAL_DIR}/data/${CHUNK_DIR}"
test -d "${LOCAL_DIR}/videos/${CHUNK_DIR}"

PARQUET_COUNT=$(find "${LOCAL_DIR}/data/${CHUNK_DIR}" -type f -name 'episode_*.parquet' | wc -l)
VIDEO_COUNT=$(find "${LOCAL_DIR}/videos/${CHUNK_DIR}" -type f \( -name '*.mp4' -o -name '*.avi' -o -name '*.mov' \) | wc -l)

if [[ "${PARQUET_COUNT}" -eq 0 ]]; then
    echo "No episode parquet files found under ${LOCAL_DIR}/data/${CHUNK_DIR}" >&2
    exit 1
fi
if [[ "${VIDEO_COUNT}" -eq 0 ]]; then
    echo "No video files found under ${LOCAL_DIR}/videos/${CHUNK_DIR}" >&2
    exit 1
fi

echo "[verify] parquet_files=${PARQUET_COUNT} video_files=${VIDEO_COUNT}"
echo "[done] subset ready at ${LOCAL_DIR}"
