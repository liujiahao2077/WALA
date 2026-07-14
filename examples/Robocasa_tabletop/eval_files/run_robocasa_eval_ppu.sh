#!/usr/bin/env bash
set -euo pipefail

# RoboCasa-GR1 tabletop evaluation on PPU.
# Usage:
#   bash examples/Robocasa_tabletop/eval_files/run_robocasa_eval_ppu.sh \
#     [results/Checkpoints/<run_id>/checkpoints/steps_xxx_pytorch_model.pt] \
#     [n_envs] [max_episode_steps] [n_action_steps]

WALA_ROOT="${WALA_ROOT:-/mnt/data/ljh/WALA}"
WALA_PYTHON="${WALA_PYTHON:-/usr/local/bin/python3}"
ROBOCASA_PYTHON="${ROBOCASA_PYTHON:-/mnt/data/ljh/envs/robocasa_eval_py310/bin/python}"
CKPT_DEFAULT="${CKPT_DEFAULT:-results/Checkpoints/wala_robocasa_gr1_tabletop/checkpoints/steps_70000_pytorch_model.pt}"

BASE_PORT="${BASE_PORT:-6666}"
NUM_GPUS="${NUM_GPUS:-2}"
N_EPISODES="${N_EPISODES:-50}"
ENV_LIMIT="${ENV_LIMIT:-0}"
SERVER_LAUNCH_GAP="${SERVER_LAUNCH_GAP:-8}"
SERVER_STARTUP_SLEEP="${SERVER_STARTUP_SLEEP:-30}"

N_ENVS_DEFAULT=1
MAX_EPISODE_STEPS_DEFAULT=720
N_ACTION_STEPS_DEFAULT=32

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '1,12p' "$0"
    exit 0
fi

cd "${WALA_ROOT}"

CKPT_PATH="${1:-$CKPT_DEFAULT}"
N_ENVS="${2:-$N_ENVS_DEFAULT}"
MAX_EPISODE_STEPS="${3:-$MAX_EPISODE_STEPS_DEFAULT}"
N_ACTION_STEPS="${4:-$N_ACTION_STEPS_DEFAULT}"

if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "Checkpoint path does not exist: ${CKPT_PATH}" >&2
    exit 1
fi
if (( NUM_GPUS < 1 )); then
    echo "NUM_GPUS must be >= 1, got ${NUM_GPUS}" >&2
    exit 1
fi

export PPU_SDK=/usr/local/PPU_SDK
export PPU_HOME=/usr/local/PPU_SDK
export PPU_PATH=/usr/local/PPU_SDK
export CUDA_HOME=/usr/local/PPU_SDK/CUDA_SDK
export CUDA_PATH=/usr/local/PPU_SDK/CUDA_SDK
export LD_LIBRARY_PATH=/usr/local/PPU_SDK/targets/x86_64-linux/lib:/usr/local/PPU_SDK/CUDA_SDK/lib64:/usr/local/PPU_SDK/lib:${LD_LIBRARY_PATH:-}
export PATH=/usr/local/PPU_SDK/CUDA_SDK/bin:/usr/local/PPU_SDK/bin:/usr/local/PPU_SDK/asight/bin:/usr/local/PPU_SDK/ppu-smi/bin:${PATH}
export PYTHONPATH="${WALA_ROOT}:/mnt/data/ljh/robocasa_eval_setup/robosuite:/mnt/data/ljh/robocasa_eval_setup/robocasa-gr1-tabletop-tasks:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-glx}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-glx}"
export ROBOCASA_ENABLE_RENDER="${ROBOCASA_ENABLE_RENDER:-1}"
export ROBOCASA_RECORD_VIDEO="${ROBOCASA_RECORD_VIDEO:-1}"

ENV_NAMES=(
  gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
)

if (( ENV_LIMIT > 0 && ENV_LIMIT < ${#ENV_NAMES[@]} )); then
    ENV_NAMES=("${ENV_NAMES[@]:0:${ENV_LIMIT}}")
fi

SAVE_ROOT="$(dirname "$(dirname "${CKPT_PATH}")")"
CKPT_NAME="$(basename "${CKPT_PATH}" .pt)"
LOG_DIR="${CKPT_PATH}.log/eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

SERVER_PIDS=()
EVAL_PIDS=()

cleanup() {
    local pid
    for pid in "${EVAL_PIDS[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
    for pid in "${SERVER_PIDS[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
}
trap cleanup INT TERM EXIT

safe_name() {
    local name="$1"
    name="${name//\//_}"
    name="${name//[^A-Za-z0-9_.-]/_}"
    printf '%s' "${name}"
}

run_eval_env() {
    local gpu_id="$1"
    local port="$2"
    local env_name="$3"
    local log_name
    local video_out_path
    log_name="$(safe_name "${env_name}")"
    video_out_path="${SAVE_ROOT}/videos/${CKPT_NAME}/n_action_steps_${N_ACTION_STEPS}_max_episode_steps_${MAX_EPISODE_STEPS}_n_envs_${N_ENVS}_${env_name}"
    mkdir -p "${video_out_path}"

    echo "Launching evaluation | GPU ${gpu_id} | Port ${port} | Env ${env_name}"

    local cmd=("${ROBOCASA_PYTHON}" examples/Robocasa_tabletop/eval_files/simulation_env_ppu.py
        --args.env_name "${env_name}"
        --args.port "${port}"
        --args.n_episodes "${N_EPISODES}"
        --args.n_envs "${N_ENVS}"
        --args.max_episode_steps "${MAX_EPISODE_STEPS}"
        --args.n_action_steps "${N_ACTION_STEPS}"
        --args.video_out_path "${video_out_path}"
        --args.pretrained_path "${CKPT_PATH}"
    )

    if command -v xvfb-run >/dev/null 2>&1 && [[ -z "${DISPLAY:-}" ]]; then
        cmd=(xvfb-run -a -s "-screen 0 1280x1024x24 -ac +extension GLX +render -noreset" "${cmd[@]}")
    fi

    CUDA_VISIBLE_DEVICES="${gpu_id}" "${cmd[@]}" > "${LOG_DIR}/eval_env_${log_name}_gpu${gpu_id}.log" 2>&1
}

wait_batch() {
    local failed=0
    local pid
    for pid in "$@"; do
        if ! wait "${pid}"; then
            failed=1
        fi
    done
    return "${failed}"
}

echo "=== RoboCasa PPU Evaluation ==="
echo "WALA Root          : ${WALA_ROOT}"
echo "WALA Python        : ${WALA_PYTHON}"
echo "RoboCasa Python    : ${ROBOCASA_PYTHON}"
echo "Checkpoint Path    : ${CKPT_PATH}"
echo "GPUs               : ${NUM_GPUS}"
echo "Number of Envs     : ${N_ENVS}"
echo "Max Episode Steps  : ${MAX_EPISODE_STEPS}"
echo "Action Chunk Length: ${N_ACTION_STEPS}"
echo "Episodes per Env   : ${N_EPISODES}"
echo "Env Count          : ${#ENV_NAMES[@]}"
echo "Log Directory      : ${LOG_DIR}"
echo "==============================="

for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    port=$((BASE_PORT + gpu_id))
    echo "Starting policy server | GPU ${gpu_id} | Port ${port}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${WALA_PYTHON}" deployment/model_server/server_policy.py \
        --ckpt_path "${CKPT_PATH}" \
        --port "${port}" \
        --use_bf16 \
        > "${LOG_DIR}/server_gpu${gpu_id}_port${port}.log" 2>&1 &
    SERVER_PIDS+=("$!")
    sleep "${SERVER_LAUNCH_GAP}"
done

sleep "${SERVER_STARTUP_SLEEP}"

for i in "${!SERVER_PIDS[@]}"; do
    pid="${SERVER_PIDS[$i]}"
    if ! kill -0 "${pid}" 2>/dev/null; then
        port=$((BASE_PORT + i))
        echo "Policy server on GPU ${i}, port ${port} exited during startup." >&2
        echo "See ${LOG_DIR}/server_gpu${i}_port${port}.log" >&2
        exit 1
    fi
done

declare -a batch_pids=()
failed=0
count=0
for env_name in "${ENV_NAMES[@]}"; do
    gpu_id=$((count % NUM_GPUS))
    port=$((BASE_PORT + gpu_id))
    run_eval_env "${gpu_id}" "${port}" "${env_name}" &
    pid="$!"
    EVAL_PIDS+=("${pid}")
    batch_pids+=("${pid}")

    count=$((count + 1))
    if (( ${#batch_pids[@]} >= NUM_GPUS )); then
        if ! wait_batch "${batch_pids[@]}"; then
            failed=1
        fi
        batch_pids=()
    fi
    sleep 2
done

if (( ${#batch_pids[@]} > 0 )); then
    if ! wait_batch "${batch_pids[@]}"; then
        failed=1
    fi
fi

if (( failed != 0 )); then
    echo "One or more evaluation jobs failed. Check logs in ${LOG_DIR}" >&2
    exit 1
fi

echo "=== Evaluation Finished ==="
