#!/usr/bin/env bash
set -euo pipefail

WALA_ROOT="${WALA_ROOT:-/mnt/data/ljh/WALA}"
WALA_PYTHON="${WALA_PYTHON:-/usr/local/bin/python3}"
CKPT_PATH="${1:-${CKPT_PATH:-}}"
BASE_PORT="${2:-6666}"

if [[ -z "${CKPT_PATH}" ]]; then
    echo "Usage: bash examples/Robotwin/eval_files/run_robotwin_policy_servers_ppu.sh <results/Checkpoints/<run_id>/final_model/pytorch_model.pt> [base_port]" >&2
    exit 1
fi
JOBS_PER_GPU="${JOBS_PER_GPU:-4}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3}"
LOG_DIR="${LOG_DIR:-$(dirname "${CKPT_PATH}")/robotwin_ppu_server_logs/$(date +%Y%m%d_%H%M%S)}"

export PPU_SDK=/usr/local/PPU_SDK
export PPU_HOME=/usr/local/PPU_SDK
export PPU_PATH=/usr/local/PPU_SDK
export CUDA_HOME=/usr/local/PPU_SDK/CUDA_SDK
export CUDA_PATH=/usr/local/PPU_SDK/CUDA_SDK
export LD_LIBRARY_PATH=/usr/local/PPU_SDK/targets/x86_64-linux/lib:/usr/local/PPU_SDK/CUDA_SDK/lib64:/usr/local/PPU_SDK/lib:${LD_LIBRARY_PATH:-}
export PATH=/usr/local/PPU_SDK/CUDA_SDK/bin:/usr/local/PPU_SDK/bin:/usr/local/PPU_SDK/asight/bin:/usr/local/PPU_SDK/ppu-smi/bin:${PATH}
export PYTHONPATH="${WALA_ROOT}:${PYTHONPATH:-}"

mkdir -p "${LOG_DIR}"

IFS=',' read -ra GPU_IDS <<< "${GPU_IDS_CSV}"
pids=()

cleanup() {
    local pid
    for pid in "${pids[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

echo "[INFO] WALA_ROOT=${WALA_ROOT}"
echo "[INFO] WALA_PYTHON=${WALA_PYTHON}"
echo "[INFO] CKPT_PATH=${CKPT_PATH}"
echo "[INFO] BASE_PORT=${BASE_PORT}"
echo "[INFO] GPU_IDS=${GPU_IDS_CSV}"
echo "[INFO] JOBS_PER_GPU=${JOBS_PER_GPU}"
echo "[INFO] LOG_DIR=${LOG_DIR}"

slot=0
for gpu in "${GPU_IDS[@]}"; do
    for ((repeat = 0; repeat < JOBS_PER_GPU; repeat++)); do
        port=$((BASE_PORT + slot))
        log_file="${LOG_DIR}/server_gpu${gpu}_port${port}.log"
        echo "[INFO] Launching server gpu=${gpu} port=${port} log=${log_file}"
        (
            cd "${WALA_ROOT}"
            CUDA_VISIBLE_DEVICES="${gpu}" "${WALA_PYTHON}" deployment/model_server/server_policy.py \
                --ckpt_path "${CKPT_PATH}" \
                --port "${port}" \
                --use_bf16
        ) > "${log_file}" 2>&1 &
        pids+=("$!")
        slot=$((slot + 1))
    done
done

echo "[INFO] Started ${#pids[@]} server processes."
echo "[INFO] Press Ctrl-C to stop them."
wait
