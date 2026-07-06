#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SEEDS=(42 43 44 45 46)
NUM_EPOCHS=30
NUM_FACTOR=96
HIDDEN_SIZE=64
NUM_PORTFOLIO=128
SAVE_DIR="./best_models"
CONDA_ENV="factorvae"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

checkpoint_epoch() {
  local checkpoint="$1"
  python -c "import sys, torch
path = sys.argv[1]
try:
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
except TypeError:
    checkpoint = torch.load(path, map_location='cpu')
print(checkpoint['epoch'])" "${checkpoint}"
}

run_one() {
  local seed="$1"
  local universe="$2"
  local dataset="$3"
  local run_name="FactorVAE-${universe}"
  local checkpoint="${SAVE_DIR}/${run_name}_factor_${NUM_FACTOR}_hdn_${HIDDEN_SIZE}_port_${NUM_PORTFOLIO}_seed_${seed}_checkpoint.pt"

  echo "============================================================"
  echo "Seed: ${seed} | Universe: ${universe} | Dataset: ${dataset}"
  echo "============================================================"

  local resume_args=()
  if [[ -f "${checkpoint}" ]]; then
    local epoch
    epoch="$(checkpoint_epoch "${checkpoint}")"
    if (( epoch + 1 >= NUM_EPOCHS )); then
      echo "Checkpoint is already complete through epoch $((epoch + 1))/${NUM_EPOCHS}. Skipping."
      return
    fi
    echo "Found checkpoint: ${checkpoint}"
    echo "Resuming this run from epoch $((epoch + 2))."
    resume_args=(--resume auto)
  else
    echo "No checkpoint found. Starting this run from scratch."
  fi

  python main.py \
    --dataset "${dataset}" \
    --run_name "${run_name}" \
    --seed "${seed}" \
    --num_epochs "${NUM_EPOCHS}" \
    --num_factor "${NUM_FACTOR}" \
    --hidden_size "${HIDDEN_SIZE}" \
    --num_portfolio "${NUM_PORTFOLIO}" \
    "${resume_args[@]}"
}

for seed in "${SEEDS[@]}"; do
  run_one "${seed}" "csi300" "./data/csi_data.pkl"
  run_one "${seed}" "sp500" "./data/sp500_data.pkl"
done

echo "All queued training runs finished."
