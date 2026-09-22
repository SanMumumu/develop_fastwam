#!/usr/bin/env bash
# Launch RoutedWAM training. Mirrors scripts/train_action_dream_threshold.sh:
# the only difference is the python entrypoint, which selects
# RoutedWan22Trainer so the router warmup, the distillation ramp and the EMA
# teacher update receive optimizer-step progress.
set -euo pipefail

NPROC_PER_NODE="${1:?Usage: bash scripts/train_routed_wam.sh <nproc_per_node> [hydra_overrides...]}"
shift

NUM_MACHINES="${NNODES:-1}"
MACHINE_RANK="${NODE_RANK:-0}"
MAIN_PROCESS_IP="${MASTER_ADDR:-127.0.0.1}"
MAIN_PROCESS_PORT="${MASTER_PORT:-29500}"

# `accelerate launch --num_processes` is the GLOBAL rank count across all nodes.
TOTAL_PROCESSES=$((NPROC_PER_NODE * NUM_MACHINES))

accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes "${TOTAL_PROCESSES}" \
  --num_machines "${NUM_MACHINES}" \
  --machine_rank "${MACHINE_RANK}" \
  --main_process_ip "${MAIN_PROCESS_IP}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  --deepspeed_multinode_launcher standard \
  scripts/train_routed_wam.py \
  "$@"
