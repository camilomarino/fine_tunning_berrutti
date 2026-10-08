#!/bin/bash
#SBATCH --job-name=berrutti-qwen38-lora
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=110G
#SBATCH --time=12:00:00
set -euo pipefail
umask 077
: "${EXPERIMENT_ROOT:?Set the private experiment root in WORKDIR}"
: "${MODE:?Set MODE=sanity or MODE=full}"
FEASIBILITY_ROOT="$WORKDIR/berrutti-finetune-feasibility/20261007-r1"
export HF_HOME="$FEASIBILITY_ROOT/hf-cache"
KERNEL_PATH="$(cat "$FEASIBILITY_ROOT/kernel-hub-path.txt")"
export PYTHONPATH="$KERNEL_PATH:$EXPERIMENT_ROOT/code:$EXPERIMENT_ROOT/code/src"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
RUN_NAME="${RUN_NAME:-$MODE}"
"$FEASIBILITY_ROOT/env/bin/python" -m experiments.berrutti_qwen38.train \
  --dataset "$EXPERIMENT_ROOT/dataset" \
  --split "$EXPERIMENT_ROOT/code/experiments/berrutti_qwen38/split-v1.json" \
  --output "$EXPERIMENT_ROOT/runs/$RUN_NAME" \
  --mode "$MODE" \
  --max-pixels 2097152 \
  --max-new-tokens 4096
