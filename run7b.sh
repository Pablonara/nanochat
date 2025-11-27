#!/bin/bash

# The 7B model tier of nanochat
# Designed to train a ~7.35B parameter model (depth=52) on 8 GPUs with 1.5TB total VRAM
# Expected runtime: ~70-80 hours on 8XH100 with ~187.5GB VRAM per GPU
# Expected cost: ~$1,680-1,920 at $24/hr
#
# NOTE: This configuration requires ~1.5 epochs over the available dataset (1822 shards)
# If you want to avoid multiple epochs, consider lowering depth to 50 (~6.6B params)
# or reducing target_param_data_ratio from 20 to ~13

# Setup
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi
python -m nanochat.report reset
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# Train tokenizer on ~4B characters and kick off download of the rest for pretraining
python -m nanochat.dataset -n 16
# Start downloading ALL available shards for 7B model training
# 7.35B params * 20 tokens/param = 147B tokens
# 147B tokens * 4.8 chars/token = 705.6B chars
# 705.6B chars / 250M chars/shard = 2,822 shards needed
# Dataset only has 1822 shards, so we'll need ~1.55 epochs
# Download all 1822 shards
python -m nanochat.dataset -n 1822 &
python -m scripts.tok_train --max_chars=4000000000
python -m scripts.tok_eval

# Model configuration for 7B parameters:
# depth=52
# model_dim=3328 (52 * 64)
# num_heads=26 (head_dim=128)
# num_params ~7.35B
# 
# With 187.5GB VRAM per GPU, we target device_batch_size=24
# Set total_batch_size to 393,216 tokens (24 * 2048 * 8)
# 
# WARNING: This is close to the VRAM limit. If you encounter OOM errors:
# - Reduce device_batch_size to 20, 16, or even 12
# - The code will automatically adjust gradient accumulation steps
# 
# Training details:
# - Tokens / micro-batch / rank: 24 x 2048 = 49,152
# - Tokens / micro-batch (global): 393,216 (8 GPUs)
# - Total batch size: 393,216 => gradient accumulation steps: 1
# - Target tokens: 147B (Chinchilla 20:1 ratio)
# - Estimated FLOPs per token: ~1.76e10
# - Total training FLOPs: ~2.6e21

# Number of processes/GPUs to use
NPROC_PER_NODE=8

# Base model pretraining
echo "Starting base model training (7B parameters, depth=52)..."
echo "NOTE: This will train for ~70-80 hours. Consider using screen or tmux."
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=52 --device_batch_size=24 --total_batch_size=393216 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval

# Midtraining
# Use the same device_batch_size and total_batch_size as base training for consistency
echo "Starting midtraining..."
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- --device_batch_size=24 --total_batch_size=393216 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i mid

# Supervised Finetuning
# SFT defaults work well; optional to adjust if desired
echo "Starting SFT..."
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft

# Generate final report
python -m nanochat.report generate

# Talk to it
echo "Training complete! To chat with your model, run:"
echo "python -m scripts.chat_web"
