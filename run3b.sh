#!/bin/bash

# The 3B model tier of nanochat
# Designed to train a ~3.02B parameter model (depth=38) on 8 GPUs with 1.5TB total VRAM
# Expected runtime: ~18-22 hours on 8XH100 with ~187.5GB VRAM per GPU
# Expected cost: ~$432-528 at $24/hr

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
# Start downloading shards for 3B model training
# 3.02B params * 20 tokens/param = 60.4B tokens
# 60.4B tokens * 4.8 chars/token = 289.92B chars
# 289.92B chars / 250M chars/shard = 1,160 shards needed
# Start with 1200 for safety (will use ~1.03 epochs if only 1,160 needed)
python -m nanochat.dataset -n 1200 &
python -m scripts.tok_train --max_chars=4000000000
python -m scripts.tok_eval

# Model configuration for 3B parameters:
# depth=38
# model_dim=2432 (38 * 64)
# num_heads=19 (head_dim=128)
# num_params ~3.02B
# 
# With 187.5GB VRAM per GPU, we target device_batch_size=48
# Set total_batch_size to 786,432 tokens to avoid gradient accumulation (48 * 2048 * 8)
# 
# Training details:
# - Tokens / micro-batch / rank: 48 x 2048 = 98,304
# - Tokens / micro-batch (global): 786,432 (8 GPUs)
# - Total batch size: 786,432 => gradient accumulation steps: 1
# - Target tokens: 60.4B (Chinchilla 20:1 ratio)
# - Estimated FLOPs per token: ~7.3e9
# - Total training FLOPs: ~4.4e20

# Number of processes/GPUs to use
NPROC_PER_NODE=8

# Base model pretraining
echo "Starting base model training (3B parameters, depth=38)..."
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=38 --device_batch_size=48 --total_batch_size=786432 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval

# Midtraining
# Use the same device_batch_size and total_batch_size as base training for consistency
echo "Starting midtraining..."
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- --device_batch_size=48 --total_batch_size=786432 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i mid

# Supervised Finetuning
# SFT defaults work well; optional to adjust --target_examples_per_step if desired
echo "Starting SFT..."
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft

# Generate final report
python -m nanochat.report generate

# Talk to it
echo "Training complete! To chat with your model, run:"
echo "python -m scripts.chat_web"
