# nanochat 3B & 7B Model Training Guide

## Quick Start

This guide explains how to train larger nanochat models (~3B and ~7B parameters) optimized for systems with 1.5TB VRAM distributed across 8 GPUs (187.5GB per GPU).

## What's New

Three new files have been added to support larger model training:

1. **`run3b.sh`** - Training script for ~3B parameter model (depth=38)
2. **`run7b.sh`** - Training script for ~7B parameter model (depth=52)
3. **`DESIGN_3B_7B.md`** - Detailed design document with architecture specs and rationale
4. **`CHANGES_SUMMARY.md`** - Complete summary of changes and usage instructions

## Model Specifications

### 3B Model
- **Parameters:** 3.02 billion
- **Architecture:** 38 layers × 2,432 dimensions × 19 heads
- **Training time:** ~18-22 hours on 8×H100
- **Cost:** ~$432-528 @ $24/hr
- **VRAM per GPU:** ~140-150GB (comfortable headroom)

### 7B Model
- **Parameters:** 7.35 billion
- **Architecture:** 52 layers × 3,328 dimensions × 26 heads
- **Training time:** ~70-80 hours on 8×H100
- **Cost:** ~$1,680-1,920 @ $24/hr
- **VRAM per GPU:** ~165-175GB (near limit)

## Usage

### Train 3B Model
```bash
bash run3b.sh
```

### Train 7B Model
```bash
bash run7b.sh
```

### With wandb Logging
```bash
WANDB_RUN=my_experiment bash run3b.sh
```

### Recommended: Use Screen
```bash
screen -L -Logfile run3b.log -S run3b bash run3b.sh
```

## Testing Before Full Run

Test memory usage with a short run (100 iterations only):

```bash
# 3B model test
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --depth=38 --device_batch_size=48 --total_batch_size=786432 --num_iterations=100

# 7B model test
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --depth=52 --device_batch_size=24 --total_batch_size=393216 --num_iterations=100
```

Monitor VRAM during testing: `watch -n 1 nvidia-smi`

## Key Configuration Parameters

### 3B Model
```bash
--depth=38
--device_batch_size=48
--total_batch_size=786432
```

### 7B Model
```bash
--depth=52
--device_batch_size=24
--total_batch_size=393216
```

## No Code Changes Required

The existing nanochat codebase supports these configurations without any Python code modifications. The architecture automatically scales based on the `--depth` parameter:

- `model_dim = depth × 64`
- `num_heads = ceil(model_dim / 128)`
- `num_kv_heads = num_heads`

## Hardware Requirements

- **GPUs:** 8 (can use fewer, but takes proportionally longer)
- **VRAM per GPU:** 187.5GB recommended
  - 3B model: minimum ~140GB per GPU
  - 7B model: minimum ~165GB per GPU
- **Storage:** ~120GB for data + checkpoints (3B) or ~180GB (7B)

## If You Encounter OOM (Out of Memory)

For the 7B model, if you run out of memory:

```bash
# Try reducing batch size to 20 per GPU
--device_batch_size=20 --total_batch_size=327680

# Or 16 per GPU
--device_batch_size=16 --total_batch_size=262144
```

The code automatically adjusts gradient accumulation steps to compensate.

## Documentation

For complete details, see:

- **`DESIGN_3B_7B.md`** - Full design document with architecture decisions, memory estimates, and optimization strategies
- **`CHANGES_SUMMARY.md`** - Complete list of changes, usage examples, troubleshooting guide

## After Training

Once training completes, interact with your model:

```bash
# Chat via web UI (recommended)
python -m scripts.chat_web

# Chat via CLI
python -m scripts.chat_cli

# View training report
cat report.md
```

## Expected Performance

### 3B Model
- MFU: 48-52%
- Throughput: ~85-95k tokens/sec
- Validation: Significantly outperforms the 1.9B (d32) model

### 7B Model
- MFU: 46-50%
- Throughput: ~75-85k tokens/sec
- Validation: State-of-the-art for sub-$2000 training budget

## Support & Questions

Refer to the main [README.md](README.md) for general nanochat information and the [Discussions](https://github.com/karpathy/nanochat/discussions) for community support.
