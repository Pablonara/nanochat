# Model Architecture Design for 3B and 7B Parameters

## Overview
This document outlines the design decisions for scaling nanochat models to ~3B and ~7B parameters, optimized for training with 1.5TB VRAM distributed across 8 GPUs (187.5GB per GPU).

## Hardware Specifications
- **Total VRAM**: 1.5TB (1,500GB)
- **Number of GPUs**: 8
- **VRAM per GPU**: 187.5GB
- **Comparison**: This is ~2.34× more VRAM per GPU than an 80GB H100 SXM

## Model Configurations

### 3B Model Configuration (depth=38)
**Architecture Parameters:**
- `depth`: 38 layers
- `model_dim`: 2,432 (depth × 64)
- `num_heads`: 19 (head dimension = 128)
- `num_kv_heads`: 19 (1:1 GQA ratio)
- `vocab_size`: 65,536
- `max_seq_len`: 2048

**Parameter Count Calculation:**
```
Embeddings: 2 × 65,536 × 2,432 = 318,767,104
Transformer layers: 38 × 12 × 2,432² = 2,697,507,840
Total: ~3.02B parameters
```

**Training Configuration:**
- `device_batch_size`: 48 tokens per GPU
- `total_batch_size`: 786,432 tokens (48 × 2048 × 8)
- `gradient_accumulation_steps`: 1 (no accumulation required)
- Estimated VRAM usage: ~140–150GB per GPU (comfortable headroom)
- `target_param_data_ratio`: 20 (Chinchilla-optimal)
- Training tokens: 3.02B × 20 = 60.4B tokens
- Training data shards: ~(60.4B × 4.8) / 250M ≈ 1,160 shards → download 1,200 for safety

**Expected Training Time & Cost:**
- FLOPs per token: ~7.3e9
- Total FLOPs: ~4.4e20
- Estimated time @ 50% MFU on 8×H100: ~18–22 hours
- Estimated cost @ $24/hr: ~$432–528

### 7B Model Configuration (depth=52)
**Architecture Parameters:**
- `depth`: 52 layers
- `model_dim`: 3,328 (depth × 64)
- `num_heads`: 26 (head dimension = 128)
- `num_kv_heads`: 26 (1:1 GQA ratio)
- `vocab_size`: 65,536
- `max_seq_len`: 2048

**Parameter Count Calculation:**
```
Embeddings: 2 × 65,536 × 3,328 = 436,207,616
Transformer layers: 52 × 12 × 3,328² = 6,911,244,288
Total: ~7.35B parameters
```

**Training Configuration:**
- `device_batch_size`: 24 tokens per GPU
- `total_batch_size`: 393,216 tokens (24 × 2048 × 8)
- `gradient_accumulation_steps`: 1 (no accumulation required)
- Estimated VRAM usage: ~165–175GB per GPU (close to limit; monitor OOM)
- `target_param_data_ratio`: 20 (Chinchilla-optimal)
- Training tokens: 7.35B × 20 = 147B tokens
- Training data shards required for full coverage: ~(147B × 4.8) / 250M ≈ 2,822 shards
- Available data: 1,822 shards → ~1.55 epochs (acceptable for this scale)

**Expected Training Time & Cost:**
- FLOPs per token: ~1.76e10
- Total FLOPs: ~2.6e21
- Estimated time @ 50% MFU on 8×H100: ~70–80 hours
- Estimated cost @ $24/hr: ~$1,680–1,920

### Alternative 6.6B Model Configuration (depth=50)
If 7B proves too expensive or memory-intensive, a depth=50 (≈6.56B params) variant offers a balanced compromise:

**Training Hints:**
- `device_batch_size`: 20–24
- `total_batch_size`: 327,680–393,216 tokens
- Requires ~2,520 shards (≈1.4 epochs)
- Runtime: ~58–65 hours at 50% MFU

## Design Decisions & Rationale

1. **Aspect Ratio (64)**
   - Retained the default width-to-depth ratio to stay aligned with existing initialization and optimizer heuristics.

2. **Head Dimension (128)**
   - Matches established practice in contemporary transformer models, balancing expressiveness and efficiency.

3. **GQA Ratio (1:1)**
   - Stuck with full multi-head attention as the available VRAM comfortably supports it. Future work could explore grouped-query attention to save memory.

4. **Batch Sizing**
   - Chosen to maximize GPU utilization while avoiding gradient accumulation. Larger `total_batch_size` values ensure the base training assertion (`total_batch_size % world_tokens == 0`) holds.

5. **Data Requirements**
   - 3B configuration fits within available shards.
   - 7B requires reusing data (~1.55 epochs). Acceptable for experimentation; reduce `target_param_data_ratio` if strict single-pass training is desired.

6. **Sequence Length (2048)**
   - Kept at 2048 to avoid quadratic memory blow-ups. Longer sequence lengths would demand reduced batch sizes or activation checkpointing.

7. **Optimizer Settings**
   - No changes required; the existing Muon + AdamW setup automatically scales learning rates based on model dimension.

## Memory Estimates

### 3B Model per GPU
```
Parameters (bf16):             ~6.0 GB
Optimizer states:              ~18.0 GB
Activations (batch=48):        ~95–100 GB
Gradients:                     ~6.0 GB
Buffers / overhead:            ~15–20 GB
Total:                         ~140–150 GB
```

### 7B Model per GPU
```
Parameters (bf16):             ~14.7 GB
Optimizer states:              ~44.1 GB
Activations (batch=24):        ~80–90 GB
Gradients:                     ~14.7 GB
Buffers / overhead:            ~15–20 GB
Total:                         ~165–175 GB
```

If OOM occurs on the 7B run:
- Reduce `device_batch_size` to 20, 16, or 12.
- Optional future work: add activation checkpointing or mixed-precision optimizer state sharding.

## Implementation Summary

### New Training Scripts
1. `run3b.sh`
   - Depth=38 with `device_batch_size=48` and `total_batch_size=786432`.
   - Downloads 1,200 shards to cover the 60.4B-token training corpus.

2. `run7b.sh`
   - Depth=52 with `device_batch_size=24` and `total_batch_size=393216`.
   - Downloads all 1,822 available shards and documents the resulting ~1.55 epochs.

Both scripts mirror the structure of `run1000.sh`/`speedrun.sh` while updating comments for clarity and including explicit memory considerations.

### No Changes to Core Model Code
- The existing scaling logic (`model_dim = depth * 64`, `head_dim = 128`) remains valid.
- CLI overrides handle new batch sizes and total batch sizes without requiring code changes.

## Testing & Validation Strategy

Before committing to the full runs:
1. **Dry Run:** Invoke each script with `--num_iterations=100` (passed through CLI) to sanity-check memory usage.
2. **Monitor VRAM:** Run `watch -n 1 nvidia-smi` during early steps; ensure usage stays below ~180GB.
3. **Metrics:** Confirm MFU stabilizes above 45% once warm-up completes.
4. **Data Pipeline:** Validate shard downloads finish prior to training start (scripts block on completion where necessary).

## Future Optimizations
- **Activation Checkpointing:** Would allow even larger batch sizes or deeper models.
- **FlashAttention / xFormers:** Potential to lower activation memory footprint.
- **Optimizer State Sharding (ZeRO):** Could unlock >7B models within the same VRAM budget.
- **Sequence Length Scaling:** With reduced batch sizes, explore 4096-token context to improve usability.

## Conclusion
The new configuration presets deliver a clear path to training nanochat-sized models at ~3B and ~7B parameters on a single node with 1.5TB VRAM. The accompanying scripts document the required dataset sizes, expected runtimes, and actionable fallback options should memory pressure arise.
