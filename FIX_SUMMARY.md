# Fix Summary: Distributed Training Initialization Issue

## Problem

When running the 3B/7B training scripts with `torchrun`, the following error occurred:

```
ValueError: Default process group has not been initialized, please make sure to call init_process_group.
```

This happened at line 143 in `scripts/base_train.py` when calling `model.setup_optimizers()`, which internally tries to create a `DistMuon` optimizer that requires an initialized distributed process group.

## Root Cause

In `nanochat/common.py`, the `compute_init()` function only initialized the distributed process group when both conditions were met:
1. DDP mode was detected (via environment variables from `torchrun`)
2. `device_type == "cuda"`

However, when PyTorch failed to detect GPUs (autodetecting as `cpu`), the function would skip DDP initialization even though `torchrun` was being used with 8 processes. This left the processes in an inconsistent state where:
- They were launched by `torchrun` with DDP environment variables set
- But the process group was never initialized
- Causing the error when distributed optimizers tried to use `dist.get_rank()`

## Solution

Modified `nanochat/common.py` to properly initialize the distributed process group for all device types:

### Before:
```python
# Distributed setup: Distributed Data Parallel (DDP), optional, and requires CUDA
ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
if ddp and device_type == "cuda":
    device = torch.device("cuda", ddp_local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()
else:
    device = torch.device(device_type)
```

### After:
```python
# Distributed setup: Distributed Data Parallel (DDP), optional
ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
if ddp:
    if device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
        dist.init_process_group(backend=backend, device_id=device)
    else:
        device = torch.device(device_type)
        backend = "gloo"
        dist.init_process_group(backend=backend)
    dist.barrier()
else:
    device = torch.device(device_type)
```

### Key Changes:
1. **Separated DDP detection from device type:** Now initializes DDP whenever `torchrun` is used, regardless of device
2. **Backend selection:** Uses `nccl` for CUDA and `gloo` for CPU/MPS
3. **Consistent initialization:** All ranks properly join the process group before proceeding

Also improved `compute_cleanup()`:
```python
# Before:
if is_ddp():
    dist.destroy_process_group()

# After:
if dist.is_available() and dist.is_initialized():
    dist.destroy_process_group()
```

This is more robust and checks the actual state of the distributed system.

## Additional Documentation

Created/updated three files:

1. **TROUBLESHOOTING.md** - Comprehensive troubleshooting guide covering:
   - This specific DDP initialization error
   - GPU detection issues (NVIDIA and AMD ROCm)
   - OOM errors and solutions
   - Performance debugging
   - AMD ROCm-specific guidance

2. **CHANGES_SUMMARY.md** - Added Q&A entries for:
   - AMD ROCm GPU support
   - DDP initialization error

3. **FIX_SUMMARY.md** (this file) - Technical details of the fix

## Testing the Fix

### Verify DDP Initialization Works:
```bash
# With CUDA GPUs:
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --depth=38 --device_batch_size=48 --num_iterations=10

# With CPU (for testing):
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --device_type=cpu --depth=12 --device_batch_size=4 --num_iterations=10
```

### Expected Output:
```
Autodetected device type: cuda  # or cpu if no GPUs
2025-11-27 XX:XX:XX,XXX - nanochat.common - INFO - Distributed world size: 8
Vocab size: 65,536
num_layers: 38
model_dim: 2432
...
Scaling the LR for the AdamW parameters ∝1/√(2432/768) = 0.561951
[Training proceeds without DDP initialization error]
```

## Impact

This fix enables:
1. ✅ Training on CPU with `torchrun` (useful for testing/debugging)
2. ✅ Training on AMD ROCm GPUs (MI300X, MI250X, etc.)
3. ✅ Better error handling in distributed scenarios
4. ✅ Consistent behavior across device types

## For AMD ROCm Users

The original issue you encountered had two components:

1. **DDP initialization bug** - Now fixed ✅
2. **GPU detection showing CPU** - This means PyTorch isn't detecting your MI300X GPUs

To fix GPU detection on AMD ROCm:

```bash
# Check if ROCm is visible to PyTorch
python -c "import torch; print('CUDA/ROCm available:', torch.cuda.is_available())"

# If False, reinstall PyTorch with ROCm support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

Once PyTorch properly detects your GPUs, the training scripts will automatically use them.

## Files Changed

1. `nanochat/common.py` - Fixed DDP initialization
2. `CHANGES_SUMMARY.md` - Added troubleshooting Q&A
3. `TROUBLESHOOTING.md` - New comprehensive troubleshooting guide
4. `FIX_SUMMARY.md` - This file

## Backward Compatibility

This fix is fully backward compatible:
- Existing scripts continue to work unchanged
- Single GPU training still works (DDP is optional)
- CUDA training behavior is identical
- New functionality added for CPU/MPS distributed training
