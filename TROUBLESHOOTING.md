# Troubleshooting Guide for 3B/7B Training

## Common Issues and Solutions

### 1. "Default process group has not been initialized" Error

**Symptom:**
```
ValueError: Default process group has not been initialized, please make sure to call init_process_group.
```

**Cause:**
The distributed process group wasn't being initialized for non-CUDA devices (CPU/MPS) when using `torchrun`.

**Solution:**
This has been fixed in `nanochat/common.py`. The `compute_init()` function now properly initializes the process group using the `gloo` backend for CPU/MPS devices and `nccl` for CUDA devices.

**Verification:**
After the fix, you should see successful initialization when running:
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=38 --num_iterations=10
```

---

### 2. Device Detected as CPU Instead of GPU

**Symptom:**
```
Autodetected device type: cpu
```
Even though you have GPUs available (NVIDIA or AMD).

**Causes and Solutions:**

#### For NVIDIA GPUs:
Check if CUDA is available:
```bash
source .venv/bin/activate
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"
```

If `False`, your PyTorch installation might not have CUDA support:
```bash
# Reinstall PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

#### For AMD ROCm GPUs (MI300X, MI250X, etc.):
AMD ROCm uses CUDA API compatibility, so `torch.cuda.is_available()` should still return `True`.

Check ROCm installation:
```bash
rocm-smi  # Should show your GPUs
```

Verify PyTorch ROCm support:
```bash
source .venv/bin/activate
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('HIP version:', torch.version.hip if hasattr(torch.version, 'hip') else 'N/A')"
```

If ROCm support is missing, install the ROCm-compatible PyTorch:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

**Manual Override:**
If autodetection fails but GPUs are available, you can manually specify the device type:
```bash
# Modify the run script to use --device_type=cuda
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --device_type=cuda --depth=38 --device_batch_size=48 --total_batch_size=786432
```

---

### 3. Out of Memory (OOM) Errors

**Symptom:**
```
RuntimeError: CUDA out of memory.
```

**Solutions:**

For 3B model (if OOM with device_batch_size=48):
```bash
# Try reducing to 32
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --depth=38 --device_batch_size=32 --total_batch_size=524288

# Or 24
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --depth=38 --device_batch_size=24 --total_batch_size=393216
```

For 7B model (if OOM with device_batch_size=24):
```bash
# Try reducing to 20
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --depth=52 --device_batch_size=20 --total_batch_size=327680

# Or 16
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --depth=52 --device_batch_size=16 --total_batch_size=262144
```

The code will automatically adjust gradient accumulation steps to maintain consistent training dynamics.

---

### 4. Missing Checkpoint Directory

**Symptom:**
```
FileNotFoundError: [Errno 2] No such file or directory: '/root/.cache/nanochat/base_checkpoints'
```

**Cause:**
This error occurs when evaluation scripts (`base_loss.py`, `base_eval.py`) run before training has completed and saved checkpoints.

**Solution:**
This is expected if `base_train.py` failed to complete. Fix the underlying training error first (see issues #1 and #2 above), then re-run the full script.

If you want to run evaluation scripts independently, ensure training has completed:
```bash
# Check if checkpoints exist
ls -la ~/.cache/nanochat/base_checkpoints/

# If no checkpoints, run training first
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=38 --device_batch_size=48
```

---

### 5. Extremely Long Training Times

**Expected Times:**
- **3B model:** 18-22 hours on 8×H100 GPUs
- **7B model:** 70-80 hours on 8×H100 GPUs

**If training is much slower:**

1. **Check MFU (Model FLOPs Utilization):**
   Look for this in training output: `mfu: 48.52`
   - Target: 45-52% for H100
   - If below 30%, something is wrong

2. **Verify GPU utilization:**
   ```bash
   watch -n 1 nvidia-smi
   # Or for AMD:
   watch -n 1 rocm-smi
   ```
   - GPU utilization should be 95-100%
   - GPU memory should be at expected levels (140-150GB for 3B, 165-175GB for 7B)

3. **Check if running on CPU:**
   If you see `Autodetected device type: cpu`, see issue #2 above

4. **Verify data loading:**
   Ensure data shards are pre-downloaded and not downloading during training:
   ```bash
   ls -la ~/.cache/nanochat/tokenized_data/ | wc -l
   ```

---

### 6. Training Script Fails Immediately

**Symptom:**
Script exits without clear error message or with torchrun errors.

**Debug Steps:**

1. **Test single GPU first:**
   ```bash
   python -m scripts.base_train -- --depth=20 --num_iterations=10
   ```

2. **Test with minimal settings:**
   ```bash
   torchrun --standalone --nproc_per_node=2 -m scripts.base_train -- \
     --depth=12 --device_batch_size=8 --num_iterations=10
   ```

3. **Check environment variables:**
   ```bash
   echo $NANOCHAT_BASE_DIR
   echo $WANDB_RUN
   # Ensure they're set or use defaults
   ```

4. **Verify dependencies:**
   ```bash
   source .venv/bin/activate
   uv sync --extra gpu
   python -c "import torch, rust_bpe; print('OK')"
   ```

---

### 7. AMD ROCm Specific Issues

**Additional checks for MI300X/MI250X systems:**

1. **Verify ROCm drivers:**
   ```bash
   rocm-smi --showproductname
   ```

2. **Check PyTorch ROCm compatibility:**
   ```bash
   python -c "import torch; print('ROCm version:', torch.version.hip if hasattr(torch.version, 'hip') else 'Not installed')"
   ```

3. **Set ROCm environment variables (if needed):**
   ```bash
   export HSA_OVERRIDE_GFX_VERSION=9.4.2  # For MI250X
   export HSA_OVERRIDE_GFX_VERSION=9.4.0  # For MI300X (check your specific version)
   ```

4. **Test basic ROCm functionality:**
   ```bash
   python -c "import torch; x = torch.randn(100, 100).cuda(); print('ROCm works:', x.device)"
   ```

---

## Getting Help

If issues persist:

1. **Collect diagnostic information:**
   ```bash
   python -m scripts.base_train -- --help  # Check available options
   nvidia-smi  # Or rocm-smi for AMD
   cat ~/.cache/nanochat/report.md  # If training completed
   ```

2. **Check the GitHub issues:** [nanochat issues](https://github.com/karpathy/nanochat/issues)

3. **Minimal reproduction:** Try to reproduce with the smallest possible configuration:
   ```bash
   python -m scripts.base_train -- --depth=12 --num_iterations=100 --device_batch_size=16
   ```

---

## Quick Reference

### Working Configurations (1.5TB VRAM across 8 GPUs)

**3B Model (Conservative):**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --depth=38 --device_batch_size=48 --total_batch_size=786432 --run=3b-test
```

**7B Model (Near VRAM Limit):**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --depth=52 --device_batch_size=24 --total_batch_size=393216 --run=7b-test
```

**Test Run (Fast, 10 minutes):**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --depth=20 --device_batch_size=32 --num_iterations=100
```
