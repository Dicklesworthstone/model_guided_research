# GPU Environment Notes (Repro + Parity)

Purpose: capture the full GPU stack (driver, CUDA, torch/JAX builds, and env flags) so runs are reproducible and comparable across machines.

This note mirrors the known-good settings observed in `bio_inspired_nanochat` and highlights the flags that matter most for FlexAttention/torch.compile.

## Quick capture (one-shot)

Run these before a training/benchmark run and paste the output into your run log:

```bash
nvidia-smi
python - <<'PY'
import os
import platform
import torch
try:
    import jax
except Exception as e:
    jax = None
print('python:', platform.python_version())
print('torch:', torch.__version__)
print('torch.cuda.is_available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('torch.cuda.device_count:', torch.cuda.device_count())
    print('torch.cuda.get_device_name(0):', torch.cuda.get_device_name(0))
    print('torch.version.cuda:', torch.version.cuda)
    print('torch.backends.cuda.matmul.allow_tf32:', torch.backends.cuda.matmul.allow_tf32)
    print('torch.backends.cudnn.allow_tf32:', torch.backends.cudnn.allow_tf32)
print('jax:', getattr(jax, '__version__', 'not installed'))
if jax is not None:
    try:
        print('jax.default_backend:', jax.default_backend())
        print('jax.devices:', jax.devices())
    except Exception as e:
        print('jax backend error:', e)
print('ENV: PYTORCH_CUDA_ALLOC_CONF=', os.environ.get('PYTORCH_CUDA_ALLOC_CONF'))
print('ENV: CUDA_VISIBLE_DEVICES=', os.environ.get('CUDA_VISIBLE_DEVICES'))
print('ENV: TORCH_LOGS=', os.environ.get('TORCH_LOGS'))
print('ENV: TORCHDYNAMO_VERBOSE=', os.environ.get('TORCHDYNAMO_VERBOSE'))
print('ENV: JAX_PLATFORM_NAME=', os.environ.get('JAX_PLATFORM_NAME'))
print('ENV: XLA_PYTHON_CLIENT_PREALLOCATE=', os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE'))
print('ENV: XLA_PYTHON_CLIENT_MEM_FRACTION=', os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION'))
PY
```

## Known-good flags from bio_inspired_nanochat

These flags are consistently set in the bio_inspired_nanochat GPU scripts and help avoid fragmentation or allocator stalls:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Recommended for torch.compile stability debugging (use only when needed):

```bash
export TORCH_LOGS=+dynamo
export TORCHDYNAMO_VERBOSE=1
```

## FlexAttention + torch.compile requirements

FlexAttention is only available on PyTorch 2.5+ and requires the FlexAttention import path:

```python
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
```

When using FlexAttention in nanochat:

- Ensure torch>=2.5.
- Prefer `--compile` during FlexAttention correctness/bench runs.
- If you hit dtype issues with bf16, test fp16 autocast.

Example (nanochat):

```bash
python -m nanochat.train \
  --attention-type standard \
  --use-flex-attention \
  --compile
```

## JAX GPU flags (when running demos)

For JAX demos, CPU is the default. To enable GPU explicitly:

```bash
export JAX_PLATFORM_NAME=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Optional: set a memory fraction if sharing GPUs
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
```

## Repro checklist

- Record `nvidia-smi` output (driver + CUDA runtime).
- Record torch + jax versions, and whether CUDA is detected.
- Record allocator flags (`PYTORCH_CUDA_ALLOC_CONF`).
- Record compile flags (`--compile` and any TorchDynamo envs).
- Record attention type and optimizer/scheduler selection.
- Keep the uv environment locked (`uv.lock`) and note any local overrides.

