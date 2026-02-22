# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Edinburgh Machine Learning Systems (EDIN MLS) Spring 2026 course repository. Teaches GPU programming for ML workloads via two tracks:

- **Triton**: OpenAI's cross-platform GPU compiler (recommended, works on sm70+)
- **cuTile**: NVIDIA-specific framework, targeting Blackwell (sm_120) GPUs

The main assignment (`hw1-asr/`) is implementing GPU-accelerated kernels for a speech recognition model (GLM-ASR).

## Environment Setup

```bash
# Triton track
source utils/setup-triton.sh -y

# cuTile track (Blackwell GPUs only)
source utils/setup-cutile.sh -y

# Verify environment
python triton-tutorial/0-environment/check.py
python cutile-tutorial/0-environment/check.py
```

Setup scripts auto-install conda environments with Python 3.11, torch, triton/cuTile, cupy, and ML dependencies.

## Running Tutorials

```bash
python triton-tutorial/1-vectoradd/vectoradd.py
python triton-tutorial/7-attention/attention.py
```

Each lesson has a `README.md` with explanation and a Python file to run directly.

## HW1: ASR Benchmarking

```bash
cd hw1-asr

# Run benchmarks (pass directory name, not path)
./benchmark.sh glm_asr_triton_example       # reference
./benchmark.sh glm_asr_triton_template      # student implementation

# Detailed profiling
./benchmark_detailed.sh glm_asr_triton_template

# Interactive demo (requires streamlit)
streamlit run demo.py
```

Or directly:
```bash
python hw1-asr/benchmark_student.py glm_asr_triton_template
python hw1-asr/benchmark_detailed.py glm_asr_triton_example
```

## HW1 Code Architecture

**Data flow**: Audio (wav) → AudioEncoder → Projector → TextDecoder → Text

Each implementation variant (`glm_asr_scratch`, `glm_asr_triton_example`, `glm_asr_cutile_example`, `glm_asr_triton_template`, `glm_asr_cutile_template`) shares this file layout:

| File | Role |
|------|------|
| `layers.py` | Linear, LayerNorm, RMSNorm, MLP, Embedding, Conv primitives |
| `attention.py` | Multi-head self-attention with FlashAttention patterns |
| `rope.py` | Rotary Position Embeddings (RoPE) |
| `conv.py` | 1D convolution and subsampling |
| `model.py` | Full AudioEncoder + Projector + TextDecoder assembly |
| `weight_loader.py` | Pre-trained weight loading |

**`glm_asr_scratch`** is the PyTorch CPU baseline — read this first to understand the logic before looking at GPU implementations.

**Student templates** have `TODO` markers in `layers.py` and `attention.py` where kernel implementations are required.

### Minimum HW1 Requirements

1. **Tile/block size tuning**: Test ≥2 configurations, pick optimal for target GPU
2. **Kernel fusion**: Fuse at least one pair of operations (e.g., linear + activation)
3. **FlashAttention-style attention**: Blockwise softmax with tiled K/V loading

## Teaching Cluster (Slurm)

See `Teaching Cluster.md` for full details. Key points:

- Request GPU jobs: `srun --gres=gpu:1 --pty bash`
- Blackwell nodes (cuTile): request specifically with `--constraint=blackwell`
- SSH tunnel for Streamlit on cluster: `bash hw1-asr/show_tunnel.sh <port>`
- **Memory allocation note**: On the teaching cluster, pre-allocate memory explicitly to avoid OOM errors (documented in `Teaching Cluster.md`)

## File Structure

```
edin-mls-26-spring/
│
├── README.md                           # Main entry point: quick start, tool comparison, GPU compatibility
├── Teaching Cluster.md                 # Slurm setup, SSH config, memory allocation tips
├── requirements-blackwell.lock         # Pinned deps for Blackwell GPU nodes
│
├── utils/
│   ├── setup-triton.sh                 # Creates conda env with torch + triton + cupy
│   ├── setup-cutile.sh                 # Creates conda env with CUDA 13.x + cuTile
│   ├── setup-cutile-fix.sh             # Same as above + extra ML deps (transformers, streamlit)
│   └── hack-hopper/                    # Stub cuTile shim to run cuTile code on Hopper GPUs
│       └── cuda/tile/__init__.py       # Monkey-patches cuTile API for sm_90 compatibility
│
├── triton-tutorial/                    # 7-lesson Triton tutorial track
│   ├── 0-environment/check.py          # Verifies torch + triton install
│   ├── 1-vectoradd/vectoradd.py        # Hello world: element-wise vector addition kernel
│   ├── 2-execution-model/
│   │   ├── sigmoid_1d.py               # 1D grid: per-element sigmoid
│   │   └── grid_2d.py                  # 2D grid: block indexing demo
│   ├── 3-data-model/data_types.py      # FP16 vs FP32 precision and casting
│   ├── 4-transpose/grid_2d.py          # Tiled matrix transpose (coalesced vs naive)
│   ├── 5-secret-notes/README.md        # Notes-only lesson (no runnable file)
│   ├── 6-performance-tuning/autotune_benchmark.py  # triton.autotune decorator usage
│   └── 7-attention/attention.py        # FlashAttention-style kernel in Triton
│
├── hw1-asr/                            # Homework 1: GPU-accelerated speech recognition
│   ├── README.md                       # Assignment overview and quickstart
│   ├── GUIDE.md                        # Step-by-step implementation walkthrough
│   ├── benchmark.sh                    # Wrapper: runs benchmark_student.py for a given impl dir
│   ├── benchmark_student.py            # Measures end-to-end inference latency
│   ├── benchmark_detailed.sh           # Wrapper: runs benchmark_detailed.py
│   ├── benchmark_detailed.py           # Per-layer profiling with torch profiler
│   ├── demo.py                         # Streamlit web UI for live transcription demo
│   ├── show_tunnel.sh                  # Prints SSH tunnel command for Streamlit on Slurm
│   ├── test_audio.wav                  # Sample audio for smoke-testing implementations
│   ├── test_audio.txt                  # Expected transcription for test_audio.wav
│   │
│   ├── glm_asr_scratch/                # PyTorch CPU reference — read this first
│   │   ├── config.py                   # Model hyperparameters (hidden size, heads, layers)
│   │   ├── layers.py                   # Pure PyTorch: Linear, Norm, MLP, Embedding
│   │   ├── attention.py                # Pure PyTorch multi-head attention
│   │   ├── rope.py                     # Rotary position embedding computation
│   │   ├── conv.py                     # 1D conv subsampling for audio
│   │   ├── encoder.py                  # AudioEncoder: conv stack + transformer layers
│   │   ├── decoder.py                  # TextDecoder: transformer + LM head
│   │   ├── model.py                    # Top-level: encoder + projector + decoder
│   │   ├── audio_features.py           # Mel spectrogram extraction from raw audio
│   │   ├── tokenizer.py                # Text tokenizer wrapper
│   │   ├── torch_glm.py                # Inference entrypoint (load weights, run model)
│   │   └── weight_loader.py            # HuggingFace checkpoint → model state dict
│   │
│   ├── glm_asr_triton_example/         # Complete Triton GPU reference implementation
│   │   └── (same files as scratch, minus config/audio/tokenizer/torch_glm)
│   │
│   ├── glm_asr_triton_template/        # Student starting point for Triton track
│   │   └── (same files; layers.py + attention.py have TODO stubs)
│   │
│   ├── glm_asr_cutile_example/         # Complete cuTile GPU reference implementation
│   │   └── (same files as triton_example, using cuTile/CuPy kernels)
│   │
│   └── glm_asr_cutile_template/        # Student starting point for cuTile track
│       └── (same files; layers.py + attention.py have TODO stubs)
```

## GPU Compatibility

| Architecture | Triton | cuTile |
|---|---|---|
| Blackwell (sm_120) | Yes | Yes (native) |
| Hopper (sm_90) | Yes | Via `utils/hack-hopper/` |
| Ada/Ampere (sm_89/80) | Yes | No |

## Triton Secret Notes

### 1. `tl.constexpr` is Required for Block Shapes

Any value used in `tl.arange`, block shapes, or static loops must be a `tl.constexpr`.

```python
@triton.jit
def kernel(X, Y, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)  # OK
```

### 2. Always Use Masks for Out-of-Bounds

When `N` or matrix sizes are not multiples of block sizes, mask accesses:

```python
mask = offs < N
x = tl.load(X + offs, mask=mask, other=0.0)
tl.store(Y + offs, x, mask=mask)
```

### 3. Be Explicit About Type Casting

Triton does not always implicitly cast between FP16 and FP32. Cast explicitly when you want higher-precision math:

```python
x_fp32 = x.to(tl.float32)
y_fp16 = y_fp32.to(tl.float16)
```

### 4. Strides Are in Elements, Not Bytes

Torch provides strides in **elements**, not bytes. Use them directly in pointer arithmetic:

```python
ptrs = X + offs_m * stride_m + offs_n * stride_n
```

### 5. Tune `num_warps` and `num_stages`

```python
kernel[grid](..., num_warps=4, num_stages=2)
```

- **`num_warps`**: controls parallelism inside a program
- **`num_stages`**: controls pipelining depth for memory ops

### 6. Use `triton.autotune` for Performance

```python
@triton.autotune(
    configs=[triton.Config({"BLOCK": 128}, num_warps=4),
             triton.Config({"BLOCK": 256}, num_warps=8)],
    key=["N"],
)
@triton.jit
def kernel(...):
    ...
```

### 7. Be Careful with Reductions

Reductions require extra care with accumulation precision and masks. Use FP32 accumulation when possible.

### 8. Debugging Tips

- Use `tl.device_print` if your Triton version supports it (slow, can overwhelm output)
- Start with small shapes and block sizes
- Validate against PyTorch or NumPy references