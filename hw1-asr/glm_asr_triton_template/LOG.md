# LOG.md

This file tracks all implementations, experiments, and code changes made to complete HW1, so teammates can follow the work done.

---

## Phase 1 — Element-wise Activations (2026-02-22)

### What was implemented

Two element-wise activation kernels in `layers.py`:

**`silu_kernel`** (SiLU / Swish activation):
- Formula: `y = x * sigmoid(x) = x / (1 + exp(-x))`
- Pattern: `pid → offs = pid*BLOCK_SIZE + arange(0, BLOCK_SIZE) → mask → load → compute sigmoid → multiply → store`
- Accumulation in FP32 via `.to(tl.float32)` on load

**`gelu_kernel`** (GELU tanh approximation):
- Formula: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715*x³)))`
- Same 1D grid/mask pattern as SiLU
- Uses `tl.libdevice.tanh` for the tanh call (consistent with existing fused kernels in the file)

### Why
Phase 1 of PLAN.md — these are the simplest kernels, purely element-wise with no reductions.
The remaining TODOs (rmsnorm, layernorm, softmax, linear) are unimplemented stubs and will fall through to CPU/torch fallbacks until Phase 2+.

### How to test
```bash
cd hw1-asr/glm_asr_triton_template && python layers.py
```
Expected: each section prints `Input: ... -> Output: ...` with matching shapes and no errors.