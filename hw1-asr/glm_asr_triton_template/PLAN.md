# HW1-ASR Implementation Plan

## Context

The assignment is to implement GPU kernels (in Triton) for the GLM-ASR speech recognition model. The template has 10 kernel stubs (`pass`) across 3 files. Implementing them correctly gives 60 pts (correctness), then optimising for speed gives 30 pts (performance vs. example baseline), and code quality gives 10 pts. A reference solution already exists in `glm_asr_triton_example/` — read it when stuck.

**Do NOT modify**: `model.py`, `weight_loader.py`, `conv.py`

---

## Files to Edit

| File | Kernels to implement |
|------|----------------------|
| `hw1-asr/glm_asr_triton_template/layers.py` | `silu_kernel`, `gelu_kernel`, `softmax_kernel`, `rmsnorm_kernel`, `layernorm_kernel`, `linear_kernel_tf32` |
| `hw1-asr/glm_asr_triton_template/attention.py` | `attention_scores_kernel`, `softmax_inplace_kernel`, `attention_output_kernel` |
| `hw1-asr/glm_asr_triton_template/rope.py` | `compute_freqs_kernel` |

Reference for every kernel: the matching file in `hw1-asr/glm_asr_triton_example/`

---

## Step-by-Step Plan

### Phase 0 — Verify baseline runs (before writing any code)

```bash
cd hw1-asr
./benchmark.sh glm_asr_triton_example
```
Expected: `Accuracy: 100.0%  Status: PASS`. If it fails, fix environment first.

---

### Phase 1 — Element-wise activations (`layers.py`)

**Kernel: `silu_kernel`** (SiLU / Swish)
- Formula: `y = x * sigmoid(x)  =  x / (1 + exp(-x))`
- Grid: `(triton.cdiv(n_elements, BLOCK_SIZE),)` — already set by caller
- Pattern: `pid → offs → mask → load → compute → store`
- Key ops: `tl.exp(-x)`, element-wise multiply

**Kernel: `gelu_kernel`** (GELU tanh approximation)
- Formula: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))`
- Same 1D grid/mask pattern as silu
- Key op: `tl.libdevice.tanh(...)`

**Test after phase 1:**
```bash
cd hw1-asr/glm_asr_triton_template && python layers.py
```
Both activations should print correct output shapes without errors.

---

### Phase 2 — Reductions (`layers.py` + `attention.py`)

**Kernel: `rmsnorm_kernel`** (one row per program)
- Grid: `(batch_size,)` — each pid processes one row of `hidden_size` elements
- Steps: load row via `stride_x * pid + tl.arange(0, BLOCK_SIZE)`, mask → compute variance = `tl.sum(x*x) / hidden_size` → normalize `x / tl.sqrt(var + eps)` → multiply weight → store
- Accumulate in FP32, cast back to input dtype

**Kernel: `layernorm_kernel`** (one row per program)
- Grid: `(batch_size,)` — same shape as rmsnorm
- Steps: load row → compute mean = `tl.sum(x) / hidden_size` → center `x - mean` → compute var = `tl.sum(centered^2) / hidden_size` → normalize → multiply weight + add bias → store

**Kernel: `softmax_kernel`** (standalone softmax in `layers.py`)
- Grid: `(num_rows,)` — each pid processes one row of `num_cols` elements
- Steps: load row with mask → `row_max = tl.max(row, axis=0)` → `exp_x = tl.exp(row - row_max)` → `norm = tl.sum(exp_x)` → `out = exp_x / norm` → store

**Kernel: `softmax_inplace_kernel`** (in `attention.py` — used during attention)
- Grid: `(batch * num_heads * seq_q,)` — one program per attention row
- Same numerically stable softmax pattern as above, but in-place (writes back to same pointer)

**Test after phase 2:**
```bash
cd hw1-asr/glm_asr_triton_template && python layers.py   # RMSNorm, LayerNorm, Softmax
cd hw1-asr/glm_asr_triton_template && python attention.py  # partial (softmax_inplace)
```

---

### Phase 3 — Tiled matrix multiply (`layers.py`)

**Kernel: `linear_kernel_tf32`**
- Grid: `(triton.cdiv(M, TILE_M), triton.cdiv(N, TILE_N))`
- Each program computes a `TILE_M × TILE_N` output tile
- Loop over K tiles: load A tile `(TILE_M, TILE_K)` and B tile `(TILE_K, TILE_N)`, accumulate `acc += tl.dot(a, b)`
- After loop: store acc to C
- TILE defaults: `TILE_M=64, TILE_N=64, TILE_K=32` (already set by the `Linear` class)
- Use `input_precision="tf32"` in `tl.dot` for TF32 tensor cores

**Test after phase 3:**
```bash
cd hw1-asr/glm_asr_triton_template && python layers.py   # Linear layer test should pass
```

---

### Phase 4 — Attention kernels (`attention.py`)

**Kernel: `attention_scores_kernel`**
- Grid: `(batch * num_heads, seq_q)` — one program per (head, query position)
- Each program computes one row of the score matrix: `scores[q, :] = Q[q] @ K^T * scale`
- Load Q vector `(head_dim,)` once; loop over K rows in tiles, compute dot product accumulating scores; store score row

**Kernel: `attention_output_kernel`**
- Grid: `(batch * num_heads, seq_q)` — one program per (head, query position)
- Each program computes one row of the output: `out[q, :] = attn_weights[q, :] @ V`
- Load attn_weights row `(seq_k,)` once; loop over V rows in tiles; accumulate weighted sum; store output row `(head_dim,)`

**Test after phase 4:**
```bash
cd hw1-asr/glm_asr_triton_template && python attention.py  # all 4 attention tests should pass
```

---

### Phase 5 — Rotary Position Embeddings (`rope.py`)

**Kernel: `compute_freqs_kernel`**
- Grid: `(seq_len,)` — one program per sequence position
- Each program: load position index → load `inv_freq` vector `(half_dim,)` → compute `freqs = pos * inv_freq` → compute `cos_vals = tl.cos(freqs)`, `sin_vals = tl.sin(freqs)` → store `[cos_vals, cos_vals, sin_vals, sin_vals]` (duplicated halves) to output table

**Test after phase 5:**
```bash
cd hw1-asr/glm_asr_triton_template && python rope.py  # RoPE cos/sin and rotation tests
```

---

### Phase 6 — End-to-end correctness test

```bash
cd hw1-asr
./benchmark.sh glm_asr_triton_template
```
Expected: `Accuracy: 100.0%  Status: PASS`

If accuracy is 0% → some kernel still has `pass` or produces zeros.
If accuracy is partial → math bug in one kernel (compare output vs example with `diff` or print tensors).

---

### Phase 7 — Performance optimisations (mandatory for 30 pts)

Three mandatory items:

**7a. Enable fused kernels** — already implemented in template, just flip flags in `layers.py`:
```python
class MLP:
    FUSED = True   # enables swiglu_fused_kernel (gate+silu+multiply in one kernel)

class EncoderMLP:
    FUSED = True   # enables linear_gelu_kernel (matmul+GELU in one kernel)
```

**7b. Tile/block size tuning** — experiment with at least 2 configurations for `linear_kernel_tf32`. Try:
- Config A (default): `TILE_M=64, TILE_N=64, TILE_K=32, num_warps=4`
- Config B: `TILE_M=128, TILE_N=64, TILE_K=32, num_warps=8`
- Profile with `./benchmark_detailed.sh glm_asr_triton_template` to pick winner

**7c. FlashAttention-style attention** — the template's current attention is 3 separate kernels (scores → softmax → output). Optionally, fuse into a single blockwise FlashAttention kernel (like `triton-tutorial/7-attention/attention.py` implemented in the tutorial session) to further reduce memory bandwidth.

**Benchmark comparison:**
```bash
./benchmark.sh glm_asr_triton_example    # baseline time to beat
./benchmark.sh glm_asr_triton_template   # your time
./benchmark_detailed.sh glm_asr_triton_template  # per-operator breakdown
```

---

## Verification Checklist

- [ ] `python layers.py` passes all layer tests (RMSNorm, LayerNorm, GELU, SiLU, Linear, Softmax)
- [ ] `python attention.py` passes all 4 attention tests
- [ ] `python rope.py` passes RoPE tests
- [ ] `./benchmark.sh glm_asr_triton_template` → `Accuracy: 100.0%  Status: PASS`
- [ ] `./benchmark.sh glm_asr_triton_template` inference time < `./benchmark.sh glm_asr_triton_example` time
- [ ] `MLP.FUSED = True` and `EncoderMLP.FUSED = True` are set
- [ ] Block sizes tested with ≥2 configurations
