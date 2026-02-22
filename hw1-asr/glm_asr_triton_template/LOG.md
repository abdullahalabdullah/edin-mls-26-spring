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

---

## Phase 2 — Reductions (2026-02-22)

### What was implemented

**`rmsnorm_kernel`** (`layers.py`):
- Grid: `(batch_size,)` — one program per row
- Load row → cast to FP32 → `var = sum(x*x) / hidden_size` → `x_norm = x * rsqrt(var + eps)` → multiply weight → store

**`layernorm_kernel`** (`layers.py`):
- Grid: `(batch_size,)` — same shape as RMSNorm
- Load row → mean → center → variance → normalize → multiply weight + add bias → store

**`softmax_kernel`** (`layers.py`):
- Grid: `(num_rows,)` — one program per row
- Numerically stable: load with `other=-inf` → subtract max → exp → sum → divide → store

**`softmax_inplace_kernel`** (`attention.py`):
- Grid: `(batch * num_heads * seq_q,)` — one program per attention score row
- Same numerically stable softmax, reads and writes back to the same pointer

### How to test
```bash
cd hw1-asr/glm_asr_triton_template && python layers.py   # RMSNorm, LayerNorm, Softmax
cd hw1-asr/glm_asr_triton_template && python attention.py  # partial (softmax_inplace only)
```

---

## Phase 3 — Tiled Matrix Multiply (2026-02-22)

### What was implemented

**`linear_kernel_tf32`** (`layers.py`):
- Grid: `(M // BLOCK_M, N // BLOCK_N)` — one program per output tile
- Each program accumulates a `BLOCK_M × BLOCK_N` output tile in FP32
- Loops over K in steps of `BLOCK_K`, loading A tile `(BLOCK_M, BLOCK_K)` and B tile `(BLOCK_K, BLOCK_N)`, accumulating via `tl.dot(..., input_precision="tf32")` for TF32 tensor cores
- Stores result with bounds mask
- Default tiles: `BLOCK_M=64, BLOCK_N=64, BLOCK_K=32`

Also updated `__main__` test to run both Triton and Torch backends and print the max diff for correctness validation.

### How to test
```bash
cd hw1-asr/glm_asr_triton_template && python layers.py
```
Expected: `Max diff vs torch: ~0.0` (small numerical difference due to TF32 rounding is OK)

---

## Phase 4 — Attention Kernels (2026-02-22)

### What was implemented

**`attention_scores_kernel`** (`attention.py`):
- Grid: `(batch * num_heads, seq_q)` — one program per (head, query position)
- Loads Q vector `(head_dim,)` for this query position
- Loads all K vectors `(seq_k, head_dim)` for this head in one block
- Computes `scores = sum(k * q[None, :], axis=1) * scale` — dot product for each key position
- Stores score row `(seq_k,)`

**`attention_output_kernel`** (`attention.py`):
- Grid: `(batch * num_heads, seq_q)` — one program per (head, query position)
- Loads attention weights row `(seq_k,)` (post-softmax)
- Loads all V vectors `(seq_k, head_dim)` for this head in one block
- Computes `out = sum(v * w[:, None], axis=0)` — weighted sum over key positions
- Stores output row `(head_dim,)`

### How to test
```bash
cd hw1-asr/glm_asr_triton_template && python attention.py
```
Expected: all 4 tests pass with non-zero output statistics (Mean/Std/Min/Max non-zero)

---

## Phase 5 — Rotary Position Embeddings (2026-02-22)

### What was implemented

**`compute_freqs_kernel`** (`rope.py`):
- Grid: `(seq_len,)` — one program per sequence position
- Loads position scalar `pos` from the positions array
- Loads `inv_freq` vector of size `half_dim` (= `rotary_dim // 2`)
- Computes `freqs = pos * inv_freq` (element-wise)
- Computes `cos_half = cos(freqs)`, `sin_half = sin(freqs)`
- Stores both halves duplicated: `cos_cache[pos, :half_dim] = cos_half` and `cos_cache[pos, half_dim:] = cos_half` (same for sin). This gives `cos_cache` shape `(seq_len, rotary_dim)` with the first and second halves identical, which is what `_apply_rope_single` expects.

### How to test
```bash
cd hw1-asr/glm_asr_triton_template && python rope.py
```
Expected: shapes print correctly and no errors

---

## Phase 7b — Tile Config C restored (2026-02-22)

Reverted tile settings back to the best config found during tuning (Config C):
- `Linear.TILE_M = 128, TILE_N = 64, TILE_K = 64, num_warps = 8`
- `MLP.TILE_M, TILE_N, TILE_K = 128, 64, 64, num_warps = 8`
- `EncoderMLP.TILE_M, TILE_N, TILE_K = 128, 64, 64, num_warps = 8`

Config C was the overall winner from tile tuning experiments (1374.0ms, vs 1440.2ms for the Config A baseline).

---

## Phase 7c — FlashAttention (2026-02-22)

### What was implemented

**`flash_attention_kernel`** (`attention.py`):
- Grid: `(batch * num_heads, seq_q)` — one program per query position
- Loads Q vector `(head_dim,)` once for this query position
- Iterates over K/V in `BLOCK_K=64`-tile chunks with **online softmax** (no full score matrix materialised in HBM)
- Per tile:
  1. Load K tile `(BLOCK_K, head_dim)` → compute `scores = dot(q, k^T) * scale` → shape `(BLOCK_K,)`
  2. Apply causal mask if `is_causal` (compile-time constexpr branch)
  3. Online softmax update: `m_new = max(m_i, max(scores_block))`, rescale previous `acc` by `exp(m_i - m_new)`, accumulate `exp(scores - m_new) * V`
  4. Load V tile `(BLOCK_K, head_dim)` → weighted sum into `acc (BLOCK_D,)`
- Normalize: `out = acc / l_i`
- Store result with `mask=offs_d < head_dim`
- SRAM footprint: `O(seq_k * head_dim)` per program vs. `O(seq_q * seq_k)` for the 3-kernel approach

The original 3-kernel approach (attention_scores_kernel → softmax_inplace_kernel → attention_output_kernel) is commented out in `scaled_dot_product_attention` and preserved for reference.

**Selection logic in `scaled_dot_product_attention`**:
- `use_flash = q.is_cuda and head_dim_padded <= MAX_ATTENTION_DIM and attention_mask is None`
- When `attention_mask is not None` (e.g. cross-attention padding), falls back to torch

### How to test
```bash
cd hw1-asr/glm_asr_triton_template && python attention.py
./benchmark.sh glm_asr_triton_template
```
Expected: attention.py tests pass; benchmark shows 100% accuracy.