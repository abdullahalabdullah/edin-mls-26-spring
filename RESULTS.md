# HW1 End-to-End Benchmark Results

GPU: saxa cluster node

---

## Reference Baseline — `glm_asr_triton_example`

```
Time:   1478.9ms (+/- 0.5ms)
Tokens: 13
Speed:  113.76ms/token

Transcription: Concord returned to its place amidst the tents.
Accuracy: 100.0%
Status:   PASS
```

---

## Template — Phases 1–6, fused kernels OFF

Config: `MLP.FUSED = False, EncoderMLP.FUSED = False`

Kernels: silu, gelu, rmsnorm, layernorm, softmax, softmax_inplace, linear_kernel_tf32, attention_scores, attention_output, compute_freqs (RoPE)

```
Time:   1478.7ms (+/- 0.2ms)
Tokens: 13
Speed:  113.75ms/token

Accuracy: 100.0%
Status:   PASS
```

---

## Template — Phases 1–6, fused kernels ON

Config: `MLP.FUSED = True, EncoderMLP.FUSED = True`

```
Time:   1440.2ms (+/- 0.2ms)
Tokens: 13
Speed:  110.78ms/token

Transcription: Concord returned to its place amidst the tents.
Accuracy: 100.0%
Status:   PASS
```

### Fusion Impact

Fused kernels combine matmul + activation into a single kernel pass, eliminating intermediate memory round-trips.

| | Fused ON | Fused OFF | Delta |
|---|---|---|---|
| Time | 1440.2ms | 1478.7ms | **-38.5ms (2.6% faster)** |
| Speed | 110.78ms/tok | 113.75ms/tok | **-2.97ms/tok** |

### Comparison vs Example Baseline (fused ON)

| | Template (fused ON) | Example | Delta |
|---|---|---|---|
| Time | 1440.2ms | 1478.9ms | **-38.7ms (2.6% faster)** |
| Speed | 110.78ms/tok | 113.76ms/tok | **-2.98ms/tok** |
| Accuracy | 100% | 100% | — |

---

## Phase 7b — Tile/Block Size Tuning

All configs use fused kernels ON. Applied to `Linear`, `MLP` (swiglu_fused_kernel), and `EncoderMLP` (linear_gelu_kernel).

### Config A (baseline): `TILE_M=64, TILE_N=64, TILE_K=32, num_warps=4`

Starting point — default tile sizes.

```
Time:   1440.2ms (+/- 0.2ms)
Speed:  110.78ms/token
Accuracy: 100.0% — PASS
```

### Config B: `TILE_M=128, TILE_N=128, TILE_K=32, num_warps=8`

- Doubled both output tile dimensions so each program computes a 128×128 block instead of 64×64
- Doubled num_warps 4→8 to match the larger register footprint per program
- Larger tiles amortise kernel launch overhead and improve L2 cache reuse for the model's large hidden sizes (audio: 1280, text: 2048)

```
Time:   1377.8ms (+/- 15.4ms)
Speed:  105.98ms/token
Accuracy: 100.0% — PASS
```

### Config C: `TILE_M=128, TILE_N=64, TILE_K=64, num_warps=8`

- Kept TILE_M=128 and num_warps=8 from Config B
- Narrowed TILE_N from 128→64 (smaller output tile width)
- Doubled TILE_K from 32→64 (deeper K-dimension accumulation per loop iteration)
- Halving the K-loop iterations reduces loop overhead and improves arithmetic intensity per kernel launch

```
Time:   1374.0ms (+/- 10.7ms)
Speed:  105.69ms/token
Accuracy: 100.0% — PASS
```

### Config D: `TILE_M=256, TILE_N=64, TILE_K=32, num_warps=8`

- Doubled TILE_M from 128→256 (each program processes 256 output rows at once)
- Hypothesis: taller M tile would reduce kernel launch overhead for long sequences
- Result: **slower than C** — 256-row tiles exceed register budget, causing register spilling

```
Time:   1436.8ms (+/- 1.8ms)
Speed:  110.53ms/token
Accuracy: 100.0% — PASS
```

### Summary

| | Config A | Config B | Config C | Config D |
|---|---|---|---|---|
| Tiles (M×N×K) | 64×64×32 | 128×128×32 | 128×64×64 | 256×64×32 |
| num_warps | 4 | 8 | 8 | 8 |
| Time | 1440.2ms | 1377.8ms | **1374.0ms** | 1436.8ms |
| Speed | 110.78ms/tok | 105.98ms/tok | **105.69ms/tok** | 110.53ms/tok |
| vs A | — | -4.3% | **-4.6%** | -0.2% |

**Winner: Config C** (`TILE_M=128, TILE_N=64, TILE_K=64, num_warps=8`) — best balance of tile size and register pressure. Active configuration.

### Final result vs example reference

| | Template (Config C) | Example | Delta |
|---|---|---|---|
| Time | 1374.0ms | 1478.9ms | **-104.9ms (7.1% faster)** |
| Speed | 105.69ms/tok | 113.76ms/tok | **-8.07ms/tok** |
| Accuracy | 100% | 100% | — |
