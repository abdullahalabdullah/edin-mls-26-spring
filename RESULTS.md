# HW1 End-to-End Benchmark Results


### glm_asr_triton_example (reference baseline)

```
Time:   1478.9ms (+/- 0.5ms)
Tokens: 13
Speed:  113.76ms/token

Transcription: Concord returned to its place amidst the tents.
Accuracy: 100.0%
Status:   PASS
```

GPU: saxa cluster node

### Comparison

| | Template | Example | Delta |
|---|---|---|---|
| Time | 1440.2ms | 1478.9ms | **-38.7ms (2.6% faster)** |
| Speed | 110.78ms/tok | 113.76ms/tok | **-2.98ms/tok** |
| Accuracy | 100% | 100% | — |

Template is already faster than the example before Phase 7 optimisations.


### glm_asr_triton_template (all kernels implemented - Phases 1-6, fused kernels OFF)


```
Time:   1478.7ms (+/- 0.2ms)
Tokens: 13
Speed:  113.75ms/token

Accuracy: 100.0%
Status:   PASS
```

GPU: saxa cluster node
Config: MLP.FUSED = False, EncoderMLP.FUSED = False

GPU: saxa cluster node
Config: MLP.FUSED = True, EncoderMLP.FUSED = True
Kernels implemented: silu, gelu, rmsnorm, layernorm, softmax, softmax_inplace, linear_kernel_tf32, attention_scores, attention_output, compute_freqs (RoPE)


### glm_asr_triton_template (all kernels implemented - Phases 1-6, fused kernels ON)

```
Time:   1440.2ms (+/- 0.2ms)
Tokens: 13
Speed:  110.78ms/token

Transcription: Concord returned to its place amidst the tents.
Expected:      CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS

Accuracy: 100.0%
Status:   PASS
```

GPU: saxa cluster node
Config: MLP.FUSED = True, EncoderMLP.FUSED = True
Kernels implemented: silu, gelu, rmsnorm, layernorm, softmax, softmax_inplace, linear_kernel_tf32, attention_scores, attention_output, compute_freqs (RoPE)



### Fusion Impact

| | Fused ON | Fused OFF | Speedup |
|---|---|---|---|
| Time | 1440.2ms | 1478.7ms | **2.6% faster** |
| Speed | 110.78ms/tok | 113.75ms/tok | **-2.97ms/tok** |

Fused kernels save ~38ms by combining matmul + activation into a single kernel pass, eliminating intermediate memory round-trips.



### Tile/block size tuning

**Baseline (Config A):** `TILE_M=64, TILE_N=64, TILE_K=32, num_warps=4`

**Config B:** `TILE_M=128, TILE_N=128, TILE_K=32, num_warps=8`
- Doubled both output tile dimensions (M and N) so each program computes a 128×128 output block instead of 64×64
- Doubled num_warps from 4→8 to match the larger register footprint per program
- Applied to `Linear`, `MLP` (swiglu_fused_kernel), and `EncoderMLP` (linear_gelu_kernel)
- Larger tiles amortise kernel launch overhead and improve L2 cache reuse for the model's large hidden sizes (audio: 1280, text: 2048)

```
Time:   1377.8ms (+/- 15.4ms)
Tokens: 13
Speed:  105.98ms/token

Accuracy: 100.0%
Status:   PASS
```

GPU: saxa cluster node

| | Config A (baseline) | Config B | Delta |
|---|---|---|---|
| Tiles | 64×64×32 | 128×128×32 | — |
| num_warps | 4 | 8 | — |
| Time | 1440.2ms | 1377.8ms | **-62.4ms (4.3% faster)** |
| Speed | 110.78ms/tok | 105.98ms/tok | **-4.8ms/tok** |

Config B is the new best — **62ms faster** than fused baseline, **101ms faster** than the example reference.
