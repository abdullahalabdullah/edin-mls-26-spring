# HW1 End-to-End Benchmark Results

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
