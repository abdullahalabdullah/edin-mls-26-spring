# Notes

## Block Size Performance Impact

Block size has a big impact on performance because it affects:
- **Occupancy** — how many programs fit on the GPU simultaneously
- **Memory coalescing** — larger blocks read memory more efficiently
- **Register pressure** — too large and programs compete for registers

## FlashAttention vs Two-Pass Attention Benchmark

Both kernels produce correct softmax attention output. Two-pass reads K twice (once to find the row max, once to normalize). FlashAttention reads K once using an online softmax trick.

```
== Attention Performance Benchmark (both correct) ==
Seq Len    | TwoPass (ms)    | Flash (ms)     | Speedup
----------------------------------------------------------
128        | 0.0282          | 0.0189         | 1.50x
256        | 0.0544          | 0.0521         | 1.04x
512        | 0.1060          | 0.0562         | 1.89x
1024       | 0.3563          | 0.2018         | 1.77x
2048       | 1.3512          | 0.8001         | 1.69x
```

- Flash is consistently faster (~1.5–1.9x) because it eliminates the second pass over K
- The speedup is noisy at small sizes due to cache effects and kernel launch overhead
- The trend would be cleaner and larger at seq_len 4096+ where memory bandwidth dominates

## HW1 End-to-End Benchmark Results 

### glm_asr_triton_template (all kernels implemented - Phases 1-6 completed)

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
Kernels implemented: silu, gelu, rmsnorm, layernorm, softmax, softmax_inplace, linear_kernel_tf32, attention_scores, attention_output, compute_freqs (RoPE)

## Grading

| | Points | Requirement |
|---|---|---|
| **Correctness** | 60 | Transcription accuracy > 80% |
| **Performance** | 30 | Faster than the example baseline |
| **Code quality** | 10 | Clean, readable kernels |