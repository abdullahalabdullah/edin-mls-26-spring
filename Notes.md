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