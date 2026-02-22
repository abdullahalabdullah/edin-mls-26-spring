# Notes

## Block Size Performance Impact

Block size has a big impact on performance because it affects:
- **Occupancy** — how many programs fit on the GPU simultaneously
- **Memory coalescing** — larger blocks read memory more efficiently
- **Register pressure** — too large and programs compete for registers