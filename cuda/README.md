# CUDA performance layer

Custom kernels for the looped transformer. The recurrent core runs the same
block `r` times, so the loop step is **memory-bandwidth bound**: every pass
re-reads the hidden state. These kernels cut that traffic.

| Kernel | File | What it fuses |
|---|---|---|
| Prompt re-injection + RMSNorm | `inject_norm.cu` | `y = rmsnorm(x + s·P(prompt))` in one pass over `x` (plus the cuBLAS GEMM for `P`) |
| Ring KV cache | `kv_ring.cu` | K/V write + gather across recurrent passes; pass *k+1* reuses pass *k*'s K/V without recompute |

## Build

```bash
nvcc -O3 -arch=sm_90 -c inject_norm.cu
nvcc -O3 -arch=sm_90 -c kv_ring.cu
```

## Verification status

**Not compiled or run on this host** — no CUDA toolchain and no GPU here.
The kernels are written against the CUDA C++ API and reviewed by reading, not
by `nvcc`. They must be built and tested on a CUDA machine before any
performance claim is made.
