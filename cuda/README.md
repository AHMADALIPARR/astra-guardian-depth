<!-- SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0 -->
<!-- Copyright (C) 2026 SnapKitty Collective -->
# CUDA performance layer

Custom kernels for the looped transformer. The recurrent core runs the same
block `r` times, so the loop step is **memory-bandwidth bound**: every pass
re-reads the hidden state. These kernels cut that traffic.

| Kernel | File | What it does |
|---|---|---|
| Prompt re-injection + RMSNorm | `inject_norm.cu` | Fuses `y = rmsnorm(x + s·P(prompt))` — the cuBLAS GEMM for `P` stays outside; the elementwise add+norm is one kernel |
| Ring KV cache | `kv_ring.cu` | K/V write + gather across recurrent passes; pass *k+1* reuses pass *k*'s K/V without recompute |

## Optimizations applied

`inject_norm.cu`:
- `half2` vectorized loads/stores (2 elements per instruction); scalar tail
  for odd dims.
- Warp-shuffle reduction for the norm denominator — no 1 KB shared-memory
  array, one `__syncthreads`, lower shared-memory pressure.
- `__launch_bounds__` to guide register allocation; block size is always a
  multiple of 32.

`kv_ring.cu`:
- `half2` vectorized copies (asserts even `d_head`).
- One block per head — the head index comes from `blockIdx`, removing a
  division per thread; `const __restrict__` enables read-only cache loads.

## Build

```bash
nvcc -O3 -arch=sm_90 -c inject_norm.cu
nvcc -O3 -arch=sm_90 -c kv_ring.cu
```

## Verification status

**Not compiled, profiled, or run on this host** — no CUDA toolchain and no
GPU here. The optimizations above are code-level (fewer instructions, less
shared memory, vectorized traffic) and were verified by reading only. They
must be built and benchmarked with `nvcc` + `nsys`/`ncu` on a CUDA machine
before any performance claim is made.
