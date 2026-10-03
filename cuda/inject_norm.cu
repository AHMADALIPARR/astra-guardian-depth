// SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
// Copyright (C) 2026 SnapKitty Collective
// Fused prompt re-injection + RMSNorm for one recurrent loop pass.
//
// Every loop pass does:  h = h + s * P(prompt)   (prompt re-injection, GEMM)
//                        h = rmsnorm(h)          (input norm of the shared block)
//
// This kernel fuses the elementwise tail into:
//     y = rmsnorm(x + s * p; w)
// The GEMM producing P(prompt) stays a cuBLAS call; this kernel takes its
// output `p` and fuses the add + norm.
//
// Optimizations vs a naive fusion:
//   * half2 vectorized loads/stores: 2 elements per instruction, halves the
//     load/store instruction count on the memory-bound path.
//   * warp-shuffle reduction for sumsq: no 1 KB shared-memory array, single
//     __syncthreads, lower shared-memory pressure -> higher occupancy.
//   * __launch_bounds__ to guide the compiler's register allocation.
//   * Scalar tail for odd dim (model dims are even in practice).
//
// The norm still needs two passes over x/p (sumsq must be complete before
// any output is written); the win is fusing add+norm into those passes
// instead of running separate kernels.
//
// Build: nvcc -O3 -arch=sm_90 -c inject_norm.cu
// Status: NOT compiled here (no CUDA toolchain on this host).

#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void __launch_bounds__(1024, 4) inject_norm_fwd(
    const half* __restrict__ x,    // (rows, dim) evolving hidden state
    const half* __restrict__ p,    // (rows, dim) P(prompt), precomputed by cuBLAS
    const half* __restrict__ w,    // (dim) rmsnorm weight
    half* __restrict__ y,          // (rows, dim) normalized output
    int dim, float scale, float eps) {
  // One thread-block per row; block size is a multiple of 32 (see launcher).
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int lane = tid & 31;
  const int wid = tid >> 5;
  const int nwarps = nthreads >> 5;

  // Pass 1: sumsq of (x + scale*p), half2-vectorized when dim is even
  // (row*dim is then even, so the half2 base stays aligned).
  float sumsq = 0.0f;
  if ((dim & 1) == 0) {
    const half2* x2 = reinterpret_cast<const half2*>(x + row * dim);
    const half2* p2 = reinterpret_cast<const half2*>(p + row * dim);
    const int dim2 = dim >> 1;
    for (int j = tid; j < dim2; j += nthreads) {
      float2 a = __half22float2(x2[j]);
      float2 b = __half22float2(p2[j]);
      float v0 = a.x + scale * b.x;
      float v1 = a.y + scale * b.y;
      sumsq += v0 * v0 + v1 * v1;
    }
  } else {
    for (int j = tid; j < dim; j += nthreads) {
      float v = __half2float(x[row * dim + j])
              + scale * __half2float(p[row * dim + j]);
      sumsq += v * v;
    }
  }

  // Warp-shuffle reduction, then a 32-float shared staging area.
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    sumsq += __shfl_down_sync(0xffffffff, sumsq, off);
  __shared__ float warp_sums[32];
  if (lane == 0) warp_sums[wid] = sumsq;
  __syncthreads();
  float total = (tid < nwarps) ? warp_sums[tid] : 0.0f;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    total += __shfl_down_sync(0xffffffff, total, off);
  const float inv = rsqrtf(total / (float)dim + eps);

  // Pass 2: normalize and write.
  if ((dim & 1) == 0) {
    const half2* x2 = reinterpret_cast<const half2*>(x + row * dim);
    const half2* p2 = reinterpret_cast<const half2*>(p + row * dim);
    const half2* w2 = reinterpret_cast<const half2*>(w);
    half2* y2 = reinterpret_cast<half2*>(y + row * dim);
    const int dim2 = dim >> 1;
    for (int j = tid; j < dim2; j += nthreads) {
      float2 a = __half22float2(x2[j]);
      float2 b = __half22float2(p2[j]);
      float2 ww = __half22float2(w2[j]);
      float2 o;
      o.x = (a.x + scale * b.x) * inv * ww.x;
      o.y = (a.y + scale * b.y) * inv * ww.y;
      y2[j] = __float22half2_rn(o);
    }
  } else {
    for (int j = tid; j < dim; j += nthreads) {
      float v = __half2float(x[row * dim + j])
              + scale * __half2float(p[row * dim + j]);
      float wv = __half2float(w[j]);
      y[row * dim + j] = __float2half(v * inv * wv);
    }
  }
}

// Host launcher: grid = rows, block = pow2 in [32, 1024] covering dim.
extern "C" void astra_inject_norm(
    const half* x, const half* p, const half* w, half* y,
    int rows, int dim, float scale, float eps, cudaStream_t stream) {
  int block = 32;
  while (block < dim && block < 1024) block <<= 1;
  inject_norm_fwd<<<rows, block, 0, stream>>>(x, p, w, y, dim, scale, eps);
}
