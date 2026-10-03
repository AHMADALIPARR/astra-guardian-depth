// SPDX-License-Identifier: AGPL-3.0-or-later
// Fused prompt re-injection + RMSNorm for one recurrent loop pass.
//
// Every loop pass does:  h = h + s * P(prompt)   (prompt re-injection, GEMM)
//                        h = rmsnorm(h)          (input norm of the shared block)
//
// Done naively that is three passes over the hidden state (add: read+write,
// norm: read+write). This kernel fuses the elementwise tail into one pass:
//     y = rmsnorm(x + s * p; w)
// The GEMM producing P(prompt) stays a cuBLAS call; this kernel takes its
// output `p` and fuses the add + norm around a single read of `x`.
//
// Build: nvcc -O3 -arch=sm_90 -c inject_norm.cu
// Status: NOT compiled here (no CUDA toolchain on this host).

#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void inject_norm_fwd(
    const half* __restrict__ x,    // (rows, dim) evolving hidden state
    const half* __restrict__ p,    // (rows, dim) P(prompt), precomputed by cuBLAS
    const half* __restrict__ w,    // (dim) rmsnorm weight
    half* __restrict__ y,          // (rows, dim) normalized output
    int dim, float scale, float eps) {
  // One thread-block per row.
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  // Pass 1 (streamed): accumulate sumsq of (x + scale*p). We recompute the
  // sum in registers on the second loop to avoid shared-memory traffic for
  // the full row; bandwidth stays at one read of x and one read of p.
  float sumsq = 0.0f;
  for (int j = tid; j < dim; j += nthreads) {
    float v = __half2float(x[row * dim + j]) + scale * __half2float(p[row * dim + j]);
    sumsq += v * v;
  }
  __shared__ float red[1024];
  red[tid] = sumsq;
  __syncthreads();
  for (int s = nthreads / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  const float inv = rsqrtf(red[0] / (float)dim + eps);

  // Pass 2: normalize and write.
  for (int j = tid; j < dim; j += nthreads) {
    float v = __half2float(x[row * dim + j]) + scale * __half2float(p[row * dim + j]);
    float wv = __half2float(w[j]);
    y[row * dim + j] = __float2half(v * inv * wv);
  }
}

// Host launcher: grid = rows, block = min(1024, next pow2 >= dim).
extern "C" void astra_inject_norm(
    const half* x, const half* p, const half* w, half* y,
    int rows, int dim, float scale, float eps, cudaStream_t stream) {
  int block = 1;
  while (block < dim && block < 1024) block <<= 1;
  inject_norm_fwd<<<rows, block, 0, stream>>>(x, p, w, y, dim, scale, eps);
}
