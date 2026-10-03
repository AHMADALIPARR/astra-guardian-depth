// SPDX-License-Identifier: AGPL-3.0-or-later
// Ring-buffer KV cache for recurrent passes.
//
// In a looped transformer the SAME block runs r times over the sequence, so a
// naive implementation recomputes K/V for all previous tokens on every pass.
// The ring buffer keeps the K/V of the last `capacity` (token, pass) pairs and
// lets pass k+1 attend to the keys/values written by pass k without
// recomputation, at the cost of capacity * d_model * 2 * sizeof(half) bytes.
//
// Layout: [head][ring_slot][d_head] for K and V separately, ring_slot =
// (pass * seq_len + pos) % capacity. When the ring wraps, the oldest
// (pass, pos) pair is evicted -- acceptable because later passes dominate the
// representation (see the convergence audit in pytorch/astra/monitor.py).
//
// Build: nvcc -O3 -arch=sm_90 -c kv_ring.cu
// Status: NOT compiled here (no CUDA toolchain on this host).

#include <cuda_fp16.h>
#include <cuda_runtime.h>

struct KvRing {
  half* k;          // (n_heads, capacity, d_head)
  half* v;          // (n_heads, capacity, d_head)
  int n_heads;
  int capacity;
  int d_head;
  unsigned long long cursor;  // next slot to write; monotonic
};

__global__ void kv_ring_write(
    half* __restrict__ buf,            // one of k/v: (n_heads, capacity, d_head)
    const half* __restrict__ src,      // (n_heads, seq_len, d_head) fresh K or V
    unsigned long long cursor,
    int n_heads, int seq_len, int capacity, int d_head) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_heads * seq_len * d_head;
  if (idx >= total) return;
  const int dh = idx % d_head;
  const int pos = (idx / d_head) % seq_len;
  const int h = idx / (d_head * seq_len);
  const unsigned long long slot = (cursor + pos) % (unsigned long long)capacity;
  buf[(h * capacity + slot) * d_head + dh] = src[idx];
}

__global__ void kv_ring_gather(
    const half* __restrict__ buf,      // (n_heads, capacity, d_head)
    half* __restrict__ dst,            // (n_heads, seq_len, d_head) assembled window
    unsigned long long cursor,         // cursor AFTER the corresponding writes
    int n_heads, int seq_len, int capacity, int d_head) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_heads * seq_len * d_head;
  if (idx >= total) return;
  const int dh = idx % d_head;
  const int pos = (idx / d_head) % seq_len;
  const int h = idx / (d_head * seq_len);
  // Most recent `seq_len` slots ending at cursor-1.
  const unsigned long long slot =
      (cursor - seq_len + pos) % (unsigned long long)capacity;
  dst[idx] = buf[(h * capacity + slot) * d_head + dh];
}

extern "C" {

void astra_kv_ring_write(KvRing* ring, const half* k_src, const half* v_src,
                         int seq_len, cudaStream_t stream) {
  const int total = ring->n_heads * seq_len * ring->d_head;
  const int block = 256;
  const int grid = (total + block - 1) / block;
  kv_ring_write<<<grid, block, 0, stream>>>(
      ring->k, k_src, ring->cursor, ring->n_heads, seq_len,
      ring->capacity, ring->d_head);
  kv_ring_write<<<grid, block, 0, stream>>>(
      ring->v, v_src, ring->cursor, ring->n_heads, seq_len,
      ring->capacity, ring->d_head);
  ring->cursor += seq_len;
}

void astra_kv_ring_gather(const KvRing* ring, half* k_dst, half* v_dst,
                          int seq_len, cudaStream_t stream) {
  const int total = ring->n_heads * seq_len * ring->d_head;
  const int block = 256;
  const int grid = (total + block - 1) / block;
  kv_ring_gather<<<grid, block, 0, stream>>>(
      ring->k, k_dst, ring->cursor, ring->n_heads, seq_len,
      ring->capacity, ring->d_head);
  kv_ring_gather<<<grid, block, 0, stream>>>(
      ring->v, v_dst, ring->cursor, ring->n_heads, seq_len,
      ring->capacity, ring->d_head);
}

}  // extern "C"
