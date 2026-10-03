// SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
// Copyright (C) 2026 SnapKitty Collective
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
// Optimizations:
//   * half2 vectorized copies: 2 elements per transaction (d_head is even in
//     practice; asserted in the launcher).
//   * One block per head: the head index comes from blockIdx, eliminating a
//     division per thread; the remaining pos/dh decode is a single div per
//     half2 element, amortized over the vectorized copy.
//   * const __restrict__ pointers let the compiler emit LDG (read-only cache)
//     loads for src/buf.
//
// Build: nvcc -O3 -arch=sm_90 -c kv_ring.cu
// Status: NOT compiled here (no CUDA toolchain on this host).

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cassert>

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
    int seq_len, int capacity, int d_head) {
  // One block per head; threads cooperatively copy half2 vectors.
  const int h = blockIdx.x;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int d_head2 = d_head >> 1;
  const int total = seq_len * d_head2;

  const half2* __restrict__ src2 =
      reinterpret_cast<const half2*>(src) + (size_t)h * seq_len * d_head2;
  half2* __restrict__ buf2 = reinterpret_cast<half2*>(buf);

  for (int i = tid; i < total; i += nthreads) {
    const int pos = i / d_head2;
    const int dh2 = i - pos * d_head2;
    const unsigned long long slot = (cursor + (unsigned long long)pos)
                                  % (unsigned long long)capacity;
    buf2[((size_t)h * capacity + slot) * d_head2 + dh2] = src2[i];
  }
}

__global__ void kv_ring_gather(
    const half* __restrict__ buf,      // (n_heads, capacity, d_head)
    half* __restrict__ dst,            // (n_heads, seq_len, d_head) assembled window
    unsigned long long cursor,         // cursor AFTER the corresponding writes
    int seq_len, int capacity, int d_head) {
  // One block per head; threads cooperatively copy half2 vectors.
  const int h = blockIdx.x;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int d_head2 = d_head >> 1;
  const int total = seq_len * d_head2;

  const half2* __restrict__ buf2 = reinterpret_cast<const half2*>(buf);
  half2* __restrict__ dst2 =
      reinterpret_cast<half2*>(dst) + (size_t)h * seq_len * d_head2;

  for (int i = tid; i < total; i += nthreads) {
    const int pos = i / d_head2;
    const int dh2 = i - pos * d_head2;
    // Most recent `seq_len` slots ending at cursor-1.
    // Precondition: cursor >= seq_len (gather follows write).
    const unsigned long long slot =
        (cursor - (unsigned long long)seq_len + (unsigned long long)pos)
        % (unsigned long long)capacity;
    dst2[i] = buf2[((size_t)h * capacity + slot) * d_head2 + dh2];
  }
}

extern "C" {

void astra_kv_ring_write(KvRing* ring, const half* k_src, const half* v_src,
                         int seq_len, cudaStream_t stream) {
  assert((ring->d_head & 1) == 0 && "d_head must be even for half2 vectorization");
  assert(ring->capacity >= seq_len &&
         "capacity must cover seq_len or the ring overwrites live slots");
  assert(seq_len > 0 && "seq_len must be positive");
  const int block = 256;
  kv_ring_write<<<ring->n_heads, block, 0, stream>>>(
      ring->k, k_src, ring->cursor, seq_len, ring->capacity, ring->d_head);
  kv_ring_write<<<ring->n_heads, block, 0, stream>>>(
      ring->v, v_src, ring->cursor, seq_len, ring->capacity, ring->d_head);
  ring->cursor += seq_len;
}

void astra_kv_ring_gather(const KvRing* ring, half* k_dst, half* v_dst,
                          int seq_len, cudaStream_t stream) {
  assert((ring->d_head & 1) == 0 && "d_head must be even for half2 vectorization");
  assert(ring->capacity >= seq_len &&
         "capacity must cover seq_len or the ring overwrites live slots");
  assert(ring->cursor >= (unsigned long long)seq_len &&
         "gather requires cursor >= seq_len (call after the matching write)");
  assert(seq_len > 0 && "seq_len must be positive");
  const int block = 256;
  kv_ring_gather<<<ring->n_heads, block, 0, stream>>>(
      ring->k, k_dst, ring->cursor, seq_len, ring->capacity, ring->d_head);
  kv_ring_gather<<<ring->n_heads, block, 0, stream>>>(
      ring->v, v_dst, ring->cursor, seq_len, ring->capacity, ring->d_head);
}

}  // extern "C"
