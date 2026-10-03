// SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
// Copyright (C) 2026 SnapKitty Collective
//! Production serving layer for the Astra recurrent-depth transformer.
//!
//! Responsibilities:
//!   * batch incoming requests and schedule them on the GPU,
//!   * choose the recurrent loop count `r` per request (adaptive depth),
//!   * own the CUDA kernels in `../../cuda/` through FFI.
//!
//! Status: written, NOT compiled here (no Rust toolchain on this host).

use std::collections::VecDeque;
use std::os::raw::{c_float, c_int};
use std::ffi::c_void;

// ---------------------------------------------------------------------------
// FFI to the CUDA kernels (cuda/inject_norm.cu, cuda/kv_ring.cu)
//
// Feature-gated (`cuda`): the kernels are built separately with nvcc and
// linked in. With the feature off (default, incl. `cargo test`) this module
// is compiled out, so the pure-Rust scheduling logic is testable without a
// GPU toolchain.
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
mod ffi {
    use std::os::raw::{c_float, c_int};
    use std::ffi::c_void;

    pub type CudaStream = *mut c_void;

    #[repr(C)]
    pub struct KvRing {
        pub k: *mut u16, // half*
        pub v: *mut u16,
        pub n_heads: c_int,
        pub capacity: c_int,
        pub d_head: c_int,
        pub cursor: u64,
    }

    extern "C" {
        pub fn astra_inject_norm(
            x: *const u16, p: *const u16, w: *const u16, y: *mut u16,
            rows: c_int, dim: c_int, scale: c_float, eps: c_float,
            stream: CudaStream,
        );
        pub fn astra_kv_ring_write(ring: *mut KvRing, k_src: *const u16,
                                   v_src: *const u16, seq_len: c_int,
                                   stream: CudaStream);
        pub fn astra_kv_ring_gather(ring: *const KvRing, k_dst: *mut u16,
                                    v_dst: *mut u16, seq_len: c_int,
                                    stream: CudaStream);
    }

    /// Safe wrapper: one fused prompt re-injection + norm over `rows` states.
    pub fn inject_norm(x: &[u16], p: &[u16], w: &[u16], y: &mut [u16],
                       rows: usize, dim: usize, scale: f32, stream: CudaStream) {
        assert_eq!(x.len(), rows * dim);
        assert_eq!(p.len(), rows * dim);
        assert_eq!(y.len(), rows * dim);
        assert_eq!(w.len(), dim);
        unsafe {
            astra_inject_norm(x.as_ptr(), p.as_ptr(), w.as_ptr(), y.as_mut_ptr(),
                              rows as c_int, dim as c_int, scale, 1e-6, stream);
        }
    }
}

#[cfg(feature = "cuda")]
pub use ffi::{KvRing, CudaStream, inject_norm};

// ---------------------------------------------------------------------------
// Request batching with adaptive recurrent depth
// ---------------------------------------------------------------------------

/// A single inference request.
pub struct Request {
    pub id: u64,
    pub prompt_tokens: Vec<u32>,
    /// Difficulty estimate in [0, 1]; higher -> more recurrent passes.
    /// Produced upstream (classifier, heuristics, or user tier).
    pub difficulty: f32,
    pub max_tokens: usize,
}

/// Maps difficulty to a loop count. Depth is compute, not parameters:
/// a hard prompt gets more passes through the SAME shared block.
pub fn adaptive_loops(difficulty: f32, r_min: u32, r_max: u32) -> u32 {
    let d = difficulty.clamp(0.0, 1.0);
    r_min + ((r_max - r_min) as f32 * d).round() as u32
}

/// A micro-batch of requests padded to one sequence length, sharing one
/// prelude/coda launch. Requests keep their own `r`: the scheduler groups
/// by loop count so the recurrent core runs in lockstep per group.
pub struct Batch {
    pub loop_count: u32,
    pub requests: Vec<Request>,
}

pub struct Scheduler {
    queue: VecDeque<Request>,
    next_id: u64,
    r_min: u32,
    r_max: u32,
    max_batch: usize,
}

impl Scheduler {
    pub fn new(r_min: u32, r_max: u32, max_batch: usize) -> Self {
        Self { queue: VecDeque::new(), next_id: 0, r_min, r_max, max_batch }
    }

    pub fn submit(&mut self, prompt_tokens: Vec<u32>, difficulty: f32,
                  max_tokens: usize) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.queue.push_back(Request { id, prompt_tokens, difficulty, max_tokens });
        id
    }

    /// Drain one batch: pop up to `max_batch` requests with the same loop
    /// count (the most common one currently queued).
    pub fn next_batch(&mut self) -> Option<Batch> {
        if self.queue.is_empty() {
            return None;
        }
        // Count loop-count frequencies in the queue head window.
        let window = self.queue.iter().take(self.max_batch * 2);
        let mut freq = std::collections::HashMap::new();
        for req in window {
            let r = adaptive_loops(req.difficulty, self.r_min, self.r_max);
            *freq.entry(r).or_insert(0usize) += 1;
        }
        let target = freq.into_iter().max_by_key(|&(_, c)| c).map(|(r, _)| r)?;
        let mut batch = Batch { loop_count: target, requests: Vec::new() };
        let mut i = 0;
        while i < self.queue.len() && batch.requests.len() < self.max_batch {
            let r = adaptive_loops(self.queue[i].difficulty, self.r_min, self.r_max);
            if r == target {
                batch.requests.push(self.queue.remove(i).unwrap());
            } else {
                i += 1;
            }
        }
        Some(batch)
    }

    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }
}

// ---------------------------------------------------------------------------
// Python API sketch (requires pyo3; not wired here)
// ---------------------------------------------------------------------------
//
// The production integration wraps `Scheduler` for Python:
//
// ```python
// import astra_serve
// srv = astra_serve.Scheduler(r_min=2, r_max=16, max_batch=32)
// rid = srv.submit(prompt_tokens=[...], difficulty=0.8, max_tokens=128)
// batch = srv.next_batch()  # -> {"loop_count": 13, "request_ids": [...]}
// ```
//
// Wiring: add `pyo3 = { version = "0.22", features = ["extension-module"] }`
// and `#[pymodule]` wrappers around Scheduler/Batch. Deliberately left
// unwritten until the crate compiles on a Rust toolchain.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_loops_span() {
        assert_eq!(adaptive_loops(0.0, 2, 16), 2);
        assert_eq!(adaptive_loops(1.0, 2, 16), 16);
        assert_eq!(adaptive_loops(0.5, 2, 16), 9);
    }

    #[test]
    fn scheduler_groups_by_loop_count() {
        let mut s = Scheduler::new(2, 16, 8);
        for _ in 0..4 {
            s.submit(vec![1, 2, 3], 0.0, 16); // r = 2
        }
        for _ in 0..2 {
            s.submit(vec![1, 2, 3], 1.0, 16); // r = 16
        }
        let b = s.next_batch().unwrap();
        assert_eq!(b.loop_count, 2);
        assert_eq!(b.requests.len(), 4);
        assert_eq!(s.queue_len(), 2);
    }
}
