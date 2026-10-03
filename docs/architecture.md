# SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
# Copyright (C) 2026 SnapKitty Collective
# Astra architecture: recurrent depth + latent reasoning

## The core idea

Instead of scaling depth by stacking dozens of uniquely parameterized layers
(a traditional 32-layer model), Astra reuses **one shared Transformer block**
multiple times. Depth becomes *compute*, not *parameters*.

## Three-phase flow

```
tokens
  │
  ▼
┌──────────┐
│ PRELUDE  │  n_pre uniquely-parameterized layers embed the prompt
└────┬─────┘
     │ h₀  (prompt representation — saved for re-injection)
     ▼
┌──────────┐   ┌─────────────────────────────────────────┐
│ RECURRENT│   │ pass k:  h = h + P(h₀)   (re-inject)    │
│ CORE     │──▶│          h = Block(h)     (SHARED block) │
│ (× r)    │   │ no tokens are generated inside the loop │
└────┬─────┘   └─────────────────────────────────────────┘
     │ hᵣ  (refined continuous hidden state)
     ▼
┌──────────┐
│  CODA    │  RMSNorm + unembedding head → visible tokens
└──────────┘
```

1. **Prelude** — initial layers process and embed the incoming prompt.
2. **Recurrent core (the loop)** — the shared block takes the evolving hidden
   state and loops over it `r` times. The prompt representation `h₀` is
   re-introduced every pass (residual add or learned gate) so the loop cannot
   drift from the initial context.
3. **Coda** — final layers map the refined hidden state to visible tokens.

## Hidden chain-of-thought

Earlier reasoning models verbalized their thinking: thousands of visible
tokens before the answer. Here the loop refines representations **entirely in
latent space** — continuous vector activations, no intermediate tokens.

Consequences:

- **Computational depth vs parameter count.** A compact model gets deep logical
  reasoning (math, coding, strategy) while token generation stays fast and
  conversational.
- **Adaptive compute.** Easy prompt → small `r`; hard prompt → large `r`. The
  serving layer (`rust/astra-serve`) maps a difficulty estimate to a loop
  count per request, batching requests that share `r`.
- **Safety/monitoring challenge.** The reasoning is silent. Auditing *how* a
  conclusion was reached needs internal probes, not transcript reading. The
  `pytorch/astra/monitor.py` tools record per-pass delta norms (is the loop
  settling?), prompt drift (is re-injection holding?), and per-pass coda
  readouts (when did the model lock onto its answer?). These are behavioral
  probes over activations — they make the loop legible, not transparent.

## The three implementation layers

| Layer | Language | Role | Location |
|---|---|---|---|
| Research & training | Python (PyTorch) | Prototype the loop, loss functions, gradient updates through `r` passes | `pytorch/` |
| Performance | C++/CUDA | Custom kernels: the loop step is memory-bandwidth bound (same block re-reads state `r` times), so prompt re-injection + norm are fused and K/V rides a ring buffer across passes | `cuda/` |
| Production serving | Rust | Request batching, adaptive-`r` scheduling, FFI to the CUDA kernels, Python API | `rust/astra-serve/` |

## What this repo proves (executed)

- The looped block is **one** shared module: `n_pre` prelude blocks + 1 loop
  block, parameter count independent of `r` (unit-tested).
- Prompt re-injection measurably changes the loop trajectory (unit-tested).
- **Depth-vs-parameters probe**: one shared block trained with per-pass
  supervision to rotate bits right by one; looping it `k` times yields
  per-pass exact-match ≈ `0.492^k` — the signature of a correctly learned
  iterated function with independently compounding errors. Same 107,200
  weights throughout; see README for the full table.
- **Causal masking sets a provable ceiling**: position 0 of a right-rotation
  needs the last bit, which the causal mask hides — capping the demo at
  0.9375 bit accuracy / 0.5 exact-match. The model sits on that ceiling; the
  residual error is architectural, not under-training.
- The loop does **not** extrapolate past its training horizon (`r_train=4`):
  passes 5–8 collapse toward chance. Stated, not hidden.

## Hard-won implementation notes

- The model originally had **no positional embeddings**, making it
  permutation-equivariant — incapable, in principle, of rotation or
  neighbor-based denoising. Caught by experiment, fixed with learned
  position embeddings.
- The iterated task must respect causality: rotate-*left* needs attending to
  the future, which the causal mask forbids. Rotate-*right* is the
  causal-compatible direction. Task and mask must agree.
- Two negative results are kept in `pytorch/astra/tasks.py` with notes:
  `ParityTask` (stalls at chance through a shared loop) and `RefineTask`
  (i.i.d. bits are un-denoisable in principle — the Bayes-optimal predictor
  is the identity). `ChainRefineTask` trains to 0.824–0.826 bit accuracy,
  flat across `r` — a weak denoiser that extra passes don't help.
  Measured result is mixed: the loop learns an approximate single rotation
  (pass-1 bit accuracy 0.937) but errors compound per pass and extrapolation
  past `r_train` collapses toward chance; the audit probes (growing deltas,
  prompt drift → 1.0) confirm the instability. Kept as a negative result.

## What this repo does NOT claim

- The CUDA kernels are **not compiled or benchmarked here** (no GPU on the
  build host). No performance numbers are stated for them.
- The Rust crate is **not compiled here** (no Rust toolchain). Its unit tests
  are written, not run.
- The parity demo is a synthetic probe of the depth mechanism, not evidence
  about reasoning quality on real tasks.
