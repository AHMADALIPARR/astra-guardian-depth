<!-- SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0 -->
<!-- Copyright (C) 2026 SnapKitty Collective -->
# C11 looped-transformer training backend

Complete forward and backward for the parameter layout committed at `5ad2721`:
one prelude, one shared block applied four times, residual h0 reinjection,
four-head causal attention, RMSNorm, SwiGLU, positional/token embeddings, and
the output head. The default configuration has **95,936 parameters**. It is not
the older 107,200-parameter PyTorch GELU configuration.

This directory is additive. The original `c11/looped` files remain untouched.
Use this directory's header and implementation together: parameter structs
match the original, but the activation cache intentionally differs.

```sh
make test
make sanitize
make train-looped
./train-looped 500 32 100
```

The training driver balances k=1..4 in each batch, averages accumulated
gradients before an Adam update, uses learning rate 0.003, and reports examples
seen and gradient norm. Its final evaluation enumerates all 256 bit strings
for every k and reports exact, bit, visible, hidden, and per-position accuracy.
Training and evaluation do not share or advance the same random stream.
The default 500 updates at batch 32 expose the model to 16,000 examples.
This driver has been smoke-tested; no multi-seed convergence or throughput
comparison is claimed for the transformer yet.

## Mirrored tensors

`LoopedGrad` is an alias of `LoopedModel`; each named parameter tensor has an
identically shaped gradient tensor. `BlockTensorGrad` similarly mirrors
`BlockTensors`, the intermediate activation layout. Attention gradients retain
separate head, query-position, and key-position axes.

The forward cache saves a distinct activation tape for each execution of the
shared block. Backward visits these tapes in reverse and adds parameter
gradients to the one `g.shared` object. For `hin = hprev + W*h0`, it sends
the gradient to hprev, accumulates `dW += dhin outer h0`, and adds
`dh0 += transpose(W)*dhin` across all recurrent passes. The prelude receives
the sum of that reinjection gradient and the normal recurrent-path gradient.

Softmax backward uses `ds = p * (dp - sum(p * dp))`; the query/key gradients
include the `1/sqrt(head_width)` scaling. Value/key gradients accumulate from
all queries that use them. Masked future positions receive no gradient.

## API and ownership

Heap-allocate `LoopedModel`, `LoopedGrad`, and `LoopedCache`; the cache is large.
Parameters, public logits, and accumulated gradients are float. Activations
and internal arithmetic are double, enabling informative numerical checks.
Call forward, loss, zero-grad, backward, then update weights. Check statuses.
The backward call adds to existing gradients, supporting batches and multiple
loss contributions. Do not change model weights between forward and backward.
Use separate caches and gradient objects for concurrent calls. No static
mutable backward workspace exists. `free(cache)` frees the complete tape.

`looped_forward_r` supports 1 through `LT_R_LOOP` passes; the default wrapper
uses `LT_R_LOOP`. Sequence length is 1 through `LT_MAX_T`. Cross-entropy ignores
target -1; an entirely ignored sequence has zero loss and zero dlogits.
NULL cache is allowed for inference. The original one-sided random initializer
has been replaced with symmetric fan-in-scaled weights, unit norm weights,
zero biases, and a small reinjection matrix.

## Executed validation

- 1,251 finite-difference checks on a 404-parameter model, including every
  parameter at (T=5, r=1), (T=5, r=4), and (T=1, r=4).
- 157 checks on the default 95,936-parameter model, including nonzero Q/K/V,
  attention output, SwiGLU, normalization, embedding, reinjection, and head
  gradients. Key-bias gradients are separately checked to be zero, as expected
  from softmax's invariance to constant row shifts.
- Gradient accumulation, causal probability normalization, exact zero masked
  probabilities/future positional gradients, prefix invariance, cache ownership,
  invalid inputs, inference without a cache, and maximum sequence length.
- Numerical cross-entropy derivatives and ignored targets.
- Fixed-batch SGD smoke test: default-model loss 2.282493 to 0.045344 in 70
  updates. This demonstrates learning on that batch, not generalization.
- ASan and UBSan passed. Leak detection is disabled by default because the
  managed executor's ptrace environment blocks LeakSanitizer; on another host
  enable it with `ASAN_OPTIONS=detect_leaks=1 make sanitize`.

Inside the original repository, `make parity` compares against its untouched
forward source at `../looped/looped.c`, using identical weights and inputs.
Nine cases (three seeds, lengths 1/5/10) passed with maximum logit difference
6.56e-7. The standalone archive's normal tests do not require that reference.
