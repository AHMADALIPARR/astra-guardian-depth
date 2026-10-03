// SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
// Copyright (C) 2026 SnapKitty Collective
// C11 tensor with manual autograd for Mamba SSM training.
// No Python. No PyTorch. Just C11.

#ifndef ASTRA_TENSOR_H
#define ASTRA_TENSOR_H

#include <stddef.h>

typedef struct Tensor Tensor;
typedef void (*BackwardFn)(Tensor *t);

struct Tensor {
    float *data;      // forward values
    float *grad;      // gradients (NULL if not requires_grad)
    int *shape;       // dimensions
    int ndim;
    size_t numel;
    int requires_grad;

    // autograd graph
    BackwardFn backward_fn;
    Tensor *parents[4];
    int n_parents;
    // saved tensors for backward
    void *ctx;
};

Tensor *tensor_new(int ndim, const int *shape, int requires_grad);
void tensor_free(Tensor *t);
void tensor_zero_grad(Tensor *t);
size_t tensor_numel(int ndim, const int *shape);

// Ops (all create graph nodes if any parent requires grad)
Tensor *tensor_matmul(Tensor *a, Tensor *b);       // 2D matmul
Tensor *tensor_add(Tensor *a, Tensor *b);          // broadcast add (same shape)
Tensor *tensor_mul(Tensor *a, Tensor *b);         // elementwise
Tensor *tensor_silu(Tensor *a);
Tensor *tensor_rmsnorm(Tensor *a, Tensor *weight, float eps);
Tensor *tensor_embedding(Tensor *weight, const int *ids, int n); // (n, dim)

// Backward: call on scalar loss
void tensor_backward(Tensor *loss);

#endif
