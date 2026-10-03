# SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
# Copyright (C) 2026 SnapKitty Collective
"""Astra recurrent-depth transformer: prelude -> looped core (r passes) -> coda.

The model reuses ONE shared Transformer block for `r` recurrent passes over the
evolving hidden state. The prompt representation is re-injected every pass so
the loop cannot drift from the initial context. Reasoning happens in latent
space: no intermediate tokens are generated during the loop.
"""

from .model import (
    RMSNorm,
    CausalSelfAttention,
    TransformerBlock,
    LoopedTransformer,
    LoopTrace,
)

__all__ = [
    "RMSNorm",
    "CausalSelfAttention",
    "TransformerBlock",
    "LoopedTransformer",
    "LoopTrace",
]
