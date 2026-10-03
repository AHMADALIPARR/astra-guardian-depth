# SPDX-License-Identifier: AGPL-3.0-or-later
"""Looped transformer core: prelude -> recurrent shared block (r passes) -> coda.

Architecture (three-phase flow):
  1. Prelude: token embedding + `n_pre` uniquely-parameterized transformer layers
     that embed the incoming prompt.
  2. Recurrent core: ONE shared TransformerBlock applied `r` times to the evolving
     hidden state. The prompt representation is re-injected every pass
     (residual or gated) so the loop cannot drift from the initial context.
  3. Coda: final RMSNorm + unembedding head mapping the refined hidden state to
     visible tokens.

The loop performs multi-step computation entirely in latent space (continuous
vector activations) -- hidden chain-of-thought with no intermediate tokens.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoopTrace:
    """Per-pass audit record of the silent (latent) reasoning loop."""

    passes: int
    delta_norms: list = field(default_factory=list)   # ||h_{k+1} - h_k|| per pass
    prompt_drift: list = field(default_factory=list)  # cosine distance from prompt rep
    readouts: list = field(default_factory=list)      # per-pass argmax tokens (coda readout)
    entropies: list = field(default_factory=list)     # per-pass output entropy
    pass_logits: list = field(default_factory=list)   # per-pass coda logits (for deep supervision)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split(t_: torch.Tensor) -> torch.Tensor:
            return t_.view(b, t, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = split(q), split(k), split(v)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(b, t, d)
        return self.out(y)


class TransformerBlock(nn.Module):
    """One pre-norm transformer block (attention + MLP)."""

    def __init__(self, dim: int, n_heads: int, ffn_mult: int = 4):
        super().__init__()
        self.n1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads)
        self.n2 = RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, ffn_mult * dim, bias=False),
            nn.GELU(),
            nn.Linear(ffn_mult * dim, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x


class LoopedTransformer(nn.Module):
    """Prelude -> shared looped block (r passes, prompt re-injected) -> coda."""

    def __init__(
        self,
        vocab_size: int,
        dim: int = 128,
        n_heads: int = 4,
        n_pre: int = 2,
        ffn_mult: int = 4,
        inject: str = "residual",
        max_seq_len: int = 64,
    ):
        super().__init__()
        assert inject in ("residual", "gated")
        self.inject = inject
        self.dim = dim

        self.embed = nn.Embedding(vocab_size, dim)
        # Learned positional embeddings: without these the model is
        # permutation-equivariant and cannot represent positional functions
        # (rotation, neighbor-based denoising, ...) at all.
        self.pos_embed = nn.Embedding(max_seq_len, dim)
        self.prelude = nn.Sequential(
            *[TransformerBlock(dim, n_heads, ffn_mult) for _ in range(n_pre)]
        )
        # THE recurrent core: a single shared block, reused r times.
        self.loop = TransformerBlock(dim, n_heads, ffn_mult)
        self.prompt_proj = nn.Linear(dim, dim, bias=False)
        if inject == "gated":
            self.inject_gate = nn.Linear(2 * dim, dim, bias=False)

        self.coda_norm = RMSNorm(dim)
        self.coda_head = nn.Linear(dim, vocab_size, bias=False)

    # -- coda (also used by the monitor for per-pass readouts) -----------------
    def coda(self, h: torch.Tensor) -> torch.Tensor:
        return self.coda_head(self.coda_norm(h))

    def _inject_prompt(self, h: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        if self.inject == "residual":
            return h + self.prompt_proj(prompt)
        gate = torch.sigmoid(self.inject_gate(torch.cat([h, prompt], dim=-1)))
        return gate * h + (1.0 - gate) * self.prompt_proj(prompt)

    def forward(self, ids: torch.Tensor, r: int = 4, record: bool = False):
        """Run the three-phase flow.

        Args:
            ids: (B, T) token ids.
            r: number of recurrent passes through the shared block.
            record: if True, also return a LoopTrace auditing the latent loop.
        """
        assert r >= 1, "need at least one recurrent pass"
        b, t = ids.shape
        assert t <= self.pos_embed.num_embeddings, "sequence longer than max_seq_len"
        h = self.embed(ids) + self.pos_embed(
            torch.arange(t, device=ids.device).unsqueeze(0).expand(b, t)
        )
        h = self.prelude(h)
        prompt = h  # prompt representation, re-injected every pass

        trace = LoopTrace(passes=r) if record else None
        for _ in range(r):
            h_prev = h
            h = self._inject_prompt(h, prompt)
            h = self.loop(h)
            if record:
                delta = (h - h_prev).norm(dim=-1).mean().item()
                cos = F.cosine_similarity(h, prompt, dim=-1).mean().item()
                logits = self.coda(h)
                probs = F.softmax(logits[:, -1, :], dim=-1)
                ent = -(probs * (probs + 1e-12).log()).sum(dim=-1).mean().item()
                trace.delta_norms.append(delta)
                trace.prompt_drift.append(1.0 - cos)
                trace.entropies.append(ent)
                trace.readouts.append(logits.argmax(dim=-1).tolist())
                trace.pass_logits.append(logits)

        logits = self.coda(h)
        return (logits, trace) if record else logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def looped_block_instances(self) -> int:
        """How many TransformerBlock modules exist (n_pre prelude + 1 shared)."""
        return sum(1 for m in self.modules() if isinstance(m, TransformerBlock))
