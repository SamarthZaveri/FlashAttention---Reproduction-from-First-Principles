"""
Phase 1 — Self-Attention from scratch.

Implements Q/K/V projections, scaled dot-product scores, numerically stable
softmax, and the weighted sum, then verifies the output matches PyTorch's
own scaled_dot_product_attention (SDPA) reference exactly (within fp
tolerance).

Run:
    python self_attention.py
Expect:
    "max abs diff vs torch SDPA: <tiny number>"  followed by an OK assertion.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def stable_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax via the x - max(x) trick, built from primitives
    (not torch.softmax) so Phase 0's numerical-stability lesson is load-bearing here."""
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = 1.0 / math.sqrt(embed_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        """x: (batch, seq_len, embed_dim) -> (batch, seq_len, embed_dim)"""
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, S, S)

        if causal:
            seq_len = x.shape[1]
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        probs = stable_softmax(scores, dim=-1)
        out = torch.matmul(probs, v)
        return out


def _reference_sdpa(x, q_proj, k_proj, v_proj, causal=False):
    """Reference computed with the SAME weights, via torch's built-in SDPA kernel."""
    q, k, v = q_proj(x), k_proj(x), v_proj(x)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch, seq_len, embed_dim = 4, 128, 64
    x = torch.randn(batch, seq_len, embed_dim, device=device)

    attn = SelfAttention(embed_dim).to(device)

    for causal in (False, True):
        out = attn(x, causal=causal)
        ref = _reference_sdpa(x, attn.q_proj, attn.k_proj, attn.v_proj, causal=causal)
        diff = (out - ref).abs().max().item()
        print(f"[causal={causal}] max abs diff vs torch SDPA: {diff:.8f}")
        assert diff < 1e-4, "SelfAttention does not match torch SDPA reference"

    print("OK: from-scratch SelfAttention matches torch.nn.functional.scaled_dot_product_attention.")
