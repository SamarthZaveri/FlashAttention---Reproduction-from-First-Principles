"""
Phase 2 — Multi-Head Attention from scratch.

Implements head splitting/concatenation on top of Phase 1's primitives and
verifies output matches torch.nn.MultiheadAttention exactly, by copying
nn.MultiheadAttention's packed in_proj weights into our separate per-head
projections so both sides run identical math on identical weights.

Run:
    python multihead_attention.py
Expect:
    "max abs diff vs nn.MultiheadAttention: <tiny number>"
"""

import math

import torch
import torch.nn as nn

from self_attention import stable_softmax


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, S, E) -> (B, num_heads, S, head_dim)
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, num_heads, S, head_dim) -> (B, S, E)
        batch, num_heads, seq_len, head_dim = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, num_heads * head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, S, S)
        probs = stable_softmax(scores, dim=-1)
        out = torch.matmul(probs, v)  # (B, H, S, head_dim)

        out = self._merge_heads(out)
        return self.out_proj(out)

    def load_from_torch_mha(self, torch_mha: nn.MultiheadAttention):
        """Copy weights out of a torch.nn.MultiheadAttention (batch_first=True,
        packed in_proj_weight of shape (3E, E)) into our separate Q/K/V/out
        projections so both modules compute on identical parameters."""
        E = self.embed_dim
        w = torch_mha.in_proj_weight.data
        b = torch_mha.in_proj_bias.data

        self.q_proj.weight.data.copy_(w[0:E])
        self.k_proj.weight.data.copy_(w[E:2 * E])
        self.v_proj.weight.data.copy_(w[2 * E:3 * E])
        self.q_proj.bias.data.copy_(b[0:E])
        self.k_proj.bias.data.copy_(b[E:2 * E])
        self.v_proj.bias.data.copy_(b[2 * E:3 * E])

        self.out_proj.weight.data.copy_(torch_mha.out_proj.weight.data)
        self.out_proj.bias.data.copy_(torch_mha.out_proj.bias.data)


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch, seq_len, embed_dim, num_heads = 4, 128, 64, 8
    x = torch.randn(batch, seq_len, embed_dim, device=device)

    torch_mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, bias=True).to(device)
    torch_mha.eval()

    ours = MultiHeadAttention(embed_dim, num_heads).to(device)
    ours.eval()
    ours.load_from_torch_mha(torch_mha)

    with torch.no_grad():
        ref_out, _ = torch_mha(x, x, x, need_weights=False)
        out = ours(x)

    diff = (out - ref_out).abs().max().item()
    print(f"max abs diff vs nn.MultiheadAttention: {diff:.8f}")
    assert diff < 1e-4, "MultiHeadAttention does not match torch.nn.MultiheadAttention"
    print("OK: from-scratch MultiHeadAttention matches torch.nn.MultiheadAttention.")
