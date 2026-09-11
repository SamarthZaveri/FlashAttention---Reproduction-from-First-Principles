"""
Phase 9b — Fused FlashAttention forward kernel in Triton.

Non-causal, single-head-per-program-id-1 forward pass. Implements the same
algorithm as flash/naive_flash_attention.py (Phase 8) but fused into one
GPU kernel: for each query block, stream over key/value blocks, maintain a
running max and running (unnormalized) sum for online softmax, and
accumulate the output — the full (seq_len x seq_len) score matrix is never
materialized in HBM, only a (BLOCK_M x BLOCK_N) tile at a time in registers.

Requires (Windows): pip install triton-windows torch
Requires (Linux/WSL2): pip install triton torch

Run:
    python flash_attention_triton.py
Expect:
    "max abs diff vs torch reference attention: <tiny number>"
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attn_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    seq_len, head_dim,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Grid: (num_query_blocks, batch * num_heads)
    start_m = tl.program_id(0)
    bh = tl.program_id(1)

    q_ptr += bh * stride_qh
    k_ptr += bh * stride_kh
    v_ptr += bh * stride_vh
    o_ptr += bh * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Online softmax running state.
    m_i = tl.full((BLOCK_M,), value=float("-inf"), dtype=tl.float32)  # running max
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)                       # running sum
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)               # running output

    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = k_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k_mask = (offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # Scores for this tile: (BLOCK_M, BLOCK_N)
        scores = tl.dot(q, tl.trans(k)) * sm_scale
        score_mask = (offs_m[:, None] < seq_len) & (offs_n[None, :] < seq_len)
        scores = tl.where(score_mask, scores, float("-inf"))

        # --- online softmax update ---
        m_ij = tl.max(scores, axis=1)                  # block-local max
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_new)                     # rescale factor for old accumulator
        p = tl.exp(scores - m_new[:, None])              # unnormalized probs for this tile

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = v_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v_mask = (offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]

    o_ptrs = o_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    o_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)
    tl.store(o_ptrs, acc.to(tl.float16), mask=o_mask)


def flash_attention_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    q, k, v: (batch, num_heads, seq_len, head_dim), fp16, contiguous, CUDA.
    Non-causal forward pass. Returns (batch, num_heads, seq_len, head_dim).
    """
    batch, num_heads, seq_len, head_dim = q.shape
    assert k.shape == q.shape and v.shape == q.shape
    assert head_dim <= 128, "increase BLOCK_D / add head_dim tiling for larger dims"

    o = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(head_dim)

    BLOCK_M, BLOCK_N = 64, 64
    BLOCK_D = triton.next_power_of_2(head_dim)

    grid = (triton.cdiv(seq_len, BLOCK_M), batch * num_heads)

    _flash_attn_fwd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        seq_len, head_dim,
        sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )
    return o


def _reference_attention(q, k, v):
    """Standard (non-fused) attention — Phase 1's reference, for correctness checks."""
    sm_scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(probs, v)


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda"
    batch, num_heads, seq_len, head_dim = 2, 4, 512, 64

    q = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)

    out_triton = flash_attention_triton(q, k, v)
    out_ref = _reference_attention(q, k, v)

    diff = (out_triton.float() - out_ref.float()).abs().max().item()
    print(f"max abs diff vs torch reference attention: {diff:.6f}")
    assert diff < 5e-2, "mismatch too large — check online softmax rescaling"
    print("OK: Triton FlashAttention kernel matches reference attention within fp16 tolerance.")

    # Quick memory sanity check vs standard attention: standard materializes
    # an (seq_len x seq_len) matrix per (batch, head); flash never does.
    torch.cuda.reset_peak_memory_stats()
    _ = flash_attention_triton(q, k, v)
    torch.cuda.synchronize()
    flash_peak = torch.cuda.max_memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    _ = _reference_attention(q, k, v)
    torch.cuda.synchronize()
    ref_peak = torch.cuda.max_memory_allocated()

    print(f"peak mem — flash: {flash_peak/1e6:.1f} MB, standard: {ref_peak/1e6:.1f} MB")
