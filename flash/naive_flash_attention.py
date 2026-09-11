"""
Phase 8 — Naive FlashAttention (pure PyTorch, no custom kernels).

Folds Phase 6's online softmax and Phase 7's tiling into a single fused
pass: for each query block, stream over key/value blocks exactly once,
maintaining running (max, sum, weighted-output-accumulator) statistics and
rescaling the accumulator in place as the running max updates. This is the
same algorithm kernels/flash_attention_triton.py (Phase 9) implements as an
actual fused GPU kernel — this version proves the algorithm is correct
using only plain PyTorch ops, before paying the complexity cost of Triton.

Run:
    python naive_flash_attention.py
Expect:
    correctness check vs standard attention, then a latency + peak-memory
    benchmark table across sequence lengths.
"""

import math
import time

import torch


def flash_attention_naive(q, k, v, block_size_q=128, block_size_kv=128):
    """
    q, k, v: (batch, num_heads, seq_len, head_dim)
    Single-pass blocked attention with online-softmax rescaling — the exact
    algorithm FlashAttention describes, implemented with plain PyTorch ops
    (no fused kernel, so intermediate (block_size_q x block_size_kv) score
    tiles ARE materialized as real tensors — but never the full N x N
    matrix).
    """
    batch, num_heads, seq_len, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)
    device, dtype = q.device, q.dtype

    out = torch.empty_like(q)

    for q_start in range(0, seq_len, block_size_q):
        q_end = min(q_start + block_size_q, seq_len)
        q_block = q[:, :, q_start:q_end, :]
        bq = q_end - q_start

        m_i = torch.full((batch, num_heads, bq, 1), float("-inf"), device=device, dtype=dtype)
        l_i = torch.zeros((batch, num_heads, bq, 1), device=device, dtype=dtype)
        acc = torch.zeros((batch, num_heads, bq, head_dim), device=device, dtype=dtype)

        for k_start in range(0, seq_len, block_size_kv):
            k_end = min(k_start + block_size_kv, seq_len)
            k_block = k[:, :, k_start:k_end, :]
            v_block = v[:, :, k_start:k_end, :]

            scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale  # (B,H,bq,bk)

            block_max = scores.max(dim=-1, keepdim=True).values
            new_m = torch.maximum(m_i, block_max)

            alpha = torch.exp(m_i - new_m)          # rescales OLD accumulator + sum
            p = torch.exp(scores - new_m)            # unnormalized probs for THIS block

            l_i = l_i * alpha + p.sum(dim=-1, keepdim=True)
            acc = acc * alpha + torch.matmul(p, v_block)
            m_i = new_m

        out[:, :, q_start:q_end, :] = acc / l_i

    return out


def _reference_attention(q, k, v):
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def _benchmark(fn, q, k, v, num_iters=10, warmup=3):
    device = q.device.type
    for _ in range(warmup):
        fn(q, k, v)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(num_iters):
        fn(q, k, v)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_iters
    peak_mem = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    return elapsed, peak_mem


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    batch, num_heads, head_dim = 2, 8, 64
    q = torch.randn(batch, num_heads, 512, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, num_heads, 512, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, num_heads, 512, head_dim, device=device, dtype=dtype)

    out = flash_attention_naive(q, k, v)
    ref = _reference_attention(q, k, v)
    diff = (out - ref).abs().max().item()
    print(f"max abs diff vs reference attention: {diff:.8f}")
    assert diff < 1e-4, "flash_attention_naive does not match reference"
    print("OK: naive FlashAttention matches standard attention exactly.\n")

    print(f"{'seq_len':>8} | {'standard (ms)':>14} | {'flash (ms)':>11} | {'standard (MB)':>14} | {'flash (MB)':>11}")
    for seq_len in (256, 512, 1024, 2048, 4096):
        q = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)

        try:
            std_time, std_mem = _benchmark(_reference_attention, q, k, v)
            std_time_s, std_mem_s = f"{std_time*1000:.2f}", f"{std_mem/1e6:.1f}"
        except torch.cuda.OutOfMemoryError:
            std_time_s, std_mem_s = "OOM", "OOM"
            torch.cuda.empty_cache()

        flash_time, flash_mem = _benchmark(flash_attention_naive, q, k, v)

        print(f"{seq_len:>8} | {std_time_s:>14} | {flash_time*1000:>11.2f} | {std_mem_s:>14} | {flash_mem/1e6:>11.1f}")
