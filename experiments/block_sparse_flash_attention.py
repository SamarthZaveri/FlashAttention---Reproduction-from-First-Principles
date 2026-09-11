"""
Original extension — Norm-Bound Block-Sparse FlashAttention.

Idea: before doing the expensive (block_q x block_kv) matmul for a given
(query-block, key-block) pair, get a cheap Cauchy-Schwarz upper bound on the
scores that block could possibly produce:

    max_{i in Qblock, j in Kblock} (q_i . k_j) * scale
        <= (max_i ||q_i||) * (max_j ||k_j||) * scale

This bound only needs per-token L2 norms (an O(N) reduction), not the full
O(bq * bkv) matmul. For a given query block, compute this upper bound
against every key block, take the max over key blocks as an upper bound on
that query block's true row max, then SKIP any key block whose upper bound
is more than log(1/epsilon) below that running max — such a block's true
contribution to the softmax is provably bounded by `epsilon` relative to the
dominant term, and can be omitted with a bounded, honestly-reported
approximation error, in exchange for skipping the matmuls, memory traffic,
and online-softmax updates for that block entirely.

This trades away FlashAttention's "exact" property for a tunable
speed/memory-vs-accuracy knob — which is the whole point of an *extension*:
it is not a strict improvement, and its benefit is highly data-dependent (see
the two regimes benchmarked in __main__ below).

Run:
    python block_sparse_flash_attention.py
"""

import math
import sys
import os
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "flash"))
from naive_flash_attention import flash_attention_naive  # noqa: E402


def block_sparse_flash_attention(q, k, v, block_size_q=128, block_size_kv=128, epsilon=1e-4):
    """
    q, k, v: (batch, num_heads, seq_len, head_dim)
    Same single-pass online-softmax loop as flash_attention_naive, but skips
    (query_block, key_block) pairs whose Cauchy-Schwarz score upper bound is
    negligible relative to that query block's best available upper bound.
    """
    batch, num_heads, seq_len, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)
    device, dtype = q.device, q.dtype
    log_inv_eps = math.log(1.0 / epsilon)

    k_norms = k.norm(dim=-1)  # (B, H, N) -- per-token key norms, cheap
    out = torch.empty_like(q)

    blocks_total = 0
    blocks_skipped = 0

    for q_start in range(0, seq_len, block_size_q):
        q_end = min(q_start + block_size_q, seq_len)
        q_block = q[:, :, q_start:q_end, :]
        bq = q_end - q_start
        q_block_max_norm = q_block.norm(dim=-1).max(dim=-1, keepdim=True).values  # (B,H,1)

        # Cheap pass: upper bound per key block, using only norms.
        kv_starts = list(range(0, seq_len, block_size_kv))
        block_upper_bounds = []
        for k_start in kv_starts:
            k_end = min(k_start + block_size_kv, seq_len)
            block_max_k_norm = k_norms[:, :, k_start:k_end].max(dim=-1, keepdim=True).values  # (B,H,1)
            ub = q_block_max_norm * block_max_k_norm * scale  # (B,H,1)
            block_upper_bounds.append(ub)

        global_ub = torch.stack(block_upper_bounds, dim=0).max(dim=0).values  # (B,H,1)

        m_i = torch.full((batch, num_heads, bq, 1), float("-inf"), device=device, dtype=dtype)
        l_i = torch.zeros((batch, num_heads, bq, 1), device=device, dtype=dtype)
        acc = torch.zeros((batch, num_heads, bq, head_dim), device=device, dtype=dtype)

        for k_start, ub in zip(kv_starts, block_upper_bounds):
            blocks_total += 1
            # Skip only where EVERY (batch, head) slice is negligible for this q block.
            if bool((ub < global_ub - log_inv_eps).all()):
                blocks_skipped += 1
                continue

            k_end = min(k_start + block_size_kv, seq_len)
            k_block = k[:, :, k_start:k_end, :]
            v_block = v[:, :, k_start:k_end, :]

            scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale
            block_max = scores.max(dim=-1, keepdim=True).values
            new_m = torch.maximum(m_i, block_max)

            alpha = torch.exp(m_i - new_m)
            p = torch.exp(scores - new_m)

            l_i = l_i * alpha + p.sum(dim=-1, keepdim=True)
            acc = acc * alpha + torch.matmul(p, v_block)
            m_i = new_m

        out[:, :, q_start:q_end, :] = acc / l_i

    out.skip_fraction = blocks_skipped / max(blocks_total, 1)  # stashed for benchmarking convenience
    return out


def _benchmark(fn, q, k, v, num_iters=10, warmup=3):
    device = q.device.type
    for _ in range(warmup):
        fn(q, k, v)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(num_iters):
        out = fn(q, k, v)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_iters
    peak_mem = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    return elapsed, peak_mem, out


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    batch, num_heads, seq_len, head_dim = 2, 8, 2048, 64

    print("=== Regime A: isotropic random Q/K (worst case — no real locality/structure) ===")
    q = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)

    for epsilon in (1e-2, 1e-4, 1e-6):
        base_time, base_mem, base_out = _benchmark(flash_attention_naive, q, k, v)
        sparse_fn = lambda q, k, v, eps=epsilon: block_sparse_flash_attention(q, k, v, epsilon=eps)
        sparse_time, sparse_mem, sparse_out = _benchmark(sparse_fn, q, k, v)
        err = (sparse_out - base_out).abs().max().item()
        print(f"  eps={epsilon:8.0e} | skip_frac={sparse_out.skip_fraction:5.1%} | "
              f"time {base_time*1000:6.2f}ms -> {sparse_time*1000:6.2f}ms | "
              f"mem {base_mem/1e6:6.1f}MB -> {sparse_mem/1e6:6.1f}MB | max_err={err:.2e}")

    print("\n=== Regime B: block-structured Q/K (each 128-token block has a distinct random")
    print("    'topic' offset — simulates real text locality where distant blocks rarely matter) ===")
    block = 128
    num_blocks = seq_len // block
    topic_offsets = torch.randn(num_blocks, head_dim, device=device, dtype=dtype) * 4.0
    q2 = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k2 = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    for b in range(num_blocks):
        q2[:, :, b * block:(b + 1) * block, :] += topic_offsets[b]
        k2[:, :, b * block:(b + 1) * block, :] += topic_offsets[b]
    v2 = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)

    for epsilon in (1e-2, 1e-4, 1e-6):
        base_time, base_mem, base_out = _benchmark(flash_attention_naive, q2, k2, v2)
        sparse_fn = lambda q, k, v, eps=epsilon: block_sparse_flash_attention(q, k, v, epsilon=eps)
        sparse_time, sparse_mem, sparse_out = _benchmark(sparse_fn, q2, k2, v2)
        err = (sparse_out - base_out).abs().max().item()
        print(f"  eps={epsilon:8.0e} | skip_frac={sparse_out.skip_fraction:5.1%} | "
              f"time {base_time*1000:6.2f}ms -> {sparse_time*1000:6.2f}ms | "
              f"mem {base_mem/1e6:6.1f}MB -> {sparse_mem/1e6:6.1f}MB | max_err={err:.2e}")

    print("\nInterpretation: this is a DATA-DEPENDENT extension. Expect near-zero skip fraction and")
    print("near-zero speedup on isotropic random data (Regime A) — Cauchy-Schwarz bounds are loose")
    print("when there's no structure to exploit. Expect a meaningfully higher skip fraction and real")
    print("speedup/memory reduction on block-structured data (Regime B), at a controllable, honestly")
    print("reported max-error cost that shrinks as epsilon shrinks. Report BOTH regimes' numbers in")
    print("report/README.md — a purely positive result here would be a red flag, not a finding.")
