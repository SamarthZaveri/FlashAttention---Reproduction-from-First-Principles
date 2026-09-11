"""
Phase 7 — Blocked (tiled) attention.

Processes attention in tiled blocks instead of materializing the full
(seq_len x seq_len) matrix at once, and benchmarks the memory savings vs.
standard attention. This is a deliberately two-pass version (compute
row statistics by streaming block scores once, then recompute block scores
a second time to accumulate the output) — simpler than a fully fused
single-pass kernel, and a useful intermediate checkpoint before Phase 8
folds both passes into one loop using online-softmax rescaling.

Run:
    python blocked_attention.py
Expect:
    correctness check against a reference implementation, then a peak-memory
    comparison against standard (fully materialized) attention.
"""

import math

import torch


def blocked_attention(q, k, v, block_size=128):
    """
    q, k, v: (batch, num_heads, seq_len, head_dim)
    Two-pass tiled attention: never holds more than a (block_size x seq_len)
    (actually (block_size x block_size)) score tile in memory at once.
    """
    batch, num_heads, seq_len, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)
    device, dtype = q.device, q.dtype

    out = torch.empty_like(q)

    for q_start in range(0, seq_len, block_size):
        q_end = min(q_start + block_size, seq_len)
        q_block = q[:, :, q_start:q_end, :]  # (B, H, bq, D)

        # --- Pass 1: stream over K blocks to get the true row max and sum ---
        row_max = torch.full((batch, num_heads, q_end - q_start, 1), float("-inf"), device=device, dtype=dtype)
        row_sum = torch.zeros((batch, num_heads, q_end - q_start, 1), device=device, dtype=dtype)

        for k_start in range(0, seq_len, block_size):
            k_end = min(k_start + block_size, seq_len)
            k_block = k[:, :, k_start:k_end, :]

            scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale  # (B,H,bq,bk) tile only
            block_max = scores.max(dim=-1, keepdim=True).values
            new_max = torch.maximum(row_max, block_max)

            row_sum = row_sum * torch.exp(row_max - new_max) + torch.exp(scores - new_max).sum(dim=-1, keepdim=True)
            row_max = new_max

        # --- Pass 2: recompute score tiles (cheap FLOPs, avoids storing them) and accumulate PV ---
        acc = torch.zeros((batch, num_heads, q_end - q_start, head_dim), device=device, dtype=dtype)
        for k_start in range(0, seq_len, block_size):
            k_end = min(k_start + block_size, seq_len)
            k_block = k[:, :, k_start:k_end, :]
            v_block = v[:, :, k_start:k_end, :]

            scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale
            probs = torch.exp(scores - row_max) / row_sum  # exact, since row_max/row_sum are the true global values
            acc += torch.matmul(probs, v_block)

        out[:, :, q_start:q_end, :] = acc

    return out


def _reference_attention(q, k, v):
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # fp32 for a tight correctness bound; see flash/naive_flash_attention.py for fp16

    batch, num_heads, seq_len, head_dim = 2, 4, 1024, 64
    q = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)

    out = blocked_attention(q, k, v, block_size=128)
    ref = _reference_attention(q, k, v)
    diff = (out - ref).abs().max().item()
    print(f"max abs diff vs reference attention: {diff:.8f}")
    assert diff < 1e-4, "blocked_attention does not match reference"
    print("OK: blocked (tiled) attention matches standard attention exactly.\n")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        _ = blocked_attention(q, k, v, block_size=128)
        torch.cuda.synchronize()
        blocked_peak = torch.cuda.max_memory_allocated()

        torch.cuda.reset_peak_memory_stats()
        _ = _reference_attention(q, k, v)
        torch.cuda.synchronize()
        standard_peak = torch.cuda.max_memory_allocated()

        print(f"peak mem — blocked: {blocked_peak/1e6:.1f} MB, standard: {standard_peak/1e6:.1f} MB "
              f"({standard_peak/blocked_peak:.2f}x more for standard)")
    else:
        print("(CUDA not available — memory comparison skipped; run on GPU for the real numbers.)")
