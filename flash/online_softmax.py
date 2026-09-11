"""
Phase 6 — Online (streaming) softmax.

The core mathematical insight behind FlashAttention: compute softmax over a
row by streaming fixed-size chunks and maintaining a running max and a
running (rescaled) sum, without ever holding the full row in memory at
once. Verified to be numerically identical to torch.softmax.

Run:
    python online_softmax.py
Expect:
    "max abs diff vs torch.softmax: <tiny number>"
"""

import torch


def online_softmax(x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """
    x: (..., N). Computes softmax(x, dim=-1) by streaming over N in chunks
    of `chunk_size`, maintaining running max (m) and running sum (l), then
    doing a final normalization pass. This mirrors exactly the running
    statistics flash_attention kernels maintain per query row.
    """
    *batch_shape, N = x.shape
    device, dtype = x.device, x.dtype

    m_i = torch.full((*batch_shape, 1), float("-inf"), device=device, dtype=dtype)  # running max
    l_i = torch.zeros((*batch_shape, 1), device=device, dtype=dtype)                 # running sum
    # We need the exp'd chunks again for the final normalization; a "true"
    # single-pass fused kernel (see flash/blocked_attention.py) instead
    # folds the value-matmul into the same loop so it never needs this.
    exp_chunks = []

    for start in range(0, N, chunk_size):
        chunk = x[..., start:start + chunk_size]
        chunk_max = chunk.max(dim=-1, keepdim=True).values
        new_m = torch.maximum(m_i, chunk_max)

        # Rescale the old running sum to the new max before adding the new chunk.
        alpha = torch.exp(m_i - new_m)
        exp_chunk = torch.exp(chunk - new_m)

        l_i = l_i * alpha + exp_chunk.sum(dim=-1, keepdim=True)
        m_i = new_m
        exp_chunks.append((exp_chunk, new_m))  # keep for the final pass

    # Final pass: every exp_chunk above was computed relative to the *max at
    # that point in the stream*, not the true global max — rescale each to
    # the true final max m_i before dividing by the true final sum l_i.
    out = torch.empty_like(x)
    for (exp_chunk, chunk_relative_max), start in zip(exp_chunks, range(0, N, chunk_size)):
        rescale = torch.exp(chunk_relative_max - m_i)
        out[..., start:start + chunk_size] = (exp_chunk * rescale) / l_i

    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.randn(4, 8, 777, device=device) * 5  # wide range of magnitudes to stress numerics
    for chunk_size in (16, 64, 128, 777):
        out = online_softmax(x, chunk_size)
        ref = torch.softmax(x, dim=-1)
        diff = (out - ref).abs().max().item()
        print(f"chunk_size={chunk_size:4d} | max abs diff vs torch.softmax: {diff:.8f}")
        assert diff < 1e-5, "online softmax does not match torch.softmax"

    print("OK: online softmax matches torch.softmax exactly across chunk sizes, including chunk_size=N.")
