"""
Phase 4 — Measure Attention Cost: empirically observe O(n^2) scaling.

Benchmarks runtime and peak GPU memory of standard (materialized-matrix)
attention across sequence lengths 128 -> 4096, saves the raw numbers as JSON
(consumed by the dashboard) and a publication-quality matplotlib plot.

Run:
    python measure_attention_cost.py
Outputs:
    results/attention_cost.json
    results/attention_cost.png
"""

import json
import math
import os
import time

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def standard_attention(q, k, v):
    """The thing FlashAttention exists to replace: materializes the full
    (seq_len x seq_len) score matrix in HBM."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # <-- O(n^2) materialization
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def benchmark_one(seq_len, batch, num_heads, head_dim, device, dtype, num_iters=20, warmup=5):
    q = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)

    for _ in range(warmup):
        standard_attention(q, k, v)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(num_iters):
        out = standard_attention(q, k, v)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_iters

    peak_mem = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    return elapsed, peak_mem


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    batch, num_heads, head_dim = 2, 8, 64

    seq_lens = [128, 256, 512, 1024, 2048, 4096]
    latencies_ms, peak_mems_mb = [], []

    print(f"device={device}, dtype={dtype}, batch={batch}, heads={num_heads}, head_dim={head_dim}\n")

    for seq_len in seq_lens:
        try:
            elapsed, peak_mem = benchmark_one(seq_len, batch, num_heads, head_dim, device, dtype)
            latencies_ms.append(elapsed * 1000)
            peak_mems_mb.append(peak_mem / 1e6)
            print(f"seq_len={seq_len:5d} | latency={elapsed*1000:8.3f} ms | peak_mem={peak_mem/1e6:8.1f} MB")
        except torch.cuda.OutOfMemoryError:
            print(f"seq_len={seq_len:5d} | OOM materializing the full attention matrix (this IS the point)")
            latencies_ms.append(None)
            peak_mems_mb.append(None)
            torch.cuda.empty_cache()

    os.makedirs("results", exist_ok=True)
    payload = {
        "seq_lens": seq_lens,
        "latency_ms": latencies_ms,
        "peak_mem_mb": peak_mems_mb,
        "config": {"batch": batch, "num_heads": num_heads, "head_dim": head_dim, "device": device},
    }
    with open("results/attention_cost.json", "w") as f:
        json.dump(payload, f, indent=2)
    with open("results/attention_cost.js", "w") as f:
        f.write("window.ATTENTION_COST_DATA = ")
        json.dump(payload, f, indent=2)
        f.write(";\n")

    valid = [(s, l, m) for s, l, m in zip(seq_lens, latencies_ms, peak_mems_mb) if l is not None]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    xs = [v[0] for v in valid]
    ax1.plot(xs, [v[1] for v in valid], marker="o", color="#d62728")
    ax1.set_xlabel("sequence length")
    ax1.set_ylabel("latency (ms)")
    ax1.set_title("Standard attention latency vs sequence length")
    ax1.grid(alpha=0.3)

    ax2.plot(xs, [v[2] for v in valid], marker="o", color="#1f77b4")
    ax2.set_xlabel("sequence length")
    ax2.set_ylabel("peak GPU memory (MB)")
    ax2.set_title("Standard attention memory vs sequence length (O(n^2))")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("results/attention_cost.png", dpi=150)
    print("\nSaved results/attention_cost.json and results/attention_cost.png")
