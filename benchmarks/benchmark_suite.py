"""
Phase 10 — Full benchmark suite: standard vs. naive FlashAttention vs.
Triton FlashAttention vs. the Phase-11 extension, across sequence lengths.

Measures latency, throughput (tokens/sec), peak GPU memory, and (if
`pynvml` is installed) GPU utilization %. Saves:
    results/benchmark_results.json   (raw data)
    results/benchmark_results.js     (same data as `window.BENCHMARK_DATA`,
                                       so dashboard/index.html can load it
                                       via <script src> and avoid the
                                       file:// CORS issues fetch() hits)
    results/benchmark_latency.png
    results/benchmark_memory.png
    results/benchmark_speedup.png

Run (from attention-lab/benchmarks/):
    python benchmark_suite.py

Triton and the block-sparse extension are optional: if triton isn't
installed, or import fails, that method's line is simply omitted from the
results/plots rather than crashing the whole run.
"""

import json
import math
import os
import sys
import time

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "flash"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kernels"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from naive_flash_attention import flash_attention_naive  # noqa: E402

try:
    from flash_attention_triton import flash_attention_triton
    HAS_TRITON = True
except Exception as e:
    print(f"[warn] Triton FlashAttention unavailable ({e}); skipping in benchmark suite.")
    HAS_TRITON = False

try:
    from block_sparse_flash_attention import block_sparse_flash_attention
    HAS_EXTENSION = True
except Exception as e:
    print(f"[warn] Extension unavailable ({e}); skipping in benchmark suite.")
    HAS_EXTENSION = False

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    HAS_NVML = True
except Exception:
    HAS_NVML = False


def standard_attention(q, k, v):
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


METHODS = {"standard": standard_attention, "naive_flash": flash_attention_naive}
if HAS_TRITON:
    METHODS["triton_flash"] = flash_attention_triton
if HAS_EXTENSION:
    METHODS["block_sparse_extension"] = block_sparse_flash_attention


def sample_gpu_util():
    if not HAS_NVML:
        return None
    return pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE).gpu


def benchmark_method(fn, q, k, v, num_iters=15, warmup=5):
    device = q.device.type
    for _ in range(warmup):
        fn(q, k, v)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    utils = []
    start = time.perf_counter()
    for _ in range(num_iters):
        fn(q, k, v)
        u = sample_gpu_util()
        if u is not None:
            utils.append(u)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_iters

    peak_mem = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    avg_util = sum(utils) / len(utils) if utils else None
    return elapsed, peak_mem, avg_util


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (device == "cuda" and HAS_TRITON) else torch.float32
    batch, num_heads, head_dim = 2, 8, 64
    seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192]

    print(f"device={device}, dtype={dtype}, methods={list(METHODS)}\n")

    results = {name: {"seq_lens": [], "latency_ms": [], "peak_mem_mb": [], "gpu_util_pct": [], "tokens_per_sec": []}
               for name in METHODS}

    for seq_len in seq_lens:
        q = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)

        row = [f"seq_len={seq_len}"]
        for name, fn in METHODS.items():
            try:
                elapsed, peak_mem, util = benchmark_method(fn, q, k, v)
                tokens_per_sec = (batch * seq_len) / elapsed
                results[name]["seq_lens"].append(seq_len)
                results[name]["latency_ms"].append(elapsed * 1000)
                results[name]["peak_mem_mb"].append(peak_mem / 1e6)
                results[name]["gpu_util_pct"].append(util)
                results[name]["tokens_per_sec"].append(tokens_per_sec)
                row.append(f"{name}: {elapsed*1000:.2f}ms / {peak_mem/1e6:.0f}MB")
            except torch.cuda.OutOfMemoryError:
                row.append(f"{name}: OOM")
                torch.cuda.empty_cache()
        print(" | ".join(row))

    os.makedirs("results", exist_ok=True)
    payload = {"config": {"batch": batch, "num_heads": num_heads, "head_dim": head_dim,
                           "device": device, "dtype": str(dtype)},
               "results": results}

    with open("results/benchmark_results.json", "w") as f:
        json.dump(payload, f, indent=2)
    with open("results/benchmark_results.js", "w") as f:
        f.write("window.BENCHMARK_DATA = ")
        json.dump(payload, f, indent=2)
        f.write(";\n")

    colors = {"standard": "#d62728", "naive_flash": "#ff7f0e",
              "triton_flash": "#2ca02c", "block_sparse_extension": "#9467bd"}

    # --- latency plot ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, data in results.items():
        if data["seq_lens"]:
            ax.plot(data["seq_lens"], data["latency_ms"], marker="o", label=name, color=colors.get(name))
    ax.set_xlabel("sequence length")
    ax.set_ylabel("latency (ms)")
    ax.set_title("Attention latency: standard vs. FlashAttention variants")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/benchmark_latency.png", dpi=150)

    # --- memory plot ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, data in results.items():
        if data["seq_lens"]:
            ax.plot(data["seq_lens"], data["peak_mem_mb"], marker="o", label=name, color=colors.get(name))
    ax.set_xlabel("sequence length")
    ax.set_ylabel("peak GPU memory (MB)")
    ax.set_title("Peak memory: standard (O(n^2)) vs. FlashAttention variants (~O(n))")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/benchmark_memory.png", dpi=150)

    # --- speedup plot (vs standard, where both exist) ---
    fig, ax = plt.subplots(figsize=(7, 5))
    if results["standard"]["seq_lens"]:
        std_map = dict(zip(results["standard"]["seq_lens"], results["standard"]["latency_ms"]))
        for name, data in results.items():
            if name == "standard" or not data["seq_lens"]:
                continue
            xs, speedups = [], []
            for s, lat in zip(data["seq_lens"], data["latency_ms"]):
                if s in std_map:
                    xs.append(s)
                    speedups.append(std_map[s] / lat)
            ax.plot(xs, speedups, marker="o", label=name, color=colors.get(name))
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("sequence length")
    ax.set_ylabel("speedup vs. standard attention (x)")
    ax.set_title("Speedup over standard attention")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/benchmark_speedup.png", dpi=150)

    print("\nSaved results/benchmark_results.{json,js} and 3 plots to results/.")
    print("Open dashboard/index.html in a browser to explore interactively.")
