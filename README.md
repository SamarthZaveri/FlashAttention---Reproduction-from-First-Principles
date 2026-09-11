# attention-lab

FlashAttention, reproduced from first principles and extended. Built to
Project 3's spec: every stage is verified against a reference before moving
to the next, nothing is trusted just because it runs.

You have a Blackwell GPU, so every phase below should run natively —
Triton's Windows fork (`triton-windows`) officially supports Blackwell on
Triton ≥3.3 + CUDA ≥12.8; if you're on Linux/WSL2 just use stock `triton`.

## Setup

```bash
pip install torch matplotlib
pip install triton-windows   # Windows + NVIDIA GPU
# pip install triton         # Linux / WSL2, instead of the line above
pip install pynvml           # optional: enables GPU-utilization% in Phase 10
```

## Run order

```
vanilla/self_attention.py           # Phase 1 — verify vs torch SDPA
vanilla/multihead_attention.py      # Phase 2 — verify vs nn.MultiheadAttention
vanilla/transformer_encoder.py      # Phase 3 — train on synthetic task, check val acc
benchmarks/measure_attention_cost.py# Phase 4 — O(n^2) blowup, writes results/attention_cost.{json,js,png}
report/flashattention_summary.md    # Phase 5 — read (already written, no code to run)
flash/online_softmax.py             # Phase 6 — verify vs torch.softmax
flash/blocked_attention.py          # Phase 7 — verify + memory savings vs standard
flash/naive_flash_attention.py      # Phase 8 — verify + latency/memory table
kernels/matmul_triton.py            # Phase 9a — Triton warm-up
kernels/flash_attention_triton.py   # Phase 9b — fused GPU kernel, verify + memory check
benchmarks/benchmark_suite.py       # Phase 10 — full suite, writes results/benchmark_results.{json,js}
experiments/block_sparse_flash_attention.py  # Extension — two-regime honesty check
dashboard/index.html                # Open in a browser AFTER the two benchmark scripts above
```

Each script is self-contained (`python <file>.py`) and asserts its own
correctness — if a script's assertion fails, something upstream broke and
later phases shouldn't be trusted until it's fixed.

**Dashboard**: run `measure_attention_cost.py` and `benchmark_suite.py` from
inside `benchmarks/` first (they write `results/*.js` there), then just open
`dashboard/index.html` directly in a browser — no server needed, it loads
the `results/*.js` files via `<script src>` rather than `fetch()` so it
works straight off `file://`.

## Repo structure

```
attention-lab/
├── vanilla/            Phases 1–3: self-attn, multi-head, full encoder + training
├── flash/              Phases 6–8: online softmax, blocked attention, naive flash
├── kernels/            Phase 9: Triton matmul warm-up + fused flash-attn kernel
├── benchmarks/          Phases 4 & 10: cost measurement + full benchmark suite
│   └── results/         (generated) json/js/png outputs, gitignore-able
├── experiments/         Original extension: norm-bound block-sparse flash attention
├── report/               Phase 5 paper summary + this README
└── dashboard/            Interactive HTML dashboard over benchmarks/results/*.js
```

## The extension: norm-bound block-sparse FlashAttention

Cauchy-Schwarz gives a cheap (O(N), norms-only) upper bound on the scores a
given key block could possibly produce; blocks whose bound is provably
negligible relative to the running max get skipped entirely — full matmul,
memory traffic, and softmax update all avoided for that block. This is an
approximation with a tunable, honestly-bounded error (`epsilon`), not a free
lunch: `experiments/block_sparse_flash_attention.py` benchmarks it in two
regimes — isotropic random Q/K (where it should barely help, because
Cauchy-Schwarz bounds are loose without real structure) and block-structured
Q/K simulating text locality (where it should skip a real fraction of blocks
and show a genuine speed/memory win). Report both numbers; a suspiciously
uniform win across both regimes would mean the benchmark itself is broken,
not that the idea is unreasonably good.

## Success-metric checklist (per project spec §3.8)

- [ ] Every phase's assertion passes before moving to the next.
- [ ] `naive_flash_attention.py`'s benchmark table shows real memory savings
      at longer sequence lengths (check the printed MB columns).
- [ ] `block_sparse_flash_attention.py` reports a measurable effect in BOTH
      regimes, positive or negative — don't cherry-pick the favorable one
      for the writeup.
