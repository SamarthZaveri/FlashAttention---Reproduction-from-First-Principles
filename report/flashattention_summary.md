# FlashAttention — Technical Summary

## 1. What problem exists

Standard implementations of attention are slow and memory-hungry at long
sequence lengths, but *not* primarily because of the number of floating-point
operations they perform. The QKᵀ matmul, softmax, and the probability×V
matmul all involve roughly O(N²·d) FLOPs for sequence length N and head
dimension d — a completely tractable amount of compute for a modern GPU.
The actual bottleneck is **memory traffic**: how many bytes get moved
between GPU high-bandwidth memory (HBM, the several-GB-to-TB main GPU
memory) and the much smaller, much faster on-chip SRAM (shared
memory/registers, tens of KB per streaming multiprocessor). GPUs have
compute throughput that has grown far faster than memory bandwidth, so
operations that are dominated by data movement rather than arithmetic —
"memory-bandwidth-bound" operations — leave most of the GPU's FLOPs idle
while it waits on memory transfers. Standard attention is exactly this kind
of operation, and it gets worse quadratically as sequence length grows.

## 2. What causes it

A standard attention implementation executes as a sequence of distinct
kernels, each of which reads its inputs from HBM and writes its full output
back to HBM before the next kernel starts:

1. Compute S = QKᵀ, write the full N×N score matrix to HBM.
2. Read S back from HBM, compute the softmax, write the full N×N
   probability matrix P back to HBM.
3. Read P and V back from HBM, compute O = PV, write O to HBM.

The culprit is step 1–2: **materializing the full N×N attention matrix**
(scores and then probabilities) in HBM. For N=4096, storing this matrix in
fp16 for a single head already costs 32MB, and it must be written and then
re-read at least twice (once by softmax, once by the PV matmul) — and that
cost is paid per head and per batch element, and it scales as O(N²) in both
compute *and* memory, unlike the rest of the transformer's operations which
scale closer to O(N).

## 3. Why that's costly

Each of those HBM reads/writes is orders of magnitude slower than an
equivalent access to on-chip SRAM (roughly 1–2 TB/s HBM bandwidth vs.
~19 TB/s SRAM bandwidth on a GPU like the A100, i.e. an order of magnitude
gap). Because the N×N matrix is far too large to fit in SRAM at once for any
non-trivial sequence length, standard implementations have no choice but to
round-trip it through HBM between each of the three kernels above. The GPU
ends up spending most of its wall-clock time waiting on these memory
transfers rather than doing arithmetic — it is memory-bandwidth bound, not
compute bound. This is also precisely what causes the O(N²) *memory usage*
that makes long-context attention run out of GPU memory well before it runs
out of compute budget: the N×N matrix has to be allocated and held
somewhere, even though it's only ever used as scratch space for a much
smaller final output.

## 4. FlashAttention's core insight: never materialize the full matrix

FlashAttention restructures the three-kernel computation above into a
**single fused kernel** that never writes the N×N matrix to HBM at all. It
does this with two ideas working together:

**Tiling.** Q, K, and V are split into fixed-size blocks. For a given block
of Q, the kernel loops over blocks of K and V, and at each step computes
only the (block_size × block_size) tile of the score matrix — small enough
to live entirely in fast SRAM. Only the final (block_size × head_dim) output
tile and a couple of scalar per-row statistics ever get written back to
HBM.

**Online (streaming) softmax.** The obstacle to tiling is that softmax's
denominator needs the sum over the *entire* row, and numerically stable
softmax needs the row's *max*, both of which are normally only knowable
after seeing the whole row — i.e., after materializing the whole matrix.
FlashAttention resolves this by maintaining, for each query row, a running
maximum and a running (appropriately rescaled) sum as it streams over key
blocks, updating the accumulated output incrementally with each new block
using a numerically-exact rescaling factor. By the time the last key block
has been processed, the running statistics are mathematically identical to
what a single-pass softmax over the full row would have produced — no
approximation, just algebraic reformulation. (This is the same technique
this project's Phase 6 implements and numerically verifies against
`torch.softmax` before Phase 7 tiles it and Phase 8 turns it into a
complete drop-in attention function.)

The net effect is that attention's FLOP count is essentially unchanged, but
its HBM traffic drops from O(N²) to roughly O(N²/M) where M is the SRAM
tile size — in practice this is what lets FlashAttention run several times
faster than standard attention *and* use memory that scales linearly, not
quadratically, in sequence length, which is exactly what this project's
Phase 4 benchmark (standard attention) and Phase 8/9 benchmarks (naive and
Triton FlashAttention) are designed to demonstrate side by side.

## 5. Scope note

The original paper also covers the backward pass (which recomputes
attention probabilities on-the-fly during the backward kernel, trading a
small amount of extra compute for avoiding the storage of the N×N matrix
from the forward pass), block-sparse variants, and detailed IO-complexity
proofs. This project's scope (per §3.4) is the forward pass only, which is
sufficient to demonstrate and benchmark the tiling + online-softmax
mechanism that is the paper's central contribution.
