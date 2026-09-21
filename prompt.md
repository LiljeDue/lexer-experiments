# Optimising Pass 1 of a Parallel Lexer on GPU

## Background

This repository implements a parallel lexer on the GPU. The lexer tokenises a
stream of bytes by running a DFA (deterministic finite automaton) over the
input. Because the DFA transitions are data-dependent, a naive parallel
implementation cannot process each byte independently — each byte's state
depends on the cumulative state of all preceding bytes.

The solution is a **two-pass approach**:

- **Pass 1 (P1)**: compute a prefix scan over the input, where the scan
  operator is DFA state composition. The result is a flat array of prefix
  states, one per input byte.
- **Pass 2 (P2)**: read the prefix states and emit tokens.

P1 is the bottleneck. This document describes the P1 kernel, what we have
tried to optimise it, and where we currently stand.

## The DFA

The lexer recognises a simple parenthesised-identifier language. The DFA has
**12 states**. Each byte maps to an initial state via a 256-entry `to_state[]`
table. States are composed using a 12×12 composition table (`compose[]`).

`state_t` is `uint16_t` (2 bytes). The identity element for the scan is
`IDENTITY = 74`.

## Pass 1 Algorithm: Decoupled Lookback Scan

P1 implements a **decoupled lookback** prefix scan (the same algorithm used
internally by CUB's `DeviceScan`). The key ideas:

1. Each block processes a contiguous tile of `BLOCK_SIZE × ITEMS_PER_THREAD`
   bytes.
2. Within a block, CUB `BlockScan` computes an intra-block inclusive scan
   using the DFA composition operator.
3. Across blocks, a **tile descriptor array** (`ScanTileState`) holds
   per-tile status (INVALID / PARTIAL / INCLUSIVE) and partial aggregates,
   packed into a single `uint32_t` per tile for atomic access.
4. Each block performs a **lookback**: it walks backwards through predecessor
   tile descriptors, accumulating the exclusive prefix, until it finds a tile
   marked INCLUSIVE (or hits the out-of-bounds sentinel).
5. The exclusive prefix is combined with the intra-block scan result to
   produce the final inclusive prefix for each element.

Block assignment is **dynamic** — a global atomic counter (`dyn_index_ptr`)
assigns tile indices at runtime, ensuring load balance regardless of SM
scheduling order.

### Key parameters (current)

| Parameter | Value |
|---|---|
| `BLOCK_SIZE` | 256 |
| `ITEMS_PER_THREAD` | 22 |
| Registers/thread (sm_80) | 40 |
| Theoretical occupancy (A100) | 75% (6 blocks/SM, limited by registers) |
| Dynamic shmem/block | 11264 bytes (states array) |
| Static shmem/block | 1416 bytes (compose + to_state tables) |

`__launch_bounds__(256, 6)` is applied on sm_80+ to hint the compiler to
target 6 blocks/SM, reducing register count from 47 to 40.

## Benchmark Setup

Input: `data/tokens_dense_500MiB.in` — 500 MiB of dense token data.

Hardware: **NVIDIA A100 (sm_80)**, 108 SMs, 2 TB/s HBM2e bandwidth.

All timings are mean over 100 runs with 500 warmup iterations, reported as
`μs` with 95% CI and effective GB/s (input bytes read + output bytes written).

```
make bench_p1    # builds and runs p1_bench on dense 500MiB input
make profile_p1  # builds with -lineinfo -DPROFILE, runs ncu --set full,
                 # writes profile_p1.txt and profile_p1.ncu-rep
```

## Current Benchmark Results (A100)

```
BW ceiling BS256/IPT22 (read only):    537μs   1331 GB/s
2Pass P1 BS256/IPT22 (add scan):      2032μs    774 GB/s
2Pass P1 BS256/IPT22 (DeviceScan):    2173μs    724 GB/s
2Pass P1 BS256/IPT22 (NregNone):      2261μs    696 GB/s
2Pass P1 BS128/IPT15 (cub-style):     2795μs    563 GB/s
```

The BW ceiling (read-only, no scan) achieves 1331 GB/s — essentially saturating
HBM2e. `p1_nregnone` achieves 696 GB/s, roughly **52% of the memory bandwidth
ceiling**. IPT=22 was confirmed optimal by a sweep over IPT=16/20/24/28/32.

## Variant Descriptions

### BW ceiling
Reads input using the same u64 coalesced load pattern as P1, XORs into a
per-thread accumulator, writes one u64 per thread. No scan, no shmem tables.
Upper bound on achievable P1 throughput.

### Add scan (`p1_add`)
Identical to `p1_nregnone` but replaces DFA composition with integer addition.
Output is meaningless. Used to isolate scan synchronisation overhead from
composition cost. Result: composition accounts for ~10% of P1 time at 774 GB/s;
the remaining ~90% is scan synchronisation overhead.

### DeviceScan
Uses `cub::DeviceScan::InclusiveScan` with a
`thrust::make_transform_iterator` to fuse byte→state mapping into the scan.
The compose table lives in global memory (no shmem control inside DeviceScan).
CUB selects the `DefaultPolicy` (128 threads, 15 IPT,
`BLOCK_LOAD_WARP_TRANSPOSE`) because `ComposeOp` is not a primitive operator.
Result: 724 GB/s — comparable to our handwritten kernel despite 72
registers/thread (43% theoretical occupancy vs our 75%).

### NregNone (`p1_nregnone`)
Full P1: u64 coalesced loads, byte→state via shmem `to_state[]`, inclusive
prefix scan with DFA composition via shmem `compose[]`, u64 coalesced writes.
Best real implementation: **696 GB/s**.

### CUB-style (`p1_cub_style`)
Matches CUB `DefaultPolicy` exactly: 128 threads, 15 IPT,
`BLOCK_LOAD_WARP_TRANSPOSE`, `BLOCK_STORE_WARP_TRANSPOSE`, `BLOCK_SCAN_WARP_SCANS`,
but adds compose table in static shmem. Result: 563 GB/s — worse than NregNone
despite correct load/store patterns, because 128T/15IPT has more tiles and
lower per-tile throughput than 256T/22IPT.

## Profiling Findings (ncu --set full, A100)

### `p1_nregnone` (pre-relaxed-store)

| Metric | Value |
|---|---|
| DRAM throughput | 25.3% of peak |
| L1/TEX throughput | 63.8% |
| Achieved occupancy | 67.8% |
| Warp cycles/instruction | 20.0 |
| Top stall | CTA barrier (44% of cycles) |
| Shared store bank conflicts | 58.7% excessive wavefronts (2.5-way avg) |

### Roofline position

The roofline model reported `p1_nregnone` at the ridge point — simultaneously
at the memory-bandwidth and compute ceiling for its arithmetic intensity.
**This was misleading**: the bank conflicts added replays that the roofline
counted as compute utilisation, and the `__threadfence()` in `SetPartial`/
`SetInclusive` added serialising barriers not visible as memory pressure.
Switching to `st.relaxed.gpu` PTX stores improved NregNone from 642 → 696 GB/s.

### `DeviceScanKernel` (CUB)

| Metric | Value |
|---|---|
| Registers/thread | 72 (limits occupancy to 43%) |
| Theoretical occupancy | 43.75% |
| Top stall | CTA barrier (40% of cycles) |

CUB uses `st.relaxed.gpu` / `ld.relaxed.gpu` in its `ScanTileState`, avoiding
`__threadfence()`. This is why it outperformed our kernel at 724 vs 642 GB/s
before we fixed our tile state implementation.

## What We Tried and Why It Didn't Help

### BS=32 (warp-scan, no intra-block barriers)
Eliminates `__syncthreads()` inside `BlockScan` by using a single warp per
block. Result: 3.7× slower. Reason: killing occupancy (1 block/SM vs 5–6)
removes all latency hiding. Barriers are not the bottleneck — occupancy is
needed to hide the lookback latency.

### `__launch_bounds__(256, 6)` on sm_80
Forces the compiler to target 6 blocks/SM, reducing registers from 47 to 40.
Result: ~11–15% speedup. Applied via arch-conditional `LB_P1` macro (sm_80+
only, to avoid ptxas warnings on sm_75).

### Shmem padding (IPT=21/24/32, padded stride)
Intended to eliminate the 2.5-way shmem store bank conflicts reported by the
profiler. Multiple approaches tried (scalar write path, register-direct write,
nodiv load). All measured ~155 GB/s. Root causes: non-coalesced global stores
(blocked layout), expensive non-power-of-two integer division in load phase,
and ultimately the bank conflicts are masked by the dominant barrier stall.

### IPT sweep (16/20/24/28/32)
Swept ITEMS_PER_THREAD to find whether fewer tiles (larger IPT) reduces lookback
stall. IPT=22 is optimal. IPT=32 collapses to 305 GB/s despite no register
spill — likely instruction pressure from the larger unrolled `st[]` array.

### CUB-style kernel (128T/15IPT, BLOCK_LOAD_WARP_TRANSPOSE)
Matched DeviceScan's DefaultPolicy exactly but added compose table in shmem.
Result: 563 GB/s — worse than NregNone (696 GB/s). 128T/15IPT has ~3× more
tiles than 256T/22IPT, increasing lookback synchronisation overhead.

### `p1_bench` static shmem layout
`p1_bench` places the compose and to_state tables in **static** shmem
(vs dynamic shmem in `cuda_lexer.cu`). This reduces dynamic shmem per block
from 11552 to 11264 bytes.

## What Actually Helped

| Change | Before | After | Gain |
|---|---|---|---|
| Static shmem for tables | ~550 GB/s | 642 GB/s | +92 GB/s |
| `__launch_bounds__(256,6)` on sm_80 | ~560 GB/s | 642 GB/s | ~80 GB/s |
| `st.relaxed.gpu` tile state stores | 642 GB/s | 696 GB/s | +54 GB/s |

## Conclusion

P1 is limited by the decoupled lookback scan algorithm itself:

- ~90% of P1 time is scan synchronisation overhead (CTA barrier stalls from
  the intra-block `BlockScan` and the cross-tile lookback).
- ~10% is the DFA composition operator cost (add scan at 774 vs NregNone at 696 GB/s).
- IPT=22, BS=256 is the optimal tile configuration.
- The gap to BW ceiling (696 vs 1331 GB/s) is irreducible with decoupled lookback.

Meaningful improvement would require a fundamentally different scan algorithm
that avoids cross-tile synchronisation, which is not possible for a general
prefix scan with a data-dependent operator.
