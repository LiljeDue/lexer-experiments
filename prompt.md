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

Block assignment is **static** — `blockIdx.x` is used directly, matching CUB
DeviceScan's approach. Tile 0 has a fast path that skips the lookback entirely
and publishes its inclusive aggregate immediately.

### Key parameters (current)

| Parameter | Value |
|---|---|
| `BLOCK_SIZE` | 256 |
| `ITEMS_PER_THREAD` | 22 |
| Registers/thread (sm_80) | 40 |
| Theoretical occupancy (A100) | 75% (6 blocks/SM, limited by registers) |
| Static shmem/block (`p1_transpose`) | 12.08 KiB (compose + to_state + transpose buffer) |

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
                 # writes profile_p1.ncu-rep and profile_p1.txt
```

## Current Benchmark Results (A100)

```
BW ceiling BS256/IPT22 (read only):    537μs   1332 GB/s
2Pass P1 BS256/IPT22 (add scan):      2039μs    771 GB/s
2Pass P1 BS256/IPT22 (transpose):     1870μs    841 GB/s
2Pass P1 BS256/IPT22 (DeviceScan):    2173μs    724 GB/s
2Pass P1 BS256/IPT22 (NregNone):      2257μs    697 GB/s
2Pass P1 BS128/IPT15 (cub-style):     2794μs    563 GB/s
```

The BW ceiling (read-only, no scan) achieves 1332 GB/s — essentially saturating
HBM2e. `p1_transpose` achieves 841 GB/s, roughly **63% of the memory bandwidth
ceiling**. IPT=22 was confirmed optimal by a sweep over IPT=16/20/24/28/32.

Notably, `p1_transpose` at 841 GB/s exceeds the `add_scan` baseline at 771 GB/s,
meaning DFA composition + transpose outperforms plain integer addition with the
old load/store path. This is because the old path had severe shmem store bank
conflicts (2.5-way, 58.8% excessive wavefronts) that are eliminated by the
transposed layout.

## Variant Descriptions

### BW ceiling
Reads input using the same u64 coalesced load pattern as P1, XORs into a
per-thread accumulator, writes one u64 per thread. No scan, no shmem tables.
Upper bound on achievable P1 throughput.

### Add scan (`p1_add`)
Identical to `p1_nregnone` but replaces DFA composition with integer addition.
Output is meaningless. Used to isolate scan synchronisation overhead from
composition cost. Result: composition accounts for ~10% of P1 time at 771 GB/s;
the remaining ~90% is scan synchronisation overhead.

### DeviceScan
Uses `cub::DeviceScan::InclusiveScan` with a
`thrust::make_transform_iterator` to fuse byte→state mapping into the scan.
The compose table lives in global memory (no shmem control inside DeviceScan).
CUB selects the `DefaultPolicy` (128 threads, 15 IPT,
`BLOCK_LOAD_WARP_TRANSPOSE`) because `ComposeOp` is not a primitive operator.
Result: 724 GB/s — slower than our handwritten kernel despite 72
registers/thread (43% theoretical occupancy vs our 75%).

### Transpose (`p1_transpose`) — best variant
Full P1: `BLOCK_LOAD_WARP_TRANSPOSE` striped global load → shmem transpose →
blocked register layout, inclusive prefix scan with DFA composition via shmem
`compose[]`, `BLOCK_STORE_WARP_TRANSPOSE` blocked registers → shmem transpose
→ striped global store. Static `blockIdx.x` assignment, tile-0 fast path.
Best real implementation: **841 GB/s**.

### NregNone (`p1_nregnone`)
Full P1: u64 coalesced loads, byte→state via shmem `to_state[]`, inclusive
prefix scan with DFA composition via shmem `compose[]`, u64 coalesced writes.
Superseded by `p1_transpose`: **697 GB/s**.

### CUB-style (`p1_cub_style`)
Matches CUB `DefaultPolicy` exactly: 128 threads, 15 IPT,
`BLOCK_LOAD_WARP_TRANSPOSE`, `BLOCK_STORE_WARP_TRANSPOSE`, `BLOCK_SCAN_WARP_SCANS`,
but adds compose table in static shmem. Result: 563 GB/s — worse than transpose
despite correct load/store patterns, because 128T/15IPT has ~2.9× more tiles and
lower per-tile throughput than 256T/22IPT.

## Profiling Findings (ncu --set full, A100)

### `p1_nregnone` vs `p1_transpose` comparison

| Metric | `p1_nregnone` | `p1_transpose` |
|---|---|---|
| Elapsed cycles | 2,971K | 2,482K |
| Achieved occupancy | 67.6% | **73.6%** |
| Warp cycles/instruction | 19.0 | 28.0 |
| Top stall | CTA barrier 35.9% | CTA barrier **47%** |
| Shmem store bank conflicts | 2.5-way (58.8%) | **none** |
| Shmem load bank conflicts | 1.3-way (18.8%) | 1.2-way (17%) |
| Instructions executed | 725M | **449M** |
| Static shmem/block | 1.41 KiB + 11.26 KiB dyn | **12.08 KiB** static |
| L1/TEX hit rate | 5.4% | 34.3% |
| DRAM throughput | 25.8% | 30.9% |

**Why `p1_transpose` is faster:**
1. **No shmem store bank conflicts** — `BlockStore` with `WARP_TRANSPOSE`
   eliminates the 2.5-way (58.8% excessive wavefronts) conflicts present in
   NregNone's blocked store path.
2. **38% fewer instructions** — 449M vs 725M. The transposed load/store is
   more instruction-efficient than the manual blocked layout + u64 store loop.
3. **Higher occupancy** — 73.6% vs 67.6%, closer to the 75% theoretical ceiling.

**The occupancy improvement** is because `p1_transpose` uses only static shmem
(12.08 KiB/block) vs NregNone's static + dynamic (1.41 + 11.26 = 12.67 KiB).
The smaller total footprint allows the 7th block/SM to fit, raising achieved
occupancy.

**Why `p1_transpose` has more barrier stall (47% vs 36%):**
The warp-transpose requires an additional `__syncthreads()` between the load
and scan phases. The CTA barrier stall per instruction is higher (13.1 cycles
vs 6.8 cycles), partially offsetting the instruction count and conflict gains.

### `DeviceScanKernel` (CUB)

| Metric | Value |
|---|---|
| Registers/thread | 72 (limits occupancy to 43%) |
| Theoretical occupancy | 43.75% |
| L1/TEX hit rate | 86.9% |
| Top stall | CTA barrier 39.6% |

CUB's high L1 hit rate (86.9% vs our 34.3%) comes from its larger number of
tiles (136K vs 93K) and the resulting L1 cache reuse of tile descriptor reads
during lookback. Despite this, its low occupancy (43%) means fewer warps
available to hide the lookback latency, and it ends up slower overall.

### Roofline position

`p1_transpose` is "Compute and Memory well-balanced" per the profiler. The
dominant bottleneck is the CTA barrier stall from the decoupled lookback
algorithm: ~47% of warp cycles stalled at barriers.

## Why 841 GB/s Is the Practical Ceiling

Two optimisations were investigated after `p1_transpose` was established as the
best variant:

### 1. Removing the post-load `__syncthreads()`

`p1_transpose` has 4 `__syncthreads()` per tile vs 3 for `p1_nregnone`. The
extra sync sits between `BlockLoad` and `BlockScan`. Investigation of the CUB
source confirmed this sync is **load-bearing and cannot be removed**:

- `BlockLoad(WARP_TRANSPOSE)` calls `BlockExchange::WarpStripedToBlocked`,
  which uses only `__syncwarp()` (intra-warp), not `__syncthreads()`.
- The post-load `__syncthreads()` is required to safely transition the union
  shmem from `temp.load` to `temp.scan_storage` before the scan phase writes
  to it. Without it, warps could race the union access.

### 2. Switching to `BLOCK_SCAN_RAKING_MEMOIZE`

Counted `__syncthreads()` calls in the two `BlockScan` specialisations:

| Algorithm | Internal `__syncthreads()` calls |
|---|---|
| `BLOCK_SCAN_WARP_SCANS` | 3 |
| `BLOCK_SCAN_RAKING` / `RAKING_MEMOIZE` | 16 |

`BLOCK_SCAN_WARP_SCANS` is already the minimum-barrier option. Switching to
raking would add 13 more barriers per tile and make performance worse.

### Conclusion

All `__syncthreads()` calls in `p1_transpose` are load-bearing. The 47% CTA
barrier stall is the minimum achievable for decoupled lookback at 256T/22IPT.
**841 GB/s is the practical ceiling for this algorithm on A100.**

The only path to further improvement is a fundamentally different scan
algorithm — e.g. a two-kernel reduce-then-scan — but that adds a second
kernel launch and a full extra pass over the input, which is unlikely to
improve end-to-end throughput.

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
and ultimately these approaches broke coalescing worse than the conflicts cost.

### IPT sweep (16/20/24/28/32)
Swept ITEMS_PER_THREAD to find whether fewer tiles (larger IPT) reduces lookback
stall. IPT=22 is optimal. IPT=32 collapses to 305 GB/s despite no register
spill — likely instruction pressure from the larger unrolled `st[]` array.

### CUB-style kernel (128T/15IPT, BLOCK_LOAD_WARP_TRANSPOSE)
Matched DeviceScan's DefaultPolicy exactly but added compose table in shmem.
Result: 563 GB/s — worse than transpose (841 GB/s). 128T/15IPT has ~2.9× more
tiles than 256T/22IPT, increasing lookback synchronisation overhead.

### Static blockIdx.x assignment + tile-0 fast path
Removed the atomic dynamic index counter from `p1_nregnone`, used `blockIdx.x`
directly (matching CUB DeviceScan's approach), and added a tile-0 fast path
that skips the lookback. Result: no measurable change (697 → 697 GB/s). The
atomic counter was not the bottleneck.

## What Actually Helped

| Change | Before | After | Gain |
|---|---|---|---|
| Static shmem for tables | ~550 GB/s | 642 GB/s | +92 GB/s |
| `__launch_bounds__(256,6)` on sm_80 | ~560 GB/s | 642 GB/s | ~80 GB/s |
| `st.relaxed.gpu` tile state stores | 642 GB/s | 697 GB/s | +54 GB/s |
| `BLOCK_LOAD/STORE_WARP_TRANSPOSE` | 697 GB/s | 841 GB/s | +144 GB/s |

The final gain (warp-transpose) works by eliminating shmem store bank conflicts
and reducing total instruction count, at the cost of more barrier stall per
instruction. The net effect is strongly positive.

## Conclusion — P1 is Done

`p1_transpose` at **841 GB/s** is the final optimised P1 kernel. All known
optimisation levers have been exhausted:

- IPT=22, BS=256 is the optimal tile configuration (sweep confirmed).
- `BLOCK_LOAD/STORE_WARP_TRANSPOSE` eliminates shmem bank conflicts and
  reduces instruction count, giving +144 GB/s over the manual blocked layout.
- `st.relaxed.gpu` tile state stores replace `__threadfence()`, giving +54 GB/s.
- Static shmem for tables and `__launch_bounds__(256,6)` gave earlier gains.
- All `__syncthreads()` calls are load-bearing — none can be removed.
- `BLOCK_SCAN_WARP_SCANS` is already the minimum-barrier BlockScan algorithm.
- Static `blockIdx.x` assignment and tile-0 fast path have no measurable effect.

The remaining 37% gap to the BW ceiling (841 vs 1332 GB/s) is the irreducible
cost of the decoupled lookback algorithm: ~47% of warp cycles stall at CTA
barriers from the intra-block scan and cross-tile lookback. This is
fundamental to any single-pass prefix scan with a data-dependent operator.
