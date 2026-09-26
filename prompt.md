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

Hardware: **NVIDIA A100 (sm_80)**, 108 SMs. Spec DRAM bandwidth depends on the
model (40 GB HBM2: ~1.55 TB/s; 80 GB HBM2e: ~2 TB/s); which model the cluster
node has is not recorded here.

All timings are mean over 100 runs with 500 warmup iterations, reported as
`μs` with 95% CI and effective GB/s (input bytes read + output bytes written).

**Bug (fixed): benchmark statistics and byte counts.**
- *What / how it manifested:* the 95% CIs printed by `print_stats`
  (`p1_bench.cu`) and `compute_descriptors` (`common/util.cu.h`) were always
  about ±9.5% of the mean, whatever the run-to-run spread. The "variance"
  was the mean of t² (not E[t²] − mean²), so the "standard deviation" was
  roughly the mean itself, and the multiplier was 0.95 instead of 1.96.
  Separately, in `cuda_lexer.cu` the ladder step "Big tile S1 (load only)"
  was credited with the full lexer's traffic (input + outputs) and reported
  an impossible 3378 GB/s, and the (now removed) older lexer tests counted
  every block's shared-memory table copy (served from L2) as DRAM traffic.
- *How it was identified:* identical ±9.5% intervals on every line
  regardless of variance, and GB/s above the A100's physical bandwidth.
- *Fix:* Bessel-corrected sample variance, 1.96 multiplier, floating-point
  byte factor; every line counts only the bytes that variant moves (S1:
  input only). All earlier CIs in this document are the old, meaningless
  ones; run-to-run spread was judged from repeated runs instead (~±5 μs).

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

The BW ceiling (read-only, no scan) achieves 1332 GB/s — ~66% of spec on an
80 GB A100, ~85% on a 40 GB A100. It is the best this load pattern achieved,
not a proven DRAM limit. `p1_transpose` achieves 841 GB/s, roughly **63% of
that measured ceiling**. IPT=22 was confirmed optimal by a sweep over IPT=16/20/24/28/32.

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

**The occupancy improvement is not explained by shmem.** Theoretical occupancy
is capped at 6 blocks/SM (75%) by registers: 40 regs × 256 threads = 10,240
regs/block, and 65,536 / 10,240 = 6. A 7th block cannot fit regardless of
shmem, and 6 blocks × 12.67 KiB ≈ 76 KiB is well under the SM's shmem
capacity, so shmem does not limit either kernel. Both kernels share the same
75% ceiling; the gain from 67.6% to 73.6% *achieved* occupancy comes from
something else (e.g. less time with blocks stalled or idle at barriers, or a
smaller tail effect). The cause has not been determined.

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

## Remaining Headroom

Two barrier-reduction ideas were investigated after `p1_transpose` was
established as the best variant. Both were ruled out, but neither result bounds
P1 throughput — see "What these results do not show" below.

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

Counted `__syncthreads()` occurrences in the CUB source of the two `BlockScan`
specialisations:

| Algorithm | `__syncthreads()` in source |
|---|---|
| `BLOCK_SCAN_WARP_SCANS` | 3 |
| `BLOCK_SCAN_RAKING` / `RAKING_MEMOIZE` | 16 |

These are source occurrences, not barriers executed per tile (some sit in
alternative code paths). Raking was not benchmarked; `WARP_SCANS` is kept
because it is very likely the lower-barrier option.

### What these results do not show

The investigations above show that **no barrier in `p1_transpose` can be
deleted**. They do not show that the barrier *stall* is minimal, or that
841 GB/s is a ceiling:

- **Barrier stall measures waiting, not barrier count.** In decoupled lookback,
  one warp performs the lookback while the other warps of the block wait at the
  next `__syncthreads()`. The 47% CTA-barrier stall therefore largely reflects
  **lookback latency** exposed through a barrier. Reducing that latency, or
  overlapping it with useful work, would reduce the stall without removing any
  barrier.
- **Decoupled lookback is not inherently this far from bandwidth.** CUB's
  `DeviceScan` gets close to memcpy throughput for primitive operators. The gap
  here is specific to this kernel/operator, not a property of single-pass scans.

### Traffic floor

P1 moves 3 bytes per input byte (1 byte read, 2-byte `state_t` written).
For the 500 MiB input:

| | Bytes moved | Time at 1332 GB/s |
|---|---|---|
| P1 (single pass) | 1573 MB | ~1180 μs |
| Reduce-then-scan (input read twice) | 2097 MB | ~1574 μs |
| `p1_transpose` measured | 1573 MB | **1870 μs** |

(1332 GB/s is the measured read-only ceiling, not the A100's spec bandwidth.)

A two-kernel reduce-then-scan has a traffic floor **below** the current
measured time, so it cannot be dismissed on bandwidth grounds. Whether it wins
depends on how close each of its passes gets to the ceiling; it has not been
implemented.

## Speed-of-Light Investigation (A100)

Goal: a P1 that runs at the speed of light (SoL) for its own traffic
(1 B read + 2 B written per input byte). Timings are means over 100 runs; the
95% CIs printed by `print_stats` are not valid (it uses sqrt(E[t²]) as the
standard deviation), so run-to-run spread was judged from repeated runs
(`p1_transpose`: 1852–1894 μs across five runs).

### Speed-of-light references

| Kernel | Traffic/byte | Time |
|---|---|---|
| memcpy D2D | 2 B | 773 μs (1356 GB/s) |
| map-only, u8 out | 2 B | 779 μs |
| map-only, 4-bit out | 1.5 B | 594 μs |
| map-only, u16 out, coalesced (L0) | 3 B | 1257 μs |

memcpy at 1356 GB/s is ~87% of a 40 GB A100's spec bandwidth. The u16 floor
at that bandwidth is ~1159 μs. Smaller outputs only move the floor; they do
not remove the scan overhead, so all work below stays at u16.

### Cost ladder

Each step adds one piece of `p1_transpose` (all u16 out, BS=256, IPT=22):

| Step | Kernel | Time | Added |
|---|---|---|---|
| L0 | map-only, coalesced | 1257 μs | – |
| L1 | + warp-transpose load/store | 1239 μs | 0 |
| L2 | + block scan, no lookback | 1524 μs | +285 μs |
| L3 | + decoupled lookback (`p1_transpose`) | 1877–1894 μs | +360 μs |

L1 is the practical u16 SoL (94% of the floor). The whole gap is the scan:
~45% block scan, ~55% lookback.

### Profile of the ladder (ncu, SM clock locked at 765 MHz)

- **Block scan (L1→L2):** +227M warp instructions (~14 thread instructions
  and ~2.5 shmem loads per element), LSU 73% busy, L1/shmem pipe 84% — the
  block scan is throughput-bound on shmem/LSU. The `compose` line shows
  mostly short-scoreboard stalls, but removing that dependency (column
  compose, below) made it slower.
- **Lookback (L2→L3):** only +52M instructions but +564K cycles; 24.8% of all
  stall samples sit on the barrier where 7 of 8 warps wait for warp 0's
  lookback (`block_scan_warp_scans.cuh:429`).

### Lookback statistics (instrumented `p1_transpose`, BS=256)

Two runs (the second with finer depth buckets and a window counter):

- tile-1 at the first poll: INVALID 25–28%, PARTIAL 72–75%, INCLUSIVE 0.0%
- re-polls per tile: 1.30–1.47
- depth to the nearest INCLUSIVE tile: 63–69 tiles on average; 89% of tiles
  walk back ≥ 64 tiles
- 32-tile windows walked per tile: 3.00 (89% walk 3 or more)

The depth is a fixed **time lag**, not a fixed tile count. Dividing depth by
the tile completion rate gives the time between a tile publishing PARTIAL and
publishing INCLUSIVE — about the duration of one lookback:

| Config | Tiles/μs | Depth | Lag |
|---|---|---|---|
| BS=256 | 49.6 | 69 | 1.40 μs |
| BS=512 | 23.7 | 32 | 1.36 μs |
| BS=256, 64-tile window | 39.4 | 68 | 1.73 μs |

So changing how many tiles or windows the lookback walks does not shorten it.
The cost is that 7 of 8 warps of every block sit idle for ~1.4 μs per tile.

### Vectorized load/store (`p1_vec`)

The block scan was bound by load/store-unit (MIO) throughput, and ~7 of ~9
memory-pipe instructions per element came from the byte-granular load/store
path (see "Memory-pipe instructions per element"). `p1_vec` keeps the same
blocked-register scan (CUB `BlockScan`, same lookback) but moves 8-byte input
vectors and 16-byte output vectors:

- load: 3 × `LDG.64` per lane (256 contiguous bytes per warp instruction) →
  `STS.64` into a per-warp buffer → `LDS.64` of the lane's own 24 bytes →
  `to_state[]`
- store: states packed 8 per 16 bytes → `STS.128` (blocked) → `LDS.128`
  (striped) → `STG.128` (512 contiguous bytes per warp instruction)
- the exchanges are warp-local (`__syncwarp` only), removing
  `p1_transpose`'s two block barriers around load and store; at IPT=24 all
  four shmem patterns are bank-conflict-free (lane strides of 6 / 12 words)

~3.75 memory-pipe instructions per element instead of ~9.

| Step | L (IPT=22) | V (IPT=24) |
|---|---|---|
| 1: load/store | 1237 μs | 1256 μs |
| 2: + block scan | 1523 μs (+286) | 1245 μs (−11) |
| 3: + lookback | 1876 μs (+353) | **1530 μs** (+285) |

The block scan is now completely hidden behind memory time (V2 ≈ V1): with
the load/store path no longer saturating the MIO queue, the scan's two
compose lookups per element fit in the shadow of the loads. The lookback is
the only remaining cost. (V uses IPT=24 vs L's 22; the earlier IPT sweep had
24 slightly *worse* than 22 on the old path, so the gain is the
vectorization.)

### Persistent kernel with `cp.async` prefetch (`p1_vec_pipe`)

In `p1_vec` a block issues no memory requests while it waits in the
lookback. `p1_vec_pipe` makes V3 persistent — grid = resident blocks/SM ×
SMs (6 × 108 = 648), block `b` processes tiles `b, b+648, …` — and before
working on a tile, each warp issues `cp.async` 16-byte copies
(`LDGSTS.E.BYPASS.128`: global → shared, no registers, no L1) of its share
of the block's *next* tile into a second input buffer. Those loads are in
flight during the current tile's scan, lookback and store.

- 25.4 KB shmem/block (2 input buffers + output buffer), so 6 blocks/SM need
  the maximum shared-memory carveout; 40 registers, no spills.
- All blocks must be co-resident: a tile's lookback spins on its
  predecessors, so every block owning an earlier tile has to be running.
  The grid is sized from the occupancy API to guarantee this.
- One extra `__syncthreads()` per tile (BlockScan temp storage is reused).

| | V2 | V3 | V3 persistent + cp.async |
|---|---|---|---|
| time | 1244 μs | 1530 μs | **1488 μs** |
| lookback cost (vs V2) | – | 286 μs | 244 μs |
| windows / re-polls / depth | – | 2.92 / 1.13 / 68 | 3.49 / 1.55 / 85 |

Each lookback got *longer* (tiles finish faster, so more are in flight during
the ~1.4 μs lag), but it costs less because memory stays busy meanwhile.

## What We Tried and Why It Didn't Help

### Removing the lookback sleeps (sleep modes)
Variants of `p1_transpose` that (1) skipped the initial
200–550 ns sleep before the first poll, or (2) also shortened the between-poll
sleep from 350 ns to 32 ns. Result: 2019–2025 μs for both vs 1877–1894 μs for
the baseline in the same runs (+125–148 μs).
Why: the initial sleep is a well-tuned wait, not overhead. Without it the
first poll usually finds tile-1 not yet published and the retry lands later
than the tuned delay would have. The poll interval made no difference. The
sleep samples in the profile were time spent waiting for the predecessor.

### Column compose (`p1_column`)
Replaced the two in-thread compose chains with ALU-only lookups: for fixed
x, compose(a, x) over the 12 states packs into a u64 "column", so
acc = (col[x] >> 4*acc) & 15 and the shmem load address depends only on the
input. Result: no-lookback 1760–1800 μs vs 1524 μs (+~270 μs); with lookback
2052 μs vs 1877 μs. Why: the block scan is bound by shmem/LSU throughput,
not latency. Column compose issues more shmem loads per element (2 × 8-byte
column loads + an index→state load vs 2 compose loads), so it lost despite
removing the dependent chain. (Only 5 distinct input columns exist for this
DFA, so a register-resident variant is possible but DFA-specific.)

### Skipping the sleep before later lookback windows
Kept the tuned first-window sleep but dropped the 350 ns sleep before
windows 2, 3, …. Result: 1878 μs vs 1877 μs — no effect. Why: that sleep
overlaps time the warp would otherwise spend waiting for INVALID
predecessors to publish.

### 64-tile lookback window (2 predecessor tiles per lane)
Each lane checked 2 tile descriptors so one window covers 64 tiles and the
typical depth (~69) needs fewer round trips. Result: 2363 μs vs 1877 μs
(+486 μs); 2426 μs combined with no later-window sleep. Windows per tile
dropped from 3.00 to 1.83, but re-polls rose from 1.30 to 2.72. Why: the warp
re-polls until *every* descriptor in its window is past INVALID; a window
twice as wide more often contains a slow predecessor, so it waits longer.

### Larger blocks (BS=512 / BS=1024, IPT=22)
Halves / quarters the number of tiles to cut the number of lookbacks.

| BS | L3 | lookback (L3−L2) | block scan (L2−L1) |
|---|---|---|---|
| 256 | 1894 μs | 370 μs | 285 μs |
| 512 | 1962 μs | 320 μs | 349 μs |
| 1024 | 2827 μs | 308 μs | 794 μs |

Why it failed: lookback cost barely depends on the tile count (halving the
tiles cut it by 14%) — the cost is how long each lookback takes, not how many
there are. Larger blocks make the block scan more expensive (more warps per
barrier; BS=1024 also drops to 50% occupancy at 63 registers).

### More resident blocks (8 blocks/SM), carveout, register packing
At 40 registers the SM holds at most 48 warps whatever the block shape, so
more resident warps need fewer registers. `__launch_bounds__(256, 8)` caps
registers at 32 (8 blocks/SM, 98% achieved occupancy), with the shared memory
carveout raised so 8 × 13 KB fits.

| Variant | L2 (block scan) | L3 | lookback (L3−L2) |
|---|---|---|---|
| 6 blocks/SM, 40 regs (baseline) | 1524–1538 μs | 1874–1878 μs | ~340–350 μs |
| 8 blocks/SM, max carveout (~28 KB L1) | 1701–1706 μs | 1941–1942 μs | 241 μs |
| 8 blocks/SM, 132 KB carveout (~60 KB L1) | 1697 μs | 1942 μs | 245 μs |
| register-packed, 6 blocks/SM | 1535 μs | 1890 μs | 355 μs |
| register-packed, 8 blocks/SM, 132 KB | 1741 μs | 1919 μs | 178 μs |

(Register-packed: the in-thread scan loops hold two u16 states per 32-bit
register; BlockScan scans only the per-thread aggregates.)

Why it failed: more resident warps do hide the lookback (its cost fell from
~345 μs to 178–245 μs, even though each lookback took longer: 1.84 μs vs
1.30 μs lag), but the block scan got slower by ~170–220 μs. At 32 registers
the kernels spill (~5–6 local loads/stores per warp per tile, inside CUB's
BlockScan); the ncu profile of L2 at 8 blocks shows +12.5M instructions,
doubled L2-cache reads from spill loads missing L1, and higher MIO-throttle
stalls. The carveout size made no difference, and register packing did not
remove the spills (they are in the block-scan phase, not in the state
array). More warps help the latency-bound lookback but hurt the block scan,
which is bound by load/store-unit (MIO) throughput.

### Memory-pipe instructions per element (why the block scan is LSU-bound)
Counted from the full-tile SASS path of L2 (BS=256, IPT=22):

| Step | per element |
|---|---|
| `LDG.U8` byte load | 1 |
| `to_state` lookup (LDS) | 1 |
| load warp-transpose (STS + LDS) | 2 |
| compose, reduce + scan (LDS) | 2 |
| store warp-transpose (STS + LDS) | 2 |
| `STG.U16` state store | 1 |

~9 memory-pipe instructions per element, of which the scan contributes 2. L1
(load/store only) hides the other 7 behind DRAM time; adding the scan's 2
tips the MIO queue into throttling. Every change that added LSU work
(column compose, spills at 8 blocks/SM) made P1 slower.

### `p1_vec` at 8 blocks/SM and at IPT=16 / IPT=32

With the load/store unit no longer saturated, more occupancy and other tile
sizes were retried on V3:

| Variant | V2 (block scan) | V3 | lookback (V3−V2) |
|---|---|---|---|
| IPT=24, 6 blocks/SM (best) | 1245 μs | 1527–1530 μs | ~282 μs |
| IPT=24, 8 blocks/SM (32 regs, spills) | – | 1698 μs | – |
| IPT=16 (128K tiles, 36 regs) | 1229 μs | 1935 μs | 706 μs |
| IPT=32 (64K tiles, spills, 2–4-way conflicts) | 1392 μs | 1676 μs | 284 μs |

Why they failed:
- 8 blocks/SM: spills (40 B stores / 68 B loads) and more resident blocks
  slowed each tile's publication — re-polls per tile rose from 1.13 to 2.34
  and INVALID first polls from 23% to 37%.
- IPT=16: 1.5× the tiles but 2.5× the lookback cost. Once a tile's work is
  short relative to a ~1.4 μs lookback, the idle time dominates.
- IPT=32: the lookback costs the same as at IPT=24, while spills and bank
  conflicts bring the block-scan cost back (+136 μs over V1). Even with those
  fixed it would land at about the IPT=24 time.

IPT=24 at 6 blocks/SM is the sweet spot. V3's lookback behaves like L3's
(~3 windows per tile, depth ~68, re-polls 1.13).

### Bug: persistent kernel deadlock from named-barrier usage (`p1_vec_lbwarp`)

**What:** the first A100 run of `p1_vec_lbwarp` (dedicated lookback warp,
persistent grid) hung; it passed every check locally.

**How it manifested:** `make bench_p1` stopped after the
"V3 persistent + cp.async" line and never finished (the next kernel's name
was not printed because stdout was not flushed before running it).

**How it was identified:**
1. `compute-sanitizer --tool synccheck` on a debug build: 0 errors, so the
   barrier protocol itself was correct.
2. `ptxas -v` reported `used 16 barriers` for the kernel: the named barrier
   ids were passed as registers (`bar.sync %0, %1` with `"r"(id)`), so ptxas
   could not tell which ids are used and reserved all 16 per block.
3. `cuda_occupancy.h`: sm_80 has 2 × 32 = 64 barriers per SM, so at 16 per
   block only 4 blocks fit — but `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
   only applies the barrier limit for compute capability ≥ 9.0 and reported
   5. The persistent grid (5 × 108 = 540 blocks) therefore had 108 blocks
   that could never become resident; tiles waiting on their predecessors
   spun forever. Locally (sm_75: 32 barriers/SM, 2 blocks of 16) every block
   was resident, so it passed.

**How it was solved:** (the kernel was later removed — see "Dedicated lookback
warp") barrier ids are compile-time immediates
(`named_bar_sync<ID, THREADS>()`, parity selected with a branch), so ptxas
reserves only ids 0–5 (`used 6 barriers`; 64 / 6 = 10 blocks/SM, no limit
at 5). `bench` now flushes stdout before each kernel so a hang is
attributable.

**Trade-offs:** the parity choice becomes a (block-uniform) branch instead
of an id computed in a register. Persistent grids still rely on the
occupancy API, which does not model barrier limits on sm_80 — any kernel
that uses more named barriers must be checked against 64 / barriers-used by
hand. **Alternative:** dynamic tile tickets (`atomicAdd` for the next tile)
would make the persistent kernels safe even if not all blocks are resident,
at the cost of an atomic per tile and broadcasting the tile index.

### Lookback cost with an integer-add operator; first-poll sleep sweep

To separate the lookback *protocol* from the compose operator, `p1_vec_pipe`
was also run with 16-bit integer addition (as in the decoupled look-back
paper; output checked against a host running sum), and with the first-poll
sleep base (200 ns + 50 × (tile % 8)) varied:

| Variant | Time | lookback cost (vs V2 1247 μs) |
|---|---|---|
| `p1_vec_pipe`, compose, sleep 200+ | 1490 μs | 243 μs |
| sleep 100+ / 400+ / 600+ / 900+ | 1491 / 1489 / 1490 / 1486 μs | no change |
| **integer add** | **1357 μs (1159 GB/s)** | **110 μs** |

- **Compose costs ~133 μs of the lookback.** Not in the block scan (V2 ≈ V1
  showed that is free) but inside the lookback: each 32-tile window is
  reduced with `TailSegmentedReduce` — 5 dependent shuffle + compose steps,
  each compose a shared-memory table load — and tiles walk ~3 windows. The
  add version re-polls *more* (1.83 vs 1.48 per tile) yet is faster, so the
  cost is the dependent compose chain, not waiting.
- **Sleep tuning is closed:** 100–900 ns bases make no difference. (The
  100/400/600 variants even had near-identical lookback stats; `__nanosleep`
  only guarantees a sleep in [0, 2t], so these settings may not really
  differ.) Removing the sleep entirely did hurt earlier (+142 μs, L3).

At 1159 GB/s (85% of memcpy) the add version is roughly what the decoupled
look-back protocol achieves here; the gap to it is the compose chain.

### Pipelined two-window lookback

To shorten the compose chain inside the lookback, a pipelined variant loaded
two 32-tile windows per round trip, prefetched the next pair while reducing,
reduced both windows in one interleaved shuffle/compose loop, and waited only
on INVALID lanes up to a window's first INCLUSIVE/OOB lane.

| | `p1_vec_pipe` | pipelined | `p1_vec_pipe`, add | pipelined, add |
|---|---|---|---|---|
| time | 1490 μs | 1616 μs | 1351 μs | 1578 μs |
| re-polls per tile | 1.47 | 2.65 | 1.83 | 2.93 |
| windows / depth | 3.25 / 78 | 4.02 / 106 | 3.06 / 72 | 3.86 / 101 |

Why it failed: the speculatively loaded windows are **stale**. Window B (and
the next pair) is loaded together with window A; by the time A turns out to
have no INCLUSIVE tile, B's data is older than a fresh load after the usual
350 ns wait would be, so it more often still shows INVALID lanes, each costing
a re-poll (350 ns sleep + round trip). The longer lookback deepened the band
(depth 106), adding windows. It lost with the add operator too, so the
protocol, not compose, was the problem. (A synthetic unit test of the
lookback — 216 cases, mutation-checked — passed; the variant was correct.)

### Comparison with the decoupled look-back paper

Merrill & Garland, *Single-pass Parallel Prefix Scan with Decoupled
Look-back* (NVIDIA NVR-2016-002; copy in `.claude-artifacts/`). Their ceiling
is memcpy ("an ideal performance ceiling for prefix scan because it shares
the same minimum I/O workload"); for 32-bit integer prefix sum CUB matches it
on K40/M40 (26.7 vs 26.8 and 30.8 vs 31.0 G items/s) and reaches ~91% on the
C2050 — 80% of that card's stated 144 GB/s theoretical bandwidth.

We already use the paper's key techniques (parallel warp-wide look-back
window, fence-free combined status/value word, aggregate/prefix protocol).
Two differences explain the larger gap here:

1. **Operator.** Addition needs no memory access; compose is a shared-memory
   table lookup on the look-back's dependent reduction chain. Same kernel:
   add 1351 μs (86% of memcpy) vs compose 1490 μs (78%). The add variant is
   the realistic upper bound for this design, not memcpy parity.
2. **Tile rate.** Each look-back costs a roughly fixed latency, so the number
   of tiles per second matters. The paper's setting (M40, 8 B per 32-bit item,
   2048-item tiles in their example) is ~15M tiles/s; P1 on the A100 (3 B per
   item, 6144-item tiles, at memcpy speed) would be ~74M tiles/s — ~5× less
   time per tile to hide the same latency. The paper notes that when
   signalling latency limits throughput, larger partitions are the remedy.

| | % of theoretical (1555 GB/s) | % of memcpy (1356 GB/s) |
|---|---|---|
| `p1_vec_pipe`, compose (1057 GB/s) | 68% | 78% |
| `p1_vec_pipe`, integer add (1164 GB/s) | 75% | 86% |

### Larger tiles for V3: 512 threads per block

Motivated by the tile-rate comparison with the paper: V1–V3 at 512 threads
per block (12,288-item tiles, 3 blocks/SM = same 48 warps, half the tiles).

| | V2 | V3 | lookback cost | windows | depth (tiles) | re-polls |
|---|---|---|---|---|---|---|
| 256 threads (6/SM) | 1246 μs | 1529 μs | 283 μs | 2.93 | 68 | 1.13 |
| **512 threads (3/SM)** | 1242 μs | **1491 μs** | **249 μs** | **1.87** | **35** | 1.06 |

The block scan stays free at 512 threads (V2 unchanged). The look-back lag in
*time* stays about the same, so with twice the items per tile the depth in
tiles halves and most look-backs walk 2 windows instead of 3. V3 at 512
threads (1055 GB/s, 78% of memcpy, 91% of the integer-add ceiling) matches
the persistent `cp.async` kernel at 256 threads by a different mechanism.

### 768 threads per block; persistent `cp.async` at 512 threads

| Variant | V2 | V3 | lookback cost | windows | depth | re-polls |
|---|---|---|---|---|---|---|
| V, 512 threads (3/SM) | 1242 μs | **1491 μs** | 249 μs | 1.87 | 35 | 1.07 |
| V, 768 threads (2/SM) | 1264 μs | 1528 μs | 264 μs | 1.26 | 21 | 0.67 |
| persistent + `cp.async`, 256 threads | – | **1491 μs** | ~244 μs | 3.25 | 78 | 1.48 |
| persistent + `cp.async`, 512 threads | – | 1515 μs | – | 2.24 | 47 | 1.60 |

- **768 threads:** the shortest look-back walks (1.26 windows) but slower.
  With 2 blocks/SM a block waiting in its look-back idles half the SM (a
  third at 512 threads), and the block scan starts to cost (+22 μs on V2).
  512 threads balances fewer look-backs against idle fraction.
- **Persistent + 512 threads** (buffers in dynamic shared memory, 48 KB per
  block): the two gains do not add up — both hide the same look-back idle
  time, and re-polls/depth rose again (1.60 / 47 vs 1.07 / 35).

### Dedicated lookback warp (`p1_vec_lbwarp`)

Persistent, `cp.async` prefetch, 288 threads = 8 compute warps + 1 lookback
warp. The compute warps ran a tile-local scan (custom block scan on a
compute-only named barrier), published PARTIAL and handed the aggregate to
the lookback warp (`bar.arrive`), then fixed up and stored the *previous*
tile once its prefix arrived (`bar.sync`, out = compose(P, local) from a
shared-memory pending buffer). The lookback warp ran each tile's lookback
while the compute warps worked on the next tile.

| | `p1_vec_pipe` (6 blocks/SM) | `p1_vec_lbwarp` (5 blocks/SM) |
|---|---|---|
| time | 1492 μs | 1837 μs (+345) |
| INVALID first polls | 31% | 9% |
| re-polls / windows / depth | 1.82 / 3.47 / 85 | 2.23 / 4.07 / 105 |

Why it failed: PARTIAL is published before the lookback starts, so fewer
first polls meet INVALID — but every tile's INCLUSIVE is published later
(when the lookback warp gets to it), so the band of PARTIAL-only tiles grew
(depth 105, >4 windows) and each lookback got longer. On the cost side:
one block fewer per SM (40 compute warps instead of 48; 288 threads × 40
registers), 56 B of spills (the current tile's 24 states stay live during the
previous tile's fix-up), and one extra compose lookup plus a pending-buffer
round trip per element. Which of these dominates was not profiled.

### 4-chain reduce and deferred lookback (on top of `p1_vec_pipe`)

| Variant | Time | vs pipe (1491 μs) |
|---|---|---|
| 4-chain reduce (per-thread aggregate from 4 interleaved chains, depth 9 instead of 24) | 1482 μs | −9 μs (within run-to-run spread) |
| deferred lookback + fix-up, no pre-poll sleeps | 1791 μs | +300 μs |
| deferred lookback + fix-up, with pre-poll sleeps | 1584 μs | +93 μs |

- **4-chain reduce:** publishing PARTIAL earlier did not reduce waiting —
  INVALID first polls stayed ~30% and re-polls rose (1.61 → 1.80). Too small
  to keep the extra code.
- **Deferred lookback** (`p1_vec_defer`): each round a block first finished
  its previous tile (lookback, INCLUSIVE, out = compose(P, local) from a
  shared-memory pending buffer, store), then scanned the current tile
  locally. The assumption was that the previous tile's predecessors would
  have published a round earlier. That held locally (0.9% INVALID first
  polls on the GTX 1660 Ti) but not on the A100: 28.9% INVALID — with 648
  resident blocks, neighbouring blocks drift out of phase, so the owner of
  tile p−1 is often still in its previous round. Without pre-poll sleeps the
  walk re-polled 2.04× per tile; the fix-up also adds a compose lookup per
  element and a barrier.

A bug found while testing `p1_vec_defer` (not in any measured code): the
loop condition treated `tile_idx - gridDim.x` as a pending tile even past
`num_tiles`, walking out of bounds (illegal memory access). Fixed before
measuring by also requiring the pending tile to be `< num_tiles`.

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
| Vectorized load/store (`p1_vec`, IPT=24) | 838 GB/s (1876 μs) | 1028 GB/s (1530 μs) | +190 GB/s |
| Larger tiles: 512 threads/block (`p1_vec<512, 24, 3>`) | 1028 GB/s (1529 μs) | 1055 GB/s (1491 μs) | +27 GB/s |

(Persistent + `cp.async` prefetch at 256 threads, `p1_vec_pipe`, reaches the
same 1491 μs by a different mechanism; the two do not stack.)

## Conclusion — P1 is Final

**Final P1: `p1_vec<512, 24, 3>` — 1491 μs / 1055 GB/s**, 20% faster than the
starting point (`p1_transpose`, 1870 μs), at 78% of memcpy and 91% of the
integer-add reference. It is chosen over `p1_vec_pipe` (same time) because
it is simpler: no persistent grid (so no co-residency requirement and none
of the deadlock risk hit in `p1_vec_lbwarp`), no `cp.async`, no shared-memory
carveout. `p1_transpose` (L3), `p1_vec_pipe` and its integer-add variant stay
in `p1_bench.cu` as references.

### Why we stop here

This is not a proof of optimality, but every part of the remaining time is
bounded by a measurement, and every lever we know for the part that is left
has been tried.

1. **Memory floor (~1242 μs).** P1 must read 1 B and write 2 B per element.
   At measured memcpy bandwidth that is 1159 μs; the simplest possible
   kernel with this output (L0, byte → state map with no scan at all) takes
   1257 μs, and V2 (load/store + full block scan) 1242 μs. The block scan is
   already free. Nothing short of changing the output format (u8: 773 μs,
   4-bit: 594 μs floors) can go below ~1242 μs.
2. **Look-back (249 μs = 1491 − 1242) splits into two measured parts:**
   - **Operator, ~135 μs.** The same kernel with integer addition runs in
     1351–1357 μs. Compose is a table lookup on the look-back's dependent
     reduction chain (5 dependent steps per 32-tile window); addition is
     memory-less. Register-based compose alternatives (shuffle lookups,
     select chains) cost a comparable latency per step, so this part is
     inherent to a table-driven DFA operator. This is also why the paper's
     memcpy parity (for integer addition) is not the target here.
   - **Protocol, ~110 μs** (1356 − 1247 with the free operator). P1 needs
     ~5× the paper's tiles per second (1 B in + 2 B out per element), so
     each tile has ~5× less time to hide a look-back of roughly fixed
     latency (L2 round trips).
3. **Every look-back lever was measured:**
   - *Hide it:* more resident warps (register spills on a saturated LSU),
     a dedicated look-back warp (later INCLUSIVE, lower occupancy), a
     deferred look-back (blocks drift out of phase), persistent `cp.async`
     prefetch (**−38 μs**, kept as reference).
   - *Shorten it:* sleep changes (no effect; removing it hurts), wider
     windows (waiting on stragglers), a pipelined two-window look-back
     (stale speculative loads), a faster per-thread reduce (no effect).
   - *Do fewer:* larger tiles (**512 threads: −38 μs**; 768 threads worse —
     idle fraction and block-scan cost), IPT 16/32 (worse).
   - *Combine:* persistent + 512 threads (1515 μs — the two gains hide the
     same idle time and do not stack).

So within the decoupled look-back design on the A100 with u16 output, P1
lies between 1242 μs (no look-back) and 1356 μs (memory-less operator); it
is at 1491 μs, and the remaining ~135 μs is the cost of a table-lookup
operator. Moving further needs a different contract or hardware, not more
tuning:

- a smaller output (u8 or 4-bit states; would require P2 changes),
- an operator whose composition is not a memory lookup (not available for
  this DFA),
- hardware support such as Hopper's distributed shared memory / thread
  block clusters, which could shorten look-back latency (not on the A100).

Two-kernel reduce-then-scan is no option: its traffic floor (~1574 μs) is
above the result. `p1_vec_pipe`'s techniques have also been ported into the
single-pass lexer (`lexerVecPipe` in `cuda_lexer.cu`).

## Single-pass Lexer (`cuda_lexer.cu`)

Speed of light: 1 B read per input byte + 5 B written per token (u32 index +
u8 token). At memcpy bandwidth (1356 GB/s): dense 940 μs (150.1M tokens),
moderate 428 μs (11.3M), sparse 391 μs (1.2M). At 6144-byte tiles that is
90–218M tiles/s — 3–8× P1's tile rate — with two look-backs (state, index)
per tile. The two look-backs cannot simply be merged: a tile's token count
depends on its incoming state.

### `lexerBig` (large tiles)

Tile = 256 threads × 96 bytes (24 KB) kept in shared memory. Pass A: per-thread
state reduction, block scan + state look-back. Pass B: rescan, produce flags
in registers, tokens written in place over the consumed bytes; block scan of
counts + index look-back. Pass C: coalesced emission (originally: owner
lane by binary search over lane counts, element by k-th set bit; now per-lane
expansion, see below). DFA tables are
still copied from global into shared memory at kernel start.

A100 (μs):

| | transpose | vecPipe | Big S1 (load) | Big S2 (no look-backs) | Big S3 (full) |
|---|---|---|---|---|---|
| dense | 3324 | 3074 | 377 | 3126 | 3183 |
| moderate | 2995 | 2436 | 378 | 1171 | 1258 |
| sparse | 2823 | 2319 | 378 | 1071 | 1132 |

`lexerBig` halves moderate/sparse vs `lexerVecPipe`; look-backs now cost only
60–90 μs. Dense was no better.

**Performance issue: `__fns` in the emission.** ncu (dense, S2): 1418M warp
instructions (~87 per input byte), issue slots 78.5% busy, almost no stalls —
instruction-bound. `__fns` (k-th set bit) was a subroutine call (`CALL` in
SASS) costing ~676M warp instructions (45% of the kernel), once per output
token — hence dense (150M tokens) was slow and sparse was not. Replaced by
`select_bit`: word chosen with two `popc`, then a 5-step branchless `popc`
binary search (~25 instructions, no call; checked against a naive select on
~32M (mask, k) pairs). A100 S3: dense 3183 → 2055 μs, moderate 1258 → 1179,
sparse 1132 → 1119 (S2: 2002 / 1118 / 1065).

Remaining cost after the fix (S2 − S1): passes A/B ≈ 690 μs on every dataset
(instruction-bound: two dependent shared lookups plus index extraction per
byte, twice), emission ≈ 935 μs extra on dense (the per-slot owner search
and select, ~40 instructions per token), look-backs ≈ 55 μs.

### Chain tables (kept) and per-lane staged emission (failed)

Two changes were tried together (`10c634b`), were slower together, and were
then split with template flags (`635e6a0`) to measure each on the A100 and
with ncu. Result: keep the chain tables, drop the staged emission.

**Passes A/B — derived chain tables (kept).** At kernel start each block
builds two tables in shared memory from the DFA tables it loaded (so the DFA
stays runtime data):
- `row_of[256]` (u16): byte offset of the compose row of `to_state[b]`;
- `comp[144]` (u16): compose results packed as index·2 (bits 1–4), produce
  (bit 5), token (bits 6–8), accept (bit 9).

One chain step is `v = comp[row_of[b] + (v & 0x1f)]` (byte addressing): an
AND, an add and two shared loads, of which only the `comp` load is on the
dependency chain, instead of `to_state` load → index extraction → multiply-add
→ `compose` load. Produce/token/accept are bit tests on `v`. The block scan
and the look-back still use `ShmemCompose` on the plain state index
(`(v & 0x1f) >> 1`). 40 registers, 25.8 KB shared memory, 6 blocks/SM.

**Emission — per-lane staged expansion (failed, removed).** Each lane walked
its own produce mask (`__ffs`, clear lowest bit) and wrote element positions
into a 256-slot per-warp u16 staging area, in rounds of 256 warp-local slots;
the warp then copied each round out coalesced. It replaced the owner binary
search + `select_bit` per output slot. The staging area added 4 KB of shared
memory per block → 5 blocks/SM (launch bounds `(256, 5)`, 48 registers).

A100 split, μs (compose + select = `7b34870`):

| S3 | compose + select | chain + select | compose + staged | chain + staged |
|---|---|---|---|---|
| dense | 2057 | **1959** | 2260 | 2185 |
| moderate | 1178 | **1144** | 1260 | 1227 |
| sparse | 1123 | **1086** | 1178 | 1132 |

| S2 | compose + select | chain + select | compose + staged | chain + staged |
|---|---|---|---|---|
| dense | 1996 | 1918 | 2169 | 2079 |
| moderate | 1119 | 1103 | 1158 | 1108 |
| sparse | 1065 | 1053 | 1089 | 1054 |

S3 − S2 (look-back cost): select variants 33–61 μs, staged variants 78–119 μs.

ncu, dense (clocks locked at 765 MHz):

| | blocks/SM | warps active | warp instr. (S2) | S2 | S3 − S2 |
|---|---|---|---|---|---|
| compose + select | 6 | 73% | 843M | 3.23 ms | 0.092 ms |
| chain + select | 6 | 73% | 782M | 3.03 ms | 0.090 ms |
| compose + staged | 5 | 59% | 1033M | 3.64 ms | 0.167 ms |
| chain + staged | 5 | 59% | 976M | 3.47 ms | 0.190 ms |

Causes:
1. *Chain tables*: −61M warp instructions (−7%), all datasets faster; the
   look-backs also got ~20 μs cheaper on the A100.
2. *Staged emission executes more instructions, not fewer* (+190M, +22%).
   Per source line, the expansion loop costs ~631M warp instructions vs
   ~520M for the owner search + `select_bit` + shuffles. 443M of it is the
   three-way `if (m0) / else if (m1) / else` word selection: lanes whose next
   token is in different mask words diverge, so the warp runs the paths one
   after another; and the loop runs as many iterations as the busiest lane
   has tokens, not the average. My estimate (~12 instructions per token)
   ignored both.
3. *5 blocks/SM*: warps active 73% → 59%, and the look-back cost roughly
   doubles (fewer tiles in flight to hide predecessor waits). This is why
   staged also lost on sparse, which has almost no emission work.

Alternatives not pursued: a branchless 96-bit lowest-set-bit (removes the
divergent word selection but not the busiest-lane iteration count) would still
pay the 5-blocks/SM occupancy and look-back cost; staging in the consumed
input buffer instead of a separate area would keep 6 blocks/SM but needs the
buffer free before the whole block's tokens are read.

**Current `lexerBig` (chain + select), A100 S3:** dense 1959 μs (48% of
speed of light), moderate 1144 μs (37%), sparse 1086 μs (36%). Its SASS is
identical to the benchmarked chain + select variant (S2, S3; S1 differs only
by the table setup). Local: debug tests and all three datasets pass S3.

### Pass B: independent lookups (kept) and packed flags

Profile of chain + select (dense S2): emission ~50% of warp instructions
(~43% of stall samples), passes A/B 33% (37%). A chain step costs ~5.8
instructions per byte; pass B added ~4.4 per byte for the produce test,
`set_bit` and token packing (~17.5 per byte for A + B).

Three pass B variants were measured (`db3621e`, template parameter `PASSB`):
- **0**: chain rescan, per-byte produce test, tokens written in place.
- **1 — packed flags**: chain rescan, but the state byte (low byte of the
  chain state) is written in place, and produce flags are gathered 4 bytes at
  a time: `((w & 0x01010101) * 0x10204080) >> 28` (checked exhaustively
  against a per-byte loop on 20M random words). Element mask = produce mask
  of states shifted by one; the emission extracts the token as `byte >> 5`.
- **2 — independent lookups** (kept): as 1, and pass A also writes each
  prefix function F_i (state byte) in place; pass B computes state i as
  `compose(prefix, F_i)` from `comp_pf[p * 16 + f]` (192 B, u8 state bytes),
  four at a time: `((w >> 1) & 0x0f0f0f0f) | prefix * 16 * 0x01010101` gives
  four table indices, then `PRMT` + `LDS.U8` per byte — no serial chain in
  pass B. Needs one more `__syncthreads` (the next thread's first byte is
  read before pass A overwrites it), and in partial tiles the elements from
  bytes past the valid input are masked off.

**Requirement on the DFA (variant 2):** every state index must have a single
full state value in the compose table (flags a function of the index);
otherwise `compose(p, F_i)` can differ in its produce/token flags from the
chain's state i. Checked for this DFA: 12 indices, 12 distinct values, and
the table is associative on full values. A DFA loaded at run time must
satisfy this (it holds when the states are the endomorphisms with fixed
flags, as generated here).

To make the state byte complete, the chain state layout changed to produce
bit 0, index·2 bits 1–4, token bits 5–7, accept bit 8 (step mask `0x1e`).
All variants: 40 registers, 25.8–26.0 KB shared memory (6 blocks/SM).

A100, μs (variant 0 ≈ chain + select: 1949 / 1143 / 1088 vs 1959 / 1144 / 1086):

| S3 | 0 | 1 | 2 |
|---|---|---|---|
| dense | 1949 | 1892 (−57) | **1844 (−105)** |
| moderate | 1143 | 1115 (−28) | **986 (−157)** |
| sparse | 1088 | 1056 (−32) | **925 (−163)** |

| S2 | 0 | 1 | 2 |
|---|---|---|---|
| dense | 1900 | 1886 (CI ±11) | 1808 (CI ±14) |
| moderate | 1088 | 1063 | 920 |
| sparse | 1043 | 1021 | 872 |

ncu (dense S2, clocks locked at 765 MHz): warp instructions 775M / 738M /
700M (variant 2: −10%), duration 2.99 / 2.93 / 2.76 ms; warps active ~73% and
the stall mix unchanged across variants. Per source line, variant 0 → 2:
pass B ~185M → 82M instructions (~5 per byte; the shared-memory latency of
the serial chain is gone), pass A ~100M → 114M (~7 per byte: storing F_i
costs ~1.2 per byte), emission unchanged at ~540M incl. shuffles (now ~2/3
of the instructions and ~47% of stall samples on dense). The extra
`__syncthreads` shows as the load + next-byte region rising from 3% to 8% of
stall samples.

Variant 1 helped but is dominated by variant 2 and was removed; variant 0
was removed. **Current `lexerBig` (variant 2), A100 S3:** dense 1844 μs (51%
of speed of light), moderate 986 μs (43%), sparse 925 μs (42%). SASS
identical to the benchmarked variant 2 (S1–S3); debug tests and all three
datasets pass S3 locally.

Next targets: the emission on dense (~2/3 of instructions); pass A's chain
step (5.8 instructions per byte) for moderate/sparse; the extra barrier
(lanes 0-30 can get the next byte by warp shuffle; lane 31 needs the next
warp's first byte, e.g. from global memory, to avoid the race).

### Emission and pass A variants (all failed, removed)

Starting point `be346d9` (dense 1844 μs; emission ~2/3 of instructions on
dense, ~115 warp instructions per 32 output slots). Five candidates were
measured in `d3e61d9`, each as **one change against the base**
(`lexerBig<..., EMIT, FLAGLESS, NOBAR>`), then all removed: the kernel is back
to the `be346d9` code.

1. **emit-words-dense** (`EMIT 1`): CHUNK = 96 = 3 × 32, so each 32-bit
   produce-mask word covers 32 consecutive elements. The warp walked the
   owner lanes with tokens (ballot, `__ffs`) and their non-zero mask words
   (one shuffle each); lane l took bit l, its slot was
   `base + popc(w & lower_lanes)` — no owner search, no `select_bit`. Used
   when the warp had more than 4 tokens per non-zero word (count by
   `__reduce_add_sync`), otherwise the owner search.
2. **emit-words-all** (`EMIT 2`): word-aligned for every warp.
3. **emit-owner-packed** (`EMIT 3`): owner search on one packed word per lane
   (incl 31–20, count 19–13, popc(m0) 12–7, popc(m0)+popc(m1) 6–0): one
   shuffle for the owner's rank base and word popcounts instead of one
   shuffle and two `POPC`s, at the cost of field extraction.
4. **passA-flagless** (`FLAGLESS`): pass A stepped through `comp_a` (index·2
   only), removing the per-byte `& 0x1e`.
5. **passA-no-barrier** (`NOBAR`): next byte from the next lane by
   `__shfl_down_sync` (lane 31 from global memory); the extra `__syncthreads`
   removed.

A100, S3 μs (change vs base; dense rows have CIs of ±10–17 μs, even base S2
1834 vs S3 1833, so dense differences under ~20 μs are noise):

| | dense (1833) | moderate (985) | sparse (925) |
|---|---|---|---|
| emit-words-dense | +52 | +52 | +2 |
| emit-words-all | +43 | +131 | −4 |
| emit-owner-packed | +63 | −1 | −2 |
| passA-flagless | +19 | −2 | +4 |
| passA-no-barrier | +8 | +1 | +1 |

ncu (dense S2, clocks locked at 765 MHz):

| | warp instr. | duration | global store requests | store sectors to L2 | DRAM write |
|---|---|---|---|---|---|
| base | 700M | 2.76 ms | 9.5M | 31.2M | 0.74 GB |
| emit-words-dense | 659M | 2.59 ms | 32.8M | 53.6M | 0.74 GB |
| emit-words-all | 653M | 2.58 ms | 32.8M | 53.6M | 0.74 GB |
| emit-owner-packed | 716M | 2.87 ms | 9.5M | 31.2M | 0.74 GB |
| passA-flagless | 688M | 2.69 ms | 9.5M | 31.2M | 0.74 GB |
| passA-no-barrier | 701M | 2.76 ms | 9.5M | 31.2M | 0.74 GB |

Causes:
1. *Word-aligned emission*: only −6–7% instructions (estimated ~−35%: the
   second emission estimate of mine that was wrong in direction), and each
   store instruction writes ~9 slots instead of 32 → 3.4× the store requests
   and +72% partially filled store sectors on the L1→L2 path. L2 merges them
   (DRAM writes unchanged), but at full clock the store path limits: slower
   by ~45–50 μs on dense. On moderate the per-warp choice still routed
   warps to the word loop (or cost its reduction), +52 μs; always using it
   costs +131 μs (few tokens per word). Under ncu's locked 765 MHz SM clock
   the same variants were *faster* (instruction issue dominates there).
2. *Packed owner search*: field extraction costs more than the shuffle and
   `POPC`s it removes (+2% instructions).
3. *Flagless pass A*: −1.7% instructions, no measurable time change — pass A
   is bound by the latency of its serial chain, not instruction count.
4. *No barrier*: no change in instructions or time; the barrier was not on
   the critical path.

**Profiling lesson:** ncu locks the SM clock to base by default, which can
invert the ranking of variants that shift work between the SM and the memory
path. `make profile` now passes `--clock-control none` so profiles run at
the real clock (A100: GPU boost clock instead of 765 MHz; durations are then
comparable to the bench).

Current `lexerBig` unchanged (`be346d9` code): A100 S3 dense 1833–1844 μs,
moderate 985, sparse 925.

### Diagnostic: load/compute overlap and the power cap

**Question.** Estimated issue time alone on dense is ~1.15 ms (700M warp
instructions / 432 schedulers at 1.41 GHz), the load alone (S1) 375 μs, and
the bench's S2 (1834 μs) looked like S1 + compute + stores. Do DRAM reads and
compute add up instead of overlapping?

**Test** (`b43e6c3`, removed afterwards): `lexerBig<..., L2IN = true>` made
every block read the input of tile `blockIdx.x % 64` (1.5 MB, stays in L2;
outputs still written to DRAM). The first 64 tiles have the whole file's
token density (tokens per byte: dense 0.2862 vs 0.2864, moderate 0.0219 vs
0.0216, sparse 0.0022 vs 0.0023), so the emission work is representative.

**Result: load and compute overlap; the kernel is compute-bound.** ncu at
the real clock (`--clock-control none`, ~1.40 GHz), dense, single launches:

| | duration | DRAM read | warp instr. | issue active |
|---|---|---|---|---|
| S1 (load only) | 0.366 ms | 524 MB | 12M | 6% |
| S2 | 1.511 ms | 529 MB | 700M | 77% |
| S2, L2-resident input | 1.499 ms | 6 MB | 700M | 78% |
| S3 | 1.575 ms | 527 MB | 709M | 75% |
| S3, L2-resident input | 1.537 ms | 5 MB | 709M | 77% |

Removing the DRAM reads saves only 12 μs (S2) / ~38 μs (S3): prefetching or
pipelining the next tile would not help. Targets remain instruction count and
issue efficiency (pass A, emission).

**Finding: the bench is power-capped.** ncu measured dense S2 at 1.51 ms, the
bench at 1834 μs (1808–1900 μs over recent runs), 18% slower. The bench's
timed region (launch + `cudaDeviceSynchronize` between the events) adds only
a few μs. Logging during the bench on the A100 (PCIe, 250 W limit):

```
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,power.limit,clocks_throttle_reasons.active \
  --format=csv -lms 200 > smi.log & SMI=$!; timeout 1500 make bench; kill $SMI
```

| bench rows (in order) | SM clock | power | software power cap (0x4) |
|---|---|---|---|
| dense transpose, vecPipe | 1335–1395 MHz | ~250 W | yes |
| dense Big S2 / S3 | 1140–1230 MHz | 250–258 W | yes |
| dense Big S2/S3, L2-resident | 1245–1275 MHz | ~255 W | yes |
| moderate transpose | 1410 MHz | 221 W | no |
| moderate Big S2 / S3 | 1320–1395 MHz | ~255 W | yes |
| sparse transpose | 1410 MHz | 219 W | no |
| sparse Big S2 / S3 | 1335–1365 MHz | ~255 W | yes |

(Rows assigned to the ~1.2 s high-power bursts in bench order.) At ~1.2 GHz
instead of ~1.40 GHz, ncu's 1.51 ms becomes ~1.76 ms, most of the 1834 μs.

Consequences:
1. All `lexerBig` rows run at the 250 W cap; the memory-bound transpose
   kernels on moderate/sparse do not. Energy per byte (instructions, shared
   memory traffic) therefore sets the clock: saving instructions pays twice
   (less issue time and a higher clock), most on dense.
2. The dense rows' wide CIs (±10–17 μs) come from the clock varying under the
   cap (1140–1275 MHz within one row), not from the kernel.
3. ncu durations (single launches with idle gaps, no cap) are shorter than
   the bench's sustained numbers; compare variants within one tool only.

### Moderate/sparse are shared-memory bound; shared-memory variants (A100 numbers pending)

`make profile DATA=moderate|sparse` (real clock, `be346d9` kernel), S2:

| | moderate | sparse | dense |
|---|---|---|---|
| duration | 0.853 ms | 0.820 ms | 1.511 ms |
| warp instructions | 291M | 264M | 700M |
| issue active | 57% | 54% | 77% |
| shared-memory pipe (data wavefronts) | 92% | 93% | 74% |
| top stalls | mio_throttle, short_scoreboard | same | — |

Moderate and sparse are bound by the shared-memory pipe, not by issue. Sparse
S2: 109M shared wavefronts, 33.5M (31%) of them from bank conflicts:

| source | wavefronts | from conflicts |
|---|---|---|
| pass A chain step (`row_of[byte]` + `comp[...]`, 2 loads/byte) | 49.0M | 15.9M |
| own-chunk `LDS/STS.128`, stride 96 B (pass A/B read + write, 4 × 6) | 32.8M | 16.4M (all 2-way) |
| pass B `comp_pf` lookups | 16.4M | 0 |
| tile load `STS.128` | 4.1M | 0 |

Three variants, each one change against the base (`lexerBig<..., SWZ, RB8,
PFREG>`, bench rows `Big S2/S3 swizzle | row_of-u8 | comp_pf-regs`):

1. **swizzle** (`SWZ`): vector k of thread t's chunk is stored at slot
   `k ^ ((t >> 2) & 1)`. With stride 96 B, lanes t and t+4 of an 8-lane
   `LDS.128` phase hit the same bank group; flipping the slot's parity for
   lanes 4–7 puts them in the other half (bank units `6t + k` mod 8: lanes
   0–3 even + k, lanes 4–7 even + (k^1)). The coalesced tile load, the partial
   tile load, the next-byte read and the emission's token read use the same
   mapping (byte offset ^ 16 when CHUNK % 32 == 0). +32 SASS instructions.
2. **row_of-u8** (`RB8`): `row_of8[byte] = class * 3` (row offset in 8-byte
   units), step address `comp + (row_of8[b] << 3) + (v & 0x1e)`. A 256-byte
   table maps all bytes < 128 to different banks (no conflicts for ASCII);
   193 `LDS.U16` become `LDS.U8`, no extra instructions.
3. **comp_pf-regs** (`PFREG`): each thread loads its prefix's 16-byte
   `comp_pf` row once (`LDS.128`) and selects pass B's state bytes four at a
   time: nibble selector from `f & 7` (3 ALU), `PRMT` from entries 0–7 and
   8–15, blend mask by `prmt.b32` sign replication of bit 3 of f (inline PTX:
   `__byte_perm` ignores the sign bit of the selector — the first version
   using it mismatched 65511 of 65536 cases), `LOP3` blend: ~10 ALU per 4
   bytes, no shared loads (96 `LDS.U8` per thread removed). Checked
   exhaustively on the GPU against a table lookup (all 16^4 f combinations).

Checks: base SASS byte-identical to `be346d9`; all variants ≤ 40 registers, no
spills, ≤ 26.0 KB shared memory (6 blocks/SM); other kernels unchanged; debug
tests and all three datasets pass S3 for every variant (local, sm_75).

ncu per variant (real clock, average per launch), sparse:

| | S2 | S3 | shared wavefronts | of which conflicts | issue active |
|---|---|---|---|---|---|
| base | 0.820 ms | 0.865 ms | 115.0M | 34.2M | 54% |
| swizzle | 0.710 (−13%) | 0.779 (−10%) | 98.4M | 17.1M | 64% |
| row_of-u8 | 0.720 (−12%) | 0.795 (−8%) | 99.2M | 18.4M | 60% |
| comp_pf-regs | 0.734 (−10%) | 0.808 (−7%) | 99.2M | 34.3M | 57% |

dense:

| | S2 | S3 | warp instr. | shared pipe |
|---|---|---|---|---|
| base | 1.511 ms | 1.576 ms | 700M | 74% |
| swizzle | 1.517 (+0.4%) | 1.585 (+0.6%) | 717M | 66% |
| row_of-u8 | 1.488 (−1.5%) | 1.552 (−1.5%) | 692M | 69% |
| comp_pf-regs | 1.488 (−1.6%) | 1.560 (−1.0%) | 688M | 68% |

Each variant removed the traffic it targeted (~16M wavefronts each on
sparse); they hit different sources. On dense (issue-bound) swizzle costs
+17M instructions (mostly the XOR on each emitted token's read) and the other
two help slightly. Combined rows added: **all three** and **row_of-u8 +
comp_pf-regs** (without swizzle, for dense). Combined variants: ≤ 40
registers, no spills, 25.7 KB shared memory; debug tests and all datasets
pass S3 locally. A100 bench numbers pending.

### state_t width and DFA size

Requirement: the state-specific optimizations must work for `state_t` =
`uint8_t`, `uint16_t` or `uint32_t`.

- **Width-independent already:** the packed chain state, `comp`, `row_of`,
  `comp_pf`, the swizzle, row_of-u8, comp_pf-regs and the 4-at-a-time
  produce flags only read states through `get_index` / `get_token` /
  `is_produce` / `is_accept` and use their own u16/u8 formats.
- **Fixed — table copies:** the compose / to_state tables were copied as
  `uint64_t` with `NUM_STATES * NUM_STATES / 4` and `256 / 4` words (4 states
  per word: 2-byte states only; silently wrong for 1- or 4-byte states) in
  `lexerTranspose`, `lexerVecPipe` and `lexerBig`. Now
  `copy_states_to_shared<N, BLOCK_SIZE>` copies `N * sizeof(state_t)` bytes
  (8 bytes at a time when the size allows, else per state). SASS of all
  existing kernels unchanged for `uint16_t`.
- **Fixed — look-back descriptors:** `TxnWordTraits<uint8_t>` (16-bit
  descriptor: status in bits 7–0, value in 15–8) and a 16-bit
  `st.relaxed.gpu.u16`. Checked with a standalone decoupled look-back scan
  (4M elements, 32768 tiles) for uint8 (add and an order-sensitive
  "last non-zero" operator), uint16 and uint32: no mismatches.
- **Encoding limits → generic path:** the fast path needs at most 16 states
  (index in 4 bits) and at most 8 tokens (3 token bits):
  `LEXER_CHAIN_FITS = NUM_STATES <= 16 && popcount(TOKEN_MASK) <= 3`. A DFA
  that does not fit (whatever the width of `state_t`) compiles `lexerBig`'s
  generic path: passes A and B as plain `compose(s, to_state[b])` chains over
  `state_t` in shared memory, per-byte produce test, tokens stored as bytes;
  look-backs and emission are shared (the swizzle also applies). RB8 and
  PFREG require the fast path (`static_assert`). Before, > 16 states failed
  to compile and > 8 tokens were silently truncated.
- **Tested:** `FORCE_GENERIC` runs the generic path on this DFA: debug tests
  (with and without swizzle) and all three datasets pass S3; bench row
  `Big S3 generic path (forced)` shows its cost. Not tested: a real
  `uint8_t`/`uint32_t` DFA (none available; this DFA needs 9 bits).

### Shared-memory variants: results and adoption (all three kept)

ncu (real clock, `f8682ba`), S3 average per launch, change vs base:

| | dense | moderate | sparse |
|---|---|---|---|
| base | 1.574 ms | 0.917 ms | 0.868 ms |
| swizzle | +0.7% | −9% | −10% |
| row_of-u8 | −1.2% | −7% | −8% |
| comp_pf-regs | −1.1% | −6% | −7% |
| **all three** | −0.9% | **−16%** (0.771) | **−19%** (0.703) |
| row_of-u8 + comp_pf-regs | −2.6% | −13% (0.800) | −15% (0.740) |
| generic path (forced, unswizzled) | +16% | +21% | +22% |

All three together (S2): shared wavefronts sparse 115.0M → 66.5M, bank conflicts
34.2M → 1.4M (moderate 35.3M → 2.0M), shared pipe 93% → 74%, issue 54% → 71%:
moderate/sparse are no longer shared-memory bound. On dense (issue-bound)
the swizzle's extra instructions cancel most of its gain.

**Adopted: all three** (best on moderate/sparse, −0.9% on dense vs −2.6%
without the swizzle). The `SWZ`/`RB8`/`PFREG` flags and the code they
switched off were removed; the swizzle also applies to the generic path.
`row_of8` generalized: 8-byte units when compose rows are a multiple of 8
bytes, else 2-byte units (class · NUM_STATES ≤ 240 fits a byte). S2/S3 SASS
identical to the benchmarked all-three variant up to shared-memory offsets;
debug tests (fast and forced generic) and all three datasets pass S3.
Decided on ncu numbers; confirmed by the sustained (power-capped) A100 bench
at `76be63b`:

| S3 | before (`be346d9` code) | now | change | % of speed of light |
|---|---|---|---|---|
| dense | 1833 μs | 1815 μs | −1.0% | 51.3% → 51.8% |
| moderate | 985 μs | 840 μs | −14.7% | 43.5% → 51.0% |
| sparse | 925 μs | 769 μs | −16.9% | 42.3% → 50.8% |

Bench S3 − S2 (look-backs): moderate 62 → 88 μs, sparse 54 → 88 μs (dense
within noise). Forced generic path: dense 2062, moderate 1135, sparse
1074 μs (+14% / +35% / +40% over the fast path).

**Next:** with less compute per tile, the look-backs show on moderate and
sparse: S3 − S2 grew from 52–68 μs (base) to 107–122 μs (all three) in ncu,
54–62 → 88 μs in the bench.

### Look-back cost: delay sweep and 384-thread tiles (failed, removed)

Sparse profile (all three shared-memory changes), S3 vs S2: barrier stalls
1.7 → 6.6 per issue (7 warps wait at the block scans while warp 0 does the
state and then the index look-back), sleeping 0.2 (the look-back warp's
`__nanosleep`), long scoreboard 0.84 → 1.14 (descriptor loads). Each
look-back sleeps 450 ns before its first poll (tuned for P1): ~0.9 μs per
tile, with ~33 tiles per block slot (21334 tiles / 648 resident blocks).

Variants measured at `e1a604d`:
- **look-back delay 0 / 100 / 200 ns** instead of 450 (first poll only; the
  350 ns sleeps between later polls unchanged);
- **BS384**: 384 threads × 96 B = 36 KB tiles, 38 KB shared memory,
  `__launch_bounds__(384, 4)`: 4 blocks/SM (same 1536 threads/SM), 14222
  instead of 21334 tiles;
- **BS384 with delay 0 / 100 / 200**.

A100 bench, S3 μs (change vs base of the same run; dense CIs ±10–17 μs):

| | dense (1821) | moderate (848) | sparse (780) |
|---|---|---|---|
| delay 0 | +31 | +18 | +18 |
| delay 100 | +17 | −6 | −1 |
| delay 200 | +26 | −2 | −1 |
| BS384 | +24 | +16 | +12 |
| BS384 delay 0 | +43 | +42 | +36 |
| BS384 delay 100 | +22 | +15 | +3 |
| BS384 delay 200 | +25 | +15 | +7 |

Causes:
1. *Delay 0* is slower (+18 μs on moderate/sparse): polling without the
   initial sleep adds L2 traffic that slows the other blocks. 100–200 ns
   gain at most 6 μs, within the run-to-run drift (this run's base is
   8–11 μs slower than the previous one).
2. *BS384* shortens the look-back share (moderate S3 − S2 93 → 79 μs) but
   the compute gets slower (S2 +26–30 μs: block scans and barriers over 12
   warps instead of 8, and fewer, larger blocks per SM).

Removed from the bench and kernel. Kept, since they change no current SASS:
`FIRST_DELAY_NS` (template parameter of `TilePrefixCallbackOp`, default 450)
and `LB_BIG(BS)` (lexerBig's launch bounds follow the block size, 1536
threads/SM).

Current state (`be346d9` + shared-memory changes, `76be63b` code), A100 S3:
dense 1815–1821 μs, moderate 840–848, sparse 769–780 (~51% of speed of light
on all three). The look-backs (~90 μs on moderate/sparse) remain the largest
cost not reduced; an idea not yet investigated: stop a look-back early at a
predecessor whose aggregate state is a constant function (maps every
incoming state to the same state), common in lexers.

### Derived tables built at compile time (kept)

Sparse S3 per-region profile (adopted kernel): building the derived tables in
every tile (copy compose, compute `row_of8`, `comp`, `comp_pf` with scalar
global loads and packing) cost ~8% of the warp instructions and ~12% of the
stall samples (its global-load latency sits at the start of every tile).
Other regions: pass A chain step 24% / pass A loop 13%, pass B 19%, block
scans + look-back glue 13%, emission 5%.

Change: `LexerChainTables { row_of8[256]; comp[NS·NS]; comp_pf[NS·16] }`
(736 B, 16-byte aligned) is computed by `constexpr make_lexer_chain_tables()`
from `h_to_state` / `h_compose` (now `constexpr`; the state accessors and
`pack_chain_state` are `constexpr __host__ __device__`), uploaded once by
`LexerCtxShmem`, and copied per tile as 46 `uint4`s. The tables stay data in
global memory (loadable at run time in principle); the generator is
unchanged and the packed formats stay defined next to the kernel.

SASS: S2/S3 −496 static instructions (setup only: 9 fewer global loads, 9
fewer shared stores, ~190 fewer IMAD/LOP3/LEA/SHF); hot-loop LDS/PRMT/POPC
counts unchanged; 40 registers, no spills, same shared memory; other kernels
unchanged. Debug tests (fast + generic) and all three datasets pass S3.

A100 bench (`e46e7db`) vs the previous run:

| | S2 | S3 | % of speed of light |
|---|---|---|---|
| dense | noise (±15 μs) | 1821 → 1820 μs | 51.6% |
| moderate | 755 → 739 μs (−2.1%) | 848 → 830 μs (−2.1%) | 51.6% |
| sparse | 690 → 670 μs (−2.9%) | 780 → 764 μs (−2.1%) | 51.2% |

ncu, sparse (real clock): S2 0.596 → 0.586 ms, warp instructions 252M → 239M
(−5%); S3 0.703 → 0.704 ms with barrier stalls 6.6 → 7.3 per issue. The
instruction cut is real, but much of the removed setup overlapped with other
blocks, and in S3 the faster tiles wait longer on their predecessors: the
look-back wait absorbs compute savings on moderate/sparse. Next: hide the
look-back wait behind other work instead of only cutting instructions.

### More, smaller blocks to absorb the look-back wait (A100 numbers pending)

The look-back wait is idle time of a whole block (7 warps at the barrier,
warp 0 spinning); loads already overlap (L2-resident test), so only other
blocks' compute can fill it. Registers cap 256-thread blocks at 6 per SM
(40 × 256 × 6); 128-thread blocks (12 KB tiles) fit 11 per SM by shared
memory (13.4 KB each, 40 registers, no spills; `LB_BIG(128)` =
`__launch_bounds__(128, 12)`): 44 warps in 11 independent blocks instead of
48 warps in 6, at twice the tiles and look-backs. Bench rows `Big S2/S3
BS128`; S3 passes on all three datasets locally.

Considered and deferred: a warp-specialized persistent kernel (8 compute
warps + 1 look-back warp, pass A of tile k+1 while tile k's look-back
resolves, double buffer in dynamic shared memory → 3 blocks/SM); large
rewrite with occupancy risk, decided on after this test.
