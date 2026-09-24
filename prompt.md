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

The warp-transpose gain came from eliminating shmem store bank conflicts and
reducing instruction count. The vectorized load/store gain came from cutting
memory-pipe instructions per element from ~9 to ~3.75, which makes the block
scan free (see "Vectorized load/store").

## Conclusion — Current State of P1

`p1_vec<256, 24, 3>` (V3) at **1530 μs / 1028 GB/s** is the best P1 kernel.
`p1_transpose` (1876 μs) stays in `p1_bench.cu` as the L3 reference; the other
variants listed above were removed from the code and their numbers are kept
as a historical record.

Where the time goes (V ladder): load/store at the practical speed of light
(V1 1256 μs vs 1159 μs u16 floor), block scan free (V2 1245 μs), lookback
+285 μs. V3 is at 82% of V1 and 76% of the u16 floor.

Levers exhausted:

- Lookback sleep changes, a 64-tile lookback window, larger blocks, column
  compose and 8 blocks/SM for the IPT=22 kernel (with or without register
  packing / carveout changes) all made P1 slower or had no effect (see
  "What We Tried").
- `BLOCK_LOAD/STORE_WARP_TRANSPOSE` (+144 GB/s) and then vectorized
  warp-local exchanges (+190 GB/s) removed the load/store path's cost.
- `st.relaxed.gpu` tile state stores replace `__threadfence()`: +54 GB/s.
- Static shmem for tables and `__launch_bounds__(256,6)` gave earlier gains.
- Static `blockIdx.x` assignment and tile-0 fast path have no measurable effect.

Open directions — the remaining gap is the lookback (~285 μs):

1. **Hide the lookback instead of shortening it** — the lookback takes
   ~1.4 μs per tile regardless of sleeps, window size or tile count (see
   "Lookback statistics"); the cost is the block's other warps idling for
   it. More resident warps or other tile sizes did not help (see "More
   resident blocks" and "`p1_vec` at 8 blocks/SM and at IPT=16 / IPT=32").
   Being measured: `p1_vec_pipe`, a persistent V3 that prefetches its next
   tile with `cp.async` so the block keeps loads in flight during the
   lookback.
2. **Two-kernel reduce-then-scan** — traffic floor ~1574 μs, now *above*
   V3's 1530 μs, so it can no longer win on traffic.
