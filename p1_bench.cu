// p1_bench.cu — standalone benchmark for the two-pass lexer P1 kernel.
//
// Pass 1 (P1): maps each input byte to a DFA state, then runs an inclusive
// prefix scan using the DFA composition table as the scan operator.
// Result is a flat array of prefix states written to global memory for P2.
//
// Usage: ./p1_bench <input_file>
//   input_file: raw bytes, e.g. data/tokens_dense_500MiB.in

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/iterator/transform_iterator.h>

// ---------------------------------------------------------------------------
// GPU error checking
// ---------------------------------------------------------------------------

#define gpuAssert(x) _gpuAssert(x, __FILE__, __LINE__)
static void _gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPU error: %s  (%s:%d)\n",
                cudaGetErrorString(code), file, line);
        exit(1);
    }
}

// ---------------------------------------------------------------------------
// Timing helper: prints "mean CI GB/s\n" given per-run millisecond times
// and the total bytes transferred.
// ---------------------------------------------------------------------------
static void print_stats(float* ms, int runs, size_t bytes) {
    double mean = 0, var = 0, gbps = 0;
    double factor = (double)bytes / (1000.0 * runs);
    for (int i = 0; i < runs; i++) {
        double t = fmax(ms[i] * 1e3, 0.5);   // microseconds
        mean += t / runs;
        var  += (t * t) / runs;
        gbps += factor / t;
    }
    double std  = sqrt(var);
    double bound = 0.95 * std / sqrt((double)runs);
    printf("%.0fμs (95%% CI: [%.1fμs, %.1fμs]); %.0fGB/s\n",
           mean, mean - bound, mean + bound, gbps);
}

// ---------------------------------------------------------------------------
// DFA tables for a simple parenthesised-identifier lexer.
//
// state_t encodes: end-state index (bits 3:0), token type (bits 6:4),
// accept flag (bit 7), produce flag (bit 8).
// IDENTITY (74) is the scan identity: compose(x, IDENTITY) == x.
// ---------------------------------------------------------------------------
using state_t = uint16_t;

static const uint32_t NUM_STATES = 12;
static const state_t  IDENTITY   = 74;

static state_t h_to_state[256] = {
    75, 75, 75, 75, 75, 75, 75, 75, 75, 128, 128, 75, 75, 128,
    75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
    75, 75, 75, 128, 75, 75, 75, 75, 75, 75, 75, 161, 178, 75,
    75, 75, 75, 75, 75, 147, 147, 147, 147, 147, 147, 147, 147,
    147, 147, 75, 75, 75, 75, 75, 75, 75, 147, 147, 147, 147,
    147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 147,
    147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 75, 75,
    75, 75, 75, 75, 147, 147, 147, 147, 147, 147, 147, 147, 147,
    147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 147,
    147, 147, 147, 147, 147, 147, 75, 75, 75, 75, 75, 75, 75, 75,
    75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
    75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
    75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
    75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
    75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
    75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
    75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
    75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
    75, 75, 75, 75
};

static state_t h_compose[NUM_STATES * NUM_STATES] = {
    132, 392, 392, 392, 132, 392, 392, 392, 132, 392, 128, 75,
    421, 421, 421, 421, 421, 421, 421, 421, 421, 421, 161, 75,
    438, 438, 438, 438, 438, 438, 438, 438, 438, 438, 178, 75,
    407, 407, 407, 153, 407, 407, 407, 153, 407, 153, 147, 75,
    132, 132, 132, 132, 132, 132, 132, 132, 132, 132, 132, 75,
    421, 421, 421, 421, 421, 421, 421, 421, 421, 421, 421, 75,
    438, 438, 438, 438, 438, 438, 438, 438, 438, 438, 438, 75,
    407, 407, 407, 407, 407, 407, 407, 407, 407, 407, 407, 75,
    392, 392, 392, 392, 392, 392, 392, 392, 392, 392, 392, 75,
    153, 153, 153, 153, 153, 153, 153, 153, 153, 153, 153, 75,
    128, 161, 178, 147, 132, 421, 438, 407, 392, 153, 74, 75,
    75,  75,  75,  75,  75,  75,  75,  75,  75,  75,  75,  75,
};

// ---------------------------------------------------------------------------
// Decoupled lookback scan infrastructure (from common/sps.cu.h)
// ---------------------------------------------------------------------------

const uint8_t  LG_WARP          = 5;
const uint8_t  WARP             = 1 << LG_WARP;
const uint32_t TILE_STATUS_PADDING = WARP;

enum ScanTileStatus : uint32_t {
    SCAN_TILE_OOB       = 0,
    SCAN_TILE_INVALID   = 1,
    SCAN_TILE_PARTIAL   = 2,
    SCAN_TILE_INCLUSIVE = 3,
};

// state_t is uint16_t: pack status+value into a single uint32_t so that
// publish and read are each one atomic-width transaction.
struct ScanTileState {
    // Layout: bits[31:16] = value, bits[15:0] = status
    uint32_t* d_tile_descriptors;

    __host__ static size_t AllocationSize(int num_tiles) {
        return (num_tiles + TILE_STATUS_PADDING) * sizeof(uint32_t);
    }

    __device__ void InitializeStatus(int num_tiles) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_tiles)
            d_tile_descriptors[TILE_STATUS_PADDING + idx] =
                (uint32_t(state_t()) << 16) | uint32_t(SCAN_TILE_INVALID);
        if (blockIdx.x == 0 && threadIdx.x < TILE_STATUS_PADDING)
            d_tile_descriptors[threadIdx.x] =
                (uint32_t(state_t()) << 16) | uint32_t(SCAN_TILE_OOB);
    }

    __device__ __forceinline__ void SetPartial(int tile_idx, state_t value) {
        uint32_t word = (uint32_t(value) << 16) | uint32_t(SCAN_TILE_PARTIAL);
        asm volatile("st.relaxed.gpu.u32 [%0], %1;"
                     : : "l"(d_tile_descriptors + TILE_STATUS_PADDING + tile_idx), "r"(word)
                     : "memory");
    }

    __device__ __forceinline__ void SetInclusive(int tile_idx, state_t value) {
        uint32_t word = (uint32_t(value) << 16) | uint32_t(SCAN_TILE_INCLUSIVE);
        asm volatile("st.relaxed.gpu.u32 [%0], %1;"
                     : : "l"(d_tile_descriptors + TILE_STATUS_PADDING + tile_idx), "r"(word)
                     : "memory");
    }

    // Sleeps initial_delay_ns, then polls until no lane in the warp sees
    // INVALID (350 ns between polls). first_status / retries report what the
    // first poll saw and how many times the warp re-polled (used only by
    // instrumented variants).
    __device__ __forceinline__ void WaitForValid(int tile_idx,
                                                  uint32_t& status,
                                                  state_t& value,
                                                  uint32_t initial_delay_ns,
                                                  uint32_t& first_status,
                                                  uint32_t& retries) {
        __nanosleep(initial_delay_ns);
        uint32_t w;
        asm volatile("ld.relaxed.gpu.u32 %0, [%1];"
                     : "=r"(w)
                     : "l"(d_tile_descriptors + TILE_STATUS_PADDING + tile_idx)
                     : "memory");
        first_status = w & 0xffffu;
        while (__any_sync(0xffffffff, (w & 0xffffu) == uint32_t(SCAN_TILE_INVALID))) {
            retries++;
            __nanosleep(350);
            asm volatile("ld.relaxed.gpu.u32 %0, [%1];"
                         : "=r"(w)
                         : "l"(d_tile_descriptors + TILE_STATUS_PADDING + tile_idx)
                         : "memory");
        }
        status = w & 0xffffu;
        value  = state_t(w >> 16);
    }
};

// Lookback statistics, recorded only by instrumented (STATS = true) variants.
// Counters are spread over STAT_SLOTS slots by tile index to limit atomic
// contention; the host sums the slots.
//   FIRST_*:  status of tile-1 at the first poll (after the initial sleep)
//   RETRY_*:  warp-level re-polls of a window with an INVALID entry, per tile
//   WIN_*:    32-tile lookback windows walked per tile
//   DEPTH_*:  tiles walked back from tile-1 to the nearest INCLUSIVE (or OOB)
//             tile; 0 = tile-1 was already INCLUSIVE
enum LookbackStat : uint32_t {
    STAT_TILES,
    STAT_FIRST_INVALID, STAT_FIRST_PARTIAL, STAT_FIRST_INCLUSIVE,
    STAT_RETRIES, STAT_RETRY_0, STAT_RETRY_1, STAT_RETRY_2PLUS,
    STAT_WINDOWS, STAT_WIN_1, STAT_WIN_2, STAT_WIN_3PLUS,
    STAT_DEPTH_SUM, STAT_DEPTH_0, STAT_DEPTH_1_7, STAT_DEPTH_8_31,
    STAT_DEPTH_32_63, STAT_DEPTH_64PLUS,
    STAT_COUNT
};
constexpr uint32_t STAT_SLOTS = 32;
__device__ unsigned long long g_lookback_stats[STAT_SLOTS * STAT_COUNT];

__device__ __forceinline__ void record_lookback_stats(
    int tile_idx, uint32_t first_status, uint32_t retries, uint32_t windows, uint32_t depth)
{
    unsigned long long* s = g_lookback_stats + (tile_idx % STAT_SLOTS) * STAT_COUNT;
    atomicAdd(s + STAT_TILES, 1ull);
    atomicAdd(s + (first_status == uint32_t(SCAN_TILE_INVALID) ? STAT_FIRST_INVALID
                 : first_status == uint32_t(SCAN_TILE_PARTIAL) ? STAT_FIRST_PARTIAL
                                                               : STAT_FIRST_INCLUSIVE), 1ull);
    atomicAdd(s + STAT_RETRIES, (unsigned long long)retries);
    atomicAdd(s + (retries == 0 ? STAT_RETRY_0 : retries == 1 ? STAT_RETRY_1 : STAT_RETRY_2PLUS), 1ull);
    atomicAdd(s + STAT_WINDOWS, (unsigned long long)windows);
    atomicAdd(s + (windows == 1 ? STAT_WIN_1 : windows == 2 ? STAT_WIN_2 : STAT_WIN_3PLUS), 1ull);
    atomicAdd(s + STAT_DEPTH_SUM, (unsigned long long)depth);
    atomicAdd(s + (depth == 0 ? STAT_DEPTH_0 : depth < 8 ? STAT_DEPTH_1_7 : depth < 32 ? STAT_DEPTH_8_31
                 : depth < 64 ? STAT_DEPTH_32_63 : STAT_DEPTH_64PLUS), 1ull);
}

// Prefix callback used with CUB BlockScan (decoupled lookback).
template<typename ScanOpT, bool STATS = false>
struct PrefixCallbackOp {
    using WarpReduceT = cub::WarpReduce<state_t, WARP>;

    ScanTileState& tile_state;
    ScanOpT        scan_op;
    int            tile_idx;
    state_t        identity;
    state_t        exclusive_prefix;

    struct TempStorage {
        typename WarpReduceT::TempStorage warp_reduce;
        state_t exclusive_prefix;
        state_t block_aggregate;
    };
    TempStorage& temp_storage;

    __device__ __forceinline__
    PrefixCallbackOp(ScanTileState& ts, TempStorage& tmp,
                     ScanOpT op, int idx, state_t id)
        : tile_state(ts), scan_op(op), tile_idx(idx),
          identity(id), temp_storage(tmp) {}

    __device__ __forceinline__ state_t
    ProcessWindow(int predecessor_idx, uint32_t& predecessor_status,
                  uint32_t delay_ns, uint32_t& first_status, uint32_t& retries) {
        state_t value;
        tile_state.WaitForValid(predecessor_idx, predecessor_status, value, delay_ns,
                                first_status, retries);
        int is_oob    = (predecessor_status == uint32_t(SCAN_TILE_OOB));
        int tail_flag = (predecessor_status == uint32_t(SCAN_TILE_INCLUSIVE)) | is_oob;
        state_t eff   = is_oob ? identity : value;
        auto flipped  = [&](state_t a, state_t b) { return scan_op(b, a); };
        return WarpReduceT(temp_storage.warp_reduce)
                   .TailSegmentedReduce(eff, tail_flag, flipped);
    }

    __device__ __forceinline__ state_t operator()(state_t block_aggregate) {
        if (threadIdx.x == 0) {
            temp_storage.block_aggregate = block_aggregate;
            tile_state.SetPartial(tile_idx, block_aggregate);
        }
        int      predecessor_idx = tile_idx - threadIdx.x - 1;
        uint32_t predecessor_status;
        // Seed initial delay with tile_idx to spread out thundering-herd polling.
        uint32_t initial_delay = 200 + (uint32_t)(tile_idx % 8) * 50;
        uint32_t first_status, later_status, retries = 0, windows = 1;
        exclusive_prefix = ProcessWindow(predecessor_idx, predecessor_status, initial_delay,
                                         first_status, retries);
        while (__all_sync(0xffffffff,
                          predecessor_status != uint32_t(SCAN_TILE_INCLUSIVE) &&
                          predecessor_status != uint32_t(SCAN_TILE_OOB))) {
            predecessor_idx -= WARP;
            windows++;
            state_t w = ProcessWindow(predecessor_idx, predecessor_status, 350,
                                      later_status, retries);
            exclusive_prefix = scan_op(w, exclusive_prefix);
        }
        if constexpr (STATS) {
            uint32_t done  = __ballot_sync(0xffffffff,
                                           predecessor_status == uint32_t(SCAN_TILE_INCLUSIVE) ||
                                           predecessor_status == uint32_t(SCAN_TILE_OOB));
            uint32_t depth = (windows - 1) * WARP + (__ffs(done) - 1);
            if (threadIdx.x == 0)
                record_lookback_stats(tile_idx, first_status, retries, windows, depth);
        }
        state_t ep = (state_t)__shfl_sync(0xffffffff, (uint32_t)exclusive_prefix, 0);
        if (threadIdx.x == 0) {
            tile_state.SetInclusive(tile_idx, scan_op(ep, block_aggregate));
            temp_storage.exclusive_prefix = ep;
        }
        return ep;
    }

    __device__ __forceinline__ state_t GetExclusivePrefix() {
        return temp_storage.exclusive_prefix;
    }
};

// ---------------------------------------------------------------------------
// Scan operators
// ---------------------------------------------------------------------------

// DFA composition: compose(a, b) gives the state reached by first applying
// transition a, then transition b.
struct ComposeOp {
    state_t* d_compose;   // NUM_STATES*NUM_STATES table (shmem or global)

    __device__ __forceinline__ state_t
    operator()(state_t a, state_t b) const {
        return d_compose[(b & 15u) * NUM_STATES + (a & 15u)];
    }
};


// Functor for TransformInputIterator: maps a raw byte to its initial state.
struct ByteToState {
    state_t* d_to_state;
    __device__ __forceinline__ state_t operator()(uint8_t byte) const {
        return d_to_state[byte];
    }
};

// ---------------------------------------------------------------------------
// Shared memory helpers
// ---------------------------------------------------------------------------

// Load NUM_STATES*NUM_STATES compose table and 256-entry to_state table
// from global memory into shared memory using 64-bit loads.
template<uint32_t BLOCK_SIZE>
__device__ inline void loadTablesToShmem(
    state_t* __restrict__ d_compose_glb,
    state_t* __restrict__ d_to_state_glb,
    volatile state_t* shmem_compose,    // NUM_STATES*NUM_STATES entries
    volatile state_t* shmem_to_state)   // 256 entries
{
    // Each entry is 2 bytes; load 8 bytes (4 entries) at a time.
    // compose table: 144 bytes = 18 uint64_t loads, 1 pass for BLOCK_SIZE>=18
    for (uint32_t i = threadIdx.x; i < NUM_STATES * NUM_STATES / 4; i += BLOCK_SIZE)
        reinterpret_cast<volatile uint64_t*>(shmem_compose)[i] =
            reinterpret_cast<uint64_t*>(d_compose_glb)[i];
    // to_state table: 512 bytes = 64 uint64_t loads
    for (uint32_t i = threadIdx.x; i < 256 / 4; i += BLOCK_SIZE)
        reinterpret_cast<volatile uint64_t*>(shmem_to_state)[i] =
            reinterpret_cast<uint64_t*>(d_to_state_glb)[i];
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Init kernel for tile state array
// ---------------------------------------------------------------------------
__global__ void initTileState(ScanTileState ts, int num_tiles) {
    ts.InitializeStatus(num_tiles);
}

static void initScanTileState(ScanTileState& ts, int num_tiles) {
    int threads = 256;
    int blocks  = (num_tiles + TILE_STATUS_PADDING + threads - 1) / threads;
    initTileState<<<blocks, threads>>>(ts, num_tiles);
}

// ---------------------------------------------------------------------------
// P1 kernels
// ---------------------------------------------------------------------------

// minnctapersm=6 is valid only on sm_80+ (A100 has 65536 regs/SM;
// 6*256*40 = 61440 <= 65536). sm_75 has only 32768 and would warn.
#if __CUDA_ARCH__ >= 800
#define LB_P1 __launch_bounds__(256, 6)
#else
#define LB_P1 __launch_bounds__(256)
#endif


template<uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD, bool STATS = false>
__global__ LB_P1
void p1_transpose(
    state_t* __restrict__ d_compose_glb,
    state_t* __restrict__ d_to_state_glb,
    const uint8_t* __restrict__ d_in,
    state_t* __restrict__ d_states_out,
    ScanTileState tile_state,
    uint32_t size,
    uint32_t num_tiles)
{
    using TransformIter = thrust::transform_iterator<ByteToState, const uint8_t*>;
    using BlockLoadT  = cub::BlockLoad <state_t, BLOCK_SIZE, ITEMS_PER_THREAD,
                                        cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockStoreT = cub::BlockStore<state_t, BLOCK_SIZE, ITEMS_PER_THREAD,
                                        cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockScanT  = cub::BlockScan <state_t, BLOCK_SIZE,
                                        cub::BLOCK_SCAN_WARP_SCANS>;
    using PrefixOp    = PrefixCallbackOp<ComposeOp, STATS>;

    __shared__ union {
        typename BlockLoadT::TempStorage  load;
        typename BlockStoreT::TempStorage store;
        struct {
            typename PrefixOp::TempStorage  prefix;
            typename BlockScanT::TempStorage scan;
        } scan_storage;
    } temp;

    __shared__ __align__(8) state_t shmem_compose[NUM_STATES * NUM_STATES];
    __shared__ __align__(8) state_t shmem_to_state[256];

    loadTablesToShmem<BLOCK_SIZE>(
        d_compose_glb, d_to_state_glb, shmem_compose, shmem_to_state);

    uint32_t tile_idx = blockIdx.x;
    uint32_t glb_offs = tile_idx * BLOCK_SIZE * ITEMS_PER_THREAD;
    uint32_t valid    = (uint32_t)min((uint64_t)BLOCK_SIZE * ITEMS_PER_THREAD,
                                      (uint64_t)size - glb_offs);

    ByteToState byte_to_state{shmem_to_state};
    TransformIter d_in_states(d_in + glb_offs, byte_to_state);
    state_t st[ITEMS_PER_THREAD];
    if (glb_offs + BLOCK_SIZE * ITEMS_PER_THREAD <= size)
        BlockLoadT(temp.load).Load(d_in_states, st);
    else
        BlockLoadT(temp.load).Load(d_in_states, st, valid, IDENTITY);
    __syncthreads();

    ComposeOp compose_op{shmem_compose};

    if (tile_idx == 0) {
        state_t block_aggregate;
        BlockScanT(temp.scan_storage.scan).InclusiveScan(st, st, compose_op, block_aggregate);
        if (threadIdx.x == 0)
            tile_state.SetInclusive(0, block_aggregate);
    } else {
        PrefixOp prefix_op(tile_state, temp.scan_storage.prefix, compose_op, (int)tile_idx, IDENTITY);
        BlockScanT(temp.scan_storage.scan).InclusiveScan(st, st, compose_op, prefix_op);
    }
    __syncthreads();

    if (glb_offs + BLOCK_SIZE * ITEMS_PER_THREAD <= size)
        BlockStoreT(temp.store).Store(d_states_out + glb_offs, st);
    else
        BlockStoreT(temp.store).Store(d_states_out + glb_offs, st, valid);
}

// ---------------------------------------------------------------------------
// Map-only baselines: speed of light for P1
//
// Byte → state lookup with no scan. Reads the same input as P1 and writes one
// state per byte at OUT_BITS each: 16 = full state_t, 8 = state index as
// uint8_t, 4 = two state indices packed per byte (element 2j in the low
// nibble of byte j). The state index (bits 3:0) determines the full state_t,
// so the 8- and 4-bit outputs lose no information.
//
// Each thread issues CHUNKS 16-byte loads (coalesced across the block) before
// doing any work, to keep enough loads in flight.
// ---------------------------------------------------------------------------

__host__ __device__ __forceinline__ uint32_t map_out_bytes(uint32_t size, uint32_t out_bits) {
    return (uint32_t)(((uint64_t)size * out_bits + 7) / 8);
}

template<uint32_t OUT_BITS, uint32_t BLOCK_SIZE, uint32_t CHUNKS>
__global__ __launch_bounds__(BLOCK_SIZE)
void map_only(
    const state_t* __restrict__ d_to_state_glb,
    const uint8_t* __restrict__ d_in,
    uint8_t* __restrict__ d_out,
    uint32_t size)
{
    static_assert(OUT_BITS == 16 || OUT_BITS == 8 || OUT_BITS == 4, "OUT_BITS must be 16, 8 or 4");

    __shared__ state_t shmem_to_state[256];
    for (uint32_t i = threadIdx.x; i < 256; i += BLOCK_SIZE)
        shmem_to_state[i] = d_to_state_glb[i];
    __syncthreads();

    auto map = [&](uint32_t byte) -> uint32_t {
        state_t s = shmem_to_state[byte];
        return OUT_BITS == 16 ? s : (s & 15u);
    };

    const uint32_t nvec = size / 16;
    const uint4* in_vec = reinterpret_cast<const uint4*>(d_in);
    const uint32_t base = blockIdx.x * BLOCK_SIZE * CHUNKS + threadIdx.x;

    uint4 v[CHUNKS];
    #pragma unroll
    for (uint32_t c = 0; c < CHUNKS; c++) {
        uint32_t idx = base + c * BLOCK_SIZE;
        if (idx < nvec) v[c] = in_vec[idx];
    }

    #pragma unroll
    for (uint32_t c = 0; c < CHUNKS; c++) {
        uint32_t idx = base + c * BLOCK_SIZE;
        if (idx >= nvec) break;
        const uint32_t w[4] = {v[c].x, v[c].y, v[c].z, v[c].w};
        auto in_byte = [&](uint32_t k) { return (w[k / 4] >> (8 * (k % 4))) & 0xffu; };

        if constexpr (OUT_BITS == 16) {
            uint32_t o[8];
            #pragma unroll
            for (uint32_t j = 0; j < 8; j++)
                o[j] = map(in_byte(2 * j)) | (map(in_byte(2 * j + 1)) << 16);
            uint4* out_vec = reinterpret_cast<uint4*>(d_out);
            out_vec[2 * idx]     = make_uint4(o[0], o[1], o[2], o[3]);
            out_vec[2 * idx + 1] = make_uint4(o[4], o[5], o[6], o[7]);
        } else if constexpr (OUT_BITS == 8) {
            uint32_t o[4];
            #pragma unroll
            for (uint32_t j = 0; j < 4; j++)
                o[j] = map(in_byte(4 * j))           | (map(in_byte(4 * j + 1)) << 8)
                     | (map(in_byte(4 * j + 2)) << 16) | (map(in_byte(4 * j + 3)) << 24);
            reinterpret_cast<uint4*>(d_out)[idx] = make_uint4(o[0], o[1], o[2], o[3]);
        } else {
            uint32_t o[2] = {0, 0};
            #pragma unroll
            for (uint32_t k = 0; k < 16; k++)
                o[k / 8] |= map(in_byte(k)) << (4 * (k % 8));
            reinterpret_cast<uint2*>(d_out)[idx] = make_uint2(o[0], o[1]);
        }
    }

    // Tail (size % 16 bytes): one thread, scalar. Starts 16-aligned, so the
    // nibble pairs never straddle the vector part.
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (uint32_t i = nvec * 16; i < size; i += (OUT_BITS == 4 ? 2 : 1)) {
            if constexpr (OUT_BITS == 16)
                reinterpret_cast<state_t*>(d_out)[i] = (state_t)map(d_in[i]);
            else if constexpr (OUT_BITS == 8)
                d_out[i] = (uint8_t)map(d_in[i]);
            else
                d_out[i / 2] = (uint8_t)(map(d_in[i]) | (i + 1 < size ? map(d_in[i + 1]) << 4 : 0));
        }
    }
}

// Host reference check for map_only output.
static bool check_map_only(const uint8_t* input, uint32_t size,
                           uint32_t out_bits, const uint8_t* out) {
    for (uint32_t i = 0; i < size; i++) {
        state_t s = h_to_state[input[i]];
        uint32_t expect = out_bits == 16 ? s : (s & 15u);
        uint32_t got;
        if (out_bits == 16)     got = reinterpret_cast<const state_t*>(out)[i];
        else if (out_bits == 8) got = out[i];
        else                    got = (out[i / 2] >> (4 * (i % 2))) & 15u;
        if (got != expect) {
            fprintf(stderr, "map_only<%u> mismatch at %u: got %u, expected %u\n",
                    out_bits, i, got, expect);
            return false;
        }
    }
    return true;
}

// Map-only u16 with fully coalesced stores: each thread maps 8 input bytes
// (one 8-byte load) to 8 states (one 16-byte store), so consecutive threads
// write consecutive 16-byte segments. map_only<16> instead issues two 16-byte
// stores 32 bytes apart per thread.
template<uint32_t BLOCK_SIZE, uint32_t CHUNKS>
__global__ __launch_bounds__(BLOCK_SIZE)
void map_only_u16_coalesced(
    const state_t* __restrict__ d_to_state_glb,
    const uint8_t* __restrict__ d_in,
    state_t* __restrict__ d_out,
    uint32_t size)
{
    __shared__ state_t shmem_to_state[256];
    for (uint32_t i = threadIdx.x; i < 256; i += BLOCK_SIZE)
        shmem_to_state[i] = d_to_state_glb[i];
    __syncthreads();

    const uint32_t nvec = size / 8;
    const uint2* in_vec = reinterpret_cast<const uint2*>(d_in);
    uint4* out_vec      = reinterpret_cast<uint4*>(d_out);
    const uint32_t base = blockIdx.x * BLOCK_SIZE * CHUNKS + threadIdx.x;

    uint2 v[CHUNKS];
    #pragma unroll
    for (uint32_t c = 0; c < CHUNKS; c++) {
        uint32_t idx = base + c * BLOCK_SIZE;
        if (idx < nvec) v[c] = in_vec[idx];
    }

    #pragma unroll
    for (uint32_t c = 0; c < CHUNKS; c++) {
        uint32_t idx = base + c * BLOCK_SIZE;
        if (idx >= nvec) break;
        const uint32_t w[2] = {v[c].x, v[c].y};
        auto s = [&](uint32_t k) -> uint32_t {
            return shmem_to_state[(w[k / 4] >> (8 * (k % 4))) & 0xffu];
        };
        out_vec[idx] = make_uint4(s(0) | (s(1) << 16), s(2) | (s(3) << 16),
                                  s(4) | (s(5) << 16), s(6) | (s(7) << 16));
    }

    // Tail (size % 8 bytes): one thread, scalar.
    if (blockIdx.x == 0 && threadIdx.x == 0)
        for (uint32_t i = nvec * 8; i < size; i++)
            d_out[i] = shmem_to_state[d_in[i]];
}

// ---------------------------------------------------------------------------
// P1 cost ladder: p1_transpose with pieces removed, to attribute its time.
//   STEP 1: warp-transpose load -> store, no scan. Output = byte->state map.
//   STEP 2: + BlockScan with compose, no lookback: every tile scans from
//           IDENTITY independently. Output is a per-tile scan (not valid P1).
// STEP 3 is p1_transpose itself (+ decoupled lookback).
// ---------------------------------------------------------------------------
template<uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD, uint32_t STEP>
__global__ LB_P1
void p1_ladder(
    state_t* __restrict__ d_compose_glb,
    state_t* __restrict__ d_to_state_glb,
    const uint8_t* __restrict__ d_in,
    state_t* __restrict__ d_states_out,
    uint32_t size)
{
    static_assert(STEP == 1 || STEP == 2, "STEP must be 1 or 2");

    using TransformIter = thrust::transform_iterator<ByteToState, const uint8_t*>;
    using BlockLoadT  = cub::BlockLoad <state_t, BLOCK_SIZE, ITEMS_PER_THREAD,
                                        cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockStoreT = cub::BlockStore<state_t, BLOCK_SIZE, ITEMS_PER_THREAD,
                                        cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockScanT  = cub::BlockScan <state_t, BLOCK_SIZE,
                                        cub::BLOCK_SCAN_WARP_SCANS>;

    __shared__ union {
        typename BlockLoadT::TempStorage  load;
        typename BlockStoreT::TempStorage store;
        typename BlockScanT::TempStorage  scan;
    } temp;

    __shared__ __align__(8) state_t shmem_compose[NUM_STATES * NUM_STATES];
    __shared__ __align__(8) state_t shmem_to_state[256];

    loadTablesToShmem<BLOCK_SIZE>(
        d_compose_glb, d_to_state_glb, shmem_compose, shmem_to_state);

    uint32_t tile_idx = blockIdx.x;
    uint32_t glb_offs = tile_idx * BLOCK_SIZE * ITEMS_PER_THREAD;
    uint32_t valid    = (uint32_t)min((uint64_t)BLOCK_SIZE * ITEMS_PER_THREAD,
                                      (uint64_t)size - glb_offs);

    ByteToState byte_to_state{shmem_to_state};
    TransformIter d_in_states(d_in + glb_offs, byte_to_state);
    state_t st[ITEMS_PER_THREAD];
    if (glb_offs + BLOCK_SIZE * ITEMS_PER_THREAD <= size)
        BlockLoadT(temp.load).Load(d_in_states, st);
    else
        BlockLoadT(temp.load).Load(d_in_states, st, valid, IDENTITY);
    __syncthreads();

    if constexpr (STEP == 2) {
        ComposeOp compose_op{shmem_compose};
        state_t block_aggregate;
        BlockScanT(temp.scan).InclusiveScan(st, st, compose_op, block_aggregate);
        __syncthreads();
    }

    if (glb_offs + BLOCK_SIZE * ITEMS_PER_THREAD <= size)
        BlockStoreT(temp.store).Store(d_states_out + glb_offs, st);
    else
        BlockStoreT(temp.store).Store(d_states_out + glb_offs, st, valid);
}

// ---------------------------------------------------------------------------
// Vectorized P1 ladder: same blocked-register scan as p1_transpose, but the
// load/store path moves 8-byte input vectors and 16-byte output vectors
// instead of one byte / one state per instruction.
//
// Exchanges are warp-local (each warp owns WARP * ITEMS_PER_THREAD
// consecutive elements), so they need only __syncwarp, no block barrier:
//   load:  lane loads 8-byte vectors striped over its warp's segment
//          (LDG.64, 256 contiguous bytes per warp instruction), writes them
//          to the warp's shmem buffer (STS.64) and reads back its own
//          ITEMS_PER_THREAD contiguous bytes (LDS.64); then byte -> state via
//          to_state[].
//   store: lane packs its states 8 per 16-byte vector, writes them blocked
//          (STS.128), reads back striped (LDS.128) and stores (STG.128, 512
//          contiguous bytes per warp instruction).
// With ITEMS_PER_THREAD = 24 all four shmem access patterns are free of bank
// conflicts (lane strides of 6 and 12 words). The last, partial tile uses a
// scalar path.
//   STEP 1: load/store only (like p1_ladder STEP 1).
//   STEP 2: + BlockScan, no lookback (like p1_ladder STEP 2).
//   STEP 3: + decoupled lookback (like p1_transpose).
// STATS: record lookback statistics (STEP 3).
// Other ITEMS_PER_THREAD multiples of 8 work but, unlike 24, have 2-4 way
// bank conflicts on the blocked shmem accesses (lane strides of 4/8 or
// 8/16 words).
// ---------------------------------------------------------------------------
template<uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD, uint32_t STEP, bool STATS = false>
__global__ LB_P1
void p1_vec(
    state_t* __restrict__ d_compose_glb,
    state_t* __restrict__ d_to_state_glb,
    const uint8_t* __restrict__ d_in,
    state_t* __restrict__ d_states_out,
    ScanTileState tile_state,
    uint32_t size)
{
    static_assert(STEP >= 1 && STEP <= 3, "STEP must be 1, 2 or 3");
    static_assert(ITEMS_PER_THREAD % 8 == 0, "ITEMS_PER_THREAD must be a multiple of 8");
    constexpr uint32_t VECS       = ITEMS_PER_THREAD / 8;   // 8-byte loads = 16-byte stores per lane
    constexpr uint32_t WARP_ITEMS = WARP * ITEMS_PER_THREAD;
    constexpr uint32_t TILE       = BLOCK_SIZE * ITEMS_PER_THREAD;

    using BlockScanT = cub::BlockScan<state_t, BLOCK_SIZE, cub::BLOCK_SCAN_WARP_SCANS>;
    using PrefixOp   = PrefixCallbackOp<ComposeOp, STATS>;

    // Per-warp exchange buffer: input bytes on load, output states on store.
    __shared__ __align__(16) uint32_t xbuf[BLOCK_SIZE / WARP][WARP_ITEMS * sizeof(state_t) / 4];
    __shared__ typename BlockScanT::TempStorage scan_temp;
    __shared__ typename PrefixOp::TempStorage   prefix_temp;
    __shared__ __align__(8) state_t shmem_compose[NUM_STATES * NUM_STATES];
    __shared__ __align__(8) state_t shmem_to_state[256];

    loadTablesToShmem<BLOCK_SIZE>(
        d_compose_glb, d_to_state_glb, shmem_compose, shmem_to_state);

    const uint32_t warp     = threadIdx.x / WARP;
    const uint32_t lane     = threadIdx.x % WARP;
    const uint32_t tile_idx = blockIdx.x;
    const uint32_t glb_offs = tile_idx * TILE;
    const bool     full     = glb_offs + TILE <= size;
    uint32_t*      wbuf     = xbuf[warp];

    state_t st[ITEMS_PER_THREAD];
    if (full) {
        const uint2* src = reinterpret_cast<const uint2*>(d_in + glb_offs + warp * WARP_ITEMS);
        uint2*       buf = reinterpret_cast<uint2*>(wbuf);
        uint2 v[VECS];
        #pragma unroll
        for (uint32_t k = 0; k < VECS; k++)
            v[k] = src[lane + k * WARP];
        #pragma unroll
        for (uint32_t k = 0; k < VECS; k++)
            buf[lane + k * WARP] = v[k];
        __syncwarp();
        #pragma unroll
        for (uint32_t k = 0; k < VECS; k++) {
            uint2 w = buf[lane * VECS + k];
            #pragma unroll
            for (uint32_t b = 0; b < 8; b++) {
                uint32_t word = b < 4 ? w.x : w.y;
                st[8 * k + b] = shmem_to_state[(word >> (8 * (b % 4))) & 0xffu];
            }
        }
    } else {
        #pragma unroll
        for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
            uint32_t idx = glb_offs + threadIdx.x * ITEMS_PER_THREAD + i;
            st[i] = idx < size ? shmem_to_state[d_in[idx]] : IDENTITY;
        }
    }

    if constexpr (STEP >= 2) {
        ComposeOp compose_op{shmem_compose};
        if (STEP == 2 || tile_idx == 0) {
            state_t block_aggregate;
            BlockScanT(scan_temp).InclusiveScan(st, st, compose_op, block_aggregate);
            if (STEP == 3 && threadIdx.x == 0)
                tile_state.SetInclusive(0, block_aggregate);
        } else {
            PrefixOp prefix_op(tile_state, prefix_temp, compose_op, (int)tile_idx, IDENTITY);
            BlockScanT(scan_temp).InclusiveScan(st, st, compose_op, prefix_op);
        }
    }

    if (full) {
        uint4* buf = reinterpret_cast<uint4*>(wbuf);
        __syncwarp();   // every lane has finished reading its input bytes
        #pragma unroll
        for (uint32_t k = 0; k < VECS; k++) {
            const state_t* s = st + 8 * k;
            buf[lane * VECS + k] = make_uint4(uint32_t(s[0]) | (uint32_t(s[1]) << 16),
                                              uint32_t(s[2]) | (uint32_t(s[3]) << 16),
                                              uint32_t(s[4]) | (uint32_t(s[5]) << 16),
                                              uint32_t(s[6]) | (uint32_t(s[7]) << 16));
        }
        __syncwarp();
        uint4* dst = reinterpret_cast<uint4*>(d_states_out + glb_offs + warp * WARP_ITEMS);
        #pragma unroll
        for (uint32_t k = 0; k < VECS; k++)
            dst[lane + k * WARP] = buf[lane + k * WARP];
    } else {
        #pragma unroll
        for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
            uint32_t idx = glb_offs + threadIdx.x * ITEMS_PER_THREAD + i;
            if (idx < size)
                d_states_out[idx] = st[i];
        }
    }
}

// ---------------------------------------------------------------------------
// cp.async helpers (sm_80+): 16-byte global -> shared copies that bypass
// registers and L1. On older architectures they fall back to a synchronous
// copy so the kernel logic can still be tested there.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
#if __CUDA_ARCH__ >= 800
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(gmem) : "memory");
#else
    *reinterpret_cast<uint4*>(smem) = *reinterpret_cast<const uint4*>(gmem);
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

// Waits until at most N of this thread's committed groups are still pending.
template<int N>
__device__ __forceinline__ void cp_async_wait() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory");
#endif
}

// ---------------------------------------------------------------------------
// Persistent p1_vec (STEP 3) with cp.async prefetch of the next tile.
//
// Each block processes tiles blockIdx.x, blockIdx.x + gridDim.x, ... . Before
// working on its current tile, it issues cp.async copies of its *next* tile's
// input into a second shmem buffer, so those loads are in flight during the
// current tile's scan, lookback and store (in p1_vec a block issues no loads
// while it waits in the lookback). Load/store otherwise as p1_vec: per-warp
// buffers, __syncwarp only, 16-byte output vectors.
//
// The grid must not exceed the number of co-resident blocks: a tile's
// lookback spins until its predecessors publish, which requires every block
// that owns an earlier tile to be running.
// ---------------------------------------------------------------------------
template<uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD, bool STATS = false>
__global__ LB_P1
void p1_vec_pipe(
    state_t* __restrict__ d_compose_glb,
    state_t* __restrict__ d_to_state_glb,
    const uint8_t* __restrict__ d_in,
    state_t* __restrict__ d_states_out,
    ScanTileState tile_state,
    uint32_t size,
    uint32_t num_tiles)
{
    static_assert(ITEMS_PER_THREAD % 8 == 0, "ITEMS_PER_THREAD must be a multiple of 8");
    constexpr uint32_t VECS       = ITEMS_PER_THREAD / 8;   // 8-byte reads = 16-byte stores per lane
    constexpr uint32_t WARP_ITEMS = WARP * ITEMS_PER_THREAD;
    constexpr uint32_t TILE       = BLOCK_SIZE * ITEMS_PER_THREAD;
    constexpr uint32_t CHUNKS     = WARP_ITEMS / 16;        // 16-byte cp.async chunks per warp
    static_assert(WARP_ITEMS % 16 == 0, "warp segment must be a multiple of 16 bytes");

    using BlockScanT = cub::BlockScan<state_t, BLOCK_SIZE, cub::BLOCK_SCAN_WARP_SCANS>;
    using PrefixOp   = PrefixCallbackOp<ComposeOp, STATS>;

    // Double-buffered per-warp input bytes, and a per-warp output buffer.
    __shared__ __align__(16) uint8_t  inbuf[2][BLOCK_SIZE / WARP][WARP_ITEMS];
    __shared__ __align__(16) uint32_t outbuf[BLOCK_SIZE / WARP][WARP_ITEMS * sizeof(state_t) / 4];
    __shared__ typename BlockScanT::TempStorage scan_temp;
    __shared__ typename PrefixOp::TempStorage   prefix_temp;
    __shared__ __align__(8) state_t shmem_compose[NUM_STATES * NUM_STATES];
    __shared__ __align__(8) state_t shmem_to_state[256];

    loadTablesToShmem<BLOCK_SIZE>(
        d_compose_glb, d_to_state_glb, shmem_compose, shmem_to_state);

    const uint32_t warp = threadIdx.x / WARP;
    const uint32_t lane = threadIdx.x % WARP;

    // Issues this warp's share of tile's input into inbuf[buf] (full tiles
    // only; the partial last tile is read directly), then commits a group —
    // always, so every lane has one group per call.
    auto prefetch = [&](uint32_t tile, uint32_t buf) {
        if (tile < num_tiles && tile * TILE + TILE <= size) {
            const uint8_t* src = d_in + tile * TILE + warp * WARP_ITEMS;
            uint8_t*       dst = inbuf[buf][warp];
            for (uint32_t c = lane; c < CHUNKS; c += WARP)
                cp_async16(dst + 16 * c, src + 16 * c);
        }
        cp_async_commit();
    };

    uint32_t buf = 0;
    prefetch(blockIdx.x, buf);
    for (uint32_t tile_idx = blockIdx.x; tile_idx < num_tiles; tile_idx += gridDim.x, buf ^= 1) {
        prefetch(tile_idx + gridDim.x, buf ^ 1);
        cp_async_wait<1>();   // this tile's group has landed (next tile's may be pending)
        __syncwarp();         // ... for every lane of the warp

        const uint32_t glb_offs = tile_idx * TILE;
        const bool     full     = glb_offs + TILE <= size;

        state_t st[ITEMS_PER_THREAD];
        if (full) {
            const uint2* in = reinterpret_cast<const uint2*>(inbuf[buf][warp]);
            #pragma unroll
            for (uint32_t k = 0; k < VECS; k++) {
                uint2 w = in[lane * VECS + k];
                #pragma unroll
                for (uint32_t b = 0; b < 8; b++) {
                    uint32_t word = b < 4 ? w.x : w.y;
                    st[8 * k + b] = shmem_to_state[(word >> (8 * (b % 4))) & 0xffu];
                }
            }
        } else {
            #pragma unroll
            for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
                uint32_t idx = glb_offs + threadIdx.x * ITEMS_PER_THREAD + i;
                st[i] = idx < size ? shmem_to_state[d_in[idx]] : IDENTITY;
            }
        }

        ComposeOp compose_op{shmem_compose};
        if (tile_idx == 0) {
            state_t block_aggregate;
            BlockScanT(scan_temp).InclusiveScan(st, st, compose_op, block_aggregate);
            if (threadIdx.x == 0)
                tile_state.SetInclusive(0, block_aggregate);
        } else {
            PrefixOp prefix_op(tile_state, prefix_temp, compose_op, (int)tile_idx, IDENTITY);
            BlockScanT(scan_temp).InclusiveScan(st, st, compose_op, prefix_op);
        }

        if (full) {
            uint4* ob = reinterpret_cast<uint4*>(outbuf[warp]);
            #pragma unroll
            for (uint32_t k = 0; k < VECS; k++) {
                const state_t* s = st + 8 * k;
                ob[lane * VECS + k] = make_uint4(uint32_t(s[0]) | (uint32_t(s[1]) << 16),
                                                 uint32_t(s[2]) | (uint32_t(s[3]) << 16),
                                                 uint32_t(s[4]) | (uint32_t(s[5]) << 16),
                                                 uint32_t(s[6]) | (uint32_t(s[7]) << 16));
            }
            __syncwarp();
            uint4* dst = reinterpret_cast<uint4*>(d_states_out + glb_offs + warp * WARP_ITEMS);
            #pragma unroll
            for (uint32_t k = 0; k < VECS; k++)
                dst[lane + k * WARP] = ob[lane + k * WARP];
        } else {
            #pragma unroll
            for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
                uint32_t idx = glb_offs + threadIdx.x * ITEMS_PER_THREAD + i;
                if (idx < size)
                    d_states_out[idx] = st[i];
            }
        }
        // BlockScan / PrefixOp temp storage and outbuf are reused next tile.
        __syncthreads();
    }
    cp_async_wait<0>();
}

// Host reference for ladder / P1 output. scan == false: plain byte->state map.
// scan == true: inclusive compose scan that restarts every tile_len elements
// (tile_len == 0: one scan over the whole input, i.e. real P1 output).
static bool check_states(const uint8_t* input, uint32_t size, const state_t* out,
                         bool scan, uint32_t tile_len, const char* name) {
    state_t acc = IDENTITY;
    for (uint32_t i = 0; i < size; i++) {
        state_t x = h_to_state[input[i]];
        if (!scan)
            acc = x;
        else if (tile_len != 0 && i % tile_len == 0)
            acc = x;
        else
            acc = i == 0 ? x : h_compose[(x & 15u) * NUM_STATES + (acc & 15u)];
        if (out[i] != acc) {
            fprintf(stderr, "%s mismatch at %u: got %u, expected %u\n",
                    name, i, (uint32_t)out[i], (uint32_t)acc);
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Launch / bench helpers
// ---------------------------------------------------------------------------

static const uint32_t BLOCK_SIZE       = 256;
static const uint32_t ITEMS_PER_THREAD = 22;
#ifdef PROFILE
static const uint32_t WARMUP_RUNS      = 1;
static const uint32_t BENCH_RUNS       = 1;
#else
static const uint32_t WARMUP_RUNS      = 500;
static const uint32_t BENCH_RUNS       = 100;
#endif

static uint32_t num_tiles(uint32_t size) {
    return (size + BLOCK_SIZE * ITEMS_PER_THREAD - 1) / (BLOCK_SIZE * ITEMS_PER_THREAD);
}

static void reset(ScanTileState& ts, uint32_t nlb) {
    initScanTileState(ts, (int)nlb);
}

// Requests the maximum shared memory carveout for kernel (8 blocks/SM x
// ~14 KB exceeds the default 100 KB configuration) and returns the resulting
// resident blocks per SM.
template<typename KernelT>
static int max_shared_blocks_per_sm(KernelT kernel) {
    gpuAssert(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                   (int)cudaSharedmemCarveoutMaxShared));
    int blocks = 0;
    gpuAssert(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, BLOCK_SIZE, 0));
    return blocks;
}

static void reset_lookback_stats() {
    void* p;
    gpuAssert(cudaGetSymbolAddress(&p, g_lookback_stats));
    gpuAssert(cudaMemset(p, 0, sizeof(unsigned long long) * STAT_SLOTS * STAT_COUNT));
}

static void print_lookback_stats(const char* name) {
    unsigned long long raw[STAT_SLOTS * STAT_COUNT];
    gpuAssert(cudaMemcpyFromSymbol(raw, g_lookback_stats, sizeof(raw)));
    unsigned long long s[STAT_COUNT] = {};
    for (uint32_t slot = 0; slot < STAT_SLOTS; slot++)
        for (uint32_t i = 0; i < STAT_COUNT; i++)
            s[i] += raw[slot * STAT_COUNT + i];
    double t = (double)s[STAT_TILES];
    auto pct = [&](uint32_t i) { return 100.0 * s[i] / t; };
    printf("%s  (%llu tile lookbacks)\n", name, s[STAT_TILES]);
    if (s[STAT_TILES] == 0) return;
    printf("  first poll of tile-1:  invalid %5.1f%%  partial %5.1f%%  inclusive %5.1f%%\n",
           pct(STAT_FIRST_INVALID), pct(STAT_FIRST_PARTIAL), pct(STAT_FIRST_INCLUSIVE));
    printf("  re-polls per tile:     %.2f  (0: %.1f%%  1: %.1f%%  2+: %.1f%%)\n",
           s[STAT_RETRIES] / t, pct(STAT_RETRY_0), pct(STAT_RETRY_1), pct(STAT_RETRY_2PLUS));
    printf("  windows per tile:      %.2f  (1: %.1f%%  2: %.1f%%  3+: %.1f%%)\n",
           s[STAT_WINDOWS] / t, pct(STAT_WIN_1), pct(STAT_WIN_2), pct(STAT_WIN_3PLUS));
    printf("  depth to inclusive:    %.2f  (0: %.1f%%  1-7: %.1f%%  8-31: %.1f%%  32-63: %.1f%%  64+: %.1f%%)\n",
           s[STAT_DEPTH_SUM] / t, pct(STAT_DEPTH_0), pct(STAT_DEPTH_1_7), pct(STAT_DEPTH_8_31),
           pct(STAT_DEPTH_32_63), pct(STAT_DEPTH_64PLUS));
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    // Read input
    FILE* f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END);
    size_t input_size = (size_t)ftell(f);
    rewind(f);
    uint8_t* input = (uint8_t*)malloc(input_size);
    assert(fread(input, 1, input_size, f) == input_size);
    fclose(f);

    assert(input_size <= UINT32_MAX && "input exceeds uint32_t range");
    uint32_t size = (uint32_t)input_size;
    uint32_t nlb  = num_tiles(size);

    printf("%s  (%zu bytes, %u tiles)\n\n", argv[1], input_size, nlb);

    // Allocate GPU buffers
    uint8_t*  d_in;
    state_t*  d_states_out;
    state_t*  d_compose_glb;
    state_t*  d_to_state_glb;
    ScanTileState ts;

    gpuAssert(cudaMalloc(&d_in,           (size_t)size * sizeof(uint8_t)));
    gpuAssert(cudaMalloc(&d_states_out,   (size_t)size * sizeof(state_t)));
    gpuAssert(cudaMalloc(&d_compose_glb,  sizeof(h_compose)));
    gpuAssert(cudaMalloc(&d_to_state_glb, sizeof(h_to_state)));
    gpuAssert(cudaMalloc(&ts.d_tile_descriptors, ScanTileState::AllocationSize(nlb)));

    gpuAssert(cudaMemcpy(d_in,           input,     (size_t)size * sizeof(uint8_t), cudaMemcpyHostToDevice));
    gpuAssert(cudaMemcpy(d_compose_glb,  h_compose, sizeof(h_compose),              cudaMemcpyHostToDevice));
    gpuAssert(cudaMemcpy(d_to_state_glb, h_to_state, sizeof(h_to_state),            cudaMemcpyHostToDevice));

    float* ms = (float*)malloc(BENCH_RUNS * sizeof(float));
    cudaEvent_t t0, t1;
    gpuAssert(cudaEventCreate(&t0));
    gpuAssert(cudaEventCreate(&t1));

    // ------------------------------------------------------------------
    // Bench helper. prep() runs untimed before every launch.
    // GB/s counts input bytes read + output bytes written.
    // ------------------------------------------------------------------
    auto bench = [&](const char* name, size_t bytes, auto prep, auto launch) {
        printf("%-40s ", name);
        fflush(stdout);   // so a hanging kernel is identifiable from the output
        for (uint32_t i = 0; i < WARMUP_RUNS; i++) {
            prep();
            launch();
            gpuAssert(cudaDeviceSynchronize());
        }
        for (uint32_t i = 0; i < BENCH_RUNS; i++) {
            prep();
            gpuAssert(cudaEventRecord(t0));
            launch();
            gpuAssert(cudaDeviceSynchronize());
            gpuAssert(cudaEventRecord(t1));
            gpuAssert(cudaEventSynchronize(t1));
            gpuAssert(cudaEventElapsedTime(ms + i, t0, t1));
        }
        print_stats(ms, BENCH_RUNS, bytes);
    };
    auto no_prep = [] {};

    const size_t u16_bytes = (size_t)size * sizeof(uint8_t) + (size_t)size * sizeof(state_t);
    state_t* h_states = (state_t*)malloc((size_t)size * sizeof(state_t));
    auto fetch_states = [&] {
        gpuAssert(cudaGetLastError());   // catches launch-config failures
        gpuAssert(cudaDeviceSynchronize());
        gpuAssert(cudaMemcpy(h_states, d_states_out, (size_t)size * sizeof(state_t),
                             cudaMemcpyDeviceToHost));
    };
    auto poison_out = [&] {
        gpuAssert(cudaMemset(d_states_out, 0xff, (size_t)size * sizeof(state_t)));
    };

    // ------------------------------------------------------------------
    // P1 cost ladder (all u16 output, same traffic as P1)
    // ------------------------------------------------------------------
    printf("P1 cost ladder (u16 out):\n");

    // L0: map-only, coalesced stores
    {
        constexpr uint32_t CHUNKS = 8;
        auto kernel     = map_only_u16_coalesced<BLOCK_SIZE, CHUNKS>;
        uint32_t nvec   = size / 8;
        uint32_t blocks = max(1u, (nvec + BLOCK_SIZE * CHUNKS - 1) / (BLOCK_SIZE * CHUNKS));
        auto launch = [&] { kernel<<<blocks, BLOCK_SIZE>>>(d_to_state_glb, d_in, d_states_out, size); };
        poison_out(); launch(); fetch_states();
        if (!check_states(input, size, h_states, false, 0, "L0")) exit(1);
        bench("L0 map-only (coalesced):", u16_bytes, no_prep, launch);
    }

    // L1: + warp-transpose load/store
    {
        auto kernel = p1_ladder<BLOCK_SIZE, ITEMS_PER_THREAD, 1>;
        auto launch = [&] { kernel<<<nlb, BLOCK_SIZE>>>(d_compose_glb, d_to_state_glb, d_in, d_states_out, size); };
        poison_out(); launch(); fetch_states();
        if (!check_states(input, size, h_states, false, 0, "L1")) exit(1);
        bench("L1 + warp-transpose load/store:", u16_bytes, no_prep, launch);
    }

    // L2: + block scan, no lookback
    {
        auto kernel = p1_ladder<BLOCK_SIZE, ITEMS_PER_THREAD, 2>;
        auto launch = [&] { kernel<<<nlb, BLOCK_SIZE>>>(d_compose_glb, d_to_state_glb, d_in, d_states_out, size); };
        poison_out(); launch(); fetch_states();
        if (!check_states(input, size, h_states, true, BLOCK_SIZE * ITEMS_PER_THREAD, "L2")) exit(1);
        bench("L2 + block scan (no lookback):", u16_bytes, no_prep, launch);
    }

    // L3: + decoupled lookback = p1_transpose
    auto run_l3 = [&](auto kernel, const char* name) {
        auto prep   = [&] { reset(ts, nlb); };
        auto launch = [&] { kernel<<<nlb, BLOCK_SIZE>>>(d_compose_glb, d_to_state_glb, d_in, d_states_out, ts, size, nlb); };
        poison_out(); prep(); launch(); fetch_states();
        if (!check_states(input, size, h_states, true, 0, name)) exit(1);
        bench(name, u16_bytes, prep, launch);
    };
    run_l3(p1_transpose<BLOCK_SIZE, ITEMS_PER_THREAD>, "L3 + lookback (p1_transpose):");

    // Vectorized load/store ladder at IPT=24: V1 ~ L1, V2 ~ L2, V3 ~ L3, then
    // V3 as a persistent kernel that prefetches its next tile with cp.async.
    printf("\nVectorized load/store (IPT=24, u16 out):\n");
    constexpr uint32_t VEC_IPT = 24;
    const uint32_t vec_tiles = (size + BLOCK_SIZE * VEC_IPT - 1) / (BLOCK_SIZE * VEC_IPT);
    assert(vec_tiles <= nlb && "tile state array is sized for nlb tiles");
    auto run_vec = [&](auto kernel, uint32_t step, const char* name) {
        auto prep   = [&] { if (step == 3) reset(ts, vec_tiles); };
        auto launch = [&] {
            kernel<<<vec_tiles, BLOCK_SIZE>>>(d_compose_glb, d_to_state_glb, d_in, d_states_out, ts, size);
        };
        poison_out(); prep(); launch(); fetch_states();
        if (!check_states(input, size, h_states, step >= 2,
                          step == 2 ? BLOCK_SIZE * VEC_IPT : 0, name)) exit(1);
        bench(name, u16_bytes, prep, launch);
    };
    run_vec(p1_vec<BLOCK_SIZE, VEC_IPT, 1>, 1, "V1 vec load/store:");
    run_vec(p1_vec<BLOCK_SIZE, VEC_IPT, 2>, 2, "V2 + block scan (no lookback):");
    run_vec(p1_vec<BLOCK_SIZE, VEC_IPT, 3>, 3, "V3 + lookback:");

    // Persistent grid: every block must be resident at once (the lookback
    // spins on predecessor tiles), so grid = resident blocks/SM x SMs.
    int num_sms = 0;
    gpuAssert(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
    auto pipe_grid = [&](auto kernel, int& bps) {
        bps = max_shared_blocks_per_sm(kernel);   // ~25 KB shmem/block
        if (bps < 1) { fprintf(stderr, "p1_vec_pipe does not fit on an SM\n"); exit(1); }
        return std::min(vec_tiles, (uint32_t)(bps * num_sms));
    };
    {
        auto kernel = p1_vec_pipe<BLOCK_SIZE, VEC_IPT>;
        int bps;
        const uint32_t grid = pipe_grid(kernel, bps);
        char name[64];
        snprintf(name, sizeof(name), "V3 persistent + cp.async [%d/SM]:", bps);
        auto prep   = [&] { reset(ts, vec_tiles); };
        auto launch = [&] {
            kernel<<<grid, BLOCK_SIZE>>>(d_compose_glb, d_to_state_glb, d_in, d_states_out, ts, size, vec_tiles);
        };
        poison_out(); prep(); launch(); fetch_states();
        if (!check_states(input, size, h_states, true, 0, name)) exit(1);
        bench(name, u16_bytes, prep, launch);
    }

    // Lookback statistics from instrumented copies of L3 and the V3 kernels,
    // summed over STATS_LAUNCHES launches. Diagnostic only: the counter
    // atomics perturb timing, so no times are reported.
    constexpr uint32_t STATS_LAUNCHES = 10;
    printf("\nLookback stats (instrumented, %u launches):\n", STATS_LAUNCHES);
    auto run_stats = [&](uint32_t tiles, auto launch_kernel, const char* name) {
        auto launch = [&] { reset(ts, tiles); launch_kernel(); };
        poison_out(); launch(); fetch_states();
        if (!check_states(input, size, h_states, true, 0, name)) exit(1);
        reset_lookback_stats();
        for (uint32_t i = 0; i < STATS_LAUNCHES; i++) {
            launch();
            gpuAssert(cudaDeviceSynchronize());
        }
        print_lookback_stats(name);
    };
    run_stats(nlb, [&] {
        p1_transpose<BLOCK_SIZE, ITEMS_PER_THREAD, true><<<nlb, BLOCK_SIZE>>>(
            d_compose_glb, d_to_state_glb, d_in, d_states_out, ts, size, nlb);
    }, "L3");
    run_stats(vec_tiles, [&] {
        p1_vec<BLOCK_SIZE, VEC_IPT, 3, true><<<vec_tiles, BLOCK_SIZE>>>(
            d_compose_glb, d_to_state_glb, d_in, d_states_out, ts, size);
    }, "V3");
    {
        auto kernel = p1_vec_pipe<BLOCK_SIZE, VEC_IPT, true>;
        int bps;
        const uint32_t grid = pipe_grid(kernel, bps);
        run_stats(vec_tiles, [&] {
            kernel<<<grid, BLOCK_SIZE>>>(d_compose_glb, d_to_state_glb, d_in, d_states_out, ts, size, vec_tiles);
        }, "V3 persistent + cp.async");
    }

    free(h_states);

    // ------------------------------------------------------------------
    // Speed-of-light references: map-only kernels and D2D memcpy.
    // ------------------------------------------------------------------
    printf("\nSpeed-of-light references:\n");
    uint8_t* h_check = (uint8_t*)malloc((size_t)size * sizeof(state_t));
    auto run_map = [&](auto out_bits_tag, const char* name) {
        constexpr uint32_t OUT_BITS = decltype(out_bits_tag)::value;
        constexpr uint32_t CHUNKS   = 4;
        auto kernel      = map_only<OUT_BITS, BLOCK_SIZE, CHUNKS>;
        uint32_t nvec    = size / 16;
        uint32_t blocks  = max(1u, (nvec + BLOCK_SIZE * CHUNKS - 1) / (BLOCK_SIZE * CHUNKS));
        uint32_t out_len = map_out_bytes(size, OUT_BITS);
        uint8_t* d_out   = reinterpret_cast<uint8_t*>(d_states_out);
        auto launch = [&] { kernel<<<blocks, BLOCK_SIZE>>>(d_to_state_glb, d_in, d_out, size); };

        gpuAssert(cudaMemset(d_out, 0xff, out_len));
        launch();
        gpuAssert(cudaDeviceSynchronize());
        gpuAssert(cudaMemcpy(h_check, d_out, out_len, cudaMemcpyDeviceToHost));
        if (!check_map_only(input, size, OUT_BITS, h_check)) exit(1);

        bench(name, (size_t)size + out_len, no_prep, launch);
    };
    run_map(std::integral_constant<uint32_t, 16>{}, "SoL map-only (u16 out, strided store):");
    run_map(std::integral_constant<uint32_t, 8>{},  "SoL map-only (u8 out):");
    run_map(std::integral_constant<uint32_t, 4>{},  "SoL map-only (4-bit out):");
    free(h_check);

    bench("SoL memcpy D2D (1 B in, 1 B out):", 2 * (size_t)size, no_prep, [&] {
        gpuAssert(cudaMemcpyAsync(d_states_out, d_in, size, cudaMemcpyDeviceToDevice));
    });

    // Cleanup
    free(ms); free(input);
    gpuAssert(cudaFree(d_in));
    gpuAssert(cudaFree(d_states_out));
    gpuAssert(cudaFree(d_compose_glb));
    gpuAssert(cudaFree(d_to_state_glb));
    gpuAssert(cudaFree(ts.d_tile_descriptors));
    return 0;
}
