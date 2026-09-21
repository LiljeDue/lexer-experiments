// p1_bench.cu — standalone benchmark for the two-pass lexer P1 kernel.
//
// Pass 1 (P1) is the bottleneck: it maps each input byte to a DFA state,
// then runs an inclusive prefix scan over those states using the DFA
// composition table as the scan operator. The result is a flat array of
// prefix states, one per input byte, written to global memory for P2.
//
// Three variants are benchmarked here:
//
//   BW ceiling  — reads input using the same u64 load pattern as P1,
//                 writes a dummy result. Upper bound on achievable P1 speed.
//
//   NregNone    — full P1: u64 coalesced loads, byte→state lookup via
//                 to_state[] in shmem, inclusive scan with DFA compose
//                 table (also in shmem), u64 coalesced writes.
//
//   Add scan    — same as NregNone but replaces the DFA compose table
//                 lookup with plain integer addition. Results are
//                 meaningless but the timing isolates how much of the P1
//                 cost comes from the composition operator vs the scan
//                 synchronisation overhead itself.
//
// Usage: ./p1_bench <input_file>
//   input_file: raw bytes, e.g. data/tokens_dense_500MiB.in

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>
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

    __device__ __forceinline__ void WaitForValid(int tile_idx,
                                                  uint32_t& status,
                                                  state_t& value,
                                                  uint32_t initial_delay_ns = 450) {
        __nanosleep(initial_delay_ns);
        uint32_t w;
        asm volatile("ld.relaxed.gpu.u32 %0, [%1];"
                     : "=r"(w)
                     : "l"(d_tile_descriptors + TILE_STATUS_PADDING + tile_idx)
                     : "memory");
        while (__any_sync(0xffffffff, (w & 0xffffu) == uint32_t(SCAN_TILE_INVALID))) {
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

// Prefix callback used with CUB BlockScan (decoupled lookback).
template<typename ScanOpT>
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
                  uint32_t delay_ns = 350) {
        state_t value;
        tile_state.WaitForValid(predecessor_idx, predecessor_status, value, delay_ns);
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
        exclusive_prefix = ProcessWindow(predecessor_idx, predecessor_status, initial_delay);
        while (__all_sync(0xffffffff,
                          predecessor_status != uint32_t(SCAN_TILE_INCLUSIVE) &&
                          predecessor_status != uint32_t(SCAN_TILE_OOB))) {
            predecessor_idx -= WARP;
            state_t w = ProcessWindow(predecessor_idx, predecessor_status);
            exclusive_prefix = scan_op(w, exclusive_prefix);
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

__device__ inline uint32_t dynamicIndex(volatile uint32_t* ptr) {
    volatile __shared__ uint32_t idx;
    if (threadIdx.x == 0) idx = atomicAdd(const_cast<uint32_t*>(ptr), 1);
    __syncthreads();
    return idx;
}

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

struct AddOp {
    __device__ __forceinline__ state_t
    operator()(state_t a, state_t b) const { return a + b; }
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

// Load ITEMS_PER_THREAD bytes per thread from d_in[glb_offs..] using
// 64-bit coalesced loads, map each byte through to_state[], write
// results to shmem in blocked layout. Thread t writes to
// [t*ITEMS_PER_THREAD, (t+1)*ITEMS_PER_THREAD).
template<uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD>
__device__ inline void loadBytesAsStates(
    const uint8_t* __restrict__ d_in,
    uint32_t glb_offs, uint32_t size,
    const volatile state_t* __restrict__ to_state,
    volatile state_t* states,
    state_t identity)
{
    const uint32_t U8    = sizeof(uint64_t);
    const uint32_t TILE  = ITEMS_PER_THREAD * BLOCK_SIZE;
    const uint32_t LOADS = 1 + ITEMS_PER_THREAD / U8;
    #pragma unroll
    for (uint32_t i = 0; i < LOADS; i++) {
        uint32_t base_byte = (i * BLOCK_SIZE + threadIdx.x) * U8;
        uint32_t gid       = glb_offs + base_byte;
        uint64_t reg;
        uint8_t* bytes = (uint8_t*)&reg;
        if (gid + U8 <= size)
            reg = *reinterpret_cast<const uint64_t*>(d_in + gid);
        else {
            reg = 0;
            #pragma unroll
            for (uint32_t j = 0; j < U8; j++)
                if (gid + j < size) bytes[j] = d_in[gid + j];
        }
        #pragma unroll
        for (uint32_t j = 0; j < U8; j++) {
            uint32_t lid = base_byte + j;
            if (lid < TILE) {
                uint32_t t   = lid / ITEMS_PER_THREAD;
                uint32_t off = lid % ITEMS_PER_THREAD;
                states[t * ITEMS_PER_THREAD + off] = (glb_offs + lid < size)
                                                 ? to_state[bytes[j]] : identity;
            }
        }
    }
}

// Write IPT*BLOCK_SIZE state_t values from shmem to global memory using
// 64-bit coalesced stores (standard, unpadded layout only).
template<uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD>
__device__ inline void writeShmemToGlb(
    uint32_t glb_offs, uint32_t size,
    const volatile state_t* shmem,
    state_t* __restrict__ d_out)
{
    const uint32_t U8        = sizeof(uint64_t);
    const uint32_t NUM_BYTES = min(ITEMS_PER_THREAD * BLOCK_SIZE,
                                   size - glb_offs) * sizeof(state_t);
    uint8_t* d_out_bytes = reinterpret_cast<uint8_t*>(d_out) + glb_offs * sizeof(state_t);
    const uint32_t TOTAL_STORES = 1 + (NUM_BYTES - 1) / U8;
    const uint32_t ITERS        = 1 + (TOTAL_STORES - 1) / (ITEMS_PER_THREAD * BLOCK_SIZE);
    #pragma unroll
    for (uint32_t j = 0; j < ITERS; j++) {
        #pragma unroll
        for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
            uint32_t lid      = j * ITEMS_PER_THREAD * BLOCK_SIZE + i * BLOCK_SIZE + threadIdx.x;
            uint32_t lid_byte = lid * U8;
            if (lid_byte + U8 < NUM_BYTES)
                reinterpret_cast<uint64_t*>(d_out_bytes)[lid] =
                    reinterpret_cast<const volatile uint64_t*>(shmem)[lid];
            else {
                #pragma unroll
                for (uint32_t k = lid_byte; k < NUM_BYTES; k++)
                    d_out_bytes[k] =
                        reinterpret_cast<const volatile uint8_t*>(shmem)[k];
            }
        }
    }
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

// Full P1: load bytes, map to states via to_state[], inclusive prefix scan
// using DFA composition, write prefixed states to d_states_out.
// Uses static blockIdx.x tile assignment (no atomic counter) and a tile-0
// fast path (no lookback needed for the first tile).
template<uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD>
__global__ LB_P1
void p1_nregnone(
    state_t* __restrict__ d_compose_glb,
    state_t* __restrict__ d_to_state_glb,
    const uint8_t* __restrict__ d_in,
    state_t* __restrict__ d_states_out,
    ScanTileState tile_state,
    uint32_t size,
    uint32_t num_tiles)
{
    using BlockScan  = cub::BlockScan<state_t, BLOCK_SIZE>;
    using PrefixOp   = PrefixCallbackOp<ComposeOp>;

    __shared__ typename BlockScan::TempStorage scan_tmp;
    __shared__ typename PrefixOp::TempStorage  prefix_tmp;
    __shared__ __align__(8) state_t shmem_compose[NUM_STATES * NUM_STATES];
    __shared__ __align__(8) state_t shmem_to_state[256];
    extern __shared__ uint16_t dyn_shmem[];
    volatile state_t* states = (volatile state_t*) dyn_shmem;

    loadTablesToShmem<BLOCK_SIZE>(
        d_compose_glb, d_to_state_glb, shmem_compose, shmem_to_state);

    uint32_t tile_idx = blockIdx.x;
    uint32_t glb_offs = tile_idx * BLOCK_SIZE * ITEMS_PER_THREAD;

    loadBytesAsStates<BLOCK_SIZE, ITEMS_PER_THREAD>(
        d_in, glb_offs, size, shmem_to_state, states, IDENTITY);
    __syncthreads();

    state_t st[ITEMS_PER_THREAD];
    #pragma unroll
    for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++)
        st[i] = states[threadIdx.x * ITEMS_PER_THREAD + i];

    ComposeOp compose_op{shmem_compose};

    if (tile_idx == 0) {
        // First tile: no lookback needed — scan and publish inclusive aggregate.
        state_t block_aggregate;
        BlockScan(scan_tmp).InclusiveScan(st, st, compose_op, block_aggregate);
        if (threadIdx.x == 0)
            tile_state.SetInclusive(0, block_aggregate);
    } else {
        PrefixOp prefix_op(tile_state, prefix_tmp, compose_op, (int)tile_idx, IDENTITY);
        BlockScan(scan_tmp).InclusiveScan(st, st, compose_op, prefix_op);
    }

    #pragma unroll
    for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++)
        states[threadIdx.x * ITEMS_PER_THREAD + i] = st[i];
    __syncthreads();
    writeShmemToGlb<BLOCK_SIZE, ITEMS_PER_THREAD>(glb_offs, size, states, d_states_out);
}

// P1 kernel matching CUB DeviceScan DefaultPolicy exactly:
//   128 threads, 15 IPT, BLOCK_LOAD_WARP_TRANSPOSE, BLOCK_STORE_WARP_TRANSPOSE,
//   BLOCK_SCAN_WARP_SCANS — but with compose table in shmem (DeviceScan uses global).
// Load/store shmem is a union with scan/prefix shmem to minimise footprint.
template<uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD>
__global__
void p1_cub_style(
    state_t* __restrict__ d_compose_glb,
    state_t* __restrict__ d_to_state_glb,
    const uint8_t* __restrict__ d_in,
    state_t* __restrict__ d_states_out,
    ScanTileState tile_state,
    uint32_t size,
    uint32_t num_tiles,
    volatile uint32_t* dyn_index_ptr)
{
    using TransformIter = thrust::transform_iterator<ByteToState, const uint8_t*>;
    using BlockLoadT  = cub::BlockLoad <state_t, BLOCK_SIZE, ITEMS_PER_THREAD,
                                        cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockStoreT = cub::BlockStore<state_t, BLOCK_SIZE, ITEMS_PER_THREAD,
                                        cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockScanT  = cub::BlockScan <state_t, BLOCK_SIZE,
                                        cub::BLOCK_SCAN_WARP_SCANS>;
    using PrefixOp    = PrefixCallbackOp<ComposeOp>;

    // Union shmem: load/store share space with prefix/scan (used at different times).
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

    // Load tables into shmem — reuse load union slot before scan phase.
    for (uint32_t i = threadIdx.x; i < NUM_STATES * NUM_STATES / 4; i += BLOCK_SIZE)
        reinterpret_cast<volatile uint64_t*>(shmem_compose)[i] =
            reinterpret_cast<uint64_t*>(d_compose_glb)[i];
    for (uint32_t i = threadIdx.x; i < 256 / 4; i += BLOCK_SIZE)
        reinterpret_cast<volatile uint64_t*>(shmem_to_state)[i] =
            reinterpret_cast<uint64_t*>(d_to_state_glb)[i];
    __syncthreads();

    uint32_t tile_idx = dynamicIndex(dyn_index_ptr);
    uint32_t glb_offs = tile_idx * BLOCK_SIZE * ITEMS_PER_THREAD;
    uint32_t valid    = (uint32_t)min((uint64_t)BLOCK_SIZE * ITEMS_PER_THREAD,
                                      (uint64_t)size - glb_offs);

    // Load: striped→blocked transpose through shmem (BLOCK_LOAD_WARP_TRANSPOSE).
    ByteToState byte_to_state{shmem_to_state};
    TransformIter d_in_states(d_in + glb_offs, byte_to_state);
    state_t st[ITEMS_PER_THREAD];
    if (glb_offs + BLOCK_SIZE * ITEMS_PER_THREAD <= size)
        BlockLoadT(temp.load).Load(d_in_states, st);
    else
        BlockLoadT(temp.load).Load(d_in_states, st, valid, IDENTITY);
    __syncthreads();

    // Scan + lookback using scan_storage union slot.
    ComposeOp compose_op{shmem_compose};
    PrefixOp  prefix_op(tile_state, temp.scan_storage.prefix, compose_op, (int)tile_idx, IDENTITY);
    BlockScanT(temp.scan_storage.scan).InclusiveScan(st, st, compose_op, prefix_op);
    __syncthreads();

    // Store: blocked→striped transpose through shmem (BLOCK_STORE_WARP_TRANSPOSE).
    if (glb_offs + BLOCK_SIZE * ITEMS_PER_THREAD <= size)
        BlockStoreT(temp.store).Store(d_states_out + glb_offs, st);
    else
        BlockStoreT(temp.store).Store(d_states_out + glb_offs, st, valid);
}

// Add scan P1: identical to p1_nregnone but uses integer addition as the
// scan operator instead of DFA composition.  No compose table needed.
// Output values are meaningless; timing isolates scan overhead vs composition.
template<uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD>
__global__ LB_P1
void p1_add(
    state_t* __restrict__ d_to_state_glb,
    const uint8_t* __restrict__ d_in,
    state_t* __restrict__ d_states_out,
    ScanTileState tile_state,
    uint32_t size,
    uint32_t num_tiles,
    volatile uint32_t* dyn_index_ptr)
{
    using BlockScan  = cub::BlockScan<state_t, BLOCK_SIZE>;
    using PrefixOp   = PrefixCallbackOp<AddOp>;

    __shared__ typename BlockScan::TempStorage scan_tmp;
    __shared__ typename PrefixOp::TempStorage  prefix_tmp;
    __shared__ __align__(8) state_t shmem_to_state[256];
    extern __shared__ uint16_t dyn_shmem_add[];
    volatile state_t* states = (volatile state_t*) dyn_shmem_add;

    for (uint32_t i = threadIdx.x; i < 256 / 4; i += BLOCK_SIZE)
        reinterpret_cast<volatile uint64_t*>(shmem_to_state)[i] =
            reinterpret_cast<uint64_t*>(d_to_state_glb)[i];
    __syncthreads();

    uint32_t tile_idx = dynamicIndex(dyn_index_ptr);
    uint32_t glb_offs = tile_idx * BLOCK_SIZE * ITEMS_PER_THREAD;

    loadBytesAsStates<BLOCK_SIZE, ITEMS_PER_THREAD>(
        d_in, glb_offs, size, shmem_to_state, states, state_t(0));
    __syncthreads();

    state_t st[ITEMS_PER_THREAD];
    #pragma unroll
    for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++)
        st[i] = states[threadIdx.x * ITEMS_PER_THREAD + i];

    AddOp    add_op;
    PrefixOp prefix_op(tile_state, prefix_tmp, add_op, (int)tile_idx, state_t(0));
    BlockScan(scan_tmp).InclusiveScan(st, st, add_op, prefix_op);

    #pragma unroll
    for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++)
        states[threadIdx.x * ITEMS_PER_THREAD + i] = st[i];
    __syncthreads();

    writeShmemToGlb<BLOCK_SIZE, ITEMS_PER_THREAD>(glb_offs, size, states, d_states_out);
}

// BW ceiling: reads input using the same u64 pattern as P1, XORs into a
// per-thread accumulator, writes one u64 per thread to prevent DCE.
// No scan, no shmem tables.  Establishes an upper bound for P1 throughput.
template<uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD>
__global__ void
bw_ceiling(const uint8_t* __restrict__ d_in, uint32_t size, uint64_t* d_out)
{
    const uint32_t U8      = sizeof(uint64_t);
    const uint32_t LOADS   = 1 + ITEMS_PER_THREAD / U8;
    uint32_t       glb_offs = blockIdx.x * BLOCK_SIZE * ITEMS_PER_THREAD;
    uint64_t regs[LOADS];
    uint8_t* bytes = (uint8_t*)regs;
    #pragma unroll
    for (uint32_t i = 0; i < LOADS; i++) {
        uint32_t gid = glb_offs + (i * BLOCK_SIZE + threadIdx.x) * U8;
        regs[i] = 0;
        if (gid + U8 <= size)
            regs[i] = __ldg(reinterpret_cast<const uint64_t*>(d_in + gid));
        else {
            #pragma unroll
            for (uint32_t j = 0; j < U8; j++)
                if (gid + j < size) bytes[i * U8 + j] = d_in[gid + j];
        }
    }
    uint64_t acc = 0;
    #pragma unroll
    for (uint32_t i = 0; i < LOADS; i++) acc ^= regs[i];
    d_out[blockIdx.x * BLOCK_SIZE + threadIdx.x] = acc;
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

// Shmem for p1_nregnone: IPT*BS state_t entries (compose table is in static shmem).
static size_t p1_nregnone_shmem(uint32_t size) {
    (void)size;
    return (size_t)ITEMS_PER_THREAD * BLOCK_SIZE * sizeof(state_t);
}

// Shmem for p1_add: same, no compose table needed.
static size_t p1_add_shmem(uint32_t size) {
    (void)size;
    return (size_t)ITEMS_PER_THREAD * BLOCK_SIZE * sizeof(state_t);
}

static void reset(ScanTileState& ts, uint32_t* d_dyn, uint32_t nlb) {
    gpuAssert(cudaMemset(d_dyn, 0, sizeof(uint32_t)));
    initScanTileState(ts, (int)nlb);
}
static void reset(ScanTileState& ts, uint32_t nlb) {
    initScanTileState(ts, (int)nlb);
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
    uint64_t* d_bw_out;
    ScanTileState ts;

    gpuAssert(cudaMalloc(&d_in,           (size_t)size * sizeof(uint8_t)));
    gpuAssert(cudaMalloc(&d_states_out,   (size_t)size * sizeof(state_t)));
    gpuAssert(cudaMalloc(&d_compose_glb,  sizeof(h_compose)));
    gpuAssert(cudaMalloc(&d_to_state_glb, sizeof(h_to_state)));
    gpuAssert(cudaMalloc(&d_bw_out,       (size_t)nlb * BLOCK_SIZE * sizeof(uint64_t)));
    gpuAssert(cudaMalloc(&ts.d_tile_descriptors, ScanTileState::AllocationSize(nlb)));

    gpuAssert(cudaMemcpy(d_in,           input,      (size_t)size * sizeof(uint8_t), cudaMemcpyHostToDevice));
    gpuAssert(cudaMemcpy(d_compose_glb,  h_compose,  sizeof(h_compose),  cudaMemcpyHostToDevice));
    gpuAssert(cudaMemcpy(d_to_state_glb, h_to_state, sizeof(h_to_state), cudaMemcpyHostToDevice));

    float* ms = (float*)malloc(BENCH_RUNS * sizeof(float));
    cudaEvent_t t0, t1;
    gpuAssert(cudaEventCreate(&t0));
    gpuAssert(cudaEventCreate(&t1));

    // ------------------------------------------------------------------
    // BW ceiling
    // ------------------------------------------------------------------
    printf("%-38s ", "BW ceiling BS256/IPT22 (read only):");
    for (uint32_t i = 0; i < WARMUP_RUNS; i++) {
        bw_ceiling<BLOCK_SIZE, ITEMS_PER_THREAD><<<nlb, BLOCK_SIZE>>>(d_in, size, d_bw_out);
        gpuAssert(cudaDeviceSynchronize());
    }
    for (uint32_t i = 0; i < BENCH_RUNS; i++) {
        gpuAssert(cudaEventRecord(t0));
        bw_ceiling<BLOCK_SIZE, ITEMS_PER_THREAD><<<nlb, BLOCK_SIZE>>>(d_in, size, d_bw_out);
        gpuAssert(cudaDeviceSynchronize());
        gpuAssert(cudaEventRecord(t1));
        gpuAssert(cudaEventSynchronize(t1));
        gpuAssert(cudaEventElapsedTime(ms + i, t0, t1));
    }
    print_stats(ms, BENCH_RUNS,
                (size_t)size * sizeof(uint8_t) +
                (size_t)nlb * BLOCK_SIZE * sizeof(uint64_t));

    // ------------------------------------------------------------------
    // P1 add scan
    // ------------------------------------------------------------------
    {
        auto kernel    = p1_add<BLOCK_SIZE, ITEMS_PER_THREAD>;
        size_t shmem   = p1_add_shmem(size);
        size_t p1_bytes = (size_t)size * sizeof(uint8_t) + (size_t)size * sizeof(state_t);

        uint32_t* d_dyn_add;
        gpuAssert(cudaMalloc(&d_dyn_add, sizeof(uint32_t)));

        gpuAssert(cudaFuncSetAttribute(kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));

        printf("%-38s \n  %-36s ", "2Pass P1 BS256/IPT22 (add scan):", "P1:");
        for (uint32_t i = 0; i < WARMUP_RUNS; i++) {
            reset(ts, d_dyn_add, nlb);
            kernel<<<nlb, BLOCK_SIZE, shmem>>>(
                d_to_state_glb, d_in, d_states_out, ts, size, nlb, d_dyn_add);
            gpuAssert(cudaDeviceSynchronize());
        }
        for (uint32_t i = 0; i < BENCH_RUNS; i++) {
            reset(ts, d_dyn_add, nlb);
            gpuAssert(cudaEventRecord(t0));
            kernel<<<nlb, BLOCK_SIZE, shmem>>>(
                d_to_state_glb, d_in, d_states_out, ts, size, nlb, d_dyn_add);
            gpuAssert(cudaDeviceSynchronize());
            gpuAssert(cudaEventRecord(t1));
            gpuAssert(cudaEventSynchronize(t1));
            gpuAssert(cudaEventElapsedTime(ms + i, t0, t1));
        }
        print_stats(ms, BENCH_RUNS, p1_bytes);
        gpuAssert(cudaFree(d_dyn_add));
    }

    // ------------------------------------------------------------------
    // P1 DeviceScan (CUB DeviceScan::InclusiveScan with TransformInputIterator)
    // Bytes are mapped to initial states on-the-fly via ByteToState functor;
    // the compose table is in global memory (no shmem available inside DeviceScan).
    // ------------------------------------------------------------------
    {
        ByteToState byte_to_state{d_to_state_glb};
        auto        d_in_states = thrust::make_transform_iterator(d_in, byte_to_state);
        ComposeOp    compose_op{d_compose_glb};
        size_t       p1_bytes = (size_t)size * sizeof(uint8_t) + (size_t)size * sizeof(state_t);

        void*  d_temp = nullptr;
        size_t temp_bytes = 0;
        cub::DeviceScan::InclusiveScan(d_temp, temp_bytes, d_in_states, d_states_out,
                                       compose_op, (int)size);
        gpuAssert(cudaMalloc(&d_temp, temp_bytes));

        printf("%-38s \n  %-36s ", "2Pass P1 BS256/IPT22 (DeviceScan):", "P1:");
        for (uint32_t i = 0; i < WARMUP_RUNS; i++) {
            cub::DeviceScan::InclusiveScan(d_temp, temp_bytes, d_in_states, d_states_out,
                                           compose_op, (int)size);
            gpuAssert(cudaDeviceSynchronize());
        }
        for (uint32_t i = 0; i < BENCH_RUNS; i++) {
            gpuAssert(cudaEventRecord(t0));
            cub::DeviceScan::InclusiveScan(d_temp, temp_bytes, d_in_states, d_states_out,
                                           compose_op, (int)size);
            gpuAssert(cudaDeviceSynchronize());
            gpuAssert(cudaEventRecord(t1));
            gpuAssert(cudaEventSynchronize(t1));
            gpuAssert(cudaEventElapsedTime(ms + i, t0, t1));
        }
        print_stats(ms, BENCH_RUNS, p1_bytes);
        gpuAssert(cudaFree(d_temp));
    }

    // ------------------------------------------------------------------
    // P1 NregNone (full DFA composition)
    // ------------------------------------------------------------------
    {
        auto kernel    = p1_nregnone<BLOCK_SIZE, ITEMS_PER_THREAD>;
        size_t shmem   = p1_nregnone_shmem(size);
        size_t p1_bytes = (size_t)size * sizeof(uint8_t) + (size_t)size * sizeof(state_t);

        gpuAssert(cudaFuncSetAttribute(kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));

        printf("%-38s \n  %-36s ", "2Pass P1 BS256/IPT22 (NregNone):", "P1:");
        for (uint32_t i = 0; i < WARMUP_RUNS; i++) {
            reset(ts, nlb);
            kernel<<<nlb, BLOCK_SIZE, shmem>>>(
                d_compose_glb, d_to_state_glb, d_in, d_states_out, ts, size, nlb);
            gpuAssert(cudaDeviceSynchronize());
        }
        for (uint32_t i = 0; i < BENCH_RUNS; i++) {
            reset(ts, nlb);
            gpuAssert(cudaEventRecord(t0));
            kernel<<<nlb, BLOCK_SIZE, shmem>>>(
                d_compose_glb, d_to_state_glb, d_in, d_states_out, ts, size, nlb);
            gpuAssert(cudaDeviceSynchronize());
            gpuAssert(cudaEventRecord(t1));
            gpuAssert(cudaEventSynchronize(t1));
            gpuAssert(cudaEventElapsedTime(ms + i, t0, t1));
        }
        print_stats(ms, BENCH_RUNS, p1_bytes);
    }

    // ------------------------------------------------------------------
    // P1 CUB-style: matches DeviceScan DefaultPolicy exactly
    //   128 threads, 15 IPT, BLOCK_LOAD_WARP_TRANSPOSE,
    //   BLOCK_STORE_WARP_TRANSPOSE, BLOCK_SCAN_WARP_SCANS
    // but with compose table in shmem (DeviceScan uses global memory).
    // ------------------------------------------------------------------
    {
        const uint32_t BS_CUB  = 128;
        const uint32_t IPT_CUB = 15;
        uint32_t nlb_cub = (size + BS_CUB * IPT_CUB - 1) / (BS_CUB * IPT_CUB);
        auto     kernel  = p1_cub_style<BS_CUB, IPT_CUB>;
        size_t   p1_bytes = (size_t)size * sizeof(uint8_t) + (size_t)size * sizeof(state_t);

        ScanTileState ts_cub;
        gpuAssert(cudaMalloc(&ts_cub.d_tile_descriptors, ScanTileState::AllocationSize(nlb_cub)));
        uint32_t* d_dyn_cub;
        gpuAssert(cudaMalloc(&d_dyn_cub, sizeof(uint32_t)));

        printf("%-38s \n  %-36s ", "2Pass P1 BS128/IPT15 (cub-style):", "P1:");
        for (uint32_t i = 0; i < WARMUP_RUNS; i++) {
            reset(ts_cub, d_dyn_cub, nlb_cub);
            kernel<<<nlb_cub, BS_CUB>>>(
                d_compose_glb, d_to_state_glb, d_in, d_states_out,
                ts_cub, size, nlb_cub, d_dyn_cub);
            gpuAssert(cudaDeviceSynchronize());
        }
        for (uint32_t i = 0; i < BENCH_RUNS; i++) {
            reset(ts_cub, d_dyn_cub, nlb_cub);
            gpuAssert(cudaEventRecord(t0));
            kernel<<<nlb_cub, BS_CUB>>>(
                d_compose_glb, d_to_state_glb, d_in, d_states_out,
                ts_cub, size, nlb_cub, d_dyn_cub);
            gpuAssert(cudaDeviceSynchronize());
            gpuAssert(cudaEventRecord(t1));
            gpuAssert(cudaEventSynchronize(t1));
            gpuAssert(cudaEventElapsedTime(ms + i, t0, t1));
        }
        print_stats(ms, BENCH_RUNS, p1_bytes);
        gpuAssert(cudaFree(ts_cub.d_tile_descriptors));
        gpuAssert(cudaFree(d_dyn_cub));
    }


    // Cleanup
    free(ms); free(input);
    gpuAssert(cudaFree(d_in));
    gpuAssert(cudaFree(d_states_out));
    gpuAssert(cudaFree(d_compose_glb));
    gpuAssert(cudaFree(d_to_state_glb));
    gpuAssert(cudaFree(d_bw_out));
    gpuAssert(cudaFree(ts.d_tile_descriptors));
    return 0;
}
