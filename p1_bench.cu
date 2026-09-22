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

// Same as ComposeOp but uses __ldg for global read-only cache access.
// Use when d_compose points to global memory rather than shmem.
struct ComposeOpLdg {
    const state_t* __restrict__ d_compose;

    __device__ __forceinline__ state_t
    operator()(state_t a, state_t b) const {
        return __ldg(d_compose + (b & 15u) * NUM_STATES + (a & 15u));
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


// P1 kernel: 256T/22IPT with BLOCK_LOAD_WARP_TRANSPOSE / BLOCK_STORE_WARP_TRANSPOSE.
// Same tile size as p1_nregnone, but uses CUB's transposed load/store pattern
// instead of manual u64 coalesced loads + shmem blocked layout.
// Static blockIdx.x assignment, tile-0 fast path — matches p1_nregnone structure.
template<uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD>
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
    using PrefixOp    = PrefixCallbackOp<ComposeOp>;

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

// Add scan P1: transpose load/store, warp scans, static tile assignment.
// Uses AddOp instead of DFA composition — output values are meaningless;
// timing isolates scan overhead vs composition cost.
template<uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD>
__global__ LB_P1
void p1_add(
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
    using PrefixOp    = PrefixCallbackOp<AddOp>;

    __shared__ union {
        typename BlockLoadT::TempStorage  load;
        typename BlockStoreT::TempStorage store;
        struct {
            typename PrefixOp::TempStorage  prefix;
            typename BlockScanT::TempStorage scan;
        } scan_storage;
    } temp;

    __shared__ __align__(8) state_t shmem_to_state[256];

    for (uint32_t i = threadIdx.x; i < 256 / 4; i += BLOCK_SIZE)
        reinterpret_cast<volatile uint64_t*>(shmem_to_state)[i] =
            reinterpret_cast<uint64_t*>(d_to_state_glb)[i];
    __syncthreads();

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
        BlockLoadT(temp.load).Load(d_in_states, st, valid, state_t(0));
    __syncthreads();

    AddOp add_op;

    if (tile_idx == 0) {
        state_t block_aggregate;
        BlockScanT(temp.scan_storage.scan).InclusiveScan(st, st, add_op, block_aggregate);
        if (threadIdx.x == 0)
            tile_state.SetInclusive(0, block_aggregate);
    } else {
        PrefixOp prefix_op(tile_state, temp.scan_storage.prefix, add_op, (int)tile_idx, state_t(0));
        BlockScanT(temp.scan_storage.scan).InclusiveScan(st, st, add_op, prefix_op);
    }
    __syncthreads();

    if (glb_offs + BLOCK_SIZE * ITEMS_PER_THREAD <= size)
        BlockStoreT(temp.store).Store(d_states_out + glb_offs, st);
    else
        BlockStoreT(temp.store).Store(d_states_out + glb_offs, st, valid);
}

// ---------------------------------------------------------------------------
// Fused load+transpose and transpose+store helpers
//
// CUB BLOCK_LOAD_WARP_TRANSPOSE does:
//   global[striped] -> registers[striped]   (LoadDirectWarpStriped)
//   registers[striped] -> shmem[striped]    \
//   __syncwarp                               > WarpStripedToBlocked
//   shmem[blocked] -> registers[blocked]   /
//
// The fused version skips the intermediate registers:
//   global[striped] -> shmem[striped]
//   __syncwarp
//   shmem[blocked] -> registers[blocked]
//
// Same for store (reverse). Saves IPT register writes + reads per thread.
//
// For IPT that is a power of 2 and > 4, CUB inserts padding
// (item_offset += item_offset >> LOG_SMEM_BANKS) to avoid bank conflicts.
// We replicate that here. For IPT=22 (not power of 2), no padding needed.
// ---------------------------------------------------------------------------

template<uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD, bool INSERT_PADDING>
__device__ __forceinline__ void
fusedLoadTranspose(
    const uint8_t* __restrict__ d_in,
    const state_t* __restrict__ d_to_state,
    volatile state_t* shmem,   // BLOCK_SIZE * ITEMS_PER_THREAD [+ padding] entries
    state_t (&st)[ITEMS_PER_THREAD],
    uint32_t glb_offs, uint32_t size)
{
    const uint32_t WARP_ID     = threadIdx.x / WARP;
    const uint32_t LANE        = threadIdx.x % WARP;
    const uint32_t WARP_OFFSET = WARP_ID * WARP * ITEMS_PER_THREAD;

    // Global -> shmem in warp-striped order (coalesced global read).
    #pragma unroll
    for (uint32_t s = 0; s < ITEMS_PER_THREAD; s++) {
        uint32_t abs        = glb_offs + WARP_OFFSET + LANE + s * WARP;
        uint8_t  byte       = (abs < size) ? __ldg(d_in + abs) : uint8_t(0);
        state_t  val        = (abs < size) ? __ldg(d_to_state + byte) : state_t(IDENTITY);
        uint32_t shmem_idx  = WARP_OFFSET + s * WARP + LANE;
        if (INSERT_PADDING) shmem_idx += shmem_idx >> 5;  // LOG_SMEM_BANKS=5 on A100
        shmem[shmem_idx] = val;
    }

    __syncwarp(0xffffffff);

    // Shmem -> registers in blocked order.
    #pragma unroll
    for (uint32_t s = 0; s < ITEMS_PER_THREAD; s++) {
        uint32_t shmem_idx = WARP_OFFSET + LANE * ITEMS_PER_THREAD + s;
        if (INSERT_PADDING) shmem_idx += shmem_idx >> 5;
        st[s] = shmem[shmem_idx];
    }
}

template<uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD, bool INSERT_PADDING>
__device__ __forceinline__ void
fusedTransposeStore(
    state_t* __restrict__ d_out,
    volatile state_t* shmem,
    state_t (&st)[ITEMS_PER_THREAD],
    uint32_t glb_offs, uint32_t size)
{
    const uint32_t WARP_ID     = threadIdx.x / WARP;
    const uint32_t LANE        = threadIdx.x % WARP;
    const uint32_t WARP_OFFSET = WARP_ID * WARP * ITEMS_PER_THREAD;

    // Registers -> shmem in blocked order.
    #pragma unroll
    for (uint32_t s = 0; s < ITEMS_PER_THREAD; s++) {
        uint32_t shmem_idx = WARP_OFFSET + LANE * ITEMS_PER_THREAD + s;
        if (INSERT_PADDING) shmem_idx += shmem_idx >> 5;
        shmem[shmem_idx] = st[s];
    }

    __syncwarp(0xffffffff);

    // Shmem -> global in warp-striped order (coalesced global write).
    #pragma unroll
    for (uint32_t s = 0; s < ITEMS_PER_THREAD; s++) {
        uint32_t abs       = glb_offs + WARP_OFFSET + LANE + s * WARP;
        uint32_t shmem_idx = WARP_OFFSET + s * WARP + LANE;
        if (INSERT_PADDING) shmem_idx += shmem_idx >> 5;
        if (abs < size) d_out[abs] = shmem[shmem_idx];
    }
}

// Full-tile fast path for store (no bounds check).
template<uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD, bool INSERT_PADDING>
__device__ __forceinline__ void
fusedTransposeStoreFull(
    state_t* __restrict__ d_out,
    volatile state_t* shmem,
    state_t (&st)[ITEMS_PER_THREAD],
    uint32_t glb_offs)
{
    const uint32_t WARP_ID     = threadIdx.x / WARP;
    const uint32_t LANE        = threadIdx.x % WARP;
    const uint32_t WARP_OFFSET = WARP_ID * WARP * ITEMS_PER_THREAD;

    #pragma unroll
    for (uint32_t s = 0; s < ITEMS_PER_THREAD; s++) {
        uint32_t shmem_idx = WARP_OFFSET + LANE * ITEMS_PER_THREAD + s;
        if (INSERT_PADDING) shmem_idx += shmem_idx >> 5;
        shmem[shmem_idx] = st[s];
    }

    __syncwarp(0xffffffff);

    #pragma unroll
    for (uint32_t s = 0; s < ITEMS_PER_THREAD; s++) {
        uint32_t abs       = glb_offs + WARP_OFFSET + LANE + s * WARP;
        uint32_t shmem_idx = WARP_OFFSET + s * WARP + LANE;
        if (INSERT_PADDING) shmem_idx += shmem_idx >> 5;
        d_out[abs] = shmem[shmem_idx];
    }
}

// ---------------------------------------------------------------------------
// p1_fused22: IPT=22, fused global-read+shmem-write (no intermediate regs),
// fused shmem-read+global-write on store. No CUB BlockLoad/BlockStore.
// Uses __ldg for DFA tables. No padding (IPT=22 not power-of-2).
// ---------------------------------------------------------------------------
template<uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD>
__global__ LB_P1
void p1_fused22(
    state_t* __restrict__ d_compose_glb,
    state_t* __restrict__ d_to_state_glb,
    const uint8_t* __restrict__ d_in,
    state_t* __restrict__ d_states_out,
    ScanTileState tile_state,
    uint32_t size,
    uint32_t num_tiles)
{
    using BlockScanT = cub::BlockScan<state_t, BLOCK_SIZE, cub::BLOCK_SCAN_WARP_SCANS>;
    using PrefixOp   = PrefixCallbackOp<ComposeOpLdg>;

    // Shmem: transpose buffer (no padding for IPT=22) unioned with scan temp.
    __shared__ union {
        volatile state_t transpose[BLOCK_SIZE * ITEMS_PER_THREAD];
        struct {
            typename PrefixOp::TempStorage   prefix;
            typename BlockScanT::TempStorage scan;
        } scan_storage;
    } temp;

    uint32_t tile_idx = blockIdx.x;
    uint32_t glb_offs = tile_idx * BLOCK_SIZE * ITEMS_PER_THREAD;

    state_t st[ITEMS_PER_THREAD];
    fusedLoadTranspose<BLOCK_SIZE, ITEMS_PER_THREAD, false>(
        d_in, d_to_state_glb, temp.transpose, st, glb_offs, size);

    __syncthreads();  // transpose -> scan_storage union transition

    ComposeOpLdg compose_op{d_compose_glb};

    if (tile_idx == 0) {
        state_t block_aggregate;
        BlockScanT(temp.scan_storage.scan).InclusiveScan(st, st, compose_op, block_aggregate);
        if (threadIdx.x == 0)
            tile_state.SetInclusive(0, block_aggregate);
    } else {
        PrefixOp prefix_op(tile_state, temp.scan_storage.prefix, compose_op, (int)tile_idx, IDENTITY);
        BlockScanT(temp.scan_storage.scan).InclusiveScan(st, st, compose_op, prefix_op);
    }

    __syncthreads();  // scan_storage -> transpose union transition

    if (glb_offs + BLOCK_SIZE * ITEMS_PER_THREAD <= size)
        fusedTransposeStoreFull<BLOCK_SIZE, ITEMS_PER_THREAD, false>(
            d_states_out, temp.transpose, st, glb_offs);
    else
        fusedTransposeStore<BLOCK_SIZE, ITEMS_PER_THREAD, false>(
            d_states_out, temp.transpose, st, glb_offs, size);
}

// ---------------------------------------------------------------------------
// p1_fused32: IPT=32, same fused approach. IPT=32 is power-of-2 so padding
// is applied (shmem_idx += shmem_idx >> 5) to avoid bank conflicts.
// Shmem = 32*32*2 + padding = 2048 + 64 = 2112 bytes per block.
// ---------------------------------------------------------------------------
template<uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD>
__global__ LB_P1
void p1_fused32(
    state_t* __restrict__ d_compose_glb,
    state_t* __restrict__ d_to_state_glb,
    const uint8_t* __restrict__ d_in,
    state_t* __restrict__ d_states_out,
    ScanTileState tile_state,
    uint32_t size,
    uint32_t num_tiles)
{
    static_assert(ITEMS_PER_THREAD == WARP, "p1_fused32 requires IPT == WARP (32)");

    using BlockScanT = cub::BlockScan<state_t, BLOCK_SIZE, cub::BLOCK_SCAN_WARP_SCANS>;
    using PrefixOp   = PrefixCallbackOp<ComposeOpLdg>;

    // Padding: CUB adds (item_offset >> LOG_SMEM_BANKS) extra slots.
    // For IPT=32, WARP=32: TIME_SLICED_ITEMS = WARP*IPT = 1024 per warp.
    // PADDING_ITEMS = 1024 >> 5 = 32 per warp. 8 warps: 8*(1024+32) = 8448 bytes.
    static constexpr uint32_t PADDED_WARP_ITEMS = WARP * ITEMS_PER_THREAD
        + (WARP * ITEMS_PER_THREAD >> 5);

    __shared__ union {
        volatile state_t transpose[BLOCK_SIZE / WARP * PADDED_WARP_ITEMS];
        struct {
            typename PrefixOp::TempStorage   prefix;
            typename BlockScanT::TempStorage scan;
        } scan_storage;
    } temp;

    uint32_t tile_idx = blockIdx.x;
    uint32_t glb_offs = tile_idx * BLOCK_SIZE * ITEMS_PER_THREAD;

    state_t st[ITEMS_PER_THREAD];
    fusedLoadTranspose<BLOCK_SIZE, ITEMS_PER_THREAD, true>(
        d_in, d_to_state_glb, temp.transpose, st, glb_offs, size);

    __syncthreads();

    ComposeOpLdg compose_op{d_compose_glb};

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
        fusedTransposeStoreFull<BLOCK_SIZE, ITEMS_PER_THREAD, true>(
            d_states_out, temp.transpose, st, glb_offs);
    else
        fusedTransposeStore<BLOCK_SIZE, ITEMS_PER_THREAD, true>(
            d_states_out, temp.transpose, st, glb_offs, size);
}

// ---------------------------------------------------------------------------
// Warp-level shuffle transpose helpers
// ---------------------------------------------------------------------------

// Coalesced (striped) load of ITEMS_PER_THREAD state_t items per thread,
// then redistribute into blocked layout using __shfl_sync.
//
// Striped layout: thread t owns items t, t+BS, t+2*BS, ...
// Blocked layout: thread t owns items t*IPT, t*IPT+1, ..., t*IPT+IPT-1
//
// Item at absolute index i is in striped thread (i % BS) slot (i / BS).
// Thread t blocked slot k needs absolute index t*IPT+k, which lives in
// striped thread (t*IPT+k) % BS at slot (t*IPT+k) / BS.
template<uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD>
__device__ __forceinline__ void
warpShflLoad(const state_t* __restrict__ base, state_t (&items)[ITEMS_PER_THREAD],
             uint32_t glb_offs, uint32_t size)
{
    const uint32_t TILE = BLOCK_SIZE * ITEMS_PER_THREAD;
    const uint32_t WARP_ID = threadIdx.x / WARP;
    const uint32_t LANE    = threadIdx.x % WARP;

    // Each thread loads ITEMS_PER_THREAD items in striped layout into regs.
    state_t striped[ITEMS_PER_THREAD];
    #pragma unroll
    for (uint32_t s = 0; s < ITEMS_PER_THREAD; s++) {
        uint32_t abs = glb_offs + WARP_ID * WARP * ITEMS_PER_THREAD + LANE + s * WARP;
        striped[s] = (abs < size) ? __ldg(base + abs) : state_t(IDENTITY);
    }

    // Shuffle striped → blocked within each warp.
    // Thread t blocked slot k needs absolute warp-local index t*IPT+k,
    // which lives in striped lane (t*IPT+k) % WARP, slot (t*IPT+k) / WARP.
    #pragma unroll
    for (uint32_t k = 0; k < ITEMS_PER_THREAD; k++) {
        uint32_t abs_local = LANE * ITEMS_PER_THREAD + k;
        uint32_t src_lane  = abs_local % WARP;
        uint32_t src_slot  = abs_local / WARP;
        items[k] = __shfl_sync(0xffffffff, striped[src_slot], src_lane);
    }
    (void)TILE;
}

// Blocked → striped shuffle then coalesced store.
template<uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD>
__device__ __forceinline__ void
warpShflStore(state_t* __restrict__ base, state_t (&items)[ITEMS_PER_THREAD],
              uint32_t glb_offs, uint32_t size)
{
    const uint32_t WARP_ID = threadIdx.x / WARP;
    const uint32_t LANE    = threadIdx.x % WARP;

    // Shuffle blocked → striped within each warp.
    // Striped slot s of lane t holds absolute warp-local index t + s*WARP,
    // which lives in blocked lane (t+s*WARP)/IPT slot (t+s*WARP)%IPT.
    state_t striped[ITEMS_PER_THREAD];
    #pragma unroll
    for (uint32_t s = 0; s < ITEMS_PER_THREAD; s++) {
        uint32_t abs_local = LANE + s * WARP;
        uint32_t src_lane  = abs_local / ITEMS_PER_THREAD;
        uint32_t src_slot  = abs_local % ITEMS_PER_THREAD;
        striped[s] = __shfl_sync(0xffffffff, items[src_slot], src_lane);
    }

    #pragma unroll
    for (uint32_t s = 0; s < ITEMS_PER_THREAD; s++) {
        uint32_t abs = glb_offs + WARP_ID * WARP * ITEMS_PER_THREAD + LANE + s * WARP;
        if (abs < size) base[abs] = striped[s];
    }
}

// Variant with full-tile fast path (avoids branch per element when in bounds).
template<uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD>
__device__ __forceinline__ void
warpShflStoreFull(state_t* __restrict__ base, state_t (&items)[ITEMS_PER_THREAD],
                  uint32_t glb_offs)
{
    const uint32_t WARP_ID = threadIdx.x / WARP;
    const uint32_t LANE    = threadIdx.x % WARP;

    state_t striped[ITEMS_PER_THREAD];
    #pragma unroll
    for (uint32_t s = 0; s < ITEMS_PER_THREAD; s++) {
        uint32_t abs_local = LANE + s * WARP;
        uint32_t src_lane  = abs_local / ITEMS_PER_THREAD;
        uint32_t src_slot  = abs_local % ITEMS_PER_THREAD;
        striped[s] = __shfl_sync(0xffffffff, items[src_slot], src_lane);
    }

    #pragma unroll
    for (uint32_t s = 0; s < ITEMS_PER_THREAD; s++) {
        uint32_t abs = glb_offs + WARP_ID * WARP * ITEMS_PER_THREAD + LANE + s * WARP;
        base[abs] = striped[s];
    }
}

// ---------------------------------------------------------------------------
// p1_shfl22: IPT=22, register-shuffle transpose, no shmem for load/store.
// Same tile size as p1_transpose. Uses __ldg for tables.
// ---------------------------------------------------------------------------
template<uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD>
__global__ LB_P1
void p1_shfl22(
    state_t* __restrict__ d_compose_glb,
    state_t* __restrict__ d_to_state_glb,
    const uint8_t* __restrict__ d_in,
    state_t* __restrict__ d_states_out,
    ScanTileState tile_state,
    uint32_t size,
    uint32_t num_tiles)
{
    using BlockScanT = cub::BlockScan<state_t, BLOCK_SIZE, cub::BLOCK_SCAN_WARP_SCANS>;
    using PrefixOp   = PrefixCallbackOp<ComposeOpLdg>;

    __shared__ struct {
        typename PrefixOp::TempStorage  prefix;
        typename BlockScanT::TempStorage scan;
    } temp;

    uint32_t tile_idx = blockIdx.x;
    uint32_t glb_offs = tile_idx * BLOCK_SIZE * ITEMS_PER_THREAD;

    const uint32_t WARP_ID = threadIdx.x / WARP;
    const uint32_t LANE    = threadIdx.x % WARP;

    // Striped byte load + to_state lookup via __ldg.
    state_t striped[ITEMS_PER_THREAD];
    #pragma unroll
    for (uint32_t s = 0; s < ITEMS_PER_THREAD; s++) {
        uint32_t abs = glb_offs + WARP_ID * WARP * ITEMS_PER_THREAD + LANE + s * WARP;
        uint8_t byte = (abs < size) ? __ldg(d_in + abs) : uint8_t(0);
        striped[s] = (abs < size) ? __ldg(d_to_state_glb + byte) : state_t(IDENTITY);
    }

    // Shuffle striped → blocked.
    state_t st[ITEMS_PER_THREAD];
    #pragma unroll
    for (uint32_t k = 0; k < ITEMS_PER_THREAD; k++) {
        uint32_t abs_local = LANE * ITEMS_PER_THREAD + k;
        uint32_t src_lane  = abs_local % WARP;
        uint32_t src_slot  = abs_local / WARP;
        st[k] = __shfl_sync(0xffffffff, striped[src_slot], src_lane);
    }

    ComposeOpLdg compose_op{d_compose_glb};

    if (tile_idx == 0) {
        state_t block_aggregate;
        BlockScanT(temp.scan).InclusiveScan(st, st, compose_op, block_aggregate);
        if (threadIdx.x == 0)
            tile_state.SetInclusive(0, block_aggregate);
    } else {
        PrefixOp prefix_op(tile_state, temp.prefix, compose_op, (int)tile_idx, IDENTITY);
        BlockScanT(temp.scan).InclusiveScan(st, st, compose_op, prefix_op);
    }

    // Shuffle blocked → striped and store.
    state_t out_striped[ITEMS_PER_THREAD];
    #pragma unroll
    for (uint32_t s = 0; s < ITEMS_PER_THREAD; s++) {
        uint32_t abs_local = LANE + s * WARP;
        uint32_t src_lane  = abs_local / ITEMS_PER_THREAD;
        uint32_t src_slot  = abs_local % ITEMS_PER_THREAD;
        out_striped[s] = __shfl_sync(0xffffffff, st[src_slot], src_lane);
    }
    if (glb_offs + BLOCK_SIZE * ITEMS_PER_THREAD <= size) {
        #pragma unroll
        for (uint32_t s = 0; s < ITEMS_PER_THREAD; s++) {
            uint32_t abs = glb_offs + WARP_ID * WARP * ITEMS_PER_THREAD + LANE + s * WARP;
            d_states_out[abs] = out_striped[s];
        }
    } else {
        #pragma unroll
        for (uint32_t s = 0; s < ITEMS_PER_THREAD; s++) {
            uint32_t abs = glb_offs + WARP_ID * WARP * ITEMS_PER_THREAD + LANE + s * WARP;
            if (abs < size) d_states_out[abs] = out_striped[s];
        }
    }
}

// ---------------------------------------------------------------------------
// p1_shfl32: IPT=32, striped == blocked (no shuffle needed), no shmem for
// load/store. Larger tile (32KB/block vs 22KB), fewer tiles, longer lookback.
// ---------------------------------------------------------------------------
template<uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD>
__global__ LB_P1
void p1_shfl32(
    state_t* __restrict__ d_compose_glb,
    state_t* __restrict__ d_to_state_glb,
    const uint8_t* __restrict__ d_in,
    state_t* __restrict__ d_states_out,
    ScanTileState tile_state,
    uint32_t size,
    uint32_t num_tiles)
{
    static_assert(ITEMS_PER_THREAD == WARP,
        "p1_shfl32 requires ITEMS_PER_THREAD == WARP (32)");

    using BlockScanT = cub::BlockScan<state_t, BLOCK_SIZE, cub::BLOCK_SCAN_WARP_SCANS>;
    using PrefixOp   = PrefixCallbackOp<ComposeOpLdg>;

    __shared__ struct {
        typename PrefixOp::TempStorage  prefix;
        typename BlockScanT::TempStorage scan;
    } temp;

    uint32_t tile_idx = blockIdx.x;
    uint32_t glb_offs = tile_idx * BLOCK_SIZE * ITEMS_PER_THREAD;
    uint32_t valid    = (uint32_t)min((uint64_t)BLOCK_SIZE * ITEMS_PER_THREAD,
                                      (uint64_t)size - glb_offs);

    // With IPT=WARP=32, striped layout IS blocked layout:
    // thread t owns items t, t+BS, t+2*BS, ... = t*32, t*32+1, ... (for BS=256, WARP=32).
    // Wait — striped: thread t slot s = abs index t + s*BS.
    // Blocked: thread t slot s = abs index t*IPT + s.
    // These are equal only if BS==IPT. BS=256, IPT=32: not equal.
    // But within a warp (32 threads): warp-striped thread t slot s = warp_base + t + s*WARP.
    // Warp-blocked thread t slot s = warp_base + t*WARP + s. These ARE equal when IPT==WARP.
    // BlockScan with WARP_SCANS operates per-warp in warp-blocked layout.
    // So load striped per-warp = load blocked per-warp. No shuffle needed.

    const uint32_t WARP_ID = threadIdx.x / WARP;
    const uint32_t LANE    = threadIdx.x % WARP;

    state_t st[ITEMS_PER_THREAD];
    if (glb_offs + BLOCK_SIZE * ITEMS_PER_THREAD <= size) {
        #pragma unroll
        for (uint32_t s = 0; s < ITEMS_PER_THREAD; s++) {
            uint32_t abs = glb_offs + WARP_ID * WARP * ITEMS_PER_THREAD + LANE + s * WARP;
            uint8_t byte = __ldg(d_in + abs);
            st[s] = __ldg(d_to_state_glb + byte);
        }
    } else {
        #pragma unroll
        for (uint32_t s = 0; s < ITEMS_PER_THREAD; s++) {
            uint32_t abs = glb_offs + WARP_ID * WARP * ITEMS_PER_THREAD + LANE + s * WARP;
            if (abs < size) {
                uint8_t byte = __ldg(d_in + abs);
                st[s] = __ldg(d_to_state_glb + byte);
            } else {
                st[s] = IDENTITY;
            }
        }
    }

    ComposeOpLdg compose_op{d_compose_glb};

    if (tile_idx == 0) {
        state_t block_aggregate;
        BlockScanT(temp.scan).InclusiveScan(st, st, compose_op, block_aggregate);
        if (threadIdx.x == 0)
            tile_state.SetInclusive(0, block_aggregate);
    } else {
        PrefixOp prefix_op(tile_state, temp.prefix, compose_op, (int)tile_idx, IDENTITY);
        BlockScanT(temp.scan).InclusiveScan(st, st, compose_op, prefix_op);
    }

    // Store: warp-striped = warp-blocked for IPT=WARP, so write directly.
    if (glb_offs + BLOCK_SIZE * ITEMS_PER_THREAD <= size) {
        #pragma unroll
        for (uint32_t s = 0; s < ITEMS_PER_THREAD; s++) {
            uint32_t abs = glb_offs + WARP_ID * WARP * ITEMS_PER_THREAD + LANE + s * WARP;
            d_states_out[abs] = st[s];
        }
    } else {
        #pragma unroll
        for (uint32_t s = 0; s < ITEMS_PER_THREAD; s++) {
            uint32_t abs = glb_offs + WARP_ID * WARP * ITEMS_PER_THREAD + LANE + s * WARP;
            if (abs < size) d_states_out[abs] = st[s];
        }
    }
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
        size_t p1_bytes = (size_t)size * sizeof(uint8_t) + (size_t)size * sizeof(state_t);

        printf("%-38s \n  %-36s ", "2Pass P1 BS256/IPT22 (add scan):", "P1:");
        for (uint32_t i = 0; i < WARMUP_RUNS; i++) {
            reset(ts, nlb);
            kernel<<<nlb, BLOCK_SIZE>>>(
                d_to_state_glb, d_in, d_states_out, ts, size, nlb);
            gpuAssert(cudaDeviceSynchronize());
        }
        for (uint32_t i = 0; i < BENCH_RUNS; i++) {
            reset(ts, nlb);
            gpuAssert(cudaEventRecord(t0));
            kernel<<<nlb, BLOCK_SIZE>>>(
                d_to_state_glb, d_in, d_states_out, ts, size, nlb);
            gpuAssert(cudaDeviceSynchronize());
            gpuAssert(cudaEventRecord(t1));
            gpuAssert(cudaEventSynchronize(t1));
            gpuAssert(cudaEventElapsedTime(ms + i, t0, t1));
        }
        print_stats(ms, BENCH_RUNS, p1_bytes);
    }

    // ------------------------------------------------------------------
    // P1 transpose: 256T/22IPT with BLOCK_LOAD_WARP_TRANSPOSE /
    // BLOCK_STORE_WARP_TRANSPOSE — same tile size as NregNone, CUB load pattern.
    // ------------------------------------------------------------------
    {
        auto kernel    = p1_transpose<BLOCK_SIZE, ITEMS_PER_THREAD>;
        size_t p1_bytes = (size_t)size * sizeof(uint8_t) + (size_t)size * sizeof(state_t);

        printf("%-38s \n  %-36s ", "2Pass P1 BS256/IPT22 (transpose):", "P1:");
        for (uint32_t i = 0; i < WARMUP_RUNS; i++) {
            reset(ts, nlb);
            kernel<<<nlb, BLOCK_SIZE>>>(
                d_compose_glb, d_to_state_glb, d_in, d_states_out, ts, size, nlb);
            gpuAssert(cudaDeviceSynchronize());
        }
        for (uint32_t i = 0; i < BENCH_RUNS; i++) {
            reset(ts, nlb);
            gpuAssert(cudaEventRecord(t0));
            kernel<<<nlb, BLOCK_SIZE>>>(
                d_compose_glb, d_to_state_glb, d_in, d_states_out, ts, size, nlb);
            gpuAssert(cudaDeviceSynchronize());
            gpuAssert(cudaEventRecord(t1));
            gpuAssert(cudaEventSynchronize(t1));
            gpuAssert(cudaEventElapsedTime(ms + i, t0, t1));
        }
        print_stats(ms, BENCH_RUNS, p1_bytes);
    }

    // ------------------------------------------------------------------
    // P1 shfl22: register-shuffle transpose, IPT=22, __ldg for tables.
    // ------------------------------------------------------------------
    {
        auto kernel    = p1_shfl22<BLOCK_SIZE, ITEMS_PER_THREAD>;
        size_t p1_bytes = (size_t)size * sizeof(uint8_t) + (size_t)size * sizeof(state_t);

        printf("%-38s \n  %-36s ", "2Pass P1 BS256/IPT22 (shfl22):", "P1:");
        for (uint32_t i = 0; i < WARMUP_RUNS; i++) {
            reset(ts, nlb);
            kernel<<<nlb, BLOCK_SIZE>>>(
                d_compose_glb, d_to_state_glb, d_in, d_states_out, ts, size, nlb);
            gpuAssert(cudaDeviceSynchronize());
        }
        for (uint32_t i = 0; i < BENCH_RUNS; i++) {
            reset(ts, nlb);
            gpuAssert(cudaEventRecord(t0));
            kernel<<<nlb, BLOCK_SIZE>>>(
                d_compose_glb, d_to_state_glb, d_in, d_states_out, ts, size, nlb);
            gpuAssert(cudaDeviceSynchronize());
            gpuAssert(cudaEventRecord(t1));
            gpuAssert(cudaEventSynchronize(t1));
            gpuAssert(cudaEventElapsedTime(ms + i, t0, t1));
        }
        print_stats(ms, BENCH_RUNS, p1_bytes);
    }

    // ------------------------------------------------------------------
    // P1 shfl32: IPT=32, no shuffle (warp-striped=warp-blocked), __ldg tables.
    // ------------------------------------------------------------------
    {
        static const uint32_t IPT32 = 32;
        uint32_t nlb32 = (size + BLOCK_SIZE * IPT32 - 1) / (BLOCK_SIZE * IPT32);
        ScanTileState ts32;
        gpuAssert(cudaMalloc(&ts32.d_tile_descriptors, ScanTileState::AllocationSize(nlb32)));

        auto kernel    = p1_shfl32<BLOCK_SIZE, IPT32>;
        size_t p1_bytes = (size_t)size * sizeof(uint8_t) + (size_t)size * sizeof(state_t);

        printf("%-38s \n  %-36s ", "2Pass P1 BS256/IPT32 (shfl32):", "P1:");
        for (uint32_t i = 0; i < WARMUP_RUNS; i++) {
            initScanTileState(ts32, (int)nlb32);
            kernel<<<nlb32, BLOCK_SIZE>>>(
                d_compose_glb, d_to_state_glb, d_in, d_states_out, ts32, size, nlb32);
            gpuAssert(cudaDeviceSynchronize());
        }
        for (uint32_t i = 0; i < BENCH_RUNS; i++) {
            initScanTileState(ts32, (int)nlb32);
            gpuAssert(cudaEventRecord(t0));
            kernel<<<nlb32, BLOCK_SIZE>>>(
                d_compose_glb, d_to_state_glb, d_in, d_states_out, ts32, size, nlb32);
            gpuAssert(cudaDeviceSynchronize());
            gpuAssert(cudaEventRecord(t1));
            gpuAssert(cudaEventSynchronize(t1));
            gpuAssert(cudaEventElapsedTime(ms + i, t0, t1));
        }
        print_stats(ms, BENCH_RUNS, p1_bytes);
        gpuAssert(cudaFree(ts32.d_tile_descriptors));
    }

    // ------------------------------------------------------------------
    // P1 fused22: fused global-read+shmem-write, IPT=22, __ldg tables.
    // ------------------------------------------------------------------
    {
        auto kernel    = p1_fused22<BLOCK_SIZE, ITEMS_PER_THREAD>;
        size_t p1_bytes = (size_t)size * sizeof(uint8_t) + (size_t)size * sizeof(state_t);

        printf("%-38s \n  %-36s ", "2Pass P1 BS256/IPT22 (fused22):", "P1:");
        for (uint32_t i = 0; i < WARMUP_RUNS; i++) {
            reset(ts, nlb);
            kernel<<<nlb, BLOCK_SIZE>>>(
                d_compose_glb, d_to_state_glb, d_in, d_states_out, ts, size, nlb);
            gpuAssert(cudaDeviceSynchronize());
        }
        for (uint32_t i = 0; i < BENCH_RUNS; i++) {
            reset(ts, nlb);
            gpuAssert(cudaEventRecord(t0));
            kernel<<<nlb, BLOCK_SIZE>>>(
                d_compose_glb, d_to_state_glb, d_in, d_states_out, ts, size, nlb);
            gpuAssert(cudaDeviceSynchronize());
            gpuAssert(cudaEventRecord(t1));
            gpuAssert(cudaEventSynchronize(t1));
            gpuAssert(cudaEventElapsedTime(ms + i, t0, t1));
        }
        print_stats(ms, BENCH_RUNS, p1_bytes);
    }

    // ------------------------------------------------------------------
    // P1 fused32: fused global-read+shmem-write, IPT=32, padding, __ldg tables.
    // ------------------------------------------------------------------
    {
        static const uint32_t IPT32 = 32;
        uint32_t nlb32 = (size + BLOCK_SIZE * IPT32 - 1) / (BLOCK_SIZE * IPT32);
        ScanTileState ts32;
        gpuAssert(cudaMalloc(&ts32.d_tile_descriptors, ScanTileState::AllocationSize(nlb32)));

        auto kernel    = p1_fused32<BLOCK_SIZE, IPT32>;
        size_t p1_bytes = (size_t)size * sizeof(uint8_t) + (size_t)size * sizeof(state_t);

        printf("%-38s \n  %-36s ", "2Pass P1 BS256/IPT32 (fused32):", "P1:");
        for (uint32_t i = 0; i < WARMUP_RUNS; i++) {
            initScanTileState(ts32, (int)nlb32);
            kernel<<<nlb32, BLOCK_SIZE>>>(
                d_compose_glb, d_to_state_glb, d_in, d_states_out, ts32, size, nlb32);
            gpuAssert(cudaDeviceSynchronize());
        }
        for (uint32_t i = 0; i < BENCH_RUNS; i++) {
            initScanTileState(ts32, (int)nlb32);
            gpuAssert(cudaEventRecord(t0));
            kernel<<<nlb32, BLOCK_SIZE>>>(
                d_compose_glb, d_to_state_glb, d_in, d_states_out, ts32, size, nlb32);
            gpuAssert(cudaDeviceSynchronize());
            gpuAssert(cudaEventRecord(t1));
            gpuAssert(cudaEventSynchronize(t1));
            gpuAssert(cudaEventElapsedTime(ms + i, t0, t1));
        }
        print_stats(ms, BENCH_RUNS, p1_bytes);
        gpuAssert(cudaFree(ts32.d_tile_descriptors));
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
