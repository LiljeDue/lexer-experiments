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
    {
        auto kernel = p1_transpose<BLOCK_SIZE, ITEMS_PER_THREAD>;
        auto prep   = [&] { reset(ts, nlb); };
        auto launch = [&] { kernel<<<nlb, BLOCK_SIZE>>>(d_compose_glb, d_to_state_glb, d_in, d_states_out, ts, size, nlb); };
        poison_out(); prep(); launch(); fetch_states();
        if (!check_states(input, size, h_states, true, 0, "L3")) exit(1);
        bench("L3 + lookback (p1_transpose):", u16_bytes, prep, launch);
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
