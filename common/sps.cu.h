#include <cuda_runtime.h>
#include <cstdint>
#include <cub/cub.cuh>

const uint8_t LG_WARP = 5;
const uint8_t WARP = 1 << LG_WARP;

// Number of padding entries prepended to the tile state array.
// Threads in block 0 look back at indices tile_idx - threadIdx.x - 1,
// which for threadIdx.x up to WARP-1 gives indices as low as -WARP.
// Padding these with SCAN_TILE_OOB means WaitForValid returns immediately
// (status != INVALID), and TailSegmentedReduce treats them as partial
// aggregates with value=identity (stop flag only set on INCLUSIVE).
const uint32_t TILE_STATUS_PADDING = WARP;

enum ScanTileStatus : uint32_t {
    SCAN_TILE_OOB       = 0,  // out-of-bounds padding
    SCAN_TILE_INVALID   = 1,  // not yet published
    SCAN_TILE_PARTIAL   = 2,  // aggregate published, prefix not yet known
    SCAN_TILE_INCLUSIVE = 3,  // inclusive prefix published
};

// ---------------------------------------------------------------------------
// Single-word tile state: packs status + value into one TxnWord so that
// publish and read are each a single atomic-width transaction.
//
// Layout: TxnWord = (value << (8*sizeof(T))) | status
//
// T=uint16_t: TxnWord=uint32_t,           status in bits [15:0],  value in bits [31:16]
// T=uint32_t: TxnWord=unsigned long long, status in bits [31:0],  value in bits [63:32]
//
// Pack/unpack done with shifts — no reinterpret_cast — to avoid alignment
// issues with local-memory stack variables.
// ---------------------------------------------------------------------------
template<typename T>
struct TxnWordTraits;

template<> struct TxnWordTraits<uint16_t> {
    using TxnWord    = uint32_t;
    using StatusWord = uint16_t;
    __device__ __forceinline__
    static TxnWord pack(StatusWord status, uint16_t value) {
        return (uint32_t(value) << 16) | uint32_t(status);
    }
    __device__ __forceinline__
    static StatusWord unpack_status(TxnWord w) { return StatusWord(w & 0xffffu); }
    __device__ __forceinline__
    static uint16_t   unpack_value(TxnWord w)  { return uint16_t(w >> 16); }
};
template<> struct TxnWordTraits<uint32_t> {
    using TxnWord    = unsigned long long;
    using StatusWord = uint32_t;
    __device__ __forceinline__
    static TxnWord pack(StatusWord status, uint32_t value) {
        return (TxnWord(value) << 32) | TxnWord(status);
    }
    __device__ __forceinline__
    static StatusWord unpack_status(TxnWord w) { return StatusWord(w & 0xffffffffull); }
    __device__ __forceinline__
    static uint32_t   unpack_value(TxnWord w)  { return uint32_t(w >> 32); }
};

// Store/load helpers using PTX acquire/release where available.
template<typename TxnWord>
__device__ __forceinline__ void store_release(TxnWord* ptr, TxnWord val) {
    __threadfence();
    *ptr = val;
}

template<typename TxnWord>
__device__ __forceinline__ TxnWord load_relaxed(const TxnWord* ptr) {
    return *const_cast<const volatile TxnWord*>(ptr);
}

// ---------------------------------------------------------------------------
// ScanTileState<T>: per-tile state array with padding.
// Allocated as (num_tiles + TILE_STATUS_PADDING) TxnWords.
// ---------------------------------------------------------------------------
template<typename T>
struct ScanTileState {
    using StatusWord = typename TxnWordTraits<T>::StatusWord;
    using TxnWord    = typename TxnWordTraits<T>::TxnWord;

    TxnWord* d_tile_descriptors;

    // Number of TxnWords to allocate: num_tiles + TILE_STATUS_PADDING
    __host__ static size_t AllocationSize(int num_tiles) {
        return (num_tiles + TILE_STATUS_PADDING) * sizeof(TxnWord);
    }

    // Initialize from device: one thread per entry.
    __device__ void InitializeStatus(int num_tiles) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_tiles) {
            d_tile_descriptors[TILE_STATUS_PADDING + idx] =
                TxnWordTraits<T>::pack(StatusWord(SCAN_TILE_INVALID), T());
        }
        if (blockIdx.x == 0 && threadIdx.x < TILE_STATUS_PADDING) {
            d_tile_descriptors[threadIdx.x] =
                TxnWordTraits<T>::pack(StatusWord(SCAN_TILE_OOB), T());
        }
    }

    // Publish aggregate (partial): single atomic-width write.
    __device__ __forceinline__ void SetPartial(int tile_idx, T value) {
        store_release(d_tile_descriptors + TILE_STATUS_PADDING + tile_idx,
                      TxnWordTraits<T>::pack(StatusWord(SCAN_TILE_PARTIAL), value));
    }

    // Publish inclusive prefix: single atomic-width write.
    __device__ __forceinline__ void SetInclusive(int tile_idx, T value) {
        store_release(d_tile_descriptors + TILE_STATUS_PADDING + tile_idx,
                      TxnWordTraits<T>::pack(StatusWord(SCAN_TILE_INCLUSIVE), value));
    }

    // Spin until tile is non-invalid, return status and value.
    __device__ __forceinline__ void WaitForValid(int tile_idx, StatusWord& status, T& value) {
        TxnWord word = load_relaxed(d_tile_descriptors + TILE_STATUS_PADDING + tile_idx);
        while (__any_sync(0xffffffff,
               TxnWordTraits<T>::unpack_status(word) == StatusWord(SCAN_TILE_INVALID))) {
            __nanosleep(64);
            word = load_relaxed(d_tile_descriptors + TILE_STATUS_PADDING + tile_idx);
        }
        status = TxnWordTraits<T>::unpack_status(word);
        value  = TxnWordTraits<T>::unpack_value(word);
    }
};

// ---------------------------------------------------------------------------
// TilePrefixCallbackOp: CUB-style prefix callback used with BlockScan.
// Called by BlockScan on the first warp only. Each thread looks back at
// predecessor tile_idx - threadIdx.x - 1. cub::WarpReduce::TailSegmentedReduce
// (shuffle-based, no __syncwarp) combines the window. The window slides back
// by WARP until a SCAN_TILE_INCLUSIVE is found.
//
// OOB padding entries have status=SCAN_TILE_OOB (not INCLUSIVE), so the while
// loop stops only on INCLUSIVE — matching CUB's single_pass_scan_operators.
// OOB values are T() (zero). For block 0's predecessors, the TailSegmentedReduce
// accumulates these zero values until the while loop terminates when the warp
// reaches no INCLUSIVE tiles at all (all OOB) — but the identity field is used
// to seed an artificial INCLUSIVE at the OOB boundary instead.
// ---------------------------------------------------------------------------
template<typename T, typename ScanOpT>
struct TilePrefixCallbackOp {
    using StatusWord  = typename ScanTileState<T>::StatusWord;
    using WarpReduceT = cub::WarpReduce<T, WARP>;

    ScanTileState<T>& tile_state;
    ScanOpT           scan_op;
    int               tile_idx;
    T                 identity;
    T                 exclusive_prefix;
    T                 inclusive_prefix;

    struct TempStorage {
        typename WarpReduceT::TempStorage warp_reduce;
        T   exclusive_prefix;
        T   inclusive_prefix;
        T   block_aggregate;
    };

    TempStorage& temp_storage;

    __device__ __forceinline__
    TilePrefixCallbackOp(ScanTileState<T>& tile_state,
                         TempStorage& temp_storage,
                         ScanOpT scan_op,
                         int tile_idx,
                         T identity)
        : tile_state(tile_state)
        , scan_op(scan_op)
        , tile_idx(tile_idx)
        , identity(identity)
        , temp_storage(temp_storage)
    {}

    // Scan one window of WARP predecessor tiles. Returns the window aggregate
    // (combined value from the rightmost INCLUSIVE/OOB tile through lane 0),
    // and sets predecessor_status to this lane's tile status.
    __device__ __forceinline__ T
    ProcessWindow(int predecessor_idx, StatusWord& predecessor_status) {
        T value;
        tile_state.WaitForValid(predecessor_idx, predecessor_status, value);

        // OOB acts like identity: not a stop flag here, but contributes identity value.
        // Only INCLUSIVE is the stop flag for TailSegmentedReduce.
        // For block 0 (all OOB predecessors), the while loop below never fires
        // (because __all_sync(!= INCLUSIVE) would be true forever), BUT
        // we handle block 0 specially: treat OOB as INCLUSIVE with identity value.
        int is_oob    = (predecessor_status == StatusWord(SCAN_TILE_OOB));
        int tail_flag = (predecessor_status == StatusWord(SCAN_TILE_INCLUSIVE)) | is_oob;
        T   eff_value = is_oob ? identity : value;

        // TailSegmentedReduce reduces op(lane_i, lane_{i+1}) toward lane 0, giving
        // op(v[0], op(v[1], ..., v[k])) where lane k holds the INCLUSIVE/OOB stop value.
        // For our scan, lane 0 is the most-recent predecessor and lane k the oldest, so
        // we need compose(oldest, ..., newest) = compose(v[k], ..., v[0]).
        // Flipping the operator gives the correct accumulation direction.
        auto flipped_op = [&](T a, T b) { return scan_op(b, a); };
        return WarpReduceT(temp_storage.warp_reduce)
                   .TailSegmentedReduce(eff_value, tail_flag, flipped_op);
    }

    // Called by BlockScan with the block aggregate; returns exclusive prefix.
    __device__ __forceinline__ T operator()(T block_aggregate) {
        if (threadIdx.x == 0) {
            temp_storage.block_aggregate = block_aggregate;
            tile_state.SetPartial(tile_idx, block_aggregate);
        }

        int predecessor_idx = tile_idx - threadIdx.x - 1;
        StatusWord predecessor_status;

        exclusive_prefix = ProcessWindow(predecessor_idx, predecessor_status);

        // Slide window back until we find an INCLUSIVE tile (or hit all-OOB).
        while (__all_sync(0xffffffff, predecessor_status != StatusWord(SCAN_TILE_INCLUSIVE)
                                   && predecessor_status != StatusWord(SCAN_TILE_OOB))) {
            predecessor_idx -= WARP;
            T window_agg = ProcessWindow(predecessor_idx, predecessor_status);
            exclusive_prefix = scan_op(window_agg, exclusive_prefix);
        }

        // Broadcast exclusive_prefix from lane 0 to all warp lanes via shuffle
        // (avoids shared-memory write + unsynchronized read across warp lanes).
        T ep = (T) __shfl_sync(0xffffffff, (uint32_t) exclusive_prefix, 0);

        if (threadIdx.x == 0) {
            inclusive_prefix = scan_op(ep, block_aggregate);
            tile_state.SetInclusive(tile_idx, inclusive_prefix);
            temp_storage.exclusive_prefix = ep;
            temp_storage.inclusive_prefix = inclusive_prefix;
        }

        return ep;
    }

    __device__ __forceinline__ T GetExclusivePrefix() { return temp_storage.exclusive_prefix; }
    __device__ __forceinline__ T GetInclusivePrefix() { return temp_storage.inclusive_prefix; }
    __device__ __forceinline__ T GetBlockAggregate()  { return temp_storage.block_aggregate;  }
};

// ---------------------------------------------------------------------------
// Kept for backward-compat with non-two-pass kernels that still call
// dynamicIndex directly.
// ---------------------------------------------------------------------------
template<typename I>
__device__ inline I dynamicIndex(volatile I* dyn_idx_ptr) {
    volatile __shared__ I dyn_idx;
    if (threadIdx.x == 0)
        dyn_idx = atomicAdd(const_cast<I*>(dyn_idx_ptr), 1);
    __syncthreads();
    return dyn_idx;
}

template<typename T, typename I, typename OP, I ITEMS_PER_THREAD>
__device__ inline void
glbToShmemCpy(const I glb_offs,
              const I size,
              const T ne,
              T* d_read,
              volatile T* shmem_write) {
    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = i * blockDim.x + threadIdx.x;
        I gid = glb_offs + lid;
        shmem_write[lid] = gid < size ? d_read[gid] : ne;
    }
    __syncthreads();
}

template<typename T, typename I, I ITEMS_PER_THREAD>
__device__ inline void
shmemToGlbCpy(const I glb_offs,
              const I size,
              T* d_write,
              volatile T* shmem_read) {
    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = blockDim.x * i + threadIdx.x;
        I gid = glb_offs + lid;
        if (gid < size)
            d_write[gid] = shmem_read[lid];
    }
    __syncthreads();
}
