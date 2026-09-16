#include <cuda_runtime.h>
#include <cstdint>

const uint8_t LG_WARP = 5;
const uint8_t WARP = 1 << LG_WARP;

// Number of padding entries prepended to the tile state array.
// Threads in block 0 look back at indices tile_idx - threadIdx.x - 1,
// which for threadIdx.x up to WARP-1 gives indices as low as -WARP.
// Padding these with SCAN_TILE_INCLUSIVE (OOB) means the lookback
// terminates immediately for block 0 without any special-casing.
const uint32_t TILE_STATUS_PADDING = WARP;

enum ScanTileStatus : uint32_t {
    SCAN_TILE_OOB       = 0,  // padding: acts as inclusive prefix of identity
    SCAN_TILE_INVALID   = 1,  // not yet published
    SCAN_TILE_PARTIAL   = 2,  // aggregate published, prefix not yet known
    SCAN_TILE_INCLUSIVE = 3,  // inclusive prefix published
};

// ---------------------------------------------------------------------------
// Single-word tile state: packs status + value into one TxnWord so that
// publish and read are each a single atomic-width transaction.
//
// Layout (little-endian):
//   TxnWord = [ value (sizeof(T) bytes) | status (sizeof(T) bytes) ]
//
// T=uint16_t: TxnWord=uint32_t, StatusWord=uint16_t
// T=uint32_t: TxnWord=uint64_t, StatusWord=uint32_t
// ---------------------------------------------------------------------------
template<typename T>
struct TxnWordTraits;

template<> struct TxnWordTraits<uint16_t> {
    using TxnWord    = uint32_t;
    using StatusWord = uint16_t;
};
template<> struct TxnWordTraits<uint32_t> {
    using TxnWord    = unsigned long long;
    using StatusWord = uint32_t;
};

template<typename T>
struct TileDescriptor {
    typename TxnWordTraits<T>::StatusWord status;
    T value;
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
// Padding entries are initialised to SCAN_TILE_OOB so that the lookback
// in block 0 terminates immediately.
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
        TxnWord val = TxnWord();
        TileDescriptor<T>* desc = reinterpret_cast<TileDescriptor<T>*>(&val);
        if (idx < num_tiles) {
            desc->status = StatusWord(SCAN_TILE_INVALID);
            d_tile_descriptors[TILE_STATUS_PADDING + idx] = val;
        }
        if (blockIdx.x == 0 && threadIdx.x < TILE_STATUS_PADDING) {
            desc->status = StatusWord(SCAN_TILE_OOB);
            d_tile_descriptors[threadIdx.x] = val;
        }
    }

    // Publish aggregate (partial): single atomic-width write.
    __device__ __forceinline__ void SetPartial(int tile_idx, T value) {
        TileDescriptor<T> desc;
        desc.status = StatusWord(SCAN_TILE_PARTIAL);
        desc.value  = value;
        TxnWord word;
        *reinterpret_cast<TileDescriptor<T>*>(&word) = desc;
        store_release(d_tile_descriptors + TILE_STATUS_PADDING + tile_idx, word);
    }

    // Publish inclusive prefix: single atomic-width write.
    __device__ __forceinline__ void SetInclusive(int tile_idx, T value) {
        TileDescriptor<T> desc;
        desc.status = StatusWord(SCAN_TILE_INCLUSIVE);
        desc.value  = value;
        TxnWord word;
        *reinterpret_cast<TileDescriptor<T>*>(&word) = desc;
        store_release(d_tile_descriptors + TILE_STATUS_PADDING + tile_idx, word);
    }

    // Spin until tile is non-invalid, return status and value.
    __device__ __forceinline__ void WaitForValid(int tile_idx, StatusWord& status, T& value) {
        TxnWord word = load_relaxed(d_tile_descriptors + TILE_STATUS_PADDING + tile_idx);
        TileDescriptor<T> desc = reinterpret_cast<TileDescriptor<T>&>(word);
        while (__any_sync(0xffffffff, desc.status == StatusWord(SCAN_TILE_INVALID))) {
            __nanosleep(64);
            word = load_relaxed(d_tile_descriptors + TILE_STATUS_PADDING + tile_idx);
            desc = reinterpret_cast<TileDescriptor<T>&>(word);
        }
        status = desc.status;
        value  = desc.value;
    }
};

// ---------------------------------------------------------------------------
// TilePrefixCallbackOp: CUB-style prefix callback used with BlockScan.
// Called by BlockScan on the first warp only. Each thread looks back at
// predecessor tile_idx - threadIdx.x - 1. A WarpReduce with tail-segment
// flag combines the window. The window slides back by WARP until a
// SCAN_TILE_INCLUSIVE is found.
// ---------------------------------------------------------------------------
template<typename T, typename ScanOpT>
struct TilePrefixCallbackOp {
    using StatusWord = typename ScanTileState<T>::StatusWord;

    ScanTileState<T>& tile_state;
    ScanOpT           scan_op;
    int               tile_idx;
    T                 identity;
    T                 exclusive_prefix;
    T                 inclusive_prefix;

    // Temporary storage for warp reduce.
    struct TempStorage {
        T   warp_vals[WARP];
        int warp_flags[WARP];
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

    // Called by BlockScan with the block aggregate; returns exclusive prefix.
    __device__ __forceinline__ T operator()(T block_aggregate) {
        // Thread 0 publishes partial aggregate.
        if (threadIdx.x == 0) {
            temp_storage.block_aggregate = block_aggregate;
            tile_state.SetPartial(tile_idx, block_aggregate);
        }

        // All 32 warp threads look back at their predecessor window simultaneously.
        int predecessor_idx = tile_idx - threadIdx.x - 1;

        StatusWord pred_status;
        T          pred_value;
        tile_state.WaitForValid(predecessor_idx, pred_status, pred_value);

        // OOB padding acts as identity: treat it as a stop (like INCLUSIVE) with
        // the scan identity as the effective value.
        int is_oob    = (pred_status == StatusWord(SCAN_TILE_OOB));
        int tail_flag = (pred_status == StatusWord(SCAN_TILE_INCLUSIVE)) | is_oob;
        T   eff_value = is_oob ? identity : pred_value;
        temp_storage.warp_vals[threadIdx.x]  = eff_value;
        temp_storage.warp_flags[threadIdx.x] = tail_flag;
        __syncwarp();

        // Scan from lane WARP-1 towards lane 0, combining until we hit a tail flag.
        T running = eff_value;
        #pragma unroll
        for (int offset = 1; offset < WARP; offset <<= 1) {
            if (threadIdx.x >= offset) {
                int   src_flag = temp_storage.warp_flags[threadIdx.x - offset];
                T     src_val  = temp_storage.warp_vals[threadIdx.x - offset];
                if (!temp_storage.warp_flags[threadIdx.x]) {
                    running = scan_op(src_val, running);
                    temp_storage.warp_flags[threadIdx.x] = src_flag;
                }
                temp_storage.warp_vals[threadIdx.x] = running;
            }
            __syncwarp();
        }

        // The exclusive prefix is in lane 0 of the warp-scan result from the
        // window. But we may need to slide the window further back.
        exclusive_prefix = temp_storage.warp_vals[0];

        // Continue sliding back only while no lane has found INCLUSIVE or OOB.
        while (__all_sync(0xffffffff, pred_status != StatusWord(SCAN_TILE_INCLUSIVE)
                                   && pred_status != StatusWord(SCAN_TILE_OOB))) {
            predecessor_idx -= WARP;
            tile_state.WaitForValid(predecessor_idx, pred_status, pred_value);

            is_oob    = (pred_status == StatusWord(SCAN_TILE_OOB));
            tail_flag = (pred_status == StatusWord(SCAN_TILE_INCLUSIVE)) | is_oob;
            eff_value = is_oob ? identity : pred_value;
            temp_storage.warp_vals[threadIdx.x]  = eff_value;
            temp_storage.warp_flags[threadIdx.x] = tail_flag;
            __syncwarp();

            running = eff_value;
            #pragma unroll
            for (int offset = 1; offset < WARP; offset <<= 1) {
                if (threadIdx.x >= offset) {
                    int   src_flag = temp_storage.warp_flags[threadIdx.x - offset];
                    T     src_val  = temp_storage.warp_vals[threadIdx.x - offset];
                    if (!temp_storage.warp_flags[threadIdx.x]) {
                        running = scan_op(src_val, running);
                        temp_storage.warp_flags[threadIdx.x] = src_flag;
                    }
                    temp_storage.warp_vals[threadIdx.x] = running;
                }
                __syncwarp();
            }

            exclusive_prefix = scan_op(temp_storage.warp_vals[0], exclusive_prefix);
        }

        // Thread 0 publishes inclusive prefix.
        if (threadIdx.x == 0) {
            inclusive_prefix = scan_op(exclusive_prefix, block_aggregate);
            tile_state.SetInclusive(tile_idx, inclusive_prefix);
            temp_storage.exclusive_prefix = exclusive_prefix;
            temp_storage.inclusive_prefix = inclusive_prefix;
        }
        __syncwarp();

        return temp_storage.exclusive_prefix;
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

