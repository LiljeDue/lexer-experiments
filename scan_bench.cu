#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "common/sps.cu.h"
#include "common/util.cu.h"

#define PAD "%-38s "

template<typename I>
struct Add {
    __device__ __host__ __forceinline__ I operator()(I a, I b) const { return a + b; }
};

template<typename T>
__global__ void initScanTileStateKernel(ScanTileState<T> state, int num_tiles) {
    state.InitializeStatus(num_tiles);
}

template<typename T>
static void initScanTileState(ScanTileState<T>& state, int num_tiles) {
    int threads = 256;
    int blocks  = (num_tiles + TILE_STATUS_PADDING + threads - 1) / threads;
    initScanTileStateKernel<T><<<blocks, threads>>>(state, num_tiles);
}

// ---------------------------------------------------------------------------
// Simple i32 addition scan using our TilePrefixCallbackOp
// ---------------------------------------------------------------------------

template<typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__ void ourScanKernel(
    const uint32_t* __restrict__ d_in,
    uint32_t* __restrict__ d_out,
    ScanTileState<uint32_t> tile_state,
    I n,
    volatile uint32_t* dyn_idx_ptr)
{
    using BlockScan   = cub::BlockScan<uint32_t, BLOCK_SIZE>;
    using PrefixOp    = TilePrefixCallbackOp<uint32_t, Add<I>>;

    __shared__ typename BlockScan::TempStorage scan_storage;
    __shared__ typename PrefixOp::TempStorage  prefix_storage;

    uint32_t tile_idx = dynamicIndex<uint32_t>(dyn_idx_ptr);
    I glb_offs = (I)tile_idx * BLOCK_SIZE * ITEMS_PER_THREAD;

    uint32_t items[ITEMS_PER_THREAD];
    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I gid = glb_offs + threadIdx.x * ITEMS_PER_THREAD + i;
        items[i] = gid < n ? d_in[gid] : 0;
    }

    PrefixOp prefix_op(tile_state, prefix_storage, Add<I>(), (int)tile_idx, uint32_t(0));
    BlockScan(scan_storage).InclusiveScan(items, items, Add<I>(), prefix_op);

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I gid = glb_offs + threadIdx.x * ITEMS_PER_THREAD + i;
        if (gid < n) d_out[gid] = items[i];
    }
}

template<typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__ void ourScanKernelStatic(
    const uint32_t* __restrict__ d_in,
    uint32_t* __restrict__ d_out,
    ScanTileState<uint32_t> tile_state,
    I n)
{
    using BlockScan = cub::BlockScan<uint32_t, BLOCK_SIZE>;
    using PrefixOp  = TilePrefixCallbackOp<uint32_t, Add<I>>;

    __shared__ typename BlockScan::TempStorage scan_storage;
    __shared__ typename PrefixOp::TempStorage  prefix_storage;

    I tile_idx = blockIdx.x;
    I glb_offs = tile_idx * BLOCK_SIZE * ITEMS_PER_THREAD;

    uint32_t items[ITEMS_PER_THREAD];
    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I gid = glb_offs + threadIdx.x * ITEMS_PER_THREAD + i;
        items[i] = gid < n ? d_in[gid] : 0;
    }

    PrefixOp prefix_op(tile_state, prefix_storage, Add<I>(), (int)tile_idx, uint32_t(0));
    BlockScan(scan_storage).InclusiveScan(items, items, Add<I>(), prefix_op);

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I gid = glb_offs + threadIdx.x * ITEMS_PER_THREAD + i;
        if (gid < n) d_out[gid] = items[i];
    }
}

// Striped layout: thread t owns items t, t+BS, t+2*BS, ... (coalesced access).
// BlockScan expects striped input when using BlockScan<..., BLOCK_SCAN_WARP_SCANS>.
// We use cub::BlockExchange to convert striped->blocked before scan and blocked->striped after.
template<typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__ void ourScanKernelStriped(
    const uint32_t* __restrict__ d_in,
    uint32_t* __restrict__ d_out,
    ScanTileState<uint32_t> tile_state,
    I n)
{
    using BlockScan     = cub::BlockScan<uint32_t, BLOCK_SIZE>;
    using BlockExchange = cub::BlockExchange<uint32_t, BLOCK_SIZE, ITEMS_PER_THREAD>;
    using PrefixOp      = TilePrefixCallbackOp<uint32_t, Add<I>>;

    __shared__ union {
        typename BlockScan::TempStorage     scan;
        typename BlockExchange::TempStorage exchange;
    } temp;
    __shared__ typename PrefixOp::TempStorage prefix_storage;

    I tile_idx = blockIdx.x;
    I glb_offs = tile_idx * BLOCK_SIZE * ITEMS_PER_THREAD;

    // Striped load: thread t reads positions glb_offs+t, glb_offs+t+BS, ...
    uint32_t items[ITEMS_PER_THREAD];
    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I gid = glb_offs + i * BLOCK_SIZE + threadIdx.x;
        items[i] = gid < n ? d_in[gid] : 0;
    }

    // Convert striped -> blocked for BlockScan
    BlockExchange(temp.exchange).StripedToBlocked(items);

    PrefixOp prefix_op(tile_state, prefix_storage, Add<I>(), (int)tile_idx, uint32_t(0));
    BlockScan(temp.scan).InclusiveScan(items, items, Add<I>(), prefix_op);

    // Convert blocked -> striped for coalesced store
    BlockExchange(temp.exchange).BlockedToStriped(items);

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I gid = glb_offs + i * BLOCK_SIZE + threadIdx.x;
        if (gid < n) d_out[gid] = items[i];
    }
}

// Kernel to fill d_in with a simple pattern (all ones) for easy verification
__global__ void fillOnes(uint32_t* d, uint32_t n) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) d[gid] = 1;
}

int main() {
    using I = uint32_t;
    const I BLOCK_SIZE       = 256;
    const I ITEMS_PER_THREAD = 16;
    const I TILE             = BLOCK_SIZE * ITEMS_PER_THREAD;

    // 500 MiB of uint32_t = 128M elements
    const I N = 128u * 1024u * 1024u;
    const size_t BYTES = (size_t)N * sizeof(uint32_t);

#ifdef PROFILE
    const int WARMUP = 1;
    const int RUNS   = 1;
#else
    const int WARMUP = 100;
    const int RUNS   = 100;
#endif

    uint32_t *d_in, *d_out_ours, *d_out_cub;
    gpuAssert(cudaMalloc(&d_in,       BYTES));
    gpuAssert(cudaMalloc(&d_out_ours, BYTES));
    gpuAssert(cudaMalloc(&d_out_cub,  BYTES));

    // Fill input with 1s
    fillOnes<<<(N + 255) / 256, 256>>>(d_in, N);
    gpuAssert(cudaDeviceSynchronize());

    // --- CUB DeviceScan ---
    void*  d_temp_cub = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_cub, temp_bytes, d_in, d_out_cub, (int)N);
    gpuAssert(cudaMalloc(&d_temp_cub, temp_bytes));

    cudaEvent_t start, stop;
    gpuAssert(cudaEventCreate(&start));
    gpuAssert(cudaEventCreate(&stop));
    float* times_cub     = (float*)malloc(sizeof(float) * RUNS);
    float* times_ours    = (float*)malloc(sizeof(float) * RUNS);
    float* times_static  = (float*)malloc(sizeof(float) * RUNS);
    float* times_striped = (float*)malloc(sizeof(float) * RUNS);

    // Warmup CUB
    for (int i = 0; i < WARMUP; i++) {
        cub::DeviceScan::InclusiveSum(d_temp_cub, temp_bytes, d_in, d_out_cub, (int)N);
        gpuAssert(cudaDeviceSynchronize());
    }
    // Bench CUB
    for (int i = 0; i < RUNS; i++) {
        gpuAssert(cudaEventRecord(start));
        cub::DeviceScan::InclusiveSum(d_temp_cub, temp_bytes, d_in, d_out_cub, (int)N);
        gpuAssert(cudaEventRecord(stop));
        gpuAssert(cudaEventSynchronize(stop));
        gpuAssert(cudaEventElapsedTime(&times_cub[i], start, stop));
    }

    // --- Our scan ---
    const I NLB = (N + TILE - 1) / TILE;
    uint32_t* d_dyn_idx;
    ScanTileState<uint32_t> tile_state;
    gpuAssert(cudaMalloc(&d_dyn_idx, sizeof(uint32_t)));
    gpuAssert(cudaMalloc(&tile_state.d_tile_descriptors,
                         ScanTileState<uint32_t>::AllocationSize(NLB)));

    auto reset_ours = [&]() {
        gpuAssert(cudaMemset(d_dyn_idx, 0, sizeof(uint32_t)));
        initScanTileState(tile_state, (int)NLB);
    };

    // Warmup ours
    for (int i = 0; i < WARMUP; i++) {
        reset_ours();
        ourScanKernel<I, BLOCK_SIZE, ITEMS_PER_THREAD>
            <<<NLB, BLOCK_SIZE>>>(d_in, d_out_ours, tile_state, N, d_dyn_idx);
        gpuAssert(cudaDeviceSynchronize());
    }
    // Bench ours
    for (int i = 0; i < RUNS; i++) {
        reset_ours();
        gpuAssert(cudaEventRecord(start));
        ourScanKernel<I, BLOCK_SIZE, ITEMS_PER_THREAD>
            <<<NLB, BLOCK_SIZE>>>(d_in, d_out_ours, tile_state, N, d_dyn_idx);
        gpuAssert(cudaEventRecord(stop));
        gpuAssert(cudaEventSynchronize(stop));
        gpuAssert(cudaEventElapsedTime(&times_ours[i], start, stop));
    }

    // --- Our scan (static blockIdx.x, striped) ---
    uint32_t* d_out_striped;
    gpuAssert(cudaMalloc(&d_out_striped, BYTES));

    auto reset_striped = [&]() { initScanTileState(tile_state, (int)NLB); };

    for (int i = 0; i < WARMUP; i++) {
        reset_striped();
        ourScanKernelStriped<I, BLOCK_SIZE, ITEMS_PER_THREAD>
            <<<NLB, BLOCK_SIZE>>>(d_in, d_out_striped, tile_state, N);
        gpuAssert(cudaDeviceSynchronize());
    }
    for (int i = 0; i < RUNS; i++) {
        reset_striped();
        gpuAssert(cudaEventRecord(start));
        ourScanKernelStriped<I, BLOCK_SIZE, ITEMS_PER_THREAD>
            <<<NLB, BLOCK_SIZE>>>(d_in, d_out_striped, tile_state, N);
        gpuAssert(cudaEventRecord(stop));
        gpuAssert(cudaEventSynchronize(stop));
        gpuAssert(cudaEventElapsedTime(&times_striped[i], start, stop));
    }

    // --- Our scan (static blockIdx.x) ---
    uint32_t* d_out_static;
    gpuAssert(cudaMalloc(&d_out_static, BYTES));

    auto reset_static = [&]() { initScanTileState(tile_state, (int)NLB); };

    for (int i = 0; i < WARMUP; i++) {
        reset_static();
        ourScanKernelStatic<I, BLOCK_SIZE, ITEMS_PER_THREAD>
            <<<NLB, BLOCK_SIZE>>>(d_in, d_out_static, tile_state, N);
        gpuAssert(cudaDeviceSynchronize());
    }
    for (int i = 0; i < RUNS; i++) {
        reset_static();
        gpuAssert(cudaEventRecord(start));
        ourScanKernelStatic<I, BLOCK_SIZE, ITEMS_PER_THREAD>
            <<<NLB, BLOCK_SIZE>>>(d_in, d_out_static, tile_state, N);
        gpuAssert(cudaEventRecord(stop));
        gpuAssert(cudaEventSynchronize(stop));
        gpuAssert(cudaEventElapsedTime(&times_static[i], start, stop));
    }

    // Verify
    std::vector<uint32_t> h_cub(N), h_ours(N), h_static(N), h_striped(N);
    gpuAssert(cudaMemcpy(h_cub.data(),     d_out_cub,     BYTES, cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(h_ours.data(),    d_out_ours,    BYTES, cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(h_static.data(),  d_out_static,  BYTES, cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(h_striped.data(), d_out_striped, BYTES, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (I i = 0; i < N && ok; i++) {
        if (h_cub[i] != h_ours[i]) {
            printf("MISMATCH (dynamic) at i=%u: cub=%u ours=%u\n", i, h_cub[i], h_ours[i]);
            ok = false;
        }
        if (h_cub[i] != h_static[i]) {
            printf("MISMATCH (static) at i=%u: cub=%u ours=%u\n", i, h_cub[i], h_static[i]);
            ok = false;
        }
        if (h_cub[i] != h_striped[i]) {
            printf("MISMATCH (striped) at i=%u: cub=%u ours=%u\n", i, h_cub[i], h_striped[i]);
            ok = false;
        }
    }
    if (ok) printf("Results match (all %u elements correct)\n\n", N);

    // Report (2*BYTES = read + write)
    printf(PAD, "CUB DeviceScan::InclusiveSum:");
    compute_descriptors(times_cub, RUNS, 2 * BYTES);
    printf(PAD, "Our TilePrefixCallbackOp (dynamic):");
    compute_descriptors(times_ours, RUNS, 2 * BYTES);
    printf(PAD, "Our TilePrefixCallbackOp (static):");
    compute_descriptors(times_static, RUNS, 2 * BYTES);
    printf(PAD, "Our TilePrefixCallbackOp (striped):");
    compute_descriptors(times_striped, RUNS, 2 * BYTES);

    free(times_cub);
    free(times_ours);
    free(times_static);
    free(times_striped);
    gpuAssert(cudaFree(d_out_static));
    gpuAssert(cudaFree(d_out_striped));
    gpuAssert(cudaFree(d_in));
    gpuAssert(cudaFree(d_out_ours));
    gpuAssert(cudaFree(d_out_cub));
    gpuAssert(cudaFree(d_temp_cub));
    gpuAssert(cudaFree(d_dyn_idx));
    gpuAssert(cudaFree(tile_state.d_tile_descriptors));
    return ok ? 0 : 1;
}
