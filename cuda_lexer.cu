#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "common/sps.cu.h"
#include "common/util.cu.h"
#include "common/data.h"
#include <math.h>
#define PAD "%-38s "

using token_t = uint8_t;
using state_t = uint16_t;

const uint32_t NUM_STATES = 12;
const uint32_t NUM_TRANS = 256;
// const token_t IGNORE_TOKEN = 0;
const state_t ENDO_MASK = 15;
const state_t ENDO_OFFSET = 0;
const state_t TOKEN_MASK = 112;
const state_t TOKEN_OFFSET = 4;
const state_t ACCEPT_MASK = 128;
const state_t ACCEPT_OFFSET = 7;
const state_t PRODUCE_MASK = 256;
const state_t PRODUCE_OFFSET = 8;
const state_t IDENTITY = 74;

state_t h_to_state[NUM_TRANS] =
        {75, 75, 75, 75, 75, 75, 75, 75, 75, 128, 128, 75, 75, 128,
         75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
         75, 75, 75, 128, 75, 75, 75, 75, 75, 75, 75, 161, 178, 75,
         75, 75, 75, 75, 75, 147, 147, 147, 147, 147, 147, 147, 147,
         147, 147, 75, 75, 75, 75, 75, 75, 75, 147, 147, 147, 147,
         147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 147,
         147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 75, 75,
         75, 75, 75, 75, 147, 147, 147, 147, 147, 147, 147, 147, 147,
         147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 147,
         147, 147, 147, 147, 147, 75, 75, 75, 75, 75, 75, 75, 75, 75,
         75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
         75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
         75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
         75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
         75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
         75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
         75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
         75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
         75, 75, 75, 75};

state_t h_compose[NUM_STATES * NUM_STATES] =
    {132, 392, 392, 392, 132, 392, 392, 392, 132, 392, 128, 75,
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
     75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75};


__device__ __host__ __forceinline__ state_t get_index(state_t state) {
    return (state & ENDO_MASK) >> ENDO_OFFSET;
}

__device__ __host__ __forceinline__ token_t get_token(state_t state) {
    return (state & TOKEN_MASK) >> TOKEN_OFFSET;
}

__device__ bool is_accept(state_t state) {
    return (state & ACCEPT_MASK) >> ACCEPT_OFFSET;
}

__device__ __host__ __forceinline__ bool is_produce(state_t state) {
    return (state & PRODUCE_MASK) >> PRODUCE_OFFSET;
}

struct LexerCtx {
    state_t* d_to_state;
    state_t* d_compose;

    LexerCtx() : d_to_state(NULL), d_compose(NULL) {
        cudaMalloc(&d_to_state, sizeof(h_to_state));
        cudaMemcpy(d_to_state, h_to_state, sizeof(h_to_state),
                cudaMemcpyHostToDevice);
        cudaMalloc(&d_compose, sizeof(h_compose));
        cudaMemcpy(d_compose, h_compose, sizeof(h_compose),
                cudaMemcpyHostToDevice);
    }

    void Cleanup() {
        if (d_to_state) cudaFree(d_to_state);
        if (d_compose) cudaFree(d_compose);
    }

    __device__ __host__ __forceinline__
    state_t operator()(const state_t &a, const state_t &b) const {
        return d_compose[get_index(b) * NUM_STATES + get_index(a)];
    }

    __device__ __host__ __forceinline__
    state_t operator()(const volatile state_t &a, const volatile state_t &b) const {
        return d_compose[get_index(b) * NUM_STATES + get_index(a)];
    }

    __device__ __host__ __forceinline__
    state_t to_state(const char &a) const {
        return d_to_state[a];
    }
};


// Like LexerCtx but the compose table is loaded into shared memory per block.
// d_compose is set by the kernel to point to the block's __shared__ copy.
struct LexerCtxShmem {
    state_t* d_to_state;
    state_t* d_compose_glb; // global memory source for the per-block shmem load
    state_t* d_compose;     // set to shared memory inside the kernel

    LexerCtxShmem() : d_to_state(NULL), d_compose_glb(NULL), d_compose(NULL) {
        cudaMalloc(&d_to_state, sizeof(h_to_state));
        cudaMemcpy(d_to_state, h_to_state, sizeof(h_to_state),
                cudaMemcpyHostToDevice);
        cudaMalloc(&d_compose_glb, sizeof(h_compose));
        cudaMemcpy(d_compose_glb, h_compose, sizeof(h_compose),
                cudaMemcpyHostToDevice);
    }

    void Cleanup() {
        if (d_to_state) cudaFree(d_to_state);
        if (d_compose_glb) cudaFree(d_compose_glb);
    }

    __device__ __forceinline__
    state_t operator()(const state_t &a, const state_t &b) const {
        return d_compose[get_index(b) * NUM_STATES + get_index(a)];
    }

    __device__ __forceinline__
    state_t operator()(const volatile state_t &a, const volatile state_t &b) const {
        return d_compose[get_index(b) * NUM_STATES + get_index(a)];
    }

    __device__ __forceinline__
    state_t to_state(const char &a) const {
        return d_to_state[a];
    }
};

template<typename I>
struct Add {
    __device__ __forceinline__ I operator()(I a, I b) const {
        return a + b;
    }
};

template <typename T, typename I, I ITEMS_PER_THREAD>
__device__ inline void
copyFromGlbToShr(
    const I glb_offs,
    const I num_items,
    const I size,
    T* glb,
    volatile T* shr
) {
    const I NUM_ITEMS_BYTES = min(num_items, size - glb_offs) * sizeof(T);
    const I TOTAL_ITERS = 1 + (NUM_ITEMS_BYTES - 1) / sizeof(uint64_t);
    const I ITERS = 1 + (TOTAL_ITERS - 1) / (ITEMS_PER_THREAD * blockDim.x);
    
    #pragma unroll
    for (I j = 0; j < ITERS; j++) { 
        #pragma unroll
        for (I i = 0; i < ITEMS_PER_THREAD; i++) {
            I lid = j * ITEMS_PER_THREAD * blockDim.x + i * blockDim.x + threadIdx.x;
            I lid_byte = lid * sizeof(uint64_t);
            if (lid_byte + sizeof(uint64_t) < NUM_ITEMS_BYTES) {
                reinterpret_cast<volatile uint64_t*>(shr)[lid] = reinterpret_cast<uint64_t*>(glb + glb_offs)[lid];
            } else {
                #pragma unroll
                for (I k = lid_byte; k < NUM_ITEMS_BYTES; k++) {
                    reinterpret_cast<volatile uint8_t*>(shr)[k] = reinterpret_cast<uint8_t*>(glb + glb_offs)[k];
                }
            }
        }
    }

    __syncthreads();
}

template <typename T, typename I, I ITEMS_PER_THREAD>
__device__ inline void
copyFromShrToGlb(
    const I glb_offs,
    const I num_items,
    const I size,
    volatile T* shr,
    T* glb
) {
    const I NUM_ITEMS_BYTES = min(num_items, size - glb_offs) * sizeof(T);
    const I TOTAL_ITERS = 1 + (NUM_ITEMS_BYTES - 1) / sizeof(uint64_t);
    const I ITERS = 1 + (TOTAL_ITERS - 1) / (ITEMS_PER_THREAD * blockDim.x);

    #pragma unroll
    for (I j = 0; j < ITERS; j++) { 
        #pragma unroll
        for (I i = 0; i < ITEMS_PER_THREAD; i++) {
            I lid = j * ITEMS_PER_THREAD * blockDim.x + i * blockDim.x + threadIdx.x;
            I lid_byte = lid * sizeof(uint64_t);
            if (lid_byte + sizeof(uint64_t) < NUM_ITEMS_BYTES) {
                reinterpret_cast<uint64_t*>(glb + glb_offs)[lid] = reinterpret_cast<volatile uint64_t*>(shr)[lid];
            } else {
                #pragma unroll
                for (I k = lid_byte; k < NUM_ITEMS_BYTES; k++) {
                    reinterpret_cast<uint8_t*>(glb + glb_offs)[k] = reinterpret_cast<volatile uint8_t*>(shr)[k];
                }
            }
        }
    }

    __syncthreads();
}

// Load ITEMS_PER_THREAD bytes per thread from d_in[glb_offs..] using 64-bit
// coalesced loads, map each byte through to_state[], and write states to shmem
// in sequential layout (byte p → states[p - glb_offs]).
// Out-of-bounds positions get `identity`.
// If EXTRA=1, also stores the one byte at position glb_offs+TILE into *next_state.
//
// REG_MEM = 1 + IPT/8 loads per thread (e.g. 4 for IPT=30). Loop is small
// and fully unrolled. Global reads are coalesced: load i by thread t reads
// global bytes glb_offs + (i*BS+t)*8 .. +7.
template<typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD, I EXTRA=0>
__device__ inline void
loadBytesAsStates(
    const uint8_t* __restrict__ d_in,
    I glb_offs, I size,
    const state_t* __restrict__ to_state,
    volatile state_t* states,
    state_t identity,
    state_t* next_state = nullptr)
{
    const I U8    = sizeof(uint64_t);
    const I TILE  = ITEMS_PER_THREAD * BLOCK_SIZE;
    const I LOADS = 1 + (ITEMS_PER_THREAD + EXTRA) / U8;
    uint64_t regs[LOADS];
    uint8_t* bytes = (uint8_t*)regs;
    #pragma unroll
    for (I i = 0; i < LOADS; i++) {
        I base      = i * BLOCK_SIZE + threadIdx.x;
        I base_byte = base * U8;
        I gid       = glb_offs + base_byte;
        if (gid + U8 <= size) {
            regs[i] = *reinterpret_cast<const uint64_t*>(d_in + gid);
        } else {
            regs[i] = 0;
            #pragma unroll
            for (I j = 0; j < U8; j++)
                if (gid + j < size) bytes[i * U8 + j] = d_in[gid + j];
        }
    }
    #pragma unroll
    for (I i = 0; i < LOADS; i++) {
        #pragma unroll
        for (I j = 0; j < U8; j++) {
            I lid = (i * BLOCK_SIZE + threadIdx.x) * U8 + j;
            if (lid < TILE) {
                states[lid] = (glb_offs + lid < size)
                              ? to_state[bytes[i * U8 + j]] : identity;
            } else if (EXTRA && lid == TILE) {
                if (glb_offs + lid < size)
                    *next_state = to_state[bytes[i * U8 + j]];
            }
        }
    }
}

// Convert register array from striped to blocked layout using shmem as scratch.
// shmem must be at least ITEMS_PER_THREAD*BLOCK_SIZE elements of type T.
template<typename T, typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__device__ inline void
stripedToBlocked(T (&regs)[ITEMS_PER_THREAD], volatile T* shmem) {
    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++)
        shmem[i * BLOCK_SIZE + threadIdx.x] = regs[i];
    __syncthreads();
    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++)
        regs[i] = shmem[threadIdx.x * ITEMS_PER_THREAD + i];
    __syncthreads();
}

// Convert register array from blocked to striped layout using shmem as scratch.
template<typename T, typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__device__ inline void
blockedToStriped(T (&regs)[ITEMS_PER_THREAD], volatile T* shmem) {
    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++)
        shmem[threadIdx.x * ITEMS_PER_THREAD + i] = regs[i];
    __syncthreads();
    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++)
        regs[i] = shmem[i * BLOCK_SIZE + threadIdx.x];
    __syncthreads();
}

template<typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__ void
lexer(LexerCtx ctx,
      uint8_t* d_in,
      uint32_t* d_index_out,
      token_t* d_token_out,
      ScanTileState<state_t> state_states,
      ScanTileState<I> index_states,
      I size,
      I num_logical_blocks,
      volatile uint32_t* dyn_index_ptr,
      volatile I* new_size,
      volatile bool* is_valid) {
    static_assert(ITEMS_PER_THREAD <= 64, "ITEMS_PER_THREAD exceeds 64-bit is_produce_state capacity");
    using BlockScanState = cub::BlockScan<state_t, BLOCK_SIZE>;
    using BlockScanI     = cub::BlockScan<I, BLOCK_SIZE>;
    using PrefixOpState  = TilePrefixCallbackOp<state_t, LexerCtx>;
    using PrefixOpIdx    = TilePrefixCallbackOp<I, Add<I>>;
    __shared__ typename BlockScanState::TempStorage state_temp;
    __shared__ typename BlockScanI::TempStorage     index_temp;
    __shared__ typename PrefixOpState::TempStorage  state_prefix_storage;
    __shared__ typename PrefixOpIdx::TempStorage    index_prefix_storage;
    volatile __shared__ state_t states[ITEMS_PER_THREAD * BLOCK_SIZE];
    volatile __shared__ uint8_t tok_stage[ITEMS_PER_THREAD * BLOCK_SIZE];
    volatile __shared__ state_t to_state_shr[256];
    __shared__ state_t next_block_first_state;

    state_t st[ITEMS_PER_THREAD];
    I prod[ITEMS_PER_THREAD];
    uint64_t is_produce_state = 0;

    uint32_t dyn_index = dynamicIndex<uint32_t>(dyn_index_ptr);
    I glb_offs = dyn_index * BLOCK_SIZE * ITEMS_PER_THREAD;

    // Load to_state map into shmem (LexerCtx has no compose table).
    #pragma unroll
    for (I i = threadIdx.x; i < 256; i += BLOCK_SIZE)
        to_state_shr[i] = ctx.to_state(i);

    if (threadIdx.x == I())
        next_block_first_state = IDENTITY;

    __syncthreads();

    loadBytesAsStates<I, BLOCK_SIZE, ITEMS_PER_THREAD, 1>(
        d_in, glb_offs, size, (const state_t*)to_state_shr,
        states, state_t(IDENTITY), &next_block_first_state);

    __syncthreads();

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++)
        st[i] = states[threadIdx.x * ITEMS_PER_THREAD + i];

    PrefixOpState state_prefix_op(state_states, state_prefix_storage, ctx, (int)dyn_index, state_t(IDENTITY));
    BlockScanState(state_temp).InclusiveScan(st, st, ctx, state_prefix_op);

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++)
        states[threadIdx.x * ITEMS_PER_THREAD + i] = st[i];

    __syncthreads();

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = threadIdx.x * ITEMS_PER_THREAD + i;
        I gid = glb_offs + lid;
        bool temp = false;
        if (gid < size) {
            if (lid == ITEMS_PER_THREAD * BLOCK_SIZE - 1) {
                temp = gid == size - 1 || is_produce(ctx(st[i], next_block_first_state));
            } else {
                temp = gid == size - 1 || is_produce(states[lid + 1]);
            }
        }
        is_produce_state |= (uint64_t)temp << i;
        prod[i] = (I)temp;
    }

    PrefixOpIdx index_prefix_op(index_states, index_prefix_storage, Add<I>(), (int)dyn_index, I(0));
    BlockScanI(index_temp).InclusiveScan(prod, prod, Add<I>(), index_prefix_op);
    I idx_pfx  = index_prefix_op.GetExclusivePrefix();
    I prod_agg = index_prefix_op.GetBlockAggregate();

    volatile uint16_t* lid_stage = (volatile uint16_t*) states;

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        if ((is_produce_state >> i) & 1) {
            I slot = prod[i] - 1 - idx_pfx;
            I lid  = threadIdx.x * ITEMS_PER_THREAD + i;
            tok_stage[slot] = get_token(st[i]);
            lid_stage[slot] = (uint16_t) lid;
        }
    }

    __syncthreads();

    for (I slot = threadIdx.x; slot < prod_agg; slot += BLOCK_SIZE) {
        I out_idx = idx_pfx + slot;
        d_index_out[out_idx] = glb_offs + lid_stage[slot];
        d_token_out[out_idx] = tok_stage[slot];
    }

    if (dyn_index == num_logical_blocks - 1 && threadIdx.x == blockDim.x - 1) {
        *new_size = Add<I>()(idx_pfx, prod_agg);
        *is_valid = is_accept(st[ITEMS_PER_THREAD - 1]);
    }
}

template<typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__ void
lexerShmemCompose(LexerCtxShmem ctx,
      uint8_t* d_in,
      uint32_t* d_index_out,
      token_t* d_token_out,
      ScanTileState<state_t> state_states,
      ScanTileState<I> index_states,
      I size,
      I num_logical_blocks,
      volatile uint32_t* dyn_index_ptr,
      volatile I* new_size,
      volatile bool* is_valid) {
    static_assert(ITEMS_PER_THREAD <= 64, "ITEMS_PER_THREAD exceeds 64-bit is_produce_state capacity");
    using BlockScanState = cub::BlockScan<state_t, BLOCK_SIZE>;
    using BlockScanI     = cub::BlockScan<I, BLOCK_SIZE>;
    using PrefixOpState  = TilePrefixCallbackOp<state_t, LexerCtxShmem>;
    using PrefixOpIdx    = TilePrefixCallbackOp<I, Add<I>>;
    __shared__ typename BlockScanState::TempStorage state_temp;
    __shared__ typename BlockScanI::TempStorage     index_temp;
    __shared__ typename PrefixOpState::TempStorage  state_prefix_storage;
    __shared__ typename PrefixOpIdx::TempStorage    index_prefix_storage;
    __shared__ __align__(8) state_t shmem_compose[NUM_STATES * NUM_STATES];
    volatile __shared__ state_t states[ITEMS_PER_THREAD * BLOCK_SIZE];
    volatile __shared__ uint8_t tok_stage[ITEMS_PER_THREAD * BLOCK_SIZE];
    volatile __shared__ state_t to_state_shr[256];
    __shared__ state_t next_block_first_state;

    state_t st[ITEMS_PER_THREAD];
    I prod[ITEMS_PER_THREAD];
    uint64_t is_produce_state = 0;

    copyFromGlbToShr<state_t, I, 1>(0, NUM_STATES * NUM_STATES, NUM_STATES * NUM_STATES, ctx.d_compose_glb, shmem_compose);
    ctx.d_compose = shmem_compose;
    copyFromGlbToShr<state_t, I, 1>(0, 256, 256, ctx.d_to_state, to_state_shr);

    uint32_t dyn_index = dynamicIndex<uint32_t>(dyn_index_ptr);
    I glb_offs = dyn_index * BLOCK_SIZE * ITEMS_PER_THREAD;

    if (threadIdx.x == I())
        next_block_first_state = IDENTITY;

    __syncthreads();

    loadBytesAsStates<I, BLOCK_SIZE, ITEMS_PER_THREAD, 1>(
        d_in, glb_offs, size, (const state_t*)to_state_shr,
        states, state_t(IDENTITY), &next_block_first_state);

    __syncthreads();

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++)
        st[i] = states[threadIdx.x * ITEMS_PER_THREAD + i];

    PrefixOpState state_prefix_op(state_states, state_prefix_storage, ctx, (int)dyn_index, state_t(IDENTITY));
    BlockScanState(state_temp).InclusiveScan(st, st, ctx, state_prefix_op);

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++)
        states[threadIdx.x * ITEMS_PER_THREAD + i] = st[i];

    __syncthreads();

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = threadIdx.x * ITEMS_PER_THREAD + i;
        I gid = glb_offs + lid;
        bool temp = false;
        if (gid < size) {
            if (lid == ITEMS_PER_THREAD * BLOCK_SIZE - 1) {
                temp = gid == size - 1 || is_produce(ctx(st[i], next_block_first_state));
            } else {
                temp = gid == size - 1 || is_produce(states[lid + 1]);
            }
        }
        is_produce_state |= (uint64_t)temp << i;
        prod[i] = (I)temp;
    }

    PrefixOpIdx index_prefix_op(index_states, index_prefix_storage, Add<I>(), (int)dyn_index, I(0));
    BlockScanI(index_temp).InclusiveScan(prod, prod, Add<I>(), index_prefix_op);
    I idx_pfx  = index_prefix_op.GetExclusivePrefix();
    I prod_agg = index_prefix_op.GetBlockAggregate();

    volatile uint16_t* lid_stage = (volatile uint16_t*) states;

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        if ((is_produce_state >> i) & 1) {
            I slot = prod[i] - 1 - idx_pfx;
            I lid  = threadIdx.x * ITEMS_PER_THREAD + i;
            tok_stage[slot] = get_token(st[i]);
            lid_stage[slot] = (uint16_t) lid;
        }
    }

    __syncthreads();

    for (I slot = threadIdx.x; slot < prod_agg; slot += BLOCK_SIZE) {
        I out_idx = idx_pfx + slot;
        d_index_out[out_idx] = glb_offs + lid_stage[slot];
        d_token_out[out_idx] = tok_stage[slot];
    }

    if (dyn_index == num_logical_blocks - 1 && threadIdx.x == blockDim.x - 1) {
        *new_size = Add<I>()(idx_pfx, prod_agg);
        *is_valid = is_accept(st[ITEMS_PER_THREAD - 1]);
    }
}

template<typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__ void
lexerShmemComposeU64(LexerCtxShmem ctx,
      uint8_t* d_in,
      uint32_t* d_index_out,
      token_t* d_token_out,
      ScanTileState<state_t> state_states,
      ScanTileState<I> index_states,
      I size,
      I num_logical_blocks,
      volatile uint32_t* dyn_index_ptr,
      volatile I* new_size,
      volatile bool* is_valid) {
    static_assert(ITEMS_PER_THREAD <= 64, "ITEMS_PER_THREAD exceeds 64-bit is_produce_state capacity");
    using BlockScanState = cub::BlockScan<state_t, BLOCK_SIZE>;
    using BlockScanI     = cub::BlockScan<I, BLOCK_SIZE>;
    using PrefixOpState  = TilePrefixCallbackOp<state_t, LexerCtxShmem>;
    using PrefixOpIdx    = TilePrefixCallbackOp<I, Add<I>>;
    __shared__ typename BlockScanState::TempStorage state_temp;
    __shared__ typename BlockScanI::TempStorage     index_temp;
    __shared__ typename PrefixOpState::TempStorage  state_prefix_storage;
    __shared__ typename PrefixOpIdx::TempStorage    index_prefix_storage;
    __shared__ __align__(8) state_t shmem_compose[NUM_STATES * NUM_STATES];
    volatile __shared__ state_t states[ITEMS_PER_THREAD * BLOCK_SIZE];
    volatile __shared__ uint8_t tok_stage[ITEMS_PER_THREAD * BLOCK_SIZE];
    volatile __shared__ state_t to_state_shr[256];
    __shared__ state_t next_block_first_state;

    const I REG_MEM = 1 + ITEMS_PER_THREAD / sizeof(uint64_t);
    uint64_t copy_reg[REG_MEM];
    uint8_t *chars_reg = (uint8_t*) copy_reg;
    state_t st[ITEMS_PER_THREAD];
    I prod[ITEMS_PER_THREAD];
    uint64_t is_produce_state = 0;

    copyFromGlbToShr<state_t, I, 1>(0, NUM_STATES * NUM_STATES, NUM_STATES * NUM_STATES, ctx.d_compose_glb, shmem_compose);
    ctx.d_compose = shmem_compose;
    copyFromGlbToShr<state_t, I, 1>(0, 256, 256, ctx.d_to_state, to_state_shr);

    uint32_t dyn_index = dynamicIndex<uint32_t>(dyn_index_ptr);
    I glb_offs = dyn_index * BLOCK_SIZE * ITEMS_PER_THREAD;

    if (threadIdx.x == I())
        next_block_first_state = IDENTITY;

    __syncthreads();

    #pragma unroll
    for (I i = 0; i < REG_MEM; i++) {
        I uint64_lid = i * blockDim.x + threadIdx.x;
        I lid = sizeof(uint64_t) * uint64_lid;
        I gid = glb_offs + lid;
        if (gid + sizeof(uint64_t) < size) {
            copy_reg[i] = *((uint64_t*) (gid + (uint8_t*) d_in));
        } else {
            for (I j = 0; j < sizeof(uint64_t); j++) {
                I loc_gid = gid + j;
                if (loc_gid < size)
                    chars_reg[sizeof(uint64_t) * i + j] = d_in[loc_gid];
            }
        }
    }

    #pragma unroll
    for (I i = 0; i < REG_MEM; i++) {
        I lid = i * blockDim.x + threadIdx.x;
        I _gid = glb_offs + sizeof(uint64_t) * lid;
        for (I j = 0; j < sizeof(uint64_t); j++) {
            I gid = _gid + j;
            I lid_off = sizeof(uint64_t) * lid + j;
            I reg_off = sizeof(uint64_t) * i + j;
            bool is_in_block = lid_off < ITEMS_PER_THREAD * BLOCK_SIZE;
            if (gid < size && is_in_block) {
                states[lid_off] = to_state_shr[chars_reg[reg_off]];
            } else if (is_in_block) {
                states[lid_off] = IDENTITY;
            } else if (lid_off == ITEMS_PER_THREAD * BLOCK_SIZE) {
                next_block_first_state = to_state_shr[chars_reg[reg_off]];
            }
        }
    }

    __syncthreads();

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++)
        st[i] = states[threadIdx.x * ITEMS_PER_THREAD + i];

    PrefixOpState state_prefix_op(state_states, state_prefix_storage, ctx, (int)dyn_index, state_t(IDENTITY));
    BlockScanState(state_temp).InclusiveScan(st, st, ctx, state_prefix_op);

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++)
        states[threadIdx.x * ITEMS_PER_THREAD + i] = st[i];

    __syncthreads();

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = threadIdx.x * ITEMS_PER_THREAD + i;
        I gid = glb_offs + lid;
        bool temp = false;
        if (gid < size) {
            if (lid == ITEMS_PER_THREAD * BLOCK_SIZE - 1) {
                temp = gid == size - 1 || is_produce(ctx(st[i], next_block_first_state));
            } else {
                temp = gid == size - 1 || is_produce(states[lid + 1]);
            }
        }
        is_produce_state |= (uint64_t)temp << i;
        prod[i] = (I)temp;
    }

    PrefixOpIdx index_prefix_op(index_states, index_prefix_storage, Add<I>(), (int)dyn_index, I(0));
    BlockScanI(index_temp).InclusiveScan(prod, prod, Add<I>(), index_prefix_op);
    I idx_pfx  = index_prefix_op.GetExclusivePrefix();
    I prod_agg = index_prefix_op.GetBlockAggregate();

    volatile uint16_t* lid_stage = (volatile uint16_t*) states;

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        if ((is_produce_state >> i) & 1) {
            I slot = prod[i] - 1 - idx_pfx;
            I lid  = threadIdx.x * ITEMS_PER_THREAD + i;
            tok_stage[slot] = get_token(st[i]);
            lid_stage[slot] = (uint16_t) lid;
        }
    }

    __syncthreads();

    for (I slot = threadIdx.x; slot < prod_agg; slot += BLOCK_SIZE) {
        I out_idx = idx_pfx + slot;
        d_index_out[out_idx] = glb_offs + lid_stage[slot];
        d_token_out[out_idx] = tok_stage[slot];
    }

    if (dyn_index == num_logical_blocks - 1 && threadIdx.x == blockDim.x - 1) {
        *new_size = Add<I>()(idx_pfx, prod_agg);
        *is_valid = is_accept(st[ITEMS_PER_THREAD - 1]);
    }
}

// Alpacc-style lexer: __ldg() for compose/to_state, CUB BlockScan on a
// blocked register tile for the state scan, and decoupledLookbackPrefix for
// the inter-block part.  Blocked layout means thread t holds the contiguous
// slice [t*IPT, (t+1)*IPT) in registers throughout the scan, eliminating the
// strided shmem reads that the original scanBlock/scanThread use.
template<typename CTX, typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__device__ void
lexerAlpaccImpl(CTX ctx,
      uint8_t* d_in,
      uint32_t* d_index_out,
      token_t* d_token_out,
      ScanTileState<state_t> state_states,
      ScanTileState<I> index_states,
      I size,
      I num_logical_blocks,
      volatile uint32_t* dyn_index_ptr,
      volatile I* new_size,
      volatile bool* is_valid,
      state_t identity) {
    static_assert(ITEMS_PER_THREAD <= 64, "ITEMS_PER_THREAD exceeds 64-bit is_produce_state capacity");
    using BlockScanState = cub::BlockScan<state_t, BLOCK_SIZE>;
    using BlockScanI     = cub::BlockScan<I, BLOCK_SIZE>;
    using PrefixOpState  = TilePrefixCallbackOp<state_t, CTX>;
    using PrefixOpIdx    = TilePrefixCallbackOp<I, Add<I>>;
    __shared__ typename BlockScanState::TempStorage state_temp;
    __shared__ typename BlockScanI::TempStorage     index_temp;
    __shared__ typename PrefixOpState::TempStorage  state_prefix_storage;
    __shared__ typename PrefixOpIdx::TempStorage    index_prefix_storage;
    volatile __shared__ state_t states[ITEMS_PER_THREAD * BLOCK_SIZE];
    volatile __shared__ uint8_t tok_stage[ITEMS_PER_THREAD * BLOCK_SIZE];
    volatile __shared__ state_t to_state_shr[256];
    __shared__ state_t next_block_first_state;

    state_t st[ITEMS_PER_THREAD];
    I prod[ITEMS_PER_THREAD];
    uint64_t is_produce_state = 0;

    uint32_t dyn_index = dynamicIndex<uint32_t>(dyn_index_ptr);
    I glb_offs = dyn_index * BLOCK_SIZE * ITEMS_PER_THREAD;

    copyFromGlbToShr<state_t, I, 1>(0, 256, 256, ctx.d_to_state, to_state_shr);

    if (threadIdx.x == I())
        next_block_first_state = identity;

    __syncthreads();

    loadBytesAsStates<I, BLOCK_SIZE, ITEMS_PER_THREAD, 1>(
        d_in, glb_offs, size, (const state_t*)to_state_shr,
        states, identity, &next_block_first_state);

    __syncthreads();

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++)
        st[i] = states[threadIdx.x * ITEMS_PER_THREAD + i];

    PrefixOpState state_prefix_op(state_states, state_prefix_storage, ctx, (int)dyn_index, state_t(IDENTITY));
    BlockScanState(state_temp).InclusiveScan(st, st, ctx, state_prefix_op);

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++)
        states[threadIdx.x * ITEMS_PER_THREAD + i] = st[i];

    __syncthreads();

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = threadIdx.x * ITEMS_PER_THREAD + i;
        I gid = glb_offs + lid;
        bool temp = false;
        if (gid < size) {
            if (lid == ITEMS_PER_THREAD * BLOCK_SIZE - 1) {
                temp = gid == size - 1 || is_produce(ctx(st[i], next_block_first_state));
            } else {
                temp = gid == size - 1 || is_produce(states[lid + 1]);
            }
        }
        is_produce_state |= (uint64_t)temp << i;
        prod[i] = (I)temp;
    }

    PrefixOpIdx index_prefix_op(index_states, index_prefix_storage, Add<I>(), (int)dyn_index, I(0));
    BlockScanI(index_temp).InclusiveScan(prod, prod, Add<I>(), index_prefix_op);
    I idx_pfx  = index_prefix_op.GetExclusivePrefix();
    I prod_agg = index_prefix_op.GetBlockAggregate();

    volatile uint16_t* lid_stage = (volatile uint16_t*) states;

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        if ((is_produce_state >> i) & 1) {
            I slot = prod[i] - 1 - idx_pfx;
            I lid  = threadIdx.x * ITEMS_PER_THREAD + i;
            tok_stage[slot] = get_token(st[i]);
            lid_stage[slot] = (uint16_t) lid;
        }
    }

    __syncthreads();

    for (I slot = threadIdx.x; slot < prod_agg; slot += BLOCK_SIZE) {
        I out_idx = idx_pfx + slot;
        d_index_out[out_idx] = glb_offs + lid_stage[slot];
        d_token_out[out_idx] = tok_stage[slot];
    }

    if (dyn_index == num_logical_blocks - 1 && threadIdx.x == blockDim.x - 1) {
        *new_size = Add<I>()(idx_pfx, prod_agg);
        *is_valid = is_accept(st[ITEMS_PER_THREAD - 1]);
    }
}

// Loads the compose table into shmem once per block.
// ITEMS_PER_THREAD is 1 less to fit the 288-byte table within 48KB shmem.
template<typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__
void lexerAlpaccShmem(LexerCtxShmem ctx,
      uint8_t* d_in, uint32_t* d_index_out, token_t* d_token_out,
      ScanTileState<state_t> state_states, ScanTileState<I> index_states,
      I size, I num_logical_blocks, volatile uint32_t* dyn_index_ptr,
      volatile I* new_size, volatile bool* is_valid) {
    // compose(288) + states(IPT*BS*2) + tok_stage(IPT*BS) + to_state(512) + sentinel(2)
    // must fit in 48KB static shmem (CUB temp_storage adds ~2KB on top).
    static_assert(288 + ITEMS_PER_THREAD * BLOCK_SIZE * 3 + 514 <= 48 * 1024 - 2048,
                  "Static shmem exceeds 48KB limit for this BLOCK_SIZE/ITEMS_PER_THREAD");
    __shared__ __align__(8) state_t shmem_compose[NUM_STATES * NUM_STATES];
    copyFromGlbToShr<state_t, I, 1>(0, NUM_STATES * NUM_STATES, NUM_STATES * NUM_STATES, ctx.d_compose_glb, shmem_compose);
    ctx.d_compose = shmem_compose;
    lexerAlpaccImpl<LexerCtxShmem, I, BLOCK_SIZE, ITEMS_PER_THREAD>(
        ctx, d_in, d_index_out, d_token_out, state_states, index_states,
        size, num_logical_blocks, dyn_index_ptr, new_size, is_valid, IDENTITY);
}

// Large-shmem variant using dynamic shared memory so >48 KB can be allocated.
// Dynamic shmem layout (in order, all aligned to state_t = uint16_t):
//   [0 .. NUM_STATES*NUM_STATES)               compose table  (288 B)
//   [NUM_STATES*NUM_STATES .. +IPT*BS)          states[]       (IPT*BS * 2 B)
//   [NUM_STATES*NUM_STATES+IPT*BS .. +IPT*BS)   tok_stage[]    (IPT*BS * 1 B, packed into uint16_t slots)
// tok_stage is stored as uint8_t but addressed through uint16_t* (same pointer,
// byte-addressed), so the layout above keeps it naturally aligned.
template<typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__device__ void
lexerAlpaccImplDyn(LexerCtxShmem ctx,
      uint8_t* d_in,
      uint32_t* d_index_out,
      token_t* d_token_out,
      ScanTileState<state_t> state_states,
      ScanTileState<I> index_states,
      I size,
      I num_logical_blocks,
      volatile uint32_t* dyn_index_ptr,
      volatile I* new_size,
      volatile bool* is_valid,
      state_t identity) {
    static_assert(ITEMS_PER_THREAD <= 64, "ITEMS_PER_THREAD exceeds 64-bit is_produce_state capacity");
    using BlockScanState = cub::BlockScan<state_t, BLOCK_SIZE>;
    using BlockScanI     = cub::BlockScan<I, BLOCK_SIZE>;
    using PrefixOpState  = TilePrefixCallbackOp<state_t, LexerCtxShmem>;
    using PrefixOpIdx    = TilePrefixCallbackOp<I, Add<I>>;
    __shared__ typename BlockScanState::TempStorage state_temp;
    __shared__ typename BlockScanI::TempStorage     index_temp;
    __shared__ typename PrefixOpState::TempStorage  state_prefix_storage;
    __shared__ typename PrefixOpIdx::TempStorage    index_prefix_storage;
    __shared__ state_t to_state_shr[256];
    __shared__ state_t next_block_first_state;

    extern __shared__ uint16_t dyn_shmem[];
    volatile state_t* shmem_compose = (volatile state_t*) dyn_shmem;
    volatile state_t* states        = shmem_compose + NUM_STATES * NUM_STATES;
    volatile uint8_t* tok_stage     = (volatile uint8_t*) (states + ITEMS_PER_THREAD * BLOCK_SIZE);
    volatile uint16_t* lid_stage    = (volatile uint16_t*) states;

    state_t st[ITEMS_PER_THREAD];
    I prod[ITEMS_PER_THREAD];
    uint64_t is_produce_state = 0;

    uint32_t dyn_index = dynamicIndex<uint32_t>(dyn_index_ptr);
    I glb_offs = dyn_index * BLOCK_SIZE * ITEMS_PER_THREAD;

    copyFromGlbToShr<state_t, I, 1>(0, NUM_STATES * NUM_STATES, NUM_STATES * NUM_STATES,
                                     ctx.d_compose_glb, shmem_compose);
    ctx.d_compose = (state_t*) shmem_compose;
    copyFromGlbToShr<state_t, I, 1>(0, 256, 256, ctx.d_to_state, to_state_shr);

    if (threadIdx.x == I())
        next_block_first_state = identity;

    __syncthreads();

    loadBytesAsStates<I, BLOCK_SIZE, ITEMS_PER_THREAD, 1>(
        d_in, glb_offs, size, (const state_t*)to_state_shr,
        states, identity, &next_block_first_state);

    __syncthreads();

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++)
        st[i] = states[threadIdx.x * ITEMS_PER_THREAD + i];

    PrefixOpState state_prefix_op(state_states, state_prefix_storage, ctx, (int)dyn_index, state_t(IDENTITY));
    BlockScanState(state_temp).InclusiveScan(st, st, ctx, state_prefix_op);

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++)
        states[threadIdx.x * ITEMS_PER_THREAD + i] = st[i];

    state_t last_state = st[ITEMS_PER_THREAD - 1];

    __syncthreads();

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = threadIdx.x * ITEMS_PER_THREAD + i;
        I gid = glb_offs + lid;
        bool temp = false;
        if (gid < size) {
            if (lid == ITEMS_PER_THREAD * BLOCK_SIZE - 1) {
                temp = gid == size - 1 || is_produce(ctx(states[lid], next_block_first_state));
            } else {
                temp = gid == size - 1 || is_produce(states[lid + 1]);
            }
        }
        is_produce_state |= (uint64_t)temp << i;
        prod[i] = (I)temp;
    }

    PrefixOpIdx index_prefix_op(index_states, index_prefix_storage, Add<I>(), (int)dyn_index, I(0));
    BlockScanI(index_temp).InclusiveScan(prod, prod, Add<I>(), index_prefix_op);
    I idx_pfx  = index_prefix_op.GetExclusivePrefix();
    I prod_agg = index_prefix_op.GetBlockAggregate();

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        if ((is_produce_state >> i) & 1) {
            I slot = prod[i] - 1 - idx_pfx;
            I lid  = threadIdx.x * ITEMS_PER_THREAD + i;
            tok_stage[slot] = get_token(states[lid]);
        }
    }

    __syncthreads();

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        if ((is_produce_state >> i) & 1) {
            I slot = prod[i] - 1 - idx_pfx;
            I lid  = threadIdx.x * ITEMS_PER_THREAD + i;
            lid_stage[slot] = (uint16_t) lid;
        }
    }

    __syncthreads();

    for (I slot = threadIdx.x; slot < prod_agg; slot += BLOCK_SIZE) {
        I out_idx = idx_pfx + slot;
        d_index_out[out_idx] = glb_offs + lid_stage[slot];
        d_token_out[out_idx] = tok_stage[slot];
    }

    if (dyn_index == num_logical_blocks - 1 && threadIdx.x == blockDim.x - 1) {
        *new_size = Add<I>()(idx_pfx, prod_agg);
        *is_valid = is_accept(last_state);
    }
}

// Dynamic-shmem kernel wrapper. The caller must set the 164 KB shmem carveout
// via cudaFuncSetAttribute before launching (see launchLexerAlpaccShmemDyn).
template<typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__ __launch_bounds__(BLOCK_SIZE)
void lexerAlpaccShmemDyn(LexerCtxShmem ctx,
      uint8_t* d_in, uint32_t* d_index_out, token_t* d_token_out,
      ScanTileState<state_t> state_states, ScanTileState<I> index_states,
      I size, I num_logical_blocks, volatile uint32_t* dyn_index_ptr,
      volatile I* new_size, volatile bool* is_valid) {
    lexerAlpaccImplDyn<I, BLOCK_SIZE, ITEMS_PER_THREAD>(
        ctx, d_in, d_index_out, d_token_out, state_states, index_states,
        size, num_logical_blocks, dyn_index_ptr, new_size, is_valid, IDENTITY);
}

// Compute the dynamic shmem size (bytes) for lexerAlpaccShmemDyn<BS, IPT>.
// Layout: compose table + states (uint16_t) + tok_stage (uint8_t packed as uint16_t).
template<typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
static inline size_t dynShmemBytes() {
    // compose + states + tok_stage, all as uint16_t slots (tok_stage needs only
    // 1 byte per slot but we round the region up to uint16_t alignment).
    size_t compose  = NUM_STATES * NUM_STATES * sizeof(state_t);
    size_t states   = (size_t) ITEMS_PER_THREAD * BLOCK_SIZE * sizeof(state_t);
    size_t tok      = (size_t) ITEMS_PER_THREAD * BLOCK_SIZE * sizeof(uint8_t);
    // Round tok up to 2-byte alignment so the overall size is even.
    tok = (tok + 1) & ~(size_t)1;
    return compose + states + tok;
}

// Host helper: set the shmem carveout and launch lexerAlpaccShmemDyn.
template<typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
static void launchLexerAlpaccShmemDyn(
      LexerCtxShmem ctx,
      uint8_t* d_in, uint32_t* d_index_out, token_t* d_token_out,
      ScanTileState<state_t> state_states, ScanTileState<I> index_states,
      I size, I num_logical_blocks, volatile uint32_t* dyn_index_ptr,
      volatile I* new_size, volatile bool* is_valid) {
    auto kernel = lexerAlpaccShmemDyn<I, BLOCK_SIZE, ITEMS_PER_THREAD>;
    size_t shmem_bytes = dynShmemBytes<I, BLOCK_SIZE, ITEMS_PER_THREAD>();
    gpuAssert(cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_bytes));
    kernel<<<num_logical_blocks, BLOCK_SIZE, shmem_bytes>>>(
        ctx, d_in, d_index_out, d_token_out, state_states, index_states,
        size, num_logical_blocks, dyn_index_ptr, new_size, is_valid);
}

// ---- Two-pass variant v2: coalesced state I/O, no next-state buffer ----
//
// Pass 1: map bytes->states (vectorized u64 load), CUB BlockScan + decoupled
// lookback, write exactly `size` prefixed states via copyFromShrToGlb.
//
// Pass 2: read IPT*BS+1 states via copyFromGlbToShr (coalesced), striped
// layout, produce flags from states[lid+1] (the +1 covers block boundaries
// without a separate next-state buffer), direct scatter.
//
// Byte accounting:
//   IN_READ           = size bytes of input
//   IN_STATE_MAP      = 256 * NUM_LOGICAL_BLOCKS * sizeof(state_t)   (to_state shmem load in pass 1)
//   COMPOSE_READ      = NUM_STATES*NUM_STATES * NUM_LOGICAL_BLOCKS * sizeof(state_t) (per-block shmem load in pass 1)
//   STATES_OUT_BYTES  = size * sizeof(state_t)  (pass 1 writes, pass 2 reads)
//   OUT_WRITE         = num_tokens * (sizeof(I) + sizeof(token_t))
//   SCAN_READ_INDEX   = size * sizeof(I)  (index decoupled lookback state traffic, approximated)

#define LEXER_TWO_PASS_V2_P1_BODY \
    using BlockScanState = cub::BlockScan<state_t, BLOCK_SIZE>; \
    using PrefixOpState  = TilePrefixCallbackOp<state_t, LexerCtxShmem>; \
    __shared__ typename BlockScanState::TempStorage temp_storage; \
    __shared__ typename PrefixOpState::TempStorage  prefix_storage; \
    __shared__ state_t to_state_shr[256]; \
    extern __shared__ uint16_t dyn_shmem_p1[]; \
    volatile state_t* shmem_compose = (volatile state_t*) dyn_shmem_p1; \
    volatile state_t* states        = shmem_compose + NUM_STATES * NUM_STATES; \
    state_t st[ITEMS_PER_THREAD]; \
    uint32_t dyn_index = dynamicIndex<uint32_t>(dyn_index_ptr); \
    I glb_offs = dyn_index * BLOCK_SIZE * ITEMS_PER_THREAD; \
    copyFromGlbToShr<state_t, I, 1>(0, NUM_STATES * NUM_STATES, NUM_STATES * NUM_STATES, \
                                     ctx.d_compose_glb, shmem_compose); \
    ctx.d_compose = (state_t*) shmem_compose; \
    copyFromGlbToShr<state_t, I, 1>(0, 256, 256, ctx.d_to_state, to_state_shr); \
    __syncthreads(); \
    loadBytesAsStates<I, BLOCK_SIZE, ITEMS_PER_THREAD>( \
        d_in, glb_offs, size, (const state_t*)to_state_shr, \
        states, identity, nullptr); \
    __syncthreads(); \
    _Pragma("unroll") \
    for (I i = 0; i < ITEMS_PER_THREAD; i++) \
        st[i] = states[threadIdx.x * ITEMS_PER_THREAD + i]; \
    PrefixOpState prefix_op(state_states, prefix_storage, ctx, (int)dyn_index, state_t(IDENTITY)); \
    BlockScanState(temp_storage).InclusiveScan(st, st, ctx, prefix_op); \
    _Pragma("unroll") \
    for (I i = 0; i < ITEMS_PER_THREAD; i++) \
        states[threadIdx.x * ITEMS_PER_THREAD + i] = st[i]; \
    __syncthreads(); \
    copyFromShrToGlb<state_t, I, ITEMS_PER_THREAD>( \
        glb_offs, ITEMS_PER_THREAD * BLOCK_SIZE, size, states, d_states_out); \
    if (dyn_index == num_logical_blocks - 1 && threadIdx.x == BLOCK_SIZE - 1) \
        *is_valid = is_accept(st[ITEMS_PER_THREAD - 1]);


#define LEXER_TWO_PASS_V2_P1_PARAMS \
    LexerCtxShmem ctx, uint8_t* d_in, state_t* d_states_out, \
    ScanTileState<state_t> state_states, \
    I size, I num_logical_blocks, volatile uint32_t* dyn_index_ptr, \
    volatile bool* is_valid, state_t identity

template<typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__
void lexerAlpaccShmemTwoPassV2P1(LEXER_TWO_PASS_V2_P1_PARAMS) {
    LEXER_TWO_PASS_V2_P1_BODY
}

template<typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__
void lexerAlpaccShmemTwoPassV2P1Nreg48(LEXER_TWO_PASS_V2_P1_PARAMS) {
    LEXER_TWO_PASS_V2_P1_BODY
}

template<typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__
void lexerAlpaccShmemTwoPassV2P1NregNone(LEXER_TWO_PASS_V2_P1_PARAMS) {
    LEXER_TWO_PASS_V2_P1_BODY
}

// P1 variant with uint32_t shmem states: same u64 load pattern but stores
// each state as 4 bytes instead of 2, reducing bank conflicts from 8-way to 4-way.
// compose table and to_state table stay uint16_t (state_t); only the scratch
// states[] array is widened.
// P1 U32: states shmem is uint32_t (conflict-free with u64 stride-8 writes).
// Global state buffer is also uint32_t so copyFromShrToGlb<uint32_t> works.
#define LEXER_TWO_PASS_V2_P1_U32_PARAMS \
    LexerCtxShmem ctx, uint8_t* d_in, uint32_t* d_states_out, \
    ScanTileState<state_t> state_states, \
    I size, I num_logical_blocks, volatile uint32_t* dyn_index_ptr, \
    volatile bool* is_valid, state_t identity

#define LEXER_TWO_PASS_V2_P1_U32_BODY \
    using BlockScanState = cub::BlockScan<state_t, BLOCK_SIZE>; \
    using PrefixOpState  = TilePrefixCallbackOp<state_t, LexerCtxShmem>; \
    __shared__ typename BlockScanState::TempStorage temp_storage; \
    __shared__ typename PrefixOpState::TempStorage  prefix_storage; \
    __shared__ state_t to_state_shr[256]; \
    extern __shared__ uint16_t dyn_shmem_p1u32[]; \
    volatile state_t*  shmem_compose = (volatile state_t*) dyn_shmem_p1u32; \
    volatile uint64_t* states_u64    = (volatile uint64_t*)(shmem_compose + NUM_STATES * NUM_STATES); \
    state_t st[ITEMS_PER_THREAD]; \
    uint32_t dyn_index = dynamicIndex<uint32_t>(dyn_index_ptr); \
    I glb_offs = dyn_index * BLOCK_SIZE * ITEMS_PER_THREAD; \
    copyFromGlbToShr<state_t, I, 1>(0, NUM_STATES * NUM_STATES, NUM_STATES * NUM_STATES, \
                                     ctx.d_compose_glb, shmem_compose); \
    ctx.d_compose = (state_t*) shmem_compose; \
    copyFromGlbToShr<state_t, I, 1>(0, 256, 256, ctx.d_to_state, to_state_shr); \
    __syncthreads(); \
    { \
        const I U8    = sizeof(uint64_t); \
        const I TILE  = ITEMS_PER_THREAD * BLOCK_SIZE; \
        const I LOADS = 1 + ITEMS_PER_THREAD / U8; \
        uint64_t regs[LOADS]; \
        uint8_t* bytes = (uint8_t*)regs; \
        _Pragma("unroll") \
        for (I i = 0; i < LOADS; i++) { \
            I base = i * BLOCK_SIZE + threadIdx.x; \
            I gid  = glb_offs + base * U8; \
            if (gid + U8 <= size) \
                regs[i] = *reinterpret_cast<const uint64_t*>(d_in + gid); \
            else { \
                regs[i] = 0; \
                _Pragma("unroll") \
                for (I j = 0; j < U8; j++) \
                    if (gid + j < size) bytes[i * U8 + j] = d_in[gid + j]; \
            } \
        } \
        volatile state_t* states_s = (volatile state_t*) states_u64; \
        _Pragma("unroll") \
        for (I i = 0; i < LOADS; i++) { \
            _Pragma("unroll") \
            for (I j = 0; j < U8; j++) { \
                I lid = (i * BLOCK_SIZE + threadIdx.x) * U8 + j; \
                if (lid < TILE) \
                    states_s[lid] = (glb_offs + lid < size) \
                                    ? to_state_shr[bytes[i * U8 + j]] : identity; \
            } \
        } \
    } \
    __syncthreads(); \
    _Pragma("unroll") \
    for (I i = 0; i < ITEMS_PER_THREAD; i++) \
        st[i] = ((volatile state_t*) states_u64)[threadIdx.x * ITEMS_PER_THREAD + i]; \
    PrefixOpState prefix_op(state_states, prefix_storage, ctx, (int)dyn_index, state_t(IDENTITY)); \
    BlockScanState(temp_storage).InclusiveScan(st, st, ctx, prefix_op); \
    _Pragma("unroll") \
    for (I i = 0; i < ITEMS_PER_THREAD; i++) \
        ((volatile state_t*) states_u64)[threadIdx.x * ITEMS_PER_THREAD + i] = st[i]; \
    __syncthreads(); \
    copyFromShrToGlb<uint32_t, I, ITEMS_PER_THREAD>( \
        glb_offs, ITEMS_PER_THREAD * BLOCK_SIZE, size, (volatile uint32_t*) states_u64, d_states_out); \
    if (dyn_index == num_logical_blocks - 1 && threadIdx.x == BLOCK_SIZE - 1) \
        *is_valid = is_accept(st[ITEMS_PER_THREAD - 1]);

template<typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__
void lexerAlpaccShmemTwoPassV2P1U32(LEXER_TWO_PASS_V2_P1_U32_PARAMS) {
    LEXER_TWO_PASS_V2_P1_U32_BODY
}

// P2 U32: reads uint32_t states from global, casts to state_t for token/produce logic.
#define LEXER_TWO_PASS_V2_P2_U32_PARAMS \
    uint32_t* d_states_in, uint32_t* d_index_out, token_t* d_token_out, \
    ScanTileState<I> index_states, \
    I size, I num_logical_blocks, volatile uint32_t* dyn_index_ptr, \
    volatile I* new_size

#define LEXER_TWO_PASS_V2_P2_U32_BODY \
    static_assert(ITEMS_PER_THREAD <= 64, "ITEMS_PER_THREAD exceeds 64-bit is_produce_state capacity"); \
    using BlockScanI  = cub::BlockScan<I, BLOCK_SIZE>; \
    using PrefixOpIdx = TilePrefixCallbackOp<I, Add<I>>; \
    extern __shared__ uint8_t dyn_shmem_p2u32[]; \
    __shared__ typename BlockScanI::TempStorage  temp_storage; \
    __shared__ typename PrefixOpIdx::TempStorage prefix_storage; \
    volatile uint8_t*  shmem_buf = dyn_shmem_p2u32; \
    volatile uint32_t* states    = (volatile uint32_t*) shmem_buf; \
    volatile uint16_t* lid_stage = (volatile uint16_t*) shmem_buf; \
    volatile uint8_t*  tok_stage = shmem_buf + ITEMS_PER_THREAD * BLOCK_SIZE * sizeof(uint16_t); \
    __shared__ I _prod_shr[ITEMS_PER_THREAD * BLOCK_SIZE]; \
    token_t tokens[ITEMS_PER_THREAD]; \
    I       prod[ITEMS_PER_THREAD]; \
    uint64_t is_produce_state = 0; \
    I idx_pfx  = 0; \
    I prod_agg = 0; \
    uint32_t dyn_index = dynamicIndex<uint32_t>(dyn_index_ptr); \
    I glb_offs = dyn_index * BLOCK_SIZE * ITEMS_PER_THREAD; \
    copyFromGlbToShr<uint32_t, I, ITEMS_PER_THREAD>( \
        glb_offs, ITEMS_PER_THREAD * BLOCK_SIZE + 1, size, d_states_in, states); \
    _Pragma("unroll") \
    for (I i = 0; i < ITEMS_PER_THREAD; i++) { \
        I lid = i * BLOCK_SIZE + threadIdx.x; \
        I gid = glb_offs + lid; \
        bool temp = false; \
        if (gid < size) { \
            tokens[i] = get_token((state_t) states[lid]); \
            temp = gid == size - 1 || is_produce((state_t) states[lid + 1]); \
        } \
        is_produce_state |= (uint64_t)temp << i; \
        prod[i] = (I)temp; \
    } \
    stripedToBlocked<I, I, BLOCK_SIZE, ITEMS_PER_THREAD>(prod, _prod_shr); \
    { \
        PrefixOpIdx prefix_op(index_states, prefix_storage, Add<I>(), (int)dyn_index, I(0)); \
        BlockScanI(temp_storage).InclusiveScan(prod, prod, Add<I>(), prefix_op); \
        idx_pfx  = prefix_op.GetExclusivePrefix(); \
        prod_agg = prefix_op.GetBlockAggregate(); \
    } \
    blockedToStriped<I, I, BLOCK_SIZE, ITEMS_PER_THREAD>(prod, _prod_shr); \
    _Pragma("unroll") \
    for (I i = 0; i < ITEMS_PER_THREAD; i++) { \
        if ((is_produce_state >> i) & 1) { \
            I slot = prod[i] - 1 - idx_pfx; \
            I lid  = i * BLOCK_SIZE + threadIdx.x; \
            tok_stage[slot] = tokens[i]; \
        } \
    } \
    __syncthreads(); \
    _Pragma("unroll") \
    for (I i = 0; i < ITEMS_PER_THREAD; i++) { \
        if ((is_produce_state >> i) & 1) { \
            I slot = prod[i] - 1 - idx_pfx; \
            I lid  = i * BLOCK_SIZE + threadIdx.x; \
            lid_stage[slot] = (uint16_t) lid; \
        } \
    } \
    __syncthreads(); \
    for (I slot = threadIdx.x; slot < prod_agg; slot += BLOCK_SIZE) { \
        I out_idx = idx_pfx + slot; \
        d_index_out[out_idx] = glb_offs + lid_stage[slot]; \
        d_token_out[out_idx] = tok_stage[slot]; \
    } \
    if (dyn_index == num_logical_blocks - 1 && threadIdx.x == blockDim.x - 1) { \
        *new_size = Add<I>()(idx_pfx, prod_agg); \
    }

template<typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__
void lexerAlpaccShmemTwoPassV2P2U32(LEXER_TWO_PASS_V2_P2_U32_PARAMS) {
    LEXER_TWO_PASS_V2_P2_U32_BODY
}

#define LEXER_TWO_PASS_V2_P2_BODY \
    static_assert(ITEMS_PER_THREAD <= 64, "ITEMS_PER_THREAD exceeds 64-bit is_produce_state capacity"); \
    using BlockScanI  = cub::BlockScan<I, BLOCK_SIZE>; \
    using PrefixOpIdx = TilePrefixCallbackOp<I, Add<I>>; \
    extern __shared__ uint8_t dyn_shmem_p2[]; \
    __shared__ typename BlockScanI::TempStorage  temp_storage; \
    __shared__ typename PrefixOpIdx::TempStorage prefix_storage; \
    volatile uint8_t*  shmem_buf = dyn_shmem_p2; \
    volatile state_t*  states    = (volatile state_t*)  shmem_buf; \
    volatile uint16_t* lid_stage = (volatile uint16_t*) shmem_buf; \
    volatile uint8_t*  tok_stage = shmem_buf + ITEMS_PER_THREAD * BLOCK_SIZE * sizeof(uint16_t); \
    __shared__ I _prod_shr[ITEMS_PER_THREAD * BLOCK_SIZE]; \
    token_t tokens[ITEMS_PER_THREAD]; \
    I       prod[ITEMS_PER_THREAD]; \
    uint64_t is_produce_state = 0; \
    I idx_pfx  = 0; \
    I prod_agg = 0; \
    uint32_t dyn_index = dynamicIndex<uint32_t>(dyn_index_ptr); \
    I glb_offs = dyn_index * BLOCK_SIZE * ITEMS_PER_THREAD; \
    copyFromGlbToShr<state_t, I, ITEMS_PER_THREAD>( \
        glb_offs, ITEMS_PER_THREAD * BLOCK_SIZE + 1, size, d_states_in, states); \
    _Pragma("unroll") \
    for (I i = 0; i < ITEMS_PER_THREAD; i++) { \
        I lid = i * BLOCK_SIZE + threadIdx.x; \
        I gid = glb_offs + lid; \
        bool temp = false; \
        if (gid < size) { \
            tokens[i] = get_token(states[lid]); \
            temp = gid == size - 1 || is_produce(states[lid + 1]); \
        } \
        is_produce_state |= (uint64_t)temp << i; \
        prod[i] = (I)temp; \
    } \
    stripedToBlocked<I, I, BLOCK_SIZE, ITEMS_PER_THREAD>(prod, _prod_shr); \
    { \
        PrefixOpIdx prefix_op(index_states, prefix_storage, Add<I>(), (int)dyn_index, I(0)); \
        BlockScanI(temp_storage).InclusiveScan(prod, prod, Add<I>(), prefix_op); \
        idx_pfx  = prefix_op.GetExclusivePrefix(); \
        prod_agg = prefix_op.GetBlockAggregate(); \
    } \
    blockedToStriped<I, I, BLOCK_SIZE, ITEMS_PER_THREAD>(prod, _prod_shr); \
    _Pragma("unroll") \
    for (I i = 0; i < ITEMS_PER_THREAD; i++) { \
        if ((is_produce_state >> i) & 1) { \
            I slot = prod[i] - 1 - idx_pfx; \
            I lid  = i * BLOCK_SIZE + threadIdx.x; \
            tok_stage[slot] = tokens[i]; \
        } \
    } \
    __syncthreads(); \
    _Pragma("unroll") \
    for (I i = 0; i < ITEMS_PER_THREAD; i++) { \
        if ((is_produce_state >> i) & 1) { \
            I slot = prod[i] - 1 - idx_pfx; \
            I lid  = i * BLOCK_SIZE + threadIdx.x; \
            lid_stage[slot] = (uint16_t) lid; \
        } \
    } \
    __syncthreads(); \
    for (I slot = threadIdx.x; slot < prod_agg; slot += BLOCK_SIZE) { \
        I out_idx = idx_pfx + slot; \
        d_index_out[out_idx] = glb_offs + lid_stage[slot]; \
        d_token_out[out_idx] = tok_stage[slot]; \
    } \
    if (dyn_index == num_logical_blocks - 1 && threadIdx.x == blockDim.x - 1) { \
        *new_size = Add<I>()(idx_pfx, prod_agg); \
    }

#define LEXER_TWO_PASS_V2_P2_PARAMS \
    state_t* d_states_in, uint32_t* d_index_out, token_t* d_token_out, \
    ScanTileState<I> index_states, \
    I size, I num_logical_blocks, volatile uint32_t* dyn_index_ptr, \
    volatile I* new_size

// Pass 2: read IPT*BS+1 states coalesced into shmem, compute produce bits in
// blocked layout (thread t owns positions [t*IPT, (t+1)*IPT)), CUB BlockScan,
// decoupled lookback, stage (lid, token) for coalesced sequential output flush.
template<typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__
void lexerAlpaccShmemTwoPassV2P2(LEXER_TWO_PASS_V2_P2_PARAMS) {
    LEXER_TWO_PASS_V2_P2_BODY
}

template<typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__
void lexerAlpaccShmemTwoPassV2P2NregNone(LEXER_TWO_PASS_V2_P2_PARAMS) {
    LEXER_TWO_PASS_V2_P2_BODY
}

// Shmem bytes for pass 1 v2: compose table + states (IPT*BS entries).
template<typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
static inline size_t dynShmemBytesP1V2() {
    size_t compose = NUM_STATES * NUM_STATES * sizeof(state_t);
    size_t states  = (size_t) ITEMS_PER_THREAD * BLOCK_SIZE * sizeof(state_t);
    return compose + states;
}


// Shmem bytes for pass 2 v2: IPT*BS*3+2 bytes (states/lid_stage/tok_stage overlap).
template<typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
static inline size_t dynShmemBytesP2V2() {
    return (size_t) ITEMS_PER_THREAD * BLOCK_SIZE * 3 + 2;
}

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

// NREG1/NREG2: 64=maxnreg(64), 48=maxnreg(48), 0=no limit
template<typename I, I BS1, I IPT1, uint32_t NREG1=64>
static void launchLexerAlpaccShmemTwoPassV2P1(
      LexerCtxShmem ctx,
      uint8_t* d_in, state_t* d_states_glb,
      ScanTileState<state_t> state_states,
      I size, I nlb1,
      volatile uint32_t* dyn_index_ptr1, volatile bool* is_valid) {
    void* kernel;
    if      (NREG1 == 48) kernel = (void*) lexerAlpaccShmemTwoPassV2P1Nreg48<I, BS1, IPT1>;
    else if (NREG1 == 0)  kernel = (void*) lexerAlpaccShmemTwoPassV2P1NregNone<I, BS1, IPT1>;
    else                  kernel = (void*) lexerAlpaccShmemTwoPassV2P1<I, BS1, IPT1>;
    size_t shmem_bytes = dynShmemBytesP1V2<I, BS1, IPT1>();
    gpuAssert(cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_bytes));
    if      (NREG1 == 48) lexerAlpaccShmemTwoPassV2P1Nreg48<I, BS1, IPT1><<<nlb1, BS1, shmem_bytes>>>(
        ctx, d_in, d_states_glb, state_states, size, nlb1, dyn_index_ptr1, is_valid, IDENTITY);
    else if (NREG1 == 0)  lexerAlpaccShmemTwoPassV2P1NregNone<I, BS1, IPT1><<<nlb1, BS1, shmem_bytes>>>(
        ctx, d_in, d_states_glb, state_states, size, nlb1, dyn_index_ptr1, is_valid, IDENTITY);
    else                  lexerAlpaccShmemTwoPassV2P1<I, BS1, IPT1><<<nlb1, BS1, shmem_bytes>>>(
        ctx, d_in, d_states_glb, state_states, size, nlb1, dyn_index_ptr1, is_valid, IDENTITY);
}



template<typename I, I BS2, I IPT2, uint32_t NREG2=64>
static void launchLexerAlpaccShmemTwoPassV2P2(
      state_t* d_states_glb,
      uint32_t* d_index_out, token_t* d_token_out,
      ScanTileState<I> index_states,
      I size, I nlb2,
      volatile uint32_t* dyn_index_ptr2, volatile I* new_size) {
    void* kernel;
    if (NREG2 == 0) kernel = (void*) lexerAlpaccShmemTwoPassV2P2NregNone<I, BS2, IPT2>;
    else            kernel = (void*) lexerAlpaccShmemTwoPassV2P2<I, BS2, IPT2>;
    size_t shmem_bytes = dynShmemBytesP2V2<I, BS2, IPT2>();
    gpuAssert(cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_bytes));
    if (NREG2 == 0) lexerAlpaccShmemTwoPassV2P2NregNone<I, BS2, IPT2><<<nlb2, BS2, shmem_bytes>>>(
        d_states_glb, d_index_out, d_token_out, index_states, size, nlb2, dyn_index_ptr2, new_size);
    else            lexerAlpaccShmemTwoPassV2P2<I, BS2, IPT2><<<nlb2, BS2, shmem_bytes>>>(
        d_states_glb, d_index_out, d_token_out, index_states, size, nlb2, dyn_index_ptr2, new_size);
}

template<typename I, I BS1, I IPT1, I BS2, I IPT2, uint32_t NREG1=64, uint32_t NREG2=64>
static void launchLexerAlpaccShmemTwoPassV2(
      LexerCtxShmem ctx,
      uint8_t* d_in, uint32_t* d_index_out, token_t* d_token_out,
      ScanTileState<state_t> state_states, ScanTileState<I> index_states,
      I size, I nlb1, I nlb2,
      volatile uint32_t* dyn_index_ptr1, volatile uint32_t* dyn_index_ptr2,
      volatile I* new_size, volatile bool* is_valid,
      state_t* d_states_glb) {
    launchLexerAlpaccShmemTwoPassV2P1<I, BS1, IPT1, NREG1>(
        ctx, d_in, d_states_glb, state_states, size, nlb1, dyn_index_ptr1, is_valid);
    gpuAssert(cudaDeviceSynchronize());
    launchLexerAlpaccShmemTwoPassV2P2<I, BS2, IPT2, NREG2>(
        d_states_glb, d_index_out, d_token_out, index_states, size, nlb2, dyn_index_ptr2, new_size);
}

// Shmem bytes for P1 U64 variant: compose table (state_t) + states (uint64_t).
template<typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
static inline size_t dynShmemBytesP1V2U32() {
    size_t compose = NUM_STATES * NUM_STATES * sizeof(state_t);
    size_t states  = (size_t) ITEMS_PER_THREAD * BLOCK_SIZE * sizeof(uint64_t);
    return compose + states;
}

// Shmem bytes for P2 U32 variant: same layout as P2 (states/lid_stage/tok_stage overlap).
template<typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
static inline size_t dynShmemBytesP2V2U32() {
    return (size_t) ITEMS_PER_THREAD * BLOCK_SIZE * sizeof(uint32_t)
         + (size_t) ITEMS_PER_THREAD * BLOCK_SIZE * sizeof(uint8_t) + 2;
}

template<typename I, I BS1, I IPT1>
static void launchLexerAlpaccShmemTwoPassV2P1U32(
      LexerCtxShmem ctx,
      uint8_t* d_in, uint32_t* d_states_glb,
      ScanTileState<state_t> state_states,
      I size, I nlb1,
      volatile uint32_t* dyn_index_ptr1, volatile bool* is_valid) {
    auto kernel = lexerAlpaccShmemTwoPassV2P1U32<I, BS1, IPT1>;
    size_t shmem_bytes = dynShmemBytesP1V2U32<I, BS1, IPT1>();
    gpuAssert(cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_bytes));
    kernel<<<nlb1, BS1, shmem_bytes>>>(
        ctx, d_in, d_states_glb, state_states, size, nlb1, dyn_index_ptr1, is_valid, IDENTITY);
}

template<typename I, I BS2, I IPT2>
static void launchLexerAlpaccShmemTwoPassV2P2U32(
      uint32_t* d_states_glb,
      uint32_t* d_index_out, token_t* d_token_out,
      ScanTileState<I> index_states,
      I size, I nlb2,
      volatile uint32_t* dyn_index_ptr2, volatile I* new_size) {
    auto kernel = lexerAlpaccShmemTwoPassV2P2U32<I, BS2, IPT2>;
    size_t shmem_bytes = dynShmemBytesP2V2U32<I, BS2, IPT2>();
    gpuAssert(cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_bytes));
    kernel<<<nlb2, BS2, shmem_bytes>>>(
        d_states_glb, d_index_out, d_token_out, index_states, size, nlb2, dyn_index_ptr2, new_size);
}

template<typename I, I BS1, I IPT1, I BS2, I IPT2>
static void launchLexerAlpaccShmemTwoPassV2U32(
      LexerCtxShmem ctx,
      uint8_t* d_in, uint32_t* d_index_out, token_t* d_token_out,
      ScanTileState<state_t> state_states, ScanTileState<I> index_states,
      I size, I nlb1, I nlb2,
      volatile uint32_t* dyn_index_ptr1, volatile uint32_t* dyn_index_ptr2,
      volatile I* new_size, volatile bool* is_valid,
      uint32_t* d_states_glb) {
    launchLexerAlpaccShmemTwoPassV2P1U32<I, BS1, IPT1>(
        ctx, d_in, d_states_glb, state_states, size, nlb1, dyn_index_ptr1, is_valid);
    gpuAssert(cudaDeviceSynchronize());
    launchLexerAlpaccShmemTwoPassV2P2U32<I, BS2, IPT2>(
        d_states_glb, d_index_out, d_token_out, index_states, size, nlb2, dyn_index_ptr2, new_size);
}

// Bandwidth ceiling kernel: reads input bytes using same u64 pattern as P1.
// Each thread writes its XOR-reduced result to a unique slot in d_out to
// prevent DCE without any synchronization or contention.
template<typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__ void
bwCeilingRead(const uint8_t* __restrict__ d_in, I size, uint64_t* d_out) {
    const I U8       = sizeof(uint64_t);
    const I REG_MEM  = 1 + ITEMS_PER_THREAD / U8;
    const I glb_offs = (I)(blockIdx.x * BLOCK_SIZE * ITEMS_PER_THREAD);
    uint64_t regs[REG_MEM];
    uint8_t* bytes = (uint8_t*)regs;
    #pragma unroll
    for (I i = 0; i < REG_MEM; i++) {
        I gid = glb_offs + (i * BLOCK_SIZE + threadIdx.x) * U8;
        regs[i] = 0;
        if (gid + U8 <= size)
            regs[i] = __ldg(reinterpret_cast<const uint64_t*>(d_in + gid));
        else {
            #pragma unroll
            for (I j = 0; j < U8; j++)
                if (gid + j < size) bytes[i * U8 + j] = d_in[gid + j];
        }
    }
    uint64_t acc = 0;
    #pragma unroll
    for (I i = 0; i < REG_MEM; i++) acc ^= regs[i];
    d_out[blockIdx.x * BLOCK_SIZE + threadIdx.x] = acc;
}

template<uint32_t BLOCK_SIZE=256, uint32_t ITEMS_PER_THREAD=22>
void testBwCeilingRead(uint8_t* input, size_t input_size) {
    using I = uint32_t;
    const I size = (I)input_size;
    const I NLB  = (size + BLOCK_SIZE * ITEMS_PER_THREAD - 1) / (BLOCK_SIZE * ITEMS_PER_THREAD);
#ifdef PROFILE
    const I WARMUP_RUNS = 1;
    const I RUNS = 1;
#else
    const I WARMUP_RUNS = 500;
    const I RUNS = 100;
#endif
    uint8_t*  d_in;
    uint64_t* d_out;
    gpuAssert(cudaMalloc((void**)&d_in,  size * sizeof(uint8_t)));
    gpuAssert(cudaMalloc((void**)&d_out, NLB * BLOCK_SIZE * sizeof(uint64_t)));
    gpuAssert(cudaMemcpy(d_in, input, size * sizeof(uint8_t), cudaMemcpyHostToDevice));

    float* temp = (float*)malloc(sizeof(float) * RUNS);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (I i = 0; i < WARMUP_RUNS; i++) {
        bwCeilingRead<I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NLB, BLOCK_SIZE>>>(d_in, size, d_out);
        cudaDeviceSynchronize();
    }
    for (I i = 0; i < RUNS; i++) {
        cudaEventRecord(start, 0);
        bwCeilingRead<I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NLB, BLOCK_SIZE>>>(d_in, size, d_out);
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(temp + i, start, stop);
    }
    // Count bytes read (input) + bytes written (one u64 per thread)
    compute_descriptors(temp, RUNS, (uint64_t)size * sizeof(uint8_t)
                                  + (uint64_t)NLB * BLOCK_SIZE * sizeof(uint64_t));
    free(temp);
    gpuAssert(cudaFree(d_in));
    gpuAssert(cudaFree(d_out));
}

void testLexerShmemCompose(uint8_t* input,
               size_t input_size,
               uint32_t* expected_indices,
               token_t* expected_tokens,
               size_t expected_size) {
    using I = uint32_t;
    const I size = input_size;
    const I BLOCK_SIZE = 256;
    const I ITEMS_PER_THREAD = 30; // 1 less than baseline to fit compose table in shmem
    const I NUM_LOGICAL_BLOCKS = (size + BLOCK_SIZE * ITEMS_PER_THREAD - 1) / (BLOCK_SIZE * ITEMS_PER_THREAD);
    const I IN_ARRAY_BYTES = size * sizeof(uint8_t);
    const I INDEX_OUT_ARRAY_BYTES = size * sizeof(I);
    const I TOKEN_OUT_ARRAY_BYTES = size * sizeof(token_t);
#ifdef PROFILE
    const I WARMUP_RUNS = 1;
    const I RUNS = 1;
#else
    const I WARMUP_RUNS = 500;
    const I RUNS = 100;
#endif

    std::vector<token_t> h_token_out(size, 0);
    std::vector<I> h_index_out(size, 0);

    uint32_t* d_dyn_index_ptr;
    I* d_new_size;
    bool* d_is_valid;
    uint8_t *d_in;
    I *d_index_out;
    token_t *d_token_out;
    ScanTileState<state_t> d_state_states;
    ScanTileState<I>       d_index_states;
    gpuAssert(cudaMalloc((void**)&d_dyn_index_ptr, sizeof(uint32_t)));
    gpuAssert(cudaMalloc((void**)&d_new_size, sizeof(I)));
    gpuAssert(cudaMalloc((void**)&d_is_valid, sizeof(bool)));
    cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t));
    cudaMemset(d_is_valid, false, sizeof(bool));
    gpuAssert(cudaMalloc((void**)&d_state_states.d_tile_descriptors,
        ScanTileState<state_t>::AllocationSize(NUM_LOGICAL_BLOCKS)));
    gpuAssert(cudaMalloc((void**)&d_index_states.d_tile_descriptors,
        ScanTileState<I>::AllocationSize(NUM_LOGICAL_BLOCKS)));
    gpuAssert(cudaMalloc((void**)&d_in, IN_ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_index_out, INDEX_OUT_ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_token_out, TOKEN_OUT_ARRAY_BYTES));
    gpuAssert(cudaMemcpy(d_in, input, IN_ARRAY_BYTES, cudaMemcpyHostToDevice));

    LexerCtxShmem ctx = LexerCtxShmem();

    auto reset = [&]() {
        cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t));
        initScanTileState(d_state_states, (int)NUM_LOGICAL_BLOCKS);
        initScanTileState(d_index_states, (int)NUM_LOGICAL_BLOCKS);
    };
    reset();

    float * temp = (float *) malloc(sizeof(float) * RUNS);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (I i = 0; i < WARMUP_RUNS; ++i) {
        lexerShmemCompose<I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
            ctx, d_in, d_index_out, d_token_out,
            d_state_states, d_index_states,
            size, NUM_LOGICAL_BLOCKS, d_dyn_index_ptr, d_new_size, d_is_valid);
        cudaDeviceSynchronize();
        reset();
        gpuAssert(cudaPeekAtLastError());
    }

    for (I i = 0; i < RUNS; ++i) {
        cudaEventRecord(start, 0);
        lexerShmemCompose<I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
            ctx, d_in, d_index_out, d_token_out,
            d_state_states, d_index_states,
            size, NUM_LOGICAL_BLOCKS, d_dyn_index_ptr, d_new_size, d_is_valid);
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(temp + i, start, stop);
        reset();
        gpuAssert(cudaPeekAtLastError());
    }

    I temp_size = 0;
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    const I OUT_WRITE = temp_size * (sizeof(I) + sizeof(token_t));
    const I IN_READ = IN_ARRAY_BYTES;
    const I IN_STATE_MAP = sizeof(state_t) * 256 * NUM_LOGICAL_BLOCKS;
    const I COMPOSE_READ = sizeof(state_t) * NUM_STATES * NUM_STATES * NUM_LOGICAL_BLOCKS;

    reset();
    lexerShmemCompose<I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
        ctx, d_in, d_index_out, d_token_out,
        d_state_states, d_index_states,
        size, NUM_LOGICAL_BLOCKS, d_dyn_index_ptr, d_new_size, d_is_valid);
    cudaDeviceSynchronize();
    gpuAssert(cudaPeekAtLastError());
    bool is_valid = false;
    gpuAssert(cudaMemcpy(h_index_out.data(), d_index_out, INDEX_OUT_ARRAY_BYTES, cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(h_token_out.data(), d_token_out, TOKEN_OUT_ARRAY_BYTES, cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(&is_valid, d_is_valid, sizeof(bool), cudaMemcpyDeviceToHost));

    bool test_passes = is_valid;

    if (!test_passes) {
        std::cout << "Lexer Test Failed: The input given to the lexer does not result in an accepting state." << std::endl;
    }

    test_passes = temp_size == expected_size;
    if (!test_passes) {
        std::cout << "Lexer Test Failed: Expected size=" << expected_size << " but got size=" << temp_size << std::endl;
    } else {
        for (I i = 0; i < expected_size; ++i) {
            if (h_index_out[i] != expected_indices[i]) {
                printf("Lexer Test Failed: index mismatch at i=%u: expected=%u got=%u\n",
                       i, expected_indices[i], h_index_out[i]);
                test_passes = false;
                break;
            }
            if (h_token_out[i] != expected_tokens[i]) {
                printf("Lexer Test Failed: token mismatch at i=%u: expected=%u got=%u\n",
                       i, (unsigned)expected_tokens[i], (unsigned)h_token_out[i]);
                test_passes = false;
                break;
            }
        }
    }

    if (test_passes) {
        compute_descriptors(temp, RUNS, IN_READ + IN_STATE_MAP + COMPOSE_READ + OUT_WRITE);
    }

    free(temp);
    gpuAssert(cudaFree(d_in));
    gpuAssert(cudaFree(d_token_out));
    gpuAssert(cudaFree(d_index_out));
    gpuAssert(cudaFree(d_index_states.d_tile_descriptors));
    gpuAssert(cudaFree(d_state_states.d_tile_descriptors));
    gpuAssert(cudaFree(d_dyn_index_ptr));
    gpuAssert(cudaFree(d_new_size));
    gpuAssert(cudaFree(d_is_valid));

    ctx.Cleanup();
}

void testLexerShmemComposeU64(uint8_t* input,
               size_t input_size,
               uint32_t* expected_indices,
               token_t* expected_tokens,
               size_t expected_size) {
    using I = uint32_t;
    const I size = input_size;
    const I BLOCK_SIZE = 256;
    const I ITEMS_PER_THREAD = 30; // 1 less than baseline to fit compose table in shmem
    const I NUM_LOGICAL_BLOCKS = (size + BLOCK_SIZE * ITEMS_PER_THREAD - 1) / (BLOCK_SIZE * ITEMS_PER_THREAD);
    const I IN_ARRAY_BYTES = size * sizeof(uint8_t);
    const I INDEX_OUT_ARRAY_BYTES = size * sizeof(I);
    const I TOKEN_OUT_ARRAY_BYTES = size * sizeof(token_t);
#ifdef PROFILE
    const I WARMUP_RUNS = 1;
    const I RUNS = 1;
#else
    const I WARMUP_RUNS = 500;
    const I RUNS = 100;
#endif

    std::vector<token_t> h_token_out(size, 0);
    std::vector<I> h_index_out(size, 0);

    uint32_t* d_dyn_index_ptr;
    I* d_new_size;
    bool* d_is_valid;
    uint8_t *d_in;
    I *d_index_out;
    token_t *d_token_out;
    ScanTileState<state_t> d_state_states;
    ScanTileState<I>       d_index_states;
    gpuAssert(cudaMalloc((void**)&d_dyn_index_ptr, sizeof(uint32_t)));
    gpuAssert(cudaMalloc((void**)&d_new_size, sizeof(I)));
    gpuAssert(cudaMalloc((void**)&d_is_valid, sizeof(bool)));
    cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t));
    cudaMemset(d_is_valid, false, sizeof(bool));
    gpuAssert(cudaMalloc((void**)&d_state_states.d_tile_descriptors,
        ScanTileState<state_t>::AllocationSize(NUM_LOGICAL_BLOCKS)));
    gpuAssert(cudaMalloc((void**)&d_index_states.d_tile_descriptors,
        ScanTileState<I>::AllocationSize(NUM_LOGICAL_BLOCKS)));
    gpuAssert(cudaMalloc((void**)&d_in, IN_ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_index_out, INDEX_OUT_ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_token_out, TOKEN_OUT_ARRAY_BYTES));
    gpuAssert(cudaMemcpy(d_in, input, IN_ARRAY_BYTES, cudaMemcpyHostToDevice));

    LexerCtxShmem ctx = LexerCtxShmem();

    auto reset = [&]() {
        cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t));
        initScanTileState(d_state_states, (int)NUM_LOGICAL_BLOCKS);
        initScanTileState(d_index_states, (int)NUM_LOGICAL_BLOCKS);
    };
    reset();

    float * temp = (float *) malloc(sizeof(float) * RUNS);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (I i = 0; i < WARMUP_RUNS; ++i) {
        lexerShmemComposeU64<I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
            ctx, d_in, d_index_out, d_token_out,
            d_state_states, d_index_states,
            size, NUM_LOGICAL_BLOCKS, d_dyn_index_ptr, d_new_size, d_is_valid);
        cudaDeviceSynchronize();
        reset();
        gpuAssert(cudaPeekAtLastError());
    }

    for (I i = 0; i < RUNS; ++i) {
        cudaEventRecord(start, 0);
        lexerShmemComposeU64<I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
            ctx, d_in, d_index_out, d_token_out,
            d_state_states, d_index_states,
            size, NUM_LOGICAL_BLOCKS, d_dyn_index_ptr, d_new_size, d_is_valid);
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(temp + i, start, stop);
        reset();
        gpuAssert(cudaPeekAtLastError());
    }

    I temp_size = 0;
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    const I OUT_WRITE = temp_size * (sizeof(I) + sizeof(token_t));
    const I IN_READ = IN_ARRAY_BYTES;
    const I IN_STATE_MAP = sizeof(state_t) * 256 * NUM_LOGICAL_BLOCKS;
    const I COMPOSE_READ = sizeof(state_t) * NUM_STATES * NUM_STATES * NUM_LOGICAL_BLOCKS;

    reset();
    lexerShmemComposeU64<I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
        ctx, d_in, d_index_out, d_token_out,
        d_state_states, d_index_states,
        size, NUM_LOGICAL_BLOCKS, d_dyn_index_ptr, d_new_size, d_is_valid);
    cudaDeviceSynchronize();
    gpuAssert(cudaPeekAtLastError());
    bool is_valid = false;
    gpuAssert(cudaMemcpy(h_index_out.data(), d_index_out, INDEX_OUT_ARRAY_BYTES, cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(h_token_out.data(), d_token_out, TOKEN_OUT_ARRAY_BYTES, cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(&is_valid, d_is_valid, sizeof(bool), cudaMemcpyDeviceToHost));

    bool test_passes = is_valid;

    if (!test_passes) {
        std::cout << "Lexer Test Failed: The input given to the lexer does not result in an accepting state." << std::endl;
    }

    test_passes = temp_size == expected_size;
    if (!test_passes) {
        std::cout << "Lexer Test Failed: Expected size=" << expected_size << " but got size=" << temp_size << std::endl;
    } else {
        for (I i = 0; i < expected_size; ++i) {
            if (h_index_out[i] != expected_indices[i]) {
                printf("Lexer Test Failed: index mismatch at i=%u: expected=%u got=%u\n",
                       i, expected_indices[i], h_index_out[i]);
                test_passes = false;
                break;
            }
            if (h_token_out[i] != expected_tokens[i]) {
                printf("Lexer Test Failed: token mismatch at i=%u: expected=%u got=%u\n",
                       i, (unsigned)expected_tokens[i], (unsigned)h_token_out[i]);
                test_passes = false;
                break;
            }
        }
    }

    if (test_passes) {
        compute_descriptors(temp, RUNS, IN_READ + IN_STATE_MAP + COMPOSE_READ + OUT_WRITE);
    }

    free(temp);
    gpuAssert(cudaFree(d_in));
    gpuAssert(cudaFree(d_token_out));
    gpuAssert(cudaFree(d_index_out));
    gpuAssert(cudaFree(d_index_states.d_tile_descriptors));
    gpuAssert(cudaFree(d_state_states.d_tile_descriptors));
    gpuAssert(cudaFree(d_dyn_index_ptr));
    gpuAssert(cudaFree(d_new_size));
    gpuAssert(cudaFree(d_is_valid));

    ctx.Cleanup();
}

template<uint32_t ITEMS_PER_THREAD = 30>
void testLexerAlpaccShmem(uint8_t* input,
               size_t input_size,
               uint32_t* expected_indices,
               token_t* expected_tokens,
               size_t expected_size) {
    using I = uint32_t;
    const I size = input_size;
    const I BLOCK_SIZE = 256;
    const I NUM_LOGICAL_BLOCKS = (size + BLOCK_SIZE * ITEMS_PER_THREAD - 1) / (BLOCK_SIZE * ITEMS_PER_THREAD);
    const I IN_ARRAY_BYTES = size * sizeof(uint8_t);
    const I INDEX_OUT_ARRAY_BYTES = size * sizeof(I);
    const I TOKEN_OUT_ARRAY_BYTES = size * sizeof(token_t);
#ifdef PROFILE
    const I WARMUP_RUNS = 1;
    const I RUNS = 1;
#else
    const I WARMUP_RUNS = 500;
    const I RUNS = 100;
#endif

    std::vector<token_t> h_token_out(size, 0);
    std::vector<I> h_index_out(size, 0);

    uint32_t* d_dyn_index_ptr;
    I* d_new_size;
    bool* d_is_valid;
    uint8_t *d_in;
    I *d_index_out;
    token_t *d_token_out;
    ScanTileState<state_t> d_state_states;
    ScanTileState<I>       d_index_states;
    gpuAssert(cudaMalloc((void**)&d_dyn_index_ptr, sizeof(uint32_t)));
    gpuAssert(cudaMalloc((void**)&d_new_size, sizeof(I)));
    gpuAssert(cudaMalloc((void**)&d_is_valid, sizeof(bool)));
    cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t));
    cudaMemset(d_is_valid, false, sizeof(bool));
    gpuAssert(cudaMalloc((void**)&d_state_states.d_tile_descriptors,
        ScanTileState<state_t>::AllocationSize(NUM_LOGICAL_BLOCKS)));
    gpuAssert(cudaMalloc((void**)&d_index_states.d_tile_descriptors,
        ScanTileState<I>::AllocationSize(NUM_LOGICAL_BLOCKS)));
    gpuAssert(cudaMalloc((void**)&d_in, IN_ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_index_out, INDEX_OUT_ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_token_out, TOKEN_OUT_ARRAY_BYTES));
    gpuAssert(cudaMemcpy(d_in, input, IN_ARRAY_BYTES, cudaMemcpyHostToDevice));

    LexerCtxShmem ctx = LexerCtxShmem();

    auto reset = [&]() {
        cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t));
        initScanTileState(d_state_states, (int)NUM_LOGICAL_BLOCKS);
        initScanTileState(d_index_states, (int)NUM_LOGICAL_BLOCKS);
    };
    reset();

    float * temp = (float *) malloc(sizeof(float) * RUNS);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (I i = 0; i < WARMUP_RUNS; ++i) {
        lexerAlpaccShmem<I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
            ctx, d_in, d_index_out, d_token_out, d_state_states, d_index_states,
            size, NUM_LOGICAL_BLOCKS, d_dyn_index_ptr, d_new_size, d_is_valid);
        cudaDeviceSynchronize();
        reset();
        gpuAssert(cudaPeekAtLastError());
    }

    for (I i = 0; i < RUNS; ++i) {
        cudaEventRecord(start, 0);
        lexerAlpaccShmem<I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
            ctx, d_in, d_index_out, d_token_out, d_state_states, d_index_states,
            size, NUM_LOGICAL_BLOCKS, d_dyn_index_ptr, d_new_size, d_is_valid);
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(temp + i, start, stop);
        reset();
        gpuAssert(cudaPeekAtLastError());
    }

    I temp_size = 0;
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    const I OUT_WRITE = temp_size * (sizeof(I) + sizeof(token_t));
    const I IN_READ = IN_ARRAY_BYTES;
    const I IN_STATE_MAP = sizeof(state_t) * 256 * NUM_LOGICAL_BLOCKS;
    const I COMPOSE_READ = sizeof(state_t) * NUM_STATES * NUM_STATES * NUM_LOGICAL_BLOCKS;

    reset();
    lexerAlpaccShmem<I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
        ctx, d_in, d_index_out, d_token_out, d_state_states, d_index_states,
        size, NUM_LOGICAL_BLOCKS, d_dyn_index_ptr, d_new_size, d_is_valid);
    cudaDeviceSynchronize();
    gpuAssert(cudaPeekAtLastError());
    bool is_valid = false;
    gpuAssert(cudaMemcpy(h_index_out.data(), d_index_out, INDEX_OUT_ARRAY_BYTES, cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(h_token_out.data(), d_token_out, TOKEN_OUT_ARRAY_BYTES, cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(&is_valid, d_is_valid, sizeof(bool), cudaMemcpyDeviceToHost));

    bool test_passes = is_valid;
    if (!test_passes) {
        std::cout << "Lexer Test Failed: The input given to the lexer does not result in an accepting state." << std::endl;
    }
    test_passes = temp_size == expected_size;
    if (!test_passes) {
        std::cout << "Lexer Test Failed: Expected size=" << expected_size << " but got size=" << temp_size << std::endl;
    } else {
        for (I i = 0; i < expected_size; ++i) {
            if (h_index_out[i] != expected_indices[i]) {
                printf("Lexer Test Failed: index mismatch at i=%u: expected=%u got=%u\n",
                       i, expected_indices[i], h_index_out[i]);
                test_passes = false;
                break;
            }
            if (h_token_out[i] != expected_tokens[i]) {
                printf("Lexer Test Failed: token mismatch at i=%u: expected=%u got=%u\n",
                       i, (unsigned)expected_tokens[i], (unsigned)h_token_out[i]);
                test_passes = false;
                break;
            }
        }
    }

    if (test_passes) {
        compute_descriptors(temp, RUNS, IN_READ + IN_STATE_MAP + COMPOSE_READ + OUT_WRITE);
    }

    free(temp);
    gpuAssert(cudaFree(d_in));
    gpuAssert(cudaFree(d_token_out));
    gpuAssert(cudaFree(d_index_out));
    gpuAssert(cudaFree(d_index_states.d_tile_descriptors));
    gpuAssert(cudaFree(d_state_states.d_tile_descriptors));
    gpuAssert(cudaFree(d_dyn_index_ptr));
    gpuAssert(cudaFree(d_new_size));
    gpuAssert(cudaFree(d_is_valid));

    ctx.Cleanup();
}

template<uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD>
void testLexerAlpaccShmemDyn(uint8_t* input,
               size_t input_size,
               uint32_t* expected_indices,
               token_t* expected_tokens,
               size_t expected_size) {
    using I = uint32_t;
    const I size = input_size;
    const I NUM_LOGICAL_BLOCKS = (size + BLOCK_SIZE * ITEMS_PER_THREAD - 1) / (BLOCK_SIZE * ITEMS_PER_THREAD);
    const I IN_ARRAY_BYTES = size * sizeof(uint8_t);
    const I INDEX_OUT_ARRAY_BYTES = size * sizeof(I);
    const I TOKEN_OUT_ARRAY_BYTES = size * sizeof(token_t);
#ifdef PROFILE
    const I WARMUP_RUNS = 1;
    const I RUNS = 1;
#else
    const I WARMUP_RUNS = 500;
    const I RUNS = 100;
#endif

    std::vector<token_t> h_token_out(size, 0);
    std::vector<I> h_index_out(size, 0);

    uint32_t* d_dyn_index_ptr;
    I* d_new_size;
    bool* d_is_valid;
    uint8_t *d_in;
    I *d_index_out;
    token_t *d_token_out;
    ScanTileState<state_t> d_state_states;
    ScanTileState<I>       d_index_states;
    gpuAssert(cudaMalloc((void**)&d_dyn_index_ptr, sizeof(uint32_t)));
    gpuAssert(cudaMalloc((void**)&d_new_size, sizeof(I)));
    gpuAssert(cudaMalloc((void**)&d_is_valid, sizeof(bool)));
    cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t));
    cudaMemset(d_is_valid, false, sizeof(bool));
    gpuAssert(cudaMalloc((void**)&d_state_states.d_tile_descriptors,
        ScanTileState<state_t>::AllocationSize(NUM_LOGICAL_BLOCKS)));
    gpuAssert(cudaMalloc((void**)&d_index_states.d_tile_descriptors,
        ScanTileState<I>::AllocationSize(NUM_LOGICAL_BLOCKS)));
    gpuAssert(cudaMalloc((void**)&d_in, IN_ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_index_out, INDEX_OUT_ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_token_out, TOKEN_OUT_ARRAY_BYTES));
    gpuAssert(cudaMemcpy(d_in, input, IN_ARRAY_BYTES, cudaMemcpyHostToDevice));

    LexerCtxShmem ctx = LexerCtxShmem();

    auto reset = [&]() {
        cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t));
        initScanTileState(d_state_states, (int)NUM_LOGICAL_BLOCKS);
        initScanTileState(d_index_states, (int)NUM_LOGICAL_BLOCKS);
    };
    reset();

    float * temp = (float *) malloc(sizeof(float) * RUNS);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (I i = 0; i < WARMUP_RUNS; ++i) {
        launchLexerAlpaccShmemDyn<I, BLOCK_SIZE, ITEMS_PER_THREAD>(
            ctx, d_in, d_index_out, d_token_out, d_state_states, d_index_states,
            size, NUM_LOGICAL_BLOCKS, d_dyn_index_ptr, d_new_size, d_is_valid);
        cudaDeviceSynchronize();
        reset();
        gpuAssert(cudaPeekAtLastError());
    }

    for (I i = 0; i < RUNS; ++i) {
        cudaEventRecord(start, 0);
        launchLexerAlpaccShmemDyn<I, BLOCK_SIZE, ITEMS_PER_THREAD>(
            ctx, d_in, d_index_out, d_token_out, d_state_states, d_index_states,
            size, NUM_LOGICAL_BLOCKS, d_dyn_index_ptr, d_new_size, d_is_valid);
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(temp + i, start, stop);
        reset();
        gpuAssert(cudaPeekAtLastError());
    }

    I temp_size = 0;
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    const I OUT_WRITE = temp_size * (sizeof(I) + sizeof(token_t));
    const I IN_READ = IN_ARRAY_BYTES;
    const I IN_STATE_MAP = sizeof(state_t) * 256 * NUM_LOGICAL_BLOCKS;
    const I COMPOSE_READ = sizeof(state_t) * NUM_STATES * NUM_STATES * NUM_LOGICAL_BLOCKS;

    reset();
    launchLexerAlpaccShmemDyn<I, BLOCK_SIZE, ITEMS_PER_THREAD>(
        ctx, d_in, d_index_out, d_token_out, d_state_states, d_index_states,
        size, NUM_LOGICAL_BLOCKS, d_dyn_index_ptr, d_new_size, d_is_valid);
    cudaDeviceSynchronize();
    gpuAssert(cudaPeekAtLastError());
    bool is_valid = false;
    gpuAssert(cudaMemcpy(h_index_out.data(), d_index_out, INDEX_OUT_ARRAY_BYTES, cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(h_token_out.data(), d_token_out, TOKEN_OUT_ARRAY_BYTES, cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(&is_valid, d_is_valid, sizeof(bool), cudaMemcpyDeviceToHost));

    bool test_passes = is_valid;
    if (!test_passes) {
        std::cout << "Lexer Test Failed: The input given to the lexer does not result in an accepting state." << std::endl;
    }
    test_passes = temp_size == expected_size;
    if (!test_passes) {
        std::cout << "Lexer Test Failed: Expected size=" << expected_size << " but got size=" << temp_size << std::endl;
    } else {
        for (I i = 0; i < expected_size; ++i) {
            if (h_index_out[i] != expected_indices[i]) {
                printf("Lexer Test Failed: index mismatch at i=%u: expected=%u got=%u\n",
                       i, expected_indices[i], h_index_out[i]);
                test_passes = false;
                break;
            }
            if (h_token_out[i] != expected_tokens[i]) {
                printf("Lexer Test Failed: token mismatch at i=%u: expected=%u got=%u\n",
                       i, (unsigned)expected_tokens[i], (unsigned)h_token_out[i]);
                test_passes = false;
                break;
            }
        }
    }

    if (test_passes) {
        compute_descriptors(temp, RUNS, IN_READ + IN_STATE_MAP + COMPOSE_READ + OUT_WRITE);
    }

    free(temp);
    gpuAssert(cudaFree(d_in));
    gpuAssert(cudaFree(d_token_out));
    gpuAssert(cudaFree(d_index_out));
    gpuAssert(cudaFree(d_index_states.d_tile_descriptors));
    gpuAssert(cudaFree(d_state_states.d_tile_descriptors));
    gpuAssert(cudaFree(d_dyn_index_ptr));
    gpuAssert(cudaFree(d_new_size));
    gpuAssert(cudaFree(d_is_valid));

    ctx.Cleanup();
}


template<uint32_t BS1, uint32_t IPT1, uint32_t BS2, uint32_t IPT2>
void testLexerAlpaccShmemTwoPassV2U32(uint8_t* input,
               size_t input_size,
               uint32_t* expected_indices,
               token_t* expected_tokens,
               size_t expected_size) {
    using I = uint32_t;
    const I size = input_size;
    const I NLB1 = (size + BS1 * IPT1 - 1) / (BS1 * IPT1);
    const I NLB2 = (size + BS2 * IPT2 - 1) / (BS2 * IPT2);
    const I IN_ARRAY_BYTES = size * sizeof(uint8_t);
    const I INDEX_OUT_ARRAY_BYTES = size * sizeof(I);
    const I TOKEN_OUT_ARRAY_BYTES = size * sizeof(token_t);
    const I STATES_GLB_BYTES = size * sizeof(uint32_t);
#ifdef PROFILE
    const I WARMUP_RUNS = 1;
    const I RUNS = 1;
#else
    const I WARMUP_RUNS = 500;
    const I RUNS = 100;
#endif
    std::vector<token_t> h_token_out(size, 0);
    std::vector<I> h_index_out(size, 0);
    uint32_t* d_dyn_index_ptr1; uint32_t* d_dyn_index_ptr2;
    I* d_new_size; bool* d_is_valid;
    uint8_t* d_in; I* d_index_out; token_t* d_token_out;
    ScanTileState<state_t> d_state_states;
    ScanTileState<I>       d_index_states;
    uint32_t* d_states_glb;
    gpuAssert(cudaMalloc((void**)&d_dyn_index_ptr1, sizeof(uint32_t)));
    gpuAssert(cudaMalloc((void**)&d_dyn_index_ptr2, sizeof(uint32_t)));
    gpuAssert(cudaMalloc((void**)&d_new_size, sizeof(I)));
    gpuAssert(cudaMalloc((void**)&d_is_valid, sizeof(bool)));
    cudaMemset(d_dyn_index_ptr1, 0, sizeof(uint32_t));
    cudaMemset(d_dyn_index_ptr2, 0, sizeof(uint32_t));
    cudaMemset(d_is_valid, false, sizeof(bool));
    gpuAssert(cudaMalloc((void**)&d_state_states.d_tile_descriptors,
        ScanTileState<state_t>::AllocationSize(NLB1)));
    gpuAssert(cudaMalloc((void**)&d_index_states.d_tile_descriptors,
        ScanTileState<I>::AllocationSize(NLB2)));
    gpuAssert(cudaMalloc((void**)&d_in, IN_ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_index_out, INDEX_OUT_ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_token_out, TOKEN_OUT_ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_states_glb, STATES_GLB_BYTES));
    gpuAssert(cudaMemcpy(d_in, input, IN_ARRAY_BYTES, cudaMemcpyHostToDevice));
    LexerCtxShmem ctx = LexerCtxShmem();
    float* temp_total = (float*) malloc(sizeof(float) * RUNS);
    float* temp_p1    = (float*) malloc(sizeof(float) * RUNS);
    float* temp_p2    = (float*) malloc(sizeof(float) * RUNS);
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    auto reset = [&]() {
        cudaMemset(d_dyn_index_ptr1, 0, sizeof(uint32_t));
        cudaMemset(d_dyn_index_ptr2, 0, sizeof(uint32_t));
        initScanTileState(d_state_states, (int)NLB1);
        initScanTileState(d_index_states, (int)NLB2);
    };
    reset();
    for (I i = 0; i < WARMUP_RUNS; ++i) {
        launchLexerAlpaccShmemTwoPassV2U32<I, BS1, IPT1, BS2, IPT2>(
            ctx, d_in, d_index_out, d_token_out, d_state_states, d_index_states,
            size, NLB1, NLB2, d_dyn_index_ptr1, d_dyn_index_ptr2,
            d_new_size, d_is_valid, d_states_glb);
        cudaDeviceSynchronize(); reset();
        gpuAssert(cudaPeekAtLastError());
    }
    for (I i = 0; i < RUNS; ++i) {
        cudaEventRecord(start, 0);
        launchLexerAlpaccShmemTwoPassV2P1U32<I, BS1, IPT1>(
            ctx, d_in, d_states_glb, d_state_states,
            size, NLB1, d_dyn_index_ptr1, d_is_valid);
        gpuAssert(cudaDeviceSynchronize());
        cudaEventRecord(stop, 0); cudaEventSynchronize(stop);
        cudaEventElapsedTime(temp_p1 + i, start, stop);
        cudaEventRecord(start, 0);
        launchLexerAlpaccShmemTwoPassV2P2U32<I, BS2, IPT2>(
            d_states_glb, d_index_out, d_token_out, d_index_states,
            size, NLB2, d_dyn_index_ptr2, d_new_size);
        gpuAssert(cudaDeviceSynchronize());
        cudaEventRecord(stop, 0); cudaEventSynchronize(stop);
        cudaEventElapsedTime(temp_p2 + i, start, stop);
        temp_total[i] = temp_p1[i] + temp_p2[i];
        reset(); gpuAssert(cudaPeekAtLastError());
    }
    I temp_size = 0;
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    const I OUT_WRITE        = temp_size * (sizeof(I) + sizeof(token_t));
    const I IN_READ          = IN_ARRAY_BYTES;
    const I STATES_GLB_WRITE = STATES_GLB_BYTES;
    const I STATES_GLB_READ  = STATES_GLB_BYTES;
    const I P1_BYTES = IN_READ + STATES_GLB_WRITE;
    const I P2_BYTES = STATES_GLB_READ + OUT_WRITE;
    reset();
    launchLexerAlpaccShmemTwoPassV2U32<I, BS1, IPT1, BS2, IPT2>(
        ctx, d_in, d_index_out, d_token_out, d_state_states, d_index_states,
        size, NLB1, NLB2, d_dyn_index_ptr1, d_dyn_index_ptr2,
        d_new_size, d_is_valid, d_states_glb);
    cudaDeviceSynchronize(); gpuAssert(cudaPeekAtLastError());
    bool is_valid = false;
    gpuAssert(cudaMemcpy(h_index_out.data(), d_index_out, INDEX_OUT_ARRAY_BYTES, cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(h_token_out.data(), d_token_out, TOKEN_OUT_ARRAY_BYTES, cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(&is_valid, d_is_valid, sizeof(bool), cudaMemcpyDeviceToHost));
    bool test_passes = is_valid;
    if (!test_passes)
        std::cout << "Lexer Test Failed: The input given to the lexer does not result in an accepting state." << std::endl;
    test_passes = temp_size == expected_size;
    if (!test_passes) {
        std::cout << "Lexer Test Failed: Expected size=" << expected_size << " but got size=" << temp_size << std::endl;
    } else {
        for (I i = 0; i < expected_size; ++i) {
            if (h_index_out[i] != expected_indices[i]) {
                printf("Lexer Test Failed: index mismatch at i=%u: expected=%u got=%u\n",
                       i, expected_indices[i], h_index_out[i]);
                test_passes = false; break;
            }
            if (h_token_out[i] != expected_tokens[i]) {
                printf("Lexer Test Failed: token mismatch at i=%u: expected=%u got=%u\n",
                       i, (unsigned)expected_tokens[i], (unsigned)h_token_out[i]);
                test_passes = false; break;
            }
        }
    }
    if (test_passes) {
        printf("\n");
        printf("  %-36s ", "P1:");
        compute_descriptors(temp_p1, RUNS, P1_BYTES);
        printf("  %-36s ", "P2:");
        compute_descriptors(temp_p2, RUNS, P2_BYTES);
        printf("  %-36s ", "Total:");
        compute_descriptors(temp_total, RUNS, P1_BYTES + P2_BYTES);
    }
    free(temp_total); free(temp_p1); free(temp_p2);
    gpuAssert(cudaFree(d_in)); gpuAssert(cudaFree(d_token_out));
    gpuAssert(cudaFree(d_index_out));
    gpuAssert(cudaFree(d_index_states.d_tile_descriptors));
    gpuAssert(cudaFree(d_state_states.d_tile_descriptors));
    gpuAssert(cudaFree(d_dyn_index_ptr1)); gpuAssert(cudaFree(d_dyn_index_ptr2));
    gpuAssert(cudaFree(d_new_size)); gpuAssert(cudaFree(d_is_valid));
    gpuAssert(cudaFree(d_states_glb));
    ctx.Cleanup();
}

template<uint32_t BS1, uint32_t IPT1, uint32_t BS2, uint32_t IPT2,
         uint32_t NREG1=64, uint32_t NREG2=64>
void testLexerAlpaccShmemTwoPassV2(uint8_t* input,
               size_t input_size,
               uint32_t* expected_indices,
               token_t* expected_tokens,
               size_t expected_size) {
    using I = uint32_t;
    const I size = input_size;
    const I NLB1 = (size + BS1 * IPT1 - 1) / (BS1 * IPT1);
    const I NLB2 = (size + BS2 * IPT2 - 1) / (BS2 * IPT2);
    const I IN_ARRAY_BYTES = size * sizeof(uint8_t);
    const I INDEX_OUT_ARRAY_BYTES = size * sizeof(I);
    const I TOKEN_OUT_ARRAY_BYTES = size * sizeof(token_t);
    // Pass 1 writes exactly `size` states; pass 2 reads `size+1` (the +1 is
    // clamped to `size-1` by copyFromGlbToShr's bounds check).
    const I STATES_GLB_BYTES = size * sizeof(state_t);
#ifdef PROFILE
    const I WARMUP_RUNS = 1;
    const I RUNS = 1;
#else
    const I WARMUP_RUNS = 500;
    const I RUNS = 100;
#endif

    std::vector<token_t> h_token_out(size, 0);
    std::vector<I> h_index_out(size, 0);

    uint32_t* d_dyn_index_ptr1;
    uint32_t* d_dyn_index_ptr2;
    I* d_new_size;
    bool* d_is_valid;
    uint8_t *d_in;
    I *d_index_out;
    token_t *d_token_out;
    ScanTileState<state_t> d_state_states;
    ScanTileState<I>       d_index_states;
    state_t* d_states_glb;
    gpuAssert(cudaMalloc((void**)&d_dyn_index_ptr1, sizeof(uint32_t)));
    gpuAssert(cudaMalloc((void**)&d_dyn_index_ptr2, sizeof(uint32_t)));
    gpuAssert(cudaMalloc((void**)&d_new_size, sizeof(I)));
    gpuAssert(cudaMalloc((void**)&d_is_valid, sizeof(bool)));
    cudaMemset(d_dyn_index_ptr1, 0, sizeof(uint32_t));
    cudaMemset(d_dyn_index_ptr2, 0, sizeof(uint32_t));
    cudaMemset(d_is_valid, false, sizeof(bool));
    gpuAssert(cudaMalloc((void**)&d_state_states.d_tile_descriptors,
        ScanTileState<state_t>::AllocationSize(NLB1)));
    gpuAssert(cudaMalloc((void**)&d_index_states.d_tile_descriptors,
        ScanTileState<I>::AllocationSize(NLB2)));
    gpuAssert(cudaMalloc((void**)&d_in, IN_ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_index_out, INDEX_OUT_ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_token_out, TOKEN_OUT_ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_states_glb, STATES_GLB_BYTES));
    gpuAssert(cudaMemcpy(d_in, input, IN_ARRAY_BYTES, cudaMemcpyHostToDevice));

    LexerCtxShmem ctx = LexerCtxShmem();

    float* temp_total = (float*) malloc(sizeof(float) * RUNS);
    float* temp_p1    = (float*) malloc(sizeof(float) * RUNS);
    float* temp_p2    = (float*) malloc(sizeof(float) * RUNS);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto reset = [&]() {
        cudaMemset(d_dyn_index_ptr1, 0, sizeof(uint32_t));
        cudaMemset(d_dyn_index_ptr2, 0, sizeof(uint32_t));
        initScanTileState(d_state_states, (int)NLB1);
        initScanTileState(d_index_states, (int)NLB2);
    };
    reset();

    for (I i = 0; i < WARMUP_RUNS; ++i) {
        launchLexerAlpaccShmemTwoPassV2<I, BS1, IPT1, BS2, IPT2, NREG1, NREG2>(
            ctx, d_in, d_index_out, d_token_out, d_state_states, d_index_states,
            size, NLB1, NLB2, d_dyn_index_ptr1, d_dyn_index_ptr2,
            d_new_size, d_is_valid, d_states_glb);
        cudaDeviceSynchronize();
        reset();
        gpuAssert(cudaPeekAtLastError());
    }

    for (I i = 0; i < RUNS; ++i) {
        // Time P1
        cudaEventRecord(start, 0);
        launchLexerAlpaccShmemTwoPassV2P1<I, BS1, IPT1, NREG1>(
            ctx, d_in, d_states_glb, d_state_states,
            size, NLB1, d_dyn_index_ptr1, d_is_valid);
        gpuAssert(cudaDeviceSynchronize());
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(temp_p1 + i, start, stop);

        // Time P2
        cudaEventRecord(start, 0);
        launchLexerAlpaccShmemTwoPassV2P2<I, BS2, IPT2, NREG2>(
            d_states_glb, d_index_out, d_token_out, d_index_states,
            size, NLB2, d_dyn_index_ptr2, d_new_size);
        gpuAssert(cudaDeviceSynchronize());
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(temp_p2 + i, start, stop);

        temp_total[i] = temp_p1[i] + temp_p2[i];
        reset();
        gpuAssert(cudaPeekAtLastError());
    }

    I temp_size = 0;
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    const I OUT_WRITE        = temp_size * (sizeof(I) + sizeof(token_t));
    const I IN_READ          = IN_ARRAY_BYTES;      // P1: 1B per input element
    const I STATES_GLB_WRITE = STATES_GLB_BYTES;    // P1: 2B per input element written
    const I STATES_GLB_READ  = STATES_GLB_BYTES;    // P2: 2B per input element read
    // to_state (512B) and compose table (2KB) are tiny and hit L2 after first block
    const I P1_BYTES = IN_READ + STATES_GLB_WRITE;
    const I P2_BYTES = STATES_GLB_READ + OUT_WRITE;

    reset();
    launchLexerAlpaccShmemTwoPassV2<I, BS1, IPT1, BS2, IPT2, NREG1, NREG2>(
        ctx, d_in, d_index_out, d_token_out, d_state_states, d_index_states,
        size, NLB1, NLB2, d_dyn_index_ptr1, d_dyn_index_ptr2,
        d_new_size, d_is_valid, d_states_glb);
    cudaDeviceSynchronize();
    gpuAssert(cudaPeekAtLastError());
    bool is_valid = false;
    gpuAssert(cudaMemcpy(h_index_out.data(), d_index_out, INDEX_OUT_ARRAY_BYTES, cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(h_token_out.data(), d_token_out, TOKEN_OUT_ARRAY_BYTES, cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(&is_valid, d_is_valid, sizeof(bool), cudaMemcpyDeviceToHost));

    bool test_passes = is_valid;
    if (!test_passes) {
        std::cout << "Lexer Test Failed: The input given to the lexer does not result in an accepting state." << std::endl;
    }
    test_passes = temp_size == expected_size;
    if (!test_passes) {
        std::cout << "Lexer Test Failed: Expected size=" << expected_size << " but got size=" << temp_size << std::endl;
    } else {
        for (I i = 0; i < expected_size; ++i) {
            if (h_index_out[i] != expected_indices[i]) {
                printf("Lexer Test Failed: index mismatch at i=%u: expected=%u got=%u\n",
                       i, expected_indices[i], h_index_out[i]);
                test_passes = false;
                break;
            }
            if (h_token_out[i] != expected_tokens[i]) {
                printf("Lexer Test Failed: token mismatch at i=%u: expected=%u got=%u\n",
                       i, (unsigned)expected_tokens[i], (unsigned)h_token_out[i]);
                test_passes = false;
                break;
            }
        }
    }

    if (test_passes) {
        printf("\n");
        printf("  %-36s ", "P1:");
        compute_descriptors(temp_p1, RUNS, P1_BYTES);
        printf("  %-36s ", "P2:");
        compute_descriptors(temp_p2, RUNS, P2_BYTES);
        printf("  %-36s ", "Total:");
        compute_descriptors(temp_total, RUNS, P1_BYTES + P2_BYTES);
    }

    free(temp_total);
    free(temp_p1);
    free(temp_p2);
    gpuAssert(cudaFree(d_in));
    gpuAssert(cudaFree(d_token_out));
    gpuAssert(cudaFree(d_index_out));
    gpuAssert(cudaFree(d_index_states.d_tile_descriptors));
    gpuAssert(cudaFree(d_state_states.d_tile_descriptors));
    gpuAssert(cudaFree(d_dyn_index_ptr1));
    gpuAssert(cudaFree(d_dyn_index_ptr2));
    gpuAssert(cudaFree(d_new_size));
    gpuAssert(cudaFree(d_is_valid));
    gpuAssert(cudaFree(d_states_glb));

    ctx.Cleanup();
}

#ifdef DEBUG
#define DEBUG_PRINT(...) fprintf(stderr, __VA_ARGS__)
#else
#define DEBUG_PRINT(...) ((void)0)
#endif

// Token values based on the state encoding:
// token = (state & TOKEN_MASK) >> TOKEN_OFFSET
#define TOKEN_SPACE  0
#define TOKEN_IDENT  1
#define TOKEN_LPAREN 2
#define TOKEN_RPAREN 3
#define TOKEN_DEAD   4

const char* token_name(uint8_t t) {
    switch (t) {
        case TOKEN_SPACE:  return "SPACE";
        case TOKEN_IDENT:  return "IDENT";
        case TOKEN_LPAREN: return "LPAREN";
        case TOKEN_RPAREN: return "RPAREN";
        case TOKEN_DEAD:   return "DEAD";
        default:           return "UNKNOWN";
    }
}

typedef struct {
    const char*    input;
    uint32_t*      expected_indices;
    uint8_t*       expected_tokens;
    size_t         expected_size;
    const char*    name;
} LexerTest;

bool runTest(LexerTest* test) {
    size_t input_size = strlen(test->input);
    uint8_t* input = (uint8_t*) test->input;

    DEBUG_PRINT("\n=== Test: %s ===\n", test->name);
    DEBUG_PRINT("Input: \"%s\" (len=%zu)\n", test->input, input_size);
    DEBUG_PRINT("Expected %zu tokens:\n", test->expected_size);
    for (size_t i = 0; i < test->expected_size; i++) {
        DEBUG_PRINT("  [%zu] index=%u token=%s(%u)\n",
                    i, test->expected_indices[i],
                    token_name(test->expected_tokens[i]),
                    test->expected_tokens[i]);
    }

    using I = uint32_t;
    const I size = (I) input_size;
    const I BLOCK_SIZE = 256;
    const I ITEMS_PER_THREAD = 31;
    const I NUM_LOGICAL_BLOCKS = (size + BLOCK_SIZE * ITEMS_PER_THREAD - 1)
                                 / (BLOCK_SIZE * ITEMS_PER_THREAD);

    uint8_t*         d_in;
    I*               d_index_out;
    token_t*         d_token_out;
    ScanTileState<state_t> d_state_states;
    ScanTileState<I>       d_index_states;
    uint32_t*        d_dyn_index_ptr;
    I*               d_new_size;
    bool*            d_is_valid;

    gpuAssert(cudaMalloc((void**)&d_in,            size * sizeof(uint8_t)));
    gpuAssert(cudaMalloc((void**)&d_index_out,     size * sizeof(I)));
    gpuAssert(cudaMalloc((void**)&d_token_out,     size * sizeof(token_t)));
    gpuAssert(cudaMalloc((void**)&d_state_states.d_tile_descriptors,
        ScanTileState<state_t>::AllocationSize(NUM_LOGICAL_BLOCKS)));
    gpuAssert(cudaMalloc((void**)&d_index_states.d_tile_descriptors,
        ScanTileState<I>::AllocationSize(NUM_LOGICAL_BLOCKS)));
    gpuAssert(cudaMalloc((void**)&d_dyn_index_ptr, sizeof(uint32_t)));
    gpuAssert(cudaMalloc((void**)&d_new_size,      sizeof(I)));
    gpuAssert(cudaMalloc((void**)&d_is_valid,      sizeof(bool)));

    gpuAssert(cudaMemcpy(d_in, input, size * sizeof(uint8_t), cudaMemcpyHostToDevice));
    gpuAssert(cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t)));
    gpuAssert(cudaMemset(d_is_valid, 0, sizeof(bool)));
    initScanTileState(d_state_states, (int)NUM_LOGICAL_BLOCKS);
    initScanTileState(d_index_states, (int)NUM_LOGICAL_BLOCKS);

    LexerCtx ctx;

    lexer<I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
        ctx, d_in, d_index_out, d_token_out,
        d_state_states, d_index_states,
        size, NUM_LOGICAL_BLOCKS, d_dyn_index_ptr, d_new_size, d_is_valid
    );
    gpuAssert(cudaDeviceSynchronize());
    gpuAssert(cudaPeekAtLastError());

    I result_size = 0;
    bool is_valid  = false;
    gpuAssert(cudaMemcpy(&result_size, d_new_size, sizeof(I),    cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(&is_valid,    d_is_valid, sizeof(bool), cudaMemcpyDeviceToHost));

    std::vector<I>       h_indices(result_size);
    std::vector<token_t> h_tokens(result_size);
    gpuAssert(cudaMemcpy(h_indices.data(), d_index_out, result_size * sizeof(I),       cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(h_tokens.data(),  d_token_out, result_size * sizeof(token_t), cudaMemcpyDeviceToHost));

    DEBUG_PRINT("Got %u tokens (valid=%s):\n", result_size, is_valid ? "true" : "false");
    for (I i = 0; i < result_size; i++) {
        DEBUG_PRINT("  [%u] index=%u token=%s(%u)\n",
                    i, h_indices[i],
                    token_name(h_tokens[i]), h_tokens[i]);
    }

    bool pass = is_valid;
    if (!pass) {
        fprintf(stderr, "FAIL [%s]: lexer did not reach accepting state\n", test->name);
    }
    if (pass && result_size != (I) test->expected_size) {
        fprintf(stderr, "FAIL [%s]: expected %zu tokens, got %u\n",
                test->name, test->expected_size, result_size);
        pass = false;
    }
    if (pass) {
        for (I i = 0; i < (I) test->expected_size; i++) {
            if (h_indices[i] != test->expected_indices[i]) {
                fprintf(stderr, "FAIL [%s]: index mismatch at %u: expected %u got %u\n",
                        test->name, i, test->expected_indices[i], h_indices[i]);
                pass = false; break;
            }
            if (h_tokens[i] != test->expected_tokens[i]) {
                fprintf(stderr, "FAIL [%s]: token mismatch at %u: expected %s got %s\n",
                        test->name, i,
                        token_name(test->expected_tokens[i]),
                        token_name(h_tokens[i]));
                pass = false; break;
            }
        }
    }

    if (pass) printf("PASS [%s]\n", test->name);

    ctx.Cleanup();
    gpuAssert(cudaFree(d_in));
    gpuAssert(cudaFree(d_index_out));
    gpuAssert(cudaFree(d_token_out));
    gpuAssert(cudaFree(d_state_states.d_tile_descriptors));
    gpuAssert(cudaFree(d_index_states.d_tile_descriptors));
    gpuAssert(cudaFree(d_dyn_index_ptr));
    gpuAssert(cudaFree(d_new_size));
    gpuAssert(cudaFree(d_is_valid));

    return pass;
}

#ifdef DEBUG
int main() {
    // Indices are the end positions (inclusive, 0-indexed) of each token.
    // Tokens: SPACE=0, IDENT=1, LPAREN=2, RPAREN=3

    // "()" -> LPAREN@0, RPAREN@1
    uint32_t idx0[] = {0, 1};
    uint8_t  tok0[] = {TOKEN_LPAREN, TOKEN_RPAREN};

    // "(a)" -> LPAREN@0, IDENT@1, RPAREN@2
    uint32_t idx1[] = {0, 1, 2};
    uint8_t  tok1[] = {TOKEN_LPAREN, TOKEN_IDENT, TOKEN_RPAREN};

    // "(foo)" -> LPAREN@0, IDENT@3, RPAREN@4
    uint32_t idx2[] = {0, 3, 4};
    uint8_t  tok2[] = {TOKEN_LPAREN, TOKEN_IDENT, TOKEN_RPAREN};

    // "( )" -> LPAREN@0, SPACE@1, RPAREN@2
    uint32_t idx3[] = {0, 1, 2};
    uint8_t  tok3[] = {TOKEN_LPAREN, TOKEN_SPACE, TOKEN_RPAREN};

    // "(foo bar)" -> LPAREN@0, IDENT@3, SPACE@4, IDENT@7, RPAREN@8
    uint32_t idx4[] = {0, 3, 4, 7, 8};
    uint8_t  tok4[] = {TOKEN_LPAREN, TOKEN_IDENT, TOKEN_SPACE, TOKEN_IDENT, TOKEN_RPAREN};

    // "(foo (bar baz))" -> LPAREN@0, IDENT@3, SPACE@4, LPAREN@5, IDENT@8,
    //                      SPACE@9, IDENT@12, RPAREN@13, RPAREN@14
    uint32_t idx5[] = {0, 3, 4, 5, 8, 9, 12, 13, 14};
    uint8_t  tok5[] = {TOKEN_LPAREN, TOKEN_IDENT, TOKEN_SPACE,
                       TOKEN_LPAREN, TOKEN_IDENT, TOKEN_SPACE,
                       TOKEN_IDENT,  TOKEN_RPAREN, TOKEN_RPAREN};

    LexerTest tests[] = {
        {"()",              idx0, tok0, 2, "empty parens"},
        {"(a)",             idx1, tok1, 3, "single char ident"},
        {"(foo)",           idx2, tok2, 3, "ident in parens"},
        {"( )",             idx3, tok3, 3, "space in parens"},
        {"(foo bar)",       idx4, tok4, 5, "two idents"},
        {"(foo (bar baz))", idx5, tok5, 9, "nested parens"},
    };

    int n_tests = sizeof(tests) / sizeof(tests[0]);
    int passed  = 0;
    for (int i = 0; i < n_tests; i++) {
        if (runTest(&tests[i])) passed++;

    }

    printf("\n%d/%d tests passed\n", passed, n_tests);
    return passed == n_tests ? 0 : 1;
}

#else

int main(int32_t argc, char *argv[]) {
    assert(argc == 4);
    size_t input_size;
    uint8_t* input = read_u8_array(argv[1], &input_size);

    size_t expected_indices_size;
    uint32_t* expected_indices = read_u32_array(argv[2], &expected_indices_size);

    size_t expected_tokens_size;
    uint8_t* expected_tokens = read_u8_array(argv[3], &expected_tokens_size);

    assert(expected_indices_size == expected_tokens_size);

    printf("%s:\n", argv[1]);

    //printf(PAD, "Lexer Shmem Compose (u64 orig):");
    //testLexerShmemComposeU64(input, input_size, expected_indices, expected_tokens, expected_indices_size);
    //printf(PAD, "Lexer Shmem Compose:");
    //testLexerShmemCompose(input, input_size, expected_indices, expected_tokens, expected_indices_size);
    //printf(PAD, "Lexer Alpacc Shmem IPT=30:");
    //testLexerAlpaccShmem<30>(input, input_size, expected_indices, expected_tokens, expected_indices_size);
    //printf(PAD, "Lexer Alpacc Shmem Dyn BS1024 IPT=44:");
    //testLexerAlpaccShmemDyn<1024, 44>(input, input_size, expected_indices, expected_tokens, expected_indices_size);
    printf(PAD, "BW ceiling BS256/IPT22 (read only):");
    testBwCeilingRead<256, 22>(input, input_size);
    printf(PAD, "2Pass V2 BS256/IPT22 (NregNone):");
    testLexerAlpaccShmemTwoPassV2<256, 22, 256, 18, 0, 0>(input, input_size, expected_indices, expected_tokens, expected_indices_size);
    printf(PAD, "2Pass V2 BS256/IPT22 (U32 shmem):");
    testLexerAlpaccShmemTwoPassV2U32<256, 22, 256, 18>(input, input_size, expected_indices, expected_tokens, expected_indices_size);

    free(input);
    free(expected_indices);
    free(expected_tokens);
    gpuAssert(cudaPeekAtLastError());
    return 0;
}


#endif
