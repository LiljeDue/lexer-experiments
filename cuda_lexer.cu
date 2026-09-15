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

// Like LexerCtx but uses __ldg() for read-only texture-cached lookups.
struct LexerCtxLdg {
    state_t* d_to_state;
    state_t* d_compose;

    LexerCtxLdg() : d_to_state(NULL), d_compose(NULL) {
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
#ifdef __CUDA_ARCH__
        return __ldg(&d_compose[get_index(b) * NUM_STATES + get_index(a)]);
#else
        return d_compose[get_index(b) * NUM_STATES + get_index(a)];
#endif
    }

    __device__ __host__ __forceinline__
    state_t operator()(const volatile state_t &a, const volatile state_t &b) const {
#ifdef __CUDA_ARCH__
        return __ldg(&d_compose[get_index(b) * NUM_STATES + get_index(a)]);
#else
        return d_compose[get_index(b) * NUM_STATES + get_index(a)];
#endif
    }

    __device__ __host__ __forceinline__
    state_t to_state(const char &a) const {
#ifdef __CUDA_ARCH__
        return __ldg(&d_to_state[(uint8_t)a]);
#else
        return d_to_state[(uint8_t)a];
#endif
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

// Inter-block decoupled lookback that operates on a register-held tile.
// Takes the block-local inclusive-scan aggregate, publishes it, waits for
// predecessors, and returns the exclusive inter-block prefix to all threads.
// Callers apply the prefix to their register tile themselves.
// Mirrors decoupledLookbackScanNoWrite but takes aggregate as a parameter.
template<typename T, typename I, typename OP>
__device__ inline T
decoupledLookbackPrefix(volatile State<T>* states,
                        OP op,
                        const T ne,
                        uint32_t dyn_idx,
                        T aggregate) {
    volatile __shared__ T values[WARP];
    volatile __shared__ Status statuses[WARP];
    volatile __shared__ T shmem_prefix;
    const uint8_t lane = threadIdx.x & (WARP - 1);
    const bool is_first = threadIdx.x == 0;
    const bool is_first_block = dyn_idx == 0;

    if (is_first) {
        states[dyn_idx].aggregate = aggregate;
    }
    if (is_first_block && is_first) {
        states[dyn_idx].prefix = aggregate;
    }
    __threadfence();
    if (is_first_block && is_first) {
        states[dyn_idx].status = Prefix;
    } else if (is_first) {
        states[dyn_idx].status = Aggregate;
    }

    T prefix;
    bool is_first_iter = true;
    if (threadIdx.x < WARP && !is_first_block) {
        I lookback_idx = threadIdx.x + dyn_idx;
        I lookback_warp = WARP;
        Status status = Aggregate;
        do {
            if (lookback_warp <= lookback_idx) {
                I idx = lookback_idx - lookback_warp;
                status = states[idx].status;
                statuses[threadIdx.x] = status;
                values[threadIdx.x] = status == Prefix ? states[idx].prefix : states[idx].aggregate;
            } else {
                statuses[threadIdx.x] = Aggregate;
            }
            scanWarp<T, I, OP>(values, statuses, op, lane, dyn_idx, lookback_warp);
            status = statuses[WARP - 1];
            if (status == Invalid) continue;
            if (is_first) {
                if (is_first_iter) {
                    prefix = values[WARP - 1];
                } else {
                    prefix = op(values[WARP - 1], prefix);
                }
            }
            lookback_warp += WARP;
            is_first_iter = false;
        } while (status != Prefix);
    }

    if (!is_first_block && is_first) {
        shmem_prefix = prefix;
    }
    __syncthreads();

    if (is_first) {
        if (!is_first_block) {
            states[dyn_idx].prefix = op(shmem_prefix, aggregate);
        }
        __threadfence();
        states[dyn_idx].status = Prefix;
    }

    return is_first_block ? ne : shmem_prefix;
}

template<typename T, typename I, typename OP, I ITEMS_PER_THREAD>
__device__ inline void
decoupledLookbackScanSuffix(volatile State<T>* states,
                            volatile state_t* suffixes,
                            volatile T* shmem,
                            OP op,
                            const T ne,
                            uint32_t dyn_idx) {
    volatile __shared__ T values[WARP];
    volatile __shared__ Status statuses[WARP];
    volatile __shared__ T shmem_prefix;
    const uint8_t lane = threadIdx.x & (WARP - 1);
    const bool is_first = threadIdx.x == 0;

    T aggregate = shmem[ITEMS_PER_THREAD * blockDim.x - 1];

    if (is_first) {
        states[dyn_idx].aggregate = aggregate;
    }
    
    if (dyn_idx == 0 && is_first) {
        states[dyn_idx].prefix = aggregate;
    }
    
    __threadfence();
    if (dyn_idx == 0 && is_first) {
        states[dyn_idx].status = Prefix;
    } else if (is_first) {
        states[dyn_idx].status = Aggregate;
    }

    T prefix = ne;
    if (threadIdx.x < WARP && dyn_idx != 0) {
        I lookback_idx = threadIdx.x + dyn_idx;
        I lookback_warp = WARP;
        Status status = Aggregate;
        do {
            if (lookback_warp <= lookback_idx) {
                I idx = lookback_idx - lookback_warp;
                status = states[idx].status;
                statuses[threadIdx.x] = status;
                values[threadIdx.x] = status == Prefix ? states[idx].prefix : states[idx].aggregate;
            } else {
                statuses[threadIdx.x] = Aggregate;
                values[threadIdx.x] = ne;
            }

            scanWarp<T, I, OP>(values, statuses, op, lane);

            T result = values[WARP - 1];
            status = statuses[WARP - 1];

            if (status == Invalid)
                continue;
                
            if (is_first) {
                prefix = op(result, prefix);
            }

            lookback_warp += WARP;
        } while (status != Prefix);
    }

    if (is_first) {
        shmem_prefix = prefix;
    }

    __syncthreads();

    prefix = shmem_prefix;
    const I offset = threadIdx.x * ITEMS_PER_THREAD;
    const I upper = offset + ITEMS_PER_THREAD;
    #pragma unroll
    for (I lid = offset; lid < upper; lid++) {
        shmem[lid] = op(prefix, shmem[lid]);
    }

    if (is_first) {
        states[dyn_idx].prefix = op(prefix, aggregate);
        suffixes[dyn_idx] = shmem[0]; 
        __threadfence();
        states[dyn_idx].status = Prefix;
    }
    
    __syncthreads();
}

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


template<typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__ void
lexer(LexerCtx ctx,
      uint8_t* d_in,
      uint32_t* d_index_out,
      token_t* d_token_out,
      volatile State<state_t>* state_states,
      volatile State<I>* index_states,
      I size,
      I num_logical_blocks,
      volatile uint32_t* dyn_index_ptr,
      volatile I* new_size,
      volatile bool* is_valid) {
    volatile __shared__ state_t states[ITEMS_PER_THREAD * BLOCK_SIZE];
    volatile __shared__ I indices[ITEMS_PER_THREAD * BLOCK_SIZE];
    volatile __shared__ I indices_aux[BLOCK_SIZE];
    __shared__ state_t next_block_first_state;
    volatile state_t* states_aux = (volatile state_t*) indices;
    const I REG_MEM = 1 + ITEMS_PER_THREAD / sizeof(uint64_t);
    uint64_t copy_reg[REG_MEM];
    uint8_t *chars_reg = (uint8_t*) copy_reg;
    uint32_t is_produce_state = 0;

    uint32_t dyn_index = dynamicIndex<uint32_t>(dyn_index_ptr);
    I glb_offs = dyn_index * BLOCK_SIZE * ITEMS_PER_THREAD;
    
    states_aux[threadIdx.x] = ctx.to_state(threadIdx.x);
    
    if (threadIdx.x == I()) {
        next_block_first_state = IDENTITY;
    }

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
                if (loc_gid < size) {
                    chars_reg[sizeof(uint64_t) * i + j] = d_in[loc_gid];
                }
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
                states[lid_off] = states_aux[chars_reg[reg_off]];
            } else if (is_in_block) {
                states[lid_off] = IDENTITY;
            } else if (lid_off == ITEMS_PER_THREAD * BLOCK_SIZE) {
                next_block_first_state = states_aux[chars_reg[reg_off]];
            }
        }
    }

    __syncthreads();

    scan<state_t, I, LexerCtx, ITEMS_PER_THREAD>(states, states_aux, state_states, ctx, dyn_index);

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = i * blockDim.x + threadIdx.x;
        I gid = glb_offs + lid;
        bool temp = false;
        if (gid < size) {
            if (lid == ITEMS_PER_THREAD * BLOCK_SIZE - 1) {
                temp = gid == size - 1 || is_produce(ctx(states[lid], next_block_first_state));
            } else {
                temp = gid == size - 1 || is_produce(states[lid + 1]);
            }
        }
        is_produce_state |= temp << i;
        indices[lid] = temp;
    }

    __syncthreads();

    scanBlock<I, I, Add<I>, ITEMS_PER_THREAD>(indices, indices_aux, Add<I>());

    I prefix = decoupledLookbackScanNoWrite<I, I, Add<I>, ITEMS_PER_THREAD>(index_states, indices, Add<I>(), I(), dyn_index);

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = blockDim.x * i + threadIdx.x;
        I gid = glb_offs + lid;
        if (gid < size && ((is_produce_state >> i) & 1)) {
            I offset = Add<I>()(prefix, indices[lid]) - 1;
            d_index_out[offset] = gid;
            d_token_out[offset] = get_token(states[lid]);
        }
    }
    
    if (dyn_index == num_logical_blocks - 1 && threadIdx.x == blockDim.x - 1) {
        *new_size = Add<I>()(prefix, indices[ITEMS_PER_THREAD * BLOCK_SIZE - 1]);
        *is_valid = is_accept(states[ITEMS_PER_THREAD * BLOCK_SIZE - 1]);
    }
}

template<typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__ void
lexerShmemCompose(LexerCtxShmem ctx,
      uint8_t* d_in,
      uint32_t* d_index_out,
      token_t* d_token_out,
      volatile State<state_t>* state_states,
      volatile State<I>* index_states,
      I size,
      I num_logical_blocks,
      volatile uint32_t* dyn_index_ptr,
      volatile I* new_size,
      volatile bool* is_valid) {
    // Compose table in shared memory: 144 entries × 2 bytes = 288 bytes per block.
    // Reducing ITEMS_PER_THREAD by 1 (from 31 to 30) frees 1536 bytes, giving
    // headroom for the 288-byte compose table within the 48 KB shmem limit.
    __shared__ __align__(8) state_t shmem_compose[NUM_STATES * NUM_STATES];
    volatile __shared__ state_t states[ITEMS_PER_THREAD * BLOCK_SIZE];
    volatile __shared__ I indices[ITEMS_PER_THREAD * BLOCK_SIZE];
    volatile __shared__ I indices_aux[BLOCK_SIZE];
    __shared__ state_t next_block_first_state;
    volatile state_t* states_aux = (volatile state_t*) indices;
    const I REG_MEM = 1 + ITEMS_PER_THREAD / sizeof(uint64_t);
    uint64_t copy_reg[REG_MEM];
    uint8_t *chars_reg = (uint8_t*) copy_reg;
    uint32_t is_produce_state = 0;

    copyFromGlbToShr<state_t, I, 1>(0, NUM_STATES * NUM_STATES, NUM_STATES * NUM_STATES, ctx.d_compose_glb, shmem_compose);
    ctx.d_compose = shmem_compose;

    uint32_t dyn_index = dynamicIndex<uint32_t>(dyn_index_ptr);
    I glb_offs = dyn_index * BLOCK_SIZE * ITEMS_PER_THREAD;

    states_aux[threadIdx.x] = ctx.to_state(threadIdx.x);

    if (threadIdx.x == I()) {
        next_block_first_state = IDENTITY;
    }

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
                if (loc_gid < size) {
                    chars_reg[sizeof(uint64_t) * i + j] = d_in[loc_gid];
                }
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
                states[lid_off] = states_aux[chars_reg[reg_off]];
            } else if (is_in_block) {
                states[lid_off] = IDENTITY;
            } else if (lid_off == ITEMS_PER_THREAD * BLOCK_SIZE) {
                next_block_first_state = states_aux[chars_reg[reg_off]];
            }
        }
    }

    __syncthreads();

    scan<state_t, I, LexerCtxShmem, ITEMS_PER_THREAD>(states, states_aux, state_states, ctx, dyn_index);

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = i * blockDim.x + threadIdx.x;
        I gid = glb_offs + lid;
        bool temp = false;
        if (gid < size) {
            if (lid == ITEMS_PER_THREAD * BLOCK_SIZE - 1) {
                temp = gid == size - 1 || is_produce(ctx(states[lid], next_block_first_state));
            } else {
                temp = gid == size - 1 || is_produce(states[lid + 1]);
            }
        }
        is_produce_state |= temp << i;
        indices[lid] = temp;
    }

    __syncthreads();

    scanBlock<I, I, Add<I>, ITEMS_PER_THREAD>(indices, indices_aux, Add<I>());

    I prefix = decoupledLookbackScanNoWrite<I, I, Add<I>, ITEMS_PER_THREAD>(index_states, indices, Add<I>(), I(), dyn_index);

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = blockDim.x * i + threadIdx.x;
        I gid = glb_offs + lid;
        if (gid < size && ((is_produce_state >> i) & 1)) {
            I offset = Add<I>()(prefix, indices[lid]) - 1;
            d_index_out[offset] = gid;
            d_token_out[offset] = get_token(states[lid]);
        }
    }

    if (dyn_index == num_logical_blocks - 1 && threadIdx.x == blockDim.x - 1) {
        *new_size = Add<I>()(prefix, indices[ITEMS_PER_THREAD * BLOCK_SIZE - 1]);
        *is_valid = is_accept(states[ITEMS_PER_THREAD * BLOCK_SIZE - 1]);
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
      volatile State<state_t>* state_states,
      volatile State<I>* index_states,
      I size,
      I num_logical_blocks,
      volatile uint32_t* dyn_index_ptr,
      volatile I* new_size,
      volatile bool* is_valid,
      state_t identity) {
    using BlockScanState = cub::BlockScan<state_t, BLOCK_SIZE>;
    using BlockScanI     = cub::BlockScan<I, BLOCK_SIZE>;
    __shared__ union {
        typename BlockScanState::TempStorage state_scan;
        typename BlockScanI::TempStorage     index_scan;
    } temp_storage;
    // Flat shmem state tile: written by the striped global load, then read
    // into blocked registers for the CUB scan. Also reused as lid_stage[]
    // (uint16_t) during the staged scatter phase.
    volatile __shared__ state_t states[ITEMS_PER_THREAD * BLOCK_SIZE];
    // Token staging buffer for the coalesced scatter phase.
    volatile __shared__ uint8_t tok_stage[ITEMS_PER_THREAD * BLOCK_SIZE];
    volatile __shared__ state_t to_state_shr[256];
    __shared__ state_t next_block_first_state;

    const I REG_MEM = 1 + ITEMS_PER_THREAD / sizeof(uint64_t);
    uint64_t copy_reg[REG_MEM];
    uint8_t *chars_reg = (uint8_t*) copy_reg;
    state_t st[ITEMS_PER_THREAD];   // blocked register tile for CUB scan
    I prod[ITEMS_PER_THREAD];       // produce flags, blocked layout
    uint32_t is_produce_state = 0;

    uint32_t dyn_index = dynamicIndex<uint32_t>(dyn_index_ptr);
    I glb_offs = dyn_index * BLOCK_SIZE * ITEMS_PER_THREAD;

    to_state_shr[threadIdx.x] = ctx.to_state(threadIdx.x);

    if (threadIdx.x == I()) {
        next_block_first_state = identity;
    }

    __syncthreads();

    // Vectorized global→register load of input bytes (striped layout).
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

    // Map bytes→states into flat shmem (same striped pattern as original lexer).
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
                states[lid_off] = identity;
            } else if (lid_off == ITEMS_PER_THREAD * BLOCK_SIZE) {
                next_block_first_state = to_state_shr[chars_reg[reg_off]];
            }
        }
    }

    __syncthreads();

    // Load flat shmem into blocked register layout for CUB scan.
    // Thread t owns positions [t*IPT, (t+1)*IPT).
    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++)
        st[i] = states[threadIdx.x * ITEMS_PER_THREAD + i];

    // CUB BlockScan: inclusive scan on the blocked register tile.
    state_t aggregate;
    BlockScanState(temp_storage.state_scan).InclusiveScan(st, st, ctx, aggregate);

    // Inter-block decoupled lookback: get exclusive prefix, apply to registers.
    state_t pfx = decoupledLookbackPrefix<state_t, I, CTX>(state_states, ctx, identity, dyn_index, aggregate);
    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++)
        st[i] = ctx(pfx, st[i]);

    // Write post-scan results back to flat shmem for cross-thread produce check.
    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++)
        states[threadIdx.x * ITEMS_PER_THREAD + i] = st[i];

    __syncthreads();

    // Compute produce flags. Reads states[lid+1] from shmem for cross-thread
    // boundaries, matching the original lexer's produce detection exactly.
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
        is_produce_state |= (uint32_t)temp << i;
        prod[i] = (I)temp;
    }

    // CUB BlockScan: inclusive Add scan over produce flags (blocked layout).
    I prod_agg;
    BlockScanI(temp_storage.index_scan).InclusiveScan(prod, prod, Add<I>(), prod_agg);

    // Inter-block decoupled lookback for the compaction scan.
    I idx_pfx = decoupledLookbackPrefix<I, I, Add<I>>(index_states, Add<I>(), I(), dyn_index, prod_agg);

    // Stage selected (gid, token) pairs into shmem at their tile-relative
    // output positions, then flush sequentially to global memory.
    // This avoids irregular scatter writes to d_index_out/d_token_out.
    //
    // Reuse states[] as the index staging buffer: store tile-relative uint16_t
    // positions (lid, max 7935) so they fit in state_t (uint16_t).
    // Reuse to_state_shr[] (256 × uint16_t = 512 bytes) as the token buffer
    // after it is no longer needed. Since prod_agg <= 256 × IPT in the worst
    // case we need up to IPT*BS bytes for tokens. to_state_shr is only 256
    // bytes — not enough. Instead pack token into the high byte of states[]:
    // states[slot] = (token << 8) | (lid_lo), with gid reconstructed as
    // glb_offs + lid stored separately. This is fragile with uint16_t.
    //
    // Simplest correct approach: use states[] for tile-local lids (uint16_t)
    // and reuse to_state_shr for tokens (uint8_t), but to_state_shr is only
    // 256 entries. Instead declare a token staging array in the union with
    // CUB temp storage (which is no longer needed after the scans complete).
    volatile uint16_t* lid_stage = (volatile uint16_t*) states;

    // Phase 1: each thread writes its producing elements into shmem at the
    // tile-relative output slot (prod[i] - 1), in blocked layout.
    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        if ((is_produce_state >> i) & 1) {
            I slot = prod[i] - 1;                  // tile-relative output index
            I lid  = threadIdx.x * ITEMS_PER_THREAD + i;
            lid_stage[slot] = (uint16_t) lid;      // tile-local input position
            tok_stage[slot] = get_token(st[i]);
        }
    }

    __syncthreads();

    // Phase 2: all threads read sequentially from shmem and write to global
    // memory in coalesced fashion. Each thread handles prod_agg/BLOCK_SIZE
    // output elements in a striped pattern.
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
__global__ __launch_bounds__(BLOCK_SIZE, 4)
void lexerAlpacc(LexerCtxLdg ctx,
      uint8_t* d_in, uint32_t* d_index_out, token_t* d_token_out,
      volatile State<state_t>* state_states, volatile State<I>* index_states,
      I size, I num_logical_blocks, volatile uint32_t* dyn_index_ptr,
      volatile I* new_size, volatile bool* is_valid) {
    lexerAlpaccImpl<LexerCtxLdg, I, BLOCK_SIZE, ITEMS_PER_THREAD>(
        ctx, d_in, d_index_out, d_token_out, state_states, index_states,
        size, num_logical_blocks, dyn_index_ptr, new_size, is_valid, IDENTITY);
}

// Like lexerAlpacc but loads the compose table into shmem once per block.
// ITEMS_PER_THREAD is 1 less to fit the 288-byte table within 48KB shmem.
template<typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__ __launch_bounds__(BLOCK_SIZE, 4)
void lexerAlpaccShmem(LexerCtxShmem ctx,
      uint8_t* d_in, uint32_t* d_index_out, token_t* d_token_out,
      volatile State<state_t>* state_states, volatile State<I>* index_states,
      I size, I num_logical_blocks, volatile uint32_t* dyn_index_ptr,
      volatile I* new_size, volatile bool* is_valid) {
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
      volatile State<state_t>* state_states,
      volatile State<I>* index_states,
      I size,
      I num_logical_blocks,
      volatile uint32_t* dyn_index_ptr,
      volatile I* new_size,
      volatile bool* is_valid,
      state_t identity) {
    using BlockScanState = cub::BlockScan<state_t, BLOCK_SIZE>;
    using BlockScanI     = cub::BlockScan<I, BLOCK_SIZE>;
    __shared__ union {
        typename BlockScanState::TempStorage state_scan;
        typename BlockScanI::TempStorage     index_scan;
    } temp_storage;
    __shared__ state_t to_state_shr[256];
    __shared__ state_t next_block_first_state;

    // Carve dynamic shmem into three regions.
    extern __shared__ uint16_t dyn_shmem[];
    volatile state_t* shmem_compose = (volatile state_t*) dyn_shmem;
    volatile state_t* states        = shmem_compose + NUM_STATES * NUM_STATES;
    volatile uint8_t* tok_stage     = (volatile uint8_t*) (states + ITEMS_PER_THREAD * BLOCK_SIZE);

    // Reuse states[] as uint16_t lid_stage during the scatter phase (same memory).
    volatile uint16_t* lid_stage    = (volatile uint16_t*) states;

    const I REG_MEM = 1 + ITEMS_PER_THREAD / sizeof(uint64_t);
    uint64_t copy_reg[REG_MEM];
    uint8_t *chars_reg = (uint8_t*) copy_reg;
    state_t st[ITEMS_PER_THREAD];
    I prod[ITEMS_PER_THREAD];
    uint32_t is_produce_state = 0;

    uint32_t dyn_index = dynamicIndex<uint32_t>(dyn_index_ptr);
    I glb_offs = dyn_index * BLOCK_SIZE * ITEMS_PER_THREAD;

    // Load compose table and to_state map into shmem.
    copyFromGlbToShr<state_t, I, 1>(0, NUM_STATES * NUM_STATES, NUM_STATES * NUM_STATES,
                                     ctx.d_compose_glb, shmem_compose);
    ctx.d_compose = (state_t*) shmem_compose;
    copyFromGlbToShr<state_t, I, 1>(0, 256, 256, ctx.d_to_state, to_state_shr);

    if (threadIdx.x == I()) {
        next_block_first_state = identity;
    }

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
                states[lid_off] = identity;
            } else if (lid_off == ITEMS_PER_THREAD * BLOCK_SIZE) {
                next_block_first_state = to_state_shr[chars_reg[reg_off]];
            }
        }
    }

    __syncthreads();

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++)
        st[i] = states[threadIdx.x * ITEMS_PER_THREAD + i];

    state_t aggregate;
    BlockScanState(temp_storage.state_scan).InclusiveScan(st, st, ctx, aggregate);

    state_t pfx = decoupledLookbackPrefix<state_t, I, LexerCtxShmem>(state_states, ctx, identity, dyn_index, aggregate);
    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++)
        st[i] = ctx(pfx, st[i]);

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
        is_produce_state |= (uint32_t)temp << i;
        prod[i] = (I)temp;
    }

    I prod_agg;
    BlockScanI(temp_storage.index_scan).InclusiveScan(prod, prod, Add<I>(), prod_agg);

    I idx_pfx = decoupledLookbackPrefix<I, I, Add<I>>(index_states, Add<I>(), I(), dyn_index, prod_agg);

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        if ((is_produce_state >> i) & 1) {
            I slot = prod[i] - 1;
            I lid  = threadIdx.x * ITEMS_PER_THREAD + i;
            lid_stage[slot] = (uint16_t) lid;
            tok_stage[slot] = get_token(st[i]);
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

// Dynamic-shmem kernel wrapper. The caller must set the 164 KB shmem carveout
// via cudaFuncSetAttribute before launching (see launchLexerAlpaccShmemDyn).
template<typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__ __launch_bounds__(BLOCK_SIZE)
void lexerAlpaccShmemDyn(LexerCtxShmem ctx,
      uint8_t* d_in, uint32_t* d_index_out, token_t* d_token_out,
      volatile State<state_t>* state_states, volatile State<I>* index_states,
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
      volatile State<state_t>* state_states, volatile State<I>* index_states,
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

void testLexer(uint8_t* input,
               size_t input_size,
               uint32_t* expected_indices,
               token_t* expected_tokens,
               size_t expected_size) {
    using I = uint32_t;
    const I size = input_size;
    const I BLOCK_SIZE = 256;
    const I ITEMS_PER_THREAD = 31;
    const I NUM_LOGICAL_BLOCKS = (size + BLOCK_SIZE * ITEMS_PER_THREAD - 1) / (BLOCK_SIZE * ITEMS_PER_THREAD);
    const I IN_ARRAY_BYTES = size * sizeof(uint8_t);
    const I INDEX_OUT_ARRAY_BYTES = size * sizeof(I);
    const I TOKEN_OUT_ARRAY_BYTES = size * sizeof(token_t);
    const I STATE_STATES_BYTES = NUM_LOGICAL_BLOCKS * sizeof(State<state_t>);
    const I INDEX_STATES_BYTES = NUM_LOGICAL_BLOCKS * sizeof(State<I>);
    const I WARMUP_RUNS = 500;
    const I RUNS = 100;

    std::vector<token_t> h_token_out(size, 0);
    std::vector<I> h_index_out(size, 0);

    uint32_t* d_dyn_index_ptr;
    I* d_new_size;
    bool* d_is_valid;
    uint8_t *d_in;
    I *d_index_out;
    token_t *d_token_out;
    State<I>* d_index_states;
    State<state_t>* d_state_states;
    gpuAssert(cudaMalloc((void**)&d_dyn_index_ptr, sizeof(uint32_t)));
    gpuAssert(cudaMalloc((void**)&d_new_size, sizeof(I)));
    gpuAssert(cudaMalloc((void**)&d_is_valid, sizeof(bool)));
    cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t));
    cudaMemset(d_is_valid, false, sizeof(bool));
    gpuAssert(cudaMalloc((void**)&d_index_states, INDEX_STATES_BYTES));
    gpuAssert(cudaMalloc((void**)&d_state_states, STATE_STATES_BYTES));
    gpuAssert(cudaMalloc((void**)&d_in, IN_ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_index_out, INDEX_OUT_ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_token_out, TOKEN_OUT_ARRAY_BYTES));
    gpuAssert(cudaMemcpy(d_in, input, IN_ARRAY_BYTES, cudaMemcpyHostToDevice));
    
    LexerCtx ctx = LexerCtx();

    float * temp = (float *) malloc(sizeof(float) * RUNS);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (I i = 0; i < WARMUP_RUNS; ++i) {
        lexer<I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
            ctx,
            d_in,
            d_index_out,
            d_token_out,
            d_state_states,
            d_index_states,
            size,
            NUM_LOGICAL_BLOCKS,
            d_dyn_index_ptr,
            d_new_size,
            d_is_valid
        );
        cudaDeviceSynchronize();
        cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t));
        cudaMemset(d_state_states, 0, STATE_STATES_BYTES);
        cudaMemset(d_index_states, 0, INDEX_STATES_BYTES);
        gpuAssert(cudaPeekAtLastError());
    }

    for (I i = 0; i < RUNS; ++i) {
        cudaEventRecord(start, 0);
        lexer<I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
            ctx,
            d_in,
            d_index_out,
            d_token_out,
            d_state_states,
            d_index_states,
            size,
            NUM_LOGICAL_BLOCKS,
            d_dyn_index_ptr,
            d_new_size,
            d_is_valid
        );
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(temp + i, start, stop);
        cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t));
        cudaMemset(d_state_states, 0, STATE_STATES_BYTES);
        cudaMemset(d_index_states, 0, INDEX_STATES_BYTES);
        gpuAssert(cudaPeekAtLastError());
    }

    I temp_size = 0;
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    const I OUT_WRITE = temp_size * (sizeof(I) + sizeof(token_t));
    const I IN_READ = IN_ARRAY_BYTES;
    const I IN_STATE_MAP = sizeof(state_t) * size;
    const I SCAN_READ = sizeof(state_t) * (size + size / 2);
    lexer<I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
        ctx,
        d_in,
        d_index_out,
        d_token_out,
        d_state_states,
        d_index_states,
        size,
        NUM_LOGICAL_BLOCKS,
        d_dyn_index_ptr,
        d_new_size,
        d_is_valid
    );
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
            test_passes &= h_index_out[i] == expected_indices[i];
            test_passes &= h_token_out[i] == expected_tokens[i];

            if (!test_passes) {
                std::cout << "Lexer Test Failed: Due to elements mismatch at index=" << i << std::endl;
                break;
            }
        } 
    }

    if (test_passes) {
        compute_descriptors(temp, RUNS, IN_READ + IN_STATE_MAP + SCAN_READ + OUT_WRITE);
    }

    free(temp);
    gpuAssert(cudaFree(d_in));
    gpuAssert(cudaFree(d_token_out));
    gpuAssert(cudaFree(d_index_out));
    gpuAssert(cudaFree(d_index_states));
    gpuAssert(cudaFree(d_state_states));
    gpuAssert(cudaFree(d_dyn_index_ptr));
    gpuAssert(cudaFree(d_new_size));
    gpuAssert(cudaFree(d_is_valid));


    ctx.Cleanup();
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
    const I STATE_STATES_BYTES = NUM_LOGICAL_BLOCKS * sizeof(State<state_t>);
    const I INDEX_STATES_BYTES = NUM_LOGICAL_BLOCKS * sizeof(State<I>);
    const I WARMUP_RUNS = 500;
    const I RUNS = 100;

    std::vector<token_t> h_token_out(size, 0);
    std::vector<I> h_index_out(size, 0);

    uint32_t* d_dyn_index_ptr;
    I* d_new_size;
    bool* d_is_valid;
    uint8_t *d_in;
    I *d_index_out;
    token_t *d_token_out;
    State<I>* d_index_states;
    State<state_t>* d_state_states;
    gpuAssert(cudaMalloc((void**)&d_dyn_index_ptr, sizeof(uint32_t)));
    gpuAssert(cudaMalloc((void**)&d_new_size, sizeof(I)));
    gpuAssert(cudaMalloc((void**)&d_is_valid, sizeof(bool)));
    cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t));
    cudaMemset(d_is_valid, false, sizeof(bool));
    gpuAssert(cudaMalloc((void**)&d_index_states, INDEX_STATES_BYTES));
    gpuAssert(cudaMalloc((void**)&d_state_states, STATE_STATES_BYTES));
    gpuAssert(cudaMalloc((void**)&d_in, IN_ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_index_out, INDEX_OUT_ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_token_out, TOKEN_OUT_ARRAY_BYTES));
    gpuAssert(cudaMemcpy(d_in, input, IN_ARRAY_BYTES, cudaMemcpyHostToDevice));

    LexerCtxShmem ctx = LexerCtxShmem();

    float * temp = (float *) malloc(sizeof(float) * RUNS);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (I i = 0; i < WARMUP_RUNS; ++i) {
        lexerShmemCompose<I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
            ctx,
            d_in,
            d_index_out,
            d_token_out,
            d_state_states,
            d_index_states,
            size,
            NUM_LOGICAL_BLOCKS,
            d_dyn_index_ptr,
            d_new_size,
            d_is_valid
        );
        cudaDeviceSynchronize();
        cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t));
        cudaMemset(d_state_states, 0, STATE_STATES_BYTES);
        cudaMemset(d_index_states, 0, INDEX_STATES_BYTES);
        gpuAssert(cudaPeekAtLastError());
    }

    for (I i = 0; i < RUNS; ++i) {
        cudaEventRecord(start, 0);
        lexerShmemCompose<I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
            ctx,
            d_in,
            d_index_out,
            d_token_out,
            d_state_states,
            d_index_states,
            size,
            NUM_LOGICAL_BLOCKS,
            d_dyn_index_ptr,
            d_new_size,
            d_is_valid
        );
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(temp + i, start, stop);
        cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t));
        cudaMemset(d_state_states, 0, STATE_STATES_BYTES);
        cudaMemset(d_index_states, 0, INDEX_STATES_BYTES);
        gpuAssert(cudaPeekAtLastError());
    }

    I temp_size = 0;
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    const I OUT_WRITE = temp_size * (sizeof(I) + sizeof(token_t));
    const I IN_READ = IN_ARRAY_BYTES;
    const I IN_STATE_MAP = sizeof(state_t) * 256 * NUM_LOGICAL_BLOCKS;
    const I COMPOSE_READ = sizeof(state_t) * NUM_STATES * NUM_STATES * NUM_LOGICAL_BLOCKS;

    lexerShmemCompose<I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
        ctx,
        d_in,
        d_index_out,
        d_token_out,
        d_state_states,
        d_index_states,
        size,
        NUM_LOGICAL_BLOCKS,
        d_dyn_index_ptr,
        d_new_size,
        d_is_valid
    );
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
            test_passes &= h_index_out[i] == expected_indices[i];
            test_passes &= h_token_out[i] == expected_tokens[i];

            if (!test_passes) {
                std::cout << "Lexer Test Failed: Due to elements mismatch at index=" << i << std::endl;
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
    gpuAssert(cudaFree(d_index_states));
    gpuAssert(cudaFree(d_state_states));
    gpuAssert(cudaFree(d_dyn_index_ptr));
    gpuAssert(cudaFree(d_new_size));
    gpuAssert(cudaFree(d_is_valid));

    ctx.Cleanup();
}

void testLexerAlpacc(uint8_t* input,
               size_t input_size,
               uint32_t* expected_indices,
               token_t* expected_tokens,
               size_t expected_size) {
    using I = uint32_t;
    const I size = input_size;
    const I BLOCK_SIZE = 256;
    const I ITEMS_PER_THREAD = 31;
    const I NUM_LOGICAL_BLOCKS = (size + BLOCK_SIZE * ITEMS_PER_THREAD - 1) / (BLOCK_SIZE * ITEMS_PER_THREAD);
    const I IN_ARRAY_BYTES = size * sizeof(uint8_t);
    const I INDEX_OUT_ARRAY_BYTES = size * sizeof(I);
    const I TOKEN_OUT_ARRAY_BYTES = size * sizeof(token_t);
    const I STATE_STATES_BYTES = NUM_LOGICAL_BLOCKS * sizeof(State<state_t>);
    const I INDEX_STATES_BYTES = NUM_LOGICAL_BLOCKS * sizeof(State<I>);
    const I WARMUP_RUNS = 500;
    const I RUNS = 100;

    std::vector<token_t> h_token_out(size, 0);
    std::vector<I> h_index_out(size, 0);

    uint32_t* d_dyn_index_ptr;
    I* d_new_size;
    bool* d_is_valid;
    uint8_t *d_in;
    I *d_index_out;
    token_t *d_token_out;
    State<I>* d_index_states;
    State<state_t>* d_state_states;
    gpuAssert(cudaMalloc((void**)&d_dyn_index_ptr, sizeof(uint32_t)));
    gpuAssert(cudaMalloc((void**)&d_new_size, sizeof(I)));
    gpuAssert(cudaMalloc((void**)&d_is_valid, sizeof(bool)));
    cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t));
    cudaMemset(d_is_valid, false, sizeof(bool));
    gpuAssert(cudaMalloc((void**)&d_index_states, INDEX_STATES_BYTES));
    gpuAssert(cudaMalloc((void**)&d_state_states, STATE_STATES_BYTES));
    gpuAssert(cudaMalloc((void**)&d_in, IN_ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_index_out, INDEX_OUT_ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_token_out, TOKEN_OUT_ARRAY_BYTES));
    gpuAssert(cudaMemcpy(d_in, input, IN_ARRAY_BYTES, cudaMemcpyHostToDevice));

    LexerCtxLdg ctx = LexerCtxLdg();

    float * temp = (float *) malloc(sizeof(float) * RUNS);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (I i = 0; i < WARMUP_RUNS; ++i) {
        lexerAlpacc<I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
            ctx, d_in, d_index_out, d_token_out, d_state_states, d_index_states,
            size, NUM_LOGICAL_BLOCKS, d_dyn_index_ptr, d_new_size, d_is_valid);
        cudaDeviceSynchronize();
        cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t));
        cudaMemset(d_state_states, 0, STATE_STATES_BYTES);
        cudaMemset(d_index_states, 0, INDEX_STATES_BYTES);
        gpuAssert(cudaPeekAtLastError());
    }

    for (I i = 0; i < RUNS; ++i) {
        cudaEventRecord(start, 0);
        lexerAlpacc<I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
            ctx, d_in, d_index_out, d_token_out, d_state_states, d_index_states,
            size, NUM_LOGICAL_BLOCKS, d_dyn_index_ptr, d_new_size, d_is_valid);
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(temp + i, start, stop);
        cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t));
        cudaMemset(d_state_states, 0, STATE_STATES_BYTES);
        cudaMemset(d_index_states, 0, INDEX_STATES_BYTES);
        gpuAssert(cudaPeekAtLastError());
    }

    I temp_size = 0;
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    const I OUT_WRITE = temp_size * (sizeof(I) + sizeof(token_t));
    const I IN_READ = IN_ARRAY_BYTES;
    const I IN_STATE_MAP = sizeof(state_t) * size;
    const I SCAN_READ = sizeof(state_t) * (size + size / 2);

    lexerAlpacc<I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
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
            test_passes &= h_index_out[i] == expected_indices[i];
            test_passes &= h_token_out[i] == expected_tokens[i];
            if (!test_passes) {
                std::cout << "Lexer Test Failed: Due to elements mismatch at index=" << i << std::endl;
                break;
            }
        }
    }

    if (test_passes) {
        compute_descriptors(temp, RUNS, IN_READ + IN_STATE_MAP + SCAN_READ + OUT_WRITE);
    }

    free(temp);
    gpuAssert(cudaFree(d_in));
    gpuAssert(cudaFree(d_token_out));
    gpuAssert(cudaFree(d_index_out));
    gpuAssert(cudaFree(d_index_states));
    gpuAssert(cudaFree(d_state_states));
    gpuAssert(cudaFree(d_dyn_index_ptr));
    gpuAssert(cudaFree(d_new_size));
    gpuAssert(cudaFree(d_is_valid));

    ctx.Cleanup();
}

void testLexerAlpaccShmem(uint8_t* input,
               size_t input_size,
               uint32_t* expected_indices,
               token_t* expected_tokens,
               size_t expected_size) {
    using I = uint32_t;
    const I size = input_size;
    const I BLOCK_SIZE = 256;
    const I ITEMS_PER_THREAD = 30;
    const I NUM_LOGICAL_BLOCKS = (size + BLOCK_SIZE * ITEMS_PER_THREAD - 1) / (BLOCK_SIZE * ITEMS_PER_THREAD);
    const I IN_ARRAY_BYTES = size * sizeof(uint8_t);
    const I INDEX_OUT_ARRAY_BYTES = size * sizeof(I);
    const I TOKEN_OUT_ARRAY_BYTES = size * sizeof(token_t);
    const I STATE_STATES_BYTES = NUM_LOGICAL_BLOCKS * sizeof(State<state_t>);
    const I INDEX_STATES_BYTES = NUM_LOGICAL_BLOCKS * sizeof(State<I>);
    const I WARMUP_RUNS = 500;
    const I RUNS = 100;

    std::vector<token_t> h_token_out(size, 0);
    std::vector<I> h_index_out(size, 0);

    uint32_t* d_dyn_index_ptr;
    I* d_new_size;
    bool* d_is_valid;
    uint8_t *d_in;
    I *d_index_out;
    token_t *d_token_out;
    State<I>* d_index_states;
    State<state_t>* d_state_states;
    gpuAssert(cudaMalloc((void**)&d_dyn_index_ptr, sizeof(uint32_t)));
    gpuAssert(cudaMalloc((void**)&d_new_size, sizeof(I)));
    gpuAssert(cudaMalloc((void**)&d_is_valid, sizeof(bool)));
    cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t));
    cudaMemset(d_is_valid, false, sizeof(bool));
    gpuAssert(cudaMalloc((void**)&d_index_states, INDEX_STATES_BYTES));
    gpuAssert(cudaMalloc((void**)&d_state_states, STATE_STATES_BYTES));
    gpuAssert(cudaMalloc((void**)&d_in, IN_ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_index_out, INDEX_OUT_ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_token_out, TOKEN_OUT_ARRAY_BYTES));
    gpuAssert(cudaMemcpy(d_in, input, IN_ARRAY_BYTES, cudaMemcpyHostToDevice));

    LexerCtxShmem ctx = LexerCtxShmem();

    float * temp = (float *) malloc(sizeof(float) * RUNS);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (I i = 0; i < WARMUP_RUNS; ++i) {
        lexerAlpaccShmem<I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
            ctx, d_in, d_index_out, d_token_out, d_state_states, d_index_states,
            size, NUM_LOGICAL_BLOCKS, d_dyn_index_ptr, d_new_size, d_is_valid);
        cudaDeviceSynchronize();
        cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t));
        cudaMemset(d_state_states, 0, STATE_STATES_BYTES);
        cudaMemset(d_index_states, 0, INDEX_STATES_BYTES);
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
        cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t));
        cudaMemset(d_state_states, 0, STATE_STATES_BYTES);
        cudaMemset(d_index_states, 0, INDEX_STATES_BYTES);
        gpuAssert(cudaPeekAtLastError());
    }

    I temp_size = 0;
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    const I OUT_WRITE = temp_size * (sizeof(I) + sizeof(token_t));
    const I IN_READ = IN_ARRAY_BYTES;
    const I IN_STATE_MAP = sizeof(state_t) * 256 * NUM_LOGICAL_BLOCKS;
    const I COMPOSE_READ = sizeof(state_t) * NUM_STATES * NUM_STATES * NUM_LOGICAL_BLOCKS;

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
            test_passes &= h_index_out[i] == expected_indices[i];
            test_passes &= h_token_out[i] == expected_tokens[i];
            if (!test_passes) {
                std::cout << "Lexer Test Failed: Due to elements mismatch at index=" << i << std::endl;
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
    gpuAssert(cudaFree(d_index_states));
    gpuAssert(cudaFree(d_state_states));
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
    const I STATE_STATES_BYTES = NUM_LOGICAL_BLOCKS * sizeof(State<state_t>);
    const I INDEX_STATES_BYTES = NUM_LOGICAL_BLOCKS * sizeof(State<I>);
    const I WARMUP_RUNS = 500;
    const I RUNS = 100;

    std::vector<token_t> h_token_out(size, 0);
    std::vector<I> h_index_out(size, 0);

    uint32_t* d_dyn_index_ptr;
    I* d_new_size;
    bool* d_is_valid;
    uint8_t *d_in;
    I *d_index_out;
    token_t *d_token_out;
    State<I>* d_index_states;
    State<state_t>* d_state_states;
    gpuAssert(cudaMalloc((void**)&d_dyn_index_ptr, sizeof(uint32_t)));
    gpuAssert(cudaMalloc((void**)&d_new_size, sizeof(I)));
    gpuAssert(cudaMalloc((void**)&d_is_valid, sizeof(bool)));
    cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t));
    cudaMemset(d_is_valid, false, sizeof(bool));
    gpuAssert(cudaMalloc((void**)&d_index_states, INDEX_STATES_BYTES));
    gpuAssert(cudaMalloc((void**)&d_state_states, STATE_STATES_BYTES));
    gpuAssert(cudaMalloc((void**)&d_in, IN_ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_index_out, INDEX_OUT_ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_token_out, TOKEN_OUT_ARRAY_BYTES));
    gpuAssert(cudaMemcpy(d_in, input, IN_ARRAY_BYTES, cudaMemcpyHostToDevice));

    LexerCtxShmem ctx = LexerCtxShmem();

    {
        cudaFuncAttributes attrs;
        cudaFuncGetAttributes(&attrs, lexerAlpaccShmemDyn<I, BLOCK_SIZE, ITEMS_PER_THREAD>);
        int optin = 0;
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
        size_t dyn = dynShmemBytes<I, BLOCK_SIZE, ITEMS_PER_THREAD>();
        printf("  BS=%u IPT=%u: static_shmem=%zu dyn_shmem=%zu total=%zu optin_max=%d\n",
               (unsigned)BLOCK_SIZE, (unsigned)ITEMS_PER_THREAD,
               attrs.sharedSizeBytes, dyn, attrs.sharedSizeBytes + dyn, optin);
        fflush(stdout);
    }

    float * temp = (float *) malloc(sizeof(float) * RUNS);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (I i = 0; i < WARMUP_RUNS; ++i) {
        launchLexerAlpaccShmemDyn<I, BLOCK_SIZE, ITEMS_PER_THREAD>(
            ctx, d_in, d_index_out, d_token_out, d_state_states, d_index_states,
            size, NUM_LOGICAL_BLOCKS, d_dyn_index_ptr, d_new_size, d_is_valid);
        cudaDeviceSynchronize();
        cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t));
        cudaMemset(d_state_states, 0, STATE_STATES_BYTES);
        cudaMemset(d_index_states, 0, INDEX_STATES_BYTES);
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
        cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t));
        cudaMemset(d_state_states, 0, STATE_STATES_BYTES);
        cudaMemset(d_index_states, 0, INDEX_STATES_BYTES);
        gpuAssert(cudaPeekAtLastError());
    }

    I temp_size = 0;
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    const I OUT_WRITE = temp_size * (sizeof(I) + sizeof(token_t));
    const I IN_READ = IN_ARRAY_BYTES;
    const I IN_STATE_MAP = sizeof(state_t) * 256 * NUM_LOGICAL_BLOCKS;
    const I COMPOSE_READ = sizeof(state_t) * NUM_STATES * NUM_STATES * NUM_LOGICAL_BLOCKS;

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
            test_passes &= h_index_out[i] == expected_indices[i];
            test_passes &= h_token_out[i] == expected_tokens[i];
            if (!test_passes) {
                std::cout << "Lexer Test Failed: Due to elements mismatch at index=" << i << std::endl;
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
    gpuAssert(cudaFree(d_index_states));
    gpuAssert(cudaFree(d_state_states));
    gpuAssert(cudaFree(d_dyn_index_ptr));
    gpuAssert(cudaFree(d_new_size));
    gpuAssert(cudaFree(d_is_valid));

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
    State<state_t>*  d_state_states;
    State<I>*        d_index_states;
    uint32_t*        d_dyn_index_ptr;
    I*               d_new_size;
    bool*            d_is_valid;

    gpuAssert(cudaMalloc((void**)&d_in,            size * sizeof(uint8_t)));
    gpuAssert(cudaMalloc((void**)&d_index_out,     size * sizeof(I)));
    gpuAssert(cudaMalloc((void**)&d_token_out,     size * sizeof(token_t)));
    gpuAssert(cudaMalloc((void**)&d_state_states,  NUM_LOGICAL_BLOCKS * sizeof(State<state_t>)));
    gpuAssert(cudaMalloc((void**)&d_index_states,  NUM_LOGICAL_BLOCKS * sizeof(State<I>)));
    gpuAssert(cudaMalloc((void**)&d_dyn_index_ptr, sizeof(uint32_t)));
    gpuAssert(cudaMalloc((void**)&d_new_size,      sizeof(I)));
    gpuAssert(cudaMalloc((void**)&d_is_valid,      sizeof(bool)));

    gpuAssert(cudaMemcpy(d_in, input, size * sizeof(uint8_t), cudaMemcpyHostToDevice));
    gpuAssert(cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t)));
    gpuAssert(cudaMemset(d_is_valid, 0, sizeof(bool)));

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
    gpuAssert(cudaFree(d_state_states));
    gpuAssert(cudaFree(d_index_states));
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

    printf(PAD, "Lexer:");
    testLexer(input, input_size, expected_indices, expected_tokens, expected_indices_size);
    printf(PAD, "Lexer Shmem Compose:");
    testLexerShmemCompose(input, input_size, expected_indices, expected_tokens, expected_indices_size);
    printf(PAD, "Lexer Alpacc:");
    testLexerAlpacc(input, input_size, expected_indices, expected_tokens, expected_indices_size);
    printf(PAD, "Lexer Alpacc Shmem:");
    testLexerAlpaccShmem(input, input_size, expected_indices, expected_tokens, expected_indices_size);
    printf(PAD, "Lexer Alpacc Shmem Dyn BS512:");
    testLexerAlpaccShmemDyn<512, 64>(input, input_size, expected_indices, expected_tokens, expected_indices_size);
    printf(PAD, "Lexer Alpacc Shmem Dyn BS1024:");
    testLexerAlpaccShmemDyn<1024, 32>(input, input_size, expected_indices, expected_tokens, expected_indices_size);

    free(input);
    free(expected_indices);
    free(expected_tokens);
    gpuAssert(cudaPeekAtLastError());
    return 0;
}


#endif
