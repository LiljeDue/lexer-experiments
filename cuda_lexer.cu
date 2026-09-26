#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "common/sps.cu.h"
#include "common/util.cu.h"
#include "common/data.h"
#include <math.h>
#define PAD "%-38s "
// Apply minnctapersm=6 only on sm_80+ (A100). sm_75 has fewer registers
// and the hint would be out of range, producing a ptxas warning.
#if __CUDA_ARCH__ >= 800
#define LB_P1 __launch_bounds__(256, 6)
#define LB_BIG(BS) __launch_bounds__(BS, 1536 / (BS))   // lexerBig: 1536 threads/SM
#else
#define LB_P1 __launch_bounds__(256)
#define LB_BIG(BS) __launch_bounds__(BS)
#endif

using token_t = uint8_t;

// DFA: the parenthesis/identifier benchmark DFA below, or with
// -DLEXER_DFA_JSON alpacc's JSON lexer (dfa/json.h).
#ifdef LEXER_DFA_JSON
#include "dfa/json.h"
#else
using state_t = uint16_t;

const uint32_t NUM_STATES = 12;
const uint32_t NUM_TRANS = 256;
const state_t ENDO_MASK = 15;
const state_t ENDO_OFFSET = 0;
const state_t TOKEN_MASK = 112;
const state_t TOKEN_OFFSET = 4;
const state_t ACCEPT_MASK = 128;
const state_t ACCEPT_OFFSET = 7;
const state_t PRODUCE_MASK = 256;
const state_t PRODUCE_OFFSET = 8;
const state_t IDENTITY = 74;

constexpr state_t h_to_state[NUM_TRANS] =
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

constexpr state_t h_compose[NUM_STATES * NUM_STATES] =
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
constexpr token_t IGNORE_TOKEN = 0;          // whitespace
constexpr bool COMPOSE_LATER_MAJOR = true;   // compose(a, b) = h_compose[b * N + a]
#endif

// compose(a, b) (a then b) as an index into the compose table, in the DFA's
// table order.
__device__ __host__ __forceinline__ constexpr uint32_t compose_index(uint32_t ia, uint32_t ib) {
    return COMPOSE_LATER_MAJOR ? ib * NUM_STATES + ia : ia * NUM_STATES + ib;
}


__device__ __host__ __forceinline__ constexpr state_t get_index(state_t state) {
    return (state & ENDO_MASK) >> ENDO_OFFSET;
}

__device__ __host__ __forceinline__ constexpr token_t get_token(state_t state) {
    return (state & TOKEN_MASK) >> TOKEN_OFFSET;
}

#ifdef LEXER_ACCEPT_TABLE
// Acceptance is a table per endofunction (h_accept), copied to d_accept at
// program start; derived tables using it are then not compile-time.
__device__ bool d_accept[NUM_STATES];
#define LEXER_CX
__device__ __host__ __forceinline__ bool is_accept(state_t state) {
#ifdef __CUDA_ARCH__
    return d_accept[get_index(state)];
#else
    return h_accept[get_index(state)];
#endif
}
#else
#define LEXER_CX constexpr
__device__ __host__ __forceinline__ constexpr bool is_accept(state_t state) {
    return (state & ACCEPT_MASK) >> ACCEPT_OFFSET;
}
#endif

__device__ __host__ __forceinline__ constexpr bool is_produce(state_t state) {
    return (state & PRODUCE_MASK) >> PRODUCE_OFFSET;
}

// Copies N state_t values from global to shared memory, 8 bytes at a time
// when the table size allows it (any sizeof(state_t)).
template<uint32_t N, uint32_t BLOCK_SIZE>
__device__ __forceinline__ void copy_states_to_shared(state_t* dst, state_t* src) {
    if constexpr ((N * sizeof(state_t)) % 8 == 0) {
        for (uint32_t i = threadIdx.x; i < N * sizeof(state_t) / 8; i += BLOCK_SIZE)
            reinterpret_cast<volatile uint64_t*>(dst)[i] = reinterpret_cast<uint64_t*>(src)[i];
    } else {
        for (uint32_t i = threadIdx.x; i < N; i += BLOCK_SIZE)
            reinterpret_cast<volatile state_t*>(dst)[i] = src[i];
    }
}

// Packed chain state for lexerBig's per-byte loop: bit 0 = produce, bits 1-4 =
// state index * 2 (a byte offset into a u16 compose row), bits 5-7 = token,
// bit 8 = accept. The low byte is a complete "state byte" (all but accept).
// Built from the DFA's state encoding, so the tables derived with it stay
// runtime data.
__device__ __host__ __forceinline__ LEXER_CX uint16_t pack_chain_state(state_t s) {
    return uint16_t(uint32_t(is_produce(s)) | (get_index(s) << 1)
                  | (uint32_t(get_token(s)) << 5) | (uint32_t(is_accept(s)) << 8));
}
// The packed chain state / state byte hold the index in 4 bits and the token
// in 3; DFAs that do not fit use lexerBig's generic path (plain state_t
// compose chains), whatever the width of state_t.
constexpr uint32_t popcount_const(uint32_t x) { return x ? (x & 1u) + popcount_const(x >> 1) : 0; }
constexpr bool LEXER_CHAIN_FITS = NUM_STATES <= 16 && popcount_const(TOKEN_MASK) <= 3;

// Derived tables for lexerBig's fast path, built at compile time from the DFA
// tables (so the kernel only copies them; they stay data in global memory):
//   row_of8[byte] = offset of the byte's compose row in (1 << LEXER_ROW_SHIFT)
//                   byte units (8-byte units when compose rows are a multiple
//                   of 8 bytes, else 2-byte units: class * NUM_STATES <= 240);
//   comp[b * N + a] = compose(a, b) packed by pack_chain_state;
//   comp_pf[p * 16 + f] = state byte of compose(p, f) (one 16-byte row per p).
// Only sized for DFAs that fit the fast path.
constexpr uint32_t LEXER_ROW_SHIFT = (NUM_STATES * sizeof(uint16_t)) % 8 == 0 ? 3 : 1;
struct alignas(16) LexerChainTables {
    uint8_t  row_of8[256];
    uint16_t comp[LEXER_CHAIN_FITS ? NUM_STATES * NUM_STATES : 8];
    uint8_t  comp_pf[LEXER_CHAIN_FITS ? NUM_STATES * 16 : 16];
};
static_assert(sizeof(LexerChainTables) % 16 == 0, "copied as 16-byte vectors");
static_assert(offsetof(LexerChainTables, comp_pf) % 16 == 0, "comp_pf rows are read as uint4");
LEXER_CX LexerChainTables make_lexer_chain_tables() {
    LexerChainTables t{};
    if constexpr (LEXER_CHAIN_FITS) {
        for (uint32_t i = 0; i < 256; i++)
            t.row_of8[i] = uint8_t(get_index(h_to_state[i]) * ((NUM_STATES * sizeof(uint16_t)) >> LEXER_ROW_SHIFT));
        for (uint32_t b = 0; b < NUM_STATES; b++)
            for (uint32_t a = 0; a < NUM_STATES; a++)
                t.comp[b * NUM_STATES + a] = pack_chain_state(h_compose[compose_index(a, b)]);
        for (uint32_t i = 0; i < NUM_STATES * 16; i++) {
            const uint32_t p = i / 16, f = i % 16;
            t.comp_pf[i] = f < NUM_STATES ? uint8_t(pack_chain_state(h_compose[compose_index(p, f)])) : 0;
        }
    }
    return t;
}
LEXER_CX LexerChainTables h_chain_tables = make_lexer_chain_tables();

// alpacc's lexer output (lexerBig MAXADD): per kept token (terminal !=
// IGNORE_TOKEN, from the DFA) the terminal, its start and its length. Starts
// come from a max-scan over "last token end + 1" (0 = input start), output
// slots from an add-scan over kept tokens; both share one look-back round via
// MaxAdd.
struct MaxAdd {
    uint32_t max;   // start of the next token: last token end + 1 (0: none yet)
    uint32_t cnt;   // kept tokens
};
struct MaxAddOp {
    __device__ __forceinline__ MaxAdd operator()(const MaxAdd& a, const MaxAdd& b) const {
        return MaxAdd{a.max > b.max ? a.max : b.max, a.cnt + b.cnt};
    }
};
// One 64-bit tile descriptor: status in bits 1-0, max in bits 32-2, cnt in
// bits 63-33 (inputs and token counts < 2^31). Same size and zero layout as
// ScanTileState<uint32_t>, so an index tile state can be reused for it.
template<> struct TxnWordTraits<MaxAdd> {
    using TxnWord    = unsigned long long;
    using StatusWord = uint32_t;
    __device__ __forceinline__
    static TxnWord pack(StatusWord status, MaxAdd v) {
        return TxnWord(status) | (TxnWord(v.max) << 2) | (TxnWord(v.cnt) << 33);
    }
    __device__ __forceinline__
    static StatusWord unpack_status(TxnWord w) { return StatusWord(w & 3ull); }
    __device__ __forceinline__
    static MaxAdd unpack_value(TxnWord w) {
        return MaxAdd{uint32_t((w >> 2) & 0x7fffffffull), uint32_t(w >> 33)};
    }
};

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
        if constexpr (COMPOSE_LATER_MAJOR) return d_compose[get_index(b) * NUM_STATES + get_index(a)];
        else                            return d_compose[get_index(a) * NUM_STATES + get_index(b)];
    }

    __device__ __host__ __forceinline__
    state_t operator()(const volatile state_t &a, const volatile state_t &b) const {
        if constexpr (COMPOSE_LATER_MAJOR) return d_compose[get_index(b) * NUM_STATES + get_index(a)];
        else                            return d_compose[get_index(a) * NUM_STATES + get_index(b)];
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
    LexerChainTables* d_chain_tables;   // lexerBig's derived tables (h_chain_tables)
    // Step table for lexerBig's generic path (STEPTBL), derived at run time
    // from compose and to_state: bytes with the same to_state endofunction
    // form a class; step[s * num_classes + c] = compose(s, class c's function),
    // so the per-byte step needs NUM_STATES x classes entries instead of the
    // full NUM_STATES x NUM_STATES compose table.
    uint8_t* d_byte_class;
    state_t* d_step;
    uint32_t num_classes;

    LexerCtxShmem() : d_to_state(NULL), d_compose_glb(NULL), d_compose(NULL), d_chain_tables(NULL),
                      d_byte_class(NULL), d_step(NULL), num_classes(0) {
        cudaMalloc(&d_to_state, sizeof(h_to_state));
        cudaMemcpy(d_to_state, h_to_state, sizeof(h_to_state),
                cudaMemcpyHostToDevice);
        cudaMalloc(&d_compose_glb, sizeof(h_compose));
        cudaMemcpy(d_compose_glb, h_compose, sizeof(h_compose),
                cudaMemcpyHostToDevice);
        cudaMalloc(&d_chain_tables, sizeof(h_chain_tables));
        cudaMemcpy(d_chain_tables, &h_chain_tables, sizeof(h_chain_tables),
                cudaMemcpyHostToDevice);

        std::vector<uint8_t> byte_class(256);
        std::vector<uint32_t> class_endo;   // endofunction index of each class
        for (uint32_t b = 0; b < 256; b++) {
            const uint32_t e = get_index(h_to_state[b]);
            uint32_t c = 0;
            while (c < class_endo.size() && class_endo[c] != e) c++;
            if (c == class_endo.size()) class_endo.push_back(e);
            byte_class[b] = uint8_t(c);
        }
        num_classes = class_endo.size();
        std::vector<state_t> step((size_t)NUM_STATES * num_classes);
        for (uint32_t s = 0; s < NUM_STATES; s++)
            for (uint32_t c = 0; c < num_classes; c++)
                step[(size_t)s * num_classes + c] = h_compose[compose_index(s, class_endo[c])];
        cudaMalloc(&d_byte_class, 256);
        cudaMemcpy(d_byte_class, byte_class.data(), 256, cudaMemcpyHostToDevice);
        cudaMalloc(&d_step, step.size() * sizeof(state_t));
        cudaMemcpy(d_step, step.data(), step.size() * sizeof(state_t), cudaMemcpyHostToDevice);
    }

    void Cleanup() {
        if (d_to_state) cudaFree(d_to_state);
        if (d_compose_glb) cudaFree(d_compose_glb);
        if (d_chain_tables) cudaFree(d_chain_tables);
        if (d_byte_class) cudaFree(d_byte_class);
        if (d_step) cudaFree(d_step);
    }

    __device__ __forceinline__
    state_t operator()(const state_t &a, const state_t &b) const {
        if constexpr (COMPOSE_LATER_MAJOR) return d_compose[get_index(b) * NUM_STATES + get_index(a)];
        else                            return d_compose[get_index(a) * NUM_STATES + get_index(b)];
    }

    __device__ __forceinline__
    state_t operator()(const volatile state_t &a, const volatile state_t &b) const {
        if constexpr (COMPOSE_LATER_MAJOR) return d_compose[get_index(b) * NUM_STATES + get_index(a)];
        else                            return d_compose[get_index(a) * NUM_STATES + get_index(b)];
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

// Loads ITEMS_PER_THREAD bytes per thread from d_in[glb_offs..] using 64-bit
// coalesced loads, maps each byte through to_state[], writes states to shmem
// in sequential layout. Out-of-bounds positions get `identity`.
// If EXTRA=1, also stores the one byte at position glb_offs+TILE into *next_state.
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

// Single-pass lexer with BLOCK_LOAD_WARP_TRANSPOSE state scan.
// Uses static blockIdx.x for both the state scan and index scan tile indices —
// no dynamic index counter. The WARP_TRANSPOSE load eliminates shmem store bank
// conflicts and reduces instruction count vs the manual u64 blocked-load path.
//
// After the state scan, st[] is written to a flat persistent shmem array for
// produce-detection (states[lid+1] lookups). The union covers BlockLoad temp
// storage + state scan storage + index scan storage (used at different times).
template<typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__ LB_P1
void lexerTranspose(
    LexerCtxShmem ctx,
    uint8_t* d_in,
    uint32_t* d_index_out,
    token_t* d_token_out,
    ScanTileState<state_t> state_states,
    ScanTileState<I> index_states,
    I size,
    I num_logical_blocks,
    volatile I* new_size,
    volatile bool* is_valid)
{
    static_assert(ITEMS_PER_THREAD <= 64, "ITEMS_PER_THREAD exceeds 64-bit is_produce_state capacity");
    struct ByteToStateFn {
        state_t* d_to_state;
        __device__ __forceinline__ state_t operator()(uint8_t b) const { return d_to_state[b]; }
    };
    using TransformIter   = cub::TransformInputIterator<state_t, ByteToStateFn, const uint8_t*>;
    using BlockLoadT      = cub::BlockLoad<state_t, BLOCK_SIZE, ITEMS_PER_THREAD,
                                           cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockScanState  = cub::BlockScan<state_t, BLOCK_SIZE, cub::BLOCK_SCAN_WARP_SCANS>;
    using BlockScanI      = cub::BlockScan<I, BLOCK_SIZE>;
    using PrefixOpState   = TilePrefixCallbackOp<state_t, LexerCtxShmem>;
    using PrefixOpIdx     = TilePrefixCallbackOp<I, Add<I>>;

    __shared__ union {
        typename BlockLoadT::TempStorage load;
        struct {
            typename PrefixOpState::TempStorage state_prefix;
            typename BlockScanState::TempStorage state_scan;
        } state_scan_storage;
        struct {
            typename PrefixOpIdx::TempStorage idx_prefix;
            typename BlockScanI::TempStorage  idx_scan;
        } idx_scan_storage;
    } temp;

    __shared__ __align__(8) state_t shmem_compose[NUM_STATES * NUM_STATES];
    __shared__ __align__(8) state_t shmem_to_state[256];
    __shared__ state_t states[ITEMS_PER_THREAD * BLOCK_SIZE];
    __shared__ uint8_t tok_stage[ITEMS_PER_THREAD * BLOCK_SIZE];
    __shared__ state_t next_block_first_state;

    // Load tables into shmem (can share the load union slot, tables are separate).
    copy_states_to_shared<NUM_STATES * NUM_STATES, BLOCK_SIZE>(shmem_compose, ctx.d_compose_glb);
    copy_states_to_shared<256, BLOCK_SIZE>(shmem_to_state, ctx.d_to_state);
    ctx.d_compose = shmem_compose;

    if (threadIdx.x == 0)
        next_block_first_state = IDENTITY;
    __syncthreads();

    I tile_idx = (I) blockIdx.x;
    I glb_offs = tile_idx * BLOCK_SIZE * ITEMS_PER_THREAD;
    I valid    = (I) min((uint64_t)BLOCK_SIZE * ITEMS_PER_THREAD,
                         (uint64_t)size - glb_offs);

    ByteToStateFn byte_to_state{shmem_to_state};
    TransformIter d_in_states(d_in + glb_offs, byte_to_state);
    state_t st[ITEMS_PER_THREAD];
    if (glb_offs + BLOCK_SIZE * ITEMS_PER_THREAD <= size)
        BlockLoadT(temp.load).Load(d_in_states, st);
    else
        BlockLoadT(temp.load).Load(d_in_states, st, valid, IDENTITY);
    __syncthreads();  // transition union: load → state_scan_storage

    // Load the extra byte after this tile for produce-detection at the tile boundary.
    if (threadIdx.x == 0) {
        I boundary = glb_offs + BLOCK_SIZE * ITEMS_PER_THREAD;
        if (boundary < size)
            next_block_first_state = shmem_to_state[d_in[boundary]];
    }

    if (tile_idx == 0) {
        state_t block_aggregate;
        BlockScanState(temp.state_scan_storage.state_scan)
            .InclusiveScan(st, st, ctx, block_aggregate);
        if (threadIdx.x == 0)
            state_states.SetInclusive(0, block_aggregate);
    } else {
        PrefixOpState state_prefix_op(state_states, temp.state_scan_storage.state_prefix,
                                      ctx, (int)tile_idx, state_t(IDENTITY));
        BlockScanState(temp.state_scan_storage.state_scan)
            .InclusiveScan(st, st, ctx, state_prefix_op);
    }

    // Write scanned states to flat shmem for produce-detection lookups.
    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++)
        states[threadIdx.x * ITEMS_PER_THREAD + i] = st[i];
    __syncthreads();  // transition union: state_scan_storage → idx_scan_storage

    I prod[ITEMS_PER_THREAD];
    uint64_t is_produce_state = 0;
    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = threadIdx.x * ITEMS_PER_THREAD + i;
        I gid = glb_offs + lid;
        bool temp_flag = false;
        if (gid < size) {
            if (lid == ITEMS_PER_THREAD * BLOCK_SIZE - 1) {
                temp_flag = gid == size - 1 || is_produce(ctx(st[i], next_block_first_state));
            } else {
                temp_flag = gid == size - 1 || is_produce(states[lid + 1]);
            }
        }
        is_produce_state |= (uint64_t)temp_flag << i;
        prod[i] = (I)temp_flag;
    }

    PrefixOpIdx index_prefix_op(index_states, temp.idx_scan_storage.idx_prefix,
                                 Add<I>(), (int)tile_idx, I(0));
    BlockScanI(temp.idx_scan_storage.idx_scan)
        .InclusiveScan(prod, prod, Add<I>(), index_prefix_op);
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

    if (tile_idx == num_logical_blocks - 1 && threadIdx.x == BLOCK_SIZE - 1) {
        *new_size = Add<I>()(idx_pfx, prod_agg);
        *is_valid = is_accept(st[ITEMS_PER_THREAD - 1]);
    }
}


// ---------------------------------------------------------------------------
// cp.async helpers (sm_80+): 16-byte global -> shared copies that bypass
// registers and L1. Older architectures fall back to a synchronous copy.
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
// Single-pass lexer with the P1 optimisations from p1_bench.cu (p1_vec_pipe):
//  - persistent grid (at most the co-resident blocks; the lookbacks spin on
//    predecessor tiles), each block prefetching its next tile with cp.async
//    into a second shmem buffer while it works on the current one;
//  - warp-local vectorized load: each warp owns WARP * ITEMS_PER_THREAD
//    consecutive bytes, lanes read their own ITEMS_PER_THREAD bytes back with
//    LDS.64 (bank-conflict-free at ITEMS_PER_THREAD = 24);
//  - relaxed tile-state publication (no __threadfence) for both lookbacks;
//  - produce detection from registers: the next position's state is the
//    thread's own next item, the next lane's first item (shuffle) or the next
//    warp's first item (shmem), instead of a flat shmem copy of all states;
//  - index scan over per-thread produce counts (one item per thread).
// Output staging for coalesced index/token writes reuses the consumed input
// buffer (tokens) plus a u16 buffer (positions within the tile).
// ---------------------------------------------------------------------------
// Position of the k-th (0-based) set bit of m, k < popc(m): branchless binary
// search with popc on halves, quarters, ... (~25 instructions).
__device__ __forceinline__ uint32_t select_bit(uint32_t m, uint32_t k) {
    uint32_t pos = 0, c;
    c = __popc(m & 0xffffu); if (k >= c) { k -= c; m >>= 16; pos += 16; }
    c = __popc(m & 0xffu);   if (k >= c) { k -= c; m >>= 8;  pos += 8;  }
    c = __popc(m & 0xfu);    if (k >= c) { k -= c; m >>= 4;  pos += 4;  }
    c = __popc(m & 0x3u);    if (k >= c) { k -= c; m >>= 2;  pos += 2;  }
    c = m & 0x1u;            if (k >= c) {                   pos += 1;  }
    return pos;
}


__device__ __forceinline__ state_t  chain_state_index(uint32_t v) { return state_t((v & 0x1eu) >> 1); }
__device__ __forceinline__ bool     chain_produce(uint32_t v)     { return v & 1u; }
__device__ __forceinline__ uint32_t chain_token(uint32_t v)       { return (v >> 5) & 7u; }
__device__ __forceinline__ bool     chain_accept(uint32_t v)      { return (v >> 8) & 1u; }

// Compose functor over the block's shared-memory table (one pointer, instead of
// the three-pointer LexerCtxShmem, to keep register pressure down).
struct ShmemCompose {
    const state_t* table;
    __device__ __forceinline__ state_t operator()(const state_t& a, const state_t& b) const {
        if constexpr (COMPOSE_LATER_MAJOR) return table[get_index(b) * NUM_STATES + get_index(a)];
        else                            return table[get_index(a) * NUM_STATES + get_index(b)];
    }
};

template<typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__ LB_P1
void lexerVecPipe(
    LexerCtxShmem ctx,
    const uint8_t* __restrict__ d_in,
    uint32_t* d_index_out,
    token_t* d_token_out,
    ScanTileState<state_t> state_states,
    ScanTileState<I> index_states,
    I size,
    I num_tiles,
    volatile I* new_size,
    volatile bool* is_valid)
{
    static_assert(ITEMS_PER_THREAD % 8 == 0, "ITEMS_PER_THREAD must be a multiple of 8");
    static_assert(ITEMS_PER_THREAD <= 32, "produce flags are kept in a 32-bit mask");
    constexpr I WARPS      = BLOCK_SIZE / WARP;
    constexpr I VECS       = ITEMS_PER_THREAD / 8;
    constexpr I WARP_ITEMS = WARP * ITEMS_PER_THREAD;
    constexpr I TILE       = BLOCK_SIZE * ITEMS_PER_THREAD;
    constexpr I CHUNKS     = WARP_ITEMS / 16;
    static_assert(WARP_ITEMS % 16 == 0, "warp segment must be a multiple of 16 bytes");
    static_assert(TILE <= 65536, "tile positions are staged as uint16_t");

    using BlockScanState = cub::BlockScan<state_t, BLOCK_SIZE, cub::BLOCK_SCAN_WARP_SCANS>;
    using BlockScanI     = cub::BlockScan<I, BLOCK_SIZE, cub::BLOCK_SCAN_WARP_SCANS>;
    using PrefixOpState  = TilePrefixCallbackOp<state_t, ShmemCompose, true>;
    using PrefixOpIdx    = TilePrefixCallbackOp<I, Add<I>, true>;

    // Input bytes (double-buffered; warp w owns [w * WARP_ITEMS, ...)). After a
    // tile's bytes are in registers its buffer stages that tile's tokens.
    __shared__ __align__(16) uint8_t  inbuf[2][TILE];
    __shared__ __align__(16) uint16_t lid_stage[TILE];
    __shared__ typename BlockScanState::TempStorage state_scan;
    __shared__ typename PrefixOpState::TempStorage  state_prefix;
    __shared__ typename BlockScanI::TempStorage     idx_scan;
    __shared__ typename PrefixOpIdx::TempStorage    idx_prefix;
    __shared__ state_t warp_first[WARPS];
    __shared__ state_t next_tile_first;
    __shared__ __align__(8) state_t shmem_compose[NUM_STATES * NUM_STATES];
    __shared__ __align__(8) state_t shmem_to_state[256];

    copy_states_to_shared<NUM_STATES * NUM_STATES, BLOCK_SIZE>(shmem_compose, ctx.d_compose_glb);
    copy_states_to_shared<256, BLOCK_SIZE>(shmem_to_state, ctx.d_to_state);
    const ShmemCompose compose{shmem_compose};
    __syncthreads();

    const I warp = threadIdx.x / WARP;
    const I lane = threadIdx.x % WARP;

    // Issues this warp's share of tile's input into inbuf[buf] (full tiles
    // only; the partial last tile is read directly), then commits a group.
    auto prefetch = [&](I tile, I buf) {
        if (tile < num_tiles && tile * TILE + TILE <= size) {
            const uint8_t* src = d_in + tile * TILE + warp * WARP_ITEMS;
            uint8_t*       dst = inbuf[buf] + warp * WARP_ITEMS;
            for (I c = lane; c < CHUNKS; c += WARP)
                cp_async16(dst + 16 * c, src + 16 * c);
        }
        cp_async_commit();
    };

    I buf = 0;
    prefetch(blockIdx.x, buf);
    for (I tile = blockIdx.x; tile < num_tiles; tile += gridDim.x, buf ^= 1) {
        prefetch(tile + gridDim.x, buf ^ 1);
        cp_async_wait<1>();   // this tile's group has landed
        __syncwarp();

        const I glb_offs = tile * TILE;
        const bool full  = glb_offs + TILE <= size;
        if (threadIdx.x == 0) {
            I boundary = glb_offs + TILE;
            next_tile_first = boundary < size ? shmem_to_state[d_in[boundary]] : state_t(IDENTITY);
        }

        state_t st[ITEMS_PER_THREAD];
        if (full) {
            const uint2* in = reinterpret_cast<const uint2*>(inbuf[buf] + warp * WARP_ITEMS);
            #pragma unroll
            for (I k = 0; k < VECS; k++) {
                uint2 w = in[lane * VECS + k];
                #pragma unroll
                for (I b = 0; b < 8; b++) {
                    uint32_t word = b < 4 ? w.x : w.y;
                    st[8 * k + b] = shmem_to_state[(word >> (8 * (b % 4))) & 0xffu];
                }
            }
        } else {
            #pragma unroll
            for (I i = 0; i < ITEMS_PER_THREAD; i++) {
                I idx = glb_offs + threadIdx.x * ITEMS_PER_THREAD + i;
                st[i] = idx < size ? shmem_to_state[d_in[idx]] : state_t(IDENTITY);
            }
        }

        // State scan (lookback 1).
        PrefixOpState state_op(state_states, state_prefix, compose, (int)tile, state_t(IDENTITY));
        BlockScanState(state_scan).InclusiveScan(st, st, compose, state_op);

        // The state after each position: own next item, next lane's first
        // item, next warp's first item, or (end of tile) the next tile's first
        // byte composed onto this tile's last state.
        if (lane == 0)
            warp_first[warp] = st[0];
        __syncthreads();   // warp_first / next_tile_first; this tile's input bytes consumed
        state_t next = (state_t)__shfl_down_sync(0xffffffff, (uint32_t)st[0], 1);
        if (lane == WARP - 1)
            next = warp + 1 < WARPS ? warp_first[warp + 1] : compose(st[ITEMS_PER_THREAD - 1], next_tile_first);

        // Produce flag and 4-bit token per position, in one pass so each
        // state can die once used; only the flags, tokens and last state stay
        // live through the index scan and its lookback.
        const state_t last_state = st[ITEMS_PER_THREAD - 1];
        // Valid items of this thread (the input ends inside the last one).
        const I       tid_base   = glb_offs + threadIdx.x * ITEMS_PER_THREAD;
        const I       valid      = tid_base < size ? size - tid_base : 0;
        uint32_t prod_bits = 0;
        uint32_t tok[VECS] = {};
        #pragma unroll
        for (I i = 0; i < ITEMS_PER_THREAD; i++) {
            state_t nxt = i + 1 < ITEMS_PER_THREAD ? st[i + 1] : next;
            bool produce = i < valid && (i + 1 == valid || is_produce(nxt));
            prod_bits |= uint32_t(produce) << i;
            tok[i / 8] |= uint32_t(get_token(st[i])) << (4 * (i % 8));
        }

        // Index scan (lookback 2) over per-thread produce counts.
        I thread_offs;
        PrefixOpIdx idx_op(index_states, idx_prefix, Add<I>(), (int)tile, I(0));
        BlockScanI(idx_scan).ExclusiveScan((I)__popc(prod_bits), thread_offs, Add<I>(), idx_op);
        const I idx_pfx  = idx_op.GetExclusivePrefix();
        const I prod_agg = idx_op.GetBlockAggregate();

        // Stage (token, position) by output slot, then write coalesced.
        uint8_t* tok_stage = inbuf[buf];
        I slot = thread_offs - idx_pfx;
        #pragma unroll
        for (I i = 0; i < ITEMS_PER_THREAD; i++) {
            if ((prod_bits >> i) & 1) {
                tok_stage[slot] = (tok[i / 8] >> (4 * (i % 8))) & 0xfu;
                lid_stage[slot] = (uint16_t)(threadIdx.x * ITEMS_PER_THREAD + i);
                slot++;
            }
        }
        __syncthreads();
        for (I s = threadIdx.x; s < prod_agg; s += BLOCK_SIZE) {
            d_index_out[idx_pfx + s] = glb_offs + lid_stage[s];
            d_token_out[idx_pfx + s] = tok_stage[s];
        }

        if (tile == num_tiles - 1 && threadIdx.x == BLOCK_SIZE - 1) {
            *new_size = idx_pfx + prod_agg;
            *is_valid = is_accept(last_state);
        }
        // Staging buffers, scan / prefix temp storage and warp_first are
        // reused next tile; inbuf[buf] is the target of the next prefetch.
        __syncthreads();
    }
    cp_async_wait<0>();
}

// ---------------------------------------------------------------------------
// Large-tile single-pass lexer.
//
// A block's tile is BLOCK_SIZE * CHUNK input bytes (24 KB at 256 x 96), kept
// in shared memory; thread t owns the CHUNK contiguous bytes
// [t * CHUNK, (t + 1) * CHUNK). Per tile:
//   A. per-thread state reduction over its chunk (chain from IDENTITY), each
//      prefix F_i stored in place over the input as a state byte; block
//      exclusive scan of the thread aggregates with the state look-back ->
//      each thread's incoming state p;
//   B. state i = compose(p, F_i), looked up independently per byte (no
//      serial chain) in the prefix's comp_pf row, held in 4 registers and
//      indexed with PRMT; state bytes written in place, produce flags
//      gathered 4 bytes at a time (multiply-shift) into a CHUNK-bit register
//      mask; block exclusive scan of the per-thread token counts with the
//      index look-back -> each thread's first output slot;
//   C. warp-cooperative emission: each warp writes 32 consecutive output
//      slots per step (coalesced). Lane r finds the lane owning slot r by a
//      binary search over the lanes' inclusive counts and the element by a
//      k-th-set-bit select in that lane's mask, then reads its token (state
//      byte >> 5) from shared memory.
// Pass A steps through derived tables (row_of8, comp) built at kernel start
// from the DFA tables: one chain step is an AND, an add and one shared load
// on a packed state (pack_chain_state), instead of two table lookups plus the
// index extraction of compose(s, to_state[byte]). row_of8 is a byte table,
// so ASCII input bytes hit different banks.
// Each chunk's 16-byte vectors are swizzled in shared memory (vector k of
// thread t at slot k ^ ((t >> 2) & 1)), which removes the 2-way bank
// conflicts of the stride-96 vector accesses.
// Pass B requires that each state index has one full state value in the
// compose table (flags a function of the index), as for this DFA; otherwise
// compose(p, F_i) could differ in its flags from the chain's state i.
// DFAs that do not fit the packed formats (LEXER_CHAIN_FITS: <= 16 states,
// <= 8 tokens), or FORCE_GENERIC, use the generic path: passes A and B as
// plain state_t compose chains, tokens stored as bytes.
// The next position's state for the chunk's last element comes from the next
// thread's first byte (read before pass A overwrites the chunks) or, for the
// block's last thread, the next tile's first byte from global memory.
//
// Ladder (STEP): 1 = load only (bytes XOR-reduced to a sink); 2 = passes A-C
// and output without look-backs (tile-local states, each tile writes its
// outputs to its own region; output is not valid); 3 = full lexer.
// ---------------------------------------------------------------------------
template<typename I, I BLOCK_SIZE, I CHUNK, I STEP, bool FORCE_GENERIC = false, bool MAXADD = false,
         bool STEPTBL = false>
__global__ LB_BIG(BLOCK_SIZE)
void lexerBig(
    LexerCtxShmem ctx,
    const uint8_t* __restrict__ d_in,
    uint32_t* d_index_out,   // token end positions (MAXADD: token starts)
    token_t* d_token_out,
    uint32_t* d_len_out,     // MAXADD: token lengths (unused otherwise)
    ScanTileState<state_t> state_states,
    ScanTileState<I> index_states,
    I size,
    I num_tiles,
    volatile I* new_size,
    volatile bool* is_valid)
{
    static_assert(STEP >= 1 && STEP <= 3, "STEP must be 1, 2 or 3");
    static_assert(CHUNK % 32 == 0 && CHUNK <= 96,
                  "CHUNK: multiple of 32 (swizzled chunk offsets keep bit 4 free), at most 96 (3-word mask)");
    // FAST: packed chain states and state bytes (needs LEXER_CHAIN_FITS);
    // otherwise the generic path (state_t compose chains in passes A and B).
    constexpr bool FAST = LEXER_CHAIN_FITS && !FORCE_GENERIC;
    // MAXADD: alpacc's output (terminal, start, length of non-ignored tokens).
    static_assert(!MAXADD || FAST, "MAXADD is implemented on the fast path");
    // STEPTBL: the generic path steps through ctx.d_step (endofunction x byte
    // class) instead of compose(s, to_state[byte]); scans and look-backs keep
    // the full compose table.
    static_assert(!STEPTBL || !FAST, "the step table is for the generic path");
    constexpr uint32_t ROW_SHIFT = LEXER_ROW_SHIFT;
    constexpr I VECS       = CHUNK / 16;
    constexpr I WARP_BYTES = WARP * CHUNK;
    constexpr I TILE       = BLOCK_SIZE * CHUNK;

    using BlockScanState = cub::BlockScan<state_t, BLOCK_SIZE, cub::BLOCK_SCAN_WARP_SCANS>;
    using BlockScanI     = cub::BlockScan<I, BLOCK_SIZE, cub::BLOCK_SCAN_WARP_SCANS>;
    using PrefixOpState  = TilePrefixCallbackOp<state_t, ShmemCompose, true>;
    using PrefixOpIdx    = TilePrefixCallbackOp<I, Add<I>, true>;
    using BlockScanMA    = cub::BlockScan<MaxAdd, BLOCK_SIZE, cub::BLOCK_SCAN_WARP_SCANS>;   // MAXADD
    using PrefixOpMA     = TilePrefixCallbackOp<MaxAdd, MaxAddOp, true>;

    __shared__ __align__(16) uint8_t bytes[TILE];   // input bytes, then F_i, then state bytes
    __shared__ typename BlockScanState::TempStorage state_scan;
    __shared__ typename PrefixOpState::TempStorage  state_prefix;
    __shared__ typename BlockScanI::TempStorage     idx_scan;
    __shared__ typename PrefixOpIdx::TempStorage    idx_prefix;
    __shared__ typename BlockScanMA::TempStorage    ma_scan;     // MAXADD (unreferenced otherwise)
    __shared__ typename PrefixOpMA::TempStorage     ma_prefix;
    // The compose table in shared memory when it fits (up to 8 KB), else it is
    // read from global memory through L1/L2 (large DFAs: e.g. JSON, 823
    // endofunctions, 1.35 MB).
    constexpr bool COMPOSE_IN_SHMEM = NUM_STATES * NUM_STATES * sizeof(state_t) <= 8192;
    __shared__ __align__(8) state_t shmem_compose[COMPOSE_IN_SHMEM ? NUM_STATES * NUM_STATES : 4];
    // Derived tables (LexerChainTables, built at compile time from the DFA
    // tables and copied here as 16-byte vectors): one chain step is
    // v = comp[(row_of8[byte] << ROW_SHIFT) + (v & 0x1e)] (bytes; row_of8 is a
    // byte table, so bytes < 128 all fall in different banks); comp_pf is
    // read as one 16-byte row per thread.
    __shared__ LexerChainTables tables;
    auto& row_of8 = tables.row_of8;
    auto& comp    = tables.comp;
    auto& comp_pf = tables.comp_pf;
    // Generic path: to_state as loaded, or with STEPTBL the byte classes
    // (unreferenced otherwise).
    __shared__ __align__(8) state_t shmem_to_state[256];
    __shared__ __align__(16) uint8_t shmem_class[256];

    if constexpr (COMPOSE_IN_SHMEM)
        copy_states_to_shared<NUM_STATES * NUM_STATES, BLOCK_SIZE>(shmem_compose, ctx.d_compose_glb);
    if constexpr (!FAST) {
        if constexpr (STEPTBL) {
            for (uint32_t i = threadIdx.x; i < 256 / 16; i += BLOCK_SIZE)
                reinterpret_cast<uint4*>(shmem_class)[i] =
                    reinterpret_cast<const uint4*>(ctx.d_byte_class)[i];
        } else {
            copy_states_to_shared<256, BLOCK_SIZE>(shmem_to_state, ctx.d_to_state);
        }
    } else {
        for (uint32_t i = threadIdx.x; i < sizeof(LexerChainTables) / 16; i += BLOCK_SIZE)
            reinterpret_cast<uint4*>(&tables)[i] =
                reinterpret_cast<const uint4*>(ctx.d_chain_tables)[i];
    }
    const ShmemCompose compose{COMPOSE_IN_SHMEM ? shmem_compose : ctx.d_compose_glb};
    // Generic path: one byte step from state s.
    auto gstep = [&](state_t s, uint32_t byte) -> state_t {
        if constexpr (STEPTBL)
            return __ldg(&ctx.d_step[get_index(s) * ctx.num_classes + shmem_class[byte]]);
        else
            return compose(s, shmem_to_state[byte]);
    };
    auto step = [&](uint32_t v, uint32_t byte) -> uint32_t {
        return *reinterpret_cast<const uint16_t*>(
            reinterpret_cast<const uint8_t*>(comp) + (uint32_t(row_of8[byte]) << ROW_SHIFT) + (v & 0x1eu));
    };

    const I tile      = blockIdx.x;
    const I warp      = threadIdx.x / WARP;
    const I lane      = threadIdx.x % WARP;
    const I tile_offs = tile * TILE;
    const bool full   = tile_offs + TILE <= size;
    const I my_offs   = tile_offs + threadIdx.x * CHUNK;          // first byte of my chunk
    const I valid     = my_offs < size ? min(size - my_offs, CHUNK) : 0;
    uint8_t* my_bytes = bytes + threadIdx.x * CHUNK;
    // Swizzle: vector k of thread t's chunk is stored at slot k ^ ((t >> 2) & 1), so
    // the 8 lanes of each 16-byte access phase (chunk stride 96 bytes) hit
    // different bank groups; byte offsets within a chunk flip bit 4.
    const I sw = (threadIdx.x >> 2) & 1;

    // Load the tile: coalesced 16-byte vectors over each warp's segment.
    if (full) {
        const uint4* src = reinterpret_cast<const uint4*>(d_in + tile_offs + warp * WARP_BYTES);
        uint4*       dst = reinterpret_cast<uint4*>(bytes + warp * WARP_BYTES);
        #pragma unroll
        for (I k = 0; k < VECS; k++)
            {
                const I v = lane + k * WARP;   // vector of the warp's segment
                dst[v ^ (((v / VECS) >> 2) & 1)] = src[v];
            }
    } else {
        for (I i = 0; i < CHUNK; i++)
            my_bytes[i ^ (sw << 4)] = i < valid ? d_in[my_offs + i] : 0;
    }
    __syncthreads();   // tables and all chunks (the next thread's first byte) loaded

    if constexpr (STEP == 1) {
        uint32_t x = 0;
        const uint4* my = reinterpret_cast<const uint4*>(my_bytes);
        #pragma unroll
        for (I k = 0; k < VECS; k++) { uint4 v = my[k]; x ^= v.x ^ v.y ^ v.z ^ v.w; }
        if (x == 0x9e3779b9u)   // practically never: keeps the loads live
            d_token_out[my_offs] = (token_t)x;
        return;
    } else {
        // Byte after my chunk: next thread's first byte, or the next tile's.
        const I next_gid  = my_offs + CHUNK;
        const bool has_nb = next_gid < size && valid == CHUNK;
        const uint8_t nb  = !has_nb ? 0
                          : threadIdx.x + 1 < BLOCK_SIZE
                            ? my_bytes[CHUNK + ((((threadIdx.x + 1) >> 2) & 1) << 4)]
                          : d_in[next_gid];
        __syncthreads();   // next bytes read before pass A overwrites the chunks

        // Pass A: per-thread state reduction (packed chain); each prefix F_i
        // (state byte of the chain from IDENTITY) is stored in place over the
        // input.
        // Generic path: plain state_t chain, input bytes left in place.
        uint4* my = reinterpret_cast<uint4*>(my_bytes);
        uint32_t va = pack_chain_state(state_t(IDENTITY));
        state_t  ga = IDENTITY;
        if constexpr (FAST) {
            #pragma unroll
            for (I k = 0; k < VECS; k++) {
                const uint4 v = my[k ^ sw];
                uint32_t tw[4] = {0, 0, 0, 0};
                #pragma unroll
                for (I b = 0; b < 16; b++) {
                    const uint32_t word = b < 4 ? v.x : b < 8 ? v.y : b < 12 ? v.z : v.w;
                    const uint32_t byte = (word >> (8 * (b % 4))) & 0xffu;
                    if (full || 16 * k + b < valid) {
                        va = step(va, byte);
                        tw[b / 4] |= (va & 0xffu) << (8 * (b % 4));
                    }
                }
                my[k ^ sw] = make_uint4(tw[0], tw[1], tw[2], tw[3]);
            }
        } else {
            #pragma unroll
            for (I k = 0; k < VECS; k++) {
                const uint4 v = my[k ^ sw];
                #pragma unroll
                for (I b = 0; b < 16; b++) {
                    const uint32_t word = b < 4 ? v.x : b < 8 ? v.y : b < 12 ? v.z : v.w;
                    const uint32_t byte = (word >> (8 * (b % 4))) & 0xffu;
                    if (full || 16 * k + b < valid)
                        ga = gstep(ga, byte);
                }
            }
        }
        // Scans and look-back only need the state index (compose masks it).
        const state_t agg = FAST ? chain_state_index(va) : ga;

        state_t prefix;
        if constexpr (STEP == 3) {
            PrefixOpState state_op(state_states, state_prefix, compose, (int)tile, state_t(IDENTITY));
            BlockScanState(state_scan).ExclusiveScan(agg, prefix, compose, state_op);
        } else {
            state_t tile_agg;
            BlockScanState(state_scan).ExclusiveScan(agg, prefix, state_t(IDENTITY), compose, tile_agg);
        }

        // Pass B: state i = compose(prefix, F_i), looked up independently (no
        // serial chain), four per word: byte j of idx is the comp_pf index of
        // F_{4q+j} under my prefix. State bytes (produce bit 0, token bits
        // 5-7) are written in place; produce flags are gathered 4 bytes at a
        // time (p bit i = state i produces).
        // Generic path: rescan with the state_t chain from the incoming state,
        // per-byte produce test, tokens written in place.
        uint32_t m0, m1, m2;
        uint32_t n0 = 0, n1 = 0, n2 = 0;   // MAXADD: bit j = token of state j != IGNORE_TOKEN
        state_t  last_g = prefix;   // generic: state after my last valid byte
        if constexpr (!FAST) {
            m0 = m1 = m2 = 0;
            state_t st = prefix;
            #pragma unroll
            for (I k = 0; k < VECS; k++) {
                const uint4 v = my[k ^ sw];
                uint32_t tw[4] = {0, 0, 0, 0};
                #pragma unroll
                for (I b = 0; b < 16; b++) {
                    const I i = 16 * k + b;
                    const uint32_t word = b < 4 ? v.x : b < 8 ? v.y : b < 12 ? v.z : v.w;
                    const uint32_t byte = (word >> (8 * (b % 4))) & 0xffu;
                    if (full || i < valid) {
                        st = gstep(st, byte);
                        // element i-1 produces if the state after it (st) does
                        if (i > 0 && is_produce(st)) {
                            if (i - 1 < 32)      m0 |= 1u << (i - 1);
                            else if (i - 1 < 64) m1 |= 1u << (i - 1 - 32);
                            else                 m2 |= 1u << (i - 1 - 64);
                        }
                        tw[b / 4] |= uint32_t(uint8_t(get_token(st))) << (8 * (b % 4));
                        last_g = st;
                    }
                }
                my[k ^ sw] = make_uint4(tw[0], tw[1], tw[2], tw[3]);
            }
        } else {
        // My prefix's 16-entry comp_pf row in 4 registers; each state byte is
        // selected with PRMT (entries 0-7 and 8-15, blended by bit 3 of f)
        // instead of a shared-memory load.
        const uint4 pf_row = reinterpret_cast<const uint4*>(comp_pf)[get_index(prefix)];
        uint32_t p0 = 0, p1 = 0, p2 = 0;
        #pragma unroll
        for (I k = 0; k < VECS; k++) {
            const uint4 v = my[k ^ sw];
            uint32_t tw[4];
            #pragma unroll
            for (I q = 0; q < 4; q++) {
                const uint32_t word = q == 0 ? v.x : q == 1 ? v.y : q == 2 ? v.z : v.w;
                const uint32_t x   = (word >> 1) & 0x07070707u;       // f & 7 per byte
                const uint32_t y   = x | (x >> 4);
                const uint32_t sel = __byte_perm(y, 0, 0x4420);       // nibble j = f_j & 7
                const uint32_t lo  = __byte_perm(pf_row.x, pf_row.y, sel);
                const uint32_t hi  = __byte_perm(pf_row.z, pf_row.w, sel);
                uint32_t hi_mask;   // 0xff in bytes with f >= 8 (bit 4 of the byte),
                                    // by PRMT sign replication (not in __byte_perm)
                asm("prmt.b32 %0, %1, 0, 0xba98;" : "=r"(hi_mask) : "r"(word << 3));
                tw[q] = (lo & ~hi_mask) | (hi & hi_mask);
            }
            my[k ^ sw] = make_uint4(tw[0], tw[1], tw[2], tw[3]);
            #pragma unroll
            for (I q = 0; q < 4; q++) {
                const I wi = 4 * k + q;   // word of the chunk: states 4 wi .. 4 wi + 3
                const uint32_t f4 = ((tw[q] & 0x01010101u) * 0x10204080u) >> 28;
                if (wi < 8)       p0 |= f4 << (4 * (wi % 8));
                else if (wi < 16) p1 |= f4 << (4 * (wi % 8));
                else              p2 |= f4 << (4 * (wi % 8));
                if constexpr (MAXADD) {
                    // token bits (7-5) of each state byte != IGNORE_TOKEN, gathered like f4
                    const uint32_t t  = (tw[q] ^ (uint32_t(IGNORE_TOKEN) * 0x20202020u)) & 0xe0e0e0e0u;
                    const uint32_t nz = ((t >> 5) | (t >> 6) | (t >> 7)) & 0x01010101u;
                    const uint32_t n4 = (nz * 0x10204080u) >> 28;
                    if (wi < 8)       n0 |= n4 << (4 * (wi % 8));
                    else if (wi < 16) n1 |= n4 << (4 * (wi % 8));
                    else              n2 |= n4 << (4 * (wi % 8));
                }
            }
        }
        // Produce flags: element j (bit j of the chunk, in word j / 32)
        // produces if state j + 1 does. Scalars, so the compiler cannot place
        // them in local memory.
        m0 = (p0 >> 1) | (p1 << 31);
        m1 = (p1 >> 1) | (p2 << 31);
        m2 = p2 >> 1;
        if (!full) {
            // states of bytes past my valid input are garbage: keep elements
            // [0, valid - 1) (the last one is set below)
            const I n = valid > 0 ? valid - 1 : 0;
            m0 &= n >= 32 ? ~0u : (1u << n) - 1;
            m1 &= n >= 64 ? ~0u : n <= 32 ? 0u : (1u << (n - 32)) - 1;
            m2 &= n <= 64 ? 0u : (1u << (n - 64)) - 1;
        }
        }
        auto set_bit = [&](I j) {
            if (j < 32)      m0 |= 1u << j;
            else if (j < 64) m1 |= 1u << (j - 32);
            else             m2 |= 1u << (j - 64);
        };
        const uint32_t last = FAST ? pack_chain_state(compose(prefix, chain_state_index(va))) : 0u;
        // Next state after my last element produces / my last state accepts.
        auto next_produces = [&]() -> bool {
            if constexpr (FAST) return chain_produce(step(last, nb));
            else                return is_produce(gstep(last_g, nb));
        };
        auto last_accepts = [&]() -> bool {
            if constexpr (FAST) return chain_accept(last);
            else                return is_accept(last_g);
        };
        if (valid > 0) {
            // last element of my chunk: next state from the following byte,
            // or the end of the input (the final element always produces)
            const I li = valid - 1;
            const bool produce = !has_nb || next_produces();
            if (produce)
                set_bit(li);
        }
        // MAXADD: emitted tokens are the kept ones (produce and terminal !=
        // IGNORE_TOKEN; bit j: element j = token of state j); m0-m2 (all
        // produce flags) still give the token starts.
        const uint32_t e0 = MAXADD ? m0 & n0 : m0;
        const uint32_t e1 = MAXADD ? m1 & n1 : m1;
        const uint32_t e2 = MAXADD ? m2 & n2 : m2;
        const I count = __popc(e0) + __popc(e1) + __popc(e2);

        // Output slots.
        I offs;
        uint32_t max_in = 0;   // MAXADD: start of my chunk's first token
        if constexpr (MAXADD) {
            // "last token end + 1" in my chunk (0: no token ends here)
            const int last_end = m2 ? 95 - __clz(m2) : m1 ? 63 - __clz(m1) : m0 ? 31 - __clz(m0) : -1;
            const MaxAdd agg{last_end >= 0 ? uint32_t(my_offs + last_end + 1) : 0u, count};
            MaxAdd pfx;
            if constexpr (STEP == 3) {
                ScanTileState<MaxAdd> ma_states{index_states.d_tile_descriptors};
                PrefixOpMA ma_op(ma_states, ma_prefix, MaxAddOp(), (int)tile, MaxAdd{0u, 0u});
                BlockScanMA(ma_scan).ExclusiveScan(agg, pfx, MaxAddOp(), ma_op);
                if (valid > 0 && my_offs + valid == size) {   // owner of the last input byte
                    *new_size = pfx.cnt + count;
                    *is_valid = last_accepts();
                }
            } else {
                MaxAdd tile_agg;
                BlockScanMA(ma_scan).ExclusiveScan(agg, pfx, MaxAdd{0u, 0u}, MaxAddOp(), tile_agg);
                pfx.cnt += tile_offs;   // each tile writes to its own region
            }
            offs   = pfx.cnt;
            max_in = pfx.max;
        } else if constexpr (STEP == 3) {
            PrefixOpIdx idx_op(index_states, idx_prefix, Add<I>(), (int)tile, I(0));
            BlockScanI(idx_scan).ExclusiveScan(count, offs, Add<I>(), idx_op);
            if (valid > 0 && my_offs + valid == size) {   // owner of the last input byte
                *new_size = offs + count;
                *is_valid = last_accepts();
            }
        } else {
            I tile_count;
            BlockScanI(idx_scan).ExclusiveScan(count, offs, I(0), Add<I>(), tile_count);
            offs += tile_offs;   // each tile writes to its own region
        }

        // Pass C: warp-cooperative emission.
        I incl = count;
        #pragma unroll
        for (I d = 1; d < WARP; d <<= 1) {
            I y = __shfl_up_sync(0xffffffff, incl, d);
            if (lane >= d) incl += y;
        }
        const I warp_total = __shfl_sync(0xffffffff, incl, WARP - 1);
        const I warp_base  = __shfl_sync(0xffffffff, offs, 0);
        const uint8_t* warp_states = bytes + warp * WARP_BYTES;
        __syncwarp();   // state bytes of all lanes written
        for (I j = 0; j < warp_total; j += WARP) {
            const I r = j + lane;
            // owner lane: smallest o with incl_o > r
            I o = 0;
            #pragma unroll
            for (I d = WARP / 2; d >= 1; d >>= 1) {
                I v = __shfl_sync(0xffffffff, incl, o + d - 1);
                if (v <= r) o += d;
            }
            const I o_incl  = __shfl_sync(0xffffffff, incl, o);
            const I o_count = __shfl_sync(0xffffffff, count, o);
            I k = r - (o_incl - o_count);   // rank within the owner's tokens
            const uint32_t om0 = __shfl_sync(0xffffffff, e0, o);
            const uint32_t om1 = __shfl_sync(0xffffffff, e1, o);
            const uint32_t om2 = __shfl_sync(0xffffffff, e2, o);
            // Word holding the k-th set bit, then the bit within it
            // (select_bit, not __fns: __fns expands to ~140 instructions).
            const I c0 = __popc(om0), c01 = c0 + __popc(om1);
            const uint32_t m = k < c0 ? om0 : k < c01 ? om1 : om2;
            const I base     = k < c0 ? 0   : k < c01 ? 32  : 64;
            k               -= k < c0 ? 0   : k < c01 ? c0  : c01;
            const I pos = base + select_bit(m, k);
            if constexpr (MAXADD) {
                // start: after the owner's previous token end (any terminal)
                // below pos, else the owner's incoming max
                const uint32_t am0 = __shfl_sync(0xffffffff, m0, o);
                const uint32_t am1 = __shfl_sync(0xffffffff, m1, o);
                const uint32_t am2 = __shfl_sync(0xffffffff, m2, o);
                const uint32_t o_max = __shfl_sync(0xffffffff, max_in, o);
                const uint32_t b0 = pos >= 32 ? am0 : am0 & ((1u << pos) - 1);
                const uint32_t b1 = pos >= 64 ? am1 : pos < 32 ? 0u : am1 & ((1u << (pos - 32)) - 1);
                const uint32_t b2 = pos < 64 ? 0u : am2 & ((1u << (pos - 64)) - 1);
                const int prev = b2 ? 95 - __clz(b2) : b1 ? 63 - __clz(b1) : b0 ? 31 - __clz(b0) : -1;
                if (r < warp_total) {
                    const I elem  = o * CHUNK + pos;   // within the warp's segment
                    const I chunk = tile_offs + warp * WARP_BYTES + o * CHUNK;
                    const I start = prev >= 0 ? chunk + prev + 1 : o_max;
                    d_index_out[warp_base + r] = start;
                    d_len_out[warp_base + r]   = chunk + pos - start + 1;
                    d_token_out[warp_base + r] = warp_states[elem ^ (((o >> 2) & 1) << 4)] >> 5;
                }
            } else {
                if (r < warp_total) {
                    const I elem = o * CHUNK + pos;   // within the warp's segment
                    d_index_out[warp_base + r] = tile_offs + warp * WARP_BYTES + elem;
                    d_token_out[warp_base + r] =
                        warp_states[elem ^ (((o >> 2) & 1) << 4)] >> (FAST ? 5 : 0);   // token
                }
            }
        }
    }
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

template<uint32_t BS, uint32_t IPT>
void testLexerTranspose(uint8_t* input,
                        size_t input_size,
                        uint32_t* expected_indices,
                        token_t* expected_tokens,
                        size_t expected_size) {
    using I = uint32_t;
    const I size = input_size;
    const I NLB  = (size + BS * IPT - 1) / (BS * IPT);
    const size_t IN_ARRAY_BYTES  = (size_t)size * sizeof(uint8_t);
    const size_t INDEX_OUT_BYTES = (size_t)size * sizeof(I);
    const size_t TOKEN_OUT_BYTES = (size_t)size * sizeof(token_t);
#ifdef PROFILE
    const I WARMUP_RUNS = 1; const I RUNS = 1;
#else
    const I WARMUP_RUNS = 500; const I RUNS = 100;
#endif
    std::vector<token_t> h_token_out(size, 0);
    std::vector<I>       h_index_out(size, 0);

    I*       d_new_size;
    bool*    d_is_valid;
    uint8_t* d_in;
    I*       d_index_out;
    token_t* d_token_out;
    ScanTileState<state_t> d_state_states;
    ScanTileState<I>       d_index_states;

    gpuAssert(cudaMalloc((void**)&d_new_size,  sizeof(I)));
    gpuAssert(cudaMalloc((void**)&d_is_valid,  sizeof(bool)));
    gpuAssert(cudaMalloc((void**)&d_state_states.d_tile_descriptors,
        ScanTileState<state_t>::AllocationSize(NLB)));
    gpuAssert(cudaMalloc((void**)&d_index_states.d_tile_descriptors,
        ScanTileState<I>::AllocationSize(NLB)));
    gpuAssert(cudaMalloc((void**)&d_in,        IN_ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_index_out, INDEX_OUT_BYTES));
    gpuAssert(cudaMalloc((void**)&d_token_out, TOKEN_OUT_BYTES));
    gpuAssert(cudaMemcpy(d_in, input, IN_ARRAY_BYTES, cudaMemcpyHostToDevice));

    LexerCtxShmem ctx = LexerCtxShmem();

    auto reset = [&]() {
        cudaMemset(d_is_valid, 0, sizeof(bool));
        initScanTileState(d_state_states, (int)NLB);
        initScanTileState(d_index_states, (int)NLB);
    };
    reset();

    float* temp_total = (float*) malloc(sizeof(float) * RUNS);
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    for (I i = 0; i < WARMUP_RUNS; ++i) {
        lexerTranspose<I, BS, IPT><<<NLB, BS>>>(
            ctx, d_in, d_index_out, d_token_out,
            d_state_states, d_index_states, size, NLB, d_new_size, d_is_valid);
        cudaDeviceSynchronize(); reset();
        gpuAssert(cudaPeekAtLastError());
    }
    for (I i = 0; i < RUNS; ++i) {
        cudaEventRecord(start, 0);
        lexerTranspose<I, BS, IPT><<<NLB, BS>>>(
            ctx, d_in, d_index_out, d_token_out,
            d_state_states, d_index_states, size, NLB, d_new_size, d_is_valid);
        gpuAssert(cudaDeviceSynchronize());
        cudaEventRecord(stop, 0); cudaEventSynchronize(stop);
        cudaEventElapsedTime(temp_total + i, start, stop);
        reset(); gpuAssert(cudaPeekAtLastError());
    }

    I temp_size = 0;
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));

    // Correctness run.
    reset();
    lexerTranspose<I, BS, IPT><<<NLB, BS>>>(
        ctx, d_in, d_index_out, d_token_out,
        d_state_states, d_index_states, size, NLB, d_new_size, d_is_valid);
    cudaDeviceSynchronize(); gpuAssert(cudaPeekAtLastError());

    bool is_valid = false;
    gpuAssert(cudaMemcpy(h_index_out.data(), d_index_out, INDEX_OUT_BYTES, cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(h_token_out.data(), d_token_out, TOKEN_OUT_BYTES, cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(&is_valid,  d_is_valid, sizeof(bool), cudaMemcpyDeviceToHost));

    bool test_passes = is_valid;
    if (!test_passes)
        std::cout << "Lexer Test Failed: The input given to the lexer does not result in an accepting state." << std::endl;
    if (test_passes && temp_size != (I)expected_size) {
        printf("Lexer Test Failed: Expected size=%zu but got size=%u\n", expected_size, temp_size);
        test_passes = false;
    }
    if (test_passes) {
        for (I i = 0; i < (I)expected_size; ++i) {
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
        const size_t TOTAL_BYTES = IN_ARRAY_BYTES
            + (size_t)temp_size * (sizeof(I) + sizeof(token_t));
        printf("\n");
        printf("  %-36s ", "Total:");
        compute_descriptors(temp_total, RUNS, TOTAL_BYTES);
    }

    free(temp_total);
    gpuAssert(cudaFree(d_in)); gpuAssert(cudaFree(d_index_out));
    gpuAssert(cudaFree(d_token_out));
    gpuAssert(cudaFree(d_index_states.d_tile_descriptors));
    gpuAssert(cudaFree(d_state_states.d_tile_descriptors));
    gpuAssert(cudaFree(d_new_size)); gpuAssert(cudaFree(d_is_valid));
    ctx.Cleanup();
}

// Grid for the persistent lexerVecPipe: every block must be co-resident (its
// lookbacks spin on predecessor tiles). Requests the maximum shared memory
// carveout (~26 KB/block) and sizes the grid from the occupancy API.
template<typename I, I BS, I IPT>
static I lexerVecPipeGrid(I num_tiles, int* blocks_per_sm) {
    auto kernel = lexerVecPipe<I, BS, IPT>;
    gpuAssert(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                   (int)cudaSharedmemCarveoutMaxShared));
    int bps = 0, sms = 0;
    gpuAssert(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel, BS, 0));
    gpuAssert(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    if (bps < 1) { fprintf(stderr, "lexerVecPipe does not fit on an SM\n"); exit(1); }
    if (blocks_per_sm) *blocks_per_sm = bps;
    return std::min(num_tiles, (I)(bps * sms));
}

template<uint32_t BS, uint32_t IPT>
void testLexerVecPipe(uint8_t* input,
                      size_t input_size,
                      uint32_t* expected_indices,
                      token_t* expected_tokens,
                      size_t expected_size) {
    using I = uint32_t;
    const I size = input_size;
    const I NLB  = (size + BS * IPT - 1) / (BS * IPT);
    const size_t IN_ARRAY_BYTES  = (size_t)size * sizeof(uint8_t);
    const size_t INDEX_OUT_BYTES = (size_t)size * sizeof(I);
    const size_t TOKEN_OUT_BYTES = (size_t)size * sizeof(token_t);
#ifdef PROFILE
    const I WARMUP_RUNS = 1; const I RUNS = 1;
#else
    const I WARMUP_RUNS = 500; const I RUNS = 100;
#endif
    std::vector<token_t> h_token_out(size, 0);
    std::vector<I>       h_index_out(size, 0);

    I*       d_new_size;
    bool*    d_is_valid;
    uint8_t* d_in;
    I*       d_index_out;
    token_t* d_token_out;
    ScanTileState<state_t> d_state_states;
    ScanTileState<I>       d_index_states;

    gpuAssert(cudaMalloc((void**)&d_new_size,  sizeof(I)));
    gpuAssert(cudaMalloc((void**)&d_is_valid,  sizeof(bool)));
    gpuAssert(cudaMalloc((void**)&d_state_states.d_tile_descriptors,
        ScanTileState<state_t>::AllocationSize(NLB)));
    gpuAssert(cudaMalloc((void**)&d_index_states.d_tile_descriptors,
        ScanTileState<I>::AllocationSize(NLB)));
    gpuAssert(cudaMalloc((void**)&d_in,        IN_ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_index_out, INDEX_OUT_BYTES));
    gpuAssert(cudaMalloc((void**)&d_token_out, TOKEN_OUT_BYTES));
    gpuAssert(cudaMemcpy(d_in, input, IN_ARRAY_BYTES, cudaMemcpyHostToDevice));

    LexerCtxShmem ctx = LexerCtxShmem();
    int bps = 0;
    const I grid = lexerVecPipeGrid<I, BS, IPT>(NLB, &bps);
    printf("[%d/SM] ", bps);
    fflush(stdout);

    auto reset = [&]() {
        cudaMemset(d_is_valid, 0, sizeof(bool));
        initScanTileState(d_state_states, (int)NLB);
        initScanTileState(d_index_states, (int)NLB);
    };
    auto launch = [&]() {
        lexerVecPipe<I, BS, IPT><<<grid, BS>>>(
            ctx, d_in, d_index_out, d_token_out,
            d_state_states, d_index_states, size, NLB, d_new_size, d_is_valid);
    };
    reset();

    float* temp_total = (float*) malloc(sizeof(float) * RUNS);
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    for (I i = 0; i < WARMUP_RUNS; ++i) {
        launch();
        cudaDeviceSynchronize(); reset();
        gpuAssert(cudaPeekAtLastError());
    }
    for (I i = 0; i < RUNS; ++i) {
        cudaEventRecord(start, 0);
        launch();
        gpuAssert(cudaDeviceSynchronize());
        cudaEventRecord(stop, 0); cudaEventSynchronize(stop);
        cudaEventElapsedTime(temp_total + i, start, stop);
        reset(); gpuAssert(cudaPeekAtLastError());
    }

    // Correctness run.
    reset();
    launch();
    cudaDeviceSynchronize(); gpuAssert(cudaPeekAtLastError());

    I temp_size = 0;
    bool is_valid = false;
    gpuAssert(cudaMemcpy(h_index_out.data(), d_index_out, INDEX_OUT_BYTES, cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(h_token_out.data(), d_token_out, TOKEN_OUT_BYTES, cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(&is_valid,  d_is_valid, sizeof(bool), cudaMemcpyDeviceToHost));

    bool test_passes = is_valid;
    if (!test_passes)
        std::cout << "Lexer Test Failed: The input given to the lexer does not result in an accepting state." << std::endl;
    if (test_passes && temp_size != (I)expected_size) {
        printf("Lexer Test Failed: Expected size=%zu but got size=%u\n", expected_size, temp_size);
        test_passes = false;
    }
    if (test_passes) {
        for (I i = 0; i < (I)expected_size; ++i) {
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
        const size_t TOTAL_BYTES = IN_ARRAY_BYTES
            + (size_t)temp_size * (sizeof(I) + sizeof(token_t));
        printf("\n");
        printf("  %-36s ", "Total:");
        compute_descriptors(temp_total, RUNS, TOTAL_BYTES);
    }

    free(temp_total);
    gpuAssert(cudaFree(d_in)); gpuAssert(cudaFree(d_index_out));
    gpuAssert(cudaFree(d_token_out));
    gpuAssert(cudaFree(d_index_states.d_tile_descriptors));
    gpuAssert(cudaFree(d_state_states.d_tile_descriptors));
    gpuAssert(cudaFree(d_new_size)); gpuAssert(cudaFree(d_is_valid));
    ctx.Cleanup();
}


// Requests the maximum shared memory carveout for lexerBig (~26 KB per block;
// 6 blocks/SM exceed the default configuration) and returns blocks/SM.
template<typename I, I BS, I CHUNK, I STEP, bool FORCE_GENERIC = false, bool MAXADD = false,
         bool STEPTBL = false>
static int lexerBigBlocksPerSM() {
    auto kernel = lexerBig<I, BS, CHUNK, STEP, FORCE_GENERIC, MAXADD, STEPTBL>;
    gpuAssert(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                   (int)cudaSharedmemCarveoutMaxShared));
    int bps = 0;
    gpuAssert(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel, BS, 0));
    return bps;
}

// alpacc's lexer output (lexerBig MAXADD) from the (token end, token) form:
// the tokens with terminal != IGNORE_TOKEN, start = previous token end + 1 (0
// for the first token), length = end - start + 1.
static size_t maxadd_reference(const uint32_t* ends, const token_t* toks, size_t n,
                               std::vector<token_t>& t, std::vector<uint32_t>& s,
                               std::vector<uint32_t>& l) {
    t.clear(); s.clear(); l.clear();
    uint32_t start = 0;
    for (size_t i = 0; i < n; i++) {
        if (toks[i] != IGNORE_TOKEN) {
            t.push_back(toks[i]); s.push_back(start); l.push_back(ends[i] - start + 1);
        }
        start = ends[i] + 1;
    }
    return t.size();
}

// STEP 1/2 are ladder steps (timing only, output not valid); STEP 3 is checked
// against the expected output (MAXADD: against maxadd_reference). GB/s always
// counts the full lexer's traffic.
template<uint32_t BS, uint32_t CHUNK, uint32_t STEP, bool FORCE_GENERIC = false, bool MAXADD = false,
         bool STEPTBL = false>
void testLexerBig(uint8_t* input,
                  size_t input_size,
                  uint32_t* expected_indices,
                  token_t* expected_tokens,
                  size_t expected_size) {
    using I = uint32_t;
    const I size = input_size;
    const I NLB  = (size + BS * CHUNK - 1) / (BS * CHUNK);
    const size_t IN_ARRAY_BYTES  = (size_t)size * sizeof(uint8_t);
    const size_t INDEX_OUT_BYTES = (size_t)size * sizeof(I);
    const size_t TOKEN_OUT_BYTES = (size_t)size * sizeof(token_t);
#ifdef PROFILE
    const I WARMUP_RUNS = 1; const I RUNS = 1;
#else
    const I WARMUP_RUNS = 500; const I RUNS = 100;
#endif
    std::vector<token_t> h_token_out(size, 0);
    std::vector<I>       h_index_out(size, 0);
    std::vector<I>       h_len_out(MAXADD ? size : 0, 0);
    std::vector<token_t> ref_t; std::vector<uint32_t> ref_s, ref_l;   // MAXADD reference
    // No expected output (expected_indices == nullptr): not verified, token
    // count taken from the kernel.
    const bool verify = expected_indices != nullptr;
    size_t out_tokens = MAXADD && verify
        ? maxadd_reference(expected_indices, expected_tokens, expected_size, ref_t, ref_s, ref_l)
        : expected_size;

    I*       d_new_size;
    bool*    d_is_valid;
    uint8_t* d_in;
    I*       d_index_out;
    token_t* d_token_out;
    I*       d_len_out = nullptr;   // MAXADD: token lengths
    ScanTileState<state_t> d_state_states;
    ScanTileState<I>       d_index_states;

    gpuAssert(cudaMalloc((void**)&d_new_size,  sizeof(I)));
    gpuAssert(cudaMalloc((void**)&d_is_valid,  sizeof(bool)));
    gpuAssert(cudaMalloc((void**)&d_state_states.d_tile_descriptors,
        ScanTileState<state_t>::AllocationSize(NLB)));
    gpuAssert(cudaMalloc((void**)&d_index_states.d_tile_descriptors,
        ScanTileState<I>::AllocationSize(NLB)));
    gpuAssert(cudaMalloc((void**)&d_in,        IN_ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_index_out, INDEX_OUT_BYTES));
    gpuAssert(cudaMalloc((void**)&d_token_out, TOKEN_OUT_BYTES));
    if (MAXADD) gpuAssert(cudaMalloc((void**)&d_len_out, INDEX_OUT_BYTES));
    gpuAssert(cudaMemcpy(d_in, input, IN_ARRAY_BYTES, cudaMemcpyHostToDevice));

    LexerCtxShmem ctx = LexerCtxShmem();
    printf("[%d/SM] ", lexerBigBlocksPerSM<I, BS, CHUNK, STEP, FORCE_GENERIC, MAXADD, STEPTBL>());
    fflush(stdout);

    auto reset = [&]() {
        cudaMemset(d_is_valid, 0, sizeof(bool));
        initScanTileState(d_state_states, (int)NLB);
        initScanTileState(d_index_states, (int)NLB);
    };
    auto launch = [&]() {
        lexerBig<I, BS, CHUNK, STEP, FORCE_GENERIC, MAXADD, STEPTBL><<<NLB, BS>>>(
            ctx, d_in, d_index_out, d_token_out, d_len_out,
            d_state_states, d_index_states, size, NLB, d_new_size, d_is_valid);
    };
    reset();

    float* temp_total = (float*) malloc(sizeof(float) * RUNS);
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    for (I i = 0; i < WARMUP_RUNS; ++i) {
        launch();
        cudaDeviceSynchronize(); reset();
        gpuAssert(cudaPeekAtLastError());
    }
    for (I i = 0; i < RUNS; ++i) {
        cudaEventRecord(start, 0);
        launch();
        gpuAssert(cudaDeviceSynchronize());
        cudaEventRecord(stop, 0); cudaEventSynchronize(stop);
        cudaEventElapsedTime(temp_total + i, start, stop);
        reset(); gpuAssert(cudaPeekAtLastError());
    }

    bool test_passes = true;
    if (STEP == 3) {
        reset();
        launch();
        cudaDeviceSynchronize(); gpuAssert(cudaPeekAtLastError());

        I temp_size = 0;
        bool is_valid = false;
        gpuAssert(cudaMemcpy(h_index_out.data(), d_index_out, INDEX_OUT_BYTES, cudaMemcpyDeviceToHost));
        gpuAssert(cudaMemcpy(h_token_out.data(), d_token_out, TOKEN_OUT_BYTES, cudaMemcpyDeviceToHost));
        gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
        gpuAssert(cudaMemcpy(&is_valid,  d_is_valid, sizeof(bool), cudaMemcpyDeviceToHost));

        if (!verify) {
            out_tokens = temp_size;
            printf("[%u tokens, %s] ", temp_size, is_valid ? "accepting" : "not accepting");
        }
        test_passes = is_valid || !verify;
        if (!test_passes)
            std::cout << "Lexer Test Failed: The input given to the lexer does not result in an accepting state." << std::endl;
        if (test_passes && temp_size != (I)out_tokens) {
            printf("Lexer Test Failed: Expected size=%zu but got size=%u\n", out_tokens, temp_size);
            test_passes = false;
        }
        if (test_passes && MAXADD) {
            gpuAssert(cudaMemcpy(h_len_out.data(), d_len_out, INDEX_OUT_BYTES, cudaMemcpyDeviceToHost));
            for (I i = 0; i < (I)out_tokens; ++i) {
                if (h_token_out[i] != ref_t[i] || h_index_out[i] != ref_s[i] || h_len_out[i] != ref_l[i]) {
                    printf("Lexer Test Failed: (token, start, length) mismatch at i=%u: expected=(%u, %u, %u) got=(%u, %u, %u)\n",
                           i, (unsigned)ref_t[i], ref_s[i], ref_l[i],
                           (unsigned)h_token_out[i], h_index_out[i], h_len_out[i]);
                    test_passes = false; break;
                }
            }
        } else if (test_passes && verify) {
            for (I i = 0; i < (I)expected_size; ++i) {
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
    }

    if (test_passes) {
        // Bytes this variant actually moves: S1 only reads the input; S2 and S3
        // also write one index and one token per output token (MAXADD: a
        // token, start and length per kept token).
        const size_t TOTAL_BYTES = IN_ARRAY_BYTES
            + (STEP == 1 ? 0 : out_tokens * ((MAXADD ? 2 : 1) * sizeof(I) + sizeof(token_t)));
        printf("\n");
        printf("  %-36s ", STEP == 3 ? "Total:" : "Total (ladder, output not valid):");
        compute_descriptors(temp_total, RUNS, TOTAL_BYTES);
    }

    free(temp_total);
    gpuAssert(cudaFree(d_in)); gpuAssert(cudaFree(d_index_out));
    gpuAssert(cudaFree(d_token_out)); if (d_len_out) gpuAssert(cudaFree(d_len_out));
    gpuAssert(cudaFree(d_index_states.d_tile_descriptors));
    gpuAssert(cudaFree(d_state_states.d_tile_descriptors));
    gpuAssert(cudaFree(d_new_size)); gpuAssert(cudaFree(d_is_valid));
    ctx.Cleanup();
}

// Debug tests (small inputs of the benchmark DFA); not built for other DFAs.
#ifndef LEXER_DFA_JSON
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

    // Same test for lexerVecPipe (single-pass, vectorized, persistent).
    {
        const I VP_IPT   = 24;
        const I vp_tiles = (size + BLOCK_SIZE * VP_IPT - 1) / (BLOCK_SIZE * VP_IPT);
        ScanTileState<state_t> vp_state_states;
        ScanTileState<I>       vp_index_states;
        gpuAssert(cudaMalloc((void**)&vp_state_states.d_tile_descriptors,
            ScanTileState<state_t>::AllocationSize(vp_tiles)));
        gpuAssert(cudaMalloc((void**)&vp_index_states.d_tile_descriptors,
            ScanTileState<I>::AllocationSize(vp_tiles)));
        initScanTileState(vp_state_states, (int)vp_tiles);
        initScanTileState(vp_index_states, (int)vp_tiles);
        gpuAssert(cudaMemset(d_is_valid, 0, sizeof(bool)));
        gpuAssert(cudaMemset(d_new_size, 0xff, sizeof(I)));
        gpuAssert(cudaMemset(d_index_out, 0xff, size * sizeof(I)));
        gpuAssert(cudaMemset(d_token_out, 0xff, size * sizeof(token_t)));

        LexerCtxShmem vp_ctx;
        const I grid = lexerVecPipeGrid<I, BLOCK_SIZE, VP_IPT>(vp_tiles, nullptr);
        lexerVecPipe<I, BLOCK_SIZE, VP_IPT><<<grid, BLOCK_SIZE>>>(
            vp_ctx, d_in, d_index_out, d_token_out, vp_state_states, vp_index_states,
            size, vp_tiles, d_new_size, d_is_valid);
        gpuAssert(cudaDeviceSynchronize());
        gpuAssert(cudaPeekAtLastError());

        I    vp_size  = 0;
        bool vp_valid = false;
        gpuAssert(cudaMemcpy(&vp_size,  d_new_size, sizeof(I),    cudaMemcpyDeviceToHost));
        gpuAssert(cudaMemcpy(&vp_valid, d_is_valid, sizeof(bool), cudaMemcpyDeviceToHost));
        bool vp_pass = vp_valid && vp_size == (I) test->expected_size;
        if (vp_pass) {
            std::vector<I>       vp_indices(vp_size);
            std::vector<token_t> vp_tokens(vp_size);
            gpuAssert(cudaMemcpy(vp_indices.data(), d_index_out, vp_size * sizeof(I),       cudaMemcpyDeviceToHost));
            gpuAssert(cudaMemcpy(vp_tokens.data(),  d_token_out, vp_size * sizeof(token_t), cudaMemcpyDeviceToHost));
            for (I i = 0; i < vp_size && vp_pass; i++)
                vp_pass = vp_indices[i] == test->expected_indices[i] &&
                          vp_tokens[i]  == test->expected_tokens[i];
        }
        if (vp_pass)
            printf("PASS [%s] (lexerVecPipe)\n", test->name);
        else
            fprintf(stderr, "FAIL [%s] (lexerVecPipe): valid=%d size=%u (expected %zu)\n",
                    test->name, (int)vp_valid, vp_size, test->expected_size);
        pass = pass && vp_pass;

        vp_ctx.Cleanup();
        gpuAssert(cudaFree(vp_state_states.d_tile_descriptors));
        gpuAssert(cudaFree(vp_index_states.d_tile_descriptors));
    }

    // Same test for lexerBig (large-tile single-pass lexer, full STEP 3), all variants.
    auto run_big = [&](auto gen_c) {
        constexpr bool GEN = decltype(gen_c)::value;
        const I BIG_CHUNK = 96;
        const I big_tiles = (size + BLOCK_SIZE * BIG_CHUNK - 1) / (BLOCK_SIZE * BIG_CHUNK);
        ScanTileState<state_t> bg_state_states;
        ScanTileState<I>       bg_index_states;
        gpuAssert(cudaMalloc((void**)&bg_state_states.d_tile_descriptors,
            ScanTileState<state_t>::AllocationSize(big_tiles)));
        gpuAssert(cudaMalloc((void**)&bg_index_states.d_tile_descriptors,
            ScanTileState<I>::AllocationSize(big_tiles)));
        initScanTileState(bg_state_states, (int)big_tiles);
        initScanTileState(bg_index_states, (int)big_tiles);
        gpuAssert(cudaMemset(d_is_valid, 0, sizeof(bool)));
        gpuAssert(cudaMemset(d_new_size, 0xff, sizeof(I)));
        gpuAssert(cudaMemset(d_index_out, 0xff, size * sizeof(I)));
        gpuAssert(cudaMemset(d_token_out, 0xff, size * sizeof(token_t)));

        LexerCtxShmem bg_ctx;
        lexerBigBlocksPerSM<I, BLOCK_SIZE, BIG_CHUNK, 3, GEN>();
        lexerBig<I, BLOCK_SIZE, BIG_CHUNK, 3, GEN><<<big_tiles, BLOCK_SIZE>>>(
            bg_ctx, d_in, d_index_out, d_token_out, nullptr, bg_state_states, bg_index_states,
            size, big_tiles, d_new_size, d_is_valid);
        gpuAssert(cudaDeviceSynchronize());
        gpuAssert(cudaPeekAtLastError());

        I    bg_size  = 0;
        bool bg_valid = false;
        gpuAssert(cudaMemcpy(&bg_size,  d_new_size, sizeof(I),    cudaMemcpyDeviceToHost));
        gpuAssert(cudaMemcpy(&bg_valid, d_is_valid, sizeof(bool), cudaMemcpyDeviceToHost));
        bool bg_pass = bg_valid && bg_size == (I) test->expected_size;
        if (bg_pass) {
            std::vector<I>       bg_indices(bg_size);
            std::vector<token_t> bg_tokens(bg_size);
            gpuAssert(cudaMemcpy(bg_indices.data(), d_index_out, bg_size * sizeof(I),       cudaMemcpyDeviceToHost));
            gpuAssert(cudaMemcpy(bg_tokens.data(),  d_token_out, bg_size * sizeof(token_t), cudaMemcpyDeviceToHost));
            for (I i = 0; i < bg_size && bg_pass; i++)
                bg_pass = bg_indices[i] == test->expected_indices[i] &&
                          bg_tokens[i]  == test->expected_tokens[i];
        }
        if (bg_pass)
            printf("PASS [%s] (lexerBig generic=%d)\n", test->name, (int)GEN);
        else
            fprintf(stderr, "FAIL [%s] (lexerBig generic=%d): valid=%d size=%u (expected %zu)\n",
                    test->name, (int)GEN, (int)bg_valid, bg_size, test->expected_size);
        pass = pass && bg_pass;

        bg_ctx.Cleanup();
        gpuAssert(cudaFree(bg_state_states.d_tile_descriptors));
        gpuAssert(cudaFree(bg_index_states.d_tile_descriptors));
    };
    run_big(std::false_type{});
    run_big(std::true_type{});

    // lexerBig MAXADD (alpacc output: terminal, start, length of non-ignored
    // tokens) against maxadd_reference.
    {
        const I BIG_CHUNK = 96;
        const I big_tiles = (size + BLOCK_SIZE * BIG_CHUNK - 1) / (BLOCK_SIZE * BIG_CHUNK);
        ScanTileState<state_t> ma_state_states;
        ScanTileState<I>       ma_index_states;
        I* d_len = nullptr;
        gpuAssert(cudaMalloc((void**)&ma_state_states.d_tile_descriptors,
            ScanTileState<state_t>::AllocationSize(big_tiles)));
        gpuAssert(cudaMalloc((void**)&ma_index_states.d_tile_descriptors,
            ScanTileState<I>::AllocationSize(big_tiles)));
        gpuAssert(cudaMalloc((void**)&d_len, size * sizeof(I)));
        initScanTileState(ma_state_states, (int)big_tiles);
        initScanTileState(ma_index_states, (int)big_tiles);
        gpuAssert(cudaMemset(d_is_valid, 0, sizeof(bool)));
        gpuAssert(cudaMemset(d_new_size, 0xff, sizeof(I)));
        gpuAssert(cudaMemset(d_index_out, 0xff, size * sizeof(I)));
        gpuAssert(cudaMemset(d_token_out, 0xff, size * sizeof(token_t)));
        gpuAssert(cudaMemset(d_len, 0xff, size * sizeof(I)));

        LexerCtxShmem ma_ctx;
        lexerBigBlocksPerSM<I, BLOCK_SIZE, BIG_CHUNK, 3, false, true>();
        lexerBig<I, BLOCK_SIZE, BIG_CHUNK, 3, false, true><<<big_tiles, BLOCK_SIZE>>>(
            ma_ctx, d_in, d_index_out, d_token_out, d_len, ma_state_states, ma_index_states,
            size, big_tiles, d_new_size, d_is_valid);
        gpuAssert(cudaDeviceSynchronize());
        gpuAssert(cudaPeekAtLastError());

        std::vector<token_t> ref_t; std::vector<uint32_t> ref_s, ref_l;
        const size_t ref_n = maxadd_reference(test->expected_indices, test->expected_tokens,
                                              test->expected_size, ref_t, ref_s, ref_l);
        I    ma_size  = 0;
        bool ma_valid = false;
        gpuAssert(cudaMemcpy(&ma_size,  d_new_size, sizeof(I),    cudaMemcpyDeviceToHost));
        gpuAssert(cudaMemcpy(&ma_valid, d_is_valid, sizeof(bool), cudaMemcpyDeviceToHost));
        bool ma_pass = ma_valid && ma_size == (I)ref_n;
        if (ma_pass) {
            std::vector<I> s(ma_size), l(ma_size); std::vector<token_t> t(ma_size);
            gpuAssert(cudaMemcpy(s.data(), d_index_out, ma_size * sizeof(I),       cudaMemcpyDeviceToHost));
            gpuAssert(cudaMemcpy(l.data(), d_len,       ma_size * sizeof(I),       cudaMemcpyDeviceToHost));
            gpuAssert(cudaMemcpy(t.data(), d_token_out, ma_size * sizeof(token_t), cudaMemcpyDeviceToHost));
            for (I i = 0; i < ma_size && ma_pass; i++)
                ma_pass = t[i] == ref_t[i] && s[i] == ref_s[i] && l[i] == ref_l[i];
        }
        if (ma_pass)
            printf("PASS [%s] (lexerBig maxadd)\n", test->name);
        else
            fprintf(stderr, "FAIL [%s] (lexerBig maxadd): valid=%d size=%u (expected %zu)\n",
                    test->name, (int)ma_valid, ma_size, ref_n);
        pass = pass && ma_pass;

        ma_ctx.Cleanup();
        gpuAssert(cudaFree(d_len));
        gpuAssert(cudaFree(ma_state_states.d_tile_descriptors));
        gpuAssert(cudaFree(ma_index_states.d_tile_descriptors));
    }

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

#endif  // !LEXER_DFA_JSON (debug tests)

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

#ifdef LEXER_DFA_JSON
// JSON DFA (dfa/json.h): lexerBig only (the other kernels keep the compose
// table in shared memory, too small for 823 endofunctions), no expected
// output (no verification); GB/s counts input + 5 B per token reported by
// the kernel.
int main(int32_t argc, char *argv[]) {
    assert(argc >= 2);
    size_t input_size;
    uint8_t* input = read_u8_array(argv[1], &input_size);
    gpuAssert(cudaMemcpyToSymbol(d_accept, h_accept, sizeof(h_accept)));

    printf("%s (JSON DFA: %u endofunctions, compose table %zu KB; not verified):\n",
           argv[1], NUM_STATES, (size_t)NUM_STATES * NUM_STATES * sizeof(state_t) / 1024);
    printf(PAD, "Big tile S1 (load only):"); fflush(stdout);
    testLexerBig<256, 96, 1>(input, input_size, nullptr, nullptr, 0);
    printf(PAD, "Big tile S2 (no look-backs):"); fflush(stdout);
    testLexerBig<256, 96, 2>(input, input_size, nullptr, nullptr, 0);
    printf(PAD, "Big tile S3 (full) BS256/CHUNK96:"); fflush(stdout);
    testLexerBig<256, 96, 3>(input, input_size, nullptr, nullptr, 0);
    // Per-byte steps through the endofunction x byte-class step table
    // (ctx.d_step) instead of the full compose table.
    printf(PAD, "Big S2 step table:"); fflush(stdout);
    testLexerBig<256, 96, 2, false, false, true>(input, input_size, nullptr, nullptr, 0);
    printf(PAD, "Big S3 step table:"); fflush(stdout);
    testLexerBig<256, 96, 3, false, false, true>(input, input_size, nullptr, nullptr, 0);
    free(input);
    gpuAssert(cudaPeekAtLastError());
    return 0;
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

    printf(PAD, "1Pass BS256/IPT20 (transpose):");
    testLexerTranspose<256, 20>(input, input_size, expected_indices, expected_tokens, expected_indices_size);
    printf(PAD, "1Pass vec + cp.async BS256/IPT24:");
    fflush(stdout);
    testLexerVecPipe<256, 24>(input, input_size, expected_indices, expected_tokens, expected_indices_size);
    printf(PAD, "Big tile S1 (load only):");
    fflush(stdout);
    testLexerBig<256, 96, 1>(input, input_size, expected_indices, expected_tokens, expected_indices_size);
    printf(PAD, "Big tile S2 (no look-backs):");
    fflush(stdout);
    testLexerBig<256, 96, 2>(input, input_size, expected_indices, expected_tokens, expected_indices_size);
    printf(PAD, "Big tile S3 (full) BS256/CHUNK96:");
    fflush(stdout);
    testLexerBig<256, 96, 3>(input, input_size, expected_indices, expected_tokens, expected_indices_size);
    // Generic path (state_t compose chains), forced on this DFA: the cost of
    // DFAs that do not fit the packed fast path.
    printf(PAD, "Big S3 generic path (forced):"); fflush(stdout);
    testLexerBig<256, 96, 3, true>(input, input_size, expected_indices, expected_tokens, expected_indices_size);
    printf(PAD, "Big S3 generic + step table (forced):"); fflush(stdout);
    testLexerBig<256, 96, 3, true, false, true>(input, input_size, expected_indices, expected_tokens, expected_indices_size);
    // alpacc's output semantics (terminal, start, length of tokens != IGNORE_TOKEN)
    // with one (max, add) look-back round.
    printf(PAD, "Big S2 max+add:"); fflush(stdout);
    testLexerBig<256, 96, 2, false, true>(input, input_size, expected_indices, expected_tokens, expected_indices_size);
    printf(PAD, "Big S3 max+add:"); fflush(stdout);
    testLexerBig<256, 96, 3, false, true>(input, input_size, expected_indices, expected_tokens, expected_indices_size);
    free(input);
    free(expected_indices);
    free(expected_tokens);
    gpuAssert(cudaPeekAtLastError());
    return 0;
}
#endif  // LEXER_DFA_JSON

#endif
