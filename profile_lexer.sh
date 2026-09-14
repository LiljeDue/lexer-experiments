#!/usr/bin/env bash
set -e

INPUT=../data/tokens_dense_500MiB.in
INDICES=tokens_indices_dense_500MiB.out
TOKENS=tokens_tokens_dense_500MiB.out
BINARY=./cuda_lexer

METRICS="gpu__time_duration.sum"
METRICS="$METRICS,sm__warps_active.avg.pct_of_peak_sustained_active"
METRICS="$METRICS,dram__bytes_read.sum"
METRICS="$METRICS,dram__bytes_write.sum"
METRICS="$METRICS,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum"
METRICS="$METRICS,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum"
METRICS="$METRICS,l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum"
METRICS="$METRICS,l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum"
METRICS="$METRICS,lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum"
METRICS="$METRICS,sm__sass_average_data_bytes_per_sector_mem_global_op_ld.pct"
METRICS="$METRICS,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"
METRICS="$METRICS,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum"
# Warp stall reasons — what are warps waiting on?
METRICS="$METRICS,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"
METRICS="$METRICS,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct"
METRICS="$METRICS,smsp__warp_issue_stalled_barrier_per_warp_active.pct"
METRICS="$METRICS,smsp__warp_issue_stalled_membar_per_warp_active.pct"
METRICS="$METRICS,smsp__warp_issue_stalled_wait_per_warp_active.pct"
METRICS="$METRICS,smsp__warp_issue_stalled_no_instructions_per_warp_active.pct"
METRICS="$METRICS,smsp__warp_issue_stalled_not_selected_per_warp_active.pct"

# Run the binary once per kernel with a regex that uniquely matches that kernel.
# lexerShmemCompose, lexerAlpacc, lexerAlpaccShmem are distinct enough.
# "^void lexer<" matches only the base lexer (not the Shmem/Alpacc variants).
declare -A KERNELS=(
  ["lexer"]="^lexer$"
  ["lexerShmemCompose"]="^lexerShmemCompose$"
  ["lexerAlpacc"]="^lexerAlpacc$"
  ["lexerAlpaccShmem"]="^lexerAlpaccShmem$"
)

for NAME in lexer lexerShmemCompose lexerAlpacc lexerAlpaccShmem; do
  REGEX="${KERNELS[$NAME]}"
  echo ""
  echo "=== $NAME ==="
  ncu \
    --metrics "$METRICS" \
    --target-processes all \
    --kernel-name-base function \
    --kernel-name regex:"$REGEX" \
    --launch-count 1 \
    --print-summary per-kernel \
    "$BINARY" "$INPUT" "$INDICES" "$TOKENS"
done
