#!/usr/bin/env bash
# Extracts the DFA tables of an alpacc-generated CUDA lexer (dense compose
# table layout: `alpacc cuda <grammar> --lexer`) into a header for
# cuda_lexer.cu, using this repository's names.
#   usage: dfa/extract_alpacc.sh <alpacc lexer .cu> > dfa/<name>.h
set -euo pipefail
SRC="$1"
cat <<'EOF'
// JSON lexer DFA, extracted from alpacc's generated CUDA lexer
// (CoderDue/alpacc: json.cu for grammars/json.alp, dense compose table) by
// dfa/extract_alpacc.sh. Selected with -DLEXER_DFA_JSON.
//
// States are endofunctions of the DFA (transition monoid elements):
// index in ENDO_MASK, terminal in TOKEN_MASK (alpacc's TERMINAL_*), produce
// in PRODUCE_MASK; acceptance is the table h_accept (no accept bit).
// alpacc's compose table is earlier-major: compose(a, b) = h_compose[a * N + b].
EOF
grep -m1 "^enum terminal_t" "$SRC" | sed 's/^enum terminal_t : ALPACC_TERMINAL_T/\/\/ terminals:/'
echo "using state_t = uint16_t;"
grep -m1 "^const size_t NUM_STATES" "$SRC" | sed 's/const size_t/const uint32_t/'
grep -m1 "^const size_t NUM_TRANS" "$SRC" | sed 's/const size_t/const uint32_t/'
grep -m1 "^#define IGNORE_TOKEN" "$SRC" | sed 's/#define IGNORE_TOKEN \([0-9]*\)/constexpr token_t IGNORE_TOKEN = \1;/'
grep -m1 "^const state_t ENDO_MASK" "$SRC"
grep -m1 "^const state_t ENDO_OFFSET" "$SRC"
grep -m1 "^const state_t TERMINAL_MASK" "$SRC" | sed 's/TERMINAL_MASK/TOKEN_MASK/'
grep -m1 "^const state_t TERMINAL_OFFSET" "$SRC" | sed 's/TERMINAL_OFFSET/TOKEN_OFFSET/'
grep -m1 "^const state_t PRODUCE_MASK" "$SRC"
grep -m1 "^const state_t PRODUCE_OFFSET" "$SRC"
grep -m1 "^const state_t IDENTITY" "$SRC"
echo "constexpr bool COMPOSE_LATER_MAJOR = false;   // compose(a, b) = h_compose[a * N + b]"
echo "#define LEXER_ACCEPT_TABLE 1"
echo
grep -m1 -A1 "^const state_t h_to_state" "$SRC"
grep -m1 -A1 "^const state_t h_compose" "$SRC"
grep -m1 -A1 "^const bool h_accept" "$SRC"
