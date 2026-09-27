// Lexer DFA extracted from alpacc's generated CUDA lexer (CoderDue/alpacc,
// dense compose table) by dfa/extract_alpacc.sh; for JSON: json.cu of
// grammars/json.alp. Selected with -DLEXER_DFA_JSON.
//
// States are endofunctions of the DFA (transition monoid elements):
// index in ENDO_MASK, terminal in TOKEN_MASK (alpacc's TERMINAL_*), produce
// in PRODUCE_MASK; acceptance is the table h_accept (no accept bit).
// alpacc's compose table is earlier-major: compose(a, b) = h_compose[a * N + b].
// The tables are loaded from the matching .bin file at run time.
// terminals: {string,number,literal_2,literal_3,literal_4,literal_5,literal_6,literal_7,literal_8,literal_9,literal_10,ignore,empty_12};
using state_t = uint16_t;
const uint32_t NUM_STATES = 823;
const uint32_t NUM_TRANS = 256;
constexpr token_t IGNORE_TOKEN = 11;
const state_t ENDO_MASK = 1023;
const state_t ENDO_OFFSET = 0;
const state_t TOKEN_MASK = 15360;
const state_t TOKEN_OFFSET = 10;
const state_t PRODUCE_MASK = 16384;
const state_t PRODUCE_OFFSET = 14;
const state_t IDENTITY = 12288;
constexpr bool COMPOSE_LATER_MAJOR = false;   // compose(a, b) = h_compose[a * N + b]
#define LEXER_ACCEPT_TABLE 1
#define LEXER_DFA_TABLES_FROM_FILE 1   // filled by load_dfa_tables(dfa/json.bin)

state_t h_to_state[NUM_TRANS];
state_t h_compose[NUM_STATES * NUM_STATES];
bool    h_accept[NUM_STATES];
