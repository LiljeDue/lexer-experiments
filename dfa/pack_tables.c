// Packs the DFA tables of an alpacc-generated CUDA lexer (h_to_state,
// h_compose, h_accept initializers) into a binary file that cuda_lexer loads
// at run time (-DLEXER_DFA_JSON), so the tables are not compiled into the
// program. Layout (little-endian):
//   "LXDFA1\0\0", u32 num_states, u32 num_trans,
//   u16 to_state[num_trans], u16 compose[num_states^2], u8 accept[num_states]
//   usage: pack_tables <alpacc lexer .cu> <output .bin>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Reads the initializer following the line starting with `decl`: numbers
// (or true/false) up to the closing '}'.
static uint32_t* read_array(FILE* f, const char* decl, size_t* n) {
    rewind(f);
    size_t cap = 1 << 20; char* line = malloc(cap); size_t len;
    int found = 0;
    while (fgets(line, (int)cap, f)) {
        if (strncmp(line, decl, strlen(decl)) == 0) { found = 1; break; }
    }
    if (!found) { fprintf(stderr, "missing %s\n", decl); exit(1); }
    size_t vcap = 1024; uint32_t* v = malloc(vcap * sizeof(uint32_t)); *n = 0;
    int c, in_braces = 0; char tok[32]; size_t tl = 0;
    while ((c = fgetc(f)) != EOF) {
        if (c == '{') { in_braces = 1; continue; }
        if (!in_braces) continue;
        if (c == ',' || c == '}' || c == ' ' || c == '\n' || c == '\t') {
            if (tl) {
                tok[tl] = 0; tl = 0;
                if (*n == vcap) { vcap *= 2; v = realloc(v, vcap * sizeof(uint32_t)); }
                v[(*n)++] = strcmp(tok, "true") == 0 ? 1 : strcmp(tok, "false") == 0 ? 0
                          : (uint32_t)strtoul(tok, NULL, 10);
            }
            if (c == '}') break;
        } else if (tl < sizeof(tok) - 1) {
            tok[tl++] = (char)c;
        }
    }
    free(line);
    (void)len;
    return v;
}

int main(int argc, char** argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s <alpacc lexer .cu> <output .bin>\n", argv[0]); return 1; }
    FILE* f = fopen(argv[1], "r");
    if (!f) { perror(argv[1]); return 1; }
    size_t nt, nc, na;
    uint32_t* to_state = read_array(f, "const state_t h_to_state", &nt);
    uint32_t* compose  = read_array(f, "const state_t h_compose", &nc);
    uint32_t* accept   = read_array(f, "const bool h_accept", &na);
    fclose(f);
    if (nc != na * na) { fprintf(stderr, "compose has %zu entries, expected %zu^2\n", nc, na); return 1; }

    FILE* o = fopen(argv[2], "wb");
    if (!o) { perror(argv[2]); return 1; }
    const char magic[8] = {'L', 'X', 'D', 'F', 'A', '1', 0, 0};
    const uint32_t hdr[2] = {(uint32_t)na, (uint32_t)nt};
    fwrite(magic, 1, 8, o); fwrite(hdr, sizeof(uint32_t), 2, o);
    for (size_t i = 0; i < nt; i++) { uint16_t x = (uint16_t)to_state[i]; fwrite(&x, 2, 1, o); }
    for (size_t i = 0; i < nc; i++) { uint16_t x = (uint16_t)compose[i]; fwrite(&x, 2, 1, o); }
    for (size_t i = 0; i < na; i++) { uint8_t x = (uint8_t)accept[i]; fwrite(&x, 1, 1, o); }
    fclose(o);
    fprintf(stderr, "packed: %zu states, %zu transitions\n", na, nt);
    return 0;
}
