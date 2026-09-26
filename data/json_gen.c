// Random JSON input for the JSON DFA benchmark (grammars/json.alp of alpacc):
// objects, arrays, strings of [a-zA-Z0-9 ], numbers -?[0-9]+(.[0-9]+)?,
// null/true/false, and whitespace runs between tokens. Top-level values are
// appended until the next one would not fit, the rest is padded with spaces,
// so the file is exactly N bytes of lexically valid JSON.
// Output: Futhark binary u8 array (the format read_u8_array expects).
//   usage: json_gen <bytes> <output file>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint32_t rnd(uint32_t n) {   // xorshift64*, uniform in [0, n)
    rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
    return (uint32_t)((rng * 0x2545f4914f6cdd1dull) >> 32) % n;
}

static char* buf; static size_t len, cap;
static void put(char c) { if (len < cap) buf[len++] = c; }
static void puts_(const char* s) { while (*s) put(*s++); }

static void ws(void) {   // whitespace run between tokens (often empty)
    static const char w[] = " \n\t\r";
    if (rnd(3) == 0) { uint32_t n = 1 + rnd(4); while (n--) put(w[rnd(4)]); }
}
static void string_(void) {
    static const char c[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ";
    uint32_t n = rnd(17);
    put('"'); while (n--) put(c[rnd(sizeof(c) - 1)]); put('"');
}
static void number(void) {
    if (rnd(4) == 0) put('-');
    uint32_t n = 1 + rnd(8); while (n--) put('0' + rnd(10));
    if (rnd(3) == 0) { put('.'); n = 1 + rnd(6); while (n--) put('0' + rnd(10)); }
}
static void value(int depth) {
    uint32_t k = depth >= 6 ? 2 + rnd(4) : rnd(8);
    switch (k) {
    case 0: case 1: {   // object / array
        const int obj = k == 0;
        put(obj ? '{' : '['); ws();
        uint32_t n = rnd(7);
        for (uint32_t i = 0; i < n; i++) {
            if (i) { put(','); ws(); }
            if (obj) { string_(); ws(); put(':'); ws(); }
            value(depth + 1); ws();
        }
        put(obj ? '}' : ']');
        break;
    }
    case 2: case 3: string_(); break;
    case 4: case 5: number(); break;
    case 6: puts_(rnd(2) ? "true" : "false"); break;
    default: puts_("null"); break;
    }
}

int main(int argc, char** argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s <bytes> <output>\n", argv[0]); return 1; }
    const size_t n = strtoull(argv[1], NULL, 10);
    char* out = malloc(n);
    size_t used = 0;
    cap = 1 << 22; buf = malloc(cap);
    for (;;) {   // one top-level value at a time, into buf
        len = 0; value(0); ws(); put('\n');
        if (len == cap || used + len > n) break;
        memcpy(out + used, buf, len); used += len;
    }
    memset(out + used, ' ', n - used);

    FILE* f = fopen(argv[2], "wb");
    if (!f) { perror(argv[2]); return 1; }
    const char header[7] = {'b', 2, 1, ' ', ' ', 'u', '8'};
    const uint64_t size = n;
    fwrite(header, 1, sizeof(header), f);
    fwrite(&size, sizeof(size), 1, f);
    fwrite(out, 1, n, f);
    fclose(f);
    return 0;
}
