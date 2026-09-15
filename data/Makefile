CC?=cc
CFLAGS?=-O3 -Wall -Werror -Wextra -pedantic -g
TOKENS=tokens

default: $(TOKENS)_dense_500MiB.in \
		 $(TOKENS)_moderate_500MiB.in \
		 $(TOKENS)_sparse_500MiB.in

.PHONY: clean

$(TOKENS): $(TOKENS).c
	$(CC) $(CFLAGS) -o $(TOKENS) $<

$(TOKENS)_dense_500MiB.in: $(TOKENS)
	./$< 524288000 0:10 > $@ 

$(TOKENS)_moderate_500MiB.in: $(TOKENS)
	./$< 524288000 100:110 > $@ 

$(TOKENS)_sparse_500MiB.in: $(TOKENS)
	./$< 524288000 1000:1010 > $@ 

clean:
	rm -f $(TOKENS) $(TOKENS)_*.in