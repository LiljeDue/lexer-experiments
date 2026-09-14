DATA_PATH=./data
COMMON_PATH=./common
CUDA_PROGRAM=cuda_lexer
CUDA_DEBUG_PROGRAM=cuda_lexer_debug
COMPILER?=nvcc
FLAGS?=-O3 --std=c++14 -diag-suppress 550 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80
RED=[31m
GREEN=[32m
DEFAULT=\033[39m

default: bench

.PHONY: clean bench test

$(DATA_PATH)/tokens_dense_500MiB.in:
	(cd $(DATA_PATH) && make)

$(DATA_PATH)/tokens_moderate_500MiB.in:
	(cd $(DATA_PATH) && make)

$(DATA_PATH)/tokens_sparse_500MiB.in:
	(cd $(DATA_PATH) && make)

tokens_indices_dense_500MiB.out: $(DATA_PATH)/tokens_dense_500MiB.in
	futhark script ./futhark_lexer 'indices ($$loaddata "$<")' -b >$@

tokens_indices_moderate_500MiB.out: $(DATA_PATH)/tokens_moderate_500MiB.in
	futhark script ./futhark_lexer 'indices ($$loaddata "$<")' -b >$@

tokens_indices_sparse_500MiB.out: $(DATA_PATH)/tokens_sparse_500MiB.in
	futhark script ./futhark_lexer 'indices ($$loaddata "$<")' -b >$@

tokens_tokens_dense_500MiB.out: $(DATA_PATH)/tokens_dense_500MiB.in
	futhark script ./futhark_lexer 'tokens ($$loaddata "$<")' -b >$@

tokens_tokens_moderate_500MiB.out: $(DATA_PATH)/tokens_moderate_500MiB.in
	futhark script ./futhark_lexer 'tokens ($$loaddata "$<")' -b >$@

tokens_tokens_sparse_500MiB.out: $(DATA_PATH)/tokens_sparse_500MiB.in
	futhark script ./futhark_lexer 'tokens ($$loaddata "$<")' -b >$@

$(CUDA_PROGRAM): cuda_lexer.cu $(COMMON_PATH)/sps.cu.h $(COMMON_PATH)/util.cu.h $(COMMON_PATH)/data.h
	$(COMPILER) $(FLAGS) -o $@ $<

$(CUDA_DEBUG_PROGRAM): cuda_lexer.cu $(COMMON_PATH)/sps.cu.h $(COMMON_PATH)/util.cu.h $(COMMON_PATH)/data.h
	$(COMPILER) $(FLAGS) -DDEBUG -o $@ $<

bench: $(DATA_PATH)/tokens_dense_500MiB.in \
       $(DATA_PATH)/tokens_moderate_500MiB.in \
       $(DATA_PATH)/tokens_sparse_500MiB.in \
       tokens_indices_dense_500MiB.out \
       tokens_indices_moderate_500MiB.out \
       tokens_indices_sparse_500MiB.out \
       tokens_tokens_dense_500MiB.out \
       tokens_tokens_moderate_500MiB.out \
       tokens_tokens_sparse_500MiB.out \
       $(CUDA_PROGRAM)
	@echo -e "$(GREEN)=== CUDA LEXER ===$(DEFAULT)"
	@./$(CUDA_PROGRAM) $(DATA_PATH)/tokens_dense_500MiB.in tokens_indices_dense_500MiB.out tokens_tokens_dense_500MiB.out
	@echo ""
	@./$(CUDA_PROGRAM) $(DATA_PATH)/tokens_moderate_500MiB.in tokens_indices_moderate_500MiB.out tokens_tokens_moderate_500MiB.out
	@echo ""
	@./$(CUDA_PROGRAM) $(DATA_PATH)/tokens_sparse_500MiB.in tokens_indices_sparse_500MiB.out tokens_tokens_sparse_500MiB.out
	@echo -e "$(GREEN)============$(DEFAULT)"

test: $(CUDA_DEBUG_PROGRAM)
	@echo -e "$(GREEN)=== CUDA LEXER DEBUG TESTS ===$(DEFAULT)"
	@./$(CUDA_DEBUG_PROGRAM)
	@echo -e "$(GREEN)==============================$(DEFAULT)"

clean:
	rm -rf $(CUDA_PROGRAM) $(CUDA_DEBUG_PROGRAM)
