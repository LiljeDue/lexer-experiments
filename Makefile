DATA_PATH=./data
COMMON_PATH=./common
FUTHARK_PROGRAM=futhark_lexer
CUDA_PROGRAM=cuda_lexer
CUDA_DEBUG_PROGRAM=cuda_lexer_debug
CUDA_PROFILE_PROGRAM=cuda_lexer_profile
COMPILER?=nvcc
FLAGS?=-O3 --std=c++14 -diag-suppress 550 -gencode arch=compute_80,code=sm_80
GREEN=[32m
DEFAULT=\033[39m

default: bench

.PHONY: clean bench test devinfo profile

$(DATA_PATH)/tokens_dense_500MiB.in:
	(cd $(DATA_PATH) && make)

$(DATA_PATH)/tokens_moderate_500MiB.in:
	(cd $(DATA_PATH) && make)

$(DATA_PATH)/tokens_sparse_500MiB.in:
	(cd $(DATA_PATH) && make)

tokens_indices_dense_500MiB.out: $(DATA_PATH)/tokens_dense_500MiB.in $(FUTHARK_PROGRAM)
	futhark script ./$(FUTHARK_PROGRAM) 'indices ($$loaddata "$<")' -b >$@

tokens_indices_moderate_500MiB.out: $(DATA_PATH)/tokens_moderate_500MiB.in $(FUTHARK_PROGRAM)
	futhark script ./$(FUTHARK_PROGRAM) 'indices ($$loaddata "$<")' -b >$@

tokens_indices_sparse_500MiB.out: $(DATA_PATH)/tokens_sparse_500MiB.in $(FUTHARK_PROGRAM)
	futhark script ./$(FUTHARK_PROGRAM) 'indices ($$loaddata "$<")' -b >$@

tokens_tokens_dense_500MiB.out: $(DATA_PATH)/tokens_dense_500MiB.in $(FUTHARK_PROGRAM)
	futhark script ./$(FUTHARK_PROGRAM) 'tokens ($$loaddata "$<")' -b >$@

tokens_tokens_moderate_500MiB.out: $(DATA_PATH)/tokens_moderate_500MiB.in $(FUTHARK_PROGRAM)
	futhark script ./$(FUTHARK_PROGRAM) 'tokens ($$loaddata "$<")' -b >$@

tokens_tokens_sparse_500MiB.out: $(DATA_PATH)/tokens_sparse_500MiB.in $(FUTHARK_PROGRAM)
	futhark script ./$(FUTHARK_PROGRAM) 'tokens ($$loaddata "$<")' -b >$@

$(FUTHARK_PROGRAM): $(FUTHARK_PROGRAM).fut
	futhark pkg sync
	futhark cuda $< -o $@ --server

$(CUDA_PROGRAM): cuda_lexer.cu $(COMMON_PATH)/sps.cu.h $(COMMON_PATH)/util.cu.h $(COMMON_PATH)/data.h
	$(COMPILER) $(FLAGS) -o $@ $<

$(CUDA_DEBUG_PROGRAM): cuda_lexer.cu $(COMMON_PATH)/sps.cu.h $(COMMON_PATH)/util.cu.h $(COMMON_PATH)/data.h
	$(COMPILER) $(FLAGS) -DDEBUG -o $@ $<

$(CUDA_PROFILE_PROGRAM): cuda_lexer.cu $(COMMON_PATH)/sps.cu.h $(COMMON_PATH)/util.cu.h $(COMMON_PATH)/data.h
	$(COMPILER) $(FLAGS) -DPROFILE -o $@ $<

bench: $(FUTHARK_PROGRAM) \
       $(DATA_PATH)/tokens_dense_500MiB.in \
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

profile: $(CUDA_PROFILE_PROGRAM) $(DATA_PATH)/tokens_dense_500MiB.in tokens_indices_dense_500MiB.out tokens_tokens_dense_500MiB.out
	{ ncu --set full \
	      --kernel-name-base function \
	      --kernel-name regex:lexerAlpacc \
	      --target-processes all \
	      ./$(CUDA_PROFILE_PROGRAM) $(DATA_PATH)/tokens_dense_500MiB.in tokens_indices_dense_500MiB.out tokens_tokens_dense_500MiB.out; \
	  ncu --set full \
	      --kernel-name-base function \
	      --kernel-name regex:lexerAlpaccShmemDyn \
	      --target-processes all \
	      ./$(CUDA_PROFILE_PROGRAM) $(DATA_PATH)/tokens_dense_500MiB.in tokens_indices_dense_500MiB.out tokens_tokens_dense_500MiB.out; \
	} > profile.txt 2>&1

diag: cuda_lexer.cu $(COMMON_PATH)/sps.cu.h $(COMMON_PATH)/util.cu.h $(COMMON_PATH)/data.h \
      $(DATA_PATH)/tokens_dense_500MiB.in tokens_indices_dense_500MiB.out tokens_tokens_dense_500MiB.out
	mkdir -p .claude-artifacts
	$(COMPILER) $(FLAGS) -DPROFILE -o .claude-artifacts/cuda_lexer_diag $<
	./.claude-artifacts/cuda_lexer_diag \
	  $(DATA_PATH)/tokens_dense_500MiB.in \
	  tokens_indices_dense_500MiB.out \
	  tokens_tokens_dense_500MiB.out

devinfo:
	$(COMPILER) $(FLAGS) -o devinfo devinfo.cu
	./devinfo
	rm -f devinfo

clean:
	rm -rf $(CUDA_PROGRAM) $(CUDA_DEBUG_PROGRAM) $(CUDA_PROFILE_PROGRAM) $(FUTHARK_PROGRAM) *.out
