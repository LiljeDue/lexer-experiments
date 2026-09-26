DATA_PATH=./data
COMMON_PATH=./common
FUTHARK_PROGRAM=futhark_lexer
CUDA_PROGRAM=cuda_lexer
CUDA_DEBUG_PROGRAM=cuda_lexer_debug
COMPILER?=nvcc
FLAGS?=-O3 --std=c++17 -diag-suppress 550 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80
GREEN=[32m
DEFAULT=\033[39m

default: bench

P1_BENCH_PROGRAM=p1_bench

.PHONY: clean bench test devinfo profile_p1 bench_p1 profile bench_json

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
	@echo "Compilation done, starting benchmarks..."

$(CUDA_DEBUG_PROGRAM): cuda_lexer.cu $(COMMON_PATH)/sps.cu.h $(COMMON_PATH)/util.cu.h $(COMMON_PATH)/data.h
	$(COMPILER) $(FLAGS) -DDEBUG -o $@ $<

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

$(P1_BENCH_PROGRAM): p1_bench.cu
	$(COMPILER) $(FLAGS) -o $@ $<

# JSON DFA (alpacc's grammars/json.alp, extracted into dfa/json.h): lexerBig
# only, on 500 MB of generated JSON; the output is not verified.
JSON_DATA=$(DATA_PATH)/json_500MiB.in
$(DATA_PATH)/json_gen: $(DATA_PATH)/json_gen.c
	$(CC) -O2 -o $@ $<

$(JSON_DATA): $(DATA_PATH)/json_gen
	$(DATA_PATH)/json_gen 524288000 $@

$(CUDA_PROGRAM)_json: cuda_lexer.cu dfa/json.h $(COMMON_PATH)/sps.cu.h $(COMMON_PATH)/util.cu.h $(COMMON_PATH)/data.h
	$(COMPILER) $(FLAGS) -DLEXER_DFA_JSON -o $@ $<

bench_json: $(CUDA_PROGRAM)_json $(JSON_DATA)
	@echo -e "$(GREEN)=== CUDA LEXER (JSON DFA) ===$(DEFAULT)"
	@./$(CUDA_PROGRAM)_json $(JSON_DATA)
	@echo -e "$(GREEN)============$(DEFAULT)"

bench_p1: $(P1_BENCH_PROGRAM) $(DATA_PATH)/tokens_dense_500MiB.in
	@echo -e "$(GREEN)=== P1 BENCH ===$(DEFAULT)"
	@./$(P1_BENCH_PROGRAM) $(DATA_PATH)/tokens_dense_500MiB.in
	@echo -e "$(GREEN)============$(DEFAULT)"

$(DATA_PATH)/tokens_dense_1GiB.in:
	$(DATA_PATH)/tokens 1073741824 0:10 > $@

$(DATA_PATH)/tokens_moderate_1GiB.in:
	$(DATA_PATH)/tokens 1073741824 100:110 > $@

$(DATA_PATH)/tokens_sparse_1GiB.in:
	$(DATA_PATH)/tokens 1073741824 1000:1010 > $@

tokens_indices_dense_1GiB.out: $(DATA_PATH)/tokens_dense_1GiB.in $(FUTHARK_PROGRAM)
	futhark script ./$(FUTHARK_PROGRAM) 'indices ($$loaddata "$<")' -b >$@

tokens_indices_moderate_1GiB.out: $(DATA_PATH)/tokens_moderate_1GiB.in $(FUTHARK_PROGRAM)
	futhark script ./$(FUTHARK_PROGRAM) 'indices ($$loaddata "$<")' -b >$@

tokens_indices_sparse_1GiB.out: $(DATA_PATH)/tokens_sparse_1GiB.in $(FUTHARK_PROGRAM)
	futhark script ./$(FUTHARK_PROGRAM) 'indices ($$loaddata "$<")' -b >$@

tokens_tokens_dense_1GiB.out: $(DATA_PATH)/tokens_dense_1GiB.in $(FUTHARK_PROGRAM)
	futhark script ./$(FUTHARK_PROGRAM) 'tokens ($$loaddata "$<")' -b >$@

tokens_tokens_moderate_1GiB.out: $(DATA_PATH)/tokens_moderate_1GiB.in $(FUTHARK_PROGRAM)
	futhark script ./$(FUTHARK_PROGRAM) 'tokens ($$loaddata "$<")' -b >$@

tokens_tokens_sparse_1GiB.out: $(DATA_PATH)/tokens_sparse_1GiB.in $(FUTHARK_PROGRAM)
	futhark script ./$(FUTHARK_PROGRAM) 'tokens ($$loaddata "$<")' -b >$@

bench1g: $(FUTHARK_PROGRAM) \
         $(DATA_PATH)/tokens_dense_1GiB.in \
         $(DATA_PATH)/tokens_moderate_1GiB.in \
         $(DATA_PATH)/tokens_sparse_1GiB.in \
         tokens_indices_dense_1GiB.out \
         tokens_indices_moderate_1GiB.out \
         tokens_indices_sparse_1GiB.out \
         tokens_tokens_dense_1GiB.out \
         tokens_tokens_moderate_1GiB.out \
         tokens_tokens_sparse_1GiB.out \
         $(CUDA_PROGRAM)
	@echo -e "$(GREEN)=== CUDA LEXER 1GiB ===$(DEFAULT)"
	@./$(CUDA_PROGRAM) $(DATA_PATH)/tokens_dense_1GiB.in tokens_indices_dense_1GiB.out tokens_tokens_dense_1GiB.out
	@echo ""
	@./$(CUDA_PROGRAM) $(DATA_PATH)/tokens_moderate_1GiB.in tokens_indices_moderate_1GiB.out tokens_tokens_moderate_1GiB.out
	@echo ""
	@./$(CUDA_PROGRAM) $(DATA_PATH)/tokens_sparse_1GiB.in tokens_indices_sparse_1GiB.out tokens_tokens_sparse_1GiB.out
	@echo -e "$(GREEN)============$(DEFAULT)"

test: $(CUDA_DEBUG_PROGRAM)
	@echo -e "$(GREEN)=== CUDA LEXER DEBUG TESTS ===$(DEFAULT)"
	@./$(CUDA_DEBUG_PROGRAM)
	@echo -e "$(GREEN)==============================$(DEFAULT)"

# Profile the lexer on one dataset: make profile [DATA=dense|moderate|sparse]
# (default dense). Writes profile_lexer_$(DATA).ncu-rep / .txt.
DATA ?= dense
profile: $(DATA_PATH)/tokens_$(DATA)_500MiB.in tokens_indices_$(DATA)_500MiB.out tokens_tokens_$(DATA)_500MiB.out
	$(COMPILER) $(FLAGS) -DPROFILE -lineinfo -o $(CUDA_PROGRAM)_profile cuda_lexer.cu
	-ncu --set full -f \
	    --clock-control none \
	    --import-source 1 \
	    --source-folders . \
	    --target-processes all \
	    --export profile_lexer_$(DATA) \
	    ./$(CUDA_PROGRAM)_profile $(DATA_PATH)/tokens_$(DATA)_500MiB.in tokens_indices_$(DATA)_500MiB.out tokens_tokens_$(DATA)_500MiB.out 2>&1
	-ncu --import profile_lexer_$(DATA).ncu-rep > profile_lexer_$(DATA).txt 2>&1

profile_p1: $(DATA_PATH)/tokens_dense_500MiB.in
	$(COMPILER) $(FLAGS) -DPROFILE -lineinfo -o $(P1_BENCH_PROGRAM)_profile p1_bench.cu
	-ncu --set full \
	    --import-source 1 \
	    --source-folders . \
	    --target-processes all \
	    --kernel-name-base demangled \
	    --kernel-id '::regex:(p1_ladder|p1_transpose)<:1' \
	    --export profile_p1 \
	    ./$(P1_BENCH_PROGRAM)_profile $(DATA_PATH)/tokens_dense_500MiB.in 2>&1
	-ncu --import profile_p1.ncu-rep > profile_p1.txt 2>&1

devinfo:
	$(COMPILER) $(FLAGS) -o devinfo devinfo.cu
	./devinfo
	rm -f devinfo

clean:
	rm -rf $(CUDA_PROGRAM) $(CUDA_PROGRAM)_json $(CUDA_DEBUG_PROGRAM) $(P1_BENCH_PROGRAM) $(P1_BENCH_PROGRAM)_profile $(FUTHARK_PROGRAM) *.out
	rm -f $(DATA_PATH)/json_gen $(JSON_DATA)
	rm -f $(DATA_PATH)/tokens_*_1GiB.in
