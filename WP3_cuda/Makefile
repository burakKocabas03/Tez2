NVCC     = nvcc
NVFLAGS  = -O3 -std=c++14 -arch=sm_75
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# sm_70 = Volta (V100), sm_75 = Turing (T4/RTX 2000),
# sm_80 = Ampere (A100/RTX 3000), sm_86 = Ampere (RTX 3060+)
# sm_89 = Ada Lovelace (RTX 4000)
# Change -arch to match your GPU. Use: nvcc --list-gpu-arch
# Or use: -gencode=arch=compute_70,code=sm_70 for multi-arch

TARGETS = tsp_cuda max_clique_cuda knapsack_cuda

all: $(TARGETS)

tsp_cuda: tsp_cuda.cu
	$(NVCC) $(NVFLAGS) -o $@ $<
	@echo "  Built: $@"

max_clique_cuda: max_clique_cuda.cu
	$(NVCC) $(NVFLAGS) -o $@ $<
	@echo "  Built: $@"

knapsack_cuda: knapsack_cuda.cu
	$(NVCC) $(NVFLAGS) -o $@ $<
	@echo "  Built: $@"

clean:
	rm -f $(TARGETS)

.PHONY: all clean
