CXX      = g++
NVCC     = nvcc
CXXFLAGS = -O3 -std=c++17
OMPFLAGS = -fopenmp
NVFLAGS  = -O3 -std=c++14 -arch=sm_75

# macOS libomp paths (Homebrew)
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S), Darwin)
    OMP_INC = -I/opt/homebrew/opt/libomp/include
    OMP_LIB = -L/opt/homebrew/opt/libomp/lib
    OMPFLAGS = -Xpreprocessor -fopenmp $(OMP_INC) $(OMP_LIB) -lomp
endif

CPU_TARGETS = tsp_cpu_opt max_clique_cpu_opt knapsack_cpu_opt
GPU_TARGETS = tsp_gpu_opt max_clique_gpu_opt knapsack_gpu_opt

.PHONY: all cpu gpu clean

all: cpu

cpu: $(CPU_TARGETS)

gpu: $(GPU_TARGETS)

tsp_cpu_opt: tsp_cpu_opt.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<

max_clique_cpu_opt: max_clique_cpu_opt.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<

knapsack_cpu_opt: knapsack_cpu_opt.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<

tsp_gpu_opt: tsp_gpu_opt.cu
	$(NVCC) $(NVFLAGS) -o $@ $<

max_clique_gpu_opt: max_clique_gpu_opt.cu
	$(NVCC) $(NVFLAGS) -o $@ $<

knapsack_gpu_opt: knapsack_gpu_opt.cu
	$(NVCC) $(NVFLAGS) -o $@ $<

clean:
	rm -f $(CPU_TARGETS) $(GPU_TARGETS) *.o
