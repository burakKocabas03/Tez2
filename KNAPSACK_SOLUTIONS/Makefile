CXX      = g++
NVCC     = nvcc
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -march=native
NVFLAGS  = -O3 -std=c++14 -arch=sm_75

# ── OpenMP flag detection ────────────────────────────────────────────────────
UNAME := $(shell uname -s)

ifeq ($(UNAME), Darwin)
    LIBOMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null)
    ifeq ($(LIBOMP_PREFIX),)
        $(warning "libomp not found. OpenMP targets will fail. Install with: brew install libomp")
    endif
    OMPFLAGS  = -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include
    OMPLDFLAGS = -L$(LIBOMP_PREFIX)/lib -lomp
else
    OMPFLAGS  = -fopenmp
    OMPLDFLAGS =
endif

# ── Targets ──────────────────────────────────────────────────────────────────
SERIAL_TARGETS  = knapsack_serial
OMP_TARGETS     = knapsack_openmp
CUDA_TARGETS    = knapsack_cuda
CPU_OPT_TARGETS = knapscak_cpu_opt
GPU_OPT_TARGETS = knapsack_gpu_opt

CPU_ALL  = $(SERIAL_TARGETS) $(OMP_TARGETS) $(CPU_OPT_TARGETS)
GPU_ALL  = $(CUDA_TARGETS) $(GPU_OPT_TARGETS)

.PHONY: all cpu gpu serial openmp cuda cpu_opt gpu_opt clean

all: cpu

cpu: $(CPU_ALL)
	@echo "  All CPU targets built."

gpu: $(GPU_ALL)
	@echo "  All GPU targets built."

serial: $(SERIAL_TARGETS)

openmp: $(OMP_TARGETS)

cuda: $(CUDA_TARGETS)

cpu_opt: $(CPU_OPT_TARGETS)

gpu_opt: $(GPU_OPT_TARGETS)

# ── Build rules ──────────────────────────────────────────────────────────────
knapsack_serial: knapsack_serial.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<
	@echo "  Built: $@"

knapsack_openmp: knapsack_openmp.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $< $(OMPLDFLAGS)
	@echo "  Built: $@"

knapsack_cuda: knapsack_cuda.cu
	$(NVCC) $(NVFLAGS) -o $@ $<
	@echo "  Built: $@"

knapscak_cpu_opt: knapscak_cpu_opt.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $< $(OMPLDFLAGS)
	@echo "  Built: $@"

knapsack_gpu_opt: knapsack_gpu_opt.cu
	$(NVCC) $(NVFLAGS) -o $@ $<
	@echo "  Built: $@"

clean:
	rm -f $(CPU_ALL) $(GPU_ALL) *.o
