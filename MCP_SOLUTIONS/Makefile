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
SERIAL_TARGETS  = max_clique_serial
OMP_TARGETS     = max_clique_openmp
CUDA_TARGETS    = max_clique_cuda
CPU_OPT_TARGETS = max_clique_cpu_opt
GPU_OPT_TARGETS = max_clique_gpu_opt

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
max_clique_serial: max_clique_serial.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<
	@echo "  Built: $@"

max_clique_openmp: max_clique_openmp.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $< $(OMPLDFLAGS)
	@echo "  Built: $@"

max_clique_cuda: max_clique_cuda.cu
	$(NVCC) $(NVFLAGS) -o $@ $<
	@echo "  Built: $@"

max_clique_cpu_opt: max_clique_cpu_opt.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $< $(OMPLDFLAGS)
	@echo "  Built: $@"

max_clique_gpu_opt: max_clique_gpu_opt.cu
	$(NVCC) $(NVFLAGS) -o $@ $<
	@echo "  Built: $@"

clean:
	rm -f $(CPU_ALL) $(GPU_ALL) *.o
