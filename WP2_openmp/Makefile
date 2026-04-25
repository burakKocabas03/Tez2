CXX      = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -march=native

TARGETS = tsp_openmp max_clique_openmp knapsack_openmp

# ── OpenMP flag detection ────────────────────────────────────────────────────
UNAME := $(shell uname -s)

ifeq ($(UNAME), Darwin)
    LIBOMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null)
    ifeq ($(LIBOMP_PREFIX),)
        $(error "libomp not found. Install with: brew install libomp")
    endif
    CXXFLAGS += -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include
    LDFLAGS   = -L$(LIBOMP_PREFIX)/lib -lomp
else
    CXXFLAGS += -fopenmp
    LDFLAGS   =
endif

all: $(TARGETS)

tsp_openmp: tsp_openmp.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)
	@echo "  Built: $@"

max_clique_openmp: max_clique_openmp.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)
	@echo "  Built: $@"

knapsack_openmp: knapsack_openmp.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)
	@echo "  Built: $@"

clean:
	rm -f $(TARGETS)

.PHONY: all clean
