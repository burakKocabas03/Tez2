CXX      = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -march=native

TARGETS = tsp_serial max_clique_serial knapsack_serial

all: $(TARGETS)

tsp_serial: tsp_serial.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<
	@echo "  Built: $@"

max_clique_serial: max_clique_serial.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<
	@echo "  Built: $@"

knapsack_serial: knapsack_serial.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<
	@echo "  Built: $@"

clean:
	rm -f $(TARGETS)

.PHONY: all clean
