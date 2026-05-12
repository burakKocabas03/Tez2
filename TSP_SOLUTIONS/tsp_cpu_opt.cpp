/**
 * WP4 – CPU-Optimized TSP
 * ========================
 * Algorithm: EAX-inspired Genetic Algorithm with 2-opt refinement
 *
 * This is a simplified EAX-inspired crossover implementation for TSP.
 * It is not a full state-of-the-art EAX implementation, but it follows the idea
 * of:
 *   - Building AB-cycles from two parent tours
 *   - Removing selected A-edges from parent A
 *   - Adding selected B-edges from parent B
 *   - Repairing/merging subtours
 *   - Refining offspring using 2-opt
 *
 * Main safety improvements:
 *   - Valid tour/permutation checking
 *   - Population snapshot during parallel crossover to avoid data races
 *   - popSize and numThreads validation
 *   - More reasonable trials per generation
 *   - Fallback to parent A if crossover repair fails
 *
 * Build:
 *   g++ -O3 -std=c++17 -fopenmp -o tsp_cpu_opt tsp_cpu_opt.cpp
 *
 * Run:
 *   ./tsp_cpu_opt random <n> [pop_size] [max_gen] [num_threads]
 *   ./tsp_cpu_opt file   <tsp_file> [pop_size] [max_gen] [num_threads]
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <omp.h>

// ---------------------------------------------------------------------------
// City & I/O
// ---------------------------------------------------------------------------

struct City {
  int id;
  double x, y;
};

std::vector<City> generateRandomCities(int n, unsigned seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> coordDist(0.0, 1000.0);

  std::vector<City> cities(n);
  for (int i = 0; i < n; ++i) {
    cities[i].id = i;
    cities[i].x = coordDist(rng);
    cities[i].y = coordDist(rng);
  }

  return cities;
}

std::vector<City> readTSPLIB(const std::string &filename) {
  std::ifstream file(filename);

  if (!file.is_open()) {
    std::cerr << "[ERROR] Cannot open file: " << filename << "\n";
    std::exit(1);
  }

  std::vector<City> cities;
  std::string line;
  bool inNodes = false;

  while (std::getline(file, line)) {
    while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) {
      line.pop_back();
    }

    if (line == "NODE_COORD_SECTION") {
      inNodes = true;
      continue;
    }

    if (line == "EOF") {
      break;
    }

    if (inNodes && !line.empty()) {
      std::istringstream iss(line);
      City c;

      if (iss >> c.id >> c.x >> c.y) {
        c.id--;
        cities.push_back(c);
      }
    }
  }

  return cities;
}

// ---------------------------------------------------------------------------
// Distance Matrix
// ---------------------------------------------------------------------------

struct DistMatrix {
  int n = 0;
  std::vector<double> data;

  explicit DistMatrix(int n_) : n(n_), data((size_t)n_ * n_, 0.0) {}

  inline double get(int i, int j) const { return data[(size_t)i * n + j]; }

  void build(const std::vector<City> &cities) {
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        double dx = cities[i].x - cities[j].x;
        double dy = cities[i].y - cities[j].y;
        double d = std::sqrt(dx * dx + dy * dy);

        data[(size_t)i * n + j] = d;
        data[(size_t)j * n + i] = d;
      }
    }
  }
};

// ---------------------------------------------------------------------------
// Tour utilities
// ---------------------------------------------------------------------------

bool isValidTour(const std::vector<int> &tour, int n) {
  if ((int)tour.size() != n) {
    return false;
  }

  std::vector<int> seen(n, 0);

  for (int city : tour) {
    if (city < 0 || city >= n) {
      return false;
    }

    seen[city]++;

    if (seen[city] > 1) {
      return false;
    }
  }

  return true;
}

double tourCost(const std::vector<int> &tour, const DistMatrix &dm) {
  double total = 0.0;
  int n = (int)tour.size();

  for (int i = 0; i < n; ++i) {
    total += dm.get(tour[i], tour[(i + 1) % n]);
  }

  return total;
}

std::vector<int> nearestNeighbourTour(const DistMatrix &dm, int startCity) {
  int n = dm.n;

  std::vector<bool> visited(n, false);
  std::vector<int> tour;
  tour.reserve(n);

  int curr = startCity;
  visited[curr] = true;
  tour.push_back(curr);

  for (int step = 1; step < n; ++step) {
    double best = std::numeric_limits<double>::infinity();
    int next = -1;

    for (int j = 0; j < n; ++j) {
      if (!visited[j] && dm.get(curr, j) < best) {
        best = dm.get(curr, j);
        next = j;
      }
    }

    if (next == -1) {
      break;
    }

    visited[next] = true;
    tour.push_back(next);
    curr = next;
  }

  return tour;
}

std::vector<int> randomTour(int n, std::mt19937 &rng) {
  std::vector<int> tour(n);
  std::iota(tour.begin(), tour.end(), 0);
  std::shuffle(tour.begin(), tour.end(), rng);
  return tour;
}

// ---------------------------------------------------------------------------
// 2-opt local search
// ---------------------------------------------------------------------------

void twoOptImprove(std::vector<int> &tour, double &cost, const DistMatrix &dm,
                   int maxPasses) {
  const int n = (int)tour.size();

  if (n < 4) {
    return;
  }

  for (int pass = 0; pass < maxPasses; ++pass) {
    bool improved = false;

    for (int i = 0; i < n - 1; ++i) {
      for (int j = i + 2; j < n; ++j) {
        if (i == 0 && j == n - 1) {
          continue;
        }

        int a = tour[i];
        int b = tour[i + 1];
        int c = tour[j];
        int d = tour[(j + 1) % n];

        double delta =
            dm.get(a, c) + dm.get(b, d) - dm.get(a, b) - dm.get(c, d);

        if (delta < -1e-10) {
          std::reverse(tour.begin() + i + 1, tour.begin() + j + 1);
          cost += delta;
          improved = true;
        }
      }
    }

    if (!improved) {
      break;
    }
  }
}

// ---------------------------------------------------------------------------
// AB-cycle decomposition
// ---------------------------------------------------------------------------

struct ABCycle {
  std::vector<std::pair<int, int>> aEdges;
  std::vector<std::pair<int, int>> bEdges;
};

static void buildAdj(const std::vector<int> &tour, int n,
                     std::vector<std::array<int, 2>> &adj) {
  adj.assign(n, {-1, -1});

  for (int i = 0; i < n; ++i) {
    int v = tour[i];
    int prev = tour[(i - 1 + n) % n];
    int next = tour[(i + 1) % n];

    adj[v] = {prev, next};
  }
}

static inline int getUnused(const std::vector<std::array<bool, 2>> &used,
                            int v) {
  if (!used[v][0]) {
    return 0;
  }

  if (!used[v][1]) {
    return 1;
  }

  return -1;
}

static void markEdgeUsed(const std::vector<std::array<int, 2>> &adj,
                         std::vector<std::array<bool, 2>> &used, int u, int v) {
  for (int s = 0; s < 2; ++s) {
    if (adj[u][s] == v && !used[u][s]) {
      used[u][s] = true;
      break;
    }
  }

  for (int s = 0; s < 2; ++s) {
    if (adj[v][s] == u && !used[v][s]) {
      used[v][s] = true;
      break;
    }
  }
}

std::vector<ABCycle> findABCycles(const std::vector<int> &tourA,
                                  const std::vector<int> &tourB, int n) {
  std::vector<std::array<int, 2>> adjA(n);
  std::vector<std::array<int, 2>> adjB(n);

  buildAdj(tourA, n, adjA);
  buildAdj(tourB, n, adjB);

  std::vector<std::array<bool, 2>> usedA(n, {false, false});
  std::vector<std::array<bool, 2>> usedB(n, {false, false});

  // Mark shared undirected edges as used.
  for (int v = 0; v < n; ++v) {
    for (int sa = 0; sa < 2; ++sa) {
      if (usedA[v][sa]) {
        continue;
      }

      int nbA = adjA[v][sa];

      for (int sb = 0; sb < 2; ++sb) {
        if (usedB[v][sb]) {
          continue;
        }

        if (adjB[v][sb] == nbA) {
          usedA[v][sa] = true;
          usedB[v][sb] = true;

          for (int s2 = 0; s2 < 2; ++s2) {
            if (adjA[nbA][s2] == v && !usedA[nbA][s2]) {
              usedA[nbA][s2] = true;
              break;
            }
          }

          for (int s2 = 0; s2 < 2; ++s2) {
            if (adjB[nbA][s2] == v && !usedB[nbA][s2]) {
              usedB[nbA][s2] = true;
              break;
            }
          }

          break;
        }
      }
    }
  }

  std::vector<ABCycle> cycles;

  for (int start = 0; start < n; ++start) {
    while (true) {
      int ai = getUnused(usedA, start);

      if (ai == -1) {
        break;
      }

      ABCycle cycle;
      int cur = start;
      bool ok = true;

      int safetyCounter = 0;
      const int safetyLimit = 4 * n + 10;

      do {
        if (++safetyCounter > safetyLimit) {
          ok = false;
          break;
        }

        int aIdx = getUnused(usedA, cur);

        if (aIdx == -1) {
          ok = false;
          break;
        }

        int next = adjA[cur][aIdx];
        markEdgeUsed(adjA, usedA, cur, next);
        cycle.aEdges.push_back({cur, next});
        cur = next;

        int bIdx = getUnused(usedB, cur);

        if (bIdx == -1) {
          ok = false;
          break;
        }

        next = adjB[cur][bIdx];
        markEdgeUsed(adjB, usedB, cur, next);
        cycle.bEdges.push_back({cur, next});
        cur = next;

      } while (cur != start);

      if (ok && !cycle.aEdges.empty() &&
          cycle.aEdges.size() == cycle.bEdges.size()) {
        cycles.push_back(std::move(cycle));
      }
    }
  }

  return cycles;
}

// ---------------------------------------------------------------------------
// Apply E-set and merge subtours
// ---------------------------------------------------------------------------

static bool removeEdgeFromAdjList(std::vector<std::vector<int>> &graph, int u,
                                  int v) {
  auto removeOne = [](std::vector<int> &list, int x) -> bool {
    auto it = std::find(list.begin(), list.end(), x);

    if (it == list.end()) {
      return false;
    }

    list.erase(it);
    return true;
  };

  bool a = removeOne(graph[u], v);
  bool b = removeOne(graph[v], u);

  return a && b;
}

static bool addEdgeToAdjList(std::vector<std::vector<int>> &graph, int u,
                             int v) {
  if (u == v) {
    return false;
  }

  if ((int)graph[u].size() >= 2 || (int)graph[v].size() >= 2) {
    return false;
  }

  if (std::find(graph[u].begin(), graph[u].end(), v) != graph[u].end()) {
    return false;
  }

  graph[u].push_back(v);
  graph[v].push_back(u);

  return true;
}

static std::vector<std::vector<int>>
extractComponentsFromDegreeTwoGraph(const std::vector<std::vector<int>> &graph,
                                    int n) {
  std::vector<bool> visited(n, false);
  std::vector<std::vector<int>> components;

  for (int s = 0; s < n; ++s) {
    if (visited[s]) {
      continue;
    }

    std::vector<int> comp;
    std::vector<int> stack = {s};
    visited[s] = true;

    while (!stack.empty()) {
      int u = stack.back();
      stack.pop_back();
      comp.push_back(u);

      for (int v : graph[u]) {
        if (!visited[v]) {
          visited[v] = true;
          stack.push_back(v);
        }
      }
    }

    components.push_back(std::move(comp));
  }

  return components;
}

static std::vector<int>
orderComponentAsPathOrCycle(const std::vector<std::vector<int>> &graph,
                            const std::vector<int> &component) {
  if (component.empty()) {
    return {};
  }

  int start = component[0];

  // Prefer an endpoint if this component is a path.
  for (int v : component) {
    if ((int)graph[v].size() <= 1) {
      start = v;
      break;
    }
  }

  std::vector<int> ordered;
  ordered.reserve(component.size());

  int prev = -1;
  int cur = start;

  for (size_t steps = 0; steps < component.size(); ++steps) {
    ordered.push_back(cur);

    int next = -1;

    for (int nb : graph[cur]) {
      if (nb != prev) {
        next = nb;
        break;
      }
    }

    if (next == -1) {
      break;
    }

    prev = cur;
    cur = next;

    if (cur == start) {
      break;
    }
  }

  // Fallback: if something was not traversed due to an unexpected structure,
  // append missing nodes in arbitrary order. Later isValidTour will verify.
  if (ordered.size() != component.size()) {
    for (int v : component) {
      if (std::find(ordered.begin(), ordered.end(), v) == ordered.end()) {
        ordered.push_back(v);
      }
    }
  }

  return ordered;
}

std::vector<int> greedyMergeComponents(std::vector<std::vector<int>> subtours,
                                       const DistMatrix &dm) {
  if (subtours.empty()) {
    return {};
  }

  while (subtours.size() > 1) {
    double bestDelta = std::numeric_limits<double>::infinity();
    int bestI = -1;
    int bestJ = -1;
    int bestPi = -1;
    int bestPj = -1;
    bool bestReverseSecond = false;

    for (int i = 0; i < (int)subtours.size(); ++i) {
      for (int j = i + 1; j < (int)subtours.size(); ++j) {
        const auto &s1 = subtours[i];
        const auto &s2 = subtours[j];

        int n1 = (int)s1.size();
        int n2 = (int)s2.size();

        for (int pi = 0; pi < n1; ++pi) {
          int a = s1[pi];
          int b = s1[(pi + 1) % n1];

          for (int pj = 0; pj < n2; ++pj) {
            int c = s2[pj];
            int d = s2[(pj + 1) % n2];

            double deltaNormal =
                dm.get(a, c) + dm.get(b, d) - dm.get(a, b) - dm.get(c, d);

            if (deltaNormal < bestDelta) {
              bestDelta = deltaNormal;
              bestI = i;
              bestJ = j;
              bestPi = pi;
              bestPj = pj;
              bestReverseSecond = false;
            }

            double deltaReversed =
                dm.get(a, d) + dm.get(b, c) - dm.get(a, b) - dm.get(c, d);

            if (deltaReversed < bestDelta) {
              bestDelta = deltaReversed;
              bestI = i;
              bestJ = j;
              bestPi = pi;
              bestPj = pj;
              bestReverseSecond = true;
            }
          }
        }
      }
    }

    if (bestI == -1 || bestJ == -1) {
      return {};
    }

    auto s1 = subtours[bestI];
    auto s2 = subtours[bestJ];

    int n1 = (int)s1.size();
    int n2 = (int)s2.size();

    std::vector<int> part1;
    part1.reserve(n1);

    for (int k = 1; k <= n1; ++k) {
      part1.push_back(s1[(bestPi + k) % n1]);
    }

    std::vector<int> part2;
    part2.reserve(n2);

    for (int k = 1; k <= n2; ++k) {
      part2.push_back(s2[(bestPj + k) % n2]);
    }

    if (!bestReverseSecond) {
      std::reverse(part2.begin(), part2.end());
    }

    std::vector<int> merged;
    merged.reserve(n1 + n2);

    merged.insert(merged.end(), part1.begin(), part1.end());
    merged.insert(merged.end(), part2.begin(), part2.end());

    if (bestJ > bestI) {
      subtours.erase(subtours.begin() + bestJ);
      subtours[bestI] = std::move(merged);
    } else {
      subtours.erase(subtours.begin() + bestI);
      subtours[bestJ] = std::move(merged);
    }
  }

  return subtours[0];
}

std::vector<int> applyESetAndMerge(const std::vector<int> &tourA,
                                   const ABCycle &cycle, const DistMatrix &dm,
                                   int n) {
  std::vector<std::vector<int>> graph(n);

  // Build undirected degree-2 graph from parent A.
  for (int i = 0; i < n; ++i) {
    int u = tourA[i];
    int v = tourA[(i + 1) % n];

    graph[u].push_back(v);
    graph[v].push_back(u);
  }

  // Remove selected A-edges.
  for (auto [u, v] : cycle.aEdges) {
    if (!removeEdgeFromAdjList(graph, u, v)) {
      return {};
    }
  }

  // Add selected B-edges.
  for (auto [u, v] : cycle.bEdges) {
    if (!addEdgeToAdjList(graph, u, v)) {
      return {};
    }
  }

  // Degree safety check.
  for (int v = 0; v < n; ++v) {
    if ((int)graph[v].size() > 2) {
      return {};
    }
  }

  auto components = extractComponentsFromDegreeTwoGraph(graph, n);

  std::vector<std::vector<int>> subtours;
  subtours.reserve(components.size());

  for (const auto &comp : components) {
    auto ordered = orderComponentAsPathOrCycle(graph, comp);

    if (!ordered.empty()) {
      subtours.push_back(std::move(ordered));
    }
  }

  auto offspring = greedyMergeComponents(std::move(subtours), dm);

  if (!isValidTour(offspring, n)) {
    return {};
  }

  return offspring;
}

// ---------------------------------------------------------------------------
// EAX-inspired crossover
// ---------------------------------------------------------------------------

std::vector<int> eaxInspiredCrossover(const std::vector<int> &tourA,
                                      const std::vector<int> &tourB,
                                      const DistMatrix &dm, int n,
                                      std::mt19937 &rng) {
  if (!isValidTour(tourA, n) || !isValidTour(tourB, n)) {
    return {};
  }

  auto cycles = findABCycles(tourA, tourB, n);

  if (cycles.empty()) {
    return {};
  }

  std::uniform_int_distribution<int> cycleDist(0, (int)cycles.size() - 1);
  int chosen = cycleDist(rng);

  auto offspring = applyESetAndMerge(tourA, cycles[chosen], dm, n);

  if (!isValidTour(offspring, n)) {
    return {};
  }

  return offspring;
}

// ---------------------------------------------------------------------------
// Population structure
// ---------------------------------------------------------------------------

struct Individual {
  std::vector<int> tour;
  double cost = std::numeric_limits<double>::infinity();
};

// ---------------------------------------------------------------------------
// Main GA
// ---------------------------------------------------------------------------

Individual runEAXInspiredGA(const DistMatrix &dm, int popSize, int maxGen,
                            int numThreads) {
  const int n = dm.n;

  popSize = std::max(popSize, 2);
  numThreads = std::max(numThreads, 1);
  numThreads = std::min(numThreads, omp_get_max_threads());

  omp_set_num_threads(numThreads);

  std::vector<Individual> population(popSize);

// Phase 1: Initialize diverse-ish population.
#pragma omp parallel for schedule(static)
  for (int i = 0; i < popSize; ++i) {
    std::mt19937 rng(1234u + (unsigned)i * 17u);

    if (i < n) {
      int startCity = (i * n) / popSize;
      population[i].tour = nearestNeighbourTour(dm, startCity);
    } else {
      population[i].tour = randomTour(n, rng);
    }

    if (!isValidTour(population[i].tour, n)) {
      population[i].tour = randomTour(n, rng);
    }

    population[i].cost = tourCost(population[i].tour, dm);
    int initTwoOptPasses = (n > 1000) ? 1 : 5;
    twoOptImprove(population[i].tour, population[i].cost, dm, initTwoOptPasses);

    // Recompute cost to avoid tiny accumulated numerical drift.
    population[i].cost = tourCost(population[i].tour, dm);
  }

  Individual globalBest = population[0];

  for (const auto &ind : population) {
    if (ind.cost < globalBest.cost) {
      globalBest = ind;
    }
  }

  int stagnation = 0;
  const int maxStagnation = std::max(100, n);

  // More meaningful than only numThreads trials per generation.
  const int trialsPerGeneration = std::min(popSize, numThreads * 2);

  for (int gen = 0; gen < maxGen && stagnation < maxStagnation; ++gen) {
    bool anyImproved = false;

#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      std::mt19937 rng(42u + (unsigned)gen * 1000003u + (unsigned)tid * 9973u);

#pragma omp for schedule(dynamic)
      for (int trial = 0; trial < trialsPerGeneration; ++trial) {
        std::uniform_int_distribution<int> popDist(0, popSize - 1);

        int pA = popDist(rng);
        int pB = popDist(rng);

        while (pB == pA) {
          pB = popDist(rng);
        }

        std::vector<int> parentA;
        std::vector<int> parentB;

#pragma omp critical
        {
          parentA = population[pA].tour;
          parentB = population[pB].tour;
        }

        auto offspring = eaxInspiredCrossover(parentA, parentB, dm, n, rng);

        // Fallback: if simplified EAX repair fails, use a randomized copy of
        // parent A. This keeps the algorithm robust instead of silently losing
        // too many trials.
        if (!isValidTour(offspring, n)) {
          offspring = parentA;

          // Mild mutation: reverse a random segment.
          if (n >= 4) {
            std::uniform_int_distribution<int> idxDist(0, n - 1);
            int i = idxDist(rng);
            int j = idxDist(rng);

            if (i > j) {
              std::swap(i, j);
            }

            if (j - i >= 2) {
              std::reverse(offspring.begin() + i, offspring.begin() + j + 1);
            }
          }
        }

        if (!isValidTour(offspring, n)) {
          continue;
        }

        double offCost = tourCost(offspring, dm);
        int offspringTwoOptPasses = (n > 1000) ? 1 : 3;
        twoOptImprove(offspring, offCost, dm, offspringTwoOptPasses);
        offCost = tourCost(offspring, dm);

        if (!isValidTour(offspring, n)) {
          continue;
        }

#pragma omp critical
        {
          int worst = 0;

          for (int k = 1; k < popSize; ++k) {
            if (population[k].cost > population[worst].cost) {
              worst = k;
            }
          }

          if (offCost < population[worst].cost) {
            population[worst].tour = std::move(offspring);
            population[worst].cost = offCost;
            anyImproved = true;

            if (offCost < globalBest.cost) {
              globalBest = population[worst];
            }
          }
        }
      }
    }

    if (!anyImproved) {
      ++stagnation;
    } else {
      stagnation = 0;
    }
  }

  for (const auto &ind : population) {
    if (ind.cost < globalBest.cost) {
      globalBest = ind;
    }
  }

  return globalBest;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage:\n"
              << "  " << argv[0]
              << " random <n> [pop_size] [max_gen] [num_threads]\n"
              << "  " << argv[0]
              << " file   <tsp_file> [pop_size] [max_gen] [num_threads]\n";

    return 1;
  }

  std::vector<City> cities;
  std::string instanceDesc;

  const std::string mode = argv[1];

  if (mode == "random") {
    int n0 = std::stoi(argv[2]);

    if (n0 < 4) {
      std::cerr << "[ERROR] Need at least 4 cities.\n";
      return 1;
    }

    cities = generateRandomCities(n0);
    instanceDesc = "random " + std::to_string(n0) + " cities (seed=42)";
  } else if (mode == "file") {
    cities = readTSPLIB(argv[2]);
    instanceDesc = argv[2];
  } else {
    std::cerr << "[ERROR] First argument must be 'random' or 'file'.\n";
    return 1;
  }

  int n = (int)cities.size();

  if (n < 4) {
    std::cerr << "[ERROR] Need at least 4 cities for this algorithm.\n";
    return 1;
  }

  int popSize = (argc > 3) ? std::stoi(argv[3]) : std::min(n, 30);
  int maxGen = (argc > 4) ? std::stoi(argv[4]) : std::min(n * 5, 3000);
  int numThreads = (argc > 5) ? std::stoi(argv[5]) : omp_get_max_threads();

  if (popSize < 2) {
    std::cerr << "[WARN] pop_size must be at least 2. Using pop_size = 2.\n";
    popSize = 2;
  }

  if (maxGen < 1) {
    std::cerr << "[WARN] max_gen must be at least 1. Using max_gen = 1.\n";
    maxGen = 1;
  }

  if (numThreads < 1) {
    std::cerr
        << "[WARN] num_threads must be at least 1. Using num_threads = 1.\n";
    numThreads = 1;
  }

  if (numThreads > omp_get_max_threads()) {
    numThreads = omp_get_max_threads();
  }

  DistMatrix dm(n);
  dm.build(cities);

  auto nnTour = nearestNeighbourTour(dm, 0);

  if (!isValidTour(nnTour, n)) {
    std::cerr
        << "[ERROR] Failed to construct initial nearest-neighbour tour.\n";
    return 1;
  }

  double nnCost = tourCost(nnTour, dm);

  auto t0 = std::chrono::high_resolution_clock::now();

  Individual best = runEAXInspiredGA(dm, popSize, maxGen, numThreads);

  auto t1 = std::chrono::high_resolution_clock::now();

  if (!isValidTour(best.tour, n)) {
    std::cerr << "[ERROR] Algorithm returned an invalid tour.\n";
    return 1;
  }

  double elapsed = std::chrono::duration<double>(t1 - t0).count();
  double improvement = 100.0 * (nnCost - best.cost) / nnCost;

  std::cout << std::fixed << std::setprecision(2);

  std::cout << "═══════════════════════════════════════════════════════\n"
            << " CPU-Optimized TSP\n"
            << " EAX-inspired Genetic Algorithm with 2-opt refinement\n"
            << " Threads        : " << numThreads << "\n"
            << "═══════════════════════════════════════════════════════\n"
            << " Instance       : " << instanceDesc << "\n"
            << " Cities         : " << n << "\n"
            << " Population     : " << popSize << "\n"
            << " Max generations: " << maxGen << "\n"
            << " NN cost        : " << nnCost << "\n"
            << " Best cost      : " << best.cost << "\n"
            << " Improvement    : " << improvement << " %\n"
            << std::setprecision(6) << " Execution time : " << elapsed << " s\n"
            << "═══════════════════════════════════════════════════════\n";

  std::cout << "CSV," << n << "," << numThreads << "," << std::setprecision(4)
            << best.cost << "," << elapsed << "\n";

  return 0;
}