/**
 * WP1 – Baseline Serial Implementation
 * ======================================
 * Problem  : Traveling Salesman Problem (TSP)
 * Algorithm: Simulated Annealing with 2-opt neighbourhood
 *
 * Thesis   : A Comparison of NP-hard Problems through Parallel Algorithms
 *            Utilizing CPU and GPU-Based Solutions
 * Authors  : Burak Kocabaş & Emin Özgür Elmalı
 * Advisor  : Prof. Dr. Hasan BULUT
 *
 * Build    : g++ -O3 -std=c++17 -o tsp_serial tsp_serial.cpp
 * Run      : ./tsp_serial <tsp_file> [max_iter] [init_temp] [cooling_rate] [seed]
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <limits>
#include <string>
#include <numeric>

// ---------------------------------------------------------------------------
//  Data structures
// ---------------------------------------------------------------------------

struct City {
    int    id;
    double x, y;
};

/**
 * Flat (row-major) distance matrix.
 * Storing d[i][j] as data[i*n + j] gives better cache locality than
 * a vector-of-vectors, especially during the inner 2-opt loop.
 */
struct DistMatrix {
    int                 n = 0;
    std::vector<double> data;

    explicit DistMatrix(int n_) : n(n_), data(static_cast<size_t>(n_) * n_, 0.0) {}

    inline double get(int i, int j) const { return data[i * n + j]; }

    void build(const std::vector<City>& cities) {
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double dx = cities[i].x - cities[j].x;
                double dy = cities[i].y - cities[j].y;
                double d  = std::sqrt(dx * dx + dy * dy);
                data[i * n + j] = d;
                data[j * n + i] = d;
            }
        }
    }
};

// ---------------------------------------------------------------------------
//  Tour utilities
// ---------------------------------------------------------------------------

double tourCost(const std::vector<int>& tour, const DistMatrix& dm) {
    double total = 0.0;
    const int n  = static_cast<int>(tour.size());
    for (int i = 0; i < n; ++i)
        total += dm.get(tour[i], tour[(i + 1) % n]);
    return total;
}

/**
 * Nearest-Neighbour heuristic: greedy construction of an initial tour.
 * Starting from startCity, always move to the closest unvisited city.
 * Time complexity: O(n^2).
 */
std::vector<int> nearestNeighbourTour(const DistMatrix& dm, int startCity = 0) {
    const int n = dm.n;
    std::vector<bool> visited(n, false);
    std::vector<int>  tour;
    tour.reserve(n);

    int curr      = startCity;
    visited[curr] = true;
    tour.push_back(curr);

    for (int step = 1; step < n; ++step) {
        double best = std::numeric_limits<double>::max();
        int    next = -1;
        for (int j = 0; j < n; ++j) {
            if (!visited[j] && dm.get(curr, j) < best) {
                best = dm.get(curr, j);
                next = j;
            }
        }
        visited[next] = true;
        tour.push_back(next);
        curr = next;
    }
    return tour;
}

// ---------------------------------------------------------------------------
//  Simulated Annealing (2-opt moves)
// ---------------------------------------------------------------------------

struct SAResult {
    std::vector<int> bestTour;
    double           bestCost;
    long long        itersRun;
    long long        acceptedMoves;
};

/**
 * Core SA loop.
 *
 * At each iteration a random 2-opt move is generated:
 *   - Two indices i < j (with j-i >= 2) are chosen.
 *   - Remove edges (tour[i], tour[i+1]) and (tour[j], tour[j+1]).
 *   - Add    edges (tour[i], tour[j])   and (tour[i+1], tour[j+1]).
 *   - This reverses the sub-segment tour[i+1 .. j].
 *
 * The move is accepted if it improves cost, or with probability
 * exp(-delta / T) if it worsens cost (Metropolis criterion).
 *
 * Temperature is cooled geometrically: T <- T * coolingRate each iteration.
 */
SAResult simulatedAnnealing(const DistMatrix& dm,
                            double            initTemp,
                            double            coolingRate,
                            long long         maxIter,
                            unsigned int      seed = 42u)
{
    const int n = dm.n;

    std::mt19937                           rng(seed);
    std::uniform_int_distribution<int>     cityDist(0, n - 1);
    std::uniform_real_distribution<double> probDist(0.0, 1.0);

    std::vector<int> tour     = nearestNeighbourTour(dm, 0);
    double           cost     = tourCost(tour, dm);
    std::vector<int> bestTour = tour;
    double           bestCost = cost;

    double    T        = initTemp;
    long long accepted = 0;
    long long iter     = 0;

    for (; iter < maxIter && T > 1e-10; ++iter) {
        int i = cityDist(rng);
        int j = cityDist(rng);
        if (i == j) continue;
        if (i > j)  std::swap(i, j);
        // Skip trivial or wrap-around degenerate moves
        if (j - i < 2)              continue;
        if (i == 0 && j == n - 1)   continue;

        const int a = tour[i],         b = tour[i + 1];
        const int c = tour[j],         d = tour[(j + 1) % n];

        // Cost change if we apply this 2-opt swap
        const double delta = dm.get(a, c) + dm.get(b, d)
                           - dm.get(a, b) - dm.get(c, d);

        if (delta < 0.0 || probDist(rng) < std::exp(-delta / T)) {
            std::reverse(tour.begin() + i + 1, tour.begin() + j + 1);
            cost += delta;
            ++accepted;
            if (cost < bestCost) {
                bestCost = cost;
                bestTour = tour;
            }
        }

        T *= coolingRate;
    }

    return {bestTour, bestCost, iter, accepted};
}

// ---------------------------------------------------------------------------
//  TSPLIB reader  (EUC_2D instances)
// ---------------------------------------------------------------------------

std::vector<City> readTSPLIB(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open file: " << filename << "\n";
        std::exit(EXIT_FAILURE);
    }

    std::vector<City> cities;
    std::string       line;
    bool              inNodes = false;

    while (std::getline(file, line)) {
        // Strip trailing CR / spaces
        while (!line.empty() && (line.back() == '\r' || line.back() == ' '))
            line.pop_back();

        if (line == "NODE_COORD_SECTION") { inNodes = true;  continue; }
        if (line == "EOF")                {                   break;    }

        if (inNodes && !line.empty()) {
            std::istringstream iss(line);
            City c;
            if (iss >> c.id >> c.x >> c.y) {
                c.id--;          // convert to 0-indexed
                cities.push_back(c);
            }
        }
    }
    return cities;
}

// ---------------------------------------------------------------------------
//  Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr
            << "Usage: " << argv[0]
            << " <tsp_file> [max_iter] [init_temp] [cooling_rate] [seed]\n\n"
            << "  tsp_file:     TSPLIB EUC_2D instance\n"
            << "  max_iter:     SA iterations      (default: n*n*100)\n"
            << "  init_temp:    initial temperature (default: 1000.0)\n"
            << "  cooling_rate: geometric factor    (default: 0.99995)\n"
            << "  seed:         RNG seed            (default: 42)\n";
        return EXIT_FAILURE;
    }

    // ── Load instance ────────────────────────────────────────────────────────
    const auto cities = readTSPLIB(argv[1]);
    const int  n      = static_cast<int>(cities.size());

    // ── Parse optional parameters ────────────────────────────────────────────
    const long long  maxIter    = (argc > 2) ? std::stoll(argv[2])         : (long long)n * n * 100;
    const double     initTemp   = (argc > 3) ? std::stod(argv[3])          : 1000.0;
    // Auto-compute coolingRate so T decays from initTemp to ~1e-9 over all iterations.
    // Using exp(log(ratio)/N) is numerically stable for large N where pow() loses precision.
    const double     coolingRate= (argc > 4) ? std::stod(argv[4])
                                 : std::exp(std::log(1e-9 / initTemp) / static_cast<double>(maxIter));
    const unsigned   seed       = (argc > 5) ? (unsigned)std::stoul(argv[5]): 42u;

    std::cout << "═══════════════════════════════════════════════════════\n";
    std::cout << " WP1 – TSP Serial (Simulated Annealing)\n";
    std::cout << "═══════════════════════════════════════════════════════\n";
    std::cout << " Instance    : " << argv[1] << "  (" << n << " cities)\n";
    std::cout << " max_iter    : " << maxIter    << "\n";
    std::cout << " init_temp   : " << initTemp   << "\n";
    std::cout << " cooling_rate: " << coolingRate << "\n";
    std::cout << " seed        : " << seed        << "\n";
    std::cout << "───────────────────────────────────────────────────────\n";

    // ── Build distance matrix ────────────────────────────────────────────────
    DistMatrix dm(n);
    dm.build(cities);

    // ── Nearest-Neighbour baseline (initial solution quality) ────────────────
    const auto   nnTour = nearestNeighbourTour(dm, 0);
    const double nnCost = tourCost(nnTour, dm);

    // ── Run SA ───────────────────────────────────────────────────────────────
    const auto t0 = std::chrono::high_resolution_clock::now();
    const auto res = simulatedAnnealing(dm, initTemp, coolingRate, maxIter, seed);
    const auto t1 = std::chrono::high_resolution_clock::now();

    const double elapsed     = std::chrono::duration<double>(t1 - t0).count();
    const double improvement = 100.0 * (nnCost - res.bestCost) / nnCost;
    const double acceptRate  = 100.0 * static_cast<double>(res.acceptedMoves)
                                     / static_cast<double>(res.itersRun);

    // ── Report ───────────────────────────────────────────────────────────────
    std::cout << std::fixed << std::setprecision(2);
    std::cout << " Nearest-Neighbour cost : " << nnCost        << "\n";
    std::cout << " SA best tour cost      : " << res.bestCost  << "\n";
    std::cout << " Improvement over NN    : " << improvement   << " %\n";
    std::cout << " Accept rate            : " << acceptRate    << " %"
              << "  (" << res.acceptedMoves << " / " << res.itersRun << ")\n";
    std::cout << std::setprecision(6);
    std::cout << " Execution time         : " << elapsed        << " s\n";
    std::cout << "═══════════════════════════════════════════════════════\n";

    // Machine-readable CSV line for benchmark aggregation
    // Format: n, cost, time_s, accepted, total_iters
    std::cout << "CSV," << n << "," << std::setprecision(4)
              << res.bestCost << "," << elapsed << ","
              << res.acceptedMoves << "," << res.itersRun << "\n";

    return EXIT_SUCCESS;
}
