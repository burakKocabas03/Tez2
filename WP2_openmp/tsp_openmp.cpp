/**
 * WP2 – CPU Parallelization (OpenMP)
 * ====================================
 * Problem   : Traveling Salesman Problem (TSP)
 * Algorithm : Parallel Simulated Annealing via Independent Chains (Island Model)
 *
 * Parallel Strategy
 * -----------------
 * P threads each run an independent SA chain for maxIter/P iterations.
 * Every chain starts from a different city (diverse initial solutions) and uses
 * a unique RNG seed.  At the end, the best tour found by any thread is returned.
 *
 * Why this gives speedup
 * -----------------------
 *   - Threads share NO mutable state during the SA loop (pure data parallelism).
 *   - The only synchronisation is a single critical section at the very end to
 *     collect results.
 *   - Expected wall-clock speedup ≈ P (near-linear scaling).
 *
 * Comparison with WP1 (serial)
 * -----------------------------
 *   - Same total number of SA iterations (maxIter) split evenly across threads.
 *   - Measure: execution time reduction (speedup = T_serial / T_parallel).
 *   - Solution quality is equal or better (P diverse starting points).
 *
 * Thesis   : A Comparison of NP-hard Problems through Parallel Algorithms
 *            Utilizing CPU and GPU-Based Solutions
 * Author   : Burak Kocabaş
 * Advisor  : Prof. Dr. Hasan BULUT
 *
 * Build    : see Makefile  (requires libomp on macOS: brew install libomp)
 * Run      : ./tsp_openmp random <n> [max_iter] [init_temp] [cooling_rate] [num_threads]
 *            ./tsp_openmp file   <tsp_file> [max_iter] [init_temp] [cooling_rate] [num_threads]
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

#include <omp.h>

// ---------------------------------------------------------------------------
//  Random instance generator
// ---------------------------------------------------------------------------

struct City {
    int    id;
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

std::vector<City> readTSPLIB(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open file: " << filename << "\n";
        std::exit(EXIT_FAILURE);
    }
    std::vector<City> cities;
    std::string line;
    bool inNodes = false;
    while (std::getline(file, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' '))
            line.pop_back();
        if (line == "NODE_COORD_SECTION") { inNodes = true;  continue; }
        if (line == "EOF")                { break; }
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
//  Data structures  (identical to WP1 for a fair comparison)
// ---------------------------------------------------------------------------

/**
 * Flat, row-major distance matrix shared (read-only) across all threads.
 * Thread-safe: no writes after construction.
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
 * Nearest-Neighbour heuristic.
 * Each thread calls this with a different startCity to generate diverse
 * initial solutions, improving overall exploration.
 */
std::vector<int> nearestNeighbourTour(const DistMatrix& dm, int startCity) {
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
//  Per-thread SA result
// ---------------------------------------------------------------------------

struct ThreadResult {
    std::vector<int> bestTour;
    double           bestCost      = std::numeric_limits<double>::max();
    long long        acceptedMoves = 0;
    long long        itersRun      = 0;
    int              threadId      = -1;
};

// ---------------------------------------------------------------------------
//  Single-thread SA kernel  (identical logic to WP1)
// ---------------------------------------------------------------------------

ThreadResult runSA(const DistMatrix& dm,
                   double            initTemp,
                   double            coolingRate,
                   long long         maxIter,
                   int               startCity,
                   unsigned int      seed,
                   int               tid)
{
    const int n = dm.n;

    std::mt19937                           rng(seed);
    std::uniform_int_distribution<int>     cityDist(0, n - 1);
    std::uniform_real_distribution<double> probDist(0.0, 1.0);

    std::vector<int> tour     = nearestNeighbourTour(dm, startCity);
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
        if (j - i < 2)             continue;
        if (i == 0 && j == n - 1)  continue;

        const int    a = tour[i],         b = tour[i + 1];
        const int    c = tour[j],         d = tour[(j + 1) % n];
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

    return {bestTour, bestCost, accepted, iter, tid};
}


// ---------------------------------------------------------------------------
//  Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr
            << "Usage:\n  " << argv[0]
            << " random <n> [max_iter] [init_temp] [cooling_rate] [num_threads]\n  "
            << argv[0]
            << " file   <tsp_file> [max_iter] [init_temp] [cooling_rate] [num_threads]\n";
        return EXIT_FAILURE;
    }

    const std::string mode = argv[1];
    std::vector<City> cities;
    std::string instanceDesc;

    if (mode == "random") {
        const int n = std::stoi(argv[2]);
        cities = generateRandomCities(n);
        instanceDesc = "random " + std::to_string(n) + " cities (seed=42)";
    } else if (mode == "file") {
        cities = readTSPLIB(argv[2]);
        instanceDesc = std::string(argv[2]);
    } else {
        std::cerr << "[ERROR] First argument must be 'random' or 'file'.\n";
        return EXIT_FAILURE;
    }

    const int n = static_cast<int>(cities.size());
    if (n < 2) {
        std::cerr << "[ERROR] Need at least 2 cities.\n";
        return EXIT_FAILURE;
    }

    // Optional args start at argv[3] for both modes
    const long long maxIter  = (argc > 3) ? std::stoll(argv[3]) : (long long)n * n * 100;
    const double    initTemp = (argc > 4) ? std::stod(argv[4]) : 1000.0;

    int numThreads = (argc > 6) ? std::stoi(argv[6]) : omp_get_max_threads();
    if (numThreads > omp_get_max_threads())
        numThreads = omp_get_max_threads();
    omp_set_num_threads(numThreads);

    const long long itersForCooling = std::max(1LL, maxIter / numThreads);
    const double    coolingRate = (argc > 5) ? std::stod(argv[5])
                                 : std::exp(std::log(1e-9 / initTemp) / static_cast<double>(itersForCooling));

    DistMatrix dm(n);
    dm.build(cities);

    const auto   nnTour = nearestNeighbourTour(dm, 0);
    const double nnCost = tourCost(nnTour, dm);

    const long long itersBase      = maxIter / numThreads;
    const long long itersRemainder = maxIter % numThreads;

    std::cout << "═══════════════════════════════════════════════════════\n";
    std::cout << " WP2 – TSP OpenMP Parallel (Simulated Annealing)\n";
    std::cout << "═══════════════════════════════════════════════════════\n";
    std::cout << " Instance       : " << instanceDesc << "\n";
    std::cout << " max_iter       : " << maxIter        << "\n";
    std::cout << " iters/thread   : " << itersBase << " (+" << itersRemainder << " remainder)\n";
    std::cout << " init_temp      : " << initTemp       << "\n";
    std::cout << " cooling_rate   : " << coolingRate    << "\n";
    std::cout << " threads        : " << numThreads     << "\n";
    std::cout << "───────────────────────────────────────────────────────\n";

    // ── Parallel SA ──────────────────────────────────────────────────────────
    std::vector<int> globalBestTour = nnTour;
    double           globalBestCost = nnCost;
    long long        totalAccepted  = 0;
    long long        totalIters     = 0;

    const auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel num_threads(numThreads)
    {
        const int tid = omp_get_thread_num();

        // Unique starting city and RNG seed per thread ensures each chain
        // explores a different region of the search space.
        const int          startCity = (tid * n) / numThreads;
        const unsigned int seed      = 42u + static_cast<unsigned>(tid) * 1000u;

        // Last thread absorbs remainder iterations
        const long long myIters = itersBase + ((tid == numThreads - 1) ? itersRemainder : 0);

        ThreadResult res = runSA(dm, initTemp, coolingRate, myIters,
                                 startCity, seed, tid);

        // Update global best — only one thread at a time
        #pragma omp critical
        {
            totalAccepted += res.acceptedMoves;
            totalIters    += res.itersRun;
            if (res.bestCost < globalBestCost) {
                globalBestCost = res.bestCost;
                globalBestTour = res.bestTour;
            }
        }
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    const double elapsed     = std::chrono::duration<double>(t1 - t0).count();
    const double improvement = 100.0 * (nnCost - globalBestCost) / nnCost;
    const double acceptRate  = 100.0 * static_cast<double>(totalAccepted)
                                     / static_cast<double>(totalIters);

    // ── Report ───────────────────────────────────────────────────────────────
    std::cout << std::fixed << std::setprecision(2);
    std::cout << " Nearest-Neighbour cost : " << nnCost          << "\n";
    std::cout << " SA best tour cost      : " << globalBestCost  << "\n";
    std::cout << " Improvement over NN    : " << improvement     << " %\n";
    std::cout << " Accept rate            : " << acceptRate      << " %"
              << "  (" << totalAccepted << " / " << totalIters << ")\n";
    std::cout << " Threads used           : " << numThreads      << "\n";
    std::cout << std::setprecision(6);
    std::cout << " Execution time         : " << elapsed         << " s\n";
    std::cout << "═══════════════════════════════════════════════════════\n";

    // Machine-readable CSV line
    // Format: n, threads, cost, time_s, accepted, total_iters
    std::cout << "CSV," << n << "," << numThreads << ","
              << std::setprecision(4)
              << globalBestCost << "," << elapsed << ","
              << totalAccepted  << "," << totalIters << "\n";

    return EXIT_SUCCESS;
}
