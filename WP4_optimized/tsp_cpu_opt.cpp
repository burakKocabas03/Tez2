/**
 * WP4 – CPU-Optimized TSP
 * ========================
 * Algorithm: Simulated Annealing + 2-opt + 3-opt (in-place segment reverse)
 *
 * CPU advantages exploited:
 *   - 3-opt provides deeper local search per iteration
 *   - Branch prediction handles complex move selection
 *   - Large L1/L2 cache holds flat distance matrix
 *   - Multi-start via OpenMP island model
 *
 * Build: g++ -O3 -std=c++17 -fopenmp -o tsp_cpu_opt tsp_cpu_opt.cpp
 * Run:   ./tsp_cpu_opt random <n> [max_iter] [init_temp] [num_threads]
 *        ./tsp_cpu_opt file   <tsp_file> [max_iter] [init_temp] [num_threads]
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

std::vector<City> readTSPLIB(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open file: " << filename << "\n";
        std::exit(1);
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

struct DistMatrix {
    int n = 0;
    std::vector<double> data;
    explicit DistMatrix(int n_) : n(n_), data((size_t)n_ * n_, 0.0) {}
    inline double get(int i, int j) const { return data[i * n + j]; }
    void build(const std::vector<City>& cities) {
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j) {
                double dx = cities[i].x - cities[j].x;
                double dy = cities[i].y - cities[j].y;
                double d = std::sqrt(dx * dx + dy * dy);
                data[i * n + j] = d;
                data[j * n + i] = d;
            }
    }
};

double tourCost(const std::vector<int>& tour, const DistMatrix& dm) {
    double total = 0.0;
    int n = (int)tour.size();
    for (int i = 0; i < n; ++i)
        total += dm.get(tour[i], tour[(i + 1) % n]);
    return total;
}

std::vector<int> nearestNeighbourTour(const DistMatrix& dm, int startCity) {
    int n = dm.n;
    std::vector<bool> visited(n, false);
    std::vector<int> tour;
    tour.reserve(n);
    int curr = startCity;
    visited[curr] = true;
    tour.push_back(curr);
    for (int step = 1; step < n; ++step) {
        double best = 1e30;
        int next = -1;
        for (int j = 0; j < n; ++j)
            if (!visited[j] && dm.get(curr, j) < best) {
                best = dm.get(curr, j);
                next = j;
            }
        visited[next] = true;
        tour.push_back(next);
        curr = next;
    }
    return tour;
}

struct SAResult {
    std::vector<int> bestTour;
    double bestCost;
};

SAResult runSA_3opt(const DistMatrix& dm, double initTemp, double coolRate,
                    long long maxIter, int startCity, unsigned seed) {
    const int n = dm.n;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> cityDist(0, n - 1);
    std::uniform_real_distribution<double> probDist(0.0, 1.0);
    std::uniform_int_distribution<int> moveDist(0, 9);

    auto tour = nearestNeighbourTour(dm, startCity);
    double cost = tourCost(tour, dm);
    auto bestTour = tour;
    double bestCost = cost;
    double T = initTemp;

    for (long long iter = 0; iter < maxIter && T > 1e-10; ++iter) {
        int moveType = moveDist(rng);

        if (moveType < 7) {
            // 2-opt (70%): reverse segment [i+1..j]
            int i = cityDist(rng), j = cityDist(rng);
            if (i == j) { T *= coolRate; continue; }
            if (i > j) std::swap(i, j);
            if (j - i < 2 || (i == 0 && j == n - 1)) { T *= coolRate; continue; }

            double delta = dm.get(tour[i], tour[j]) + dm.get(tour[i+1], tour[(j+1)%n])
                         - dm.get(tour[i], tour[i+1]) - dm.get(tour[j], tour[(j+1)%n]);

            if (delta < 0.0 || probDist(rng) < std::exp(-delta / T)) {
                std::reverse(tour.begin() + i + 1, tour.begin() + j + 1);
                cost += delta;
                if (cost < bestCost) { bestCost = cost; bestTour = tour; }
            }
        } else {
            // 3-opt (30%): pick 3 random cut points, try best reconnection
            int a = cityDist(rng), b = cityDist(rng), c = cityDist(rng);
            // Sort so a < b < c
            if (a > b) std::swap(a, b);
            if (b > c) std::swap(b, c);
            if (a > b) std::swap(a, b);
            if (b - a < 2 || c - b < 2 || c >= n - 1) { T *= coolRate; continue; }

            int A1 = tour[a], A2 = tour[a+1];
            int B1 = tour[b], B2 = tour[b+1];
            int C1 = tour[c], C2 = tour[(c+1)%n];

            double d0 = dm.get(A1,A2) + dm.get(B1,B2) + dm.get(C1,C2);

            // Try reversal of segment [a+1..b] (2-opt between a and b)
            double d1 = dm.get(A1,B1) + dm.get(A2,B2) + dm.get(C1,C2);
            // Try reversal of segment [b+1..c] (2-opt between b and c)
            double d2 = dm.get(A1,A2) + dm.get(B1,C1) + dm.get(B2,C2);
            // Try both reversals
            double d3 = dm.get(A1,B1) + dm.get(A2,C1) + dm.get(B2,C2);

            double bestDelta = 1e30;
            int bestMove = -1;
            if (d1 - d0 < bestDelta) { bestDelta = d1 - d0; bestMove = 1; }
            if (d2 - d0 < bestDelta) { bestDelta = d2 - d0; bestMove = 2; }
            if (d3 - d0 < bestDelta) { bestDelta = d3 - d0; bestMove = 3; }

            if (bestDelta < 0.0 || probDist(rng) < std::exp(-bestDelta / T)) {
                if (bestMove == 1) {
                    std::reverse(tour.begin() + a + 1, tour.begin() + b + 1);
                } else if (bestMove == 2) {
                    std::reverse(tour.begin() + b + 1, tour.begin() + c + 1);
                } else if (bestMove == 3) {
                    std::reverse(tour.begin() + a + 1, tour.begin() + b + 1);
                    std::reverse(tour.begin() + b + 1, tour.begin() + c + 1);
                }
                cost = tourCost(tour, dm);
                if (cost < bestCost) { bestCost = cost; bestTour = tour; }
            }
        }
        T *= coolRate;
    }
    return {bestTour, bestCost};
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage:\n  " << argv[0]
                  << " random <n> [max_iter] [init_temp] [num_threads]\n  "
                  << argv[0] << " file   <tsp_file> [max_iter] [init_temp] [num_threads]\n";
        return 1;
    }

    std::vector<City> cities;
    std::string instanceDesc;
    const std::string mode = argv[1];
    if (mode == "random") {
        int n0 = std::stoi(argv[2]);
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
    if (n < 2) {
        std::cerr << "[ERROR] Need at least 2 cities.\n";
        return 1;
    }

    long long maxIter = (argc > 3) ? std::stoll(argv[3]) : (long long)n * n * 10;
    double initTemp = (argc > 4) ? std::stod(argv[4]) : 1000.0;
    int numThreads = (argc > 5) ? std::stoi(argv[5]) : omp_get_max_threads();
    omp_set_num_threads(numThreads);

    DistMatrix dm(n);
    dm.build(cities);

    const long long itersBase = maxIter / numThreads;
    const long long itersRem = maxIter % numThreads;
    const long long itersForCool = std::max(1LL, itersBase + itersRem);
    double coolRate = std::exp(std::log(1e-9 / initTemp) / (double)itersForCool);

    double nnCost = tourCost(nearestNeighbourTour(dm, 0), dm);

    std::vector<int> globalBestTour;
    double globalBestCost = 1e30;

    auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int startCity = (tid * n) / numThreads;
        unsigned seed = 42u + (unsigned)tid * 1000u;
        long long myIters = itersBase + ((tid == numThreads - 1) ? itersRem : 0);

        auto res = runSA_3opt(dm, initTemp, coolRate, myIters, startCity, seed);

        #pragma omp critical
        {
            if (res.bestCost < globalBestCost) {
                globalBestCost = res.bestCost;
                globalBestTour = res.bestTour;
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "═══════════════════════════════════════════════════════\n"
              << " CPU-Optimized TSP (SA + 2-opt/3-opt, " << numThreads << " threads)\n"
              << "═══════════════════════════════════════════════════════\n"
              << " Instance       : " << instanceDesc << "\n"
              << " NN cost        : " << nnCost << "\n"
              << " Best tour cost : " << globalBestCost << "\n"
              << " Improvement    : " << 100.0*(nnCost - globalBestCost)/nnCost << " %\n"
              << std::setprecision(6)
              << " Execution time : " << elapsed << " s\n"
              << "═══════════════════════════════════════════════════════\n";

    std::cout << "CSV," << n << "," << numThreads << ","
              << std::setprecision(4) << globalBestCost << "," << elapsed << "\n";
    return 0;
}
