/**
 * WP4 – CPU-Optimized TSP
 * ========================
 * Algorithm: Simulated Annealing + 2-opt + Or-opt (segment relocation)
 *
 * Improvement over WP1/WP2:
 *   - Or-opt moves: relocate segments of 1, 2, or 3 cities to better positions
 *   - Mixed neighbourhood: 70% 2-opt, 30% Or-opt moves
 *   - Single-row reverse NN for faster initial tour
 *   - Cache-friendly flat distance matrix (same as before)
 *
 * Or-opt is better suited for CPU because:
 *   - Complex branching logic (segment size selection)
 *   - Irregular memory access patterns
 *   - Benefits from branch prediction + large L1/L2 cache
 *
 * Build: g++ -O3 -std=c++17 -fopenmp -o tsp_cpu_opt tsp_cpu_opt.cpp
 * Run:   ./tsp_cpu_opt <tsp_file> [max_iter] [init_temp] [num_threads]
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
#include <omp.h>

struct City { int id; double x, y; };

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

SAResult runSA_OrOpt(const DistMatrix& dm, double initTemp, double coolRate,
                     long long maxIter, int startCity, unsigned seed) {
    const int n = dm.n;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> cityDist(0, n - 1);
    std::uniform_real_distribution<double> probDist(0.0, 1.0);
    std::uniform_int_distribution<int> moveDist(0, 9);
    std::uniform_int_distribution<int> segDist(1, 3);

    auto tour = nearestNeighbourTour(dm, startCity);
    double cost = tourCost(tour, dm);
    auto bestTour = tour;
    double bestCost = cost;
    double T = initTemp;

    for (long long iter = 0; iter < maxIter && T > 1e-10; ++iter) {
        int moveType = moveDist(rng);

        if (moveType < 7) {
            // 2-opt move (70% of the time)
            int i = cityDist(rng), j = cityDist(rng);
            if (i == j) continue;
            if (i > j) std::swap(i, j);
            if (j - i < 2 || (i == 0 && j == n - 1)) continue;

            int a = tour[i], b = tour[i+1], c = tour[j], d = tour[(j+1)%n];
            double delta = dm.get(a,c) + dm.get(b,d) - dm.get(a,b) - dm.get(c,d);

            if (delta < 0.0 || probDist(rng) < std::exp(-delta / T)) {
                std::reverse(tour.begin() + i + 1, tour.begin() + j + 1);
                cost += delta;
                if (cost < bestCost) { bestCost = cost; bestTour = tour; }
            }
        } else {
            // Or-opt move (30%): relocate segment of 1-3 cities
            int segLen = segDist(rng);
            if (segLen >= n - 1) continue;
            int pos = cityDist(rng);
            int insertAt = cityDist(rng);
            if (insertAt == pos || std::abs(insertAt - pos) <= segLen) continue;

            // Compute cost change for removing segment and inserting elsewhere
            int segEnd = (pos + segLen - 1) % n;
            int beforeSeg = (pos - 1 + n) % n;
            int afterSeg = (segEnd + 1) % n;
            int beforeIns = insertAt;
            int afterIns = (insertAt + 1) % n;

            // Ensure we don't overlap
            if (beforeIns == segEnd || afterIns == pos) continue;

            double removeCost = dm.get(tour[beforeSeg], tour[pos])
                              + dm.get(tour[segEnd], tour[afterSeg]);
            double removeGain = dm.get(tour[beforeSeg], tour[afterSeg]);
            double insertCost = dm.get(tour[beforeIns], tour[afterIns]);
            double insertPay  = dm.get(tour[beforeIns], tour[pos])
                              + dm.get(tour[segEnd], tour[afterIns]);

            double delta = -removeCost + removeGain - insertCost + insertPay;

            if (delta < 0.0 || probDist(rng) < std::exp(-delta / T)) {
                // Extract segment
                std::vector<int> seg;
                for (int k = 0; k < segLen; ++k)
                    seg.push_back(tour[(pos + k) % n]);

                // Remove from tour
                std::vector<int> newTour;
                newTour.reserve(n);
                for (int k = 0; k < n; ++k) {
                    bool inSeg = false;
                    for (int s = 0; s < segLen; ++s)
                        if (k == (pos + s) % n) { inSeg = true; break; }
                    if (!inSeg) newTour.push_back(tour[k]);
                }

                // Find insert position in reduced tour
                int insIdx = -1;
                for (int k = 0; k < (int)newTour.size(); ++k)
                    if (newTour[k] == tour[beforeIns]) { insIdx = k + 1; break; }
                if (insIdx < 0 || insIdx > (int)newTour.size()) {
                    continue;
                }

                newTour.insert(newTour.begin() + insIdx, seg.begin(), seg.end());

                if ((int)newTour.size() == n) {
                    double newCost = tourCost(newTour, dm);
                    double realDelta = newCost - cost;
                    if (realDelta < 0.0 || probDist(rng) < std::exp(-realDelta / T)) {
                        tour = newTour;
                        cost = newCost;
                        if (cost < bestCost) { bestCost = cost; bestTour = tour; }
                    }
                }
            }
        }
        T *= coolRate;
    }
    return {bestTour, bestCost};
}

std::vector<City> readTSPLIB(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) { std::cerr << "Cannot open " << filename << "\n"; std::exit(1); }
    std::vector<City> cities;
    std::string line;
    bool inNodes = false;
    while (std::getline(file, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (line == "NODE_COORD_SECTION") { inNodes = true; continue; }
        if (line == "EOF") break;
        if (inNodes && !line.empty()) {
            std::istringstream iss(line);
            City c;
            if (iss >> c.id >> c.x >> c.y) { c.id--; cities.push_back(c); }
        }
    }
    return cities;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <tsp_file> [max_iter] [init_temp] [num_threads]\n";
        return 1;
    }

    auto cities = readTSPLIB(argv[1]);
    int n = (int)cities.size();
    long long maxIter = (argc > 2) ? std::stoll(argv[2]) : (long long)n * n * 10;
    double initTemp = (argc > 3) ? std::stod(argv[3]) : 1000.0;
    int numThreads = (argc > 4) ? std::stoi(argv[4]) : omp_get_max_threads();
    omp_set_num_threads(numThreads);

    long long itersPerThread = maxIter / numThreads;
    double coolRate = std::exp(std::log(1e-9 / initTemp) / (double)itersPerThread);

    DistMatrix dm(n);
    dm.build(cities);

    double nnCost = tourCost(nearestNeighbourTour(dm, 0), dm);

    std::vector<int> globalBestTour;
    double globalBestCost = 1e30;

    auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int startCity = (tid * n) / numThreads;
        unsigned seed = 42u + (unsigned)tid * 1000u;

        auto res = runSA_OrOpt(dm, initTemp, coolRate, itersPerThread, startCity, seed);

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
              << " CPU-Optimized TSP (SA + Or-opt, " << numThreads << " threads)\n"
              << "═══════════════════════════════════════════════════════\n"
              << " Instance       : " << argv[1] << "  (" << n << " cities)\n"
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
