%%writefile tsp_gpu_opt.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

static constexpr int INF = 0x3f3f3f3f;
static constexpr int THREADS_PER_BLOCK = 256;
static constexpr int DEFAULT_ITERATIONS = 30000;
static constexpr int DEFAULT_CHAINS_PER_SM = 64;
static constexpr int DEFAULT_LOCAL_SAMPLES = 8;
static constexpr int DEFAULT_LOCAL_ROUNDS = 12;

__device__ int d_global_best;

struct Instance {
    int n = 0;
    std::vector<double> x;
    std::vector<double> y;
    std::vector<int> dist;
};

static void die(const std::string& s) {
    std::cerr << s << "\n";
    std::exit(1);
}

static void cuda_check(cudaError_t e) {
    if (e != cudaSuccess) die(cudaGetErrorString(e));
}

static std::string upper_string(std::string s) {
    for (char& c : s) {
        if (c >= 'a' && c <= 'z') c = char(c - 'a' + 'A');
    }
    return s;
}

static Instance read_tsplib_euc2d(const std::string& path) {
    std::ifstream in(path);
    if (!in) die("cannot open input file");

    Instance inst;
    std::string line;
    bool in_coords = false;
    std::string edge_type;
    std::vector<std::tuple<int, double, double>> pts;

    while (std::getline(in, line)) {
        if (line.empty()) continue;

        std::string normalized = line;
        for (char& c : normalized) {
            if (c == ':') c = ' ';
        }

        std::istringstream ss(normalized);
        std::string key;
        ss >> key;
        key = upper_string(key);

        if (key == "DIMENSION") {
            ss >> inst.n;
        } else if (key == "EDGE_WEIGHT_TYPE") {
            ss >> edge_type;
            edge_type = upper_string(edge_type);
        } else if (key == "NODE_COORD_SECTION") {
            in_coords = true;
        } else if (key == "EOF") {
            break;
        } else if (in_coords) {
            std::istringstream cs(line);
            int id;
            double x;
            double y;
            if (cs >> id >> x >> y) {
                pts.emplace_back(id, x, y);
            }
        }
    }

    if (inst.n <= 2) die("invalid DIMENSION");
    if ((int)pts.size() != inst.n) die("NODE_COORD_SECTION count does not match DIMENSION");
    if (!edge_type.empty() && edge_type != "EUC_2D") die("this solver expects EDGE_WEIGHT_TYPE EUC_2D");

    std::sort(pts.begin(), pts.end(), [](const auto& a, const auto& b) {
        return std::get<0>(a) < std::get<0>(b);
    });

    inst.x.resize(inst.n);
    inst.y.resize(inst.n);
    inst.dist.assign((size_t)inst.n * inst.n, 0);

    for (int i = 0; i < inst.n; ++i) {
        inst.x[i] = std::get<1>(pts[i]);
        inst.y[i] = std::get<2>(pts[i]);
    }

    for (int i = 0; i < inst.n; ++i) {
        for (int j = 0; j < inst.n; ++j) {
            double dx = inst.x[i] - inst.x[j];
            double dy = inst.y[i] - inst.y[j];
            inst.dist[(size_t)i * inst.n + j] = (int)(std::sqrt(dx * dx + dy * dy) + 0.5);
        }
    }

    return inst;
}

static int cpu_dist(const Instance& inst, int a, int b) {
    return inst.dist[(size_t)a * inst.n + b];
}

static int cpu_tour_cost(const Instance& inst, const std::vector<int>& tour) {
    int n = inst.n;
    int cost = 0;

    for (int i = 0; i < n; ++i) {
        cost += cpu_dist(inst, tour[i], tour[(i + 1) % n]);
    }

    return cost;
}

static void cpu_reverse(std::vector<int>& tour, int l, int r) {
    while (l < r) {
        std::swap(tour[l], tour[r]);
        ++l;
        --r;
    }
}

static void cpu_two_opt_limited(const Instance& inst, std::vector<int>& tour, long long max_checks) {
    int n = inst.n;
    long long checks = 0;
    bool improved = true;

    while (improved && checks < max_checks) {
        improved = false;
        int best_delta = 0;
        int best_i = -1;
        int best_j = -1;

        for (int i = 1; i < n - 1 && checks < max_checks; ++i) {
            int a = tour[i - 1];
            int b = tour[i];

            for (int j = i + 1; j < n && checks < max_checks; ++j) {
                ++checks;

                int c = tour[j];
                int d = tour[(j + 1) % n];

                int delta = cpu_dist(inst, a, c) + cpu_dist(inst, b, d) - cpu_dist(inst, a, b) - cpu_dist(inst, c, d);

                if (delta < best_delta) {
                    best_delta = delta;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if (best_delta < 0) {
            cpu_reverse(tour, best_i, best_j);
            improved = true;
        }
    }
}

static int cpu_nearest_seed(const Instance& inst, std::vector<int>& best_tour) {
    int n = inst.n;
    int starts = std::min(n, 8);
    int best = INF;

    std::vector<int> tour(n);
    std::vector<char> used(n);

    for (int sidx = 0; sidx < starts; ++sidx) {
        int s = (int)((long long)sidx * n / starts);

        std::fill(used.begin(), used.end(), 0);

        tour[0] = s;
        used[s] = 1;

        for (int k = 1; k < n; ++k) {
            int last = tour[k - 1];
            int bv = -1;
            int bd = INF;

            for (int v = 0; v < n; ++v) {
                if (!used[v]) {
                    int d = cpu_dist(inst, last, v);

                    if (d < bd) {
                        bd = d;
                        bv = v;
                    }
                }
            }

            tour[k] = bv;
            used[bv] = 1;
        }

        int cost = cpu_tour_cost(inst, tour);

        if (cost < best) {
            best = cost;
            best_tour = tour;
        }
    }

    cpu_two_opt_limited(inst, best_tour, std::min(1000000LL, (long long)n * n));
    best = cpu_tour_cost(inst, best_tour);

    return best;
}

__device__ __forceinline__ uint32_t rng_u32(uint32_t& s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

__device__ __forceinline__ float rng_float(uint32_t& s) {
    return (rng_u32(s) & 0x00ffffff) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ int gpu_dist(const int* dist, int n, int a, int b) {
    return __ldg(dist + (size_t)a * n + b);
}

__device__ void reverse_parallel(int* tour, int l, int r) {
    int len = r - l + 1;
    int half = len >> 1;

    for (int k = threadIdx.x; k < half; k += blockDim.x) {
        int i = l + k;
        int j = r - k;
        int t = tour[i];
        tour[i] = tour[j];
        tour[j] = t;
    }

    __syncthreads();
}

__device__ int block_sum_int(int local, int* scratch) {
    int tid = threadIdx.x;
    scratch[tid] = local;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }

    return scratch[0];
}

__device__ int tour_cost_block(const int* dist, int n, const int* tour, int* scratch) {
    int local = 0;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        local += gpu_dist(dist, n, tour[i], tour[(i + 1) % n]);
    }

    return block_sum_int(local, scratch);
}

__device__ void copy_tour_block(int* dst, const int* src, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        dst[i] = src[i];
    }

    __syncthreads();
}

__device__ void initialize_chain_tour(const int* seed_tour, int* tour, int n, int chain, uint32_t& rng) {
    int shift = (int)((uint64_t)(chain * 1315423911u) % (uint32_t)n);

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        tour[i] = seed_tour[(i + shift) % n];
    }

    __syncthreads();

    for (int p = 0; p < 4; ++p) {
        __shared__ int sl;
        __shared__ int sr;

        if (threadIdx.x == 0) {
            int a = 1 + (rng_u32(rng) % (n - 1));
            int b = 1 + (rng_u32(rng) % (n - 1));

            if (a > b) {
                int t = a;
                a = b;
                b = t;
            }

            if (a == b) {
                if (b + 1 < n) ++b;
                else if (a > 1) --a;
            }

            sl = a;
            sr = b;
        }

        __syncthreads();
        reverse_parallel(tour, sl, sr);
    }
}

__device__ void sampled_two_opt_round(
    const int* dist,
    int n,
    int* tour,
    int& cost,
    uint32_t& rng,
    int samples_per_thread,
    int* sdelta,
    int* si,
    int* sj
) {
    int tid = threadIdx.x;
    int local_delta = 0;
    int local_i = -1;
    int local_j = -1;

    uint32_t local_rng = rng ^ (uint32_t)(tid * 747796405u + 2891336453u);

    for (int s = 0; s < samples_per_thread; ++s) {
        int i = 1 + (rng_u32(local_rng) % (n - 1));
        int j = 1 + (rng_u32(local_rng) % (n - 1));

        if (i > j) {
            int t = i;
            i = j;
            j = t;
        }

        if (i == j) continue;

        int a = tour[i - 1];
        int b = tour[i];
        int c = tour[j];
        int d = tour[(j + 1) % n];

        int delta = gpu_dist(dist, n, a, c) + gpu_dist(dist, n, b, d) - gpu_dist(dist, n, a, b) - gpu_dist(dist, n, c, d);

        if (delta < local_delta) {
            local_delta = delta;
            local_i = i;
            local_j = j;
        }
    }

    sdelta[tid] = local_delta;
    si[tid] = local_i;
    sj[tid] = local_j;

    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sdelta[tid + stride] < sdelta[tid]) {
                sdelta[tid] = sdelta[tid + stride];
                si[tid] = si[tid + stride];
                sj[tid] = sj[tid + stride];
            }
        }

        __syncthreads();
    }

    if (sdelta[0] < 0) {
        int l = si[0];
        int r = sj[0];
        int delta = sdelta[0];

        reverse_parallel(tour, l, r);

        if (tid == 0) {
            cost += delta;
        }

        __syncthreads();
    }
}

__global__ void sa_block_kernel(
    const int* __restrict__ dist,
    const int* __restrict__ seed_tour,
    int n,
    int chains,
    int iterations,
    int local_samples,
    int local_rounds,
    uint64_t seed,
    int* __restrict__ chain_costs,
    int* __restrict__ chain_tours,
    int* __restrict__ work_tours
) {
    int chain = blockIdx.x;

    if (chain >= chains) return;

    extern __shared__ int shared[];

    int* scratch = shared;
    int* sdelta = scratch + THREADS_PER_BLOCK;
    int* si = sdelta + THREADS_PER_BLOCK;
    int* sj = si + THREADS_PER_BLOCK;

    int* tour = work_tours + (size_t)chain * n;
    int* best_tour = chain_tours + (size_t)chain * n;

    __shared__ int cost;
    __shared__ int best_cost;

    uint32_t rng = (uint32_t)(seed ^ ((uint64_t)chain + 1ULL) * 0x9e3779b97f4a7c15ULL);
    rng ^= (uint32_t)(chain * 747796405u + 2891336453u);

    initialize_chain_tour(seed_tour, tour, n, chain, rng);

    int initial_cost = tour_cost_block(dist, n, tour, scratch);

    if (threadIdx.x == 0) {
        cost = initial_cost;
        best_cost = initial_cost;
    }

    __syncthreads();

    copy_tour_block(best_tour, tour, n);

    for (int r = 0; r < local_rounds; ++r) {
        sampled_two_opt_round(dist, n, tour, cost, rng, local_samples, sdelta, si, sj);

        if (threadIdx.x == 0 && cost < best_cost) {
            best_cost = cost;
            atomicMin(&d_global_best, best_cost);
        }

        __syncthreads();

        if (cost == best_cost) {
            copy_tour_block(best_tour, tour, n);
        }
    }

    float temp = fmaxf(1.0f, 0.25f * ((float)cost / (float)n));
    float final_temp = 0.0001f;
    float alpha = expf(logf(final_temp / temp) / fmaxf(1.0f, (float)iterations));

    for (int it = 0; it < iterations; ++it) {
        __shared__ int sl;
        __shared__ int sr;
        __shared__ int saccept;
        __shared__ int sdiff;

        if (threadIdx.x == 0) {
            int i = 1 + (rng_u32(rng) % (n - 1));
            int j = 1 + (rng_u32(rng) % (n - 1));

            if (i > j) {
                int t = i;
                i = j;
                j = t;
            }

            if (i == j) {
                if (j + 1 < n) ++j;
                else if (i > 1) --i;
            }

            int a = tour[i - 1];
            int b = tour[i];
            int c = tour[j];
            int d = tour[(j + 1) % n];

            int delta = gpu_dist(dist, n, a, c) + gpu_dist(dist, n, b, d) - gpu_dist(dist, n, a, b) - gpu_dist(dist, n, c, d);

            float r = rng_float(rng);
            int accept = 0;

            if (delta <= 0 || r < expf(-(float)delta / temp)) {
                accept = 1;
            }

            sl = i;
            sr = j;
            sdiff = delta;
            saccept = accept;

            temp *= alpha;
        }

        __syncthreads();

        if (saccept) {
            reverse_parallel(tour, sl, sr);

            if (threadIdx.x == 0) {
                cost += sdiff;

                if (cost < best_cost) {
                    best_cost = cost;
                    atomicMin(&d_global_best, best_cost);
                }
            }

            __syncthreads();

            if (cost == best_cost) {
                copy_tour_block(best_tour, tour, n);
            }
        }

        if ((it & 255) == 255) {
            sampled_two_opt_round(dist, n, tour, cost, rng, local_samples, sdelta, si, sj);

            if (threadIdx.x == 0 && cost < best_cost) {
                best_cost = cost;
                atomicMin(&d_global_best, best_cost);
            }

            __syncthreads();

            if (cost == best_cost) {
                copy_tour_block(best_tour, tour, n);
            }
        }

        if ((it & 4095) == 4095) {
            if (threadIdx.x == 0 && cost > best_cost + n) {
                cost = best_cost;
            }

            __syncthreads();

            if (cost == best_cost) {
                copy_tour_block(tour, best_tour, n);
            }

            __shared__ int rl;
            __shared__ int rr;

            if (threadIdx.x == 0) {
                int a = 1 + (rng_u32(rng) % (n - 1));
                int b = 1 + (rng_u32(rng) % (n - 1));

                if (a > b) {
                    int t = a;
                    a = b;
                    b = t;
                }

                if (a == b) {
                    if (b + 1 < n) ++b;
                    else if (a > 1) --a;
                }

                rl = a;
                rr = b;
            }

            __syncthreads();
            reverse_parallel(tour, rl, rr);

            int new_cost = tour_cost_block(dist, n, tour, scratch);

            if (threadIdx.x == 0) {
                cost = new_cost;
            }

            __syncthreads();
        }
    }

    for (int r = 0; r < local_rounds * 2; ++r) {
        sampled_two_opt_round(dist, n, tour, cost, rng, local_samples, sdelta, si, sj);

        if (threadIdx.x == 0 && cost < best_cost) {
            best_cost = cost;
            atomicMin(&d_global_best, best_cost);
        }

        __syncthreads();

        if (cost == best_cost) {
            copy_tour_block(best_tour, tour, n);
        }
    }

    if (threadIdx.x == 0) {
        chain_costs[chain] = best_cost;
        atomicMin(&d_global_best, best_cost);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " instance.tsp [iterations] [chains_per_sm] [local_samples] [local_rounds] [seed]\n";
        return 1;
    }

    Instance inst = read_tsplib_euc2d(argv[1]);
    int n = inst.n;

    int iterations = argc >= 3 ? std::max(1, std::atoi(argv[2])) : DEFAULT_ITERATIONS;
    int chains_per_sm = argc >= 4 ? std::max(1, std::atoi(argv[3])) : DEFAULT_CHAINS_PER_SM;
    int local_samples = argc >= 5 ? std::max(1, std::atoi(argv[4])) : DEFAULT_LOCAL_SAMPLES;
    int local_rounds = argc >= 6 ? std::max(1, std::atoi(argv[5])) : DEFAULT_LOCAL_ROUNDS;
    uint64_t seed = argc >= 7 ? std::strtoull(argv[6], nullptr, 10) : (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();

    int device = 0;
    cudaDeviceProp prop{};
    cuda_check(cudaGetDeviceProperties(&prop, device));
    cuda_check(cudaSetDevice(device));

    int sms = prop.multiProcessorCount;
    int chains = std::max(sms * chains_per_sm, sms);
    int blocks = chains;

    std::vector<int> seed_tour;
    int host_seed_cost = cpu_nearest_seed(inst, seed_tour);

    int* d_dist = nullptr;
    int* d_seed_tour = nullptr;
    int* d_chain_costs = nullptr;
    int* d_chain_tours = nullptr;
    int* d_work_tours = nullptr;

    size_t dist_bytes = (size_t)n * n * sizeof(int);
    size_t tours_bytes = (size_t)chains * n * sizeof(int);

    cuda_check(cudaMalloc(&d_dist, dist_bytes));
    cuda_check(cudaMalloc(&d_seed_tour, (size_t)n * sizeof(int)));
    cuda_check(cudaMalloc(&d_chain_costs, (size_t)chains * sizeof(int)));
    cuda_check(cudaMalloc(&d_chain_tours, tours_bytes));
    cuda_check(cudaMalloc(&d_work_tours, tours_bytes));

    cuda_check(cudaMemcpy(d_dist, inst.dist.data(), dist_bytes, cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(d_seed_tour, seed_tour.data(), (size_t)n * sizeof(int), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpyToSymbol(d_global_best, &host_seed_cost, sizeof(int)));

    size_t shared_bytes = (size_t)THREADS_PER_BLOCK * 4 * sizeof(int);

    auto t0 = std::chrono::high_resolution_clock::now();

    sa_block_kernel<<<blocks, THREADS_PER_BLOCK, shared_bytes>>>(
        d_dist,
        d_seed_tour,
        n,
        chains,
        iterations,
        local_samples,
        local_rounds,
        seed,
        d_chain_costs,
        d_chain_tours,
        d_work_tours
    );

    cuda_check(cudaGetLastError());
    cuda_check(cudaDeviceSynchronize());

    auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<int> chain_costs(chains);
    std::vector<int> best_tour(n);
    std::vector<int> all_tours;

    cuda_check(cudaMemcpy(chain_costs.data(), d_chain_costs, (size_t)chains * sizeof(int), cudaMemcpyDeviceToHost));

    int best_chain = -1;
    int best_cost = host_seed_cost;

    for (int c = 0; c < chains; ++c) {
        if (chain_costs[c] < best_cost) {
            best_cost = chain_costs[c];
            best_chain = c;
        }
    }

    if (best_chain >= 0) {
        cuda_check(cudaMemcpy(best_tour.data(), d_chain_tours + (size_t)best_chain * n, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost));
    } else {
        best_tour = seed_tour;
    }

    cpu_two_opt_limited(inst, best_tour, std::min(2000000LL, (long long)n * n));
    best_cost = cpu_tour_cost(inst, best_tour);

    double secs = std::chrono::duration<double>(t1 - t0).count();

    cudaFree(d_dist);
    cudaFree(d_seed_tour);
    cudaFree(d_chain_costs);
    cudaFree(d_chain_tours);
    cudaFree(d_work_tours);

    std::cout << "Instance " << argv[1] << "\n";
    std::cout << "Vertices " << n << "\n";
    std::cout << "SMs " << sms << "\n";
    std::cout << "Chains " << chains << "\n";
    std::cout << "Blocks " << blocks << "\n";
    std::cout << "ThreadsPerBlock " << THREADS_PER_BLOCK << "\n";
    std::cout << "Iterations " << iterations << "\n";
    std::cout << "LocalSamples " << local_samples << "\n";
    std::cout << "LocalRounds " << local_rounds << "\n";
    std::cout << "SeedCost " << host_seed_cost << "\n";
    std::cout << "BestCost " << best_cost << "\n";
    std::cout << std::fixed << std::setprecision(6) << "GpuTime " << secs << "\n";
    std::cout << "Tour";
    for (int v : best_tour) {
        std::cout << " " << (v + 1);
    }
    std::cout << "\n";
    std::cout << "CSV," << n << "," << sms << "," << chains << "," << blocks << "," << iterations << "," << local_samples << "," << local_rounds << "," << best_cost << "," << std::setprecision(6) << secs << "\n";

    return 0;
}