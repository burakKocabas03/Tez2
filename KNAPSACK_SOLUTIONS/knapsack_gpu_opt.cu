#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using i64 = long long;
using u64 = unsigned long long;

static constexpr int THREADS = 256;
static constexpr i64 NEG_INF = std::numeric_limits<i64>::min() / 4;

struct Instance {
    int n = 0;
    i64 capacity = 0;
    std::vector<i64> weight;
    std::vector<i64> value;
};

static void die(const std::string& s) {
    std::cerr << s << "\n";
    std::exit(1);
}

static void cuda_check(cudaError_t e) {
    if (e != cudaSuccess) die(cudaGetErrorString(e));
}

static Instance read_instance(const std::string& path) {
    std::ifstream in(path);
    if (!in) die("cannot open input file");

    Instance inst;
    in >> inst.n;

    if (!in || inst.n <= 0) die("invalid instance header");

    inst.weight.resize(inst.n);
    inst.value.resize(inst.n);

    for (int i = 0; i < inst.n; ++i) {
        long long id;
        long long w;
        long long v;

        in >> id >> w >> v;

        if (!in) die("invalid item line");
        if (w < 0) die("negative item weight");

        inst.weight[i] = w;
        inst.value[i] = v;
    }

    in >> inst.capacity;

    if (!in) die("missing capacity");
    if (inst.capacity < 0) die("invalid capacity");

    return inst;
}

struct MaxI64 {
    __host__ __device__ i64 operator()(const i64& a, const i64& b) const {
        return a > b ? a : b;
    }
};

__global__ void make_candidates_kernel(
    const i64* __restrict__ w,
    const i64* __restrict__ v,
    i64* __restrict__ cw,
    i64* __restrict__ cv,
    u64 m,
    i64 item_w,
    i64 item_v,
    i64 capacity,
    i64 invalid_w
) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= m) return;

    i64 base_w = w[i];
    i64 base_v = v[i];

    cw[i] = base_w;
    cv[i] = base_v;

    i64 tw = base_w + item_w;

    if (tw >= base_w && tw <= capacity) {
        cw[i + m] = tw;
        cv[i + m] = base_v + item_v;
    } else {
        cw[i + m] = invalid_w;
        cv[i + m] = NEG_INF;
    }
}

__global__ void mark_keep_kernel(
    const i64* __restrict__ w,
    const i64* __restrict__ v,
    const i64* __restrict__ prefix,
    unsigned int* __restrict__ keep,
    u64 n,
    i64 capacity
) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    i64 prev_best = i == 0 ? NEG_INF : prefix[i - 1];

    keep[i] = (w[i] <= capacity && v[i] > prev_best) ? 1u : 0u;
}

__global__ void scatter_keep_kernel(
    const i64* __restrict__ w,
    const i64* __restrict__ v,
    const unsigned int* __restrict__ keep,
    const u64* __restrict__ offsets,
    i64* __restrict__ nw,
    i64* __restrict__ nv,
    u64 n
) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    if (keep[i]) {
        u64 p = offsets[i];
        nw[p] = w[i];
        nv[p] = v[i];
    }
}

static int launch_blocks_u64(u64 n) {
    u64 b = (n + THREADS - 1) / THREADS;
    if (b > (u64)std::numeric_limits<int>::max()) die("frontier too large for one kernel launch");
    return (int)b;
}

static void solve_sparse_gpu(const Instance& inst) {
    thrust::device_vector<i64> W(1);
    thrust::device_vector<i64> V(1);

    W[0] = 0;
    V[0] = 0;

    i64 invalid_w = inst.capacity == std::numeric_limits<i64>::max() ? std::numeric_limits<i64>::max() : inst.capacity + 1;

    auto t0 = std::chrono::high_resolution_clock::now();

    u64 peak_frontier = 1;

    for (int item = 0; item < inst.n; ++item) {
        u64 m = (u64)W.size();

        thrust::device_vector<i64> CW((size_t)(2 * m));
        thrust::device_vector<i64> CV((size_t)(2 * m));

        make_candidates_kernel<<<launch_blocks_u64(m), THREADS>>>(
            thrust::raw_pointer_cast(W.data()),
            thrust::raw_pointer_cast(V.data()),
            thrust::raw_pointer_cast(CW.data()),
            thrust::raw_pointer_cast(CV.data()),
            m,
            inst.weight[item],
            inst.value[item],
            inst.capacity,
            invalid_w
        );

        cuda_check(cudaGetLastError());

        thrust::sort_by_key(CW.begin(), CW.end(), CV.begin());

        thrust::device_vector<i64> RW((size_t)(2 * m));
        thrust::device_vector<i64> RV((size_t)(2 * m));

        auto reduced_end = thrust::reduce_by_key(
            CW.begin(),
            CW.end(),
            CV.begin(),
            RW.begin(),
            RV.begin(),
            thrust::equal_to<i64>(),
            MaxI64()
        );

        u64 rsize = (u64)(reduced_end.first - RW.begin());

        RW.resize((size_t)rsize);
        RV.resize((size_t)rsize);

        if (rsize == 0) die("empty frontier");

        thrust::device_vector<i64> prefix((size_t)rsize);
        thrust::inclusive_scan(RV.begin(), RV.end(), prefix.begin(), MaxI64());

        thrust::device_vector<unsigned int> keep((size_t)rsize);

        mark_keep_kernel<<<launch_blocks_u64(rsize), THREADS>>>(
            thrust::raw_pointer_cast(RW.data()),
            thrust::raw_pointer_cast(RV.data()),
            thrust::raw_pointer_cast(prefix.data()),
            thrust::raw_pointer_cast(keep.data()),
            rsize,
            inst.capacity
        );

        cuda_check(cudaGetLastError());

        thrust::device_vector<u64> offsets((size_t)rsize);
        thrust::exclusive_scan(keep.begin(), keep.end(), offsets.begin(), (u64)0);

        unsigned int last_keep = 0;
        u64 last_offset = 0;

        cuda_check(cudaMemcpy(&last_keep, thrust::raw_pointer_cast(keep.data()) + rsize - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        cuda_check(cudaMemcpy(&last_offset, thrust::raw_pointer_cast(offsets.data()) + rsize - 1, sizeof(u64), cudaMemcpyDeviceToHost));

        u64 nsize = last_offset + (u64)last_keep;

        if (nsize == 0) die("all states pruned");

        thrust::device_vector<i64> NW((size_t)nsize);
        thrust::device_vector<i64> NV((size_t)nsize);

        scatter_keep_kernel<<<launch_blocks_u64(rsize), THREADS>>>(
            thrust::raw_pointer_cast(RW.data()),
            thrust::raw_pointer_cast(RV.data()),
            thrust::raw_pointer_cast(keep.data()),
            thrust::raw_pointer_cast(offsets.data()),
            thrust::raw_pointer_cast(NW.data()),
            thrust::raw_pointer_cast(NV.data()),
            rsize
        );

        cuda_check(cudaGetLastError());

        W.swap(NW);
        V.swap(NV);

        peak_frontier = std::max(peak_frontier, (u64)W.size());
    }

    cuda_check(cudaDeviceSynchronize());

    i64 best_value = V.back();
    i64 best_weight = W.back();

    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "Items " << inst.n << "\n";
    std::cout << "Capacity " << inst.capacity << "\n";
    std::cout << "BestValue " << best_value << "\n";
    std::cout << "BestWeight " << best_weight << "\n";
    std::cout << "FinalFrontier " << (u64)W.size() << "\n";
    std::cout << "PeakFrontier " << peak_frontier << "\n";
    std::cout << std::fixed << std::setprecision(6) << "Time " << sec << "\n";
    std::cout << "CSV," << inst.n << "," << inst.capacity << "," << best_value << "," << best_weight << "," << (u64)W.size() << "," << peak_frontier << "," << std::setprecision(6) << sec << "\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " input.txt\n";
        std::cerr << "input format: n capacity followed by n lines: weight value\n";
        return 1;
    }

    int device = 0;
    cudaDeviceProp prop{};
    cuda_check(cudaGetDeviceProperties(&prop, device));
    cuda_check(cudaSetDevice(device));

    try {
        Instance inst = read_instance(argv[1]);
        solve_sparse_gpu(inst);
    } catch (const std::bad_alloc&) {
        die("host allocation failed");
    } catch (const thrust::system_error& e) {
        die(std::string("thrust error: ") + e.what());
    } catch (const std::exception& e) {
        die(std::string("error: ") + e.what());
    }

    return 0;
}