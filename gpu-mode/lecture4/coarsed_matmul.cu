#include <utils.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <stdexcept>
#include <chrono>
#include <string>
#include <cassert>

// Thread coarsing + register tiling 
template<int BM, int BK, int BN, int COARSE_A = 1, int COARSE_B = 1>
__global__ void tiled_matmul_coarsed(const float* a, const float* b, float* c, const int m, const int n, const int k){
    static_assert(BM > 0 && BK > 0 && BN > 0);
    static_assert(COARSE_A > 0 && COARSE_B > 0);
    static_assert(BM % COARSE_A == 0 && BN % COARSE_B == 0);
    constexpr int BX = BN / COARSE_B;
    constexpr int BY = BM / COARSE_A;
    static_assert(BX * BY <= 1024);
    __shared__ float tile_a[BM][BK];
    __shared__ float tile_b[BK][BN];
    float res[COARSE_A][COARSE_B] = {};
    const int tr_base = threadIdx.y * COARSE_A;
    const int tc_base = threadIdx.x * COARSE_B;
    // Each block computes a BM x BN output tile in steps of BK.
    const int row_base = blockIdx.y * BM + tr_base;
    const int col_base = blockIdx.x * BN + tc_base;

    for(int t = 0; t < k; t += BK){
        // Keep two-dimensional loading: x threads cover A's K dimension.
        #pragma unroll // without unrolling , tile = 64 will increase duration slightly
        for(int s = threadIdx.x; s < BK; s += BX){
            for(int i = 0; i < COARSE_A; ++i){
                const int tr = tr_base + i;
                const int row = row_base + i;
                tile_a[tr][s] = row < m && t + s < k ? a[row * k + t + s] : 0.0f;
            }
        }
        // y threads cover B's K dimension independently of BM.
        #pragma unroll // without unrolling , tile = 64 will increase duration slightly
        for(int s = threadIdx.y; s < BK; s += BY){
            for(int j = 0; j < COARSE_B; ++j){
                const int tc = tc_base + j;
                const int col = col_base + j;
                tile_b[s][tc] = t + s < k && col < n ? b[(t + s) * n + col] : 0.0f;
            }
        }
        __syncthreads();
        for(int s = 0; s < BK; ++s){
            float reg_a[COARSE_A];
            float reg_b[COARSE_B]; // register tiling
            for(int i = 0; i < COARSE_A; ++i){
                reg_a[i] = tile_a[tr_base + i][s]; // column s in tile_a
            }
            for(int j = 0; j < COARSE_B; ++j){
                reg_b[j] = tile_b[s][tc_base + j]; // row s in tile_b
            }
            for(int i = 0; i < COARSE_A; ++i){
                for(int j = 0; j < COARSE_B; ++j){
                    res[i][j] += reg_a[i] * reg_b[j]; // outer product !!!
                }
            }
        }
        __syncthreads();
    }
    for(int i = 0; i < COARSE_A; ++i){
        for(int j = 0; j < COARSE_B; ++j){
            const int row = row_base + i;
            const int col = col_base + j;
            if(row < m && col < n) c[row * n + col] = res[i][j];
        }
    }
}
// Row-major matrices: A[m, k] * B[k, n] = res[m, n].
void matmul(const std::vector<float>& a, const std::vector<float>& b,
            std::vector<float>& res, int m, int n, int k) {
    if (m < 0 || n < 0 || k < 0) {
        throw std::invalid_argument("Matrix dimensions must be nonnegative");
    }
    const auto rows = static_cast<std::size_t>(m);
    const auto cols = static_cast<std::size_t>(n);
    const auto inner = static_cast<std::size_t>(k);
    if (a.size() != rows * inner || b.size() != inner * cols) {
        throw std::invalid_argument("Matrix dimensions do not match vector sizes");
    }

    // A separate result also allows res to refer to a or b.
    std::vector<float> result(rows * cols, 0.0f);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            float sum = 0.0f;
            for (std::size_t s = 0; s < inner; ++s) {
                sum = std::fma(a[row * inner + s] , b[s * cols + col], sum);
            }
            result[row * cols + col] = sum;
        }
    }
    res.swap(result);
}

int main(int argc, char* argvs[]){
    printDeviceProps();
    constexpr size_t size_m = 2048;
    constexpr size_t size_k = 1024;
    constexpr size_t size_n = 1024;
    constexpr size_t bytes_a = size_m * size_k * sizeof(float);
    constexpr size_t bytes_b = size_n * size_k * sizeof(float);
    constexpr size_t bytes_c = size_m * size_n * sizeof(float);
    std::printf("Matmul test %zux%zu @ %zux%zu\n", size_m, size_k, size_k, size_n);
    int tile_size = 32;
    std::printf("Tile size: \n");
    std::cin >> tile_size;

    std::vector<float> a, b, res(size_m*size_n);
    float *device_a, *device_b, *device_c;
    for(int i = 0; i < size_m; ++i){
        for(int j = 0; j < size_k; ++j){
            a.push_back(static_cast<float>(j) / 0.5f);
        }
    }
    for(int i = 0; i < size_k; ++i){
        for(int j = 0; j < size_n; ++j){
            b.push_back(static_cast<float>(j) / 2.5f);
        }
    }
    CUDA_CHECK(cudaMalloc(&device_a, bytes_a));
    CUDA_CHECK(cudaMalloc(&device_b, bytes_b));
    CUDA_CHECK(cudaMalloc(&device_c, bytes_c));

    CUDA_CHECK(cudaMemcpy(device_a, a.data(), bytes_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_b, b.data(), bytes_b, cudaMemcpyHostToDevice));



    const auto launch_kernel = [&]() {
        switch (tile_size){
            case 8: {
                constexpr int BM = 8, BK = 8, BN = 8;
                constexpr int CA = 2, CB = 2;
                dim3 block(BN / CB, BM / CA);
                dim3 grid(cdiv(size_n, BN), cdiv(size_m, BM));
                tiled_matmul_coarsed<BM, BK, BN, CA, CB><<<grid, block>>>(
                    device_a, device_b, device_c, size_m, size_n, size_k);
                break;
            }
            case 16: {
                constexpr int BM = 16, BK = 16, BN = 16;
                constexpr int CA = 2, CB = 2;
                dim3 block(BN / CB, BM / CA);
                dim3 grid(cdiv(size_n, BN), cdiv(size_m, BM));
                tiled_matmul_coarsed<BM, BK, BN, CA, CB><<<grid, block>>>(
                    device_a, device_b, device_c, size_m, size_n, size_k);
                break;
            }
            case 32: {
                constexpr int BM = 32, BK = 32, BN = 32;
                constexpr int CA = 2, CB = 2;
                dim3 block(BN / CB, BM / CA);
                dim3 grid(cdiv(size_n, BN), cdiv(size_m, BM));
                tiled_matmul_coarsed<BM, BK, BN, CA, CB><<<grid, block>>>(
                    device_a, device_b, device_c, size_m, size_n, size_k);
                break;
            }
            case 64: {
                constexpr int BM = 64, BK = 64, BN = 64;
                constexpr int CA = 2, CB = 2;
                dim3 block(BN / CB, BM / CA);
                dim3 grid(cdiv(size_n, BN), cdiv(size_m, BM));
                tiled_matmul_coarsed<BM, BK, BN, CA, CB><<<grid, block>>>(
                    device_a, device_b, device_c, size_m, size_n, size_k);
                break;
            }
            default: {
                std::fprintf(stderr, "tile_size=%d is not implemented", tile_size);
                std::exit(EXIT_FAILURE);
                break;
            }
        }
        CUDA_CHECK(cudaGetLastError());
    };

    const int repeats = argc > 1 ? std::atoi(argvs[1]) : 10;
    const auto [gpu_total_ms, gpu_avg_ms] = kernelTimer(launch_kernel, repeats);

    CUDA_CHECK(cudaMemcpy(res.data(), device_c, bytes_c, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_a));
    CUDA_CHECK(cudaFree(device_b));
    CUDA_CHECK(cudaFree(device_c));

    std::vector<float> cpu_res;
    if(argc > 2 && !strcmp(argvs[2], "cpu")){
        const auto cpu_start = std::chrono::steady_clock::now();
        matmul(a, b, cpu_res, size_m, size_n, size_k);
        const auto cpu_stop = std::chrono::steady_clock::now();
        const double cpu_ms =
            std::chrono::duration<double, std::milli>(cpu_stop - cpu_start).count();
        std::printf("CPU matmul: %.3f ms (includes result allocation)\n", cpu_ms);
        if (gpu_avg_ms > 0.0) {
            std::printf("CPU time / GPU kernel time: %.2fx\n", cpu_ms / gpu_avg_ms);
        }
        constexpr double eps = 1e-8;
        for (std::size_t i = 0; i < res.size(); ++i) {
            const double diff = std::fabs(static_cast<double>(res[i]) - cpu_res[i]);
            if (!std::isfinite(res[i]) || !std::isfinite(cpu_res[i]) || diff > eps) {
                std::fprintf(stderr,
                            "FAIL at [%zu, %zu]: GPU=%.9g, CPU=%.9g, "
                            "abs_error=%.17g (eps=%.1e)\n",
                            i / size_m, i % size_n, res[i], cpu_res[i], diff, eps);
                return EXIT_FAILURE;
            }
        }
        std::printf("PASS: all %zu elements match CPU results (eps=%.1e)\n",
                    res.size(), eps);
    }
    return EXIT_SUCCESS;
}
