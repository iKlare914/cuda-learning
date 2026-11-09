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

template<int TILE>
__global__ void tiled_matmul(const float* a, const float* b, float* c, const int m, const int n, const int k){
    __shared__ float tile_a[TILE][TILE];
    __shared__ float tile_b[TILE][TILE];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tr = threadIdx.y;
    int tc = threadIdx.x;
    
    float res = 0.0f;
    for(int t = 0; t < k; t += TILE){
        tile_a[tr][tc] = t + tc < k && row < m ? a[row * k + t + tc] : 0.0f;
        tile_b[tr][tc] = t + tr < k && col < n ? b[(t + tr) * n + col] : 0.0f;
        __syncthreads();
        for(int s = 0; s < TILE; ++s){
            res += tile_a[tr][s] * tile_b[s][tc];
        }
        __syncthreads();
    }
    if (row < m && col < n){
        c[row * n + col] = res;
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
                dim3 block(8, 8);
                dim3 grid(cdiv(size_n, 8), cdiv(size_m, 8)); // CUDA index is in reverse order
                tiled_matmul<8><<<grid, block, 0>>>(device_a, device_b, device_c, size_m, size_n, size_k); // params in C++ index order
                break;
            }
            case 16: {
                dim3 block(16, 16);
                dim3 grid(cdiv(size_n, 16), cdiv(size_m, 16));
                tiled_matmul<16><<<grid, block, 0>>>(device_a, device_b, device_c, size_m, size_n, size_k);
                break;
            }
            case 32: {
                dim3 block(32, 32);
                dim3 grid(cdiv(size_n, 32), cdiv(size_m, 32));
                tiled_matmul<32><<<grid, block, 0>>>(device_a, device_b, device_c, size_m, size_n, size_k);
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
