#include <utils.hpp>
#include <cuda_runtime.h>
#include <numeric>

// simple 1D sum
template<bool branch_eff = true>
__global__ void reduce_sum(float* a, float* res, size_t n){
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x;
    if constexpr (!branch_eff){
        // sum within one block (highly divergence) must slighly faster? since there are more eligible warps, may be the cause of continuous memory access within thread that make data preparation faster
        // maybe this method distibute active thread to warps evenly, so the scheduler can do better
        #pragma unroll
        for(unsigned int stride = 1; stride < blockDim.x; stride <<= 1){
            int r_idx = idx + tid + stride;
            if(tid % (stride << 1) == 0 && r_idx < n) a[idx + tid] += a[idx + tid + stride]; // load/store thread are seperated
            __syncthreads();
        }
    }else{
        #pragma unroll
        for(unsigned int stride = (blockDim.x + 1) / 2; stride > 0; stride >>= 1){
            int r_idx = idx + tid + stride;
            if(tid < stride && r_idx < n) a[idx + tid] += a[idx + tid + stride]; // load/store thread are seperated
            __syncthreads();
        }
    }
    // sum within all blocks
    if(tid == 0 && idx < n){
        atomicAdd(res, a[idx]);
    }
}

int main(int argc, char** argvs){
    printDeviceProps();
    float *device_a, *device_b;
    constexpr size_t bytes_a = size_t{256} * 1024 * 1024; // 256 MiB
    constexpr size_t n = bytes_a / sizeof(float);
    int repeats = argc > 1 ? std::atoi(argvs[1]) : 10;
    const bool branch_eff = argc > 2 ? (std::atoi(argvs[2]) == 1) : true;

    CUDA_CHECK(cudaMalloc(&device_a, bytes_a));
    CUDA_CHECK(cudaMalloc(&device_b, 1 * sizeof(float)));

    std::vector<float> a(n, 1.0f);
    float b = 0.0f, res = 0.0f;
    constexpr int BLOCK = 512;
    const int GRID = cdiv(n, BLOCK);
    auto launch_kernel = [&](){
        if (branch_eff){
            reduce_sum<true><<<GRID, BLOCK>>>(device_a, device_b, n);
        }else{
            reduce_sum<false><<<GRID, BLOCK>>>(device_a, device_b, n);
        }
    };

    auto init_data = [&](){
        CUDA_CHECK(cudaMemcpy(device_a, a.data(), bytes_a, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_b, &b, 1 * sizeof(float), cudaMemcpyHostToDevice));
    };

    auto [gpu_ms, gpu_avg_ms] = kernelTimer(launch_kernel, repeats, init_data);

    CUDA_CHECK(cudaMemcpy(&res, device_b, 1 * sizeof(float), cudaMemcpyDeviceToHost));
    double expected = std::accumulate(a.begin(), a.end(), 0.0);
    if(expected != res){
        std::printf("FAIL: expected %.1f, got %.1f\n", expected, res);
    } else {
        std::printf("PASS: expected %.1f, got %.1f\n", expected, res);
    }
    return 0;
}