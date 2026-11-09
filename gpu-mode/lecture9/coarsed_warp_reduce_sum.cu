/*
performance close to tiled coarsed reduce, but need less shared memory, whereas requires more registers
*/
#include <utils.hpp>
#include <cuda_runtime.h>
#include <numeric>

#ifndef SM_SIZE_THRES
#define SM_SIZE_THRES 16
#endif

// simple 1D sum
template<int SM_SIZE, int COARSE, bool branch_eff = true>
__global__ void reduce_sum(float* a, float* res, size_t n){
    __shared__ float memo[SM_SIZE]; 
    unsigned int tid = threadIdx.x; // thread idx
    unsigned int idx = blockDim.x * blockIdx.x; // real global block idx
    unsigned int lane = tid % 32;
    unsigned int warp = tid / 32; // warp idx in block;
    size_t base = size_t(blockDim.x) * blockIdx.x * COARSE;
    size_t offset = size_t(threadIdx.x) * COARSE;
    constexpr unsigned int mask = 0xffffffff;
    float sum = 0.0f;
    #pragma unroll
    for(int i = 0; i < COARSE; ++i){
        sum += base + offset + i < n ? a[base + offset + i] : 0.0f;
    }
    if constexpr (branch_eff){
        // still slightly slower than tiled coarsed impl
        unsigned int shift = 16;
        #pragma unroll
        for(; shift > 0; shift >>= 1){
            sum += __shfl_down_sync(mask, sum, shift);
        }
    }else{
        // highly divergent (but same perf)
        #pragma unroll
        for(unsigned int stride = 1; stride < 32; stride <<= 1){
            float val = __shfl_down_sync(mask, sum, stride, 32);
            if (lane % (stride << 1) == 0) sum += val;
        }
    }
    if (lane == 0) memo[warp] = sum;
    __syncthreads();
    if constexpr (SM_SIZE >= SM_SIZE_THRES) {
        // Since a block has 1024 thread at maximum, SM_SIZE <= 32, can be loaded to a warp
        static_assert(SM_SIZE <= 32);
        if (warp == 0){
            float f_sum = lane < SM_SIZE ? memo[lane] : 0.0f;
            unsigned int shift = 16;
            #pragma unroll
            for(; shift > 0; shift >>= 1){
                f_sum += __shfl_down_sync(mask, f_sum, shift);
            }

            // sum within all blocks
            if (lane == 0 && base < n){
                atomicAdd(res, f_sum);
            }
        }
    }else{
        if (tid == 0 && base < n){
            sum = 0.0f;
            for(int i = 0; i < SM_SIZE; ++i) sum += memo[i];
            atomicAdd(res, sum);
        }
    }
}

int main(int argc, char** argvs){
    printDeviceProps();
    float *device_a, *device_b;
    constexpr size_t bytes_a = size_t{256} * 1024 * 1024; // 256 MiB
    constexpr size_t n = bytes_a / sizeof(float);
    int repeats = argc > 1 ? std::atoi(argvs[1]) : 10;

    CUDA_CHECK(cudaMalloc(&device_a, bytes_a));
    CUDA_CHECK(cudaMalloc(&device_b, 1 * sizeof(float)));

    std::vector<float> a(n, 1.0f);
    float b = 0.0f, res = 0.0f;
    int block = 1;
    const int coarse_factor = 4;
    const bool branch_eff = argc > 2 ? (std::atoi(argvs[2]) == 1) : true;
    std::printf("block size: \n");
    std::cin >> block;
    const int BLOCK = cdiv(block, coarse_factor);
    const int GRID = cdiv(n, block);
    assert(BLOCK % 32 == 0);
    auto launch_kernel = [&](){
        switch (block) {
            case 128: {
                constexpr int sm_size = (128 / coarse_factor) / 32; // 1
                if (branch_eff) reduce_sum<sm_size, coarse_factor, true><<<GRID, BLOCK>>>(device_a, device_b, n);
                else reduce_sum<sm_size, coarse_factor, false><<<GRID, BLOCK>>>(device_a, device_b, n);
                break;
            }
            case 256: {
                constexpr int sm_size = (256 / coarse_factor) / 32; // 2
                if (branch_eff) reduce_sum<sm_size, coarse_factor, true><<<GRID, BLOCK>>>(device_a, device_b, n);
                else reduce_sum<sm_size, coarse_factor, false><<<GRID, BLOCK>>>(device_a, device_b, n);
                break;
            }
            case 512: {
                constexpr int sm_size = (512 / coarse_factor) / 32; // 4
                if (branch_eff) reduce_sum<sm_size, coarse_factor, true><<<GRID, BLOCK>>>(device_a, device_b, n);
                else reduce_sum<sm_size, coarse_factor, false><<<GRID, BLOCK>>>(device_a, device_b, n);
                break;
            }
            case 1024: {
                constexpr int sm_size = (1024 / coarse_factor) / 32; // 8
                if (branch_eff) reduce_sum<sm_size, coarse_factor, true><<<GRID, BLOCK>>>(device_a, device_b, n);
                else reduce_sum<sm_size, coarse_factor, false><<<GRID, BLOCK>>>(device_a, device_b, n);
                break;
            }
            case 2048: {
                constexpr int sm_size = (2048 / coarse_factor) / 32; // 16
                if (branch_eff) reduce_sum<sm_size, coarse_factor, true><<<GRID, BLOCK>>>(device_a, device_b, n);
                else reduce_sum<sm_size, coarse_factor, false><<<GRID, BLOCK>>>(device_a, device_b, n);
                break;
            }
            case 4096: {
                constexpr int sm_size = (4096 / coarse_factor) / 32; // 32
                if (branch_eff) reduce_sum<sm_size, coarse_factor, true><<<GRID, BLOCK>>>(device_a, device_b, n);
                else reduce_sum<sm_size, coarse_factor, false><<<GRID, BLOCK>>>(device_a, device_b, n);
                break;
            }
            default:
                std::printf("Block size: %d has not been implemented", block);
                break;
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
