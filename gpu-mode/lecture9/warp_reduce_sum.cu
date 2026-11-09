#include <utils.hpp>
#include <cuda_runtime.h>
#include <numeric>

#ifndef SM_SIZE_THRES
#define SM_SIZE_THRES 16
#endif

// simple 1D sum
template<int sm_size, bool branch_eff = true>
__global__ void reduce_sum(float* a, float* res, size_t n){
    static_assert(sm_size > 0, "sm_size must be the number of warps in the block");
    static_assert(sm_size <= 32, "a block can contain at most 32 warps");
    __shared__ float memo[sm_size]; 
    unsigned int tid = threadIdx.x; // local thread idx
    unsigned int idx = blockDim.x * blockIdx.x; // global block idx
    unsigned int lane = tid % 32;
    unsigned int warp = tid / 32; // warp idx in block;
    constexpr unsigned int mask_ = 0xffffffff;
    unsigned int mask = mask_;
    float sum = tid + idx < n ? a[tid + idx] : 0;
    if constexpr (branch_eff){
        // still slightly slower than tiled coarsed impl
        unsigned int shift = 16;
        #pragma unroll
        for(; shift > 0; mask >>= shift, shift >>= 1){
            float val = 0.0f;
            if (lane < (shift * 2)) 
                val = __shfl_down_sync(mask, sum, shift, shift * 2);
            sum += val;
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
    // Use a warp for larger partial-sum sets; keep small sets serial.
    if constexpr (sm_size >= SM_SIZE_THRES) {
        if (warp == 0) {
            float block_sum = lane < sm_size ? memo[lane] : 0.0f;
            mask = mask_;
            #pragma unroll
            for (unsigned int shift = 16; shift > 0; mask >>= shift, shift >>= 1) {
                float val = 0.0f;
                if (lane < shift * 2)
                    val = __shfl_down_sync(mask, block_sum, shift, shift * 2);
                block_sum += val;
            }
            if (lane == 0 && idx < n)
                atomicAdd(res, block_sum);
        }
    } else {
        if (tid == 0 && idx < n) {
            sum = 0.0f;
            for (int i = 0; i < sm_size; ++i) sum += memo[i];
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
    const bool branch_eff = argc > 2 ? (std::atoi(argvs[2]) == 1) : true;
    std::printf("block size: \n");
    std::cin >> block;
    const int GRID = cdiv(n, block);
    const int BLOCK = block;
    assert(BLOCK % 32 == 0);
    auto launch_kernel = [&](){
        switch (block) {
            case 64:
                if (branch_eff) reduce_sum<2, true><<<GRID, BLOCK>>>(device_a, device_b, n);
                else reduce_sum<2, false><<<GRID, BLOCK>>>(device_a, device_b, n);
                break;
            case 128:
                if (branch_eff) reduce_sum<4, true><<<GRID, BLOCK>>>(device_a, device_b, n);
                else reduce_sum<4, false><<<GRID, BLOCK>>>(device_a, device_b, n);
                break;
            case 256:
                if (branch_eff) reduce_sum<8, true><<<GRID, BLOCK>>>(device_a, device_b, n);
                else reduce_sum<8, false><<<GRID, BLOCK>>>(device_a, device_b, n);
                break;
            case 512:
                if (branch_eff) reduce_sum<16, true><<<GRID, BLOCK>>>(device_a, device_b, n);
                else reduce_sum<16, false><<<GRID, BLOCK>>>(device_a, device_b, n);
                break;
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
