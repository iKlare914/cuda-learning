#include <utils.hpp>
#include <cuda_runtime.h>
#include <numeric>

// simple 1D sum
template<int TILE, int COARSE, bool branch_eff = true>
__global__ void reduce_sum(float* a, float* res, size_t n){
    unsigned int tid = threadIdx.x; // local thread idx
    static_assert(TILE % COARSE == 0);
    constexpr unsigned int BX = TILE / COARSE;
    __shared__ float tile[BX];
    unsigned int idx = TILE * blockIdx.x; // global block idx
    // from global memory to shared memory
    float sum = 0.0f;
    #pragma unroll
    for(int i = 0; i < COARSE; ++i){
        const size_t g_idx = idx + tid * COARSE + i;
        sum += g_idx < n ? a[g_idx] : 0.0f; 
    }
    tile[tid] = sum;
    __syncthreads();
    // sum within one block (high divergence)
    if constexpr (!branch_eff) {
        #pragma unroll
        for(unsigned int stride = 1; stride < BX; stride <<= 1){
                int r_idx = idx + (tid + stride) * COARSE; // real index of a
                if(tid % (stride << 1) == 0 && r_idx < n) tile[tid] += tile[tid + stride]; // load/store thread are seperated
                __syncthreads();
        }
    }else{
        #pragma unroll
        for(unsigned int stride = (BX + 1) / 2; stride > 0; stride >>= 1){
                int r_idx = idx + (tid + stride) * COARSE;
                if(tid < stride && r_idx < n) tile[tid] += tile[tid + stride]; // load/store thread are seperated
                __syncthreads();
        }
    }
    // sum within all blocks
    if(tid == 0 && idx < n){
        atomicAdd(res, tile[0]);
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
    int tile = 1;
    constexpr int coarse_factor = 4;
    const bool branch_eff = argc > 2 ? (std::atoi(argvs[2]) == 1) : true;
    std::printf("Tile size: \n");
    std::cin >> tile;
    const int GRID = cdiv(n, tile);
    const int BLOCK = cdiv(tile, coarse_factor);
    auto launch_kernel = [&](){
        switch (tile) {
            case 64:
                if (branch_eff) reduce_sum<64, coarse_factor, true><<<GRID, BLOCK>>>(device_a, device_b, n);
                else reduce_sum<64, coarse_factor, false><<<GRID, BLOCK>>>(device_a, device_b, n);
                break;
            case 128:
                if (branch_eff) reduce_sum<128, coarse_factor, true><<<GRID, BLOCK>>>(device_a, device_b, n);
                else reduce_sum<128, coarse_factor, false><<<GRID, BLOCK>>>(device_a, device_b, n);
                break;
            case 256:
                if (branch_eff) reduce_sum<256, coarse_factor, true><<<GRID, BLOCK>>>(device_a, device_b, n);
                else reduce_sum<256, coarse_factor, false><<<GRID, BLOCK>>>(device_a, device_b, n);
                break;
            case 512:
                if (branch_eff) reduce_sum<512, coarse_factor, true><<<GRID, BLOCK>>>(device_a, device_b, n);
                else reduce_sum<512, coarse_factor, false><<<GRID, BLOCK>>>(device_a, device_b, n);
                break;
            case 1024:
                if (branch_eff) reduce_sum<1024, coarse_factor, true><<<GRID, BLOCK>>>(device_a, device_b, n);
                else reduce_sum<1024, coarse_factor, false><<<GRID, BLOCK>>>(device_a, device_b, n);
                break;
            case 2048:
                if (branch_eff) reduce_sum<2048, coarse_factor, true><<<GRID, BLOCK>>>(device_a, device_b, n);
                else reduce_sum<2048, coarse_factor, false><<<GRID, BLOCK>>>(device_a, device_b, n);
                break;
            case 4096:
                if (branch_eff) reduce_sum<4096, coarse_factor, true><<<GRID, BLOCK>>>(device_a, device_b, n);
                else reduce_sum<4096, coarse_factor, false><<<GRID, BLOCK>>>(device_a, device_b, n);
                break;
            default:
                std::printf("Tile size: %d has not been implemented", tile);
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
