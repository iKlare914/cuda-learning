#include "utils.hpp"

void printDeviceProps(){
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    std::printf("GPU: %s (compute capability %d.%d)\n", deviceProp.name,
                deviceProp.major, deviceProp.minor);
    std::printf("device max thread per block: %d\n", deviceProp.maxThreadsPerBlock);
    std::printf("device max thread per SM: %d\n", deviceProp.maxThreadsPerMultiProcessor);
    std::printf("device SM shared memory per block (Bytes): %zu\n", deviceProp.sharedMemPerBlock);
    std::printf("device SM count: %d\n", deviceProp.multiProcessorCount);
    std::printf("Registers per SM: %d\n", deviceProp.regsPerMultiprocessor);
    std::printf("Registers per block limit: %d\n", deviceProp.regsPerBlock);
    std::printf("Threads per warp: %d\n", deviceProp.warpSize);
    assert(deviceProp.warpSize == 32);
}

std::tuple<float, float> kernelTimer(std::function<void()> kernel, int repeats, std::function<void()> init) {
    float gpu_ms = 0.0f;
    for(int i = 0; i < repeats; ++i){
        if (init) init();
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));

        kernel();

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float cur_gpu_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&cur_gpu_ms, start, stop));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        gpu_ms += cur_gpu_ms;
    }

    float gpu_avg_ms = gpu_ms / repeats;
    std::printf("GPU kernel: %.5f ms (average of %d runs, excludes transfers)\n",
                gpu_avg_ms, repeats);
    return std::make_tuple(gpu_ms, gpu_avg_ms);
}
