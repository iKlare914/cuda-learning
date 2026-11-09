#pragma once

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
#include <tuple>
#include <functional>
#include <ctime>

// Report the exact CUDA call and source line if an experiment fails.
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t error = (call);                                        \
        if (error != cudaSuccess) {                                             \
            std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call,   \
                         cudaGetErrorString(error));                            \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (false)

// constexpr function can be evaluated at compile time, but can also be used at runtime.
inline constexpr int cdiv(int a, int b){
    return (a + b - 1) / b;
}

#ifdef __CUDACC__
// Device counterpart of cdiv; a >= 0, b > 0, and a + b - 1 must fit in int.
__device__ __forceinline__ constexpr int cdiv_device(int a, int b){
    return (a + b - 1) / b;
}
#endif

void printDeviceProps();

std::tuple<float, float> kernelTimer(std::function<void()> kernel, int repeats, std::function<void()> = {});

