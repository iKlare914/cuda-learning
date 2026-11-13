cmake_minimum_required(VERSION 3.24)
project(cuda_learning LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ARGN contains all source files supplied after the target name.
function(add_learning_executable target)
    add_executable(${target} ${ARGN} "${PROJECT_SOURCE_DIR}/include/utils.cpp") 
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:-lineinfo>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:-Xptxas=-v,-warn-spills> # warning of register spill
    )
    target_include_directories(${target} PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(${target} PRIVATE CUDA::cudart_static) # link static runtime libaray so that cpp files can use cuda runtime api functions
endfunction()

option(ENABLE_CUDA "Build CUDA kernel experiments" ON)
if(ENABLE_CUDA)
    # Detect the local GPU unless an architecture was explicitly supplied.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)

    add_learning_executable(matmul gpu-mode/lecture4/matmul.cu)
    add_learning_executable(naive_coarsed_matmul gpu-mode/lecture4/naive_coarsed_matmul.cu)
    add_learning_executable(coarsed_matmul gpu-mode/lecture4/coarsed_matmul.cu)
    add_learning_executable(contiguous_coarsed_matmul gpu-mode/lecture4/contiguous_coarsed_matmul.cu)
    add_learning_executable(vectorized_matmul gpu-mode/lecture4/vectorized_matmul.cu)
    add_learning_executable(coalescing_coarsed_matmul gpu-mode/lecture4/coalescing_coarsed_matmul.cu)
    add_learning_executable(naive_reduce_sum gpu-mode/lecture9/naive_reduce_sum.cu)
    add_learning_executable(tiled_coarsed_reduce_sum gpu-mode/lecture9/tiled_coarsed_reduce_sum.cu)
    add_learning_executable(warp_reduce_sum gpu-mode/lecture9/warp_reduce_sum.cu)
    add_learning_executable(coarsed_warp_reduce_sum gpu-mode/lecture9/coarsed_warp_reduce_sum.cu)
    add_learning_executable(naive_fa2_d64 exercise/cuda/fa2/naive_fa2_d64.cu)
    add_learning_executable(kv_tile_reuse_fa2_d64 exercise/cuda/fa2/kv_tile_reuse_fa2_d64.cu)
    add_learning_executable(cute_layout_test exercise/cute/layout_test.cu)
endif()
