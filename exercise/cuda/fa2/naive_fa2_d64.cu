#include <utils.hpp>
#include "cpu_attention.hpp"
#include <math_constants.h>
#include <type_traits>

template<int N>
struct VecLSTraits {
    static_assert(N > 0);
    static constexpr int WIDTH = 
        N % 4 == 0 ? 4 : 
        N % 2 == 0 ? 2 : 1;
    using Type = std::conditional_t<WIDTH == 4, float4, 
        std::conditional_t<WIDTH == 2, float2, float>>;
};

template<int BR_, int BC_, int HEADDIM_, int NWARPS_, int TCOARSE_A_, int TCOARSE_B_, bool RETURNLSU_>
struct KernelTraits {
    static constexpr int BR = BR_;
    static constexpr int BC = BC_;
    static constexpr int HEADDIM = HEADDIM_;
    static constexpr int NWARPS = NWARPS_;
    static constexpr int NTHREADS = 32 * NWARPS_;
    static constexpr int TCOARSE_A = TCOARSE_A_; // thread coarse when computing Q @ K.T and computing rowmax in S matrix
    static constexpr int TCOARSE_B = TCOARSE_B_; // thread coarse when computing Q @ K.T and computing rowmax in S matrix
    static constexpr bool RETURNLSU = RETURNLSU_;
    // static_assert(TCOARSE_A_ % 4 == 0);
    // static_assert(TCOARSE_B_ % 4 == 0);
    static_assert(BR_ % TCOARSE_A_ == 0);
    static_assert(BC_ % TCOARSE_B_ == 0);
    static_assert(HEADDIM_ % TCOARSE_B_ == 0);
    static_assert(NTHREADS == (BR / TCOARSE_A_)
                 * (BC / TCOARSE_B_));
    using VecLSA = typename VecLSTraits<TCOARSE_A_>::Type;
    using VecLSB = typename VecLSTraits<TCOARSE_B_>::Type;
    using VecLSHD = typename VecLSTraits<HEADDIM_>::Type;
    static constexpr int WIDTH_A = VecLSTraits<TCOARSE_A_>::WIDTH;
    static constexpr int WIDTH_B = VecLSTraits<TCOARSE_B_>::WIDTH;
    static constexpr int WIDTH_HD = VecLSTraits<HEADDIM_>::WIDTH;
};

void dataGen(std::vector<float> &vec, int n) {
    for(int i = 0; i < n; ++i) {
        vec[i] = (float)((rand() % 10000) + 1) / 0.5f;
    }
}

template<int M, int N, int NT, typename T, int WIDTH>
__device__ __forceinline__ void tile_load_vectorized(float* __restrict__ dst, const float* __restrict__ src, int tid, size_t src_offset, int n) {
    // n dimension 1 in [m, n]
    static_assert(N % WIDTH == 0);
    for(int q = tid; q < (M * N) / WIDTH; q += NT) {
        int r = q / (N / WIDTH);
        int c = (q % (N / WIDTH)) * WIDTH;
        size_t offset = src_offset + r * n + c;
        *reinterpret_cast<T*>(dst + r * N + c) = *reinterpret_cast<const T*>(src + offset);
    }
}

// N is Warp group size, thread in the same group apply the same reduce operation
template<int N>
__device__ __forceinline__ float reduce_max(float x) {
    static_assert(N >= 1 && N <= 32 && (N & (N - 1)) == 0); // N in [1, 32] and N is power of 2
    for(int offset = 1; offset < N; offset <<= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, offset));
    }
    return x;
}

// N is Warp group size, thread in the same group apply the same reduce operation
template<int N>
__device__ __forceinline__ float reduce_sum(float x) {
    static_assert(N >= 1 && N <= 32 && (N & (N - 1)) == 0); // N in [1, 32] and N is power of 2
    for(int offset = 1; offset < N; offset <<= 1) {
        x += __shfl_xor_sync(0xffffffffu, x, offset);
    }
    return x;
}

template<typename KTrait>
__global__ void flash_attn(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v, float* __restrict__ o, float* __restrict__ LSU, int L, int D) {
    constexpr const int BR = KTrait::BR, BC = KTrait::BC, HD = KTrait::HEADDIM, NT = KTrait::NTHREADS;
    constexpr const int WIDTH_A = KTrait::WIDTH_A, WIDTH_B = KTrait::WIDTH_B, WIDTH_HD = KTrait::WIDTH_HD;
    using VecLSA = KTrait::VecLSA;
    using VecLSB = KTrait::VecLSB;
    using VecLSHD = KTrait::VecLSHD;
    __shared__ __align__(16) float q_tile[BR][HD];
    __shared__ __align__(16) float k_tile[BC][HD]; // transpose layout, Since Q @ K.T
    __shared__ __align__(16) float v_tile[BC][HD];
    __shared__ __align__(16) float p_tile[BR][BC];
    int tr_base = (threadIdx.x / (BC / KTrait::TCOARSE_B)) * KTrait::TCOARSE_A; 
    int tc_base = (threadIdx.x % (BC / KTrait::TCOARSE_B)) * KTrait::TCOARSE_B; 
    int tid = threadIdx.x;
    const float qk_scale = rsqrt((float)(HD));
    // Q loop (in splited q)
    for(int i = 0, q_end = cdiv_device(cdiv_device(L, BR), gridDim.x); i < q_end; ++i) {
        int head = blockIdx.z;
        int batch = blockIdx.y;
        int q_block_idx = blockIdx.x;
        int q_row = (q_block_idx * q_end + i) * BR;
        if(q_row >= L) break;
        size_t g_tile_offset_q = (size_t)(L) * D * batch + (size_t)(head) * L * HD + (size_t)(q_row) * HD; // Q tile offset in global Q [B, H, L, HD]
        tile_load_vectorized<BR, HD, NT, VecLSHD, WIDTH_HD>(&q_tile[0][0], q, tid, g_tile_offset_q, HD); // load Q tile
        float m[KTrait::TCOARSE_A], m_old[KTrait::TCOARSE_A];
        float l[KTrait::TCOARSE_A] = {0}; 
        #pragma unroll
        for(int _t = 0; _t < KTrait::TCOARSE_A; ++_t) {
            m[_t] = -CUDART_INF_F;
        }
        constexpr int oThreadPerRow = BC / KTrait::TCOARSE_B;
        static_assert(HD % oThreadPerRow == 0);
        constexpr int oOutputColPerThread = HD / oThreadPerRow;
        constexpr int WIDTH_O = oOutputColPerThread % 4 == 0 ? 4 : 
            oOutputColPerThread % 2 == 0 ? 2 : 1;
        using VecLSO = std::conditional_t<WIDTH_O == 4, float4,
            std::conditional_t<WIDTH_O == 2, float2, float>>;
        float acc_o[KTrait::TCOARSE_A][oOutputColPerThread] = {};
        int o_tc_base = (tid % oThreadPerRow) * oOutputColPerThread;

        // KV loop
        for(int j = 0; j < cdiv_device(L, BC); ++j){
            float acc[KTrait::TCOARSE_A][KTrait::TCOARSE_B] = {};
            size_t g_tile_offset_kv = (size_t)(L) * D * batch + (size_t)(head) * L * HD + (size_t)(BC) * j * HD; // KV tile offset in global KV [B, H, L, HD]
            tile_load_vectorized<BC, HD, NT, VecLSHD, WIDTH_HD>(&k_tile[0][0], k, tid, g_tile_offset_kv, HD);
            tile_load_vectorized<BC, HD, NT, VecLSHD, WIDTH_HD>(&v_tile[0][0], v, tid, g_tile_offset_kv, HD);
            __syncthreads();
            // compute S = Q @ K.T [BR, BC]
            #pragma unroll
            for(int s = 0; s < HD; ++s){
                float reg_a[KTrait::TCOARSE_A], reg_b[KTrait::TCOARSE_B];
                #pragma unroll
                for(int ii = 0; ii < KTrait::TCOARSE_A; ++ii) {
                    reg_a[ii] = q_tile[tr_base + ii][s]; // bank conflict !
                }
                #pragma unroll
                for(int jj = 0; jj < KTrait::TCOARSE_B; ++jj) {
                    reg_b[jj] = k_tile[tc_base + jj][s]; // bank conflict !
               }
               // outer product
                #pragma unroll
                for(int ii = 0; ii < KTrait::TCOARSE_A; ++ii) {
                    #pragma unroll
                    for(int jj = 0; jj < KTrait::TCOARSE_B; ++jj) {
                        acc[ii][jj] += reg_a[ii] * reg_b[jj];
                    }
                }
            }
            // scaled dot product
            #pragma unroll
            for(int ii = 0; ii < KTrait::TCOARSE_A; ++ii) {
                #pragma unroll
                for(int jj = 0; jj < KTrait::TCOARSE_B; ++jj) {
                    acc[ii][jj] *= qk_scale;
                }
            }
            // save running max before incorporating the current KV tile.
            #pragma unroll
            for(int ii = 0; ii < KTrait::TCOARSE_A; ++ii) {
                m_old[ii] = m[ii];
            }
            #pragma unroll
            for(int ii = 0; ii < KTrait::TCOARSE_A; ++ii) {
                #pragma unroll
                for(int jj = 0; jj < KTrait::TCOARSE_B; jj += WIDTH_B) {
                    // S is not needed !!!
                    // *reinterpret_cast<float4*>(&p_tile[tr_base + ii][tc_base + jj]) = make_float4(acc[ii][jj], acc[ii][jj + 1], acc[ii][jj + 2], acc[ii][jj + 3]);
                    // m[ii] = fmaxf(m[ii], acc[ii][jj]);
                    // m[ii] = fmaxf(m[ii], acc[ii][jj + 1]);
                    // m[ii] = fmaxf(m[ii], acc[ii][jj + 2]);
                    // m[ii] = fmaxf(m[ii], acc[ii][jj + 3]);
                    #pragma unroll
                    for(int kk = 0; kk < WIDTH_B; ++kk) {
                        m[ii] = fmaxf(m[ii], acc[ii][jj + kk]);
                    }
                }
            }
            // __syncthreads();

            // compute rowmax
            constexpr int threadPerGroup = BC / KTrait::TCOARSE_B;
            #pragma unroll
            for(int ii = 0; ii < KTrait::TCOARSE_A; ++ii) {
                m[ii] = reduce_max<threadPerGroup>(m[ii]);
            }

            // compute P matrix, reuse s tile
            #pragma unroll
            for(int ii = 0; ii < KTrait::TCOARSE_A; ++ii) {
                #pragma unroll
                for(int jj = 0; jj < KTrait::TCOARSE_B; jj += WIDTH_B){
                    // acc[ii][jj] = expf(acc[ii][jj] - m[ii]);
                    // acc[ii][jj + 1] = expf(acc[ii][jj + 1] - m[ii]);
                    // acc[ii][jj + 2] = expf(acc[ii][jj + 2] - m[ii]);
                    // acc[ii][jj + 3] = expf(acc[ii][jj + 3] - m[ii]);
                    #pragma unroll
                    for(int kk = 0; kk < WIDTH_B; ++kk) {
                        acc[ii][jj + kk] = expf(acc[ii][jj + kk] - m[ii]);
                    }
                    if constexpr (std::is_same_v<float4, VecLSB>) {
                        *reinterpret_cast<float4*>(&p_tile[tr_base + ii][tc_base + jj]) = make_float4(acc[ii][jj], acc[ii][jj + 1], acc[ii][jj + 2], acc[ii][jj + 3]);
                    }
                    if constexpr (std::is_same_v<float2, VecLSB>) {
                        *reinterpret_cast<float2*>(&p_tile[tr_base + ii][tc_base + jj]) = make_float2(acc[ii][jj], acc[ii][jj + 1]);
                    }
                    if constexpr (std::is_same_v<float, VecLSB>) {
                        p_tile[tr_base + ii][tc_base + jj] = acc[ii][jj];
                    }
                }
            }
            __syncthreads(); // Now S tile becomes P tile
            
            // compute rowsum of P
            #pragma unroll
            for(int ii = 0; ii < KTrait::TCOARSE_A; ++ii) {
                float partial_row_sum = 0.0f;
                for(int jj = 0; jj < KTrait::TCOARSE_B; ++jj) {
                    partial_row_sum += acc[ii][jj];
                }
                float row_sum = reduce_sum<threadPerGroup>(partial_row_sum);
                l[ii] = expf(m_old[ii] - m[ii]) * l[ii] + row_sum;
            }

            // Compute O_j block
            #pragma unroll
            for(int ii = 0; ii < KTrait::TCOARSE_A; ++ii) {
                float mult = expf(m_old[ii] - m[ii]);
                #pragma unroll
                for(int jj = 0; jj < oOutputColPerThread; ++jj) {
                    acc_o[ii][jj] *= mult;
                }
            }
            // compute P @ V and add to O_j
            #pragma unroll
            for(int kk = 0; kk < BC; ++kk) {
                float reg_p[KTrait::TCOARSE_A], reg_v[oOutputColPerThread];
                #pragma unroll
                for(int ii = 0; ii < KTrait::TCOARSE_A; ++ii) {
                    reg_p[ii] = p_tile[tr_base + ii][kk];
                }
                #pragma unroll
                for(int ii = 0; ii < oOutputColPerThread; ii += WIDTH_O) {
                    int offset = kk * HD + o_tc_base + ii;
                    if constexpr (std::is_same_v<VecLSO, float4>) {
                        float4 v_read = *reinterpret_cast<float4*>(&v_tile[0][0] + offset);
                        reg_v[ii] = v_read.x;
                        reg_v[ii + 1] = v_read.y;
                        reg_v[ii + 2] = v_read.z;
                        reg_v[ii + 3] = v_read.w;
                    }
                    if constexpr (std::is_same_v<VecLSO, float2>) {
                        float2 v_read = *reinterpret_cast<float2*>(&v_tile[0][0] + offset);
                        reg_v[ii] = v_read.x;
                        reg_v[ii + 1] = v_read.y;
                    }
                    if constexpr (std::is_same_v<VecLSO, float>) {
                        reg_v[ii] = *(&v_tile[0][0] + offset);
                    }
                }
                #pragma unroll
                for(int ii = 0; ii < KTrait::TCOARSE_A; ++ii) {
                    #pragma unroll
                    for(int jj = 0; jj < oOutputColPerThread; ++jj) {
                        acc_o[ii][jj] += reg_p[ii] * reg_v[jj]; // outer product
                    }
                }
            }
            __syncthreads();
        }
        
        // normalize attention result
        #pragma unroll
        for(int ii = 0; ii < KTrait::TCOARSE_A; ++ii) {
            #pragma unroll
            for(int jj = 0; jj < oOutputColPerThread; ++jj) {
                acc_o[ii][jj] /= l[ii];
            }
        }

        // write O block back to HBM
        #pragma unroll
        for(int ii = 0; ii < KTrait::TCOARSE_A; ++ii) {
            #pragma unroll
            for(int jj = 0; jj < oOutputColPerThread; jj += WIDTH_O) {
                size_t offset = g_tile_offset_q + (size_t)(tr_base + ii) * HD + (size_t)(o_tc_base + jj);
                if constexpr (std::is_same_v<VecLSO, float4>) {
                    *reinterpret_cast<float4*>(o + offset) = make_float4(acc_o[ii][jj], acc_o[ii][jj + 1], acc_o[ii][jj + 2], acc_o[ii][jj + 3]);
                }
                if constexpr (std::is_same_v<VecLSO, float2>) {
                    *reinterpret_cast<float2*>(o + offset) = make_float2(acc_o[ii][jj], acc_o[ii][jj + 1]);
                }
                if constexpr (std::is_same_v<VecLSO, float>) {
                    o[offset] = acc_o[ii][jj];
                }
            }
        }

        // write LSU to HBM for backward prop if needed
        if constexpr (KTrait::RETURNLSU) {
            if(tid % oThreadPerRow == 0) {
                int num_head = D / HD;
                size_t offset = ((size_t)(batch) * num_head + head) * L;
                #pragma unroll
                for(int r = 0; r < KTrait::TCOARSE_A; ++r) {
                    LSU[offset + q_row + tr_base + r] = logf(l[r]) + m[r];
                }
            }
        }

    }
}

int main(int argc, char* argvs[]) {
    // std::srand(std::time(nullptr));
    std::srand(42);
    printDeviceProps();
    // QKV [B, H, L, HD]
    constexpr const int B = 1, HD = 64, H = 8;
    const bool test_cpu = argc > 2 && !std::strcmp(argvs[2], "cpu");
    const int L = argc > 3 ? std::atoi(argvs[3]) : 4096;
    const int repeats = argc > 1 ? std::atoi(argvs[1]) : 10;
    if (L <= 0 || L % 32 != 0 || repeats <= 0) {
        std::fprintf(stderr, "Usage: %s [positive repeats] [cpu|gpu] [L: positive multiple of 32]\n", argvs[0]);
        return EXIT_FAILURE;
    }
    size_t numel = (size_t)(B) * L * HD * H;
    size_t bytes = (size_t)(B) * L * HD * H * sizeof(float);
    std::vector<float> q(numel, 0), k(numel, 0), v(numel, 0), o(numel, 0), res(numel, 0), lsu(numel / HD, 0), res_lsu(numel / HD, 0);
    float *device_q, *device_k, *device_v, *device_o, *device_lsu;
    dataGen(q, numel); 
    dataGen(k, numel); 
    dataGen(v, numel);

    CUDA_CHECK(cudaMalloc(&device_q, bytes));
    CUDA_CHECK(cudaMalloc(&device_k, bytes));
    CUDA_CHECK(cudaMalloc(&device_v, bytes));
    CUDA_CHECK(cudaMalloc(&device_o, bytes));
    CUDA_CHECK(cudaMalloc(&device_lsu, (numel / HD) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(device_q, q.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_k, k.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_v, v.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_o, o.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_lsu, lsu.data(), (numel / HD) * sizeof(float), cudaMemcpyHostToDevice));

    using Case1Trait = KernelTraits<32, 32, 64, 2, 4, 4, true>;
    int case_;
    std::printf("Choose test kernel trait case:\n");
    std::printf("  1: BR=%d, BC=%d, HD=%d, NWARPS=%d, NTHREADS=%d, "
                "TCOARSE_A=%d, TCOARSE_B=%d, RETURNLSU=%s\n",
                Case1Trait::BR, Case1Trait::BC, Case1Trait::HEADDIM,
                Case1Trait::NWARPS, Case1Trait::NTHREADS,
                Case1Trait::TCOARSE_A, Case1Trait::TCOARSE_B,
                Case1Trait::RETURNLSU ? "true" : "false");
    if (!(std::cin >> case_) || case_ != 1) {
        std::fprintf(stderr, "Only kernel trait case 1 is implemented\n");
        return EXIT_FAILURE;
    }
    auto launch_kernel = [&](){
        switch (case_)
        {
        case 1: {
            // 64 threads in a block, compute 32 x 32 = 1024 elements, 16 thread coarsing, use 4x4
            const int q_loop_num = 4; 
            const int q_split_blocks = cdiv(L, 32 * q_loop_num);
            using Trait = Case1Trait;
            assert(L % Trait::BR == 0 && L % Trait::BC == 0); // full tiles required
            dim3 grid(q_split_blocks, B, H);
            flash_attn<Trait><<<grid, Trait::NTHREADS, 0>>>(device_q, device_k, device_v, device_o, device_lsu, L, HD * H);
        }
        default:
            break;
        }
        CUDA_CHECK(cudaGetLastError());
    };
    
    const auto [gpu_total_ms, gpu_avg_ms] = kernelTimer(launch_kernel, repeats);

    CUDA_CHECK(cudaMemcpy(res.data(), device_o, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(res_lsu.data(), device_lsu, (numel / HD) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_q));
    CUDA_CHECK(cudaFree(device_k));
    CUDA_CHECK(cudaFree(device_v));
    CUDA_CHECK(cudaFree(device_o));
    CUDA_CHECK(cudaFree(device_lsu));
    if (test_cpu) {
        std::vector<float> cpu_o, cpu_lsu;
        const auto cpu_start = std::chrono::steady_clock::now();
        attention_cpu(q, k, v, cpu_o, cpu_lsu, B, H, L, HD);
        const auto cpu_stop = std::chrono::steady_clock::now();
        const double cpu_ms =
            std::chrono::duration<double, std::milli>(cpu_stop - cpu_start).count();
        std::printf("CPU attention: %.3f ms (includes result allocation)\n", cpu_ms);
        if (gpu_avg_ms > 0.0) {
            std::printf("CPU time / GPU kernel time: %.2fx\n", cpu_ms / gpu_avg_ms);
        }
        const bool o_ok = check_attention("O", res, cpu_o, H, L, HD);
        const bool lsu_ok = check_attention("LSU", res_lsu, cpu_lsu, H, L, 1);
        if (!o_ok || !lsu_ok) return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
