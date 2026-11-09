#include <utils.hpp>
#include <math_constants.h>

template<int BR_, int BC_, int HEADDIM_, int NWARPS_, int TCOARSE_A_, int TCOARSE_B_, bool UseVecStoreInS_>
struct KernelTraits {
    static constexpr int BR = BR_;
    static constexpr int BC = BC_;
    static constexpr int HeadDim = HEADDIM_;
    static constexpr int NWarps = NWARPS_;
    static constexpr int NThreads = 32 * NWARPS_;
    static constexpr int SMemBytes = (BR_ + 2 * BC_) * HEADDIM_ * 4;
    static constexpr int TCOARSE_A = TCOARSE_A_;
    static constexpr int TCOARSE_B = TCOARSE_B_;
    static constexpr bool UseVecStoreInS = UseVecStoreInS_;
    static_assert(TCOARSE_A_ % 4 == 0);
    static_assert(TCOARSE_B_ % 4 == 0);
    static_assert(BR_ % 4 == 0);
    static_assert(BC_ % 4 == 0);
    static_assert(HEADDIM_ % 4 == 0);
    static_assert(BR % TCOARSE_A == 0);
    static_assert(BC % TCOARSE_B == 0);
    static_assert(NThreads == (BR / TCOARSE_A_)
                 * (BC / TCOARSE_B_));
};

void dataGen(std::vector<float> &vec, int n) {
    for(int i = 0; i < n; ++i) {
        vec[i] = (float)((rand() % 10000) + 1) / 0.5f;
    }
}

template<int M, int N, int NT>
__device__ __forceinline__ void tile_load_float4(float* __restrict__ dst, const float* __restrict__ src, int tid, size_t src_offset, int n) {
    // n dimension 1 in [m, n]
    static_assert(N > 4);
    static_assert(N % 4 == 0);
    for(int q = tid; q < (M * N) / 4; q += NT) {
        int r = q / (N / 4);
        int c = (q % (N / 4)) * 4;
        size_t offset = src_offset + r * n + c;
        *reinterpret_cast<float4*>(dst + r * N + c) = *reinterpret_cast<const float4*>(src + offset);
    }
}

// BR, BC, HeadDim must be divisible by 4
template<typename KTrait>
__global__ void flash_attn(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v, float *o, int L, int D) {
    constexpr const int BR = KTrait::BR, BC = KTrait::BC, HD = KTrait::HeadDim, NT = KTrait::NThreads;
    __shared__ __align__(16) float q_tile[BR][HD];
    __shared__ __align__(16) float k_tile[BC][HD]; // transpose layout, Since Q @ K.T
    __shared__ __align__(16) float v_tile[BC][HD];
    __shared__ __align__(16) float s_tile[BR][BC];
    int tr_base = (threadIdx.x / (BC / KTrait::TCOARSE_B)) * KTrait::TCOARSE_A; 
    int tc_base = (threadIdx.x % (BC / KTrait::TCOARSE_B)) * KTrait::TCOARSE_B; 
    int tid = threadIdx.x;
    // Q loop (in splited q)
    for(int i = 0, q_end = cdiv_device(cdiv_device(L, BR), gridDim.x); i < q_end; ++i) {
        int head = blockIdx.z;
        int batch = blockIdx.y;
        int q_block_idx = blockIdx.x;
        int q_row = (q_block_idx * q_end + i) * BR;
        if(q_row >= L) break;
        size_t g_tile_offset_q = (size_t)(L) * D * batch + (size_t)(head) * L * HD + (size_t)(q_row) * HD; // Q tile offset in global Q [B, H, L, HD]
        tile_load_float4<BR, HD, NT>(&q_tile[0][0], q, tid, g_tile_offset_q, HD); // load Q tile
        float m = -CUDART_INF_F;
        float l = 0; 
        // KV loop
        for(int j = 0; j < cdiv_device(L, BC); ++j){
            float acc[KTrait::TCOARSE_A][KTrait::TCOARSE_B] = {};
            size_t g_tile_offset_kv = (size_t)(L) * D * batch + (size_t)(head) * L * HD + (size_t)(BC) * j * HD; // KV tile offset in global KV [B, H, L, HD]
            tile_load_float4<BC, HD, NT>(&k_tile[0][0], k, tid, g_tile_offset_kv, HD);
            tile_load_float4<BC, HD, NT>(&v_tile[0][0], v, tid, g_tile_offset_kv, HD);
            if constexpr (!KTrait::UseVecStoreInS) {
                #pragma unroll
                for(int ii = 0; ii < KTrait::TCOARSE_A; ++ii) {
                    #pragma unroll
                    for(int jj = 0; jj < KTrait::TCOARSE_B; ++jj) {
                        s_tile[tr_base + ii][tc_base + jj] = 0.0f; // initialize S
                    }
                }
            }
            __syncthreads();
            // compute S = Q @ K.T
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
               if constexpr (KTrait::UseVecStoreInS) {
                    #pragma unroll
                    for(int ii = 0; ii < KTrait::TCOARSE_A; ++ii) {
                        #pragma unroll
                        for(int jj = 0; jj < KTrait::TCOARSE_B; ++jj) {
                            acc[ii][jj] += reg_a[ii] * reg_b[jj];
                        }
                    }
               }else {
                    #pragma unroll
                    for(int ii = 0; ii < KTrait::TCOARSE_A; ++ii) {
                        #pragma unroll
                        for(int jj = 0; jj < KTrait::TCOARSE_B; ++jj) {
                            s_tile[tr_base + ii][tc_base + jj] += reg_a[ii] * reg_b[jj];
                        }
                    }
                }
            }
            if constexpr (KTrait::UseVecStoreInS) {
                #pragma unroll
                for(int ii = 0; ii < KTrait::TCOARSE_A; ++ii) {
                    #pragma unroll
                    for(int jj = 0; jj < KTrait::TCOARSE_B; jj += 4) {
                        *reinterpret_cast<float4*>(&s_tile[tr_base + ii][tc_base + jj]) = make_float4(acc[ii][jj], acc[ii][jj + 1], acc[ii][jj + 2], acc[ii][jj + 3]);
                    }
                }
            }
            __syncthreads();
        }
    }
}

int main(int argc, char* argvs[]) {
    std::srand(42);
    printDeviceProps();
    // QKV [B, H, L, HD]
    constexpr const int B = 1, L = 4096, HD = 128, H = 8;
    size_t numel = (size_t)(B) * L * HD * H;
    size_t bytes = (size_t)(B) * L * HD * H * sizeof(float);
    std::vector<float> q(numel, 0), k(numel, 0), v(numel, 0), o(numel, 0), res(numel, 0);
    float *device_q, *device_k, *device_v, *device_o;
    dataGen(q, numel); 
    dataGen(k, numel); 
    dataGen(v, numel); 

    CUDA_CHECK(cudaMalloc(&device_q, bytes));
    CUDA_CHECK(cudaMalloc(&device_k, bytes));
    CUDA_CHECK(cudaMalloc(&device_v, bytes));
    CUDA_CHECK(cudaMalloc(&device_o, bytes));
    CUDA_CHECK(cudaMemcpy(device_q, q.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_k, k.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_v, v.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_o, o.data(), bytes, cudaMemcpyHostToDevice));

    int case_;
    std::printf("Choose test kernel trait case:\n");
    std::cin >> case_;
    auto launch_kernel = [&](){
        switch (case_)
        {
        case 1: {
            // 64 threads in a block, compute 32 x 32 = 1024 elements, 16 thread coarsing, use 4x4
            const int q_loop_num = 4; 
            printf("a");
            const int q_split_blocks = cdiv(L, 32 * q_loop_num);
            using Trait = KernelTraits<32, 32, 128, 2, 4, 4, true>;
            assert(L % Trait::BR == 0 && L % Trait::BC == 0); // full tiles required
            dim3 grid(q_split_blocks, B, H);
            flash_attn<Trait><<<grid, Trait::NThreads, 0>>>(device_q, device_k, device_v, device_o, L, HD * H);
        }
        default:
            break;
        }
    };
    
    const int repeats = argc > 1 ? std::atoi(argvs[1]) : 10;
    const auto [gpu_total_ms, gpu_avg_ms] = kernelTimer(launch_kernel, repeats);

    CUDA_CHECK(cudaMemcpy(res.data(), device_o, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_q));
    CUDA_CHECK(cudaFree(device_k));
    CUDA_CHECK(cudaFree(device_v));
    CUDA_CHECK(cudaFree(device_o));
    return 0;
}
