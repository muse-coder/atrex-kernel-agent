// FP8 GEMM Kernel for SM120 (RTX PRO 5000 Blackwell Desktop)
// Uses mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
// A: M×K FP8 E4M3 (row-major), B: N×K FP8 E4M3 (row-major)
// C = scale * (A @ B^T), output BF16

#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// =========================================================
// MODULE: ptx_wrappers BEGIN
// =========================================================

__device__ __forceinline__ void mma_m16n8k32_fp8(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1)
    );
}

// =========================================================
// MODULE: ptx_wrappers END
// =========================================================

// =========================================================
// MODULE: kernel BEGIN
// =========================================================

static constexpr int TILE_M = 128;
static constexpr int TILE_N = 128;
static constexpr int TILE_K = 64;
static constexpr int BLOCK_SIZE = 256;
static constexpr int WARP_SIZE = 32;

// Warp layout: (4,2) in (M,N)
static constexpr int WARP_ROWS = 4;
static constexpr int WARP_COLS = 2;
static constexpr int WARP_M = TILE_M / WARP_ROWS; // 32
static constexpr int WARP_N = TILE_N / WARP_COLS;  // 64

// MMA tile dimensions
static constexpr int MMA_M = 16;
static constexpr int MMA_N = 8;
static constexpr int MMA_K = 32;

// Per-warp MMA tile counts
static constexpr int NUM_MMA_M = WARP_M / MMA_M; // 2
static constexpr int NUM_MMA_N = WARP_N / MMA_N;  // 8
static constexpr int NUM_MMA_K = TILE_K / MMA_K;  // 2

__global__ __launch_bounds__(BLOCK_SIZE)
void fp8_gemm_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    float scale,
    int M, int N, int K)
{
    __shared__ uint8_t smem_A[TILE_M * TILE_K];
    __shared__ uint8_t smem_B[TILE_N * TILE_K];

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int warp_m = warp_id / WARP_COLS;
    const int warp_n = warp_id % WARP_COLS;

    // PTX ISA fragment layout: Group = lane_id/4 (0..7), ThreadInGroup = lane_id%4 (0..3)
    const int grp = lane_id >> 2;
    const int thd = lane_id & 3;

    const int m_start = blockIdx.x * TILE_M;
    const int n_start = blockIdx.y * TILE_N;

    float acc[NUM_MMA_M][NUM_MMA_N][4];
    #pragma unroll
    for (int i = 0; i < NUM_MMA_M; i++)
        #pragma unroll
        for (int j = 0; j < NUM_MMA_N; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;

    for (int k_start = 0; k_start < K; k_start += TILE_K) {
        {
            int row = tid >> 1;
            int col = (tid & 1) << 5;
            const uint4* src_a = reinterpret_cast<const uint4*>(
                A + (size_t)(m_start + row) * K + k_start + col);
            uint4* dst_a = reinterpret_cast<uint4*>(smem_A + row * TILE_K + col);
            dst_a[0] = src_a[0];
            dst_a[1] = src_a[1];

            const uint4* src_b = reinterpret_cast<const uint4*>(
                B + (size_t)(n_start + row) * K + k_start + col);
            uint4* dst_b = reinterpret_cast<uint4*>(smem_B + row * TILE_K + col);
            dst_b[0] = src_b[0];
            dst_b[1] = src_b[1];
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < NUM_MMA_K; kk++) {
            int k_off = kk * MMA_K;
            int k_base = 4 * thd;

            #pragma unroll
            for (int mm = 0; mm < NUM_MMA_M; mm++) {
                int m_off = warp_m * WARP_M + mm * MMA_M;
                int row0 = m_off + 2 * grp;

                const uint32_t* ap0 = reinterpret_cast<const uint32_t*>(
                    smem_A + row0 * TILE_K + k_off + k_base);
                const uint32_t* ap1 = reinterpret_cast<const uint32_t*>(
                    smem_A + (row0 + 1) * TILE_K + k_off + k_base);
                uint32_t a0 = ap0[0];
                uint32_t a1 = ap1[0];
                uint32_t a2 = ap0[4];
                uint32_t a3 = ap1[4];

                #pragma unroll
                for (int mn = 0; mn < NUM_MMA_N; mn++) {
                    int n_off = warp_n * WARP_N + mn * MMA_N;

                    const uint32_t* bp = reinterpret_cast<const uint32_t*>(
                        smem_B + (n_off + grp) * TILE_K + k_off + k_base);
                    uint32_t b0 = bp[0];
                    uint32_t b1 = bp[4];

                    mma_m16n8k32_fp8(
                        acc[mm][mn][0], acc[mm][mn][1],
                        acc[mm][mn][2], acc[mm][mn][3],
                        a0, a1, a2, a3, b0, b1);
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int mm = 0; mm < NUM_MMA_M; mm++) {
        #pragma unroll
        for (int mn = 0; mn < NUM_MMA_N; mn++) {
            int row0 = m_start + warp_m * WARP_M + mm * MMA_M + 2 * grp;
            int col0 = n_start + warp_n * WARP_N + mn * MMA_N + 2 * thd;

            C[row0 * N + col0]         = __float2bfloat16(acc[mm][mn][0] * scale);
            C[row0 * N + col0 + 1]     = __float2bfloat16(acc[mm][mn][1] * scale);
            C[(row0 + 1) * N + col0]     = __float2bfloat16(acc[mm][mn][2] * scale);
            C[(row0 + 1) * N + col0 + 1] = __float2bfloat16(acc[mm][mn][3] * scale);
        }
    }
}

// =========================================================
// MODULE: kernel END
// =========================================================

// =========================================================
// MODULE: launch BEGIN
// =========================================================

void fp8_gemm_launch(
    torch::Tensor A,
    torch::Tensor B,
    float scale_a,
    float scale_b,
    torch::Tensor C)
{
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    TORCH_CHECK(M % TILE_M == 0, "M must be divisible by ", TILE_M);
    TORCH_CHECK(N % TILE_N == 0, "N must be divisible by ", TILE_N);
    TORCH_CHECK(K % TILE_K == 0, "K must be divisible by ", TILE_K);

    dim3 grid(M / TILE_M, N / TILE_N);

    fp8_gemm_kernel<<<grid, BLOCK_SIZE>>>(
        static_cast<const uint8_t*>(A.data_ptr()),
        static_cast<const uint8_t*>(B.data_ptr()),
        static_cast<__nv_bfloat16*>(C.data_ptr()),
        scale_a * scale_b,
        M, N, K);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel launch failed: ", cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp8_gemm_launch", &fp8_gemm_launch, "FP8 GEMM kernel launch");
}

// =========================================================
// MODULE: launch END
// =========================================================
