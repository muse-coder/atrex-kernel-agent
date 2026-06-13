// FP8 GEMM Kernel for SM120 (RTX PRO 5000 Blackwell)
// C[M,N] = (scale_a * scale_b) * A[M,K] @ B[N,K]^T
// A: (M,K) FP8 E4M3 row-major, B: (N,K) FP8 E4M3 row-major
// C: (M,N) BF16 row-major
//
// Uses CUTLASS 3.x CollectiveBuilder for SM120.
// SM120 (Blackwell desktop) uses mma.sync.aligned.kind::f8f6f4 PTX instructions
// (similar to SM100's QMMA) with register accumulators. The CUTLASS SM120
// collective handles TMA loads, pipeline staging, and warp-specialized execution.
//
// Two kernel schedules are instantiated:
// - Pingpong: Better for small M (fewer waves, less 2-CTA overhead)
// - Cooperative: Better for large M (higher SM utilization with 2-CTA cooperation)
//
// Shape-based dispatch selects the optimal schedule at runtime.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

// ============================================================================
// MODULE: gemm-config BEGIN
// CUTLASS 3.x GEMM configuration template for SM120 FP8 E4M3
// Parameterized by TileShape, ClusterShape, and schedule.
// ============================================================================

template <typename TileShape_,
          typename ClusterShape_ = Shape<_1,_1,_1>,
          typename KernelSchedule_ = cutlass::gemm::collective::KernelScheduleAuto,
          typename EpilogueSchedule_ = cutlass::epilogue::collective::EpilogueScheduleAuto>
struct FP8GemmSm120Config {
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementC = void;  // no C input (beta=0)
  using ElementD = cutlass::bfloat16_t;
  using ElementAccumulator = float;
  using ElementCompute = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  static constexpr int AlignmentAB = 128 / cutlass::sizeof_bits<ElementA>::value;  // 16
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementD>::value;   // 8
  static constexpr int AlignmentD = AlignmentC;

  using TileShape = TileShape_;
  using ClusterShape = ClusterShape_;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      EpilogueSchedule_
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentAB,
      ElementB, LayoutB, AlignmentAB,
      ElementAccumulator,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule_
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// MODULE: gemm-config END

// ============================================================================
// MODULE: kernel-launch BEGIN
// Persistent workspace + kernel launch
// ============================================================================

// Persistent workspace to avoid per-call cudaMalloc/cudaFree
static void* g_workspace = nullptr;
static size_t g_workspace_size = 0;

static void ensure_workspace(size_t needed, cudaStream_t stream) {
  if (needed <= g_workspace_size) return;
  if (g_workspace) {
    cudaStreamSynchronize(stream);
    cudaFree(g_workspace);
  }
  // Round up to 1MB boundary for fewer reallocations
  size_t alloc_size = ((needed + (1 << 20) - 1) >> 20) << 20;
  cudaMalloc(&g_workspace, alloc_size);
  g_workspace_size = alloc_size;
}

template <typename Config>
void run_fp8_gemm(
    const void* A_ptr,
    const void* B_ptr,
    void* D_ptr,
    float alpha,
    int M, int N, int K,
    cudaStream_t stream) {

  using GemmKernel = typename Config::GemmKernel;
  using Gemm = typename Config::Gemm;
  using ElementA = typename Config::ElementA;
  using ElementB = typename Config::ElementB;
  using ElementD = typename Config::ElementD;
  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;

  auto prob_shape = cute::make_shape(M, N, K, 1);

  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
  StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  auto* a_data = static_cast<const ElementA*>(A_ptr);
  auto* b_data = static_cast<const ElementB*>(B_ptr);
  auto* d_data = static_cast<ElementD*>(D_ptr);

  typename GemmKernel::MainloopArguments mainloop_args{
      a_data, stride_a,
      b_data, stride_b};

  // D = alpha * Acc (beta=0, no C)
  typename GemmKernel::EpilogueArguments epilogue_args{
      {alpha, 0.0f},
      nullptr, stride_c,
      d_data, stride_d};

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      prob_shape,
      mainloop_args,
      epilogue_args,
      hw_info};

  Gemm gemm_op;
  auto status = gemm_op.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(
        std::string("CUTLASS cannot implement: ") +
        cutlassGetStatusString(status));
  }

  size_t ws_size = gemm_op.get_workspace_size(args);
  ensure_workspace(ws_size, stream);

  status = gemm_op.initialize(args, g_workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(
        std::string("CUTLASS init failed: ") +
        cutlassGetStatusString(status));
  }

  status = gemm_op.run(stream);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(
        std::string("CUTLASS run failed: ") +
        cutlassGetStatusString(status));
  }
}

// MODULE: kernel-launch END

// ============================================================================
// MODULE: dispatch BEGIN
// Shape-based dispatch: pingpong for small M, cooperative for large M
// ============================================================================

// Config 1: Pingpong 128x128x128 -- single CTA per SM, good for small M
using ConfigPingpong_128x128x128 = FP8GemmSm120Config<
    Shape<_128, _128, _128>,
    Shape<_1, _1, _1>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>;

// Config 2: Cooperative 128x128x128 -- 2-CTA per SM pair, good for large M
using ConfigCoop_128x128x128 = FP8GemmSm120Config<
    Shape<_128, _128, _128>,
    Shape<_1, _1, _1>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>;

// Note: 128x256x64 tile was tested but performed poorly (~6x slower for M=1024)
// due to suboptimal register allocation or MMA scheduling on SM120.
// Sticking with 128x128x128 which matches the proven cuBLAS tile shape.

void fp8_gemm_dispatch(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor scale_a,
    torch::Tensor scale_b,
    torch::Tensor C) {

  TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(),
              "All tensors must be on CUDA");
  TORCH_CHECK(A.dtype() == torch::kFloat8_e4m3fn, "A must be float8_e4m3fn");
  TORCH_CHECK(B.dtype() == torch::kFloat8_e4m3fn, "B must be float8_e4m3fn");
  TORCH_CHECK(C.dtype() == torch::kBFloat16, "C must be bfloat16");
  TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
  TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
  TORCH_CHECK(C.is_contiguous(), "C must be contiguous");

  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(0);

  TORCH_CHECK(B.size(1) == K, "K dimension mismatch");
  TORCH_CHECK(C.size(0) == M && C.size(1) == N, "C shape mismatch");

  float sa = scale_a.item<float>();
  float sb = scale_b.item<float>();
  float alpha = sa * sb;

  auto stream = at::cuda::getCurrentCUDAStream();

  // Shape-based dispatch:
  // M<=4096: pingpong 128x128x128 (avoids 2-CTA cooperative overhead)
  // M>4096: cooperative 128x128x128 (higher throughput with 2-CTA pairing)
  if (M <= 4096) {
    run_fp8_gemm<ConfigPingpong_128x128x128>(
        A.data_ptr(), B.data_ptr(), C.data_ptr(),
        alpha, M, N, K, stream);
  } else {
    run_fp8_gemm<ConfigCoop_128x128x128>(
        A.data_ptr(), B.data_ptr(), C.data_ptr(),
        alpha, M, N, K, stream);
  }
}

// MODULE: dispatch END

// ============================================================================
// Python binding
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_gemm", &fp8_gemm_dispatch,
        "FP8 GEMM with per-tensor scaling for SM120 (Blackwell)",
        py::arg("A"), py::arg("B"),
        py::arg("scale_a"), py::arg("scale_b"),
        py::arg("C"));
}
