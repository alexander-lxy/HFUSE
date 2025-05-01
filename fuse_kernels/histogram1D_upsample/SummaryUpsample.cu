#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

// #include <THC/THCGeneral.h>
// #include <THC/THCDeviceUtils.cuh>

#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/div_rtn.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <c10/macros/Macros.h>
#include <ATen/native/im2col_shape_check.h>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include "../cuda/UpSample.cuh"
#include "../cuda/DeviceSqrt.cuh"
#include "../cuda/LaunchUtils.h"

#include <cuda_profiler_api.h>
#include "../../timeProfiler.h"

const int NUM_KERNELS = 16;
float kernel_times[NUM_KERNELS];
const char* kernel_names[NUM_KERNELS] = {
    "0-vfuse", "0-vfuse_lb", "0-hfuse", "0-hfuse_lb",
    "1-hfuse", "1-hfuse_lb", "2-hfuse", "2-hfuse_lb",
    "3-hfuse", "3-hfuse_lb", "4-hfuse", "4-hfuse_lb",
    "5-hfuse", "5-hfuse_lb", "6-hfuse", "6-hfuse_lb"
};


namespace at {
namespace native {


__device__ __forceinline__ size_t
idx(const size_t nc,
    const size_t height,
    const size_t width,
    const size_t y,
    const size_t x) {
  return (nc * height + y) * width + x;
}

#include "upsample_bilinear2d_out_frame.inc"

using namespace at::cuda;
using namespace at::cuda::detail;

#define THRESH_NUMBER_BINS_FOR_MULTI_BLOCK_MEM 100
#define THRESH_NUMBER_BINS_FOR_GLOBAL_MEM 1000
#define FOR_KERNEL_LOOP(i, lim)                                      \
  for (IndexType i = blockIdx.x * blockDim.x + threadIdx.x; i < lim; \
       i += gridDim.x * blockDim.x)

/*
  Memory types used for the 3 histogram implementations.
  See `CUDA_tensor_histogram` below.
 */
enum class CUDAHistogramMemoryType { SHARED, MULTI_BLOCK, GLOBAL };
namespace {
  template<typename input_t, typename IndexType>
  __device__ static IndexType getBin(input_t bVal, input_t minvalue, input_t maxvalue, int nbins) {
    IndexType bin = (int)((bVal - minvalue) * nbins / (maxvalue - minvalue));
    // (only applicable for histc)
    // while each bin is inclusive at the lower end and exclusive at the higher, i.e. [start, end)
    // the last bin is inclusive at both, i.e. [start, end], in order to include maxvalue if exists
    // therefore when bin == nbins, adjust bin to the last bin
    if (bin == nbins) bin -= 1;
    return bin;
  }
}

#include "kernelHistogram1D_upsample_bilinear2d_out_frame_.inc"

/*
  Kernel for computing the histogram of the input.
 */
template <
    typename output_t,
    typename input_t,
    typename IndexType,
    int ADims,
    int PDims,
    int BDims,
    CUDAHistogramMemoryType MemoryType = CUDAHistogramMemoryType::MULTI_BLOCK,
    typename Op>
#ifdef __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_1(512)
#endif
__global__ void kernelHistogram1D(
    TensorInfo<output_t, IndexType> a, /* output */
    TensorInfo<output_t, IndexType> p, /* partial output */
    TensorInfo<input_t, IndexType> b, /* input */
    int nbins,
    input_t minvalue,
    input_t maxvalue,
    IndexType totalElements,
    Op getOp) {
    extern unsigned char my_smem[] __attribute__((shared));
    output_t *smem = nullptr;
    smem = reinterpret_cast<output_t *>(my_smem);
    for (IndexType i = threadIdx.x; i < a.sizes[0]; i += blockDim.x) {
        smem[i] = 0;
    }
    __syncthreads();
    for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x; linearIndex < totalElements; linearIndex += gridDim.x * blockDim.x) {
        const IndexType bOffset = IndexToOffset<input_t, IndexType, BDims>::get(linearIndex, b);
        const input_t bVal = b.data[bOffset];
        if (bVal >= minvalue && bVal <= maxvalue) {
            const IndexType bin = getBin<input_t, IndexType>(bVal, minvalue, maxvalue, nbins);
            atomicAdd(& smem[bin], getOp(linearIndex));
        }
    }
    __syncthreads();
    for (IndexType i = threadIdx.x; i < a.sizes[0]; i += blockDim.x) {
        const IndexType aOffset = IndexToOffset<output_t, IndexType, ADims>::get(i, a);
        atomicAdd(& a.data[aOffset], smem[i]);
    }
}


inline int64_t getFreeGlobalMemory() {
  // no need to use `cudaSetDevice`
  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);
  AT_ASSERTM(
      cudaGetLastError() == cudaSuccess,
      "CUDA_tensor_histogram failed to get free global memory");
  return static_cast<int64_t>(free_mem);
}


template <typename input_hist_t>
std::tuple<Tensor, Tensor> _histc_cuda_template(
    const Tensor& self_hist,
    int64_t nbins,
    input_hist_t min,
    input_hist_t max,
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners
  ) {
  printf("2\n");
  if (nbins <= 0) {
    AT_ERROR("bins must be > 0");
  }
  at::Tensor output_hist = at::zeros({nbins}, at::device(at::kCUDA).dtype(self_hist.scalar_type()));

  // Tensor output_hist = native::zeros({nbins}, device(DeviceType::CUDA).dtype(self_hist.scalar_type()));
  input_hist_t minvalue = min;
  input_hist_t maxvalue = max;
  if (min == max) {
    minvalue = *self_hist.min().cpu().data<input_hist_t>();
    maxvalue = *self_hist.max().cpu().data<input_hist_t>();
  }
  if (minvalue == maxvalue) {
    minvalue = minvalue - 1;
    maxvalue = maxvalue + 1;
  }

  printf("3\n");
  {
  checkBackend("CUDA_tensor_histogram", {output_hist, self_hist}, Backend::CUDA);
  auto totalElements = self_hist.numel();

  const dim3 block = getApplyBlock();
  dim3 grid;
  int64_t curDevice = current_device();

  grid.x = 10000;

  CUDAHistogramMemoryType memType = CUDAHistogramMemoryType::GLOBAL;
  auto maxSharedMem = getCurrentDeviceProperties()->sharedMemPerBlock;
  auto sharedMem = nbins * sizeof(input_hist_t) + 8; // 8 guard bytes
  auto maxGlobalMem = getFreeGlobalMemory();
  auto multiBlockMem = nbins * grid.x * sizeof(input_hist_t) + 8; // 8 guard bytes
  // determine memory type to use in the kernel
    printf("6\n");
  if (nbins < THRESH_NUMBER_BINS_FOR_MULTI_BLOCK_MEM &&
      sharedMem < maxSharedMem) {
    printf("shared\n");
    memType = CUDAHistogramMemoryType::SHARED;
  } else if (
      nbins < THRESH_NUMBER_BINS_FOR_GLOBAL_MEM &&
      multiBlockMem < (maxGlobalMem / 2)) {
    // check against half of free mem to be extra safe
    // due to cached allocator, we may anyway have slightly more free mem
    printf("mb\n");
    memType = CUDAHistogramMemoryType::MULTI_BLOCK;
  }

  // alloc memory for MULTI_BLOCK
  using IndexType = int64_t;
  auto aInfo = getTensorInfo<input_hist_t, IndexType>(output_hist);
  auto bInfo = getTensorInfo<input_hist_t, IndexType>(self_hist);
  TensorInfo<input_hist_t, IndexType> pInfo(nullptr, 0, {}, {});
  Tensor partial_output_hist;
  if (memType == CUDAHistogramMemoryType::MULTI_BLOCK) {
    // partial_output_hist = native::zeros({grid.x, nbins}, output_hist.options());
    partial_output_hist = at::zeros({grid.x, nbins}, output_hist.options());

    pInfo = getTensorInfo<input_hist_t, IndexType>(partial_output_hist);
  }

  printf("7\n");
  printf("10\n");

  Tensor output = at::empty_like(input);
  TensorArg input_arg{input, "input", 1}, output_arg{output, "output", 2};
  checkAllSameGPU("upsample_bilinear2d_out_cuda", {input_arg, output_arg});

  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  int output_height = output_size[0];
  int output_width = output_size[1];

  int nbatch = input.size(0);
  int channels = input.size(1);
  int input_height = input.size(2);
  int input_width = input.size(3);

  upsample_2d_shape_check(
      input,
      Tensor(),
      nbatch,
      channels,
      input_height,
      input_width,
      output_height,
      output_width);

  output.resize_({input.size(0), input.size(1), output_height, output_width});

  AT_ASSERT(
      input_height > 0 && input_width > 0 && output_height > 0 &&
      output_width > 0);

  const int num_kernels = output_height * output_width;
  const int num_threads = std::min(
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 512);

  printf("%d %d\n", num_kernels, num_threads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "upsample_bilinear2d_out_frame", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = input.packed_accessor<scalar_t, 4>();
        auto odata = output.packed_accessor<scalar_t, 4>();

        const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(
            input_height, output_height, align_corners);
        const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
            input_width, output_width, align_corners);

        const int num_blocks = cuda::ATenCeilDiv(num_kernels, num_threads);
        printf("%d\n", num_blocks);
        cudaDeviceSynchronize();
        cudaEvent_t start_native, stop_native;
        startTimer(start_native, stop_native);
        upsample_bilinear2d_out_frame<scalar_t, accscalar_t>
            <<<num_blocks,
               num_threads,
               0,
               getStreamFromPool(true)>>>(
                num_kernels, rheight, rwidth, align_corners, idata, odata);
        static const auto getDummyOp = [] __device__(IndexType) { return 1L; };
        kernelHistogram1D<input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED>
        <<<grid,
          block,
          sharedMem,
          getStreamFromPool(true)>>>(
            aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp);
        cudaDeviceSynchronize();
        stopTimerAndPrint("native", start_native, stop_native);
        #define CALL(i,type,thread,idx)\
        cudaEvent_t start, stop;\
        startTimer(start, stop);\
        kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_##type##_idx_##i<input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED, decltype(getDummyOp), scalar_t, accscalar_t>\
        <<<grid,\
          thread,\
          sharedMem,\
          getStreamFromPool(true)>>>(\
            aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp,\
                num_kernels, rheight, rwidth, align_corners, idata, odata\
          );\
        stopTimerAndPrint(&kernel_times[idx], start, stop);\
        cudaDeviceSynchronize()

      CALL(0, vfuse,512, 0);
      CALL(0, vfuse_lb,512, 1);
      CALL(0, hfuse,1024, 2);
      CALL(0, hfuse_lb,1024, 3);
      CALL(1, hfuse,1024, 4);
      CALL(1, hfuse_lb,1024, 5);
      CALL(2, hfuse,1024, 6);
      CALL(2, hfuse_lb,1024, 7);
      CALL(3, hfuse,1024, 8);
      CALL(3, hfuse_lb,1024, 9);
      CALL(4, hfuse,1024, 10);
      CALL(4, hfuse_lb,1024, 11);
      CALL(5, hfuse,1024, 12);
      CALL(5, hfuse_lb,1024, 13);
      CALL(6, hfuse,1024, 14);
      CALL(6, hfuse_lb,1024, 15);
        // cudaProfilerStop();
      });

  AT_ASSERTM(cudaGetLastError() == cudaSuccess, "kernelHistogram1D failed");
  return std::make_tuple(output_hist, output_hist);
}
}
} // namespace

namespace native {

std::tuple<Tensor, Tensor> _histc_upsample(
    const Tensor& self,
    int64_t nbins,
    Scalar min,
    Scalar max,
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners
  ) {
  if (self.scalar_type() == ScalarType::Half) {
    AT_ERROR("HalfTensor is not supported");
  }
    printf("0\n");
  return AT_DISPATCH_ALL_TYPES(self.scalar_type(), "histc", [&] {
    printf("1\n");
    return native::_histc_cuda_template<scalar_t>(self, nbins, min.to<scalar_t>(), max.to<scalar_t>(),
    input,
    output_size,
    align_corners
  );
  });
}

} // namespace native
} // namespace at
