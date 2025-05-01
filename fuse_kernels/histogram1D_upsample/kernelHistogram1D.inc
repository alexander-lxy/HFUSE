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