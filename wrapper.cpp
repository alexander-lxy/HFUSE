#include <torch/extension.h>

#include <cuda.h>
#include <vector>

using at::IntArrayRef;
using at::TensorList;
using torch::Tensor;

namespace at
{
namespace native
{

std::tuple<Tensor, Tensor> _histc_upsample(
    const Tensor& self,
    int64_t nbins,
    Scalar min,
    Scalar max,
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners
  );


} // namespace native
} // namespace at

Tensor histc_upsample(Tensor hist_input, Tensor input_upsample)
{
  at::native::_histc_upsample(
    hist_input, 20, 0.f, 0.f,
                              input_upsample, {2000, 2560}, true);
 return torch::randn({100, 100});
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("histc_upsample", &histc_upsample, "LLTM forward (CUDA)");
}
