#include <iostream>
#include <omp.h>
#include <dlfcn.h>

#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <torch/library.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <pybind11/pybind11.h>

#include "include/cedr_ops.h"

namespace at {
namespace native {

Tensor cedr_conv2d(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {

    // Handle optional bias
    const Tensor& bias = bias_opt.has_value() ? *bias_opt : Tensor();

    // Custom conditions for optimized convolution
    if (weight.ndimension() == 4 && stride[0] == 1 && padding[0] == 1 && 
        dilation[0] == 1 && weight.size(2) == weight.size(3)) {

        // Output tensor initialization
        at::Tensor output_tensor = at::zeros(
            {input.size(0), weight.size(0), input.size(2), input.size(3)},
            input.options());

        // Apply bias if defined
        if (bias.defined() && bias.numel() > 0) {
            auto* out_data = output_tensor.data_ptr<float>();
            const auto* bias_data = bias.data_ptr<float>();
            for (int i = 0; i < bias.numel(); ++i) {
                for (int j = 0; j < input.size(2) * input.size(3); ++j) {
                    out_data[i * input.size(2) * input.size(3) + j] = bias_data[i];
                }
            }
        }

        // Call the optimized CEDR_CONV2D implementation
        cedr_conv2D(
            input.data_ptr<float>(), weight.data_ptr<float>(),
            output_tensor.data_ptr<float>(),
            input.size(2), input.size(3),
            weight.size(2), weight.size(0), input.size(1));
        return output_tensor;
    }

    // Fallback to the general PyTorch convolution API
    return at::native::convolution(
        input, weight, bias_opt, stride, padding, dilation, false, {0, 0}, groups);
}

Tensor cedr_linear(const Tensor& input, const Tensor& weight, const c10::optional<Tensor>& bias_opt) {
  const auto input_dim = input.dim();
  const auto weight_dim = weight.dim();
  TORCH_CHECK(input_dim != 0 && weight_dim != 0,
              "both arguments to linear need to be at least 1D, but they are ",
              input_dim, "D and ", weight_dim, "D");

  //if(0){
  auto bias = bias_opt.has_value()
    ? c10::MaybeOwned<Tensor>::borrowed(*bias_opt)
#if TORCH_MAJOR_VERSION == 2 && TORCH_MINOR_VERSION >= 2
    : c10::MaybeOwned<Tensor>::owned(std::in_place);
#else
    : c10::MaybeOwned<Tensor>::owned(c10::in_place);
#endif

  at::Tensor output_tensor = at::zeros({1, weight.size(0)});

  bool bias_flag = false;
  if (bias->defined()) {
    bias_flag = true;
  }

  if (bias_flag){
    _cedr_linear((float*)input.flatten().data_ptr(), (float*) weight.flatten().data_ptr(), (float*) ((Tensor)*bias).flatten().data_ptr(), (float*)output_tensor.flatten().data_ptr(), 1, input.flatten().size(0), weight.size(0), bias_flag);
  }else{
    float* empty_bias;
    _cedr_linear((float*)input.flatten().data_ptr(), (float*) weight.flatten().data_ptr(), empty_bias, (float*)output_tensor.flatten().data_ptr(), 1, input.flatten().size(0), weight.size(0), bias_flag);
  
  }
  return output_tensor;
}

Tensor cedr_conv1d(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups){

    const Tensor& bias = bias_opt.has_value() ? *bias_opt : Tensor();
    at::Tensor output_tensor = at::zeros(
        {input.size(0), weight.size(0), input.size(2)},
        input.options());

    // Apply bias if defined
    if (bias.defined() && bias.numel() > 0) {
        auto* out_data = output_tensor.data_ptr<float>();
        const auto* bias_data = bias.data_ptr<float>();
        for (int i = 0; i < bias.numel(); ++i) {
            for (int j = 0; j < input.size(2); ++j) {
                out_data[i * input.size(2) + j] = bias_data[i];
            }
        }
    }

    _cedr_conv1D_FFT(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        output_tensor.data_ptr<float>(),
        input.size(2),
        weight.size(2), weight.size(0), input.size(1));

    return output_tensor;
}

} 
} 

TORCH_LIBRARY_IMPL(aten, CPU, m) {
    m.impl("conv2d", at::native::cedr_conv2d); 
}

TORCH_LIBRARY_IMPL(aten, CPU, m) {
    m.impl("linear", at::native::cedr_linear); 
}

TORCH_LIBRARY_IMPL(aten, CPU, m) {
    m.impl("conv1d", at::native::cedr_conv1d); 
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cedr_conv2d", &at::native::cedr_conv2d, "CEDR 2D Convolution Implementation");
}