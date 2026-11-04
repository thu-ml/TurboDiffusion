/* 
 * Copyright (c) 2025 by TurboDiffusion team.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * 
 * Citation (please cite if you use this code):
 * 
 * @article{zhang2025turbodiffusion,
 *   title={TurboDiffusion: Accelerating Video Diffusion Models by 100-200 Times},
 *   author={Zhang, Jintao and Zheng, Kaiwen and Jiang, Kai and Wang, Haoxu and Stoica, Ion and Gonzalez, Joseph E and Chen, Jianfei and Zhu, Jun},
 *   journal={arXiv preprint arXiv:2512.16093},
 *   year={2025}
 * }
 * 
 * HIP/ROCm quantization kernel.
 */

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <torch/all.h>
#include <torch/python.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

#include "common/common_hip.hpp"
#include "quant/quant_hip.hpp"

auto quant(
  torch::Tensor const& Input, 
  std::optional<torch::Tensor>& Output, 
  std::optional<torch::Tensor>& Output_S
) {
  using ElementOut = int8_t;
  static constexpr int BlockSize = 128;
  static constexpr int NumThrPerCta = 256;

  int64_t m = Input.size(0);
  int64_t n = Input.size(1);
  torch::Device const input_device = Input.device();

  create_tensor<BlockSize>(input_device, Output, Output_S, m, n);
  
  ElementOut *Optr = (ElementOut*)Output.value().data_ptr();
  float *OSptr = Output_S.value().data_ptr<float>();

  switch (Input.scalar_type()) {
    case torch::kHalf: {
      __half *Iptr = (__half*)Input.data_ptr();
      quantization<__half, BlockSize, NumThrPerCta>(
          Iptr, Optr, OSptr, m, n, at::hip::getCurrentHIPStream().stream()
      );
      break;
    }

    case torch::kBFloat16: {
      hip_bfloat16 *Iptr = (hip_bfloat16*)Input.data_ptr();
      quantization<hip_bfloat16, BlockSize, NumThrPerCta>(
          Iptr, Optr, OSptr, m, n, at::hip::getCurrentHIPStream().stream()
      );
      break;
    }

    default: {
      std::cerr << "Observing: " << Input.scalar_type() << " for the input datatype which is invalid";
      throw std::runtime_error("Unsupported input data type for quantize_to_fp4.");
    }
  }
  
  return std::make_tuple(Output, Output_S);
}

void register_quant(pybind11::module_ &m) {
    m.def("quant_cuda", &quant);
}

