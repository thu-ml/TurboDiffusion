/* 
 * Copyright (c) 2025 by TurboDiffusion team.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * 
 * HIP/ROCm RMSNorm kernel.
 */

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <torch/all.h>
#include <torch/python.h>
#include <hip/hip_runtime.h>

#include "common/common_hip.hpp"
#include "norm/rmsnorm_hip.hpp"

auto rms_norm(
  at::Tensor const& Input, 
  float eps,
  const std::optional<at::Tensor>& Weight,
  std::optional<at::Tensor>& Output
) {
  using ElementIn = float;
  using ElementOut = float;
  using ElementWeight = float;
  
  int64_t const m = Input.size(0);
  int64_t const n = Input.size(1);
  torch::Device const input_device = Input.device();
  
  if (!Output.has_value()) {
    Output.emplace(
      torch::empty(
        {m, n},
        torch::TensorOptions().device(input_device).dtype(torch::kFloat32)
      )
    );
  }

  void *Iptr = Input.data_ptr();
  void *Wptr = Weight.has_value() ? Weight.value().data_ptr() : nullptr;
  void *Optr = Output.value().data_ptr();

  CONFIG_SWITCH(n, [&]{
    rmsnorm<
      ElementIn, ElementOut, ElementWeight,
      MAX_HIDDEN_SIZE, NUM_THR_PER_CTA
    >(
      Iptr, Wptr,
      Optr, 
      eps, m, n, 
      at::hip::getCurrentHIPStream().stream()
    );
  });

  return Output;
}

void register_rms_norm(pybind11::module_ &m) {
    m.def("rms_norm_cuda", &rms_norm);
}

