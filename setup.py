""" 
Copyright (c) 2025 by TurboDiffusion team.

Licensed under the Apache License, Version 2.0 (the "License")

Citation (please cite if you use this code):

@article{zhang2025turbodiffusion,
  title={TurboDiffusion: Accelerating Video Diffusion Models by 100-200 Times},
  author={Zhang, Jintao and Zheng, Kaiwen and Jiang, Kai and Wang, Haoxu and Stoica, Ion and Gonzalez, Joseph E and Chen, Jianfei and Zhu, Jun},
  journal={arXiv preprint arXiv:2512.16093},
  year={2025}
}
"""

from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import sys

import torch

is_rocm = torch.version.hip is not None

# On Windows, deduplicate INCLUDE/LIB/LIBPATH to avoid "command line too long" errors
if sys.platform == 'win32':
    for var in ['INCLUDE', 'LIB', 'LIBPATH']:
        val = os.environ.get(var, '')
        if val:
            unique = []
            seen = set()
            for p in val.split(';'):
                if p.lower() not in seen and p:
                    seen.add(p.lower())
                    unique.append(p)
            os.environ[var] = ';'.join(unique)

ops_dir = Path(__file__).parent / "turbodiffusion" / "ops"
cutlass_dir = ops_dir / "cutlass"
rocwmma_dir = Path(__file__).parent / "rocwmma_lib" / "projects" / "rocwmma" / "library" / "include"

if is_rocm:
    # HIP/ROCm build with rocWMMA
    hip_flags = [
        "-O3",
        "-std=c++17",
        "-D__HIP_PLATFORM_AMD__",
        "-DNDEBUG",
        # Undefine PyTorch's half conversion restrictions - rocWMMA needs these
        "-U__HIP_NO_HALF_OPERATORS__",
        "-U__HIP_NO_HALF_CONVERSIONS__",
    ]
    
    # Windows-specific: add C/C++ runtime libraries for clang-cl
    extra_libraries = []
    extra_link_args = []
    if sys.platform == 'win32':
        extra_libraries = ["msvcrt", "vcruntime", "ucrt"]
        # Force linking with MSVC C++ runtime
        extra_link_args = ["/DEFAULTLIB:msvcprt"]
    
    ext_modules = [
        CUDAExtension(
            name="turbo_diffusion_ops",
            sources=[
                "turbodiffusion/ops/bindings.cpp",
                "turbodiffusion/ops/quant/quant.hip",
                "turbodiffusion/ops/norm/rmsnorm.hip",
                "turbodiffusion/ops/norm/layernorm.hip",
                "turbodiffusion/ops/gemm/gemm_rocwmma.hip",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", "-D__HIP_PLATFORM_AMD__"],
                "nvcc": hip_flags,
            },
            include_dirs=[
                str(rocwmma_dir),
                str(ops_dir),
            ],
            libraries=extra_libraries,
            extra_link_args=extra_link_args,
        )
    ]
else:
    # CUDA build with CUTLASS
    nvcc_flags = [
        "-O3",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "-lineinfo",
        "-DNDEBUG",
    ]

    cc_flag = [
        "-gencode", "arch=compute_120a,code=sm_120a", 
        "-gencode", "arch=compute_100,code=sm_100",
        "-gencode", "arch=compute_90,code=sm_90",
        "-gencode", "arch=compute_89,code=sm_89",
        "-gencode", "arch=compute_80,code=sm_80"
    ]

    ext_modules = [
        CUDAExtension(
            name="turbo_diffusion_ops",
            sources=[
                "turbodiffusion/ops/bindings.cpp",
                "turbodiffusion/ops/quant/quant.cu", 
                "turbodiffusion/ops/norm/rmsnorm.cu",
                "turbodiffusion/ops/norm/layernorm.cu",
                "turbodiffusion/ops/gemm/gemm.cu"
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": nvcc_flags + ["-DEXECMODE=0"] + cc_flag,
            },
            include_dirs=[
                str(cutlass_dir / "include"),
                str(cutlass_dir / "tools" / "util" / "include"),
                str(ops_dir),
            ],
            libraries=["cuda"],
        )
    ]

setup(
    packages=find_packages(
        exclude=("build", "csrc", "include", "tests", "dist", "docs", "benchmarks")
    ),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
