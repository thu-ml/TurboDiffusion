/*
 * Copyright (c) 2025 by TurboDiffusion team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 *
 * Platform abstraction layer for CUDA/HIP compatibility.
 * This header provides unified macros and types for both backends.
 */

#pragma once

// Detect platform
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    #define TURBO_PLATFORM_HIP 1
    #define TURBO_PLATFORM_CUDA 0
#else
    #define TURBO_PLATFORM_HIP 0
    #define TURBO_PLATFORM_CUDA 1
#endif

// Include appropriate runtime headers
#if TURBO_PLATFORM_HIP
    #include <hip/hip_runtime.h>
    #include <hip/hip_fp16.h>
#else
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <cuda_fp16.h>
#endif

// Stream type abstraction
#if TURBO_PLATFORM_HIP
    using turboStream_t = hipStream_t;
    using turboError_t = hipError_t;
    #define turboSuccess hipSuccess
    #define turboGetErrorString hipGetErrorString
    #define turboGetLastError hipGetLastError
    #define turboFuncSetAttribute hipFuncSetAttribute
    #define turboFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#else
    using turboStream_t = cudaStream_t;
    using turboError_t = cudaError_t;
    #define turboSuccess cudaSuccess
    #define turboGetErrorString cudaGetErrorString
    #define turboGetLastError cudaGetLastError
    #define turboFuncSetAttribute cudaFuncSetAttribute
    #define turboFuncAttributeMaxDynamicSharedMemorySize cudaFuncAttributeMaxDynamicSharedMemorySize
#endif

// Device function qualifiers
#if TURBO_PLATFORM_HIP
    #define TURBO_HOST __host__
    #define TURBO_DEVICE __device__
    #define TURBO_HOST_DEVICE __host__ __device__
    #define TURBO_KERNEL __global__
    #define TURBO_INLINE __forceinline__
#else
    #define TURBO_HOST __host__
    #define TURBO_DEVICE __device__
    #define TURBO_HOST_DEVICE __host__ __device__
    #define TURBO_KERNEL __global__
    #define TURBO_INLINE __forceinline__
#endif

// Pragma unroll
#if TURBO_PLATFORM_HIP
    #define TURBO_PRAGMA_UNROLL _Pragma("unroll")
    #define TURBO_PRAGMA_NO_UNROLL _Pragma("nounroll")
#else
    #define TURBO_PRAGMA_UNROLL _Pragma("unroll")
    #define TURBO_PRAGMA_NO_UNROLL _Pragma("nounroll")
#endif

// Error checking macro
#define TURBO_CHECK(call)                                                      \
    do {                                                                        \
        turboError_t err = (call);                                              \
        if (err != turboSuccess) {                                              \
            fprintf(stderr, "GPU Error at %s:%d: %s\n", __FILE__, __LINE__,     \
                    turboGetErrorString(err));                                  \
            exit(err);                                                          \
        }                                                                       \
    } while (0)

// Half precision types
#if TURBO_PLATFORM_HIP
    using half_t = __half;
    using bfloat16_t = hip_bfloat16;
    
    TURBO_DEVICE TURBO_INLINE float __int2float_rn_hip(int x) {
        return static_cast<float>(x);
    }
    #define __int2float_rn __int2float_rn_hip
    
    TURBO_DEVICE TURBO_INLINE float __int_as_float_hip(int x) {
        return __int_as_float(x);
    }
#else
    #include <cutlass/numeric_types.h>
    using half_t = cutlass::half_t;
    using bfloat16_t = cutlass::bfloat16_t;
#endif

// Warp/Wave primitives
#if TURBO_PLATFORM_HIP
    // RDNA3 uses wave32
    #define TURBO_WARP_SIZE 32
    #define TURBO_FULL_MASK 0xFFFFFFFFu
    
    TURBO_DEVICE TURBO_INLINE float warpReduceSum(float val) {
        TURBO_PRAGMA_UNROLL
        for (int offset = TURBO_WARP_SIZE / 2; offset > 0; offset >>= 1) {
            val += __shfl_xor(val, offset, TURBO_WARP_SIZE);
        }
        return val;
    }
    
    TURBO_DEVICE TURBO_INLINE float warpReduceMax(float val) {
        TURBO_PRAGMA_UNROLL
        for (int offset = TURBO_WARP_SIZE / 2; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_xor(val, offset, TURBO_WARP_SIZE));
        }
        return val;
    }
#else
    #define TURBO_WARP_SIZE 32
    #define TURBO_FULL_MASK 0xFFFFFFFFu
    
    TURBO_DEVICE TURBO_INLINE float warpReduceSum(float val) {
        TURBO_PRAGMA_UNROLL
        for (int offset = TURBO_WARP_SIZE / 2; offset > 0; offset >>= 1) {
            val += __shfl_xor_sync(TURBO_FULL_MASK, val, offset);
        }
        return val;
    }
    
    TURBO_DEVICE TURBO_INLINE float warpReduceMax(float val) {
        TURBO_PRAGMA_UNROLL
        for (int offset = TURBO_WARP_SIZE / 2; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_xor_sync(TURBO_FULL_MASK, val, offset));
        }
        return val;
    }
#endif

// Synchronization
#if TURBO_PLATFORM_HIP
    #define __syncwarp() __syncthreads()
#endif

// Kernel launch helper
template <class Kernel>
TURBO_KERNEL void device_kernel_impl(
    typename Kernel::Params const params
) {
    extern __shared__ char smem[];
    Kernel op;
    op(params, smem);
}

template <class Kernel>
void launch_kernel_unified(
    typename Kernel::Params const& params,
    dim3 grid_shape,
    dim3 cta_shape,
    size_t ShmSize,
    turboStream_t stream = nullptr
) {
    auto func = device_kernel_impl<Kernel>;
    if (ShmSize >= 48 * 1024) {
        TURBO_CHECK(turboFuncSetAttribute(
            func,
            turboFuncAttributeMaxDynamicSharedMemorySize,
            ShmSize
        ));
    }
#if TURBO_PLATFORM_HIP
    hipLaunchKernelGGL(func, dim3(grid_shape), dim3(cta_shape), ShmSize, stream, params);
#else
    func<<<grid_shape, cta_shape, ShmSize, stream>>>(params);
#endif
    TURBO_CHECK(turboGetLastError());
}

// Numeric conversion helpers
namespace turbo {

template <typename To, typename From>
TURBO_DEVICE TURBO_INLINE To convert(From val) {
    return static_cast<To>(val);
}

#if TURBO_PLATFORM_HIP
template <>
TURBO_DEVICE TURBO_INLINE int8_t convert<int8_t, float>(float val) {
    // Round to nearest with clamping
    val = fmaxf(-128.0f, fminf(127.0f, rintf(val)));
    return static_cast<int8_t>(val);
}
#else
template <>
TURBO_DEVICE TURBO_INLINE int8_t convert<int8_t, float>(float val) {
    return cutlass::NumericConverter<int8_t, float, cutlass::FloatRoundStyle::round_to_nearest>()(val);
}
#endif

} // namespace turbo

