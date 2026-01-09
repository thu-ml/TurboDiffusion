// Compatibility header for CUTLASS numeric conversion on HIP/ROCm
// This provides a minimal subset of CUTLASS functionality needed for TurboDiffusion

#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

namespace cutlass {

// FloatRoundStyle enum (subset of CUTLASS)
enum class FloatRoundStyle {
    round_to_nearest = 0,
    round_toward_zero = 1,
    round_toward_infinity = 2,
    round_toward_neg_infinity = 3,
};

// NumericConverter template - provides float to int8 conversion with rounding
template <typename To, typename From, FloatRoundStyle Round = FloatRoundStyle::round_to_nearest>
struct NumericConverter {
    __device__ __host__ __forceinline__
    To operator()(From const& val) const {
        return static_cast<To>(val);
    }
};

// Specialization for float to int8_t with round_to_nearest
template <>
struct NumericConverter<int8_t, float, FloatRoundStyle::round_to_nearest> {
    __device__ __host__ __forceinline__
    int8_t operator()(float val) const {
        // Round to nearest and clamp to int8 range [-128, 127]
        val = fmaxf(-128.0f, fminf(127.0f, rintf(val)));
        return static_cast<int8_t>(val);
    }
};

// Specialization for float to int8_t with round_toward_zero
template <>
struct NumericConverter<int8_t, float, FloatRoundStyle::round_toward_zero> {
    __device__ __host__ __forceinline__
    int8_t operator()(float val) const {
        // Truncate and clamp to int8 range [-128, 127]
        val = fmaxf(-128.0f, fminf(127.0f, truncf(val)));
        return static_cast<int8_t>(val);
    }
};

} // namespace cutlass

