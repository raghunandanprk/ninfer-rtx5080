// Native-Windows stub for Blackwell NVFP4 TMA kernels.
//
// MSVC rejects CUDA kernels that pass 128-byte-aligned CUtensorMap descriptors
// by value (C2719). The RentedNoodle GSQ/RCO artifact does not use NVFP4 weight
// kernels, so Windows builds replace those entry points with loud runtime stubs
// while keeping the Q3/Q4/Q5 GSQ/RCO paths available.

#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma_launch.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

[[noreturn]] void nvfp4_tma_unavailable() {
    throw std::runtime_error(
        "NVFP4 TMA kernels are disabled in the native Windows build because "
        "MSVC cannot pass 128-byte-aligned CUtensorMap descriptors by value. "
        "Use a Q3/Q4/Q5 GSQ/RCO artifact such as the RentedNoodle build.");
}

} // namespace

void launch_nvfp4_w4a4_tma_linear(Nvfp4Problem, const std::uint8_t*, const std::uint8_t*,
                                  const std::uint8_t*, const std::uint8_t*, __nv_bfloat16*,
                                  std::int32_t, float, cudaStream_t) {
    nvfp4_tma_unavailable();
}

void launch_nvfp4_w4a4_tma_attention(const std::uint8_t*, const std::uint8_t*,
                                     const std::uint8_t*, const std::uint8_t*, __nv_bfloat16*,
                                     __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, std::int32_t,
                                     float, cudaStream_t) {
    nvfp4_tma_unavailable();
}

void launch_nvfp4_w4a4_tma_gdn(const std::uint8_t*, const std::uint8_t*, const std::uint8_t*,
                               const std::uint8_t*, __nv_bfloat16*, __nv_bfloat16*, std::int32_t,
                               float, cudaStream_t) {
    nvfp4_tma_unavailable();
}

void launch_nvfp4_w4a4_tma_linear_add(Nvfp4Problem, const std::uint8_t*, const std::uint8_t*,
                                      const std::uint8_t*, const std::uint8_t*, __nv_bfloat16*,
                                      std::int32_t, float, cudaStream_t) {
    nvfp4_tma_unavailable();
}

void launch_nvfp4_linear_swiglu_w4a4_tma(const std::uint8_t*, const std::uint8_t*,
                                         const std::uint8_t*, const std::uint8_t*,
                                         __nv_bfloat16*, std::int32_t, float, cudaStream_t) {
    nvfp4_tma_unavailable();
}

} // namespace ninfer::ops::detail
