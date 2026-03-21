#pragma once

#include <cstdint>

namespace at {
class Tensor;
}

namespace gsplat {

void launch_spherical_harmonics_fwd_kernel(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor dirs,                // [..., 3]
    const at::Tensor coeffs,              // [..., K, C]
    const at::optional<at::Tensor> masks, // [...]
    // outputs
    at::Tensor colors // [..., C]
);

void launch_spherical_harmonics_bwd_kernel(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor dirs,                // [..., 3]
    const at::Tensor coeffs,              // [..., K, C]
    const at::optional<at::Tensor> masks, // [...]
    const at::Tensor v_colors,            // [..., C]
    // outputs
    at::Tensor v_coeffs,            // [..., K, C]
    at::optional<at::Tensor> v_dirs // [..., 3]
);

} // namespace gsplat