#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Rasterization.h"
#include "Utils.cuh"
#include "Cameras.cuh"

namespace gsplat {

namespace cg = cooperative_groups;


__device__ inline void gauss_legendre3_I_and_grads(
    const vec3& gro,
    const vec3& grd,
    const vec3& ray_segment,  // to get seg_len_world
    float& I,
    vec3& dI_dgro,
    vec3& dI_dgrd)
{
    const float grd_dot_grd = glm::dot(grd, grd);
    I = 0.0f;
    dI_dgro = vec3(0.0f);
    dI_dgrd = vec3(0.0f);

    if (grd_dot_grd <= 1e-8f) {
        // Fallback: midpoint rule on [0,1] (no gradient through gro/grd)
        const float t_mid = 0.5f;
        vec3 p = gro + t_mid * grd;
        float d2 = glm::dot(p, p);
        float f_mid = __expf(-0.5f * d2);
        I = f_mid;  // ∫_0^1 f(t) dt ≈ f(0.5)
        return;
    }

    const float a = grd_dot_grd;
    const float b = glm::dot(gro, grd);
    const float c = glm::dot(gro, gro);

    const float t0 = 0.0f;
    const float t1 = 1.0f;

    const float t_star   = -b / (a + 1e-12f);
    const float t_center = fminf(fmaxf(t_star, t0), t1);

    const float dmin2 = c - (b * b) / (a + 1e-12f);
    if (dmin2 > 9.0f) {
        // negligible contribution; leave I and grads at 0
        return;
    }

    const float sigma_t    = rsqrtf(a + 1e-12f);
    const float k_sigma    = 3.0f;
    const float half_width = k_sigma * sigma_t;

    float a_int = fmaxf(t0, t_center - half_width);
    float b_int = fminf(t1, t_center + half_width);

    if (b_int <= a_int) {
        // Degenerate window: midpoint rule on [0,1]
        const float t_mid = 0.5f * (t0 + t1);
        vec3 p = gro + t_mid * grd;
        float d2 = glm::dot(p, p);
        float f_mid = __expf(-0.5f * d2);
        I = f_mid * (t1 - t0);
        return;
    }

    // 3-point Gauss–Legendre on [a_int, b_int]
    const float x0 =  0.0f;
    const float x1 = -0.7745966692414834f;  // -sqrt(3/5)
    const float x2 =  0.7745966692414834f;  //  sqrt(3/5)

    const float w0 = 0.8888888888888888f;   // 8/9
    const float w1 = 0.5555555555555556f;   // 5/9
    const float w2 = 0.5555555555555556f;   // 5/9

    const float m = 0.5f * (a_int + b_int);
    const float h = 0.5f * (b_int - a_int);

    const float t_node0 = m + h * x0;
    const float t_node1 = m + h * x1;
    const float t_node2 = m + h * x2;

    // Node 0
    vec3 r0 = gro + t_node0 * grd;
    float r2_0 = glm::dot(r0, r0);
    float f0 = __expf(-0.5f * r2_0);

    // Node 1
    vec3 r1 = gro + t_node1 * grd;
    float r2_1 = glm::dot(r1, r1);
    float f1 = __expf(-0.5f * r2_1);

    // Node 2
    vec3 r2 = gro + t_node2 * grd;
    float r2_2 = glm::dot(r2, r2);
    float f2 = __expf(-0.5f * r2_2);

    // Integral (without seg_len_world)
    I = h * (w0 * f0 + w1 * f1 + w2 * f2);

    // Gradients, ignoring dependence of t_i and h on gro/grd:
    // ∂f/∂gro = -f * r, ∂f/∂grd = -f * t * r
    vec3 dI_dgro_local(0.0f);
    vec3 dI_dgrd_local(0.0f);

    dI_dgro_local += h * w0 * (-f0 * r0);
    dI_dgro_local += h * w1 * (-f1 * r1);
    dI_dgro_local += h * w2 * (-f2 * r2);

    dI_dgrd_local += h * w0 * (-f0 * t_node0 * r0);
    dI_dgrd_local += h * w1 * (-f1 * t_node1 * r1);
    dI_dgrd_local += h * w2 * (-f2 * t_node2 * r2);

    const float seg_len_world = glm::length(ray_segment);
    I        *= seg_len_world;
    dI_dgro   = dI_dgro_local * seg_len_world;
    dI_dgrd   = dI_dgrd_local * seg_len_world;
}


template <typename scalar_t>
__global__ void rasterize_to_pixels_ultrasound_3dgs_bwd_kernel(
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
    // fwd inputs
    const vec3 *__restrict__ means,           // [B, N, 3]
    const vec4 *__restrict__ quats,           // [B, N, 4]
    const vec3 *__restrict__ scales,          // [B, N, 3]
    const scalar_t *__restrict__ intensities,      // [B, C, N, 1] or [nnz, 1]
    const scalar_t *__restrict__ transmittances,   // [B, C, N] or [nnz]
    const scalar_t *__restrict__ backgrounds, // [B, C, 1] or [nnz, 1]
    const bool *__restrict__ masks,           // [B, C, tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size_x,
    const uint32_t tile_size_y,
    const uint32_t tile_width,
    const uint32_t tile_height,
    // camera model
    const scalar_t *__restrict__ viewmats0, // [B, C, 4, 4]
    const bool convex,
    const float near_plane,
    const float far_plane,
    const float opening_angle,
    const float opening_width,
    // intersections
    const int32_t *__restrict__ tile_offsets, // [B, C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    // fwd outputs
    const scalar_t *__restrict__ render_ultrasound,         // [B, C, image_height, image_width, 1]
    const scalar_t *__restrict__ render_echo_alphas,             // [B, C, image_height, image_width, 1]
    const scalar_t *__restrict__ render_transmittances,     // [B, C, image_height, image_width, 1]
    const int32_t *__restrict__ last_ids,                   // [B, C, image_height, image_width]
    // grad outputs
    const scalar_t *__restrict__ v_render_ultrasound, // [B, C, image_height,
                                                  // image_width, CDIM]
    const scalar_t
        *__restrict__ v_render_echo_alphas, // [B, C, image_height, image_width, 1]
    const scalar_t
        *__restrict__ v_render_transmittances, // [B, C, image_height, image_width, 1]
    // grad inputs
    vec3 *__restrict__ v_means,        // [B, N, 3]
    vec4 *__restrict__ v_quats,        // [B, N, 4]
    vec3 *__restrict__ v_scales,       // [B, N, 3]
    scalar_t *__restrict__ v_intensities,   // [B, C, N, 1] or [nnz, 1]
    scalar_t *__restrict__ v_transmittances // [B, C, N] or [nnz]
) {
    auto block = cg::this_thread_block();
    uint32_t iid = block.group_index().x;
    uint32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size_y + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size_x + block.thread_index().x;

    tile_offsets += iid * tile_height * tile_width;
    render_echo_alphas += iid * image_height * image_width;
    render_transmittances += iid * image_height * image_width;
    last_ids += iid * image_height * image_width;
    render_ultrasound += iid * image_height * image_width;
    v_render_ultrasound += iid * image_height * image_width;
    v_render_echo_alphas += iid * image_height * image_width;
    v_render_transmittances += iid * image_height * image_width;
    if (backgrounds != nullptr) {
        backgrounds += iid;
    }
    if (masks != nullptr) {
        masks += iid * tile_height * tile_width;
    }

    // when the mask is provided, do nothing and return if
    // this tile is labeled as False
    if (masks != nullptr && !masks[tile_id]) {
        return;
    }

    float px = (float)j + 0.5f;
    float py = (float)i + 0.5f;
    int32_t pix_id = min(i * image_width + j, image_width * image_height - 1);

    // Create ray for pixel (in camera space)
    vec3 ray_o_cam, ray_d_cam;
    
    if (!convex) {
        // Linear probe: space rays linearly based on px
        // px=0.5 maps to -opening_width/2, px=image_width-0.5 maps to opening_width/2
        float x_pos = -opening_width / 2.0f + (px / (float)image_width) * opening_width;
        ray_o_cam = vec3(x_pos, 0.0f, 0.0f);
        ray_d_cam = vec3(0.0f, 0.0f, 1.0f);  // Always point to +z
    } else {
        // Convex probe: fan rays from origin with varying angles
        // px=0.5 maps to -opening_angle/2, px=image_width-0.5 maps to opening_angle/2
        float angle_rad = (-opening_angle / 2.0f + (px / (float)image_width) * opening_angle) * (3.14159265359f / 180.0f);
        ray_o_cam = vec3(0.0f, 0.0f, 0.0f);  // Origin at zero
        // Rotate around y-axis: x = sin(angle), z = cos(angle)
        ray_d_cam = vec3(sinf(angle_rad), 0.0f, cosf(angle_rad));
    }
    
    // Extract viewmat for this camera (4x4 matrix)
    // viewmats0 is [B, C, 4, 4], we need to index into it for the current image
    const scalar_t *viewmat = viewmats0 + iid * 16;
    
    // Convert camera-to-world (inverse of view matrix)
    // viewmat is world-to-camera, we need camera-to-world
    // For a view matrix [R|t], the inverse is [R^T | -R^T*t]
    mat3 R = mat3(
        viewmat[0], viewmat[4], viewmat[8],
        viewmat[1], viewmat[5], viewmat[9],
        viewmat[2], viewmat[6], viewmat[10]
    );
    vec3 t = vec3(viewmat[3], viewmat[7], viewmat[11]);
    
    // Transform to world space
    // For view matrix (world-to-camera): p_cam = R * p_world + t
    // So inverse (camera-to-world): p_world = R^T * (p_cam - t)
    // But for rays from camera, we use: p_world = R^T * p_cam + cam_pos
    // where cam_pos = -R^T * t
    vec3 cam_pos = -glm::transpose(R) * t;
    vec3 ray_o = glm::transpose(R) * ray_o_cam + cam_pos;
    vec3 ray_d = glm::normalize(glm::transpose(R) * ray_d_cam);

    float dist = (far_plane - near_plane) * py / (image_height - 1) + near_plane;

    // Evaluate point on ray once per pixel
    const vec3 point_on_ray = ray_o + ray_d * dist;
    const vec3 ray_segment  = point_on_ray - ray_o;

    // keep not rasterizing threads around for reading data
    bool done = (i < image_height && j < image_width);

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (iid == B * C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    const uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

        extern __shared__ int s[];
        int32_t *id_batch = (int32_t *)s; // [block_size]
        vec3 *xyz_batch =
            reinterpret_cast<vec3 *>(&id_batch[block_size]); // [block_size]
        float *transmittance_batch =
            reinterpret_cast<float *>(&xyz_batch[block_size]); // [block_size]
        vec3 *scale_batch =
            reinterpret_cast<vec3 *>(&transmittance_batch[block_size]); // [block_size]
        vec4 *quat_batch =
            reinterpret_cast<vec4 *>(&scale_batch[block_size]); // [block_size]
        mat3 *iscl_rot_batch =
            reinterpret_cast<mat3 *>(&quat_batch[block_size]); // [block_size]
        float *rgbs_batch =
            (float *)&iscl_rot_batch[block_size]; // [block_size]

    // this is the T AFTER the last gaussian in this pixel
    float alpha_sum = render_echo_alphas[pix_id];
    float T_transmittance_final = render_transmittances[pix_id];
    float T_transmittance = T_transmittance_final;

    // coverage factor γ = 1 - exp(-alpha_sum)
    const float exp_minusS = __expf(-alpha_sum);
    const float gamma = 1.0f - exp_minusS;

    const float eps = 1e-8f;

    // index of last gaussian to contribute to this pixel
    const int32_t bin_final = done ? last_ids[pix_id] : 0;

    float v_render_t = v_render_transmittances[pix_id];
    float v_render_a = v_render_echo_alphas[pix_id];

    // per-channel gradient wrt E
    const float v_O = v_render_ultrasound[pix_id];    // dL/dO
    const float O   = render_ultrasound[pix_id];      // O

    // E = O / T
    const float E_val = O / (T_transmittance_final + eps);  

    // dL/dE from ultrasound output
    const float v_E = v_O * T_transmittance_final;

    // extra contrib to dL/dT from ultrasound (O = E * T)
    v_render_t += v_O * E_val;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    const uint32_t tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int32_t warp_bin_final =
        cg::reduce(warp, bin_final, cg::greater<int>());
    for (uint32_t b = 0; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch end is the index of the last gaussian in the batch
        // These values can be negative so must be int32 instead of uint32
        const int32_t batch_end = range_end - 1 - block_size * b;
        const int32_t batch_size = min(block_size, batch_end + 1 - range_start);
        const int32_t idx = batch_end - tr;
        if (idx >= range_start) {
            // TODO: only support 1 camera for now so it is ok to abuse the index.
            int32_t isect_id = flatten_ids[idx]; // flatten index in [B * C * N] or [nnz]
            int32_t isect_bid = isect_id / (C * N);   // intersection batch index
            // int32_t isect_cid = (isect_id / N) % C;   // intersection camera index
            int32_t isect_gid = isect_id % N;         // intersection gaussian index

            id_batch[tr] = isect_id;

            const vec3 xyz   = means[isect_bid * N + isect_gid];
            const float transmittance = transmittances[isect_id];
            const vec3 scale = scales[isect_bid * N + isect_gid];
            const vec4 quat  = quats[isect_bid * N + isect_gid];

            xyz_batch[tr] = xyz;
            transmittance_batch[tr] = transmittance;
            scale_batch[tr] = scale;
            quat_batch[tr]  = quat;

            // Precompute S * R^T once per Gaussian per tile
            mat3 R = quat_to_rotmat(quat);
            mat3 S = mat3(
                1.0f / scale.x, 0.f, 0.f,
                0.f, 1.0f / scale.y, 0.f,
                0.f, 0.f, 1.0f / scale.z
            );
            iscl_rot_batch[tr] = S * glm::transpose(R);

            rgbs_batch[tr] = intensities[isect_id];
        }
        // wait for other threads to collect the gaussians in batch
        block.sync();
        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size;
             ++t) {
            bool valid = done;
            if (batch_end - t > bin_final) {
                valid = 0;
            }
            float alpha;
            float transmittance;
            float vis;

            mat3 R, S;
            vec3 xyz;
            vec3 scale;
            vec4 quat;
            vec3 delta;
            float mahal_dist, power;
            mat3 iscl_rot;
            vec3 grd, gro;
            float alpha_transmittance, gaussian_transmittance;

            vec3 dI_dgro(0.0f);
            vec3 dI_dgrd(0.0f);
            if (valid) {
                transmittance = transmittance_batch[t];
                xyz = xyz_batch[t];
                scale = scale_batch[t];
                quat = quat_batch[t];

                // Precomputed inverse-scale * rotation^T for this Gaussian
                iscl_rot = iscl_rot_batch[t];

                // R is still needed later for v_S = v_iscl_rot_total * R
                R = quat_to_rotmat(quat);
                // Reconstruct S (inverse scale) for gradient computation
                S = mat3(
                    1.0f / scale[0], 0.f,              0.f,
                    0.f,             1.0f / scale[1],  0.f,
                    0.f,             0.f,             1.0f / scale[2]
                );

                // Mahalanobis distance at this point (for opacity)
                delta = iscl_rot * (point_on_ray - xyz);
                mahal_dist = glm::dot(delta, delta);
                power = -0.5f * mahal_dist;

                vis = __expf(power);
                alpha = min(0.999f, 1.0 * vis);
                
                // Compute transmittance alpha along the ray segment
                grd = iscl_rot * ray_segment;
                gro = iscl_rot * (ray_o - xyz);
                float I = 0.0f;
                // Same Gauss–Legendre approximation as in forward, plus grads
                gauss_legendre3_I_and_grads(gro, grd, ray_segment, I, dI_dgro, dI_dgrd);
                
                // Map integral I (>=0) → smooth alpha in [0, 1)
                alpha_transmittance = 1.0f - __expf(-I);
                alpha_transmittance = fminf(alpha_transmittance, 0.999f);
                gaussian_transmittance = 1.0f - alpha_transmittance * (1.0f - transmittance);
                
                if ((power > 0.f || alpha < 1.f / 255.f) && alpha_transmittance < 1.f /255.f) {
                    valid = false;
                }
            }

            // if all threads are inactive in this warp, skip this loop
            if (!warp.any(valid)) {
                continue;
            }
            float v_rgb_local = 0.f;
            vec3 v_mean_local = {0.f, 0.f, 0.f};
            vec3 v_scale_local = {0.f, 0.f, 0.f};
            vec4 v_quat_local = {0.f, 0.f, 0.f, 0.f};
            float v_transmittance_local = 0.f;
            
            if (valid) {
                // γ = gamma
    
                float dL_dalphaj = 0.0f;  // accumulate ∂L/∂alpha_j for this Gaussian
    
                if (alpha_sum > 1e-8f && gamma > 1e-8f) {
                    // O and E at this pixel
                    const float O_val = render_ultrasound[pix_id];
                    const float E_val_local = O_val / (T_transmittance_final + eps);

                    // Gaussian's intensity
                    const float c_j = rgbs_batch[t];

                    // μ = E / γ
                    const float mu = E_val_local / (gamma + eps);

                    // dE/dalpha_j = e^{-alpha_sum} μ + γ (c_j - μ) / alpha_sum
                    const float dE_dalphaj =
                        exp_minusS * mu +
                        gamma * (c_j - mu) / (alpha_sum + eps);

                    dL_dalphaj = v_E * dE_dalphaj;

                    // ∂L/∂c_j = v_E * γ * alpha / (S+eps)
                    v_rgb_local += v_E * gamma * alpha / (alpha_sum + eps);
                }
    
                // ∂L/∂vis_j     = dL/dalpha_j
                const float v_vis = dL_dalphaj;
    

                const float v_mahal_dist = -0.5f * vis * v_vis;
                const vec3 v_delta = 2.0f * v_mahal_dist * delta;
                
                v_mean_local += -glm::transpose(iscl_rot) * v_delta;

                const vec3 diff_point_xyz = point_on_ray - xyz;
                mat3 v_iscl_rot_opacity = glm::outerProduct(v_delta, diff_point_xyz);

                const float v_gaussian_transmittance = v_render_t * T_transmittance / (gaussian_transmittance + eps);
                v_transmittance_local = v_gaussian_transmittance * alpha_transmittance;

                const float v_alpha_transmittance = v_gaussian_transmittance * (-(1.0f - transmittance));

                // alpha_transmittance = 1 - exp(-I)
                // => d alphaT / dI = exp(-I) = 1 - alpha_transmittance
                const float d_alphaT_dI = 1.0f - alpha_transmittance;
                const float v_I = v_alpha_transmittance * d_alphaT_dI;

                // 3) Back to gro, grd via I(gro, grd)
                vec3 v_gro = v_I * dI_dgro;
                vec3 v_grd = v_I * dI_dgrd;


                // gro = iscl_rot * (ray_o - xyz)
                // grd = iscl_rot * ray_segment = iscl_rot * (point_on_ray - ray_o)
                const vec3 diff_ray_o_xyz = ray_o - xyz;
                mat3 v_iscl_rot_trans = glm::outerProduct(v_gro, diff_ray_o_xyz);
                v_iscl_rot_trans += glm::outerProduct(v_grd, ray_segment);

                mat3 v_iscl_rot_total = v_iscl_rot_opacity + v_iscl_rot_trans;

                mat3 v_S = v_iscl_rot_total * R;

                v_scale_local.x += -v_S[0][0] / (scale.x * scale.x);
                v_scale_local.y += -v_S[1][1] / (scale.y * scale.y);
                v_scale_local.z += -v_S[2][2] / (scale.z * scale.z);

                mat3 v_Rt = S * v_iscl_rot_total;
                mat3 v_R = glm::transpose(v_Rt);

                quat_to_rotmat_vjp(quat, v_R, v_quat_local);

                v_mean_local += -glm::transpose(iscl_rot) * v_gro;

                T_transmittance /= gaussian_transmittance;
            }
            warpSum(v_rgb_local, warp);
            warpSum(v_mean_local, warp);
            warpSum(v_scale_local, warp);
            warpSum(v_quat_local, warp);
            warpSum(v_transmittance_local, warp);
            if (warp.thread_rank() == 0) {
                int32_t isect_id = id_batch[t]; // flatten index in [B * C * N] or [nnz]
                int32_t isect_bid = isect_id / (C * N);   // intersection batch index
                // int32_t isect_cid = (isect_id / N) % C;   // intersection camera index
                int32_t isect_gid = isect_id % N;         // intersection gaussian index
                gpuAtomicAdd((float *)(v_intensities) + isect_id, v_rgb_local);

                float *v_mean_ptr = (float *)(v_means) + 3 * (isect_bid * N + isect_gid);
                gpuAtomicAdd(v_mean_ptr, v_mean_local.x);
                gpuAtomicAdd(v_mean_ptr + 1, v_mean_local.y);
                gpuAtomicAdd(v_mean_ptr + 2, v_mean_local.z);

                float *v_scale_ptr = (float *)(v_scales) + 3 * (isect_bid * N + isect_gid);
                gpuAtomicAdd(v_scale_ptr, v_scale_local.x);
                gpuAtomicAdd(v_scale_ptr + 1, v_scale_local.y);
                gpuAtomicAdd(v_scale_ptr + 2, v_scale_local.z);

                float *v_quat_ptr = (float *)(v_quats) + 4 * (isect_bid * N + isect_gid);
                gpuAtomicAdd(v_quat_ptr, v_quat_local.x);
                gpuAtomicAdd(v_quat_ptr + 1, v_quat_local.y);
                gpuAtomicAdd(v_quat_ptr + 2, v_quat_local.z);
                gpuAtomicAdd(v_quat_ptr + 3, v_quat_local.w);

                gpuAtomicAdd(v_transmittances + isect_id, v_transmittance_local);
            }
        }
    }
}

void launch_rasterize_to_pixels_ultrasound_3dgs_bwd_kernel(
    // Gaussian parameters
    const at::Tensor means,     // [..., N, 3]
    const at::Tensor quats,     // [..., N, 4]
    const at::Tensor scales,    // [..., N, 3]
    const at::Tensor intensities,    // [..., C, N, 1] or [nnz, 1]
    const at::Tensor transmittances, // [..., C, N] or [nnz]
    const at::optional<at::Tensor> backgrounds, // [..., C, 1]
    const at::optional<at::Tensor> masks,       // [..., C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size_x,
    const uint32_t tile_size_y,
    // camera
    const at::Tensor viewmats0,               // [..., C, 4, 4]
    const bool convex,
    const float near_plane,
    const float far_plane,
    const float opening_angle,
    const float opening_width,
    // intersections
    const at::Tensor tile_offsets,    // [..., C, tile_height, tile_width]
    const at::Tensor flatten_ids,     // [n_isects]
    // forward outputs
    const at::Tensor render_ultrasound, // [..., C, image_height, image_width, 1]
    const at::Tensor render_echo_alphas,   // [..., C, image_height, image_width, 1]
    const at::Tensor render_transmittances,   // [..., C, image_height, image_width, 1]
    const at::Tensor last_ids,        // [..., C, image_height, image_width]
    // gradients of outputs
    const at::Tensor v_render_ultrasound,       // [..., C, image_height, image_width, 1]
    const at::Tensor v_render_echo_alphas,      // [..., C, image_height, image_width, 1]
    const at::Tensor v_render_transmittances,   // [..., C, image_height, image_width, 1]
    // outputs
    at::Tensor v_means,           // [..., N, 3]
    at::Tensor v_quats,           // [..., N, 4]
    at::Tensor v_scales,          // [..., N, 3]
    at::Tensor v_intensities,     // [..., C, N, 1] or [nnz, 1]
    at::Tensor v_transmittances   // [..., C, N] or [nnz]
) {
    bool packed = transmittances.dim() == 1;
    assert (packed == false); // only support non-packed for now

    uint32_t N = packed ? 0 : means.size(-2);   // number of gaussians
    uint32_t B = means.numel() / (N * 3);       // number of batches
    uint32_t C = viewmats0.size(-3);            // number of cameras
    uint32_t I = B * C;                         // number of images
    uint32_t tile_height = tile_offsets.size(-2);
    uint32_t tile_width = tile_offsets.size(-1);
    uint32_t n_isects = flatten_ids.size(0);

    // Each block covers a tile on the image. In total there are
    // I * tile_height * tile_width blocks.
    dim3 threads = {tile_size_x, tile_size_y, 1};
    dim3 grid = {I, tile_height, tile_width};

    int64_t shmem_size =
        tile_size_x * tile_size_y *
        (sizeof(int32_t) + sizeof(vec3) + sizeof(vec2) +
         sizeof(vec3) + sizeof(vec4) + sizeof(mat3) +
         sizeof(float));

    if (n_isects == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    // TODO: an optimization can be done by passing the actual number of
    // channels into the kernel functions and avoid necessary global memory
    // writes. This requires moving the channel padding from python to C side.
    if (cudaFuncSetAttribute(
            rasterize_to_pixels_ultrasound_3dgs_bwd_kernel<float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shmem_size,
            " bytes), try lowering tile_size."
        );
    }

    rasterize_to_pixels_ultrasound_3dgs_bwd_kernel<float>
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            B,
            C,
            N,
            n_isects,
            packed,
            reinterpret_cast<vec3 *>(means.data_ptr<float>()),
            reinterpret_cast<vec4 *>(quats.data_ptr<float>()),
            reinterpret_cast<vec3 *>(scales.data_ptr<float>()),
            intensities.data_ptr<float>(),
            transmittances.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                    : nullptr,
            masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
            image_width,
            image_height,
            tile_size_x,
            tile_size_y,
            tile_width,
            tile_height,
            // camera model
            viewmats0.data_ptr<float>(),
            convex,
            near_plane,
            far_plane,
            opening_angle,
            opening_width,
            // intersections
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            render_ultrasound.data_ptr<float>(),
            render_echo_alphas.data_ptr<float>(),
            render_transmittances.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(),
            v_render_ultrasound.data_ptr<float>(),
            v_render_echo_alphas.data_ptr<float>(),
            v_render_transmittances.data_ptr<float>(),
            // outputs
            reinterpret_cast<vec3 *>(v_means.data_ptr<float>()),
            reinterpret_cast<vec4 *>(v_quats.data_ptr<float>()),
            reinterpret_cast<vec3 *>(v_scales.data_ptr<float>()),
            v_intensities.data_ptr<float>(),
            v_transmittances.data_ptr<float>()
        );
}

} // namespace gsplat
