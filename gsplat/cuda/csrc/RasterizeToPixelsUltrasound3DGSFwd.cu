#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Rasterization.h"
#include "Cameras.cuh"
#include "Utils.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

////////////////////////////////////////////////////////////////
// Forward
////////////////////////////////////////////////////////////////

template <typename scalar_t>
__global__ void rasterize_to_pixels_ultrasound_3dgs_fwd_kernel(
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const vec3 *__restrict__ means,           // [B, N, 3]
    const vec4 *__restrict__ quats,           // [B, N, 4]
    const vec3 *__restrict__ scales,          // [B, N, 3]
    const scalar_t *__restrict__ intensities,      // [B, C, N, 1] or [nnz, 1]
    const scalar_t *__restrict__ transmittances,   // [B, C, N] or [nnz]
    const scalar_t *__restrict__ backgrounds, // [B, C, 1]
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
    scalar_t
        *__restrict__ render_ultrasound,            // [B, C, image_height, image_width, 1]
    scalar_t *__restrict__ render_echo_alphas,      // [B, C, image_height, image_width, 1]
    scalar_t *__restrict__ render_transmittances,   // [B, C, image_height, image_width, 1]
    scalar_t *__restrict__ render_echoes,           // [B, C, image_height, image_width, 1]
    int32_t *__restrict__ last_ids                  // [B, C, image_height, image_width]
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    auto block = cg::this_thread_block();
    int32_t iid = block.group_index().x;
    int32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size_y + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size_x + block.thread_index().x;

    tile_offsets += iid * tile_height * tile_width;
    render_ultrasound += iid * image_height * image_width;
    render_echo_alphas += iid * image_height * image_width;
    render_transmittances += iid * image_height * image_width;
    render_echoes += iid * image_height * image_width;
    last_ids += iid * image_height * image_width;
    if (backgrounds != nullptr) {
        backgrounds += iid;
    }
    if (masks != nullptr) {
        masks += iid * tile_height * tile_width;
    }

    float px = (float)j + 0.5f;
    float py = (float)i + 0.5f;
    int32_t pix_id = i * image_width + j;

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

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);
    bool done = !inside;

    // when the mask is provided, render the background color and return
    // if this tile is labeled as False
    if (masks != nullptr && inside && !masks[tile_id]) {
        render_ultrasound[pix_id] =
            backgrounds == nullptr ? 0.0f : backgrounds[0];
        render_echo_alphas[pix_id] = 0.0f;
        render_transmittances[pix_id] = 1.0f;
        last_ids[pix_id] = -1;
        return;
    }

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (iid == B * C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s; // [block_size]
    vec3 *xyz_batch =
        reinterpret_cast<vec3 *>(&id_batch[block_size]); // [block_size]
    float *transmittance_batch =
        reinterpret_cast<float *>(&xyz_batch[block_size]); // [block_size]
    mat3 *iscl_rot_batch =
        reinterpret_cast<mat3 *>(&transmittance_batch[block_size]); // [block_size]
    
    float alpha_sum = 0.0f;
    float T_transmittance = 1.0f;
    // index of most recent gaussian to write to this thread's pixel
    uint32_t cur_idx = 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    uint32_t tr = block.thread_rank();

    float pix_out = 0.f;
    for (uint32_t b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        uint32_t batch_start = range_start + block_size * b;
        uint32_t idx = batch_start + tr;
        if (idx < range_end) {
            // TODO: only support 1 camera for now so it is ok to abuse the index.
            int32_t isect_id = flatten_ids[idx]; // flatten index in [B * C * N] or [nnz]
            int32_t isect_bid = isect_id / (C * N);   // intersection batch index
            // int32_t isect_cid = (isect_id / N) % C;   // intersection camera index
            int32_t isect_gid = isect_id % N;         // intersection gaussian index
            id_batch[tr] = isect_id;
            const vec3 xyz = means[isect_bid * N + isect_gid];
            const float transmittance = transmittances[isect_id];
            xyz_batch[tr] = xyz;
            transmittance_batch[tr] = {transmittance};
            
            const vec4 quat = quats[isect_bid * N + isect_gid];
            vec3 scale = scales[isect_bid * N + isect_gid];
            
            mat3 R = quat_to_rotmat(quat);
            mat3 S = mat3(
                1.0f / scale[0],
                0.f,
                0.f,
                0.f,
                1.0f / scale[1],
                0.f,
                0.f,
                0.f,
                1.0f / scale[2]
            );
            mat3 iscl_rot = S * glm::transpose(R);
            iscl_rot_batch[tr] = iscl_rot;
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        uint32_t batch_size = min(block_size, range_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            const float transmittance = transmittance_batch[t];
            const vec3 xyz = xyz_batch[t];
            const mat3 iscl_rot = iscl_rot_batch[t];

            //** Alpha on ray tip for intensity **
            // point_on_ray is precomputed per pixel
            const vec3 delta = iscl_rot * (point_on_ray - xyz);
            const float mahal_dist = glm::dot(delta, delta);
            const float power = -0.5f * mahal_dist;

            float alpha_opacity = min(0.999f, 1.0 * __expf(power));
            // --- Ray in Gaussian space for transmittance integral ---
            const vec3 gro = iscl_rot * (ray_o - xyz);
            const vec3 grd = iscl_rot * ray_segment;
            const float grd_dot_grd = glm::dot(grd, grd);

            // Integral I ≈ ∫_0^1 exp(-0.5 * ||gro + t*grd||^2) dt
            // using 3-point Gauss–Legendre around closest approach of the ray to the Gaussian.
            float I = 0.0f;
            if (grd_dot_grd > 1e-8f) {
                // Quadratic coefficients along ray in Gaussian space
                const float a = grd_dot_grd;                 // = ||grd||^2
                const float b = glm::dot(gro, grd);
                const float c = glm::dot(gro, gro);

                // Ray parameter range for this segment
                const float t0 = 0.0f;
                const float t1 = 1.0f;

                // Closest approach along the infinite ray
                const float t_star = -b / (a + 1e-12f);
                // Clamp to actual segment
                const float t_center = fminf(fmaxf(t_star, t0), t1);

                // Minimum squared distance in Gaussian space
                const float dmin2 = c - (b * b) / (a + 1e-12f);
                // If the ray misses the Gaussian badly in transformed space, skip
                if (dmin2 > 9.0f) { // ~exp(-4.5) ≈ 0.011
                    I = 0.0f;
                } else {
                    // Std dev of the 1D Gaussian along t
                    const float sigma_t = rsqrtf(a + 1e-12f); // 1 / sqrt(a)
                    const float k_sigma = 3.0f;               // integrate over ±3σ
                    const float half_width = k_sigma * sigma_t;

                    // Integration window: [a_int, b_int] = [t_center - kσ, t_center + kσ] ∩ [0,1]
                    float a_int = fmaxf(t0, t_center - half_width);
                    float b_int = fminf(t1, t_center + half_width);

                    if (b_int > a_int) {
                        // 3-point Gauss–Legendre on [a_int, b_int]
                        // Standard nodes/weights on [-1, 1]:
                        // x0 = 0, x1 = -sqrt(3/5), x2 = +sqrt(3/5)
                        // w0 = 8/9, w1 = w2 = 5/9
                        const float x0 =  0.0f;
                        const float x1 = -0.7745966692414834f; // -sqrt(3/5)
                        const float x2 =  0.7745966692414834f; //  sqrt(3/5)

                        const float w0 = 0.8888888888888888f;  // 8/9
                        const float w1 = 0.5555555555555556f;  // 5/9
                        const float w2 = 0.5555555555555556f;  // 5/9

                        const float m = 0.5f * (a_int + b_int); // midpoint
                        const float h = 0.5f * (b_int - a_int); // half-length

                        // t_i = m + h * x_i
                        const float t_node0 = m + h * x0;
                        const float t_node1 = m + h * x1;
                        const float t_node2 = m + h * x2;

                        // Evaluate integrand at nodes: f(t) = exp(-0.5 * ||gro + t*grd||^2)
                        vec3 v0 = gro + t_node0 * grd;
                        float r2_0 = glm::dot(v0, v0);
                        float f0 = __expf(-0.5f * r2_0);

                        vec3 v1 = gro + t_node1 * grd;
                        float r2_1 = glm::dot(v1, v1);
                        float f1 = __expf(-0.5f * r2_1);

                        vec3 v2 = gro + t_node2 * grd;
                        float r2_2 = glm::dot(v2, v2);
                        float f2 = __expf(-0.5f * r2_2);

                        // Integral over t in [a_int, b_int]
                        I = h * (w0 * f0 + w1 * f1 + w2 * f2);
                    } else {
                        // If the window degenerates (very thin), fall back to midpoint on [0,1]
                        const float t_mid = 0.5f * (t0 + t1);
                        vec3 p = gro + t_mid * grd;
                        float d2 = glm::dot(p, p);
                        I = __expf(-0.5f * d2) * (t1 - t0);
                    }
                }
            } else {
                // Degenerate ray direction in Gaussian space → fallback to midpoint
                const float t_mid = 0.5f; // midpoint of [0,1]
                vec3 p = gro + t_mid * grd;
                float d2 = glm::dot(p, p);
                I = __expf(-0.5f * d2);
            }

            // Use ray length in *world* space (simpler, no param grad)
            const float seg_len_world = glm::length(ray_segment);
            I *= seg_len_world;

            // Map integral I (>=0) → smooth alpha in [0, 1)
            // for small I, alpha ≈ I; for large I, saturates.
            float alpha_transmittance = 1.0f - __expf(-I);
            alpha_transmittance = fminf(alpha_transmittance, 0.999f);

            // transmittance at center, towards 1 at the rim
            float gaussian_transmittance = 1.0f - alpha_transmittance * (1.0f - transmittance);

            if (alpha_opacity < 1.f / 255.f && alpha_transmittance < 1.f / 255.f) {
                continue;
            }

            const float gaussian_dist = glm::length(xyz - ray_o);
            
            const float next_T_transmittance = T_transmittance * gaussian_transmittance;

            int32_t isect_id = id_batch[t];
            pix_out += fminf(intensities[isect_id], 0.999f) * alpha_opacity;
            cur_idx = batch_start + t;

            alpha_sum += alpha_opacity;
            T_transmittance = next_T_transmittance;
        }
    }

    if (inside) {
        render_echo_alphas[pix_id] = alpha_sum;
        render_transmittances[pix_id] = T_transmittance;
        const float cov_factor = 1.0f - __expf(-alpha_sum);
        const float mu = pix_out / (alpha_sum + 1e-8f);
        const float E = (1.0f - cov_factor) * 0.0f + cov_factor * mu;
        render_echoes[pix_id] = E;
        render_ultrasound[pix_id] = E * T_transmittance;
        // index in bin of last gaussian in this pixel
        last_ids[pix_id] = static_cast<int32_t>(cur_idx);
    }
}

void launch_rasterize_to_pixels_ultrasound_3dgs_fwd_kernel(
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
    const at::Tensor tile_offsets, // [..., C, tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // outputs
    at::Tensor render_ultrasound,       // [..., C, image_height, image_width, 1]
    at::Tensor render_echo_alphas,      // [..., C, image_height, image_width]
    at::Tensor render_transmittances,   // [..., C, image_height, image_width]
    at::Tensor render_echoes,           // [..., C, image_height, image_width, 1]
    at::Tensor last_ids                 // [..., C, image_height, image_width]
) {
    // Note: quats need to be normalized before passing in.
    uint32_t N = means.size(-2);                // number of gaussians
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
        (sizeof(int32_t) + sizeof(vec3) + sizeof(vec2) + sizeof(mat3));

    if (cudaFuncSetAttribute(
        rasterize_to_pixels_ultrasound_3dgs_fwd_kernel<float>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shmem_size
    ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shmem_size,
            " bytes), try lowering tile_size_x or tile_size_y."
        );
    }

    rasterize_to_pixels_ultrasound_3dgs_fwd_kernel<float>
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            B,
            C,
            N,
            n_isects,
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
            render_echoes.data_ptr<float>(),
            last_ids.data_ptr<int32_t>()
        );
}

} // namespace gsplat
