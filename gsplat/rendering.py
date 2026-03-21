import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from typing_extensions import Literal

from .cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_offset_encode_rectangular,
    isect_tiles,
    isect_tiles_rectangular,
    ultrasound_rasterize_to_pixels,
    rasterize_to_pixels,
    spherical_harmonics,
)


def ultrasound_rasterization(
    means: Tensor,  # [..., N, 3]
    quats: Tensor,  # [..., N, 4]
    scales: Tensor,  # [..., N, 3]
    transmittances: Tensor,  # [..., N]
    intensities: Tensor,  # [..., (C,) N, D] or [..., (C,) N, K, D]
    viewmats: Tensor,  # [..., C, 4, 4]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    opening_angle: Optional[float] = None,
    opening_width: Optional[float] = None,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    tile_size_x: int = 4,
    tile_size_y: int = 128,
    backgrounds: Optional[Tensor] = None,  # [..., C, D]
    channel_chunk: int = 32,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict]:
    """Physics-based ultrasound rasterization of 3D Gaussians to B-mode images.

    Renders a set of 3D Gaussians (N) to a batch of ultrasound image planes (C)
    using a custom ray casting scheme that separates transmittance and echo
    intensity: :math:`B(p) = T(p) \\cdot E(p)`.

    The probe geometry is determined by the ``opening_angle`` / ``opening_width``
    arguments (exactly one must be provided):

    - **Convex probe** (``opening_angle``): Perspective camera model where rays
      fan out from the probe origin across the specified angular field of view.
    - **Linear probe** (``opening_width``): Orthographic camera model where
      parallel rays are spaced uniformly across the specified physical width.

    Camera intrinsics are constructed internally from the probe parameters — no
    explicit ``Ks`` matrix is needed.

    .. note::
        **Batch Rasterization**: This function supports batched rendering by
        providing batched ``viewmats``.

    .. note::
        **Spherical Harmonics**: If ``sh_degree`` is set, ``intensities`` is
        treated as SH coefficients [..., (C,) N, K, D] and view-dependent echo
        intensities are computed. Otherwise ``intensities`` should be
        post-activation values [..., (C,) N, D].

    .. note::
        **Rectangular Tiles**: This function uses rectangular tiles
        (``tile_size_x`` × ``tile_size_y``, default 4×128) for efficient
        ultrasound rasterization. This is effective since a vertical stripe shares Gaussians.

    .. note::

    Args:
        means: The 3D centers of the Gaussians. [..., N, 3]
        quats: The quaternions of the Gaussians (wxyz convention). Not required
            to be normalized. [..., N, 4]
        scales: The scales of the Gaussians. [..., N, 3]
        transmittances: Per-Gaussian acoustic transmittance values. [..., N]
        intensities: Echo intensities. [..., (C,) N, D] for post-activation
            values, or [..., (C,) N, K, D] for SH coefficients.
        viewmats: The world-to-camera transformations. [..., C, 4, 4]
        width: The width of the output image in pixels.
        height: The height of the output image in pixels.
        near_plane: The near plane for clipping. Default is 0.01.
        far_plane: The far plane for clipping. Default is 1e10.
        opening_angle: The angular field of view in degrees for a convex probe.
            Mutually exclusive with ``opening_width``. Default is None.
        opening_width: The physical width (in scene units, typically cm) for a linear probe.
            Mutually exclusive with ``opening_angle``. Default is None.
        eps2d: An epsilon added to the eigenvalues of projected 2D covariance
            matrices, preventing Gaussians from becoming too small. Default is 0.3.
        sh_degree: The SH degree to use. If set, ``intensities`` is treated as
            SH coefficients; otherwise as post-activation values. Default is None.
        tile_size_x: Lateral tile size for rectangular rasterization. Default is 4.
        tile_size_y: Axial tile size for rectangular rasterization. Default is 128.
        backgrounds: Background values. [..., C, D]. Default is None.
        channel_chunk: Max channels to render per pass. Default is 32. Channels
            exceeding this are rendered in sequential chunks.

    Returns:
        A tuple of five elements:

        **render_ultrasound**: The final B-mode image (echo × transmittance).
        [..., C, height, width, D].

        **render_echo_alphas**: Accumulated echo alpha. [..., C, height, width, 1].

        **render_transmittances**: Accumulated transmittance along each ray.
        [..., C, height, width, 1].

        **render_echoes**: Echo intensity only (without transmittance).
        [..., C, height, width, D].

        **meta**: A dictionary of intermediate results including projection
        outputs, tile intersection data, and probe configuration.

    """
    meta = {}

    # Transducer can be convex or linear
    assert not (opening_angle is not None and opening_width is not None)

    device = means.device
    batch_dims = means.shape[:-2]
    num_batch_dims = len(batch_dims)
    B = math.prod(batch_dims)
    N = means.shape[-2]
    C = viewmats.shape[-3]
    I = B * C
    camera_model = "ortho" if opening_width is not None else "pinhole"

    # For projecting gaussians to 2d, we will need camera charactersitics.
    # For simplicity we emulate it by setting a very low y-fov.
    # When rendering ultrasound, we skip this entirely and just create rays manually, since we need to calculate depth anyway.
    if camera_model == "pinhole":
        # Convex transducer
        fov_x = math.radians(opening_angle)
        fov_y = math.radians(0.01)

        fx = width / (2 * math.tan(fov_x / 2))
        fy = height / (2 * math.tan(fov_y / 2))
        cx = width / 2.0
        cy = height / 2.0

        base_K = torch.tensor(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=means.dtype,
        )  # [3, 3]
    else:
        # Linear transducer
        fx = width / opening_width
        fy = height / 0.001
        cx = width / 2.0
        cy = height / 2.0

        base_K = torch.tensor(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=means.dtype,
        )  # [3, 3]

    # viewmats has shape batch_dims + (C, 4, 4)
    Ks = base_K
    while Ks.dim() < len(batch_dims) + 2:
        Ks = Ks.unsqueeze(0)
    Ks = Ks.expand(batch_dims + (C, 3, 3))
    assert means.shape == batch_dims + (N, 3), means.shape
    assert transmittances.shape == batch_dims + (N,), transmittances.shape
    assert viewmats.shape == batch_dims + (C, 4, 4), viewmats.shape
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.shape

    if sh_degree is None:
        # treat colors as post-activation values, should be in shape [..., N, D] or [..., C, N, D]
        assert (
            intensities.dim() == num_batch_dims + 2
            and intensities.shape[:-1] == batch_dims + (N,)
        ) or (
            intensities.dim() == num_batch_dims + 3
            and intensities.shape[:-1] == batch_dims + (C, N)
        ), intensities.shape
    else:
        # treat colors as SH coefficients, should be in shape [..., N, K, D] or [..., C, N, K, D]
        # Allowing for activating partial SH bands
        assert (
            intensities.dim() == num_batch_dims + 3
            and intensities.shape[:-2] == batch_dims + (N,)
        ) or (
            intensities.dim() == num_batch_dims + 4
            and intensities.shape[:-2] == batch_dims + (C, N)
        ), intensities.shape
        assert (sh_degree + 1) ** 2 <= intensities.shape[-2], intensities.shape

    proj_results = fully_fused_projection(
        means,
        None,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d=eps2d,
        packed=False,
        near_plane=near_plane,
        far_plane=far_plane,
        camera_model=camera_model,
    )

    # The results are with shape [..., C, N, ...]. Only the elements with radii > 0 are valid.
    radii, means2d, depths, conics, compensations = proj_results
    transmittances = torch.broadcast_to(
        transmittances[..., None, :], batch_dims + (C, N)
    )  # [..., C, N]
    batch_ids, camera_ids, gaussian_ids = None, None, None
    image_ids = None

    meta.update(
        {
            # global batch and camera ids
            "batch_ids": batch_ids,
            "camera_ids": camera_ids,
            # local gaussian_ids
            "gaussian_ids": gaussian_ids,
            "radii": radii,
            "means2d": means2d,
            "depths": depths,
            "conics": conics,
            "transmittances": transmittances,
        }
    )

    # Turn colors into [..., C, N, D] or [..., nnz, D] to pass into rasterize_to_pixels()
    if sh_degree is None:
        # Colors are post-activation values, with shape [..., N, D] or [..., C, N, D]
        if intensities.dim() == num_batch_dims + 2:
            # Turn [..., N, D] into [..., C, N, D]
            intensities = torch.broadcast_to(
                intensities[..., None, :, :], batch_dims + (C, N, -1)
            )
        else:
            # colors is already [..., C, N, D]
            pass
    else:
        # Colors are SH coefficients, with shape [..., N, K, D] or [..., C, N, K, D]
        campos = torch.inverse(viewmats)[..., :3, 3]  # [..., C, 3]
        D = intensities.shape[-1]  # color channels
        dirs = means[..., None, :, :] - campos[..., None, :]  # [..., C, N, 3]
        masks = (radii > 0).all(dim=-1)  # [..., C, N]
        if intensities.dim() == num_batch_dims + 3:
            # Turn [..., N, K, D] into [..., C, N, K, D]
            shs = torch.broadcast_to(
                intensities[..., None, :, :, :], batch_dims + (C, N, -1, D)
            )
        else:
            # colors is already [..., C, N, K, D]
            shs = intensities
        intensities = spherical_harmonics(
            sh_degree, dirs, shs, masks=masks
        )  # [..., C, N, D]

        # Tuned brightening compared to stock gsplat for better results
        intensities = torch.clamp_min(intensities + 1.0, 0.5)

    # Identify intersecting tiles
    tile_width = math.ceil(width / float(tile_size_x))
    tile_height = math.ceil(height / float(tile_size_y))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles_rectangular(
        means2d,
        radii,
        depths,
        tile_size_x,
        tile_size_y,
        tile_width,
        tile_height,
        packed=False,
        n_images=I,
        image_ids=image_ids,
        gaussian_ids=gaussian_ids,
    )
    # print("rank", world_rank, "Before isect_offset_encode")
    isect_offsets = isect_offset_encode_rectangular(
        isect_ids, I, tile_width, tile_height
    )
    isect_offsets = isect_offsets.reshape(batch_dims + (C, tile_height, tile_width))

    # Determine convex from opening parameters
    convex = opening_angle is not None

    meta.update(
        {
            "tile_width": tile_width,
            "tile_height": tile_height,
            "tiles_per_gauss": tiles_per_gauss,
            "isect_ids": isect_ids,
            "flatten_ids": flatten_ids,
            "isect_offsets": isect_offsets,
            "width": width,
            "height": height,
            "tile_size_x": tile_size_x,
            "tile_size_y": tile_size_y,
            "n_batches": B,
            "n_cameras": C,
            "viewmats": viewmats,
            "opening_width": opening_width,
            "near_plane": near_plane,
            "far_plane": far_plane,
            "convex": convex,
        }
    )

    if intensities.shape[-1] > channel_chunk:
        # slice into chunks
        n_chunks = (intensities.shape[-1] + channel_chunk - 1) // channel_chunk
        render_ultrasound, render_echo_alphas = [], []
        for i in range(n_chunks):
            colors_chunk = intensities[..., i * channel_chunk : (i + 1) * channel_chunk]
            backgrounds_chunk = (
                backgrounds[..., i * channel_chunk : (i + 1) * channel_chunk]
                if backgrounds is not None
                else None
            )
            (
                render_ultrasound_,
                render_echo_alphas_,
                render_transmittances_,
                render_echoes_,
            ) = ultrasound_rasterize_to_pixels(
                means,
                quats,
                scales,
                colors_chunk,
                transmittances,
                viewmats,
                width,
                height,
                tile_size_x,
                tile_size_y,
                isect_offsets,
                flatten_ids,
                backgrounds=backgrounds_chunk,
                convex=convex,
                near_plane=near_plane,
                far_plane=far_plane,
                opening_angle=opening_angle if convex else 0.0,
                opening_width=opening_width if not convex else 0.0,
            )
            render_ultrasound.append(render_ultrasound_)
            render_echo_alphas.append(render_echo_alphas_)

        render_ultrasound = torch.cat(render_ultrasound, dim=-1)
        render_echo_alphas = render_echo_alphas[0]  # discard the rest
        render_transmittances = render_transmittances_[0]  # discard the rest
        render_echoes = render_echoes_[0]  # discard the rest
    else:
        (
            render_ultrasound,
            render_echo_alphas,
            render_transmittances,
            render_echoes,
        ) = ultrasound_rasterize_to_pixels(
            means,
            quats,
            scales,
            intensities,
            transmittances,
            viewmats,
            width,
            height,
            tile_size_x,
            tile_size_y,
            isect_offsets,
            flatten_ids,
            backgrounds=backgrounds,
            masks=None,
            convex=convex,
            near_plane=near_plane,
            far_plane=far_plane,
            opening_angle=opening_angle if convex else 0.0,
            opening_width=opening_width if not convex else 0.0,
        )

    return (
        render_ultrasound,
        render_echo_alphas,
        render_transmittances,
        render_echoes,
        meta,
    )


def ultrasound_3d_rasterization(
    means: Tensor,  # [..., N, 3]
    transmittances: Tensor,  # [..., N]
    quats: Tensor,  # [..., N, 4]
    scales: Tensor,  # [..., N, 3]
    intensities: Tensor,  # [..., (C,) N, D] or [..., (C,) N, K, D]
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    tile_size: int = 16,
    backgrounds: Optional[Tensor] = None,
    render_mode: Literal["echo", "transmittance"] = "echo",
    sparse_grad: bool = False,
    absgrad: bool = False,
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
    channel_chunk: int = 32,
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
    segmented: bool = False,
    covars: Optional[Tensor] = None,
    with_eval3d: bool = False,
) -> Tuple[Tensor, Tensor, Dict]:
    """Rasterize 3D Gaussians to ultrasound images using standard splatting.

    An alternative rendering path that uses the standard ``rasterize_to_pixels``
    kernel (instead of the custom ultrasound ray casting kernel).

    The ``render_mode`` selects which ultrasound component to render:

    - ``"echo"``: SH-computed intensities are used as both opacity and color,
      producing the echo component of the B-mode image.
    - ``"transmittance"``: Opacity is set to :math:`1 - \\tau_i` (where
      :math:`\\tau_i` is the per-Gaussian transmittance), and color is constant
      white, producing the accumulated transmittance map.

    .. note::
        **Batch Rasterization**: Supports batched rendering via batched
        ``viewmats`` and ``Ks``.

    .. note::
        **Spherical Harmonics**: If ``sh_degree`` is set, ``intensities`` is
        treated as SH coefficients and view-dependent values are computed.

    .. note::
        **Antialiased Rendering**: If ``rasterize_mode`` is ``"antialiased"``,
        a view-dependent compensation factor is applied to prevent aliasing.

    .. note::
        **AbsGrad**: If ``absgrad`` is True, absolute gradients of the projected
        2D means are computed during the backward pass, accessible via
        ``meta["means2d"].absgrad``.

    .. warning::
        This function is currently not differentiable w.r.t. the camera
        intrinsics ``Ks``.

    Args:
        means: The 3D centers of the Gaussians. [..., N, 3]
        transmittances: Per-Gaussian acoustic transmittance values. [..., N]
        quats: The quaternions of the Gaussians (wxyz convention). Not required
            to be normalized. [..., N, 4]
        scales: The scales of the Gaussians. [..., N, 3]
        intensities: Echo intensities. [..., (C,) N, D] for post-activation
            values, or [..., (C,) N, K, D] for SH coefficients.
        viewmats: The world-to-camera transformations. [..., C, 4, 4]
        Ks: The camera intrinsics. [..., C, 3, 3]
        width: The width of the output image in pixels.
        height: The height of the output image in pixels.
        near_plane: The near plane for clipping. Default is 0.01.
        far_plane: The far plane for clipping. Default is 1e10.
        radius_clip: Gaussians with 2D radius (in pixels) smaller or equal to
            this value will be skipped. Default is 0.0 (disabled).
        eps2d: An epsilon added to the eigenvalues of projected 2D covariance
            matrices, preventing Gaussians from becoming too small. Default is 0.3.
        sh_degree: The SH degree to use. If set, ``intensities`` is treated as
            SH coefficients; otherwise as post-activation values. Default is None.
        tile_size: The tile size for rasterization. Default is 16.
        backgrounds: Background values. [..., C, D]. Default is None.
        render_mode: The rendering mode. ``"echo"`` renders echo intensity,
            ``"transmittance"`` renders accumulated transmittance. Default is
            ``"echo"``.
        sparse_grad: If True, gradients for {means, quats, scales} use COO
            sparse layout. Default is False.
        absgrad: If True, absolute gradients of projected 2D means are computed.
            Default is False.
        rasterize_mode: ``"classic"`` or ``"antialiased"``. Default is ``"classic"``.
        channel_chunk: Max channels to render per pass. Default is 32.
        camera_model: Camera model: ``"pinhole"``, ``"ortho"``, or ``"fisheye"``.
            Default is ``"pinhole"``.
        segmented: Whether to use segmented radix sort. Default is False.
        covars: Optional covariance matrices. If provided, ``quats`` and
            ``scales`` are ignored. [..., N, 3, 3]. Default is None.
        with_eval3d: Whether to compute Gaussian response in 3D world space
            instead of 2D image space. Default is False.

    Returns:
        A tuple of three elements:

        **render_colors**: The rendered output. [..., C, height, width, D].
        Content depends on ``render_mode``: echo intensities or transmittance.

        **render_alphas**: The rendered alphas. [..., C, height, width, 1].

        **meta**: A dictionary of intermediate results of the rasterization.

    """
    meta = {}

    batch_dims = means.shape[:-2]
    num_batch_dims = len(batch_dims)
    B = math.prod(batch_dims)
    N = means.shape[-2]
    C = viewmats.shape[-3]
    I = B * C
    assert means.shape == batch_dims + (N, 3), means.shape
    if covars is None:
        assert quats.shape == batch_dims + (N, 4), quats.shape
        assert scales.shape == batch_dims + (N, 3), scales.shape
    else:
        assert covars.shape == batch_dims + (N, 3, 3), covars.shape
        quats, scales = None, None
        # convert covars from 3x3 matrix to upper-triangular 6D vector
        tri_indices = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])
        covars = covars[..., tri_indices[0], tri_indices[1]]
    assert viewmats.shape == batch_dims + (C, 4, 4), viewmats.shape
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.shape
    assert render_mode in ["echo", "transmittance"], render_mode

    if sh_degree is None:
        # treat colors as post-activation values, should be in shape [..., N, D] or [..., C, N, D]
        assert (
            intensities.dim() == num_batch_dims + 2
            and intensities.shape[:-1] == batch_dims + (N,)
        ) or (
            intensities.dim() == num_batch_dims + 3
            and intensities.shape[:-1] == batch_dims + (C, N)
        ), intensities.shape
    else:
        # treat colors as SH coefficients, should be in shape [..., N, K, D] or [..., C, N, K, D]
        # Allowing for activating partial SH bands
        assert (
            intensities.dim() == num_batch_dims + 3
            and intensities.shape[:-2] == batch_dims + (N,)
        ) or (
            intensities.dim() == num_batch_dims + 4
            and intensities.shape[:-2] == batch_dims + (C, N)
        ), intensities.shape
        assert (sh_degree + 1) ** 2 <= intensities.shape[-2], intensities.shape

    if with_eval3d:
        assert (quats is not None) and (
            scales is not None
        ), "eval3d requires to provide quats and scales."

    # Implement the multi-GPU strategy proposed in
    # `On Scaling Up 3D Gaussian Splatting Training <https://arxiv.org/abs/2406.18533>`.
    #
    # Project Gaussians to 2D. Directly pass in {quats, scales} is faster than precomputing covars.
    proj_results = fully_fused_projection(
        means,
        covars,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d=eps2d,
        packed=False,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        sparse_grad=sparse_grad,
        calc_compensations=(rasterize_mode == "antialiased"),
        camera_model=camera_model,
    )

    # The results are with shape [..., C, N, ...]. Only the elements with radii > 0 are valid.
    radii, means2d, depths, conics, compensations = proj_results
    batch_ids, camera_ids, gaussian_ids = None, None, None
    image_ids = None

    meta.update(
        {
            # global batch and camera ids
            "batch_ids": batch_ids,
            "camera_ids": camera_ids,
            # local gaussian_ids
            "gaussian_ids": gaussian_ids,
            "radii": radii,
            "means2d": means2d,
            "depths": depths,
            "conics": conics,
        }
    )

    # Turn colors into [..., C, N, D] or [..., nnz, D] to pass into rasterize_to_pixels()
    if sh_degree is None:
        # Colors are post-activation values, with shape [..., N, D] or [..., C, N, D]
        if intensities.dim() == num_batch_dims + 2:
            # Turn [..., N, D] into [..., C, N, D]
            intensities = torch.broadcast_to(
                intensities[..., None, :, :], batch_dims + (C, N, -1)
            )
        else:
            # colors is already [..., C, N, D]
            pass
    else:
        # Colors are SH coefficients, with shape [..., N, K, D] or [..., C, N, K, D]
        D = intensities.shape[-1]  # number of color channels
        campos = torch.inverse(viewmats)[..., :3, 3]  # [..., C, 3]
        dirs = means[..., None, :, :] - campos[..., None, :]  # [..., C, N, 3]
        masks = (radii > 0).all(dim=-1)  # [..., C, N]
        if intensities.dim() == num_batch_dims + 3:
            # Turn [..., N, K, D] into [..., C, N, K, D]
            shs = torch.broadcast_to(
                intensities[..., None, :, :, :], batch_dims + (C, N, -1, D)
            )
        else:
            # colors is already [..., C, N, K, D]
            shs = intensities
        intensities = spherical_harmonics(
            sh_degree, dirs, shs, masks=masks
        )  # [..., C, N, D]
        # make it apple-to-apple with Inria's CUDA Backend.
        intensities = torch.clamp_min(intensities + 0.5, 0.0)

        # For ultrasound rendering modes
        if render_mode == "transmittance":
            # Transmittance mode: use 1-transmittance as opacity, constant 1 color
            # Broadcast transmittances from [..., N] to [..., C, N]
            opacities = 1.0 - torch.broadcast_to(
                transmittances[..., None, :], batch_dims + (C, N)
            )
            # Constant white color
            intensities = torch.ones_like(intensities)
        else:
            # Echo mode: use SH-computed intensity as opacity
            sh_alpha = torch.clamp(intensities, 0.0, 1.0)  # [..., C, N, D]
            opacities = sh_alpha.squeeze(-1)  # [..., C, N]
            intensities = sh_alpha

    # Identify intersecting tiles
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        segmented=segmented,
        packed=False,
        n_images=I,
        image_ids=image_ids,
        gaussian_ids=gaussian_ids,
    )
    isect_offsets = isect_offset_encode(isect_ids, I, tile_width, tile_height)
    isect_offsets = isect_offsets.reshape(batch_dims + (C, tile_height, tile_width))

    meta.update(
        {
            "tile_width": tile_width,
            "tile_height": tile_height,
            "tiles_per_gauss": tiles_per_gauss,
            "isect_ids": isect_ids,
            "flatten_ids": flatten_ids,
            "isect_offsets": isect_offsets,
            "width": width,
            "height": height,
            "tile_size": tile_size,
            "n_batches": B,
            "n_cameras": C,
        }
    )

    if intensities.shape[-1] > channel_chunk:
        # slice into chunks
        n_chunks = (intensities.shape[-1] + channel_chunk - 1) // channel_chunk
        render_colors, render_alphas = [], []
        for i in range(n_chunks):
            colors_chunk = intensities[..., i * channel_chunk : (i + 1) * channel_chunk]
            backgrounds_chunk = (
                backgrounds[..., i * channel_chunk : (i + 1) * channel_chunk]
                if backgrounds is not None
                else None
            )
            render_colors_, render_alphas_ = rasterize_to_pixels(
                means2d,
                conics,
                colors_chunk,
                opacities,
                width,
                height,
                tile_size,
                isect_offsets,
                flatten_ids,
                backgrounds=backgrounds_chunk,
                packed=False,
                absgrad=absgrad,
            )
            render_colors.append(render_colors_)
            render_alphas.append(render_alphas_)
        render_colors = torch.cat(render_colors, dim=-1)
        render_alphas = render_alphas[0]  # discard the rest
    else:
        render_colors, render_alphas = rasterize_to_pixels(
            means2d,
            conics,
            intensities,
            opacities,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            backgrounds=backgrounds,
            packed=False,
            absgrad=absgrad,
        )

    return render_colors, render_alphas, meta
