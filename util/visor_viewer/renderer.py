import numpy as np
import torch
from nerfview import CameraState

from .render_state import VisorRenderTabState
from gsplat.rendering import ultrasound_3d_rasterization


def render_viewer_frame(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    transmittances: torch.Tensor,
    intensities: torch.Tensor,
    viewmat: torch.Tensor,
    camera_state: CameraState,
    render_tab_state: VisorRenderTabState,
    device: torch.device,
    sh_degree: int = 0,
) -> np.ndarray:
    """
    Render a single frame for the viewer.

    Args:
        means: Gaussian means tensor [N, 3].
        quats: Gaussian quaternions tensor [N, 4].
        scales: Gaussian scales tensor [N, 3].
        transmittances: Gaussian transmittances tensor [N, 1].
        intensities: Gaussian intensities tensor [N, K, 1] or [N, K, 3].
        viewmat: View matrix tensor [1, 4, 4].
        camera_state: Camera state from the viewer.
        render_tab_state: Current render settings.
        device: Torch device to use.
        sh_degree: Spherical harmonics degree.

    Returns:
        Rendered image as numpy array [H, W, 3] with values in [0, 1].
    """
    # Determine viewport size
    if render_tab_state.preview_render:
        viewport_width = render_tab_state.render_width
        viewport_height = render_tab_state.render_height
    else:
        viewport_width = render_tab_state.viewer_width
        viewport_height = render_tab_state.viewer_height

    if render_tab_state.view_mode == "3d":
        return _render_3d(
            means,
            quats,
            scales,
            transmittances,
            intensities,
            viewmat,
            camera_state,
            render_tab_state,
            device,
            sh_degree,
            viewport_width,
            viewport_height,
        )
    else:
        return _render_ultrasound(
            means,
            quats,
            scales,
            transmittances,
            intensities,
            viewmat,
            render_tab_state,
            device,
            sh_degree,
            viewport_width,
            viewport_height,
        )


def _render_3d(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    transmittances: torch.Tensor,
    intensities: torch.Tensor,
    viewmat: torch.Tensor,
    camera_state: CameraState,
    render_tab_state: VisorRenderTabState,
    device: torch.device,
    sh_degree: int,
    viewport_width: int,
    viewport_height: int,
) -> np.ndarray:
    """Render in 3D mode using 3d rasterization."""

    K = camera_state.get_K((viewport_width, viewport_height))
    K = torch.from_numpy(K).float().to(device).unsqueeze(0)

    # In 3D mode, "ultrasound" render mode falls back to "echo"
    # Since there is not way to do our ultrasound rendering in 3D
    render_mode_3d = render_tab_state.render_mode
    if render_mode_3d == "ultrasound":
        render_mode_3d = "echo"

    render, _, _ = ultrasound_3d_rasterization(
        means,
        transmittances,
        quats,
        scales,
        intensities,
        viewmat,
        K,
        viewport_width,
        viewport_height,
        sh_degree=(
            min(render_tab_state.max_sh_degree, sh_degree)
            if sh_degree is not None
            else None
        ),
        near_plane=render_tab_state.near_plane,
        far_plane=render_tab_state.far_plane,
        radius_clip=render_tab_state.radius_clip,
        eps2d=render_tab_state.eps2d,
        backgrounds=torch.tensor([[render_tab_state.backgrounds[0]]], device=device)
        / 255.0,
        render_mode=render_mode_3d,
        rasterize_mode=render_tab_state.rasterize_mode,
        camera_model=render_tab_state.camera_model,
        with_eval3d=False,
    )

    # Convert to RGB (mono expanded to 3 channels)
    render_mono = render[0, ..., 0:1].clamp(0, 1)
    render_rgb = render_mono.expand(-1, -1, 3).cpu().numpy()
    return render_rgb


def _render_ultrasound(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    transmittances: torch.Tensor,
    intensities: torch.Tensor,
    viewmat: torch.Tensor,
    render_tab_state: VisorRenderTabState,
    device: torch.device,
    sh_degree: int,
    viewport_width: int,
    viewport_height: int,
) -> np.ndarray:
    """Render in ultrasound mode using ultrasound rasterization."""
    from gsplat.rendering import ultrasound_rasterization

    us_near = render_tab_state.ultrasound_near_plane
    us_far = render_tab_state.ultrasound_far_plane
    us_width = render_tab_state.opening_width

    # Calculate ultrasound render dimensions maintaining physical aspect ratio
    us_depth = us_far - us_near
    us_aspect_ratio = us_width / us_depth

    us_render_height = viewport_height
    us_render_width = int(viewport_height * us_aspect_ratio)

    if us_render_width > viewport_width:
        us_render_width = viewport_width
        us_render_height = int(viewport_width / us_aspect_ratio)

    # Perform ultrasound rasterization
    (
        render_us,
        _,
        render_transmittances_us,
        render_echoes_us,
        _,
    ) = ultrasound_rasterization(
        means,
        quats,
        scales,
        transmittances,
        intensities,
        viewmat,
        us_render_width,
        us_render_height,
        us_near,
        us_far,
        opening_angle=(
            render_tab_state.opening_angle
            if render_tab_state.ultrasound_type == "convex"
            else None
        ),
        opening_width=(
            render_tab_state.opening_width
            if render_tab_state.ultrasound_type == "linear"
            else None
        ),
        tile_size_x=1,
        tile_size_y=512,
        sh_degree=(
            min(render_tab_state.max_sh_degree, sh_degree)
            if sh_degree is not None
            else None
        ),
        backgrounds=torch.tensor([[render_tab_state.backgrounds[0]]], device=device)
        / 255.0,
    )

    # Select output based on render mode
    if render_tab_state.render_mode == "echo":
        us_result = render_echoes_us
    elif render_tab_state.render_mode == "transmittance":
        us_result = render_transmittances_us
    else:
        us_result = render_us

    # Convert to RGB
    render_us_mono = us_result[0, ..., 0:1].clamp(0, 1)
    render_us_rgb = render_us_mono.expand(-1, -1, 3).cpu().numpy()

    # Center the ultrasound image in the viewport if smaller
    # Will fit since we already set those resolutions accordingly
    if us_render_width < viewport_width or us_render_height < viewport_height:
        output = np.zeros((viewport_height, viewport_width, 3), dtype=np.float32)
        y_offset = (viewport_height - us_render_height) // 2
        x_offset = (viewport_width - us_render_width) // 2
        output[
            y_offset : y_offset + us_render_height,
            x_offset : x_offset + us_render_width,
            :,
        ] = render_us_rgb
        return output

    return render_us_rgb
