from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from nerfview import RenderTabState


@dataclass
class VisorRenderTabState(RenderTabState):
    """
    State container for all rendering parameters in the viewer.

    This class extends RenderTabState to include both standard 3D rendering
    parameters and ultrasound-specific settings.
    """

    # Non-controllable parameters (read-only display)
    total_gs_count: int = 0
    rendered_gs_count: int = 0

    max_sh_degree: int = 5

    # 3D rendering parameters
    near_plane: float = 1e-2
    far_plane: float = 1e2
    radius_clip: float = 0.0
    eps2d: float = 0.3
    backgrounds: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    render_mode: Literal["echo", "transmittance", "ultrasound"] = "echo"

    # Ultrasound probe parameters
    ultrasound_type: Literal["convex", "linear"] = "linear"
    opening_angle: float = 70.0
    opening_width: float = 5.0
    ultrasound_near_plane: float = 0.0
    ultrasound_far_plane: float = 9.0
    show_probe_bounds: bool = False

    # Scale filter
    enable_scale_filter: bool = False
    min_scale: float = 0.0
    max_scale: float = 1.0

    # Intensity filter
    enable_intensity_filter: bool = False
    min_intensity: float = 0.0
    max_intensity: float = 1.0

    view_mode: Literal["3d", "ultrasound"] = "3d"

    # Split mode: enable both 3D and ultrasound views simultaneously
    split_mode: bool = False

    # Probe camera-to-world matrix for ultrasound rendering
    probe_c2w: Optional[np.ndarray] = None
