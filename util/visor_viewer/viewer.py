import threading
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import viser
from nerfview import CameraState, Viewer
from scipy.spatial.transform import Rotation

from .render_state import VisorRenderTabState
from .renderer import render_viewer_frame
from .visualizers import TrainingProbeBoundsVisualizer, UltrasoundProbeVisualizer


def _quat_wxyz_to_mat(wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion [w,x,y,z] to 3x3 rotation matrix."""
    xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
    return Rotation.from_quat(xyzw).as_matrix().astype(np.float32)


def _mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w,x,y,z]."""
    xyzw = Rotation.from_matrix(R).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


class UltrasoundVisorViewer(Viewer):
    """
    Viewer for Gaussian splatting models with ultrasound simulation support.

    This viewer extends the base nerfview Viewer to support both standard 3D
    rendering and ultrasound simulation views. It manages multiple clients,
    probe visualization, and split-view modes.
    """

    @staticmethod
    def render_splats(
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        transmittances: torch.Tensor,
        intensities: torch.Tensor,
        camera_state: CameraState,
        render_tab_state: VisorRenderTabState,
        device: torch.device,
        sh_degree: int = 0,
    ) -> np.ndarray:
        """
        Render Gaussian splats from the given camera state.

        Args:
            means: Gaussian means [N, 3].
            quats: Gaussian quaternions [N, 4].
            scales: Gaussian scales [N, 3].
            transmittances: Gaussian transmittances [N, 1].
            intensities: Gaussian intensities [N, K, C].
            camera_state: Current camera state.
            render_tab_state: Current render settings.
            device: Torch device.
            sh_degree: Spherical harmonics degree.

        Returns:
            Rendered image as numpy array [H, W, 3].
        """
        c2w = camera_state.c2w
        c2w_torch = torch.from_numpy(c2w).float().to(device)
        viewmat = c2w_torch.inverse().unsqueeze(0)

        # Determine actual view mode (handles split mode detection)
        actual_view_mode = render_tab_state.view_mode

        # Problem: All arguments are shared between the two clients, so there is no way to tell which one we are rendering
        if render_tab_state.split_mode and render_tab_state.probe_c2w is not None:
            # In split mode, determine client type by checking orientation similarity
            probe_wxyz = _mat_to_quat_wxyz(render_tab_state.probe_c2w[:3, :3])
            cam_wxyz = _mat_to_quat_wxyz(camera_state.c2w[:3, :3])

            # Quaternion dot product (1.0 = same orientation, -1.0 = opposite)
            dot = abs(np.dot(probe_wxyz, cam_wxyz))

            if dot > 0.999:
                actual_view_mode = "ultrasound"
                probe_c2w = (
                    torch.from_numpy(render_tab_state.probe_c2w).float().to(device)
                )
                viewmat = probe_c2w.inverse().unsqueeze(0)
            else:
                actual_view_mode = "3d"
        elif (
            render_tab_state.view_mode == "ultrasound"
            and render_tab_state.probe_c2w is not None
        ):
            probe_c2w = torch.from_numpy(render_tab_state.probe_c2w).float().to(device)
            viewmat = probe_c2w.inverse().unsqueeze(0)

        # Apply filters
        N_all = means.shape[0]
        mask = torch.ones(N_all, dtype=torch.bool, device=device)

        if render_tab_state.enable_scale_filter:
            max_scale = scales.max(dim=-1).values
            mask &= (max_scale >= render_tab_state.min_scale) & (
                max_scale <= render_tab_state.max_scale
            )

        if render_tab_state.enable_intensity_filter:
            if intensities is not None:
                intensity = intensities.view(N_all, -1).abs().mean(dim=-1)
            else:
                intensity = torch.ones(N_all, device=device)
            mask &= (intensity >= render_tab_state.min_intensity) & (
                intensity <= render_tab_state.max_intensity
            )

        # Apply mask
        means_f = means[mask]
        quats_f = quats[mask]
        scales_f = scales[mask]
        transmittances_f = transmittances[mask]
        colors_f = intensities[mask] if intensities is not None else None

        # Update counts
        render_tab_state.total_gs_count = N_all
        render_tab_state.rendered_gs_count = int(mask.sum().item())

        # Temporarily set the actual view mode for rendering
        original_view_mode = render_tab_state.view_mode
        render_tab_state.view_mode = actual_view_mode

        result = render_viewer_frame(
            means=means_f,
            quats=quats_f,
            scales=scales_f,
            transmittances=transmittances_f,
            intensities=colors_f,
            viewmat=viewmat,
            camera_state=camera_state,
            render_tab_state=render_tab_state,
            device=device,
            sh_degree=sh_degree,
        )

        # Restore original view mode
        render_tab_state.view_mode = original_view_mode

        return result

    def __init__(
        self,
        server: viser.ViserServer,
        render_fn: Callable,
        output_dir: Path,
        mode: Literal["rendering", "training"] = "rendering",
        parser=None,
    ):
        """
        Initialize the viewer.

        Args:
            server: Viser server instance.
            render_fn: Render function callback.
            output_dir: Output directory for saved renders.
            mode: Operating mode ("rendering" or "training").
            parser: Optional dataset parser for training probe bounds.
        """
        self.probe_visualizer = UltrasoundProbeVisualizer(server)
        self.parser = parser
        self.training_probe_bounds_visualizer = TrainingProbeBoundsVisualizer(server)

        self._client_ids: List[int] = []
        self._client_lock = threading.Lock()

        self._transform_controls_handle = None

        self._probe_position: Optional[np.ndarray] = None
        self._probe_rotation: Optional[np.ndarray] = None
        self._probe_quaternion: Optional[np.ndarray] = None

        self._saved_3d_camera: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        super().__init__(server, render_fn, output_dir, mode)
        server.gui.set_panel_label("gsplat viewer")

        self._setup_client_handlers()

    def _setup_client_handlers(self) -> None:
        """Set up handlers for client connect/disconnect events."""

        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            with self._client_lock:
                if len(self._client_ids) >= 2:
                    # Too many clients, ignore this one
                    return
                if client.client_id not in self._client_ids:
                    self._client_ids.append(client.client_id)
                num_clients = len(self._client_ids)

            # Auto-enable split mode when 2 clients connect
            if num_clients == 2:
                self.render_tab_state.split_mode = True
                self.render_tab_state.view_mode = "split"
                self._update_view_mode_dropdown()
                self._create_transform_controls()
            elif num_clients == 1:
                if self.render_tab_state.view_mode == "3d":
                    show_handle = self._rendering_tab_handles.get(
                        "show_probe_handle_checkbox"
                    )
                    if show_handle is None or show_handle.value:
                        self._create_transform_controls()

            # Update probe visualization
            if self._probe_position is not None and self._probe_rotation is not None:
                self._update_probe_visualization(client)
                if self._get_client_mode(client.client_id) == "ultrasound":
                    self._set_client_camera_to_probe(client)

            @client.camera.on_update
            def _on_camera_update(camera: viser.CameraHandle) -> None:
                self._on_client_camera_update(client)

        @self.server.on_client_disconnect
        def _(client: viser.ClientHandle):
            self.probe_visualizer.remove_client(client.client_id)
            self.training_probe_bounds_visualizer.clear_for_client(client.client_id)
            if client.client_id in self._saved_3d_camera:
                del self._saved_3d_camera[client.client_id]
            with self._client_lock:
                if client.client_id in self._client_ids:
                    self._client_ids.remove(client.client_id)
                num_clients = len(self._client_ids)

            if num_clients <= 1 and self.render_tab_state.split_mode:
                self.render_tab_state.split_mode = False
                self.render_tab_state.view_mode = "3d"
                self._update_view_mode_dropdown()

    def _update_view_mode_dropdown(self) -> None:
        """Update view mode dropdown options based on client count."""
        if "view_mode_dropdown" not in self._rendering_tab_handles:
            return
        dropdown = self._rendering_tab_handles["view_mode_dropdown"]
        with self._client_lock:
            num_clients = len(self._client_ids)

        if num_clients >= 2:
            dropdown.options = ("split",)
            dropdown.value = "split"
            dropdown.disabled = True
        else:
            dropdown.options = ("3d", "ultrasound")
            if self.render_tab_state.view_mode not in ("3d", "ultrasound"):
                dropdown.value = "3d"
            else:
                dropdown.value = self.render_tab_state.view_mode
            dropdown.disabled = False

    def _update_render_mode_dropdown(self) -> None:
        """Update render mode dropdown options based on current view mode."""
        if "render_mode_dropdown" not in self._rendering_tab_handles:
            return
        dropdown = self._rendering_tab_handles["render_mode_dropdown"]
        current_value = self.render_tab_state.render_mode

        if self.render_tab_state.view_mode == "3d":
            dropdown.options = ("echo", "transmittance")
            # Fall back to "echo" if current value is not available in 3D mode
            if current_value not in ("echo", "transmittance"):
                dropdown.value = "echo"
                self.render_tab_state.render_mode = "echo"
        else:
            dropdown.options = ("ultrasound", "echo", "transmittance")
            # Keep current value if it's valid, otherwise default to "ultrasound"
            if current_value not in ("ultrasound", "echo", "transmittance"):
                dropdown.value = "ultrasound"
                self.render_tab_state.render_mode = "ultrasound"

    def _on_client_camera_update(self, client: viser.ClientHandle) -> None:
        """Handle camera updates from clients."""
        try:
            client_mode = self._get_client_mode(client.client_id)
            pos = np.array(client.camera.position, dtype=np.float32)
            wxyz = np.array(client.camera.wxyz, dtype=np.float32)

            if client_mode == "3d":
                self._saved_3d_camera[client.client_id] = (pos, wxyz)
                return

            self._probe_position = pos
            self._probe_quaternion = wxyz
            self._probe_rotation = _quat_wxyz_to_mat(wxyz)

            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = self._probe_rotation
            c2w[:3, 3] = pos
            self.render_tab_state.probe_c2w = c2w

            if self._transform_controls_handle is not None:
                self._transform_controls_handle.position = tuple(pos)
                self._transform_controls_handle.wxyz = tuple(wxyz)

            self._update_probe_for_all_3d_clients()

        except (AssertionError, AttributeError):
            pass

    def _get_client_mode(self, client_id: int) -> Literal["3d", "ultrasound"]:
        """Get the view mode for a specific client."""
        with self._client_lock:
            if not self.render_tab_state.split_mode or len(self._client_ids) < 2:
                return self.render_tab_state.view_mode
            sorted_ids = sorted(self._client_ids)
            return "3d" if client_id == sorted_ids[0] else "ultrasound"

    def _create_transform_controls(self) -> None:
        """Create transform controls for probe handle manipulation."""
        if self._transform_controls_handle is not None:
            return

        initial_position = (
            tuple(self._probe_position)
            if self._probe_position is not None
            else (0.0, 0.0, 0.0)
        )
        initial_wxyz = (
            tuple(self._probe_quaternion)
            if self._probe_quaternion is not None
            else (1.0, 0.0, 0.0, 0.0)
        )

        if self._probe_position is None:
            self._probe_position = np.array(initial_position, dtype=np.float32)
            self._probe_quaternion = np.array(initial_wxyz, dtype=np.float32)
            self._probe_rotation = _quat_wxyz_to_mat(self._probe_quaternion)
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = self._probe_rotation
            c2w[:3, 3] = self._probe_position
            self.render_tab_state.probe_c2w = c2w

        for client_id, client in self.server.get_clients().items():
            if self._get_client_mode(client_id) == "3d":
                self._transform_controls_handle = client.scene.add_transform_controls(
                    name="/probe_handle",
                    position=initial_position,
                    wxyz=initial_wxyz,
                    scale=0.5,
                )

                @self._transform_controls_handle.on_update
                def _(event: viser.TransformControlsEvent):
                    self._on_probe_handle_update(event)

                break

        self._update_probe_for_all_3d_clients()
        self._update_ultrasound_clients_camera()

    def _on_probe_handle_update(self, event: viser.TransformControlsEvent) -> None:
        """Handle probe handle position/rotation updates."""
        handle = event.target
        pos = np.array(handle.position)
        quat = np.array(handle.wxyz)

        self._probe_position = pos
        self._probe_quaternion = quat
        self._probe_rotation = _quat_wxyz_to_mat(quat)

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = self._probe_rotation
        c2w[:3, 3] = pos
        self.render_tab_state.probe_c2w = c2w

        self._update_probe_for_all_3d_clients()
        self._update_ultrasound_clients_camera()

        self.rerender_for_client = True

    def _remove_transform_controls(self) -> None:
        """Remove transform controls from the scene."""
        if self._transform_controls_handle is not None:
            self._transform_controls_handle.remove()
            self._transform_controls_handle = None
        for client_id in list(self.server.get_clients().keys()):
            self.probe_visualizer.clear_for_client(client_id)

    def _set_client_camera_to_probe(self, client: viser.ClientHandle) -> None:
        """Set a client's camera position to match the probe position."""
        if self._probe_position is None or self._probe_quaternion is None:
            return

        try:
            client.camera.position = tuple(self._probe_position)
            client.camera.wxyz = tuple(self._probe_quaternion)
        except AssertionError:
            pass

    def _update_ultrasound_clients_camera(self) -> None:
        """Update all ultrasound clients' cameras to match probe position."""
        if not self.render_tab_state.split_mode:
            return

        for client_id, client in self.server.get_clients().items():
            if self._get_client_mode(client_id) == "ultrasound":
                self._set_client_camera_to_probe(client)

    def _update_probe_visualization(self, client: viser.ClientHandle) -> None:
        """Update probe visualization for a specific client."""
        if self._probe_position is None or self._probe_rotation is None:
            return

        if self._get_client_mode(client.client_id) == "3d":
            self._create_probe_on_client(client)

    def _update_probe_for_all_3d_clients(self) -> None:
        """Update probe visualization on all 3D clients."""
        if self._probe_position is None or self._probe_rotation is None:
            return

        for client_id, client in self.server.get_clients().items():
            if self._get_client_mode(client_id) == "3d":
                self._create_probe_on_client(client)

    def _create_probe_on_client(self, client: viser.ClientHandle) -> None:
        """Create probe visualization on a specific client's scene."""
        if self._probe_position is None or self._probe_rotation is None:
            return

        if self.render_tab_state.ultrasound_type == "convex":
            self.probe_visualizer.create_convex_probe(
                client=client,
                camera_position=self._probe_position,
                camera_rotation=self._probe_rotation,
                near_plane=self.render_tab_state.ultrasound_near_plane,
                far_plane=self.render_tab_state.ultrasound_far_plane,
                opening_angle=self.render_tab_state.opening_angle,
                probe_id=0,
            )
        else:
            self.probe_visualizer.create_linear_probe(
                client=client,
                camera_position=self._probe_position,
                camera_rotation=self._probe_rotation,
                near_plane=self.render_tab_state.ultrasound_near_plane,
                far_plane=self.render_tab_state.ultrasound_far_plane,
                opening_width=self.render_tab_state.opening_width,
                probe_id=0,
            )

    def _init_rendering_tab(self) -> None:
        """Initialize the rendering tab state."""
        self.render_tab_state = VisorRenderTabState()
        self._rendering_tab_handles = {}
        self._rendering_folder = self.server.gui.add_folder("Rendering")

    def _populate_rendering_tab(self) -> None:
        """Populate the rendering tab with GUI controls."""
        server = self.server
        handles = self._rendering_tab_handles

        with self._rendering_folder:
            # View Mode controls
            with server.gui.add_folder("View Mode"):
                handles["view_mode_dropdown"] = server.gui.add_dropdown(
                    "View Mode",
                    ("3d", "ultrasound"),
                    initial_value=(
                        self.render_tab_state.view_mode
                        if self.render_tab_state.view_mode in ("3d", "ultrasound")
                        else "3d"
                    ),
                    hint="Switch between 3D and ultrasound view. With 2 clients: auto-splits.",
                )
                handles["show_probe_handle_checkbox"] = server.gui.add_checkbox(
                    "Show Probe Handle",
                    initial_value=True,
                    hint="Show draggable probe handle in 3D view for positioning ultrasound.",
                )

            # Ultrasound Probe controls
            with server.gui.add_folder("Ultrasound Probe"):
                handles["ultrasound_type_dropdown"] = server.gui.add_dropdown(
                    "Probe Type",
                    ("convex", "linear"),
                    initial_value=self.render_tab_state.ultrasound_type,
                    hint="Type of ultrasound probe.",
                )
                handles["ultrasound_near_far_plane_vec2"] = server.gui.add_vector2(
                    "Near/Far Plane",
                    initial_value=(
                        self.render_tab_state.ultrasound_near_plane,
                        self.render_tab_state.ultrasound_far_plane,
                    ),
                    min=(0, 0),
                    max=(100, 100),
                    step=0.1,
                    hint="Near and far plane distance for ultrasound probe.",
                )
                handles["opening_angle_slider"] = server.gui.add_number(
                    "Opening Angle",
                    initial_value=self.render_tab_state.opening_angle,
                    min=0.0,
                    max=180.0,
                    step=1.0,
                    disabled=self.render_tab_state.ultrasound_type != "convex",
                    hint="Opening angle for convex probe (degrees).",
                )
                handles["opening_width_slider"] = server.gui.add_number(
                    "Opening Width",
                    initial_value=self.render_tab_state.opening_width,
                    min=0.0,
                    max=100.0,
                    step=0.1,
                    disabled=self.render_tab_state.ultrasound_type != "linear",
                    hint="Opening width for linear probe.",
                )

            # 3D Rendering controls
            with server.gui.add_folder("3D Rendering"):
                handles["near_far_plane_vec2"] = server.gui.add_vector2(
                    "Near/Far",
                    initial_value=(
                        self.render_tab_state.near_plane,
                        self.render_tab_state.far_plane,
                    ),
                    min=(1e-3, 1e1),
                    max=(1e1, 1e3),
                    step=1e-3,
                    hint="Near and far plane for 3D rendering.",
                )
                handles["radius_clip_slider"] = server.gui.add_number(
                    "Radius Clip",
                    initial_value=self.render_tab_state.radius_clip,
                    min=0.0,
                    max=100.0,
                    step=1.0,
                    hint="2D radius clip for rendering.",
                )
                handles["eps2d_slider"] = server.gui.add_number(
                    "2D Epsilon",
                    initial_value=self.render_tab_state.eps2d,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    hint="Epsilon added to projected 2D covariance matrices.",
                )
                handles["rasterize_mode_dropdown"] = server.gui.add_dropdown(
                    "Anti-Aliasing",
                    ("classic", "antialiased"),
                    initial_value=self.render_tab_state.rasterize_mode,
                    hint="Whether to use classic or antialiased rasterization.",
                )
                handles["camera_model_dropdown"] = server.gui.add_dropdown(
                    "Camera Model",
                    ("pinhole", "ortho", "fisheye"),
                    initial_value=self.render_tab_state.camera_model,
                    hint="Camera model used for 3D rendering.",
                )
                handles["show_probe_bounds_checkbox"] = server.gui.add_checkbox(
                    "Show Training Probe Bounds",
                    initial_value=self.render_tab_state.show_probe_bounds,
                    hint="Show ultrasound training probe bounds in 3D view.",
                )

            # Display controls
            with server.gui.add_folder("Display"):
                handles["total_gs_count_number"] = server.gui.add_number(
                    "Total Splats",
                    initial_value=self.render_tab_state.total_gs_count,
                    disabled=True,
                    hint="Total number of splats in the scene.",
                )
                handles["rendered_gs_count_number"] = server.gui.add_number(
                    "Rendered Splats",
                    initial_value=self.render_tab_state.rendered_gs_count,
                    disabled=True,
                    hint="Number of splats rendered.",
                )
                handles["max_sh_degree_number"] = server.gui.add_number(
                    "Max SH Degree",
                    initial_value=self.render_tab_state.max_sh_degree,
                    min=0,
                    max=5,
                    step=1,
                    hint="Maximum SH degree used",
                )
                handles["backgrounds_slider"] = server.gui.add_rgb(
                    "Background",
                    initial_value=self.render_tab_state.backgrounds,
                    hint="Background color for rendering.",
                )
                handles["render_mode_dropdown"] = server.gui.add_dropdown(
                    "Render Mode",
                    (
                        ("echo", "transmittance")
                        if self.render_tab_state.view_mode == "3d"
                        else ("ultrasound", "echo", "transmittance")
                    ),
                    initial_value=self.render_tab_state.render_mode,
                    hint="What to render: echo, transmittance, or full ultrasound (US mode only).",
                )

            # Filter controls
            with server.gui.add_folder("Filters"):
                handles["enable_scale_filter_checkbox"] = server.gui.add_checkbox(
                    "Filter by Scale",
                    initial_value=self.render_tab_state.enable_scale_filter,
                    hint="Only render Gaussians whose 3D scale is inside the given range.",
                )
                handles["scale_filter_vec2"] = server.gui.add_vector2(
                    "Scale Range",
                    initial_value=(
                        self.render_tab_state.min_scale,
                        self.render_tab_state.max_scale,
                    ),
                    min=(0.0, 0.0),
                    max=(1e3, 1e3),
                    step=1e-3,
                    disabled=not handles["enable_scale_filter_checkbox"].value,
                    hint="Min / max 3D scale for visible Gaussians.",
                )
                handles["enable_intensity_filter_checkbox"] = server.gui.add_checkbox(
                    "Filter by Intensity",
                    initial_value=self.render_tab_state.enable_intensity_filter,
                    hint="Only render Gaussians whose intensity is inside the given range.",
                )
                handles["intensity_filter_vec2"] = server.gui.add_vector2(
                    "Intensity Range",
                    initial_value=(
                        self.render_tab_state.min_intensity,
                        self.render_tab_state.max_intensity,
                    ),
                    min=(0.0, 0.0),
                    max=(10.0, 10.0),
                    step=1e-3,
                    disabled=not handles["enable_intensity_filter_checkbox"].value,
                    hint="Min / max intensity for visible Gaussians.",
                )

            # Set up callbacks
            self._setup_gui_callbacks()

        super()._populate_rendering_tab()

        if self.render_tab_state.show_probe_bounds:
            self.training_probe_bounds_visualizer.create_bounds(
                self.parser, self._get_client_mode
            )

    def _setup_gui_callbacks(self) -> None:
        """Set up all GUI callback handlers."""
        handles = self._rendering_tab_handles

        # View mode callbacks
        @handles["view_mode_dropdown"].on_update
        def _(_) -> None:
            self._handle_view_mode_change()

        @handles["show_probe_handle_checkbox"].on_update
        def _(_) -> None:
            if handles["show_probe_handle_checkbox"].value:
                if (
                    self.render_tab_state.view_mode == "3d"
                    or self.render_tab_state.split_mode
                ):
                    self._create_transform_controls()
            else:
                self._remove_transform_controls()

        # Ultrasound probe callbacks
        @handles["ultrasound_type_dropdown"].on_update
        def _(_) -> None:
            self.render_tab_state.ultrasound_type = handles[
                "ultrasound_type_dropdown"
            ].value
            handles["opening_angle_slider"].disabled = (
                handles["ultrasound_type_dropdown"].value != "convex"
            )
            handles["opening_width_slider"].disabled = (
                handles["ultrasound_type_dropdown"].value != "linear"
            )
            self._update_probe_for_all_3d_clients()
            self.rerender(_)

        @handles["ultrasound_near_far_plane_vec2"].on_update
        def _(_) -> None:
            self.render_tab_state.ultrasound_near_plane = handles[
                "ultrasound_near_far_plane_vec2"
            ].value[0]
            self.render_tab_state.ultrasound_far_plane = handles[
                "ultrasound_near_far_plane_vec2"
            ].value[1]
            self._update_probe_for_all_3d_clients()
            self.rerender(_)

        @handles["opening_angle_slider"].on_update
        def _(_) -> None:
            self.render_tab_state.opening_angle = handles["opening_angle_slider"].value
            self._update_probe_for_all_3d_clients()
            self.rerender(_)

        @handles["opening_width_slider"].on_update
        def _(_) -> None:
            self.render_tab_state.opening_width = handles["opening_width_slider"].value
            self._update_probe_for_all_3d_clients()
            self.rerender(_)

        # 3D rendering callbacks
        @handles["near_far_plane_vec2"].on_update
        def _(_) -> None:
            self.render_tab_state.near_plane = handles["near_far_plane_vec2"].value[0]
            self.render_tab_state.far_plane = handles["near_far_plane_vec2"].value[1]
            self.rerender(_)

        @handles["radius_clip_slider"].on_update
        def _(_) -> None:
            self.render_tab_state.radius_clip = handles["radius_clip_slider"].value
            self.rerender(_)

        @handles["eps2d_slider"].on_update
        def _(_) -> None:
            self.render_tab_state.eps2d = handles["eps2d_slider"].value
            self.rerender(_)

        @handles["rasterize_mode_dropdown"].on_update
        def _(_) -> None:
            self.render_tab_state.rasterize_mode = handles[
                "rasterize_mode_dropdown"
            ].value
            self.rerender(_)

        @handles["camera_model_dropdown"].on_update
        def _(_) -> None:
            self.render_tab_state.camera_model = handles["camera_model_dropdown"].value
            self.rerender(_)

        @handles["show_probe_bounds_checkbox"].on_update
        def _(_) -> None:
            self.render_tab_state.show_probe_bounds = handles[
                "show_probe_bounds_checkbox"
            ].value
            if (
                handles["show_probe_bounds_checkbox"].value
                and self.render_tab_state.view_mode == "3d"
            ):
                self.training_probe_bounds_visualizer.create_bounds(
                    self.parser, self._get_client_mode
                )
            else:
                self.training_probe_bounds_visualizer.clear_all()

        # Display callbacks
        @handles["max_sh_degree_number"].on_update
        def _(_) -> None:
            self.render_tab_state.max_sh_degree = int(
                handles["max_sh_degree_number"].value
            )
            self.rerender(_)

        @handles["backgrounds_slider"].on_update
        def _(_) -> None:
            self.render_tab_state.backgrounds = handles["backgrounds_slider"].value
            self.rerender(_)

        @handles["render_mode_dropdown"].on_update
        def _(_) -> None:
            self.render_tab_state.render_mode = handles["render_mode_dropdown"].value
            self.rerender(_)

        # Filter callbacks
        @handles["enable_scale_filter_checkbox"].on_update
        def _(_) -> None:
            self.render_tab_state.enable_scale_filter = handles[
                "enable_scale_filter_checkbox"
            ].value
            handles["scale_filter_vec2"].disabled = not handles[
                "enable_scale_filter_checkbox"
            ].value
            self.rerender(_)

        @handles["scale_filter_vec2"].on_update
        def _(_) -> None:
            self.render_tab_state.min_scale = handles["scale_filter_vec2"].value[0]
            self.render_tab_state.max_scale = handles["scale_filter_vec2"].value[1]
            self.rerender(_)

        @handles["enable_intensity_filter_checkbox"].on_update
        def _(_) -> None:
            self.render_tab_state.enable_intensity_filter = handles[
                "enable_intensity_filter_checkbox"
            ].value
            handles["intensity_filter_vec2"].disabled = not handles[
                "enable_intensity_filter_checkbox"
            ].value
            self.rerender(_)

        @handles["intensity_filter_vec2"].on_update
        def _(_) -> None:
            self.render_tab_state.min_intensity = handles[
                "intensity_filter_vec2"
            ].value[0]
            self.render_tab_state.max_intensity = handles[
                "intensity_filter_vec2"
            ].value[1]
            self.rerender(_)

    def _handle_view_mode_change(self) -> None:
        """Handle view mode dropdown changes."""
        if self.render_tab_state.split_mode:
            return

        handles = self._rendering_tab_handles
        old_mode = self.render_tab_state.view_mode
        new_mode = handles["view_mode_dropdown"].value
        self.render_tab_state.view_mode = new_mode

        for client_id, client in self.server.get_clients().items():
            try:
                if old_mode == "3d" and new_mode == "ultrasound":
                    pos = np.array(client.camera.position, dtype=np.float32)
                    wxyz = np.array(client.camera.wxyz, dtype=np.float32)
                    self._saved_3d_camera[client_id] = (pos, wxyz)
                    if (
                        self._probe_position is not None
                        and self._probe_quaternion is not None
                    ):
                        client.camera.position = tuple(self._probe_position)
                        client.camera.wxyz = tuple(self._probe_quaternion)
                elif old_mode == "ultrasound" and new_mode == "3d":
                    if client_id in self._saved_3d_camera:
                        saved_pos, saved_wxyz = self._saved_3d_camera[client_id]
                        client.camera.position = tuple(saved_pos)
                        client.camera.wxyz = tuple(saved_wxyz)
            except (AssertionError, AttributeError):
                pass

        if new_mode == "3d" and handles["show_probe_handle_checkbox"].value:
            self._create_transform_controls()
        elif new_mode == "ultrasound":
            self._remove_transform_controls()

        # Update render mode dropdown options based on the new view mode
        self._update_render_mode_dropdown()

        self.rerender(None)

    def _after_render(self) -> None:
        """Update GUI elements after rendering."""
        self._rendering_tab_handles["total_gs_count_number"].value = (
            self.render_tab_state.total_gs_count
        )
        self._rendering_tab_handles["rendered_gs_count_number"].value = (
            self.render_tab_state.rendered_gs_count
        )
