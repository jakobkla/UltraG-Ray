import numpy as np
import viser

from .base_visualizer import BaseVisualizer


class UltrasoundProbeVisualizer(BaseVisualizer):
    """
    Manages ultrasound probe visualizations in the 3D viewer.

    Uses per-client scenes to avoid conflicts between clients.
    Only 3D mode clients see probe visualizations, but will synchronize with ultrasound clients.
    """

    def create_convex_probe(
        self,
        client: viser.ClientHandle,
        camera_position: np.ndarray,
        camera_rotation: np.ndarray,
        near_plane: float,
        far_plane: float,
        opening_angle: float,
        probe_id: int = 0,
    ) -> None:
        """
        Create a triangular visualization for a convex probe.
        Not really accurate since that image would not be a literal triangle, but it will suffice for visualization purporses.

        Args:
            client: The viser client handle.
            camera_position: Probe position in world space.
            camera_rotation: Probe rotation matrix (3x3).
            near_plane: Near plane distance.
            far_plane: Far plane distance.
            opening_angle: Opening angle in degrees.
            probe_id: Unique identifier for this probe.
        """
        client_id = client.client_id
        self.clear_handle(client_id, probe_id)

        angle_rad = np.deg2rad(opening_angle)
        half_angle = angle_rad / 2.0
        far_width = 2 * far_plane * np.tan(half_angle)

        # Define triangle vertices in local space (camera frame)
        apex = np.array([0, 0, near_plane])
        left_corner = np.array([-far_width / 2, 0, far_plane])
        right_corner = np.array([far_width / 2, 0, far_plane])

        # Transform vertices to world space
        vertices_local = np.array([apex, left_corner, right_corner])
        vertices_world = self.transform_to_world(
            vertices_local, camera_position, camera_rotation
        )

        # Create line segments for the triangle
        lines = np.array(
            [
                [vertices_world[0], vertices_world[1]],
                [vertices_world[1], vertices_world[2]],
                [vertices_world[2], vertices_world[0]],
            ]
        )

        handle = client.scene.add_line_segments(
            name=f"/ultrasound_probe_{probe_id}",
            points=lines,
            colors=(0.9, 0.1, 0.1),
            line_width=2.5,
        )
        self._store_handle(client_id, probe_id, handle)

    def create_linear_probe(
        self,
        client: viser.ClientHandle,
        camera_position: np.ndarray,
        camera_rotation: np.ndarray,
        near_plane: float,
        far_plane: float,
        opening_width: float,
        probe_id: int = 0,
    ) -> None:
        """
        Create a rectangular visualization for a linear probe.

        Args:
            client: The viser client handle.
            camera_position: Probe position in world space.
            camera_rotation: Probe rotation matrix (3x3).
            near_plane: Near plane distance.
            far_plane: Far plane distance.
            opening_width: Width of the probe field.
            probe_id: Unique identifier for this probe.
        """
        client_id = client.client_id
        self.clear_handle(client_id, probe_id)

        half_width = opening_width / 2.0

        # Define rectangle vertices in local space
        corners_local = np.array(
            [
                [-half_width, 0, near_plane],
                [half_width, 0, near_plane],
                [half_width, 0, far_plane],
                [-half_width, 0, far_plane],
            ]
        )

        corners_world = self.transform_to_world(
            corners_local, camera_position, camera_rotation
        )

        lines = np.array(
            [
                [corners_world[0], corners_world[1]],
                [corners_world[1], corners_world[2]],
                [corners_world[2], corners_world[3]],
                [corners_world[3], corners_world[0]],
            ]
        )

        handle = client.scene.add_line_segments(
            name=f"/ultrasound_probe_{probe_id}",
            points=lines,
            colors=(0.9, 0.1, 0.1),
            line_width=2.5,
        )
        self._store_handle(client_id, probe_id, handle)
